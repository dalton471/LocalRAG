
from __future__ import annotations

import os
import re
import json
import time
import uuid
import logging
from pathlib import Path
from datetime import datetime

import requests
import streamlit as st


# =========================================================
# CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

TRACE = LOG_DIR / "request_trace.jsonl"

SERVER1 = os.getenv("SERVER1_URL", "http://127.0.0.1:8081").rstrip("/")
SERVER2 = os.getenv("SERVER2_URL", "http://127.0.0.1:8082").rstrip("/")

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

TIMEOUT = int(os.getenv("API_TIMEOUT_SECONDS", "10"))
GEMINI_TIMEOUT = int(os.getenv("GEMINI_TIMEOUT_SECONDS", "25"))


# =========================================================
# LOGGING
# =========================================================

logger = logging.getLogger("employee_router")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.FileHandler(
        LOG_DIR / "chatbot_router.log",
        encoding="utf-8"
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(message)s")
    )
    logger.addHandler(handler)


def emit(request_id, stage, **extra):
    event = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "request_id": request_id,
        "component": "SERVER_3_ROUTER_GEMINI",
        "stage": stage,
        **extra
    }

    line = json.dumps(event, ensure_ascii=False, default=str)
    logger.info(line)

    with TRACE.open("a", encoding="utf-8") as file:
        file.write(line + "\n")


# =========================================================
# QUESTION ROUTING
# =========================================================

def normalize_question(question):
    text = question.lower().strip()

    # Handle common spelling mistakes.
    corrections = {
        "deparment": "department",
        "departmant": "department",
        "departement": "department",
        "departmnt": "department",
        "employe": "employee",
        "sallery": "salary"
    }

    for wrong, correct in corrections.items():
        text = re.sub(rf"\b{wrong}\b", correct, text)

    return text


DEPARTMENT_NAMES = [
    "Engineering", "Data Science", "Human Resources", "Finance",
    "Marketing", "Operations", "IT Support", "Sales"
]


def detect_all_department_query(question):
    q = normalize_question(question)
    if "employee" not in q:
        return None

    list_intent = any(term in q for term in [
        "all", "every", "list", "show", "which employees",
        "employees who", "employees work", "employees in",
        "employee names"
    ])

    if not list_intent:
        return None

    normalized = norm_department_text(q)
    for department in DEPARTMENT_NAMES:
        if norm_department_text(department) in normalized:
            return department

    return None


def norm_department_text(value):
    return " ".join(re.findall(r"[a-z0-9]+", str(value or "").lower()))


def extract_employee_name_lookup(question):
    q = normalize_question(question)

    match = re.search(
        r"\b(?:which|what)\s+(?:department|dept|team)\s+does\s+"
        r"([a-z][a-z'-]*(?:\s+[a-z][a-z'-]*){1,3})"
        r"\s+(?:work|works)\s+in\b",
        q
    )

    if not match:
        return None

    return " ".join(match.group(1).split()).title()

def route_question(question):
    all_department = detect_all_department_query(question)
    if all_department:
        return ["server2", "server1"]

    q = normalize_question(question)

    department_terms = [
        "department", "dept", "which team", "what team",
        "team does", "division", "business unit",
        "belongs to", "works in", "working in", "department of"
    ]

    profile_terms = [
        "employee id", "employee number", "emp id", "salary",
        "pay", "designation", "job title", "role",
        "office location", "work location", "city", "address",
        "joining date", "joined", "employment status",
        "status", "profile", "personal details", "employee details"
    ]

    department_requested = any(term in q for term in department_terms)
    employee_id_present = bool(
        re.search(r"\bEMP\d+\b", question.upper())
    )

    profile_requested = (
        any(term in q for term in profile_terms)
        or bool(re.search(r"\bid\b", q))
    )

    if department_requested and employee_id_present:
        return ["server1", "server2"]

    if department_requested and profile_requested:
        return ["server1", "server2"]

    if department_requested:
        return ["server2"]

    if profile_requested:
        return ["server1"]

    return ["server1"]


# =========================================================
# SERVER HEALTH
# =========================================================

def get_server_health(url, server_name):
    try:
        response = requests.get(url + "/health", timeout=3)
        response.raise_for_status()

        data = response.json()

        # Server 1 uses employee_count; Server 2 uses record_count.
        count = data.get(
            "record_count",
            data.get("employee_count", 0)
        )

        return {
            "online": True,
            "count": count,
            "data": data,
            "error": None
        }

    except Exception as error:
        return {
            "online": False,
            "count": 0,
            "data": {},
            "error": str(error)
        }


# =========================================================
# LOCAL RETRIEVAL SERVICES
# =========================================================

def call_server(url, server_name, question, request_id, record_ref=None, department=None, record_refs=None):
    start = time.perf_counter()

    try:
        payload = {
            "query": question,
            "top_k": 10
        }

        if department:
            payload = {
                "query": "",
                "department": department,
                "top_k": 10
            }
        elif record_refs:
            payload = {
                "query": "",
                "record_refs": record_refs,
                "top_k": 10
            }
        elif record_ref:
            payload = {
                "query": "",
                "record_ref": record_ref,
                "top_k": 1
            }

        response = requests.post(
            url + "/search",
            json=payload,
            headers={"X-Request-ID": request_id},
            timeout=TIMEOUT
        )

        response.raise_for_status()
        data = response.json()

        elapsed = round(
            (time.perf_counter() - start) * 1000, 2
        )

        return data, elapsed, None

    except Exception as error:
        elapsed = round(
            (time.perf_counter() - start) * 1000, 2
        )

        return {}, elapsed, f"{server_name}: {error}"


def compact_context(results, include_all=False):
    merged = {}

    for service, data in results.items():
        trace = data.get("retrieval_trace", {})
        rankings = trace.get("rankings", [])

        ranking_field = (
            "employee_id" if service == "server1" else "record_ref"
        )

        ranking_lookup = {
            str(item.get("identity", "")).strip(): item
            for item in rankings
        }

        for index, row in enumerate(data.get("records", [])):
            record_ref = str(row.get("record_ref", "")).strip()

            # Do not accidentally merge records without a shared reference.
            key = record_ref or f"{service}_unlinked_{index}"

            item = merged.setdefault(key, {
                "record_ref": record_ref or None,
                "employee_profile": {},
                "department_directory": {},
                "retrieval": {
                    "combined_rrf_score": 0.0,
                    "service_scores": {}
                }
            })

            if service == "server1":
                item["employee_profile"].update(row)
            elif service == "server2":
                item["department_directory"].update(row)

            identity = str(row.get(ranking_field, "")).strip()
            ranking = ranking_lookup.get(identity)

            if ranking and ranking.get("rrf_score") is not None:
                score = float(ranking["rrf_score"])
                item["retrieval"]["combined_rrf_score"] += score
                item["retrieval"]["service_scores"][service] = score

    evidence = list(merged.values())

    exact_lookup = any(
        str(data.get("retrieved_using", "")).startswith("exact_")
        for data in results.values()
    )

    if not exact_lookup:
        evidence.sort(
            key=lambda item: item["retrieval"]["combined_rrf_score"],
            reverse=True
        )

    selected = evidence if include_all else evidence[:5]

    return [{
        "source": "merged_local_employee_evidence",
        "selection_method": (
            "all_department_matches" if include_all
            else ("exact_lookup" if exact_lookup
                  else "global_combined_rrf")
        ),
        "candidate_count": len(evidence),
        "selected_count": len(selected),
        "records": selected
    }]


# =========================================================
# GEMINI CLOUD GENERATION
# =========================================================

def gemini_answer(question, context, request_id=None):

    if not GEMINI_API_KEY:
        return None, "GEMINI_API_KEY is not configured.", 0, {}

    if not context or not any(
        item.get("records") for item in context
    ):
        return None, "No retrieved records are available.", 0, {}

    record_count = sum(
        len(item.get("records", [])) for item in context
    )
    listing_instruction = (
        "For this department-wide listing, include every employee "
        "record in the supplied context. Do not omit or group employees.\n"
        if record_count > 5 else ""
    )

    prompt = (
        "You are an employee knowledge assistant.\n"
        "Answer only using the retrieved JSON records below.\n"
        "Never invent employee information.\n"
        "Server 1 contains employee profile information.\n"
        "Server 2 contains department information only.\n"
        "Use only the fields returned by the selected services.\n"
        "If the requested information is missing, say it was not found.\n"
        "Do not claim access to the complete employee database.\n"
        "Give a concise and professional answer.\n"
        + listing_instruction
        + "\nRETRIEVED CONTEXT:\n"
        + json.dumps(context, ensure_ascii=False)
        + "\n\nQUESTION:\n"
        + question
        + "\n\nANSWER:"
    )

    url = (
        "https://generativelanguage.googleapis.com/"
        f"v1beta/models/{GEMINI_MODEL}:generateContent"
    )

    start = time.perf_counter()

    selected_records = [
        record
        for group in context
        for record in group.get("records", [])
    ]
    selected_refs = [
        record.get("record_ref")
        for record in selected_records
        if record.get("record_ref")
    ]

    if request_id:
        emit(
            request_id,
            "GEMINI_REQUEST_STARTED",
            provider="Google Gemini Cloud",
            model=GEMINI_MODEL,
            endpoint=url,
            context_record_count=len(selected_records),
            selected_record_refs=selected_refs,
            prompt_characters=len(prompt)
        )

    try:
        response = requests.post(
            url,
            headers={
                "x-goog-api-key": GEMINI_API_KEY,
                "Content-Type": "application/json"
            },
            json={
                "contents": [{
                    "role": "user",
                    "parts": [{"text": prompt}]
                }],
                "generationConfig": {
                    "temperature": 0.1,
                    "maxOutputTokens": 1400 if record_count > 5 else 220
                }
            },
            timeout=GEMINI_TIMEOUT
        )

        elapsed = round(
            (time.perf_counter() - start) * 1000, 2
        )

        if not response.ok:
            try:
                message = response.json().get(
                    "error", {}
                ).get("message", response.text[:300])
            except Exception:
                message = response.text[:300]

            raise RuntimeError(
                f"Gemini HTTP {response.status_code}: {message}"
            )

        body = response.json()

        candidates = body.get("candidates", [])

        if not candidates:
            raise RuntimeError("Gemini returned no answer candidates.")

        parts = (
            candidates[0]
            .get("content", {})
            .get("parts", [])
        )

        answer = "".join(
            part.get("text", "")
            for part in parts
            if isinstance(part, dict)
        ).strip()

        if not answer:
            raise RuntimeError("Gemini returned an empty answer.")

        usage = body.get("usageMetadata", {})

        token_usage = {
            "prompt_tokens": usage.get("promptTokenCount"),
            "candidate_tokens": usage.get("candidatesTokenCount"),
            "total_tokens": usage.get("totalTokenCount")
        }

        if request_id:
            emit(
                request_id,
                "GEMINI_RESPONSE_RECEIVED",
                provider="Google Gemini Cloud",
                model=GEMINI_MODEL,
                http_status=response.status_code,
                generation_ms=elapsed,
                token_usage=token_usage,
                answer_characters=len(answer)
            )

        return answer, None, elapsed, token_usage

    except Exception as error:
        elapsed = round(
            (time.perf_counter() - start) * 1000, 2
        )

        if request_id:
            emit(
                request_id,
                "GEMINI_REQUEST_FAILED",
                provider="Google Gemini Cloud",
                model=GEMINI_MODEL,
                generation_ms=elapsed,
                error=str(error)
            )

        return None, str(error), elapsed, {}


# =========================================================
# LOCAL FALLBACK
# =========================================================

def safe_fallback(results):

    records = []

    for data in results.values():
        records.extend(data.get("records", []))

    if not records:
        return (
            "No matching employee information was found "
            "in the selected service(s)."
        )

    output = []

    for record in records:
        fields = [
            f"{key.replace('_', ' ').title()}: {value}"
            for key, value in record.items()
        ]

        output.append(" | ".join(fields))

    return (
        "Gemini is unavailable. Here is the information "
        "retrieved from the local service:\n\n"
        + "\n\n".join(output)
    )


# =========================================================
# STREAMLIT PAGE
# =========================================================

st.set_page_config(
    page_title="Employee Hybrid RAG Assistant",
    page_icon="💜",
    layout="wide"
)


# =========================================================
# PROFESSIONAL PURPLE DESIGN
# =========================================================

st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(circle at 10% 0%, #39205f 0%, transparent 35%),
        linear-gradient(135deg, #100b1c, #1d1231 55%, #281642);
    color: #f5efff;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3, p, label {
    color: #f5efff !important;
}

.hero {
    background: linear-gradient(120deg, #38205b, #5b2b91);
    border: 1px solid #8056b2;
    border-radius: 22px;
    padding: 30px;
    margin-bottom: 24px;
    box-shadow: 0 12px 35px rgba(0,0,0,0.22);
}

.hero-title {
    font-size: 30px;
    font-weight: 800;
    color: white;
    margin-bottom: 8px;
}

.hero-subtitle {
    font-size: 14px;
    color: #e4d5fa;
}

.server-card {
    background: rgba(255,255,255,0.055);
    border: 1px solid rgba(190,150,255,0.28);
    border-radius: 18px;
    padding: 20px;
    min-height: 135px;
}

.server-name {
    font-size: 13px;
    color: #cbb5ec;
    font-weight: 700;
    letter-spacing: 1px;
}

.server-count {
    font-size: 30px;
    color: white;
    font-weight: 800;
    margin-top: 10px;
}

.server-status {
    font-size: 13px;
    color: #a7f3d0;
    margin-top: 4px;
}

.section-card {
    background: rgba(255,255,255,0.045);
    border: 1px solid rgba(190,150,255,0.22);
    border-radius: 18px;
    padding: 22px;
    margin-top: 18px;
}

.stButton > button,
.stFormSubmitButton > button {
    background: linear-gradient(90deg, #7c3aed, #a855f7);
    color: white;
    border: none;
    border-radius: 12px;
    font-weight: 700;
    min-height: 45px;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    border: 1px solid #d8b4fe;
    color: white;
}

div[data-testid="stTextInput"] input {
    background: #241638;
    color: white;
    border: 1px solid #8056b2;
    border-radius: 10px;
}

div[data-testid="stExpander"] {
    border: 1px solid #68428f;
    border-radius: 12px;
    background: rgba(255,255,255,0.035);
}

div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.05);
    padding: 15px;
    border-radius: 14px;
    border: 1px solid rgba(190,150,255,0.2);
}

hr {
    border-color: #63427f;
}

.small-note {
    color: #c6b5dd;
    font-size: 12px;
}

/* High-contrast black and yellow error panel */
.error-panel {
    background: #080808;
    border: 2px solid #ffd400;
    border-left: 7px solid #ffd400;
    border-radius: 12px;
    padding: 16px 20px;
    margin: 14px 0;
    color: #ffe44d !important;
}

.error-panel-title {
    color: #ffd400 !important;
    font-size: 17px;
    font-weight: 800;
    margin-bottom: 8px;
}

.error-panel-message {
    color: #fff176 !important;
    font-size: 14px;
    overflow-wrap: anywhere;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# HERO SECTION
# =========================================================

st.markdown("""
<div class="hero">
    <div class="hero-title">
        💜 Employee Advanced Hybrid RAG Assistant
    </div>
    <div class="hero-subtitle">
        Intelligent employee knowledge retrieval ·
        Two isolated local services ·
        Gemini cloud answer generation ·
        Request routing and audit trace
    </div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# SERVER STATUS
# =========================================================

health1 = get_server_health(SERVER1, "Server 1")
health2 = get_server_health(SERVER2, "Server 2")

left, right = st.columns(2)

with left:
    status_text = "ONLINE" if health1["online"] else "OFFLINE"
    st.markdown(
        f"""
        <div class="server-card">
            <div class="server-name">SERVER 1 · EMPLOYEE PROFILE</div>
            <div class="server-count">{health1['count']} Records</div>
            <div class="server-status">{status_text}</div>
            <div class="small-note">
                Employee ID · Salary · Designation · Location
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if not health1["online"]:
        st.markdown(
            '<div class="error-panel">'
            '<div class="error-panel-title">⚠ SERVER 1 — EMPLOYEE PROFILE OFFLINE</div>'
            '<div class="error-panel-message">'
            + __import__("html").escape(str(health1["error"]))
            + '</div></div>',
            unsafe_allow_html=True
        )

with right:
    status_text = "ONLINE" if health2["online"] else "OFFLINE"
    st.markdown(
        f"""
        <div class="server-card">
            <div class="server-name">SERVER 2 · DEPARTMENT DIRECTORY</div>
            <div class="server-count">{health2['count']} Records</div>
            <div class="server-status">{status_text}</div>
            <div class="small-note">
                Employee Name · Department
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if not health2["online"]:
        st.markdown(
            '<div class="error-panel">'
            '<div class="error-panel-title">⚠ SERVER 2 — DEPARTMENT DIRECTORY OFFLINE</div>'
            '<div class="error-panel-message">'
            + __import__("html").escape(str(health2["error"]))
            + '</div></div>',
            unsafe_allow_html=True
        )


st.markdown(
    f"""
    <p class="small-note">
        Gemini model: {GEMINI_MODEL} ·
        Only retrieved records and the question are sent for answer generation.
    </p>
    """,
    unsafe_allow_html=True
)


# =========================================================
# QUESTION INPUT
# =========================================================

st.markdown('<div class="section-card">', unsafe_allow_html=True)

st.subheader("Ask your Employee Knowledge Assistant")

st.caption(
    "Ask about employee departments, IDs, salary, designation "
    "and other available employee information."
)

with st.form("question_form"):
    question = st.text_input(
        "Your question",
        placeholder="Example: Which department is Rajkumar in?"
    )

    submitted = st.form_submit_button(
        "Ask Assistant  ↗",
        use_container_width=True
    )

st.markdown('</div>', unsafe_allow_html=True)



# =========================================================
# COMPLETE-DATASET DEPARTMENT SALARY ANALYTICS
# =========================================================

def is_department_salary_analytics_question(question):
    text = question.lower()
    asks_average = any(word in text for word in (
        "average", "avg", "mean"
    ))
    asks_salary = any(word in text for word in (
        "salary", "salaries", "annual pay", "compensation"
    ))
    asks_department = any(word in text for word in (
        "department", "departments", "dept", "team"
    ))
    return asks_average and asks_salary and asks_department


def fetch_department_salary_analytics(request_id):
    from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

    endpoints = {
        "server1": SERVER1.rstrip("/") + "/records",
        "server2": SERVER2.rstrip("/") + "/records"
    }

    datasets = {}
    elapsed_times = {}

    for server, endpoint in endpoints.items():
        started = time.perf_counter()

        response = requests.get(
            endpoint,
            headers={"X-Request-ID": request_id},
            timeout=TIMEOUT
        )

        elapsed_times[server] = round(
            (time.perf_counter() - started) * 1000, 2
        )

        response.raise_for_status()
        payload = response.json()

        records = payload.get("records")
        reported_count = payload.get("record_count")

        if not isinstance(records, list):
            raise ValueError(
                f"{server} did not return a valid records list."
            )

        if reported_count != 150 or len(records) != 150:
            raise ValueError(
                f"{server} must return exactly 150 records. "
                f"Reported: {reported_count}; received: {len(records)}."
            )

        datasets[server] = records

    employees = datasets["server1"]
    departments = datasets["server2"]

    employee_refs = [
        str(row.get("record_ref", "")).strip().upper()
        for row in employees
    ]
    department_refs = [
        str(row.get("record_ref", "")).strip().upper()
        for row in departments
    ]

    if any(not ref for ref in employee_refs + department_refs):
        raise ValueError("A record is missing its record_ref.")

    if len(set(employee_refs)) != 150:
        raise ValueError("Server 1 contains duplicate record references.")

    if len(set(department_refs)) != 150:
        raise ValueError("Server 2 contains duplicate record references.")

    if set(employee_refs) != set(department_refs):
        missing_departments = sorted(set(employee_refs) - set(department_refs))
        missing_employees = sorted(set(department_refs) - set(employee_refs))

        raise ValueError(
            "The two datasets do not contain identical reference sets. "
            f"Missing department references: {len(missing_departments)}; "
            f"missing employee references: {len(missing_employees)}."
        )

    department_by_ref = {
        str(row["record_ref"]).strip().upper(): row
        for row in departments
    }

    grouped = {}

    for employee in employees:
        ref = str(employee["record_ref"]).strip().upper()
        department_record = department_by_ref[ref]

        employee_name = " ".join(
            str(employee.get("name", "")).strip().lower().split()
        )
        department_name = " ".join(
            str(department_record.get("name", "")).strip().lower().split()
        )

        if not employee_name or employee_name != department_name:
            raise ValueError(
                f"Employee name validation failed for reference {ref}."
            )

        department = str(
            department_record.get("department", "")
        ).strip()

        if not department:
            raise ValueError(
                f"Department is missing for reference {ref}."
            )

        raw_salary = employee.get("annual_salary_inr")

        try:
            salary = Decimal(str(raw_salary).replace(",", "").strip())
        except (InvalidOperation, ValueError, TypeError):
            raise ValueError(
                f"Invalid annual salary for reference {ref}."
            )

        if not salary.is_finite() or salary <= 0:
            raise ValueError(
                f"Invalid or non-positive annual salary for reference {ref}."
            )

        key = department.casefold()

        if key not in grouped:
            grouped[key] = {
                "department": department,
                "count": 0,
                "total": Decimal("0")
            }

        grouped[key]["count"] += 1
        grouped[key]["total"] += salary

    if sum(item["count"] for item in grouped.values()) != 150:
        raise ValueError(
            "Department aggregation did not account for all 150 employees."
        )

    rows = []

    for item in grouped.values():
        average = (
            item["total"] / Decimal(item["count"])
        ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        rows.append({
            "Department": item["department"],
            "Employee Count": item["count"],
            "Average Annual Salary (INR)": f"₹{average:,.2f}",
            "_average": average
        })

    rows.sort(
        key=lambda row: (-row["_average"], row["Department"].casefold())
    )

    highest_average = rows[0]["_average"]
    highest_departments = [
        row["Department"]
        for row in rows
        if row["_average"] == highest_average
    ]

    display_rows = [
        {
            "Department": row["Department"],
            "Employee Count": row["Employee Count"],
            "Average Annual Salary (INR)": row["Average Annual Salary (INR)"]
        }
        for row in rows
    ]

    result = {
        "total_employees": 150,
        "department_count": len(display_rows),
        "highest_departments": highest_departments,
        "highest_average": f"₹{highest_average:,.2f}",
        "rows": display_rows
    }

    emit(
        request_id,
        "FULL_DATASET_SALARY_ANALYTICS_COMPLETED",
        employee_records=150,
        department_records=150,
        joined_records=150,
        department_count=len(display_rows),
        highest_departments=highest_departments,
        highest_average_inr=str(highest_average),
        aggregation_engine="Python Decimal"
    )

    return result, elapsed_times


# =========================================================
# PROCESS QUESTION
# =========================================================

if submitted and question.strip():

    question = question.strip()
    request_id = str(uuid.uuid4())

    all_department = detect_all_department_query(question)
    salary_analytics_requested = is_department_salary_analytics_question(question)
    selected = route_question(question)

    if salary_analytics_requested:
        selected = ["server1", "server2"]

    emit(
        request_id,
        "QUESTION_ROUTED",
        question=question,
        selected_servers=selected
    )

    results = {}
    retrieval_traces = {}
    timings = {}
    errors = []
    clarification_message = None
    not_found_message = None

    employee_id_match = re.search(
        r"\bEMP\d+\b", question.upper()
    )
    employee_id = employee_id_match.group(0) if employee_id_match else None
    id_department_lookup = bool(employee_id and "server2" in selected)

    requested_employee_name = extract_employee_name_lookup(question)

    called_servers = []
    salary_analytics_result = None

    with st.status(
        "Retrieving employee information...",
        expanded=True
    ) as status:

        if salary_analytics_requested:
            st.write("Fetching complete employee and department datasets...")
            called_servers.extend(["server1", "server2"])

            try:
                salary_analytics_result, analytics_timings = (
                    fetch_department_salary_analytics(request_id)
                )
                timings.update(analytics_timings)
            except Exception as error:
                errors.append(
                    "Full-dataset salary analytics failed validation: "
                    + str(error)
                )
                emit(
                    request_id,
                    "FULL_DATASET_SALARY_ANALYTICS_FAILED",
                    error=str(error)
                )

        for name in ([] if salary_analytics_requested else selected):

            url = SERVER1 if name == "server1" else SERVER2

            # For an ID-based department question, Server 1 must
            # resolve the employee before Server 2 is contacted.
            record_ref = None
            department_filter = None
            record_refs = None

            if all_department and name == "server2":
                department_filter = all_department
                st.write(
                    f"Retrieving every {all_department} department record from SERVER 2..."
                )

            elif all_department and name == "server1":
                department_data = results.get("server2", {})
                department_records = department_data.get("records", [])
                record_refs = list(dict.fromkeys(
                    str(row.get("record_ref", "")).strip()
                    for row in department_records
                    if row.get("record_ref")
                ))
                expected_count = department_data.get(
                    "retrieval_trace", {}
                ).get("total_department_matches", len(department_records))

                if (
                    not record_refs
                    or len(record_refs) != len(department_records)
                    or len(record_refs) != expected_count
                ):
                    not_found_message = (
                        "The department service did not return a complete "
                        "set of unique employee references. Profile retrieval "
                        "was stopped to avoid presenting an incomplete list."
                    )
                    emit(
                        request_id,
                        "DEPARTMENT_REFERENCE_VALIDATION_FAILED",
                        department=all_department,
                        returned_records=len(department_records),
                        unique_references=len(record_refs),
                        expected_records=expected_count
                    )
                    break

                st.write(
                    f"Retrieving {len(record_refs)} matching employee profiles from SERVER 1..."
                )

            elif name == "server2" and id_department_lookup:
                employee_data = results.get("server1", {})
                employee_records = employee_data.get("records", [])

                if len(employee_records) != 1:
                    not_found_message = (
                        f"I could not uniquely verify employee "
                        f"{employee_id} in Server 1. "
                        "The department service was not queried."
                    )
                    emit(
                        request_id,
                        "EMPLOYEE_REFERENCE_NOT_RESOLVED",
                        employee_id=employee_id,
                        employee_record_count=len(employee_records)
                    )
                    break

                record_ref = employee_records[0].get("record_ref")

                if not record_ref:
                    not_found_message = (
                        "The employee was found, but no record reference "
                        "was returned. The department service was not queried."
                    )
                    emit(
                        request_id,
                        "EMPLOYEE_REFERENCE_MISSING",
                        employee_id=employee_id
                    )
                    break

                st.write("Calling SERVER 2 using the verified record reference...")

            else:
                st.write(f"Calling {name.upper()}...")

            requested_top_k = (
                len(record_refs) if record_refs
                else (1 if record_ref else 10)
            )
            retrieval_mode = (
                "exact_department_filter" if department_filter
                else (
                    "exact_record_ref_batch" if record_refs
                    else ("exact_record_ref" if record_ref else "question_query")
                )
            )

            emit(
                request_id,
                "SERVICE_CALL_STARTED",
                service=name,
                endpoint=url + "/search",
                http_method="POST",
                retrieval_mode=retrieval_mode,
                requested_top_k=requested_top_k
            )

            data, elapsed, error = call_server(
                url,
                name,
                question,
                request_id,
                record_ref=record_ref,
                department=department_filter,
                record_refs=record_refs
            )

            timings[name] = elapsed
            called_servers.append(name)

            if error:
                errors.append(error)
                emit(
                    request_id,
                    "SERVICE_ERROR",
                    service=name,
                    error=error
                )

                if all_department:
                    break

                if name == "server1" and id_department_lookup:
                    not_found_message = (
                        "I could not verify the employee in Server 1, "
                        "so I did not query the department service."
                    )
                    break

            else:
                results[name] = data
                retrieval_traces[name] = data.get("retrieval_trace", {})

                records = data.get("records", [])
                retrieval_method = data.get("retrieved_using", "")
                # Do not use similar hybrid results as evidence for a
                # person-specific department lookup without an exact name.
                if (
                    requested_employee_name
                    and name == "server2"
                    and retrieval_method != "exact_employee_name"
                ):
                    not_found_message = (
                        f"I couldn't find an exact employee named "
                        f"{requested_employee_name} in the department records. "
                        "Similar names were not used as evidence. "
                        "Please check the spelling or provide the employee ID."
                    )

                    # Remove similar-name candidates from the answer context.
                    results[name] = {**data, "records": []}

                    emit(
                        request_id,
                        "EMPLOYEE_NAME_NOT_FOUND",
                        service=name,
                        requested_name=requested_employee_name,
                        retrieval_method=retrieval_method
                    )
                    break


                emit(
                    request_id,
                    "SERVICE_RESULT",
                    service=name,
                    result_count=len(records),
                    fields=sorted({
                        key
                        for row in records
                        for key in row.keys()
                    }),
                    retrieval_trace=retrieval_traces[name]
                )

                if all_department and name == "server1":
                    if len(records) != len(record_refs):
                        message = (
                            "Server 1 returned "
                            f"{len(records)} of {len(record_refs)} requested "
                            "employee profiles. The department-wide answer "
                            "was stopped because the profile results are incomplete."
                        )
                        errors.append(message)
                        emit(
                            request_id,
                            "DEPARTMENT_PROFILE_COUNT_MISMATCH",
                            expected_count=len(record_refs),
                            returned_count=len(records)
                        )
                        break

                # A name can represent multiple different employees.
                if (
                    retrieval_method == "exact_employee_name"
                    and len(records) > 1
                ):
                    clarification_message = (
                        "I found multiple employees with that name. "
                        "Please provide the employee ID so I can identify "
                        "the correct employee without guessing."
                    )
                    emit(
                        request_id,
                        "AMBIGUOUS_EMPLOYEE_NAME",
                        service=name,
                        match_count=len(records)
                    )
                    break

                if name == "server1" and employee_id:
                    if len(records) == 0:
                        not_found_message = (
                            f"Employee {employee_id} was not found "
                            "in the employee profile records. "
                            "No employee details are available."
                        )
                        emit(
                            request_id,
                            "EMPLOYEE_ID_NOT_FOUND",
                            employee_id=employee_id
                        )
                        break

                    if id_department_lookup and len(records) != 1:
                        not_found_message = (
                            f"Employee ID {employee_id} did not return "
                            "one unique employee record. The department "
                            "service was not queried."
                        )
                        break

        context = compact_context(results, include_all=bool(all_department))
        selected_evidence = (
            context[0].get("records", []) if context else []
        )

        selected_refs = [
            item.get("record_ref")
            for item in selected_evidence
            if item.get("record_ref")
        ]

        emit(
            request_id,
            "GLOBAL_EVIDENCE_SELECTED",
            selection_method=(
                context[0].get("selection_method") if context else None
            ),
            candidate_count=(
                context[0].get("candidate_count", 0) if context else 0
            ),
            selected_count=len(selected_evidence),
            selected_record_refs=selected_refs,
            selected_scores=[
                {
                    "record_ref": item.get("record_ref"),
                    "combined_rrf_score": item.get(
                        "retrieval", {}
                    ).get("combined_rrf_score", 0.0)
                }
                for item in selected_evidence
            ]
        )

        emit(
            request_id,
            "GEMINI_CONTEXT_READY",
            services=list(results.keys()),
            record_count=len(selected_evidence),
            selected_record_refs=selected_refs
        )

        gemini_error = None
        gemini_ms = 0
        usage = {}

        if clarification_message:
            answer = clarification_message
            source = "Clarification Required"

        elif not_found_message:
            answer = not_found_message
            source = "Employee Lookup Not Completed"

        elif errors:
            answer = (
                "I couldn't calculate the department salary averages safely. "
                "The complete datasets could not be retrieved or validated. "
                "Please check the service and dataset details below."
            )
            gemini_error = "; ".join(errors)
            source = "Full-Dataset Analytics Validation Error"

        elif salary_analytics_requested and salary_analytics_result:
            highest = ", ".join(
                salary_analytics_result["highest_departments"]
            )
            answer = (
                f"**Highest average annual salary:** {highest}\n\n"
                f"**Average annual salary:** "
                f"{salary_analytics_result['highest_average']}\n\n"
                f"**Employees analyzed:** "
                f"{salary_analytics_result['total_employees']}\n\n"
                f"**Departments analyzed:** "
                f"{salary_analytics_result['department_count']}\n\n"
                "The calculation used all 150 employee records, joined "
                "to all 150 department records using unique record references."
            )
            source = "Validated Full-Dataset Analytics (Python)"

        else:
            st.write("Generating answer...")

            answer, gemini_error, gemini_ms, usage = gemini_answer(
                question,
                context,
                request_id=request_id
            )

            if answer:
                source = "Gemini Cloud"
            else:
                answer = "Gemini could not generate an answer. Please check the Gemini error details below."
                source = "Gemini Error"

        emit(
            request_id,
            "REQUEST_COMPLETED",
            source=source,
            gemini_error=gemini_error,
            selected_servers=selected,
            called_servers=called_servers
        )

        status.update(
            label="Request completed",
            state="complete",
            expanded=False
        )


    # =====================================================
    # ANSWER
    # =====================================================

    st.markdown('<div class="section-card">', unsafe_allow_html=True)

    st.subheader("✨ Assistant Answer")
    st.write(answer)

    if salary_analytics_requested and salary_analytics_result:
        st.subheader("Department-wise Salary Analysis")
        st.caption(
            "All 150 employee records validated and included. "
            "Averages are calculated in INR."
        )
        st.table(salary_analytics_result["rows"])

    st.caption(f"Answer source: {source}")

    st.caption(
        "Services called: "
        + (", ".join(server.upper() for server in called_servers) or "None")
    )

    st.markdown('</div>', unsafe_allow_html=True)


    # =====================================================
    # ERRORS
    # =====================================================

    if errors:
        st.markdown(
            '<div class="error-panel">'
            '<div class="error-panel-title">⚠ SYSTEM ERROR — RETRIEVAL SERVICE</div>'
            '<div class="error-panel-message">'
            + "<br>".join(
                __import__("html").escape(str(error)) for error in errors
            )
            + '</div></div>',
            unsafe_allow_html=True
        )

    elif gemini_error:
        st.markdown(
            '<div class="error-panel">'
            '<div class="error-panel-title">⚠ GEMINI GENERATION ERROR</div>'
            '<div class="error-panel-message">'
            + __import__("html").escape(str(gemini_error))
            + '</div></div>',
            unsafe_allow_html=True
        )


    # =====================================================
    # METRICS
    # =====================================================

    st.subheader("Request Performance")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Server 1 Response",
        f"{timings.get('server1', 0)} ms"
        if "server1" in timings else "Not called"
    )

    c2.metric(
        "Server 2 Response",
        f"{timings.get('server2', 0)} ms"
        if "server2" in timings else "Not called"
    )

    c3.metric(
        "Gemini Tokens",
        usage.get("total_tokens")
        if usage.get("total_tokens") is not None
        else "Not reported"
    )


    # =====================================================
    # TECHNICAL DETAILS
    # =====================================================

    with st.expander("Request Routing and Returned Fields"):

        st.json({
            "request_id": request_id,
            "selected_servers": selected,
            "called_servers": called_servers,
            "service_fields": {
                name: sorted({
                    field
                    for row in data.get("records", [])
                    for field in row.keys()
                })
                for name, data in results.items()
            },
            "record_counts": {
                name: len(data.get("records", []))
                for name, data in results.items()
            },
            "retrieval_traces": retrieval_traces,
            "global_selection": context[0] if context else {},
            "token_usage": usage
        })


    with st.expander("Retrieved Records — Local Services"):
        st.json(results)

    with st.expander("Selected Evidence Sent to Gemini"):
        st.json(context)

    with st.expander("Hybrid Retrieval Traces"):
        st.json(retrieval_traces)


    with st.expander("Request Audit Trace"):

        events = []

        if TRACE.exists():
            for line in TRACE.read_text(
                encoding="utf-8"
            ).splitlines():

                try:
                    event = json.loads(line)

                    if event.get("request_id") == request_id:
                        events.append(event)

                except Exception:
                    pass

        st.markdown("### RAG Execution Timeline")

        if not events:
            st.info("No audit events were found for this request.")

        else:
            for event in events:
                stage = event.get("stage", "UNKNOWN")
                timestamp = event.get("timestamp", "")

                st.markdown(f"**{timestamp} — {stage}**")

                if stage == "QUESTION_ROUTED":
                    st.write("Entry point: Streamlit chatbot / Server 3")
                    st.write("Selected services:", event.get("selected_servers", []))

                elif stage == "SERVICE_CALL_STARTED":
                    st.write(
                        f"Backend: {event.get('service')} · "
                        f"{event.get('http_method')} · "
                        f"{event.get('endpoint')}"
                    )
                    st.write(
                        f"Retrieval mode: {event.get('retrieval_mode')} · "
                        f"Requested top_k: {event.get('requested_top_k')}"
                    )

                elif stage == "SERVICE_RESULT":
                    trace = event.get("retrieval_trace", {})
                    st.write(
                        f"Backend: {event.get('service')} · "
                        f"Returned records: {event.get('result_count')}"
                    )
                    st.write(
                        f"Method: {trace.get('retrieval_method', 'Not reported')} · "
                        f"Embedding provider: {trace.get('embedding_provider', 'Not applicable')} · "
                        f"Embedding model: {trace.get('embedding_model', 'Not applicable')}"
                    )

                    if trace:
                        st.write(
                            f"Corpus records: {trace.get('corpus_record_count', 'Not reported')} · "
                            f"Cache hit: {trace.get('cache_hit', 'Not reported')} · "
                            f"RRF K: {trace.get('rrf_k', 'Not applicable')}"
                        )
                        st.write(
                            f"Index build: {trace.get('index_build_ms', 'Not reported')} ms · "
                            f"Query embedding: {trace.get('query_embedding_ms', 'Not reported')} ms · "
                            f"Retrieval: {trace.get('total_retrieval_ms', 'Not reported')} ms"
                        )

                        rankings = trace.get("rankings", [])
                        if rankings:
                            st.caption("Actual returned ranking details")
                            st.dataframe(rankings, use_container_width=True)

                    st.write("Returned fields:", event.get("fields", []))

                elif stage == "SERVICE_ERROR":
                    st.warning(
                        f"{event.get('service')}: {event.get('error')}"
                    )

                elif stage == "GLOBAL_EVIDENCE_SELECTED":
                    selection_limit = (
                        "all matching records"
                        if event.get("selection_method") == "all_department_matches"
                        else "maximum 5"
                    )
                    st.write(
                        f"Selection: {event.get('selection_method')} · "
                        f"Candidates: {event.get('candidate_count')} · "
                        f"Selected: {event.get('selected_count')} "
                        f"({selection_limit})"
                    )
                    st.write("Selected record references:", event.get("selected_record_refs", []))
                    st.write("Combined RRF scores:", event.get("selected_scores", []))

                elif stage == "GEMINI_CONTEXT_READY":
                    st.write(
                        f"Evidence prepared: {event.get('record_count')} records · "
                        f"Services: {event.get('services')}"
                    )
                    st.write("Evidence references:", event.get("selected_record_refs", []))

                elif stage == "GEMINI_REQUEST_STARTED":
                    st.write(
                        f"Cloud handoff: {event.get('provider')} · "
                        f"Model: {event.get('model')}"
                    )
                    st.write("Endpoint:", event.get("endpoint"))
                    st.write(
                        f"Context records: {event.get('context_record_count')} · "
                        f"Prompt characters: {event.get('prompt_characters')}"
                    )

                elif stage == "GEMINI_RESPONSE_RECEIVED":
                    st.write(
                        f"HTTP status: {event.get('http_status')} · "
                        f"Generation time: {event.get('generation_ms')} ms"
                    )
                    st.write("Token usage:", event.get("token_usage", {}))

                elif stage == "GEMINI_REQUEST_FAILED":
                    st.warning(
                        f"Gemini generation failed: {event.get('error')}"
                    )

                elif stage == "REQUEST_COMPLETED":
                    st.write("Final answer source:", event.get("source"))
                    st.write("Called services:", event.get("called_servers"))
                    st.write("Selected services:", event.get("selected_servers"))

                else:
                    st.write({
                        key: value
                        for key, value in event.items()
                        if key not in ("timestamp", "request_id", "component", "stage")
                    })

                st.divider()

        with st.expander("Raw Audit Events (JSON)"):
            st.json(events)


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Employee Advanced Hybrid RAG · "
    "Local retrieval services · Gemini cloud generation · "
    "Request routing and audit logging"
)

st.caption(
    "Logs: logs/server1_employee.log · "
    "logs/server2_department.log · "
    "logs/chatbot_router.log · "
    "logs/request_trace.jsonl"
)

from __future__ import annotations
import csv, re, json, logging, time, uuid
from pathlib import Path
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from shared.hybrid_retrieval import hybrid_retrieve, HybridRetrievalError

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "server1_employee_records.csv"
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("server1_employee")
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.FileHandler(LOG_DIR / "server1_employee.log", encoding="utf-8")
    h.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
    logger.addHandler(h)
    logger.addHandler(logging.StreamHandler())

FIELDS = [
    "employee_id", "name", "designation", "annual_salary_inr",
    "joining_date", "office_location", "employment_status", "record_ref"
]

def load_rows():
    with DATA.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    if any("department" in r for r in rows):
        raise RuntimeError("Security boundary violation: Server 1 must not contain department.")

    if any(not r.get("record_ref", "").strip() for r in rows):
        raise RuntimeError("Data validation error: Every employee must have a record_ref.")

    return rows

def norm(s):
    return " ".join(re.findall(r"[a-z0-9]+", str(s or "").lower()))

def employee_document(row):
    fields = [
        "employee_id", "name", "designation", "annual_salary_inr",
        "joining_date", "office_location", "employment_status"
    ]
    return " | ".join(
        f"{field.replace('_', ' ')}: {row.get(field, '')}"
        for field in fields
    )


def retrieve(query, rows, top_k=5, record_refs=None):
    if record_refs:
        rows_by_ref = {
            str(r.get("record_ref", "")).strip().upper(): r
            for r in rows
        }
        matches = [
            rows_by_ref[ref.strip().upper()]
            for ref in record_refs
            if ref.strip().upper() in rows_by_ref
        ]
        trace = {
            "retrieval_method": "exact_record_ref_batch",
            "exact_match": True,
            "requested_reference_count": len(record_refs),
            "returned_record_count": len(matches),
        }
        return matches, "exact_record_ref_batch", trace

    q = norm(query)
    id_match = re.search(r"\bEMP\d+\b", str(query).upper())

    if id_match:
        employee_id = id_match.group(0)
        found = [
            r for r in rows
            if r["employee_id"].upper() == employee_id
        ]
        trace = {
            "retrieval_method": "exact_employee_id",
            "exact_match": True,
            "returned_record_count": len(found),
        }
        return found, "exact_employee_id", trace

    found = [
        r for r in rows
        if norm(r["name"]) and norm(r["name"]) in q
    ]
    if found:
        selected = found[:top_k]
        trace = {
            "retrieval_method": "exact_employee_name",
            "exact_match": True,
            "returned_record_count": len(selected),
        }
        return selected, "exact_employee_name", trace

    return hybrid_retrieve(
        query=query,
        rows=rows,
        text_builder=employee_document,
        top_k=top_k,
        cache_namespace="server1_employee",
        identity_field="employee_id",
    )

class SearchRequest(BaseModel):
    query: str = ""
    top_k: int = Field(default=5, ge=1, le=10)
    record_refs: list[str] | None = None

app = FastAPI(
    title="Server 1 - Employee Profile RAG API",
    version="1.1",
    description="Private employee profile retrieval. Contains employee IDs and profile fields; department data is excluded."
)

@app.get("/")
def root():
    return {
        "server": "Server 1 - Employee Profile",
        "health": "/health",
        "search": "/search"
    }

@app.get("/health")
def health():
    rows = load_rows()
    return {
        "status": "healthy",
        "server": "Server 1 - Employee Profile",
        "record_count": len(rows),
        "stored_fields": FIELDS,
        "department_field_present": False,
        "record_ref_present": True
    }

@app.post("/search")
def search(req: SearchRequest, x_request_id: str | None = Header(default=None)):
    rid = x_request_id or str(uuid.uuid4())
    start = time.perf_counter()
    rows = load_rows()
    try:
        matches, method, retrieval_trace = retrieve(req.query, rows, req.top_k, req.record_refs)
    except HybridRetrievalError as error:
        logger.error(json.dumps({
            "request_id": rid,
            "server": "SERVER_1_EMPLOYEE_PROFILE",
            "stage": "RETRIEVAL_ERROR",
            "error": str(error),
        }, ensure_ascii=False))
        raise HTTPException(status_code=503, detail=str(error))

    ms = round((time.perf_counter() - start) * 1000, 2)

    logger.info(json.dumps({
        "request_id": rid,
        "server": "SERVER_1_EMPLOYEE_PROFILE",
        "stage": "RETRIEVAL_COMPLETE",
        "method": method,
        "result_count": len(matches),
        "duration_ms": ms
    }, ensure_ascii=False))

    return {
        "request_id": rid,
        "server": "server1",
        "retrieved_using": method,
        "record_count": len(matches),
        "records": matches,
        "duration_ms": ms,
        "retrieval_trace": retrieval_trace
    }


@app.get("/records")
def all_employee_records(x_request_id: str | None = Header(default=None)):
    rid = x_request_id or str(uuid.uuid4())
    start = time.perf_counter()

    rows = load_rows()
    records = [
        {field: row.get(field, "") for field in FIELDS}
        for row in rows
    ]

    refs = [str(row["record_ref"]).strip().upper() for row in records]

    if len(refs) != len(set(refs)):
        raise HTTPException(
            status_code=500,
            detail="Duplicate employee record references detected."
        )

    ms = round((time.perf_counter() - start) * 1000, 2)

    logger.info(json.dumps({
        "request_id": rid,
        "server": "SERVER_1_EMPLOYEE_PROFILE",
        "stage": "COMPLETE_EMPLOYEE_DATASET_READ",
        "record_count": len(records),
        "duration_ms": ms
    }, ensure_ascii=False))

    return {
        "request_id": rid,
        "server": "server1",
        "record_count": len(records),
        "records": records,
        "duration_ms": ms
    }

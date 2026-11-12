from __future__ import annotations
import csv, re, json, logging, time, uuid
from pathlib import Path
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from shared.hybrid_retrieval import hybrid_retrieve, HybridRetrievalError

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "server2_department_records.csv"
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("server2_department")
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.FileHandler(LOG_DIR / "server2_department.log", encoding="utf-8")
    h.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
    logger.addHandler(h)
    logger.addHandler(logging.StreamHandler())

ALLOWED_FIELDS = {"record_ref", "name", "department"}
FORBIDDEN_FIELDS = {"employee_id", "annual_salary_inr", "designation",
                    "joining_date", "office_location", "employment_status"}

def load_rows():
    with DATA.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        if fields != ALLOWED_FIELDS:
            raise RuntimeError(
                f"Security boundary violation: Server 2 fields must be exactly "
                f"{sorted(ALLOWED_FIELDS)}; found {sorted(fields)}"
            )
        rows = list(reader)

    if any(not r.get("record_ref", "").strip() for r in rows):
        raise RuntimeError("Data validation error: Every department record needs a record_ref.")

    return rows

def norm(s):
    return " ".join(re.findall(r"[a-z0-9]+", str(s or "").lower()))

def department_document(row):
    fields = ["record_ref", "name", "department"]
    return " | ".join(
        f"{field.replace('_', ' ')}: {row.get(field, '')}"
        for field in fields
    )


def retrieve(query, rows, top_k=5, record_ref=None, department=None):
    if department:
        matches = [
            r for r in rows
            if norm(r.get("department", "")) == norm(department)
        ]
        trace = {
            "retrieval_method": "exact_department_filter",
            "exact_match": True,
            "department_filter": department,
            "returned_record_count": len(matches),
            "total_department_matches": len(matches),
        }
        return matches, "exact_department_filter", trace

    if record_ref:
        matches = [
            r for r in rows
            if r["record_ref"].strip().upper() == record_ref.strip().upper()
        ]
        trace = {
            "retrieval_method": "exact_record_ref",
            "exact_match": True,
            "returned_record_count": len(matches[:1]),
        }
        return matches[:1], "exact_record_ref", trace

    q = norm(query)
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
        text_builder=department_document,
        top_k=top_k,
        cache_namespace="server2_department",
        identity_field="record_ref",
    )

class SearchRequest(BaseModel):
    query: str = ""
    record_ref: str | None = None
    department: str | None = None
    top_k: int = Field(default=5, ge=1, le=10)

app = FastAPI(
    title="Server 2 - Department Directory API",
    version="1.1",
    description="Department-only lookup service. Contains record references, employee names and department labels only."
)

@app.get("/")
def root():
    return {"server": "Server 2 - Department Directory", "health": "/health", "search": "/search"}

@app.get("/health")
def health():
    rows = load_rows()
    return {
        "status": "healthy",
        "server": "Server 2 - Department Directory",
        "record_count": len(rows),
        "stored_fields": ["record_ref", "name", "department"],
        "employee_id_present": False,
        "salary_present": False,
        "record_ref_present": True
    }

@app.post("/search")
def search(req: SearchRequest, x_request_id: str | None = Header(default=None)):
    rid = x_request_id or str(uuid.uuid4())
    start = time.perf_counter()
    rows = load_rows()
    try:
        matches, method, retrieval_trace = retrieve(
            req.query, rows, req.top_k, req.record_ref, req.department
        )
    except HybridRetrievalError as error:
        logger.error(json.dumps({
            "request_id": rid,
            "server": "SERVER_2_DEPARTMENT_ONLY",
            "stage": "RETRIEVAL_ERROR",
            "error": str(error),
        }, ensure_ascii=False))
        raise HTTPException(status_code=503, detail=str(error))

    ms = round((time.perf_counter() - start) * 1000, 2)

    logger.info(json.dumps({
        "request_id": rid,
        "server": "SERVER_2_DEPARTMENT_ONLY",
        "stage": "RETRIEVAL_COMPLETE",
        "method": method,
        "result_count": len(matches),
        "duration_ms": ms
    }, ensure_ascii=False))

    return {
        "request_id": rid,
        "server": "server2",
        "retrieved_using": method,
        "record_count": len(matches),
        "records": matches,
        "retrieval_trace": retrieval_trace,
        "duration_ms": ms
    }


@app.get("/records")
def all_department_records(x_request_id: str | None = Header(default=None)):
    rid = x_request_id or str(uuid.uuid4())
    start = time.perf_counter()

    rows = load_rows()
    records = [
        {
            "record_ref": row["record_ref"],
            "name": row["name"],
            "department": row["department"]
        }
        for row in rows
    ]

    refs = [str(row["record_ref"]).strip().upper() for row in records]

    if len(refs) != len(set(refs)):
        raise HTTPException(
            status_code=500,
            detail="Duplicate department record references detected."
        )

    ms = round((time.perf_counter() - start) * 1000, 2)

    logger.info(json.dumps({
        "request_id": rid,
        "server": "SERVER_2_DEPARTMENT_ONLY",
        "stage": "COMPLETE_DEPARTMENT_DATASET_READ",
        "record_count": len(records),
        "duration_ms": ms
    }, ensure_ascii=False))

    return {
        "request_id": rid,
        "server": "server2",
        "record_count": len(records),
        "records": records,
        "duration_ms": ms
    }

# Employee Advanced RAG — Two Isolated Local Servers + Gemini Cloud

## Architecture and data boundaries

- **Server 1 — Employee Profile API (FastAPI, port 8081):** stores employee ID, name, designation, annual salary, joining date, office location, and employment status. It deliberately does **not** store department.
- **Server 2 — Department Directory API (FastAPI, port 8082):** stores only `name` and `department`. It deliberately does **not** store employee IDs, salary, addresses, or other profile fields.
- **Chatbot / Router (Streamlit, port 8501):** classifies each question and calls only the service(s) required. It sends only the selected service result(s), not the CSV files, to Gemini.
- **Gemini Cloud:** generates a natural-language answer from the small retrieved context. It does not receive the full 150-row dataset.

## Routing rules

| Example question | Service(s) called |
|---|---|
| Which department is Aarav Sharma in? | Server 2 only |
| What is the employee ID of Aarav Sharma? | Server 1 only |
| What is Aarav Sharma's salary? | Server 1 only |
| Give Aarav Sharma's employee ID and department | Server 1 + Server 2 |

The router and each API write trace events. Returned field names are logged; full payload logging is not enabled.

## Important data note

The supplied source CSV contains `office_location`, not a residential address. Server 1 preserves the available office location and does not invent a home address. The original combined CSV is split into two separate files in `data/`; there is no combined master CSV in this deliverable.

## Windows setup

Open PowerShell in this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, do not get stuck. Run the interpreter directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Set Gemini key in the current PowerShell window (do not paste it into chat or source code):

```powershell
$env:GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
```

Open **three separate PowerShell windows**, each with the project folder as its current directory.

### Window 1 — Start Server 1

```powershell
python -m uvicorn server1_employee_api.main:app --host 127.0.0.1 --port 8081
```
Health: http://127.0.0.1:8081/health  
API docs: http://127.0.0.1:8081/docs

### Window 2 — Start Server 2

```powershell
python -m uvicorn server2_department_api.main:app --host 127.0.0.1 --port 8082
```
Health: http://127.0.0.1:8082/health  
API docs: http://127.0.0.1:8082/docs

### Window 3 — Start Streamlit chatbot

```powershell
python -m streamlit run chatbot/app.py
```
Open the local URL Streamlit prints (normally http://localhost:8501).

## Quick API tests

Server 1 only:
```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8081/search -ContentType 'application/json' -Body '{"query":"What is the salary of Aarav Sharma?"}'
```

Server 2 only:
```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8082/search -ContentType 'application/json' -Body '{"query":"Which department is Aarav Sharma in?"}'
```

## Validation checklist

1. `/health` on port 8081 reports 150 records, no department field.
2. `/health` on port 8082 reports 150 records, only name and department fields.
3. Ask a department-only question and verify the UI says **SERVER2** only.
4. Ask an employee-ID or salary question and verify the UI says **SERVER1** only.
5. Ask for both ID and department and verify both servers are called.
6. If Gemini key is not configured, local retrieval still returns a fallback result. Gemini cloud generation requires a valid API key and internet access.

## Honest implementation note

This starter uses deterministic exact-name/ID matching and weighted lexical retrieval over structured employee fields. It does **not** claim dense embeddings, a vector database, BM25, or RRF. Those require a separately implemented and evaluated indexing pipeline. The immediate deliverable prioritizes the requested server isolation and routing boundary without misrepresenting components.

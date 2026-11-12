@echo off
cd /d "%~dp0"
python -m uvicorn server1_employee_api.main:app --host 127.0.0.1 --port 8081
pause

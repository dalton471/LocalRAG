@echo off
cd /d "%~dp0"
python -m uvicorn server2_department_api.main:app --host 127.0.0.1 --port 8082
pause

@echo off
cd /d "%~dp0"
python -m streamlit run chatbot\app.py
pause

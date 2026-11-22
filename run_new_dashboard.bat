@echo off
setlocal
cd /d "%~dp0"
set PYTHONPATH=%~dp0;%~dp0Target-MAS-Dataset-main;%PYTHONPATH%

echo ========================================================
echo   TraceProfiler Multi-Agent Optimization Dashboard
echo ========================================================
echo Starting Streamlit app on Python 3.12 (dashboard.py)...
"C:\Users\radhi\AppData\Local\Programs\Python\Python312\python.exe" -m streamlit run dashboard.py
pause


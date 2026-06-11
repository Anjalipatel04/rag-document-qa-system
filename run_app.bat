@echo off
REM ============================================================
REM  RAG Document QA — App Launcher
REM  Double-click to start the Streamlit app
REM ============================================================

echo  Starting RAG Document QA...
call venv\Scripts\activate.bat
streamlit run app.py --server.port 8501 --server.headless false
pause

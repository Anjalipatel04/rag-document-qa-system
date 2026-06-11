@echo off
REM ============================================================
REM  RAG Document QA — Windows Setup Script
REM  Double-click this file OR run in Command Prompt
REM ============================================================

echo.
echo  ================================================
echo   RAG-Powered Document QA — Setup
echo  ================================================
echo.

REM Check Python
python --version >nul 2>&1
IF ERRORLEVEL 1 (
    echo [ERROR] Python not found. Download from https://python.org
    echo         Make sure to check "Add Python to PATH" during install.
    pause
    exit /b 1
)

echo [1/4] Creating virtual environment...
python -m venv venv
IF ERRORLEVEL 1 ( echo [ERROR] Failed to create venv. & pause & exit /b 1 )

echo [2/4] Activating virtual environment...
call venv\Scripts\activate.bat

echo [3/4] Installing dependencies (this takes 3-5 min first time)...
pip install --upgrade pip -q
pip install -r requirements.txt
IF ERRORLEVEL 1 ( echo [ERROR] pip install failed. Check your internet. & pause & exit /b 1 )

echo [4/4] Setting up .env file...
IF NOT EXIST .env (
    copy .env.example .env
    echo.
    echo  [ACTION NEEDED] Open .env in Notepad and add your Groq API key.
    echo  Get a FREE key at: https://console.groq.com
    echo.
    notepad .env
)

echo.
echo  ================================================
echo   Setup complete!
echo   Run the app with:  streamlit run app.py
echo   OR double-click:   run_app.bat
echo  ================================================
echo.
pause

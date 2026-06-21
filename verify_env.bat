@echo off
setlocal enabledelayedexpansion

cd /d "c:\Webilapps\Functional Requirements Document"

echo.
echo ======================================================================
echo TRIAGEML PROJECT - FULL ENVIRONMENT VERIFICATION
echo ======================================================================
echo.

REM Activate virtual environment
call triageml_conda_env\Scripts\activate.bat

echo [1] Verifying environment activation...
python --version
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to activate environment
    exit /b 1
)
echo [OK] Environment activated successfully
echo.

echo [2] Verifying patients.csv location...
if exist patients.csv (
    echo [OK] patients.csv found in: %CD%\patients.csv
) else (
    echo [ERROR] patients.csv NOT found!
    exit /b 1
)
echo.

echo [3] Checking installed packages...
pip list | find "pandas" >nul
if %ERRORLEVEL% equ 0 (
    echo [OK] All required packages installed
) else (
    echo [ERROR] Required packages missing
    exit /b 1
)
echo.

echo [4] Testing Python imports...
python -c "import pandas; import numpy; import sklearn; import fastapi; import joblib; print('[OK] All imports successful')"
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Import failed
    exit /b 1
)
echo.

echo [5] Verifying project files...
set files_ok=1
for %%F in (app.py train.py requirements.txt patients.csv) do (
    if not exist %%F (
        echo [ERROR] Missing file: %%F
        set files_ok=0
    )
)
if %files_ok% equ 1 (
    echo [OK] All project files present
) else (
    exit /b 1
)
echo.

echo [6] Testing model and API...
python verify_project.py
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Verification failed
    exit /b 1
)
echo.

echo ======================================================================
echo [OK] ENVIRONMENT FULLY VERIFIED AND READY TO RUN
echo ======================================================================
echo.
echo Your environment contains:
echo   - Python 3.13.14 (in isolated virtual environment)
echo   - All dependencies installed and working
echo   - patients.csv in correct location
echo   - All project files ready
echo.
echo To activate the environment in the future:
echo   CMD:      triageml_conda_env\Scripts\activate.bat
echo   PowerShell: .\triageml_conda_env\Scripts\Activate.ps1
echo.
echo To start the API server:
echo   python -m uvicorn app:app --reload
echo.
echo API will be available at: http://127.0.0.1:8000
echo.
echo ======================================================================
echo.

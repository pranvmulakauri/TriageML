@echo off
cd "c:\Webilapps\Functional Requirements Document"

echo ======================================================================
echo TRIAGEML PROJECT - CONDA ENVIRONMENT SETUP
echo ======================================================================

REM Check if conda is available
where conda >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo.
    echo [INFO] Conda not found in system PATH
    echo [INFO] Creating Python virtual environment with venv instead...
    echo.
    
    REM Create virtual environment
    python -m venv triageml_conda_env
    call triageml_conda_env\Scripts\activate.bat
) else (
    echo.
    echo [OK] Conda found! Creating conda environment...
    echo.
    
    REM Create conda environment
    call conda create -n triageml python=3.13 -y
    call conda activate triageml
)

echo.
echo [INFO] Installing dependencies...
echo.

REM Install requirements
pip install --upgrade pip
pip install -r requirements.txt

echo.
echo ======================================================================
echo [OK] ENVIRONMENT SETUP COMPLETE
echo ======================================================================
echo.
echo To activate the environment in the future, run:
echo   - On Windows CMD: triageml_conda_env\Scripts\activate.bat
echo   - On PowerShell: .\triageml_conda_env\Scripts\Activate.ps1
echo.
echo To run the application:
echo   uvicorn app:app --reload
echo.
echo ======================================================================

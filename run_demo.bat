@echo off
TITLE ENGINE-TWIN: DRDO SIH26054 Digital Twin Ground Station
echo ======================================================================
echo   ENGINE-TWIN: MALE UAV Aero Piston Engine Digital Twin (SIH26054)
echo ======================================================================
echo   Starting Real-Time Physics Simulator, EKF, AI Engine and Dashboard...
echo.
cd /d "%~dp0"

python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not found in your system PATH.
    echo Please install Python 3.10 or 3.11 and check "Add Python to PATH".
    pause
    exit /b 1
)

python main.py
if errorlevel 1 (
    echo.
    echo ======================================================================
    echo [ERROR] An error occurred while running ENGINE-TWIN.
    echo If modules are missing, install dependencies with:
    echo   pip install -r requirements.txt
    echo ======================================================================
)
pause

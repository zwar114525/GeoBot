@echo off
REM GeoBot - One-time setup when opening repo on a new device
echo Installing GeoBot dependencies...
pip install -r requirements.txt
if %ERRORLEVEL% EQU 0 (
    echo.
    echo Setup complete. Run: streamlit run app.py
) else (
    echo Setup failed. Check your Python/pip installation.
    exit /b 1
)

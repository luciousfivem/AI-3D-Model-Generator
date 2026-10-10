@echo off
rem Starts Forge3D without leaving a console window open.
cd /d "%~dp0.."
if not exist ".venv\Scripts\pythonw.exe" (
    echo Forge3D is not installed yet. Run:  powershell -ExecutionPolicy Bypass -File scripts\install.ps1
    pause
    exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" -m forge3d

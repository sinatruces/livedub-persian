@echo off
rem Starts the livedub web page on http://localhost:8000 (Windows).
setlocal
cd /d "%~dp0"
title livedub

where py >nul 2>nul && (set "PY=py -3") || (set "PY=python")
%PY% -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>nul
if errorlevel 1 (
  echo Python 3.11 or newer is needed. Install it from https://www.python.org/downloads/
  echo and tick "Add python.exe to PATH" during setup, then run this file again.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating the Python environment ^(first run only^)...
  %PY% -m venv .venv || goto :failed
)
echo Installing/updating packages...
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -q -r requirements.txt || goto :failed

where ffmpeg >nul 2>nul || echo WARNING: ffmpeg was not found. Install it with:  winget install Gyan.FFmpeg

".venv\Scripts\python.exe" -m livedub.web --open
pause
exit /b 0

:failed
echo Setup failed; see the messages above.
pause
exit /b 1

@echo off
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (set PY=py -3) else (set PY=python)
if not exist .venv (
  %PY% --version >nul 2>nul || (
    echo Python isn't installed. Get it from https://www.python.org/downloads/ and tick "Add python.exe to PATH" during setup, then run this again.
    pause
    exit /b 1
  )
  echo First run: setting up, this takes a minute...
  %PY% -m venv .venv
  .venv\Scripts\python -m pip install --upgrade pip >nul
)
:run
rem Picks up anything new in requirements.txt after an update. Quick when nothing changed.
.venv\Scripts\python -m pip install -q -r requirements.txt
.venv\Scripts\python -m dwarf_overview
rem Exit code 3 means an update was installed: start again without opening another browser tab.
if %errorlevel%==3 (
  set DWARF_OVERVIEW_NO_BROWSER=1
  goto run
)
pause

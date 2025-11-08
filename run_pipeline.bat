@echo off
setlocal enabledelayedexpansion

rem Determine repo root (folder of this script)
set "ROOT=%~dp0"
rem Trim trailing backslash if present
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

rem Locate python.exe from likely locations (current folder first, then erd-env)
set "PY_EXE="
if exist "%ROOT%\python.exe" set "PY_EXE=%ROOT%\python.exe"
if not defined PY_EXE if exist "%ROOT%\Scripts\python.exe" set "PY_EXE=%ROOT%\Scripts\python.exe"
if not defined PY_EXE if exist "%ROOT%\erd-env\python.exe" set "PY_EXE=%ROOT%\erd-env\python.exe"
if not defined PY_EXE if exist "%ROOT%\erd-env\Scripts\python.exe" set "PY_EXE=%ROOT%\erd-env\Scripts\python.exe"
if not defined PY_EXE (
  echo [ERROR] Could not find python.exe in "%ROOT%" or "%ROOT%\erd-env". >&2
  echo         Ensure your packed env is extracted in this folder or under "erd-env". >&2
  exit /b 1
)

rem Accept config path as first argument; default to repo config.yaml
set "CONFIG=%~1"
if "%CONFIG%"=="" set "CONFIG=%ROOT%\config.yaml"

echo Running ERD pipeline...
"%PY_EXE%" "%ROOT%\run_erd_pipeline.py" --config "%CONFIG%"
set ERR=%ERRORLEVEL%
if %ERR% NEQ 0 (
  echo [ERROR] Pipeline failed with exit code %ERR%. >&2
  exit /b %ERR%
)

echo Done.
exit /b 0

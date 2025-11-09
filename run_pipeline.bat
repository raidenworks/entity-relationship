@echo off
setlocal enabledelayedexpansion

rem Determine repo root (folder of this script)
set "ROOT=%~dp0"
rem Trim trailing backslash if present
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

rem Expected environment directory (update if different)
set "ENV_DIR=%ROOT%\erd-env"

rem Validate env exists
if not exist "%ENV_DIR%\python.exe" if not exist "%ENV_DIR%\Scripts\python.exe" (
  echo [ERROR] Environment not found under "%ENV_DIR%". >&2
  echo         Run unpack_env.bat or place the env under "erd-env". >&2
  exit /b 1
)

rem Detect if env already active (python from PATH within ENV_DIR)
set "ALREADY="
for /f "delims=" %%P in ('where python 2^>nul') do (
  set "PY_FOUND=%%P"
  goto :gotpy
)
:gotpy
if defined PY_FOUND (
  set "TMP=!PY_FOUND:%ENV_DIR%=!"
  if not "!TMP!"=="!PY_FOUND!" set "ALREADY=1"
)

if defined ALREADY (
  echo [INFO] Using already-active environment: %ENV_DIR%
) else (
  rem "Activate" by prepending env paths for this process
  set "_OLD_PATH=%PATH%"
  set "PATH=%ENV_DIR%;%ENV_DIR%\Scripts;%ENV_DIR%\Library\bin;%ENV_DIR%\Library\usr\bin;%PATH%"
  echo [INFO] Activated environment: %ENV_DIR%
)

rem Accept config path as first argument; default to repo config (yaml|yml)
set "CONFIG=%~1"
if "%CONFIG%"=="" (
  if exist "%ROOT%\config.yaml" (
    set "CONFIG=%ROOT%\config.yaml"
  ) else if exist "%ROOT%\config.yml" (
    set "CONFIG=%ROOT%\config.yml"
  ) else (
    echo [ERROR] No config file found. Provide a path or add config.yaml/.yml in the repo root.>&2
    exit /b 1
  )
)

echo Running ERD pipeline...
python "%ROOT%\run_erd_pipeline.py" --config "%CONFIG%"
set ERR=%ERRORLEVEL%
if %ERR% NEQ 0 (
  echo [ERROR] Pipeline failed with exit code %ERR%. >&2
  exit /b %ERR%
)

echo Done.
exit /b 0

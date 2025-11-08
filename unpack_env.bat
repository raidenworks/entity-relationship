@echo off
setlocal

rem Unpack a conda-packed environment into the SAME directory as the tar.gz
rem Usage:
rem   unpack_env.bat [PATH_TO_CONDA_PACK_TAR_GZ]
rem If no argument is provided, looks for "erd-env.tar.gz" next to this script.

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

set "PACK=%~1"
if "%PACK%"=="" set "PACK=%ROOT%\erd-env.tar.gz"

if not exist "%PACK%" (
  echo [ERROR] File not found: "%PACK%" >&2
  echo         Place erd-env.tar.gz next to this script or pass a tar.gz path.>&2
  exit /b 1
)

rem Directory containing the tarball (base for extraction)
set "PACK_DIR=%~dp1"
if "%PACK%"=="%ROOT%\erd-env.tar.gz" set "PACK_DIR=%ROOT%"
if "%PACK_DIR:~-1%"=="\" set "PACK_DIR=%PACK_DIR:~0,-1%"
set "DEST=%PACK_DIR%\erd-env"

rem Try to extract with tar (Windows 10+ includes bsdtar as tar)
if not exist "%DEST%" mkdir "%DEST%"
echo Extracting "%PACK%" to "%DEST%" ...
tar -xzf "%PACK%" -C "%DEST%" 2>nul
if errorlevel 1 (
  tar -xf "%PACK%" -C "%DEST%"
  if errorlevel 1 (
    echo [ERROR] Extraction failed. Ensure 'tar' is available on PATH.>&2
    exit /b 1
  )
)

rem Run conda-unpack if present to fix absolute paths
set "UNPACK_EXE=%DEST%\Scripts\conda-unpack.exe"
set "MARK_FILE=%DEST%\.conda-unpacked"
if exist "%UNPACK_EXE%" (
  if not exist "%MARK_FILE%" (
    echo Running conda-unpack ...
    "%UNPACK_EXE%"
    if errorlevel 1 (
      echo [WARN] conda-unpack reported a failure. You may need to re-run this.>&2
    ) else (
      >"%MARK_FILE%" echo unpacked
    )
  ) else (
    echo [INFO] Environment already unpacked (marker found). Skipping.
  )
) else (
  echo [INFO] conda-unpack.exe not found at "%UNPACK_EXE%". Skipping path fix.
)

echo [OK] Environment ready at "%DEST%".
exit /b 0

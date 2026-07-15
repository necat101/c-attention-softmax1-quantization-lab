@echo off
setlocal EnableDelayedExpansion

cd /d "%~dp0"

REM resolve zig
if defined ZIG_BIN if exist "%ZIG_BIN%" (
  set "ZIG=%ZIG_BIN%"
  goto :zig_found
)
where zig >nul 2>nul
if %ERRORLEVEL%==0 (
  for /f "delims=" %%i in ('where zig') do set "ZIG=%%i" & goto :zig_found
)
if exist "%USERPROFILE%\.local\bin\zig.exe" set "ZIG=%USERPROFILE%\.local\bin\zig.exe" & goto :zig_found
if exist "%USERPROFILE%\bin\zig.exe" set "ZIG=%USERPROFILE%\bin\zig.exe" & goto :zig_found
echo zig not found - set ZIG_BIN or install zig 0.14+ >&2
exit /b 1

:zig_found
REM resolve python
if defined PYTHON_BIN if exist "%PYTHON_BIN%" (
  set "PY=%PYTHON_BIN%"
  goto :py_found
)
where python >nul 2>nul
if %ERRORLEVEL%==0 (
  for /f "delims=" %%i in ('where python') do set "PY=%%i" & goto :py_found
)
where python3 >nul 2>nul
if %ERRORLEVEL%==0 (
  for /f "delims=" %%i in ('where python3') do set "PY=%%i" & goto :py_found
)
echo python not found >&2
exit /b 1

:py_found
set "ZIG_BIN=%ZIG%"
set "PYTHON_BIN=%PY%"

echo zig:
"%ZIG%" version
echo python:
"%PY%" --version
echo.

"%PY%" run_lab.py
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
echo.
echo running tests...
"%PY%" -m unittest -v

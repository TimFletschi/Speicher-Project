@echo off
setlocal
set "ROOT=%~dp0"
if not exist "%ROOT%\.venv\Scripts\python.exe" set "ROOT=%~dp0.."
set "PY=%ROOT%\.venv\Scripts\python.exe"
set "SCRIPT=%ROOT%\Storage_Test.py"
"%PY%" "%SCRIPT%" %*
endlocal

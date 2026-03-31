@echo off
setlocal
set "ROOT=%~dp0"
if not exist "%ROOT%\.venv\Scripts\python.exe" set "ROOT=%~dp0.."
set "PY=%ROOT%\.venv\Scripts\python.exe"
set "SCRIPT=%ROOT%\plotting_machine.py"
"%PY%" "%SCRIPT%" %*
endlocal

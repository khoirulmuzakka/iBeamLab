@echo off
setlocal
cd /d "%~dp0"
python tools\build_local.py %*
exit /b %errorlevel%

@echo off
setlocal
cd /d "%~dp0"
python tools\package_gpu_runtime.py %*
exit /b %errorlevel%

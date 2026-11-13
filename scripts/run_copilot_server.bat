@echo off
setlocal
set "TERRA_ROOT=%~dp0"
if exist "%~dp0serve_qwen_copilot.py" if not exist "%~dp0train_gpu.py" goto :ready
set "TERRA_ROOT=%~dp0.."
:ready
cd /d "%TERRA_ROOT%"
if defined VIRTUAL_ENV goto :run
if not exist ".venv\Scripts\activate.bat" (
  echo Run run_training.bat first to create the trained environment and adapter.
  exit /b 1
)
call ".venv\Scripts\activate.bat"
:run
python scripts\serve_qwen_copilot.py --auto-port %*
exit /b %errorlevel%

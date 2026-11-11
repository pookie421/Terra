@echo off
setlocal
cd /d "%~dp0"
where python >nul 2>nul
if errorlevel 1 goto :failed
python -c "import sys; assert (3,11) <= sys.version_info[:2] < (3,13), 'Use Python 3.11 or 3.12 for the pinned environment'"
if errorlevel 1 goto :failed
if defined VIRTUAL_ENV goto :environment_ready
if not exist ".venv\Scripts\python.exe" python -m venv .venv
if errorlevel 1 goto :failed
call ".venv\Scripts\activate.bat"
:environment_ready
python -m pip install --upgrade pip
if errorlevel 1 goto :failed
python -m pip install -r requirements.txt
if errorlevel 1 goto :failed
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 goto :failed
rem Dependencies are installed before CUDA visibility and actual kernel checks.
python train.py --smoke-only --check-torch
if errorlevel 1 goto :failed
if /I "%~1"=="--qlora" goto :qlora
python train.py --skip-dataset-build
if errorlevel 1 goto :failed
echo XGBoost finished. QLoRA is optional: run_training.bat --qlora
exit /b 0
:qlora
python -m pip install -r requirements-qwen.txt
if errorlevel 1 goto :failed
python train.py --train-llm
if errorlevel 1 goto :failed
exit /b 0
:failed
echo Training failed. Preserve the output and inspect artifacts. No readiness is implied.
exit /b 1

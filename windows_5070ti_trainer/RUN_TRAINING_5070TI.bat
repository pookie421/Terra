@echo off
setlocal
cd /d "%~dp0"

echo TERRA-SHIELD portable RTX 5070 Ti trainer
where python >nul 2>nul
if errorlevel 1 (
  echo ERROR: Python 3.11 or newer is required. Install Python and enable PATH.
  pause
  exit /b 1
)
python -c "import sys; raise SystemExit(0 if sys.version_info >= (3,11) else 1)"
if errorlevel 1 (
  echo ERROR: Python 3.11 or newer is required.
  pause
  exit /b 1
)
where nvidia-smi >nul 2>nul
if errorlevel 1 (
  echo ERROR: NVIDIA driver tools were not found. Install a current NVIDIA driver.
  pause
  exit /b 1
)
nvidia-smi
if errorlevel 1 (
  echo ERROR: nvidia-smi could not query the NVIDIA GPU.
  pause
  exit /b 1
)

if not exist ".venv_5070ti\Scripts\python.exe" (
  python -m venv .venv_5070ti
  if errorlevel 1 goto :failed
)
call ".venv_5070ti\Scripts\activate.bat"
python -m pip install --upgrade pip
if errorlevel 1 goto :failed

echo Installing CUDA 13.0 PyTorch wheels for Blackwell / RTX 50 series...
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
if errorlevel 1 goto :failed
python -c "import torch; assert torch.cuda.is_available(), 'PyTorch cannot see the NVIDIA GPU'; print(torch.cuda.get_device_name(0))"
if errorlevel 1 goto :failed
python -m pip install -r requirements.txt
if errorlevel 1 goto :failed
python -m pip install -r requirements-qwen.txt
if errorlevel 1 goto :failed

python train_all_on_5070ti.py --train-llm
set RESULT=%ERRORLEVEL%
if not "%RESULT%"=="0" (
  echo Training did not complete. Review the message above and artifacts\scientific_validation_report.json.
  pause
  exit /b %RESULT%
)
echo Training artifacts are in %CD%\artifacts
pause
exit /b 0

:failed
echo Setup failed. Check the NVIDIA driver, internet connection, and Python installation.
pause
exit /b 1

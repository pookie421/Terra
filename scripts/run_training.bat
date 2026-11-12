@echo off
setlocal
rem Works both in canonical scripts/ and as the generated bundle root launcher.
set "TERRA_ROOT=%~dp0"
set "TERRA_ENTRY=train.py"
if exist "%~dp0train.py" goto :root_ready
if not exist "%~dp0train_gpu.py" goto :failed
set "TERRA_ROOT=%~dp0.."
set "TERRA_ENTRY=scripts\train_gpu.py"
:root_ready
cd /d "%TERRA_ROOT%"
set "TERRA_DEVICE=cuda"
set "TERRA_QLORA=0"
set "TERRA_LLM_ONLY=0"
set "TERRA_SMOKE_TEST=0"
set "TERRA_SMOKE_ONLY=0"
:parse
if "%~1"=="" goto :parsed
if /I "%~1"=="--cpu" (
  set "TERRA_DEVICE=cpu"
  shift
  goto :parse
)
if /I "%~1"=="--qlora" (
  set "TERRA_QLORA=1"
  shift
  goto :parse
)
if /I "%~1"=="--llm-only" (
  set "TERRA_QLORA=1"
  set "TERRA_LLM_ONLY=1"
  shift
  goto :parse
)
if /I "%~1"=="--smoke-test" (
  set "TERRA_QLORA=1"
  set "TERRA_SMOKE_TEST=1"
  shift
  goto :parse
)
if /I "%~1"=="--smoke-only" (
  set "TERRA_SMOKE_ONLY=1"
  shift
  goto :parse
)
echo Unknown option: %~1
echo Usage: run_training.bat [--cpu] [--qlora ^| --llm-only ^| --smoke-test ^| --smoke-only]
exit /b 2
:parsed
if "%TERRA_DEVICE%"=="cpu" if "%TERRA_QLORA%"=="1" (
  echo QLoRA requires BF16 CUDA; CPU and FP16 fallbacks are disabled.
  exit /b 2
)
if "%TERRA_SMOKE_ONLY%"=="1" if "%TERRA_QLORA%"=="1" (
  echo Use --smoke-only for XGBoost, or --smoke-test for the longest-trajectory QLoRA test.
  exit /b 2
)
where python >nul 2>nul
if errorlevel 1 goto :failed
python -c "import sys; assert (3,11) <= sys.version_info[:2] < (3,13), 'Use Python 3.11 or 3.12 for the pinned environment'"
if errorlevel 1 goto :failed
if defined VIRTUAL_ENV goto :environment_ready
if not exist ".venv\Scripts\python.exe" python -m venv .venv
if errorlevel 1 goto :failed
call ".venv\Scripts\activate.bat"
:environment_ready
python -c "import sys; assert (3,11) <= sys.version_info[:2] < (3,13), 'The selected virtual environment must use Python 3.11 or 3.12'"
if errorlevel 1 goto :failed
set "TOKENIZERS_PARALLELISM=false"
python -m pip install --upgrade pip
if errorlevel 1 goto :failed
python -m pip install -r requirements-training.txt
if errorlevel 1 goto :failed
if "%TERRA_DEVICE%"=="cpu" goto :run
rem Never accept CUDA visibility without actual architecture/kernel verification.
python scripts\hardware_preflight.py --output artifacts\hardware_preflight.json
if not errorlevel 1 goto :cuda_ready
echo Installing the official PyTorch CUDA 12.8 build. A compatible NVIDIA driver is also required.
python -m pip install --upgrade --force-reinstall torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 goto :failed
rem Re-apply the numerical pins if torch's dependency resolver changed NumPy.
python -m pip install -r requirements-training.txt
if errorlevel 1 goto :failed
python scripts\hardware_preflight.py --output artifacts\hardware_preflight.json
if errorlevel 1 goto :failed
:cuda_ready
if "%TERRA_QLORA%"=="0" goto :run
rem Require the native Windows wheel; do not source-compile bitsandbytes or FlashAttention.
python -m pip install --only-binary=bitsandbytes -r requirements-qwen.txt
if errorlevel 1 goto :failed
rem HF installs may alter torch. Re-check actual kernels after the complete stack is installed.
python scripts\hardware_preflight.py --require-bf16 --check-bitsandbytes --output artifacts\hardware_preflight.json
if errorlevel 1 goto :failed
:run
if "%TERRA_SMOKE_ONLY%"=="1" goto :xgb_smoke
if "%TERRA_SMOKE_TEST%"=="1" goto :qlora_smoke
if "%TERRA_LLM_ONLY%"=="1" goto :llm_only
if "%TERRA_QLORA%"=="1" goto :qlora
python "%TERRA_ENTRY%" --device %TERRA_DEVICE%
if errorlevel 1 goto :failed
echo XGBoost finished. Optional QLoRA: run_training.bat --qlora
exit /b 0
:xgb_smoke
python "%TERRA_ENTRY%" --device %TERRA_DEVICE% --smoke-only
if errorlevel 1 goto :failed
exit /b 0
:qlora_smoke
python "%TERRA_ENTRY%" --device cuda --smoke-test
if errorlevel 1 goto :failed
exit /b 0
:llm_only
python "%TERRA_ENTRY%" --device cuda --llm-only
if errorlevel 1 goto :failed
exit /b 0
:qlora
python "%TERRA_ENTRY%" --device cuda --train-llm
if errorlevel 1 goto :failed
exit /b 0
:failed
echo Training failed. Preserve the output and inspect artifacts. No readiness is implied.
exit /b 1

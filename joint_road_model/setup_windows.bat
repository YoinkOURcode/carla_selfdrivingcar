@echo off
REM Usage: setup_windows.bat [CARLA version]   e.g. setup_windows.bat 0.9.15   (default 0.9.16)
setlocal
cd /d "%~dp0"
set "CARLA_VER=%~1"
if "%CARLA_VER%"=="" set "CARLA_VER=0.9.16"

py -3.10 --version >nul 2>&1
if errorlevel 1 goto nopython

if not exist ".venv\Scripts\python.exe" py -3.10 -m venv .venv
if errorlevel 1 goto failed
set "PY=.venv\Scripts\python.exe"
"%PY%" -m pip install --upgrade pip
if errorlevel 1 goto failed

where nvidia-smi >nul 2>&1
if errorlevel 1 goto cpu
echo NVIDIA GPU found: installing PyTorch with CUDA support.
"%PY%" -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 goto failed
goto rest

:cpu
echo No NVIDIA GPU found: installing CPU PyTorch. The model will run at only a few frames per second.
"%PY%" -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto failed

:rest
"%PY%" -m pip install -r requirements_windows.txt
if errorlevel 1 goto failed
"%PY%" -m pip install carla==%CARLA_VER%
if errorlevel 1 goto nocarla

"%PY%" -c "import torch,carla,cv2,flask,ultralytics;print('torch',torch.__version__,'| CUDA available:',torch.cuda.is_available(),'| ultralytics',ultralytics.__version__)"
if errorlevel 1 goto failed
echo.
echo Setup finished. Next run: run_selftest.bat
exit /b 0

:nopython
echo Python 3.10 was not found. Install 64-bit Python 3.10 from python.org, then run this again.
exit /b 1
:nocarla
echo Could not install carla==%CARLA_VER% for Python 3.10. This setup supports CARLA 0.9.15 and 0.9.16.
exit /b 1
:failed
echo Setup failed. Read the error above.
exit /b 1

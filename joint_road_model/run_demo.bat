@echo off
REM Browser demo. Optional: run_demo.bat C:\path\to\Mark   to browse your full frame folder.
setlocal
cd /d "%~dp0"
set "DEVICE=cpu"
for /f %%d in ('.venv\Scripts\python.exe -c "import torch;print('cuda' if torch.cuda.is_available() else 'cpu')"') do set "DEVICE=%%d"
start "" http://127.0.0.1:8790/
if "%~1"=="" (
  .venv\Scripts\python.exe demo_app.py --device %DEVICE%
) else (
  .venv\Scripts\python.exe demo_app.py --device %DEVICE% --frames "%~1"
)

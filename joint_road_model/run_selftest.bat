@echo off
setlocal
cd /d "%~dp0"
set "DEVICE=cpu"
for /f %%d in ('.venv\Scripts\python.exe -c "import torch;print('cuda' if torch.cuda.is_available() else 'cpu')"') do set "DEVICE=%%d"
.venv\Scripts\python.exe selftest.py --device %DEVICE%
exit /b %errorlevel%

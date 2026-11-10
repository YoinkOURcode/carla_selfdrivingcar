@echo off
REM Start CARLA first. All arguments go to carla_live_test.py, e.g.:
REM   run_carla_test.bat --check
REM   run_carla_test.bat --town Town03 --seconds 120 --out carla_runs\town03
setlocal
cd /d "%~dp0"
.venv\Scripts\python.exe carla_live_test.py %*
exit /b %errorlevel%

@echo off
setlocal
cd /d "%~dp0"

where python >nul 2>nul
if %errorlevel%==0 (
  python local_server.py
  exit /b %errorlevel%
)

set "BLENDER_PY=C:\Program Files\Blender Foundation\Blender 4.3\4.3\python\bin\python.exe"
if exist "%BLENDER_PY%" (
  "%BLENDER_PY%" local_server.py
  exit /b %errorlevel%
)

echo Python non trovato. Installa Python oppure crea il pacchetto .exe dell'app.
pause
exit /b 1

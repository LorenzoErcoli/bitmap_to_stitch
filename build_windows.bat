@echo off
setlocal
cd /d "%~dp0"

where python >nul 2>nul
if %errorlevel%==0 (
  set "PYTHON=python"
) else (
  set "PYTHON=C:\Program Files\Blender Foundation\Blender 4.3\4.3\python\bin\python.exe"
)

if not exist "%PYTHON%" if not "%PYTHON%"=="python" (
  echo Python non trovato.
  exit /b 1
)

"%PYTHON%" -m PyInstaller --clean --noconfirm bitmap_to_stitch_local.spec
if errorlevel 1 exit /b %errorlevel%

echo.
echo Build completata:
echo dist\BitmapToStitch\BitmapToStitch.exe

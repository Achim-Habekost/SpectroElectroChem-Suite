@echo off
setlocal
cd /d "%~dp0"

set "PYTHONPATH=%~dp0src;%PYTHONPATH%"

echo Starting SpectroElectroChem Suite v6.0.0 ...
echo.

py main.py
if %errorlevel%==0 goto end

echo.
echo Starting with py failed. Trying python ...
echo.

python main.py
if %errorlevel%==0 goto end

echo.
echo Python could not start the SpectroElectroChem Suite.
echo If packages are missing, run:
echo     Install_required_Python_packages.bat
echo.

:end
endlocal
exit /b 0

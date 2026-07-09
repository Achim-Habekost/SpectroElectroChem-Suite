@echo off
setlocal
cd /d "%~dp0"

echo Installing required Python packages for SpectroElectroChem Suite v3.0 Final ...
echo.

py -m pip install -r requirements.txt
if %errorlevel%==0 goto end

echo.
echo Installation with py failed. Trying python ...
echo.

python -m pip install -r requirements.txt

:end
echo.
echo Finished.
pause

@echo off
setlocal
cd /d "%~dp0\.."

py -m pip install --upgrade pip
py -m pip install -r requirements.txt
pyinstaller packaging\SpectroElectroChem_Suite.spec --noconfirm --clean

echo.
echo Build finished. See dist\SpectroElectroChem_Suite
pause

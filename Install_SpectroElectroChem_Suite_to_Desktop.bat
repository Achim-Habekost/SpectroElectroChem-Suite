@echo off
setlocal

set "TARGET=%USERPROFILE%\OneDrive\Desktop\SpectroElectroChem_Suite_v6_0_0"

if not exist "%USERPROFILE%\OneDrive\Desktop" (
    set "TARGET=%USERPROFILE%\Desktop\SpectroElectroChem_Suite_v6_0_0"
)

echo Installing SpectroElectroChem Suite v6.0.0 to:
echo %TARGET%
echo.

mkdir "%TARGET%" 2>nul
xcopy "%~dp0*" "%TARGET%\" /E /I /Y

echo.
echo Installation finished.
echo Start with:
echo %TARGET%\Start_SpectroElectroChem_Suite.bat
echo.

pause

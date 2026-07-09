@echo off
setlocal
cd /d "%~dp0"
set "PYTHONPATH=%~dp0src;%PYTHONPATH%"

echo Testing SpectroElectroChem Suite installation ...
echo.

py -c "import sys; sys.path.insert(0, 'src'); import spectroelectrochem_suite; print('Package OK:', spectroelectrochem_suite.__version__)"
if %errorlevel% neq 0 goto error

py -c "import numpy, pandas, scipy, matplotlib, openpyxl, plotly; print('Scientific packages OK')"
if %errorlevel% neq 0 goto error

py -c "import PySide6; print('PySide6 OK')"
if %errorlevel% neq 0 (
    echo.
    echo PySide6 is missing. Please run Install_required_Python_packages.bat
)

echo.
echo Test finished.
pause
exit /b

:error
echo.
echo Test failed. Please run Install_required_Python_packages.bat
pause

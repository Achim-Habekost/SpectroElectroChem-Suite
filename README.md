## Quick start on Windows

1. Unzip this folder completely.
2. Run `Install_required_Python_packages.bat` once.
3. Run `Test_Installation.bat`.
4. Start the suite with `Start_SpectroElectroChem_Suite.bat`.

This Final package is directly runnable from the unpacked folder. It sets the local `src` folder automatically, so the error `No module named spectroelectrochem_suite` should no longer occur.

# SpectroElectroChem Suite

**Version 3.0.0**

SpectroElectroChem Suite is an open-source software package for the analysis and visualization of spectro-electrochemical data.

## Version 3.0 highlights

- Modern Qt/PySide6-based main window
- Menu bar, toolbar, module cards and status line
- Unified appearance for all modules
- Plugin registry for future methods such as IR, UV/Vis or NIR spectro-electrochemistry
- Project-folder creation with standardized subfolders
- PyInstaller configuration for Windows `.exe` builds
- Inno Setup script for a real Windows installer
- GitHub Actions workflow for automatic Windows builds
- Local PDF manual and HTML help page
- Update-check infrastructure for future GitHub Releases

## Modules

1. Raman Spectrum Analysis
2. SERS / Raman Voltammogram
3. Absorpto- / Fluorovoltammogram

## Start from source

```bash
pip install -r requirements.txt
python main.py
```

On Windows:

```text
Start_SpectroElectroChem_Suite.bat
```

## Build Windows executable

```cmd
scripts\build_exe_windows.bat
```

## Build Windows installer

After the PyInstaller build, install Inno Setup and compile:

```text
installer\SpectroElectroChem_Suite_InnoSetup.iss
```

## Acknowledgement

Parts of the source code were developed with the assistance of OpenAI ChatGPT and were subsequently validated, modified and extended by the author.

## License

MIT License.

## Citation

Please see `CITATION.cff`.


## v3.0 Final Raman waterfall update

The SERS / Raman Voltammogram module now includes:
- `Waterfall vertical offset`
- Excel sheets for custom Raman waterfall plots:
  - `Waterfall_Shifted_Values`
  - `Waterfall_Unshifted_Values`
  - `Waterfall_Offsets`

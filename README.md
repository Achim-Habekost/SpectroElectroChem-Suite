# SpectroElectroChem Suite

**Version 4.0.0 — First Public Release**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![GitHub release](https://img.shields.io/badge/release-v4.0.0-blue.svg)](https://github.com/Achim-Habekost/SpectroElectroChem-Suite/releases)
[![DOI](https://img.shields.io/badge/DOI-Zenodo%20pending-lightgrey.svg)](https://zenodo.org/)

**SpectroElectroChem Suite** is open-source software for the analysis and visualization of spectro-electrochemical data. It supports Raman spectra, Raman/SERS voltammograms, absorptovoltammograms and fluorovoltammograms.

## Main features

- Raman spectrum analysis
- Raman and SERS voltammograms
- Absorptovoltammograms
- Fluorovoltammograms
- Interactive 3D surface plots
- Interactive waterfall plots
- Waterfall values exported to Excel
- Baseline correction
- Savitzky–Golay smoothing
- Heatmap and contour visualization
- Excel, HTML, PNG and PDF export
- Plugin-ready project architecture
- Documentation and packaging templates for Windows builds

## Modules

1. **Raman Spectrum Analysis**  
   Baseline correction, smoothing, peak detection and peak assignment.

2. **SERS / Raman Voltammogram**  
   Matrix CSV input with potentials in the first row and Raman shifts in the first column. Output includes interactive surface, heatmap, contour and waterfall visualizations.

3. **Absorpto- / Fluorovoltammogram**  
   Matrix CSV input with potentials in the first row and wavelengths in the first column. Output includes Excel data, surface plots, heatmaps, contour plots and waterfall values.

## Quick start on Windows

1. Download the release archive from GitHub.
2. Unzip the archive completely.
3. Run once:

```text
Install_required_Python_packages.bat
```

4. Optional test:

```text
Test_Installation.bat
```

5. Start the program:

```text
Start_SpectroElectroChem_Suite.bat
```

## Start from source

```bash
pip install -r requirements.txt
python main.py
```

## System requirements

Recommended:

- Windows 10 or Windows 11
- Python 3.10 or newer

Required Python packages are listed in `requirements.txt` and can be installed automatically with `Install_required_Python_packages.bat`.

## Build Windows executable

```text
scripts\build_exe_windows.bat
```

## Build Windows installer

After the PyInstaller build, install Inno Setup and compile:

```text
installer\SpectroElectroChem_Suite_InnoSetup.iss
```

## File formats

The voltammogram modules use matrix CSV files:

- first row: potentials
- first column: Raman shifts or wavelengths
- remaining cells: intensities, absorptions or fluorescence values

## Citation

If you use this software in scientific work, please cite the accompanying publication and the archived software release.

Recommended software citation:

> Habekost, A. *SpectroElectroChem Suite: Software for Raman, SERS, absorption and fluorescence spectro-electrochemical data analysis*. Version 4.0.0. GitHub / Zenodo.

A formal DOI will be added after Zenodo archiving.

## Acknowledgement

Parts of the source code were developed with the assistance of OpenAI ChatGPT and were subsequently validated, modified and extended by the author.

## License

This project is distributed under the MIT License. See `LICENSE`.

## Author

Prof. Dr. Achim Habekost

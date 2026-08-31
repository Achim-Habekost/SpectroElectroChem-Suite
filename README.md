# SpectroElectroChem Suite v6.0.0

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![GitHub release](https://img.shields.io/badge/release-v6.0.0-blue.svg)](https://github.com/Achim-Habekost/SpectroElectroChem-Suite/releases)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21283231-blue.svg)](https://doi.org/10.5281/zenodo.21283231)

**SpectroElectroChem Suite** is open-source scientific software for the analysis, visualization and quantitative evaluation of spectroscopic, spectroelectrochemical and electrochemical data.

Version 6.0.0 contains **ten analysis modules** and adds the new **Electrochemical Surface Activation & SERS Analysis** workflow.

## Analysis modules

1. **Raman Spectrum Analysis**  
   Baseline correction, smoothing, peak detection, peak parameters and waterfall visualization.

2. **SERS / Raman Voltammogram**  
   Potential-resolved Raman/SERS matrices with surface, heatmap, contour and rotatable waterfall plots.

3. **Absorpto- / Fluorovoltammogram**  
   Potential-resolved absorption and fluorescence matrices with numerical and interactive output.

4. **RRDE Analysis**  
   Disk/ring-current analysis, background correction, collection efficiency, H2O2 yield, electron-transfer number, Levich, Koutecky-Levich, Tafel and Butler-Volmer evaluation.

5. **Spectro-CV Synchronization**  
   Synchronization of cyclic voltammetry with time-resolved absorption, fluorescence/ECL and Raman spectra, including multi-cycle analysis and spectral derivatives.

6. **ECL Synchronization (Integrated Signal)**  
   Synchronization of CV and non-wavelength-resolved ECL signals, cycle averaging, signal integration, concentration-series calibration and unknown-sample evaluation.

7. **Diffusion Coefficient Analysis**  
   Cottrell, Anson and Randles-Sevcik analysis, supplemented by Nicholson electron-transfer kinetics and scan-rate diagnostics.

8. **EIS Analysis**  
   Nyquist and Bode visualization, Randles-type equivalent-circuit fitting, residual analysis and Kramers-Kronig consistency checks.

9. **Stripping Voltammetry**  
   SWV and DPV concentration-series analysis with 3D waterfalls, cursor-defined integration, manual peak-current selection, calibration and unknown-sample determination.

10. **Electrochemical Surface Activation & SERS Analysis**  
    Au/Ag Raman characterization before and after electrochemical activation, apparent Raman enhancement, peak-specific enhancement and shifts, optional activation-CV analysis and wavelength-to-Raman-shift conversion.

## Outputs

Depending on the selected module, the suite creates processed Excel workbooks, static PNG/PDF figures and interactive HTML visualizations. Numerical processing steps and metadata are exported where implemented to support reproducibility and independent inspection.

## Windows installation

The recommended distribution is the ready-to-run Windows build or installer attached to the GitHub release. These variants do not require a separate Python installation.

For the Python source distribution:

1. Download and extract the release archive completely.
2. Run `Install_required_Python_packages.bat` once.
3. Optionally run `Test_Installation.bat`.
4. Start with `Start_SpectroElectroChem_Suite.bat`.

Python 3.10 or newer is required for the source distribution.

## Start from source

```bash
python -m pip install -r requirements.txt
python main.py
```

## Build Windows executable

```text
scripts\build_exe_windows.bat
```

The PyInstaller specification is located in `packaging/SpectroElectroChem_Suite.spec`.

## Build Windows installer

After building the executable, compile:

```text
installer\SpectroElectroChem_Suite_InnoSetup.iss
```

with Inno Setup.

## Documentation

The complete **User Manual v6.0** is included in `docs/User_Manual.pdf` and `docs/User_Manual_v6_0.pdf`.

Persistent concept DOI: **10.5281/zenodo.21283231**

Repository: https://github.com/Achim-Habekost/SpectroElectroChem-Suite

## Citation

If you use the suite in scientific work, please cite the archived software release through Zenodo and use the metadata in `CITATION.cff`.

Suggested form:

> Habekost, A. *SpectroElectroChem Suite*, version 6.0.0. GitHub/Zenodo, 2026. https://doi.org/10.5281/zenodo.21283231

## Scientific note on Module 10

The enhancement values reported by Module 10 are described as **apparent Raman enhancement** or **surface-activation enhancement**. They should not be interpreted as a classical SERS enhancement factor unless the numbers of molecules contributing to the reference and SERS signals are known. Before/after comparisons require comparable optical acquisition conditions.

## Acknowledgement

Parts of the source code were developed with the assistance of OpenAI ChatGPT and were subsequently reviewed, scientifically validated, modified and substantially extended by the author.

## License

MIT License. See `LICENSE`.

## Author

Prof. Dr. Achim Habekost

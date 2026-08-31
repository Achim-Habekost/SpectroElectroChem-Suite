# Changelog

## v6.0.0 - 2026-08-31

### Major release

- Expanded SpectroElectroChem Suite to **ten analysis modules**.
- Added **Module 5: Spectro-CV Synchronization** for cyclic voltammetry synchronized with absorption, fluorescence/ECL and Raman spectra.
- Added **Module 6: ECL Synchronization (Integrated Signal)** with cycle averaging, integration, concentration calibration and unknown-sample evaluation.
- Added **Module 7: Diffusion Coefficient Analysis** using Cottrell, Anson and Randles-Sevcik methods, with Nicholson kinetics and scan-rate diagnostics.
- Added **Module 8: EIS Analysis** with Nyquist/Bode visualization, equivalent-circuit fitting, residuals and Kramers-Kronig consistency checks.
- Added **Module 9: Stripping Voltammetry** for simultaneous SWV/DPV concentration-series analysis, calibration, LOD/LOQ-oriented output and unknown-sample determination.
- Added **Module 10: Electrochemical Surface Activation & SERS Analysis** for Au/Ag before/after Raman comparison, apparent enhancement, peak shifts and activation-CV analysis.
- Extended RRDE analysis with improved background correction, collection-efficiency handling, H2O2/electron-number evaluation, Levich/Koutecky-Levich, Tafel and Butler-Volmer workflows.
- Added or refined interactive 3D plots, zoom controls, cycle means, Raman baseline processing and Excel/PNG/HTML exports across modules.
- Updated the central plugin registry and launcher for all ten modules.
- Updated the User Manual to version 6.0.
- Updated repository metadata, citation information, installer/build configuration and GitHub Actions asset names for release 6.0.0.

## v4.0.0 — First Public Release

- First official public release of SpectroElectroChem Suite.
- Cleaned repository structure.
- Added `.gitignore` and `.gitattributes`.
- Removed Python cache and compiled files.
- Updated README for publication and user installation.
- Updated citation metadata.
- Added release notes.
- Raman/SERS voltammogram module includes waterfall vertical offset.
- Raman/SERS voltammogram module exports waterfall data to Excel:
  - `Waterfall_Shifted_Values`
  - `Waterfall_Unshifted_Values`
  - `Waterfall_Offsets`
- Included Raman spectrum analysis, SERS/Raman voltammograms, absorptovoltammograms and fluorovoltammograms.
- Included documentation and packaging templates for Windows builds.

## Earlier development versions

- Raman spectrum analysis.
- SERS/Raman voltammogram visualization.
- Absorpto-/fluorovoltammogram visualization.
- Baseline correction, smoothing and Excel/HTML/PDF export.
- Plugin-ready structure and central launcher architecture.

## v5.4.5 Preview
- Improved diffusion-limited plateau selection wording and diagnostics.
- Added plateau slope and relative slope to the Tafel Excel report.
- Introduced graded plateau-drift warnings.
- Clarified the scope of absolute cathodic disk currents in Levich/Koutecký–Levich calculations.

## v5.4.6 Preview
- Restored a clearly structured Tafel Excel Summary sheet.
- Added formatted sections for general settings, Tafel analysis, limiting-current analysis, and warnings.
- Added a compact limiting-current table with rotation rate, limiting current, drift, relative scatter, and plateau slope.
- Rounded rotation rates in the summary table for improved readability.
- Collected all quality warnings in a dedicated highlighted section.

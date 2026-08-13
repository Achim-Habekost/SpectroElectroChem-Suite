# Changelog

All notable changes to SpectroElectroChem Suite are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [5.5.3] - 2026-08-13

### Added

- Potential-interpolated background correction using a separate N2/background RRDE measurement.
- Excel documentation of original, interpolated background, corrected and smoothed-corrected disk and ring currents.
- Application and installer icon support.

### Changed

- Background measurements are matched to the O2 measurement by rotation rate.
- Background currents are interpolated onto the potential grid of the O2 measurement before subtraction.
- RRDE Excel reporting was extended with a dedicated background-correction worksheet.
- PyInstaller and Inno Setup configurations were updated for version 5.5.3.

### Fixed

- Improved handling of small potential-grid differences between O2 and N2/background measurements.
- Added validation of point count, scan direction, potential range and rotation-rate matching for background subtraction.

## [5.5.0] - 2026-07-30

### Added

- Rotating ring–disk electrode (RRDE) analysis module.
- Direct CSV import for paired disk and ring current data.
- Calculation of the electron-transfer number.
- Levich and Koutecký–Levich analyses.
- Mass-transport-corrected kinetic current calculation.
- Tafel analysis with Tafel slope, exchange current and exchange current density.
- Reaction-aware Butler–Volmer evaluation.
- Calculation of the standard heterogeneous rate constant for applicable simple redox couples.
- Manual and measurement-based ring-background correction.
- Plateau diagnostics based on current scatter, drift and plateau slope.
- Structured Excel reports containing settings, results, diagnostics and warnings.
- Interactive 2D and 3D RRDE visualizations.
- Export of interactive plots as HTML.
- Separate disk- and ring-current unit settings.
- Default current display in microamperes.

### Changed

- Revised graphical main window integrating the RRDE module.
- Improved scientific terminology and explanatory GUI text.
- Improved validation of potential ranges and numerical input.
- Reorganized and expanded exported Excel workbooks.
- Updated project documentation for electrochemical and spectroelectrochemical workflows.
- Updated README with screenshots and an overview of all analysis modules.

### Fixed

- Improved handling of CSV files containing unnamed or empty columns.
- Improved interpolation of disk currents at selected Levich and Koutecký–Levich potentials.
- Improved handling of cathodic-current signs in Levich and Koutecký–Levich calculations.
- Improved warnings for insufficient data points and non-ideal diffusion-limited plateaus.

## [4.0.1] - 2026-07-10

### Added

- Raman spectrum analysis with baseline correction, smoothing, peak detection and peak assignment.
- Raman and SERS voltammogram analysis.
- Absorption and fluorescence voltammogram analysis.
- Heat maps, contour plots, waterfall plots and interactive three-dimensional visualizations.
- Excel, PNG, PDF and HTML export.
- User manual and example CSV templates.
- Archived Zenodo release with version DOI 10.5281/zenodo.21283232.

[Unreleased]: https://github.com/Achim-Habekost/SpectroElectroChem-Suite/compare/v5.5.0...HEAD
[5.5.0]: https://github.com/Achim-Habekost/SpectroElectroChem-Suite/compare/v4.0.1...v5.5.0
[4.0.1]: https://github.com/Achim-Habekost/SpectroElectroChem-Suite/releases/tag/v4.0.1

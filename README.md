# SpectroElectroChem Suite

**Version 5.5.3**

Open-source desktop software for spectroelectrochemical data processing, visualization, and quantitative analysis.

Designed for research, higher education, and advanced electrochemical data analysis.

![Platform](https://img.shields.io/badge/platform-Windows-blue)
![Python](https://img.shields.io/badge/Python-3.11+-green)
![License](https://img.shields.io/badge/license-MIT-yellow)
![Version](https://img.shields.io/badge/version-v5.5.3-red)

---

## Overview

SpectroElectroChem Suite is an integrated desktop application for the analysis, visualization, and quantitative evaluation of spectroelectrochemical data. It combines Raman spectroelectrochemistry, SERS, absorptovoltammetry, fluorovoltammetry, and rotating ring–disk electrode (RRDE) analysis within a single graphical user interface.

To our knowledge, SpectroElectroChem Suite is one of the first open-source desktop applications integrating these complementary techniques into a unified workflow.

---

## Documentation

The complete documentation is available in this repository.

- **User Manual (PDF):** `docs/User_Manual.pdf`

---

## Main Features

### Raman & SERS
- Raman spectrum analysis
- Raman and SERS voltammograms
- Baseline correction
- Savitzky–Golay smoothing
- Peak detection
- Heat maps, contour plots and waterfall plots
- Interactive 3D visualization

### UV/Vis & Fluorescence
- Absorptovoltammetry
- Fluorovoltammetry
- Interactive surface plots
- Excel and HTML export

### RRDE Analysis
- CSV import
- Disk/ring visualization
- Potential-interpolated background correction using a separate N₂ measurement
- Levich analysis
- Koutecký–Levich analysis
- Tafel analysis
- Butler–Volmer analysis
- H₂O₂ yield calculation
- Electron transfer number (n)
- Excel reports

---

## Installation

Download the latest release from GitHub and follow the instructions in the User Manual.

When using the source code:

```bash
pip install -r requirements.txt
python main.py
```

---

## System Requirements

- Windows 10 or Windows 11
- Python 3.11 or newer
- Required packages listed in `requirements.txt`

---

## Citation

If you use SpectroElectroChem Suite in scientific work, please cite both the software and its Zenodo archive.

Zenodo DOI:

https://doi.org/10.5281/zenodo.21283231

---

## License

This project is distributed under the MIT License.

See `LICENSE` for details.

---

## Acknowledgement

Parts of the software were developed with the assistance of OpenAI ChatGPT and were subsequently reviewed, validated, modified, and extended by the author.

---

## Author

Prof. Dr. Achim Habekost

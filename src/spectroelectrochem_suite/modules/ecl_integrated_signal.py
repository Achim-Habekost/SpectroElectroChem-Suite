from __future__ import annotations

import csv
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit, brentq

from PySide6.QtCore import Qt

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QFileDialog, QMessageBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QGroupBox, QTableWidget, QTableWidgetItem, QHeaderView, QDialog, QComboBox, QScrollArea, QLineEdit
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import re


def _num(x):
    try:
        return float(str(x).strip().replace(',', '.'))
    except Exception:
        return np.nan


def _read_two_blocks(path: str):
    """Read: E_CV, n currents, blank column(s), E_ECL, n ECL signals.
    Header/non-numeric rows are ignored. Number of cycles is inferred from both blocks.
    """
    rows=[]
    with open(path, 'r', encoding='utf-8-sig', errors='replace', newline='') as f:
        sample=f.read(4096); f.seek(0)
        delim=';' if sample.count(';') >= sample.count(',') else ','
        for r in csv.reader(f, delimiter=delim): rows.append(r)
    width=max(map(len,rows)); rows=[r+['']*(width-len(r)) for r in rows]
    # find first row containing numeric data in both separated blocks
    start=None; split=None
    for ir,r in enumerate(rows):
        vals=[_num(v) for v in r]
        numeric=[i for i,v in enumerate(vals) if np.isfinite(v)]
        if len(numeric)<4: continue
        # blank run after first numeric block
        for j in range(2,width-1):
            if not str(r[j]).strip() and any(str(r[k]).strip() for k in range(j+1,width)):
                k=j
                while k<width and not str(r[k]).strip(): k+=1
                if k<width and np.isfinite(_num(r[0])) and np.isfinite(_num(r[k])):
                    start=ir; split=k; break
        if start is not None: break
    if start is None: raise ValueError('Could not identify the two numeric data blocks separated by blank columns.')
    left=[]; right=[]
    for r in rows[start:]:
        lv=[_num(x) for x in r[:split]]
        rv=[_num(x) for x in r[split:]]
        # trim trailing blank/nan left columns
        while lv and not np.isfinite(lv[-1]): lv.pop()
        while rv and not np.isfinite(rv[-1]): rv.pop()
        if len(lv)>=2 and len(rv)>=2 and np.isfinite(lv[0]) and np.isfinite(rv[0]):
            left.append(lv); right.append(rv)
    nl=min(len(x) for x in left); nr=min(len(x) for x in right)
    L=np.asarray([x[:nl] for x in left],float); R=np.asarray([x[:nr] for x in right],float)
    n=min(L.shape[1]-1,R.shape[1]-1)
    if n<1: raise ValueError('No matching CV/ECL cycle columns were found.')
    return L[:,0],L[:,1:n+1],R[:,0],R[:,1:n+1],n




def _read_numeric_blocks(path: str):
    """Return contiguous numeric column blocks separated by blank columns.

    Header rows are tolerated. A block is considered active if its column contains
    numeric data in the measurement region. Each returned block is a 2-D float array.
    """
    raw = pd.read_csv(path, header=None, sep=None, engine="python", dtype=str)
    num = raw.apply(lambda c: pd.to_numeric(
        c.astype(str).str.strip().str.replace(",", ".", regex=False),
        errors="coerce"
    ))

    active = [bool(num.iloc[:, j].notna().sum() >= 3) for j in range(num.shape[1])]
    runs = []
    start = None
    for j, on in enumerate(active + [False]):
        if on and start is None:
            start = j
        elif not on and start is not None:
            runs.append((start, j))
            start = None

    blocks = []
    for a, b in runs:
        g = num.iloc[:, a:b].copy()
        # Keep rows where the first column (potential) is numeric.
        g = g[g.iloc[:, 0].notna()]
        # Drop columns that became completely empty.
        g = g.dropna(axis=1, how="all")
        if g.shape[0] >= 3 and g.shape[1] >= 2:
            blocks.append(g.to_numpy(dtype=float))
    return blocks


def _read_reference_csv(path: str, n_cycles: int):
    """Reference CSV: CV block + one ECL block."""
    blocks = _read_numeric_blocks(path)
    if len(blocks) < 2:
        raise ValueError(
            "Reference CSV must contain two numeric blocks: "
            "CV (Potential + currents) and ECL (Potential + signals)."
        )
    cvb, eclb = blocks[0], blocks[1]
    expected = n_cycles + 1
    if cvb.shape[1] < expected or eclb.shape[1] < expected:
        raise ValueError(
            f"For n={n_cycles}, each block must contain Potential + {n_cycles} cycle columns."
        )
    return {
        "ecv": cvb[:, 0],
        "cv": cvb[:, 1:expected],
        "ecl_blocks": [(eclb[:, 0], eclb[:, 1:expected])],
        "detected_blocks": len(blocks),
    }


def _read_series_csv(path: str, n_cycles: int, n_concentrations: int):
    """Series CSV: one CV block + m ECL blocks, each with Potential + n signals."""
    blocks = _read_numeric_blocks(path)
    expected_blocks = 1 + n_concentrations
    if len(blocks) != expected_blocks:
        raise ValueError(
            f"For {n_concentrations} concentrations the CSV must contain exactly "
            f"{expected_blocks} numeric blocks: 1 CV block + {n_concentrations} ECL blocks. "
            f"Detected: {len(blocks)}."
        )

    expected_cols = n_cycles + 1
    for i, b in enumerate(blocks):
        if b.shape[1] < expected_cols:
            label = "CV" if i == 0 else f"ECL block {i}"
            raise ValueError(
                f"{label} has {b.shape[1]-1} signal columns, but n={n_cycles} requires "
                f"{n_cycles} cycle columns."
            )
        if b.shape[1] > expected_cols:
            raise ValueError(
                f"Block {i+1} has {b.shape[1]-1} signal columns. "
                f"For n={n_cycles}, exactly {n_cycles} are expected."
            )

    cvb = blocks[0]
    ecl_blocks = [(b[:, 0], b[:, 1:expected_cols]) for b in blocks[1:]]
    return {
        "ecv": cvb[:, 0],
        "cv": cvb[:, 1:expected_cols],
        "ecl_blocks": ecl_blocks,
        "detected_blocks": len(blocks),
    }

class ECLIntegratedWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ECL Synchronization (Integrated Signal)")
        self.resize(1320, 980)

        self.reference_path = None
        self.series_path = None
        self.reference_data = None
        self.series_data = None
        self.unknown_path = None
        self.unknown_data = None
        self._block_integrals_cache = None
        self._block_integrals_cache_signature = None

        c = QWidget()
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(c)
        self.setCentralWidget(self.scroll_area)
        outer = QVBoxLayout(c)

        title = QLabel("ECL Synchronization (Integrated Signal)")
        title.setStyleSheet("font-size:22px; font-weight:700; color:#17365D; padding:4px 0 6px 0;")
        outer.addWidget(title)

        info = QLabel(
            "Reference CSV: CV block + one ECL block.  "
            "Coreactant-series CSV: one CV block followed by one ECL block for each concentration.  "
            "Every block contains Potential + n cycle columns and blocks are separated by blank columns."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color:#4A5568; padding-bottom:4px;")
        outer.addWidget(info)

        self.setStyleSheet("""
            QPushButton#loadReferenceButton {
                background-color: #E8F1F8;
                color: #17365D;
                border: 1px solid #9FBAD0;
                border-radius: 5px;
                padding: 5px 10px;
                font-weight: 600;
            }
            QPushButton#loadSeriesButton {
                background-color: #EAF6EC;
                color: #285A35;
                border: 1px solid #A6C8AD;
                border-radius: 5px;
                padding: 5px 10px;
                font-weight: 600;
            }
            QPushButton#primaryActionButton {
                background-color: #2F75B5;
                color: white;
                border: 1px solid #245C8F;
                border-radius: 5px;
                padding: 5px 10px;
                font-weight: 600;
            }
            QPushButton#primaryActionButton:hover { background-color: #3F86C6; }
            QPushButton#exportButton {
                background-color: #F3F5F7;
                color: #2D3748;
                border: 1px solid #BFC7D0;
                border-radius: 5px;
                padding: 5px 10px;
            }
            QComboBox, QSpinBox {
                background: white;
                border: 1px solid #BFC7D0;
                border-radius: 4px;
                padding: 2px 5px;
            }
        """)

        # Experiment dimensions.
        setup = QHBoxLayout()
        setup.addWidget(QLabel("Number of cycles, n"))
        self.n_cycles = QSpinBox()
        self.n_cycles.setRange(1, 20)
        self.n_cycles.setValue(5)
        setup.addWidget(self.n_cycles)

        setup.addSpacing(30)
        setup.addWidget(QLabel("Number of coreactant concentrations, m"))
        self.n_conc = QSpinBox()
        self.n_conc.setRange(1, 20)
        self.n_conc.setValue(5)
        setup.addWidget(self.n_conc)
        setup.addStretch(1)
        outer.addLayout(setup)

        # Exactly two input files.
        files = QHBoxLayout()
        b_ref = QPushButton("Load without coreactant CSV")
        b_ref.setObjectName("loadReferenceButton")
        b_ref.clicked.connect(self.load_reference)
        files.addWidget(b_ref)
        self.ref_label = QLabel("No reference CSV selected")
        files.addWidget(self.ref_label, 1)

        b_series = QPushButton("Load with coreactant concentration-series CSV")
        b_series.setObjectName("loadSeriesButton")
        b_series.clicked.connect(self.load_series)
        files.addWidget(b_series)
        self.series_label = QLabel("No concentration-series CSV selected")
        files.addWidget(self.series_label, 1)
        outer.addLayout(files)

        # Explicit concentration input: one concentration value for every ECL block.
        conc_box = QGroupBox("Coreactant concentration input")
        conc_box.setStyleSheet("""
            QGroupBox {
                font-weight: 700;
                color: #17365D;
                border: 1px solid #B8C7D9;
                border-radius: 7px;
                margin-top: 10px;
                padding-top: 10px;
                background: #F7FAFD;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        conc_layout = QVBoxLayout(conc_box)
        conc_note = QLabel(
            "Enter one concentration for each ECL block in the concentration-series CSV. "
            "ECL block 1 corresponds to concentration 1, ECL block 2 to concentration 2, etc."
        )
        conc_note.setWordWrap(True)
        conc_layout.addWidget(conc_note)

        self.conc_table = QTableWidget(0, 3)
        self.conc_table.setHorizontalHeaderLabels([
            "Concentration",
            "Assigned ECL block",
            "Coreactant concentration / mol L⁻¹"
        ])
        self.conc_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.conc_table.verticalHeader().setDefaultSectionSize(26)
        self.conc_table.setMinimumHeight(150)
        self.conc_table.setMaximumHeight(185)
        conc_box.setMaximumHeight(245)
        self.conc_table.setAlternatingRowColors(True)
        self.conc_table.setStyleSheet("""
            QTableWidget {
                gridline-color: #D9E2EC;
                alternate-background-color: #F1F6FB;
                background-color: white;
                selection-background-color: #D6E8FA;
                selection-color: #17365D;
            }
            QHeaderView::section {
                background-color: #DCEAF7;
                color: #17365D;
                font-weight: 700;
                padding: 5px;
                border: 1px solid #C3D5E6;
            }
            QLineEdit {
                background: #FFFDF5;
                border: 1px solid #C7B978;
                border-radius: 3px;
                padding: 3px 5px;
            }
            QLineEdit:focus {
                border: 1px solid #2F75B5;
                background: #FFFFFF;
            }
        """)
        conc_layout.addWidget(self.conc_table)
        outer.addWidget(conc_box)

        self.n_conc.valueChanged.connect(self._rebuild_concentration_rows)
        self.n_conc.valueChanged.connect(self._update_polynomial_degree_limit)
        self._rebuild_concentration_rows()
        self._update_polynomial_degree_limit()

        # Processing.
        # Optional unknown-concentration analysis (data loading only in this version).
        unknown_box = QGroupBox("Unknown concentration")
        unknown_box.setMinimumHeight(150)
        unknown_box.setStyleSheet("""
            QGroupBox {
                font-size: 15px;
                font-weight: 700;
                color: #7A3E00;
                border: 2px solid #E6A15A;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 12px;
                background-color: #FFF4E3;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
            }
            QPushButton {
                background-color: #F39C34;
                color: white;
                border: 1px solid #C8791F;
                border-radius: 5px;
                padding: 7px 14px;
                font-weight: 700;
            }
            QComboBox {
                background: white;
                border: 1px solid #D6A05E;
                border-radius: 4px;
                padding: 4px 8px;
                min-width: 80px;
            }
            QLabel {
                font-size: 13px;
            }
        """)
        unknown_layout = QVBoxLayout(unknown_box)
        unknown_choice = QHBoxLayout()
        unknown_choice.addWidget(QLabel("Do you want to analyze an unknown concentration?"))
        self.unknown_choice = QComboBox()
        self.unknown_choice.addItems(["No", "Yes"])
        self.unknown_choice.currentIndexChanged.connect(self._toggle_unknown_section)
        unknown_choice.addWidget(self.unknown_choice)
        unknown_choice.addStretch(1)
        unknown_layout.addLayout(unknown_choice)

        self.unknown_input_widget = QWidget()
        unknown_input_layout = QHBoxLayout(self.unknown_input_widget)
        unknown_input_layout.setContentsMargins(0, 4, 0, 0)
        self.unknown_load_button = QPushButton("Load CSV for unknown concentration")
        self.unknown_load_button.clicked.connect(self.load_unknown)
        unknown_input_layout.addWidget(self.unknown_load_button)
        self.unknown_label = QLabel("No unknown-concentration CSV selected")
        unknown_input_layout.addWidget(self.unknown_label, 1)
        unknown_layout.addWidget(self.unknown_input_widget)
        self.unknown_input_widget.setVisible(False)
        outer.addWidget(unknown_box)

        processing_box = QGroupBox("Processing and calibration")
        processing_box.setStyleSheet("QGroupBox { background:#F8F4FF; color:#5A3D85; font-weight:700; border:1px solid #C9B7E8; margin-top:10px; padding-top:10px; }")
        opts = QHBoxLayout(processing_box)
        self.smooth = QCheckBox("Savitzky–Golay smoothing")
        opts.addWidget(self.smooth)

        opts.addWidget(QLabel("Window"))
        self.win = QSpinBox()
        self.win.setRange(3, 101)
        self.win.setSingleStep(2)
        self.win.setValue(11)
        opts.addWidget(self.win)

        opts.addWidget(QLabel("SG polynomial"))
        self.poly = QSpinBox()
        self.poly.setRange(1, 20)
        self.poly.setValue(2)
        opts.addWidget(self.poly)

        self.showmean = QCheckBox("Show cycle mean")
        self.showmean.setChecked(True)
        opts.addWidget(self.showmean)

        self.n_cycles.valueChanged.connect(self._invalidate_integral_cache)
        self.smooth.stateChanged.connect(self._invalidate_integral_cache)
        self.win.valueChanged.connect(self._invalidate_integral_cache)
        self.poly.valueChanged.connect(self._invalidate_integral_cache)

        opts.addWidget(QLabel("Displayed concentration"))
        self.display_combo = QComboBox()
        self.display_combo.setMinimumWidth(190)
        self.display_combo.setToolTip(
            "Select which coreactant concentration (ECL block) is shown in the CV/ECL preview and PNG."
        )
        self.display_combo.currentIndexChanged.connect(self._display_concentration_changed)
        opts.addWidget(self.display_combo)

        opts.addSpacing(18)
        opts.addWidget(QLabel("Calibration fit"))
        self.calibration_fit = QComboBox()
        self.calibration_fit.addItem("Linear", "linear")
        self.calibration_fit.addItem("Polynomial degree 2", "poly2")
        self.calibration_fit.addItem("Polynomial degree 3", "poly3")
        self.calibration_fit.addItem("Polynomial degree 4", "poly4")
        self.calibration_fit.addItem("Logarithmic", "log")
        self.calibration_fit.addItem("Exponential", "exp")
        self.calibration_fit.addItem("Power law", "power")
        self.calibration_fit.addItem("4-parameter logistic (4PL)", "4pl")
        self.calibration_fit.setCurrentIndex(0)
        self.calibration_fit.setMinimumWidth(230)
        self.calibration_fit.currentIndexChanged.connect(self._calibration_fit_changed)
        self.calibration_fit.setToolTip(
            "Calibration model for integrated ECL intensity vs concentration. "
            "Linear is the default; polynomial degree is limited by the number of calibration points."
        )
        opts.addWidget(self.calibration_fit)

        opts.addStretch(1)
        outer.addWidget(processing_box)

        acts_box = QGroupBox("Analysis and export")
        acts_box.setStyleSheet("QGroupBox { background:#F3FAF3; color:#285A35; font-weight:700; border:1px solid #A6C8AD; margin-top:10px; padding-top:10px; }")
        acts = QHBoxLayout(acts_box)
        p = QPushButton("Synchronize + Plot")
        p.setObjectName("primaryActionButton")
        p.clicked.connect(self.plot)
        acts.addWidget(p)

        e = QPushButton("Export Excel")
        e.setObjectName("exportButton")
        e.clicked.connect(self.export_excel)
        acts.addWidget(e)

        png = QPushButton("Export PNG")
        png.setObjectName("exportButton")
        png.clicked.connect(self.export_png)
        acts.addWidget(png)

        cp = QPushButton("Plot ECL integral vs concentration")
        cp.setObjectName("primaryActionButton")
        cp.clicked.connect(self.plot_concentration_series)
        acts.addWidget(cp)
        acts.addStretch(1)
        outer.addWidget(acts_box)

        # Large plot.
        self.fig = Figure(figsize=(12, 7.5), tight_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setMinimumHeight(470)
        self.preview_toolbar = NavigationToolbar(self.canvas, self)
        self.preview_toolbar.setToolTip(
            "Use Home to reset, Pan to move and Zoom to select a potential region."
        )
        outer.addWidget(self.preview_toolbar)
        outer.addWidget(self.canvas, 1)

        # Summary: reference integral, currently displayed concentration integral, quotient.
        self.summary = QTableWidget(1, 3)
        self.summary.setHorizontalHeaderLabels([
            "Integral without coreactant / a.u.·V",
            "Integral with coreactant / a.u.·V",
            "Ratio without / with"
        ])
        self.summary.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        outer.addWidget(self.summary)


    def _toggle_unknown_section(self):
        enabled = self.unknown_choice.currentText() == "Yes"
        self.unknown_input_widget.setVisible(enabled)
        if not enabled:
            self.unknown_path = None
            self.unknown_data = None
            self.unknown_label.setText("No unknown-concentration CSV selected")

    def load_unknown(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Load CSV for unknown concentration", "", "CSV files (*.csv);;All files (*)"
        )
        if not p:
            return
        try:
            blocks = _read_numeric_blocks(p)
            n = int(self.n_cycles.value())
            candidates = [b for b in blocks if b.ndim == 2 and b.shape[1] >= n + 1]
            if not candidates:
                raise ValueError(
                    f"The unknown-concentration CSV must contain Potential + {n} ECL cycle columns."
                )
            block = candidates[0][:, :n+1]
            self.unknown_path = p
            self.unknown_data = {"potential": block[:, 0], "ecl": block[:, 1:n+1]}
            self.unknown_label.setText(
                f"{Path(p).name} — {n} ECL curves loaded"
            )
        except Exception as exc:
            self.unknown_path = None
            self.unknown_data = None
            QMessageBox.critical(self, "Unknown concentration", f"Could not load CSV:\n{exc}")

    def _update_polynomial_degree_limit(self, *args):
        """Disable polynomial fits whose degree exceeds n_calibration_points - 1."""
        if not hasattr(self, "calibration_fit"):
            return
        npts = int(self.n_conc.value())
        model = self.calibration_fit.model()
        for i in range(self.calibration_fit.count()):
            key = str(self.calibration_fit.itemData(i))
            enabled = True
            if key.startswith("poly"):
                enabled = int(key[4:]) <= max(1, npts - 1)
            elif key == "4pl":
                enabled = npts >= 4
            item = model.item(i)
            if item is not None:
                item.setEnabled(enabled)
        cur = str(self.calibration_fit.currentData())
        if cur.startswith("poly") and int(cur[4:]) > max(1, npts-1):
            self.calibration_fit.setCurrentIndex(0)
        if cur == "4pl" and npts < 4:
            self.calibration_fit.setCurrentIndex(0)

    def _calibration_fit_changed(self, *args):
        """Refresh an open/available calibration result on the next plot and keep Excel consistent."""
        # Selection is read directly by plotting/export routines; no stale cached fit is retained.
        pass

    def _rebuild_concentration_rows(self):
        old_conc = []
        old_blocks = []
        for r in range(self.conc_table.rowCount()):
            w = self.conc_table.cellWidget(r, 2)
            old_conc.append(w.text().strip() if isinstance(w, QLineEdit) else "")
            b = self.conc_table.cellWidget(r, 1)
            old_blocks.append(int(b.currentData()) if isinstance(b, QComboBox) else r)

        m = int(self.n_conc.value())
        self.conc_table.setRowCount(m)

        for r in range(m):
            self.conc_table.setItem(r, 0, QTableWidgetItem(f"c{r+1}"))

            block_combo = QComboBox()
            for bi in range(m):
                block_combo.addItem(f"ECL block {bi+1}", bi)
            default_block = old_blocks[r] if r < len(old_blocks) else r
            default_block = max(0, min(default_block, m - 1))
            block_combo.setCurrentIndex(default_block)
            block_combo.currentIndexChanged.connect(self._refresh_display_combo)
            self.conc_table.setCellWidget(r, 1, block_combo)

            w = QLineEdit()
            w.setPlaceholderText("e.g. 1e-4, 8e-5, 5e-5")
            if r < len(old_conc) and old_conc[r]:
                w.setText(old_conc[r])
            w.editingFinished.connect(self._refresh_display_combo)
            self.conc_table.setCellWidget(r, 2, w)

        self._refresh_display_combo()
        self._invalidate_integral_cache()

        if self.series_path:
            self.series_data = None
            self.series_label.setText(
                f"{Path(self.series_path).name} — reload required after changing m"
            )

    @staticmethod
    def _parse_concentration_text(text):
        """Parse concentration input robustly, including spaced scientific notation."""
        text = str(text).strip()
        if not text:
            return np.nan
        text = (
            text.replace(",", ".")
                .replace("−", "-")
                .replace("–", "-")
                .replace("—", "-")
                .replace("×10^", "e")
                .replace("x10^", "e")
        )
        text = re.sub(r"\s+", "", text)
        try:
            return float(text)
        except Exception:
            return np.nan

    def _refresh_display_combo(self):
        current = self.display_combo.currentIndex() if hasattr(self, "display_combo") else 0
        if not hasattr(self, "display_combo"):
            return
        self.display_combo.blockSignals(True)
        self.display_combo.clear()

        concentrations = self._concentrations()
        assignments = self._assigned_blocks()
        for r in range(int(self.n_conc.value())):
            c = concentrations[r] if r < len(concentrations) else np.nan
            b = assignments[r] if r < len(assignments) else r
            ctext = f"{c:.6g}" if np.isfinite(c) else "not set"
            self.display_combo.addItem(
                f"c{r+1} = {ctext} mol L⁻¹ → B{b+1}", r
            )

        if self.display_combo.count():
            self.display_combo.setCurrentIndex(
                min(max(current, 0), self.display_combo.count()-1)
            )
        self.display_combo.blockSignals(False)

    def _concentrations(self):
        vals = []
        for r in range(self.conc_table.rowCount()):
            w = self.conc_table.cellWidget(r, 2)
            if isinstance(w, QLineEdit):
                vals.append(self._parse_concentration_text(w.text()))
            else:
                vals.append(np.nan)
        return vals

    def _assigned_blocks(self):
        vals = []
        for r in range(self.conc_table.rowCount()):
            w = self.conc_table.cellWidget(r, 1)
            vals.append(int(w.currentData()) if isinstance(w, QComboBox) else r)
        return vals

    def load_reference(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Select without-coreactant CSV", "", "CSV (*.csv);;All files (*)"
        )
        if not p:
            return
        try:
            d = _read_reference_csv(p, int(self.n_cycles.value()))
        except Exception as exc:
            QMessageBox.critical(self, "Import error", str(exc))
            return
        self.reference_path = p
        self.reference_data = d
        self.ref_label.setText(
            f"{Path(p).name} — 1 ECL block × {self.n_cycles.value()} cycles"
        )
        self._fill_summary()

    def load_series(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Select with-coreactant concentration-series CSV", "",
            "CSV (*.csv);;All files (*)"
        )
        if not p:
            return
        n = int(self.n_cycles.value())
        m = int(self.n_conc.value())
        try:
            d = _read_series_csv(p, n, m)
        except Exception as exc:
            QMessageBox.critical(self, "Import error", str(exc))
            return
        self.series_path = p
        self.series_data = d
        self._invalidate_integral_cache()
        self.series_label.setText(
            f"{Path(p).name} — {m} ECL blocks × {n} cycles = {m*n} ECL traces"
        )
        self._refresh_display_combo()
        self._fill_summary()

    def _smooth(self, y):
        y = np.asarray(y, dtype=float)
        if not self.smooth.isChecked():
            return y
        n = len(y)
        w = min(int(self.win.value()), n if n % 2 else n - 1)
        if w < 3:
            return y
        if w % 2 == 0:
            w -= 1
        po = min(int(self.poly.value()), w - 1)
        return savgol_filter(y, w, po, mode="interp")

    @staticmethod
    def _turn_index(e):
        e = np.asarray(e, dtype=float)
        if len(e) < 4:
            return len(e) // 2
        d = np.diff(e)
        changes = np.flatnonzero(np.sign(d[:-1]) != np.sign(d[1:]))
        if len(changes):
            return int(changes[0] + 1)
        return int(np.argmax(np.abs(e - e[0])))

    def _sync_trace(self, target_e, source_e, source_y):
        target_e = np.asarray(target_e, float)
        source_e = np.asarray(source_e, float)
        source_y = np.asarray(source_y, float)

        tc = self._turn_index(target_e)
        ts = self._turn_index(source_e)
        out = np.full_like(target_e, np.nan, dtype=float)

        for t_sl, s_sl in [
            (slice(0, tc + 1), slice(0, ts + 1)),
            (slice(tc, None), slice(ts, None)),
        ]:
            x = source_e[s_sl]
            y = source_y[s_sl]
            ok = np.isfinite(x) & np.isfinite(y)
            x, y = x[ok], y[ok]
            if len(x) < 2:
                continue
            order = np.argsort(x)
            x, y = x[order], y[order]
            x, unique_idx = np.unique(x, return_index=True)
            y = y[unique_idx]
            out[t_sl] = np.interp(target_e[t_sl], x, y, left=np.nan, right=np.nan)
        return out

    def _processed_dataset(self, d, block_index=0):
        ecv = np.asarray(d["ecv"], float)
        cv = np.asarray(d["cv"], float).copy()
        eecl, ecl_raw = d["ecl_blocks"][block_index]
        ecl_raw = np.asarray(ecl_raw, float)

        n = int(self.n_cycles.value())
        cv = cv[:, :n]
        ecl_raw = ecl_raw[:, :n]

        cvp = np.column_stack([self._smooth(cv[:, k]) for k in range(n)])
        eclp = np.column_stack([
            self._sync_trace(ecv, eecl, self._smooth(ecl_raw[:, k]))
            for k in range(n)
        ])
        return ecv, cvp, eclp

    @staticmethod
    def _integrate_trace(e, y):
        e = np.asarray(e, float)
        y = np.asarray(y, float)
        ok = np.isfinite(e) & np.isfinite(y)
        pair = ok[:-1] & ok[1:]
        if not np.any(pair):
            return np.nan
        return float(np.sum(
            0.5 * (y[:-1][pair] + y[1:][pair]) *
            np.abs(np.diff(e)[pair])
        ))

    def _mean_integral(self, d, block_index=0):
        if d is None:
            return np.nan
        e, _, ecl = self._processed_dataset(d, block_index)
        mean_ecl = np.nanmean(ecl, axis=1)
        return self._integrate_trace(e, mean_ecl)

    def _selected_block_index(self):
        if not self.series_data:
            return 0
        row = self.display_combo.currentData()
        try:
            row = int(row)
        except Exception:
            row = 0
        blocks = self._assigned_blocks()
        if 0 <= row < len(blocks):
            return int(blocks[row])
        return 0

    def _fill_summary(self):
        i0 = self._mean_integral(self.reference_data, 0) if self.reference_data else np.nan
        iw = np.nan
        if self.series_data:
            iw = self._mean_integral(self.series_data, self._selected_block_index())
        ratio = i0 / iw if np.isfinite(i0) and np.isfinite(iw) and abs(iw) > 1e-30 else np.nan

        vals = [
            f"{i0:.6g}" if np.isfinite(i0) else "",
            f"{iw:.6g}" if np.isfinite(iw) else "",
            f"{ratio:.6g}" if np.isfinite(ratio) else "",
        ]
        for j, v in enumerate(vals):
            self.summary.setItem(0, j, QTableWidgetItem(v))

    def _display_concentration_changed(self, *args):
        # Do not auto-open messages while the combo is being rebuilt.
        if getattr(self, "series_data", None) is None:
            return
        # Update both the numerical summary and the complete CV/ECL preview immediately.
        # Thus every concentration selected in the combo box displays its own mean curves
        # and its corresponding individual-cycle inset without requiring another button click.
        try:
            self._fill_summary()
            self.plot()
        except Exception:
            pass

    def plot(self):
        try:
            self._refresh_display_combo()

            if self.series_data is None:
                QMessageBox.information(
                    self, "ECL",
                    "Please load the with-coreactant concentration-series CSV."
                )
                return

            concentrations = self._concentrations()
            bi = self._selected_block_index()
            if bi >= len(concentrations) or not np.isfinite(concentrations[bi]):
                QMessageBox.information(
                    self, "Coreactant concentration",
                    f"Please enter a valid concentration for ECL block {bi+1}."
                )
                return

            e, cv, ecl = self._processed_dataset(self.series_data, bi)
            if cv.size == 0 or ecl.size == 0:
                raise ValueError("No CV/ECL data available after synchronization.")

            finite_ecl = np.isfinite(ecl)
            if not np.any(finite_ecl):
                raise ValueError(
                    "Synchronization produced no finite ECL values. "
                    "Please check the CV and ECL potential ranges."
                )

            styles = ["-", "--", ":", "-."]

            self.fig.clear()
            ax = self.fig.add_subplot(111)
            ax2 = ax.twinx()

            # Main preview: show only the cycle means, large and uncluttered.
            mean_cv = np.nanmean(cv, axis=1)
            mean_ecl = np.nanmean(ecl, axis=1)
            ax.plot(e, mean_cv, color="black", lw=3.2, label="Mean CV")
            ax2.plot(e, mean_ecl, lw=3.2, label="Mean ECL")

            # Inset: retain all individual cycles so reproducibility remains visible.
            # It intentionally has no full legend; cycle identity is conveyed by line style.
            ins = ax.inset_axes([0.055, 0.55, 0.36, 0.38])
            ins2 = ins.twinx()
            for k in range(cv.shape[1]):
                ins.plot(e, cv[:, k], color="black", ls=styles[k % len(styles)],
                         lw=0.8, alpha=0.70)
            for k in range(ecl.shape[1]):
                ins2.plot(e, ecl[:, k], ls=styles[k % len(styles)],
                          lw=0.9, alpha=0.65)
            # Means are also shown in the inset as reference.
            ins.plot(e, mean_cv, color="black", lw=1.8)
            ins2.plot(e, mean_ecl, lw=1.8)
            ins.set_title("Individual cycles", fontsize=8)
            ins.tick_params(axis="both", labelsize=7)
            ins2.tick_params(axis="y", labelsize=7)
            ins.grid(alpha=0.18)

            cval = concentrations[bi]
            ax.set_title(
                f"Coreactant concentration: {cval:.6g} mol L⁻¹"
            )
            ax.set_xlabel("Potential / V")
            ax.set_ylabel("Current / µA")
            ax2.set_ylabel("ECL intensity / a.u.")
            ax.grid(alpha=0.25)

            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, loc="best", fontsize=8, ncol=2)

            self.fig.tight_layout()
            self.canvas.draw()
            self.canvas.flush_events()
            self._fill_summary()

            # Make the updated plot visible even when the upper controls are in view.
            if hasattr(self, "scroll_area"):
                self.scroll_area.ensureWidgetVisible(self.canvas, 20, 20)

        except Exception as exc:
            QMessageBox.critical(
                self,
                "Synchronize + Plot",
                f"Could not synchronize and plot the data:\n{exc}"
            )

    def _validate_concentrations(self):
        vals = self._concentrations()
        if len(vals) != int(self.n_conc.value()):
            return False

        bad = [i+1 for i,v in enumerate(vals) if (not np.isfinite(v)) or v < 0]
        if bad:
            QMessageBox.information(
                self, "Coreactant concentrations",
                "Please enter a valid non-negative concentration for every row. "
                "Invalid row(s): " + ", ".join(map(str, bad))
            )
            return False

        blocks = self._assigned_blocks()
        if len(set(blocks)) != len(blocks):
            QMessageBox.information(
                self, "ECL block assignment",
                "Each concentration must be assigned to a different ECL block."
            )
            return False
        return True

    def _invalidate_integral_cache(self, *args):
        self._block_integrals_cache = None
        self._block_integrals_cache_signature = None

    def _integral_cache_signature(self):
        return (
            id(self.series_data),
            int(self.n_cycles.value()),
            bool(self.smooth.isChecked()),
            int(self.win.value()),
            int(self.poly.value()),
        )

    def _get_block_integrals(self):
        """Calculate/freeze integrals strictly in physical left-to-right CSV block order."""
        if self.series_data is None:
            return []

        sig = self._integral_cache_signature()
        if (
            self._block_integrals_cache is not None
            and self._block_integrals_cache_signature == sig
        ):
            return list(self._block_integrals_cache)

        vals = []
        for block_index in range(len(self.series_data["ecl_blocks"])):
            vals.append(float(self._mean_integral(self.series_data, block_index)))

        self._block_integrals_cache = tuple(vals)
        self._block_integrals_cache_signature = sig
        return list(vals)

    def _calibration_model_key(self):
        try:
            return str(self.calibration_fit.currentData())
        except Exception:
            return "linear"

    def _effective_polynomial_degree(self, n_points):
        key = self._calibration_model_key()
        if key == "linear": return 1
        if key.startswith("poly"):
            return max(1, min(int(key[4:]), int(n_points)-1))
        return 1

    def _fit_calibration(self, x, y):
        x=np.asarray(x,float); y=np.asarray(y,float); key=self._calibration_model_key()
        xmin,xmax=float(np.min(x)),float(np.max(x))
        if key == "linear" or key.startswith("poly"):
            deg = 1 if key=="linear" else min(int(key[4:]), len(x)-1)
            coeff=np.polyfit(x,y,deg); f=np.poly1d(coeff)
            label="Linear fit" if deg==1 else f"Polynomial fit (degree {deg})"
            params=("poly", coeff)
        elif key == "log":
            if np.any(x<=0): raise ValueError("Logarithmic calibration requires concentrations > 0.")
            a,b=np.polyfit(np.log(x),y,1); f=lambda z: a*np.log(np.asarray(z))+b
            label="Logarithmic fit"; params=("log",(a,b))
        elif key == "exp":
            def fn(z,a,b,c): return a*np.exp(b*z)+c
            p0=(max(np.ptp(y),1e-12), 1.0/max(np.ptp(x),1e-12), float(np.min(y)))
            popt,_=curve_fit(fn,x,y,p0=p0,maxfev=50000); f=lambda z: fn(np.asarray(z),*popt)
            label="Exponential fit"; params=("exp",popt)
        elif key == "power":
            if np.any(x<=0): raise ValueError("Power-law calibration requires concentrations > 0.")
            def fn(z,a,b,c): return a*np.power(z,b)+c
            p0=(max(np.ptp(y),1e-12),1.0,float(np.min(y)))
            popt,_=curve_fit(fn,x,y,p0=p0,maxfev=50000); f=lambda z: fn(np.asarray(z),*popt)
            label="Power-law fit"; params=("power",popt)
        elif key == "4pl":
            if np.any(x<=0): raise ValueError("4PL calibration requires concentrations > 0.")
            def fn(z,a,d,c,b): return d+(a-d)/(1.0+np.power(np.asarray(z)/c,b))
            p0=(float(y[0]),float(y[-1]),float(np.median(x)),1.0)
            popt,_=curve_fit(fn,x,y,p0=p0,maxfev=100000); f=lambda z: fn(np.asarray(z),*popt)
            label="4-parameter logistic fit (4PL)"; params=("4pl",popt)
        else: raise ValueError(f"Unknown calibration model: {key}")
        yhat=np.asarray(f(x),float); ss_res=float(np.sum((y-yhat)**2)); ss_tot=float(np.sum((y-np.mean(y))**2))
        r2=1.0-ss_res/ss_tot if ss_tot>0 else np.nan
        xfit=np.linspace(xmin,xmax,500); yfit=np.asarray(f(xfit),float)
        return f,label,r2,xfit,yfit,params

    def _fit_equation_text(self, fit_params):
        """Human-readable equation for the currently fitted calibration model."""
        kind, params = fit_params
        if kind == "poly":
            coeff = np.asarray(params, dtype=float)
            deg = len(coeff) - 1
            if deg == 1:
                a, b = coeff
                return f"y = {a:.6E} x + {b:.6E}"
            terms = []
            for i, c in enumerate(coeff):
                power = deg - i
                if power == 0:
                    terms.append(f"{c:+.6E}")
                elif power == 1:
                    terms.append(f"{c:+.6E} x")
                else:
                    terms.append(f"{c:+.6E} x^{power}")
            return "y = " + " ".join(terms).lstrip("+")
        if kind == "log":
            a, b = params
            return f"y = {a:.6E} ln(x) + {b:.6E}"
        if kind == "exp":
            a, b, c = params
            return f"y = {a:.6E} exp({b:.6E} x) + {c:.6E}"
        if kind == "power":
            a, b, c = params
            return f"y = {a:.6E} x^{b:.6E} + {c:.6E}"
        if kind == "4pl":
            a, d, c, b = params
            return (
                f"y = {d:.6E} + ({a:.6E} - {d:.6E}) / "
                f"(1 + (x/{c:.6E})^{b:.6E})"
            )
        return ""

    @staticmethod
    def _estimate_x_from_fit(f, y_unknown, xmin, xmax):
        grid=np.linspace(float(xmin),float(xmax),4001); vals=np.asarray(f(grid),float)-float(y_unknown)
        roots=[]
        for i in range(len(grid)-1):
            if not (np.isfinite(vals[i]) and np.isfinite(vals[i+1])): continue
            if vals[i]==0: roots.append(float(grid[i]))
            elif vals[i]*vals[i+1] < 0:
                try: roots.append(float(brentq(lambda z: float(np.asarray(f(z)))-float(y_unknown),grid[i],grid[i+1])))
                except Exception: pass
        if not roots:
            j=int(np.nanargmin(np.abs(vals)))
            if abs(vals[j]) <= max(1e-8, 1e-3*max(abs(float(y_unknown)),1.0)): roots=[float(grid[j])]
        roots=sorted(set(round(r,15) for r in roots))
        return (roots[0] if roots else np.nan), roots

    def _analysis_table(self):
        """Single source of truth using explicit concentration -> ECL-block assignment."""
        if self.series_data is None:
            return pd.DataFrame(columns=[
                "Concentration_index",
                "ECL_block",
                "Coreactant_concentration_mol_L-1",
                "Integrated_mean_ECL_a.u._V",
            ])

        concentrations = self._concentrations()
        assignments = self._assigned_blocks()
        integrals = self._get_block_integrals()
        nblocks = len(self.series_data["ecl_blocks"])

        rows = []
        for ci, (c, block_index) in enumerate(zip(concentrations, assignments), start=1):
            if block_index < 0 or block_index >= nblocks:
                raise ValueError(f"Invalid ECL block assignment for c{ci}.")
            rows.append({
                "Concentration_index": ci,
                "ECL_block": block_index + 1,
                "Coreactant_concentration_mol_L-1": float(c),
                "Integrated_mean_ECL_a.u._V": float(integrals[block_index]),
            })
        return pd.DataFrame(rows)

    def _analysis_table_sorted(self):
        """Same rows as _analysis_table(), sorted only as complete rows for plotting."""
        df = self._analysis_table()
        if df.empty:
            return df
        return df.sort_values(
            "Coreactant_concentration_mol_L-1",
            kind="mergesort"
        ).reset_index(drop=True)

    def _series_integrals(self):
        """Compatibility wrapper: complete row records from the exact sorted analysis table."""
        df = self._analysis_table_sorted()
        return [
            {
                "ECL_block": int(row.ECL_block),
                "concentration": float(row.Coreactant_concentration_mol_L_1),
                "integral": float(row.Integrated_mean_ECL_a_u_V),
            }
            for row in df.rename(columns={
                "Coreactant_concentration_mol_L-1": "Coreactant_concentration_mol_L_1",
                "Integrated_mean_ECL_a.u._V": "Integrated_mean_ECL_a_u_V",
            }).itertuples(index=False)
        ]

    def _unknown_mean_integral(self):
        """Integral of the mean ECL trace in the optional unknown-sample CSV."""
        if self.unknown_data is None:
            return np.nan
        e = np.asarray(self.unknown_data["potential"], float)
        traces = np.asarray(self.unknown_data["ecl"], float)
        n = min(int(self.n_cycles.value()), traces.shape[1])
        if n < 1:
            return np.nan
        processed = np.column_stack([self._smooth(traces[:, k]) for k in range(n)])
        mean_trace = np.nanmean(processed, axis=1)
        return self._integrate_trace(e, mean_trace)

    @staticmethod
    def _estimate_x_from_polynomial(coeff, y_unknown, xmin, xmax, x_data, y_data):
        """Return a physically admissible polynomial intersection inside calibration range."""
        roots = np.roots(np.array(coeff, dtype=float) - np.r_[np.zeros(len(coeff)-1), y_unknown])
        valid = sorted(float(r.real) for r in roots
                       if abs(r.imag) <= max(1e-12, 1e-7 * max(1.0, abs(r.real)))
                       and xmin - 1e-12 <= r.real <= xmax + 1e-12)
        if not valid:
            return np.nan, []
        # If the polynomial is non-monotonic there may be several mathematical roots.
        # Choose the root closest to the local linear inverse estimate from the two
        # calibration points whose intensities bracket/approach the unknown value.
        order = np.argsort(np.abs(np.asarray(y_data, float) - y_unknown))[:2]
        if len(order) == 2 and abs(y_data[order[1]] - y_data[order[0]]) > 1e-30:
            x_hint = x_data[order[0]] + (y_unknown-y_data[order[0]]) * (x_data[order[1]]-x_data[order[0]]) / (y_data[order[1]]-y_data[order[0]])
        else:
            x_hint = float(np.mean(valid))
        return min(valid, key=lambda v: abs(v-x_hint)), valid

    def plot_concentration_series(self):
        try:
            self._refresh_display_combo()
            if not self._validate_concentrations():
                return
            if self.series_data is None:
                QMessageBox.information(self, "Concentration series", "Please load the concentration-series CSV first.")
                return

            plot_df = self._analysis_table_sorted()
            if len(plot_df) < 2:
                QMessageBox.information(self, "Concentration series", "At least two concentration blocks are required.")
                return

            x = plot_df["Coreactant_concentration_mol_L-1"].to_numpy(dtype=float)
            y = plot_df["Integrated_mean_ECL_a.u._V"].to_numpy(dtype=float)
            cache = self._get_block_integrals()
            physical = self._analysis_table().sort_values("ECL_block")
            for _, row in physical.iterrows():
                bi = int(row["ECL_block"]) - 1
                if not np.isclose(float(row["Integrated_mean_ECL_a.u._V"]), float(cache[bi]), rtol=1e-12, atol=1e-12):
                    raise RuntimeError(f"Internal block mapping mismatch at ECL block {bi+1}.")

            # Calibration fit selected explicitly by the user.
            fitfun, fit_name, r2, xfit, yfit, fit_params = self._fit_calibration(x, y)

            unknown_y = np.nan
            unknown_x = np.nan
            roots = []
            analyze_unknown = self.unknown_choice.currentText() == "Yes" and self.unknown_data is not None
            if analyze_unknown:
                unknown_y = self._unknown_mean_integral()
                if np.isfinite(unknown_y):
                    unknown_x, roots = self._estimate_x_from_fit(
                        fitfun, unknown_y, float(np.min(x)), float(np.max(x))
                    )

            dlg = QDialog(self)
            dlg.setWindowTitle("Integrated ECL intensity vs coreactant concentration")
            lay = QVBoxLayout(dlg)

            toolbar_row = QHBoxLayout()
            plot_column = QVBoxLayout()
            fig = Figure(figsize=(9.2, 7.0), tight_layout=True)
            canvas = FigureCanvas(fig)
            toolbar = NavigationToolbar(canvas, dlg)
            plot_column.addWidget(toolbar)
            plot_column.addWidget(canvas)

            content_row = QHBoxLayout()
            content_row.addLayout(plot_column, 5)

            result_panel = QGroupBox("Unknown sample")
            result_panel.setMinimumWidth(275)
            result_panel.setMaximumWidth(340)
            result_panel.setStyleSheet("""
                QGroupBox {
                    font-size: 15px;
                    font-weight: 700;
                    color: #A00000;
                    border: 2px solid #E04A4A;
                    border-radius: 8px;
                    margin-top: 10px;
                    padding-top: 12px;
                    background-color: #FFF7F7;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 5px;
                }
            """)
            result_layout = QVBoxLayout(result_panel)
            result_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
            content_row.addWidget(result_panel, 1)

            lay.addLayout(content_row)

            ax = fig.add_subplot(111)
            ax.plot(x, y, "o", markersize=6, label="Calibration data")
            ax.plot(xfit, yfit, linewidth=2.0, label=fit_name)
            for _, row in plot_df.iterrows():
                ax.annotate(f"B{int(row['ECL_block'])}",
                            (float(row["Coreactant_concentration_mol_L-1"]), float(row["Integrated_mean_ECL_a.u._V"])),
                            xytext=(5, 5), textcoords="offset points", fontsize=8)

            if analyze_unknown and np.isfinite(unknown_y) and np.isfinite(unknown_x):
                # Red guide arrows: measured integral -> fitted curve -> concentration axis.
                xmin_plot = min(0.0, float(np.min(x)))
                ax.annotate("", xy=(unknown_x, unknown_y), xytext=(xmin_plot, unknown_y),
                            arrowprops=dict(arrowstyle="->", color="red", lw=1.8))
                ax.annotate("", xy=(unknown_x, 0.0), xytext=(unknown_x, unknown_y),
                            arrowprops=dict(arrowstyle="->", color="red", lw=1.8))
                ax.plot([unknown_x], [unknown_y], "o", color="red", markersize=7, label="Unknown sample")
                ax.text(unknown_x, 0.03, f"{unknown_x:.3E} mol L⁻¹",
                        transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                        fontsize=13, fontweight="bold", color="red",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="red", alpha=0.9))

            ax.set_title("Integrated ECL intensity vs coreactant concentration")
            ax.set_xlabel("Coreactant concentration / mol L⁻¹")
            ax.set_ylabel("Integrated ECL intensity / a.u.·V")
            ax.grid(alpha=0.25)
            ax.legend(loc="best")
            ax.text(0.02, 0.02, f"{fit_name}; R² = {r2:.5f}", transform=ax.transAxes, fontsize=9)

            fit_label = QLabel(
                f"<b>Calibration fit</b><br>{fit_name}<br>R² = {r2:.5f}"
            )
            fit_label.setWordWrap(True)
            fit_label.setStyleSheet(
                "font-size: 12px; padding: 8px; background:#F2F7FF; "
                "border:1px solid #A9C4E4; border-radius:5px;"
            )
            result_layout.addWidget(fit_label)

            if analyze_unknown and np.isfinite(unknown_y) and np.isfinite(unknown_x):
                result_label = QLabel(
                    "<b>Integrated mean ECL</b><br>"
                    f"<span style='font-size:15px;color:#C00000'>{unknown_y:.6g} a.u.·V</span><br><br>"
                    "<b>Estimated concentration</b><br>"
                    f"<span style='font-size:20px;color:red;font-weight:700'>{unknown_x:.4E} mol L⁻¹</span>"
                )
                result_label.setWordWrap(True)
                result_label.setStyleSheet(
                    "padding:10px; background:#FFF0F0; border:1px solid #E56B6B; "
                    "border-radius:6px;"
                )
                result_layout.addWidget(result_label)
            elif analyze_unknown:
                result_label = QLabel(
                    "The unknown ECL integral has no usable intersection "
                    "with the selected calibration fit in the calibration range."
                )
                result_label.setWordWrap(True)
                result_label.setStyleSheet(
                    "padding:10px; color:#A00000; background:#FFF0F0; "
                    "border:1px solid #E56B6B; border-radius:6px;"
                )
                result_layout.addWidget(result_label)
            else:
                result_layout.addWidget(QLabel("No unknown sample selected."))

            result_layout.addStretch(1)

            table = QTableWidget(len(plot_df), 3)
            table.setHorizontalHeaderLabels(["ECL block", "Concentration / mol L⁻¹", "Integrated mean ECL / a.u.·V"])
            table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            for r, (_, row) in enumerate(plot_df.iterrows()):
                table.setItem(r, 0, QTableWidgetItem(str(int(row["ECL_block"]))))
                table.setItem(r, 1, QTableWidgetItem(f'{float(row["Coreactant_concentration_mol_L-1"]):.8g}'))
                table.setItem(r, 2, QTableWidgetItem(f'{float(row["Integrated_mean_ECL_a.u._V"]):.8g}'))
            table.setMaximumHeight(190)
            lay.addWidget(table)

            if analyze_unknown:
                if np.isfinite(unknown_y) and np.isfinite(unknown_x):
                    txt = f"Unknown: integrated mean ECL = {unknown_y:.6g} a.u.·V  →  concentration = {unknown_x:.6g} mol L⁻¹"
                    if len(roots) > 1:
                        txt += f"   ({len(roots)} mathematical intersections in calibration range; nearest local branch selected)"
                    result = QLabel(txt)
                    result.setStyleSheet("font-weight: 700; color: #C00000; padding: 6px;")
                    lay.addWidget(result)
                elif self.unknown_data is not None:
                    lay.addWidget(QLabel("Unknown ECL integral lies outside the usable polynomial calibration intersections."))

            canvas.draw()
            dlg.resize(1350, 900)
            dlg.exec()

        except Exception as exc:
            QMessageBox.critical(self, "Concentration series", f"Could not calculate or plot the concentration series:\n{exc}")

    def export_excel(self):
        try:
            self._refresh_display_combo()
            if not self._validate_concentrations():
                return
            if self.series_data is None:
                QMessageBox.information(
                    self, "Excel", "Please load the concentration-series CSV first."
                )
                return

            p, _ = QFileDialog.getSaveFileName(
                self, "Save Excel", "ECL_integrated_analysis.xlsx", "Excel (*.xlsx)"
            )
            if not p:
                return

            conc = self._concentrations()
            with pd.ExcelWriter(p, engine="xlsxwriter") as w:
                workbook = w.book
                colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9",
                          "#E69F00", "#F0E442", "#999999"]
                dash_types = ["solid", "dash", "dot", "dash_dot"]

                def write_dataset_sheet(sheet_name, e, cv, ecl, concentration=None):
                    d = {"Potential_V": e}
                    if concentration is not None:
                        d["Coreactant_concentration_mol_L-1"] = np.full(len(e), concentration)
                    for k in range(cv.shape[1]):
                        d[f"CV_cycle_{k+1}_uA"] = cv[:, k]
                    for k in range(ecl.shape[1]):
                        d[f"ECL_cycle_{k+1}_a.u."] = ecl[:, k]
                    d["Mean_CV_uA"] = np.nanmean(cv, axis=1)
                    d["Mean_ECL_a.u."] = np.nanmean(ecl, axis=1)
                    df = pd.DataFrame(d)
                    df.to_excel(w, sheet_name=sheet_name, index=False)

                    ws = w.sheets[sheet_name]
                    ws.freeze_panes(1, 0)
                    ws.set_column(0, df.shape[1]-1, 18)

                    chart = workbook.add_chart({"type": "scatter", "subtype": "smooth"})
                    headers = list(df.columns)
                    xcol = headers.index("Potential_V")

                    # CV cycles on primary axis.
                    cv_headers = [h for h in headers if h.startswith("CV_cycle_")]
                    for k, h in enumerate(cv_headers):
                        ycol = headers.index(h)
                        line = {"color": "#000000", "width": 1.5}
                        dash = dash_types[k % len(dash_types)]
                        if dash != "solid":
                            line["dash_type"] = dash
                        chart.add_series({
                            "name": h,
                            "categories": [sheet_name, 1, xcol, len(df), xcol],
                            "values": [sheet_name, 1, ycol, len(df), ycol],
                            "line": line,
                            "marker": {"type": "none"},
                        })

                    # ECL cycles on secondary axis.
                    ecl_headers = [h for h in headers if h.startswith("ECL_cycle_")]
                    for k, h in enumerate(ecl_headers):
                        ycol = headers.index(h)
                        line = {"color": colors[k % len(colors)], "width": 1.8}
                        dash = dash_types[k % len(dash_types)]
                        if dash != "solid":
                            line["dash_type"] = dash
                        chart.add_series({
                            "name": h,
                            "categories": [sheet_name, 1, xcol, len(df), xcol],
                            "values": [sheet_name, 1, ycol, len(df), ycol],
                            "y2_axis": True,
                            "line": line,
                            "marker": {"type": "none"},
                        })

                    # Means.
                    if "Mean_CV_uA" in headers:
                        ycol = headers.index("Mean_CV_uA")
                        chart.add_series({
                            "name": "Mean CV",
                            "categories": [sheet_name, 1, xcol, len(df), xcol],
                            "values": [sheet_name, 1, ycol, len(df), ycol],
                            "line": {"color": "#000000", "width": 3.0},
                            "marker": {"type": "none"},
                        })
                    if "Mean_ECL_a.u." in headers:
                        ycol = headers.index("Mean_ECL_a.u.")
                        chart.add_series({
                            "name": "Mean ECL",
                            "categories": [sheet_name, 1, xcol, len(df), xcol],
                            "values": [sheet_name, 1, ycol, len(df), ycol],
                            "y2_axis": True,
                            "line": {"color": "#8C564B", "width": 3.0},
                            "marker": {"type": "none"},
                        })

                    chart.set_x_axis({"name": "Potential / V"})
                    chart.set_y_axis({"name": "Current / µA"})
                    chart.set_y2_axis({"name": "ECL intensity / a.u."})
                    chart.set_legend({"position": "bottom"})
                    chart.set_size({"width": 1100, "height": 620})
                    ws.insert_chart("L2", chart)

                # Reference.
                if self.reference_data is not None:
                    e, cv, ecl = self._processed_dataset(self.reference_data, 0)
                    write_dataset_sheet("Without coreactant", e, cv, ecl)

                # One sheet + chart per concentration using explicit ECL-block assignment.
                assignments = self._assigned_blocks()
                for i, block_index in enumerate(assignments):
                    e, cv, ecl = self._processed_dataset(self.series_data, block_index)
                    cval = conc[i] if i < len(conc) else np.nan
                    write_dataset_sheet(f"Coreactant c{i+1}"[:31], e, cv, ecl, cval)

                # Integrals vs concentration.
                mapping_df = self._analysis_table()
                mapping_df["Mapping_source"] = "explicit concentration-to-ECL-block assignment"
                integral_df = self._analysis_table_sorted()[
                    [
                        "Coreactant_concentration_mol_L-1",
                        "Integrated_mean_ECL_a.u._V",
                        "ECL_block",
                    ]
                ].copy()
                rows = [
                    {
                        "ECL_block": int(row["ECL_block"]),
                        "concentration": float(row["Coreactant_concentration_mol_L-1"]),
                        "integral": float(row["Integrated_mean_ECL_a.u._V"]),
                    }
                    for _, row in integral_df.iterrows()
                ]
                # Physical left-to-right block order for verification.
                mapping_df.sort_values("ECL_block").to_excel(
                    w, sheet_name="Block mapping", index=False
                )

                integral_df.to_excel(w, sheet_name="Concentration integrals", index=False)
                ws_int = w.sheets["Concentration integrals"]
                ws_int.set_column(0, 2, 24)

                if len(integral_df) >= 1:
                    # Use exactly the same calibration table and polynomial logic as the GUI.
                    x_cal = integral_df["Coreactant_concentration_mol_L-1"].to_numpy(dtype=float)
                    y_cal = integral_df["Integrated_mean_ECL_a.u._V"].to_numpy(dtype=float)
                    fitfun, fit_name, r2, x_fit, y_fit, fit_params = self._fit_calibration(x_cal, y_cal)
                    # The Excel fit is generated by the same selector/model as the preview.
                    current_model = self._current_calibration_model() if hasattr(self, "_current_calibration_model") else None
                    if current_model is not None and str(current_model).lower().startswith("linear") and "Linear" not in fit_name:
                        raise RuntimeError("Excel/GUI calibration-fit mismatch: linear was selected but Excel did not receive a linear fit.")
                    fit_equation = self._fit_equation_text(fit_params)
                    # Visible calibration summary next to the exported data.
                    fit_summary_fmt = workbook.add_format({
                        "bold": True, "bg_color": "#DCEAF7", "border": 1
                    })
                    fit_value_fmt = workbook.add_format({
                        "bg_color": "#FFFFFF", "border": 1
                    })
                    r2_value_fmt = workbook.add_format({
                        "bold": True, "font_color": "#C00000",
                        "bg_color": "#FFF2F2", "border": 1,
                        "num_format": "0.00000"
                    })
                    ws_int.write("E2", "Calibration fit", fit_summary_fmt)
                    ws_int.write("F2", fit_name, fit_value_fmt)
                    ws_int.write("E3", "R²", fit_summary_fmt)
                    if np.isfinite(r2):
                        ws_int.write_number("F3", float(r2), r2_value_fmt)
                    else:
                        ws_int.write("F3", "", r2_value_fmt)
                    ws_int.write("E4", "Fit equation", fit_summary_fmt)
                    ws_int.write("F4", fit_equation, fit_value_fmt)
                    coeff = fit_params[1] if fit_params[0] == "poly" else None

                    unknown_y = np.nan
                    unknown_x = np.nan
                    analyze_unknown = self.unknown_choice.currentText() == "Yes" and self.unknown_data is not None
                    if analyze_unknown:
                        unknown_y = self._unknown_mean_integral()
                        if np.isfinite(unknown_y):
                            unknown_x, _roots = self._estimate_x_from_fit(
                                fitfun, unknown_y, float(np.min(x_cal)), float(np.max(x_cal))
                            )

                    # Helper data for the fitted curve and, if available, the unknown sample arrows.
                    helper_col = 4  # E
                    ws_int.write(0, helper_col, "Fit_x")
                    ws_int.write(0, helper_col + 1, "Fit_y")
                    if fitfun is not None:
                        x_fit = np.linspace(float(np.min(x_cal)), float(np.max(x_cal)), 201)
                        y_fit = fitfun(x_fit)
                        for rr, (xx, yy) in enumerate(zip(x_fit, y_fit), start=1):
                            ws_int.write_number(rr, helper_col, float(xx))
                            ws_int.write_number(rr, helper_col + 1, float(yy))

                    chart = workbook.add_chart({"type": "scatter", "subtype": "straight_with_markers"})
                    chart.add_series({
                        "name": "Calibration data",
                        "categories": ["Concentration integrals", 1, 0, len(integral_df), 0],
                        "values": ["Concentration integrals", 1, 1, len(integral_df), 1],
                        "line": {"none": True},
                        "marker": {"type": "circle", "size": 6},
                    })
                    if fitfun is not None:
                        chart.add_series({
                            "name": fit_name,
                            "categories": ["Concentration integrals", 1, helper_col, 201, helper_col],
                            "values": ["Concentration integrals", 1, helper_col + 1, 201, helper_col + 1],
                            "line": {"width": 2.25},
                            "marker": {"type": "none"},
                        })

                    # Calibration model summary: this is the SAME fit used by the GUI.
                    fmt_fit_head = workbook.add_format({
                        "bold": True, "font_size": 14, "font_color": "#17365D",
                        "bg_color": "#DCEAF7", "border": 1, "align": "center"
                    })
                    fmt_fit_label = workbook.add_format({
                        "bold": True, "bg_color": "#F3F7FB", "border": 1
                    })
                    fmt_fit_value = workbook.add_format({
                        "bg_color": "#FFFFFF", "border": 1
                    })
                    fmt_r2 = workbook.add_format({
                        "bold": True, "font_color": "#C00000", "font_size": 13,
                        "bg_color": "#FFF2F2", "border": 1
                    })

                    ws_int.merge_range("N8:P8", "Calibration fit", fmt_fit_head)
                    ws_int.merge_range("N9:O9", "Model", fmt_fit_label)
                    ws_int.write("P9", fit_name, fmt_fit_value)
                    ws_int.merge_range("N10:O10", "R²", fmt_fit_label)
                    if np.isfinite(r2):
                        ws_int.write_number("P10", float(r2), fmt_r2)
                    else:
                        ws_int.write("P10", "", fmt_r2)
                    ws_int.merge_range("N11:O12", "Equation", fmt_fit_label)
                    ws_int.write("P11", fit_equation, fmt_fit_value)
                    ws_int.set_column("N:O", 20)
                    ws_int.set_column("P:P", 28)

                    if analyze_unknown:
                        ws_int.write("E205", "Unknown integrated mean ECL / a.u.·V")
                        if np.isfinite(unknown_y):
                            ws_int.write_number("F205", float(unknown_y))
                        ws_int.write("E206", "Estimated concentration / mol L⁻¹")
                        if np.isfinite(unknown_x):
                            ws_int.write_number("F206", float(unknown_x))
                            red = "#FF0000"
                            xmin = float(np.min(x_cal))
                            ymin = 0.0
                            # Horizontal arrow-like guide (unknown intensity -> fit).
                            ws_int.write("H205", xmin); ws_int.write("I205", unknown_y)
                            ws_int.write("H206", unknown_x); ws_int.write("I206", unknown_y)
                            chart.add_series({
                                "name": "Unknown ECL → fit",
                                "categories": ["Concentration integrals", 204, 7, 205, 7],
                                "values": ["Concentration integrals", 204, 8, 205, 8],
                                "line": {"color": red, "dash_type": "dash", "width": 1.5},
                                "marker": {"type": "none"},
                            })
                            # Vertical guide to concentration axis.
                            ws_int.write("J205", unknown_x); ws_int.write("K205", unknown_y)
                            ws_int.write("J206", unknown_x); ws_int.write("K206", ymin)
                            chart.add_series({
                                "name": "Fit → concentration",
                                "categories": ["Concentration integrals", 204, 9, 205, 9],
                                "values": ["Concentration integrals", 204, 10, 205, 10],
                                "line": {"color": red, "dash_type": "dash", "width": 1.5},
                                "marker": {"type": "none"},
                            })
                            # Unknown point.
                            ws_int.write("L205", unknown_x); ws_int.write("M205", unknown_y)
                            chart.add_series({
                                "name": "Unknown sample",
                                "categories": ["Concentration integrals", 204, 11, 204, 11],
                                "values": ["Concentration integrals", 204, 12, 204, 12],
                                "line": {"none": True},
                                "marker": {"type": "circle", "size": 8, "border": {"color": red}, "fill": {"color": red}},
                            })
                            # Prominent unknown-sample result beside the calibration diagram.
                            # Keep it high on the sheet so it is visible together with the chart.
                            fmt_head = workbook.add_format({
                                "bold": True, "font_size": 15, "font_color": "#7A3E00",
                                "bg_color": "#FFF2CC", "border": 1, "align": "center", "valign": "vcenter"
                            })
                            fmt_label = workbook.add_format({
                                "bold": True, "font_size": 12, "bg_color": "#FFF4E3",
                                "border": 1, "valign": "vcenter"
                            })
                            fmt_red = workbook.add_format({
                                "bold": True, "font_color": red, "font_size": 16,
                                "bg_color": "#FFF4E3", "border": 1, "align": "center", "valign": "vcenter"
                            })
                            ws_int.merge_range("N2:P2", "Unknown concentration", fmt_head)
                            ws_int.merge_range("N3:O4", "Integrated mean ECL / a.u.·V", fmt_label)
                            ws_int.write_number("P3", float(unknown_y), fmt_red)
                            ws_int.merge_range("N5:O6", "Estimated concentration / mol L⁻¹", fmt_label)
                            ws_int.write("P5", f"{unknown_x:.3E}", fmt_red)
                            ws_int.set_column("N:O", 20)
                            ws_int.set_column("P:P", 18)
                            ws_int.set_row(1, 24)
                            ws_int.set_row(2, 25)
                            ws_int.set_row(4, 25)

                    chart.set_title({
                        "name": (
                            f"Integrated ECL intensity — {fit_name}, "
                            f"R² = {r2:.5f}" if np.isfinite(r2)
                            else f"Integrated ECL intensity — {fit_name}"
                        )
                    })
                    chart.set_x_axis({"name": "Coreactant concentration / mol L⁻¹"})
                    chart.set_y_axis({"name": "Integrated ECL intensity / a.u.·V", "min": 0})
                    chart.set_legend({"position": "bottom"})
                    chart.set_size({"width": 950, "height": 580})
                    ws_int.insert_chart("E2", chart)

                # Without/with comparison for all concentrations.
                i0 = self._mean_integral(self.reference_data, 0) if self.reference_data else np.nan
                comparisons = []
                for r in rows:
                    cval = r["concentration"]
                    iw = r["integral"]
                    block = r["ECL_block"]
                    ratio = i0/iw if np.isfinite(i0) and np.isfinite(iw) and abs(iw)>1e-30 else np.nan
                    comparisons.append({
                        "Coreactant_concentration_mol_L-1": cval,
                        "Integral_without_coreactant_a.u._V": i0,
                        "Integral_with_coreactant_a.u._V": iw,
                        "Ratio_without_over_with": ratio,
                        "ECL_block": block,
                    })
                comp_df = pd.DataFrame(comparisons)
                comp_df.to_excel(w, sheet_name="Integral comparison", index=False)
                ws_cmp = w.sheets["Integral comparison"]
                ws_cmp.set_column(0, comp_df.shape[1]-1, 26)

                if len(comp_df) >= 1:
                    chart = workbook.add_chart({"type": "scatter", "subtype": "straight_with_markers"})
                    chart.add_series({
                        "name": "Ratio without / with",
                        "categories": ["Integral comparison", 1, 0, len(comp_df), 0],
                        "values": ["Integral comparison", 1, 3, len(comp_df), 3],
                        "line": {"width": 2.0},
                        "marker": {"type": "circle", "size": 6},
                    })
                    chart.set_x_axis({"name": "Coreactant concentration / mol L⁻¹"})
                    chart.set_y_axis({"name": "Integral ratio without / with"})
                    chart.set_legend({"none": True})
                    chart.set_size({"width": 850, "height": 520})
                    ws_cmp.insert_chart("G2", chart)

            QMessageBox.information(self, "Excel", "Excel export completed.")
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Export Excel",
                f"Could not export Excel file:\n{exc}"
            )
    def export_png(self):
        if self.series_data is None:
            return
        self.plot()
        p, _ = QFileDialog.getSaveFileName(
            self, "Save PNG", "ECL_integrated_plot.png", "PNG (*.png)"
        )
        if p:
            old_size = self.fig.get_size_inches().copy()
            try:
                self.fig.set_size_inches(16, 9)
                self.fig.savefig(p, dpi=300, bbox_inches="tight")
            finally:
                self.fig.set_size_inches(old_size)
                self.canvas.draw_idle()


def main():
    app = QApplication.instance() or QApplication([])
    w = ECLIntegratedWindow()
    w.show()
    app.exec()


if __name__ == "__main__":
    main()

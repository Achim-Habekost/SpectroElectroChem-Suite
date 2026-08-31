from __future__ import annotations

import sys
import tempfile
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.interpolate import PchipInterpolator
from scipy import sparse
from scipy.sparse.linalg import spsolve
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from PySide6.QtCore import Qt, QLocale, QTimer
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QGroupBox, QLineEdit, QFileDialog,
    QCheckBox, QSpinBox, QDoubleSpinBox, QMessageBox, QDialog, QRadioButton,
    QButtonGroup, QDialogButtonBox, QListWidget, QComboBox, QScrollArea
)


class ReliableDoubleSpinBox(QDoubleSpinBox):
    """Standard QDoubleSpinBox using English decimal notation."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setLocale(QLocale.c())
        self.setGroupSeparatorShown(False)


class ReliableSpinBox(QSpinBox):
    """Standard QSpinBox with native, symmetric arrow stepping."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setLocale(QLocale.c())
        self.setGroupSeparatorShown(False)


class SpectralModeDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Spectro-CV Synchronization")
        self.setMinimumWidth(430)
        layout = QVBoxLayout(self)

        title = QLabel("Select spectral data type")
        title.setStyleSheet("font-size: 18px; font-weight: 700;")
        layout.addWidget(title)
        layout.addWidget(QLabel("Choose the type of spectra to synchronize with the cyclic voltammogram."))

        self.group = QButtonGroup(self)
        self.radios = []
        for i, text in enumerate(("Absorbance", "Fluorescence and ECL (wavelength-resolved)", "Raman")):
            rb = QRadioButton(text)
            if i == 0:
                rb.setChecked(True)
            self.group.addButton(rb)
            self.radios.append(rb)
            layout.addWidget(rb)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def selected_mode(self):
        for rb in self.radios:
            if rb.isChecked():
                label = rb.text()
                if label.startswith("Fluorescence"):
                    return "Fluorescence"
                return label
        return "Absorbance"


class MainWindow(QMainWindow):
    def __init__(self, spectral_mode: str):
        super().__init__()
        self.spectral_mode = spectral_mode
        self.setWindowTitle(f"Spectro-CV Synchronization – {spectral_mode}")
        self.resize(1180, 940)

        self.wavelengths = None
        self.spectra = None
        self.selected_wavelengths = []
        self.selected_ranges = []
        self.pending_range_start = None
        self.drag_range_index = None
        self.drag_range_edge = None
        self.selection_lines = []
        self.drag_selection_index = None
        self.selection_colors = ["#7C3AED", "#F97316", "#0EA5E9", "#16A34A", "#DC2626", "#A16207", "#DB2777", "#475569"]
        self.cv_potential = None
        self.cv_currents = None
        self.last_plot_html = None

        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        outer = QVBoxLayout(central)

        header = QLabel(f"Spectro-CV Synchronization — {self.spectral_mode}")
        header.setStyleSheet("font-size: 24px; font-weight: 700; color: #18324a;")
        outer.addWidget(header)

        subtitle = QLabel(
            "Load CV and spectral data, define acquisition parameters, select smoothing and choose wavelengths directly from the spectral preview."
        )
        subtitle.setWordWrap(True)
        outer.addWidget(subtitle)

        # Input files
        files_box = QGroupBox("Data input")
        files_grid = QGridLayout(files_box)
        self.cv_path = QLineEdit(); self.cv_path.setReadOnly(True)
        self.spec_path = QLineEdit(); self.spec_path.setReadOnly(True)
        b1 = QPushButton("Load CV CSV")
        b2 = QPushButton("Load spectra CSV")
        b1.clicked.connect(self._load_cv_file)
        b2.clicked.connect(self._load_spectra_file)
        files_grid.addWidget(b1,0,0); files_grid.addWidget(self.cv_path,0,1)
        files_grid.addWidget(b2,1,0); files_grid.addWidget(self.spec_path,1,1)
        self.spec_format_label = QLabel("")
        self.spec_format_label.setStyleSheet("color: #52606d; font-style: italic;")
        files_grid.addWidget(self.spec_format_label,2,1)
        outer.addWidget(files_box)

        params_row = QHBoxLayout()

        # CV parameters
        cv_box = QGroupBox("Cyclic voltammetry")
        cv_grid = QGridLayout(cv_box)
        self.e_start = self._dspin(0.8, -10, 10, 3, 0.05)
        self.e_switch = self._dspin(-0.8, -10, 10, 3, 0.05)
        self.scan_rate = self._dspin(0.05, 0.0001, 100, 4, 0.005)
        self.increment = self._dspin(0.005, 0.000001, 10, 6, 0.001)
        self.cycles = ReliableSpinBox(); self.cycles.setRange(1,100); self.cycles.setSingleStep(1); self.cycles.setValue(3)
        labels = [
            ("Start potential / V", self.e_start),
            ("Switching potential / V", self.e_switch),
            ("Scan rate / V s⁻¹", self.scan_rate),
            ("Potential increment / V", self.increment),
            ("Number of cycles", self.cycles),
        ]
        for r,(lab,w) in enumerate(labels):
            cv_grid.addWidget(QLabel(lab),r,0); cv_grid.addWidget(w,r,1)
        params_row.addWidget(cv_box)

        # Spectral parameters
        sp_box = QGroupBox(f"{self.spectral_mode} spectra")
        sp_grid = QGridLayout(sp_box)
        self.integration = self._dspin(40.0, 0.001, 1e7, 3, 1.0)
        self.averages = ReliableSpinBox(); self.averages.setRange(1,100000); self.averages.setSingleStep(1); self.averages.setValue(30)
        self.wl_min = self._dspin(500.0, 0, 100000, 2, 1.0)
        self.wl_max = self._dspin(800.0, 0, 100000, 2, 1.0)
        range_from_label = "Raman shift from / cm⁻¹" if self.spectral_mode == "Raman" else "Spectral range from / nm"
        range_to_label = "Raman shift to / cm⁻¹" if self.spectral_mode == "Raman" else "Spectral range to / nm"
        splabels = [
            ("Integration time / ms", self.integration),
            ("Averages", self.averages),
            (range_from_label, self.wl_min),
            (range_to_label, self.wl_max),
        ]
        for r,(lab,w) in enumerate(splabels):
            sp_grid.addWidget(QLabel(lab),r,0); sp_grid.addWidget(w,r,1)
        params_row.addWidget(sp_box)
        outer.addLayout(params_row)

        # Smoothing
        smooth_box = QGroupBox("Smoothing")
        smooth_grid = QGridLayout(smooth_box)
        self.cv_smooth = QCheckBox("Smooth CV")
        self.cv_window = ReliableSpinBox(); self.cv_window.setRange(3,501); self.cv_window.setSingleStep(2); self.cv_window.setValue(11)
        self.spec_smooth = QCheckBox("Smooth spectra")
        self.spec_smooth.setChecked(True)
        self.spec_window = ReliableSpinBox(); self.spec_window.setRange(3,501); self.spec_window.setSingleStep(2); self.spec_window.setValue(41)
        smooth_grid.addWidget(self.cv_smooth,0,0); smooth_grid.addWidget(QLabel("Window points"),0,1); smooth_grid.addWidget(self.cv_window,0,2)
        smooth_grid.addWidget(self.spec_smooth,1,0); smooth_grid.addWidget(QLabel("Window points"),1,1); smooth_grid.addWidget(self.spec_window,1,2)

        self.derivative_presentation = QComboBox()
        self.derivative_presentation.addItems(["None", "Savitzky–Golay", "PCHIP"])
        self.derivative_presentation.setCurrentText("PCHIP")
        self.derivative_window = ReliableSpinBox()
        self.derivative_window.setRange(3, 101)
        self.derivative_window.setSingleStep(2)
        self.derivative_window.setValue(7)
        smooth_grid.addWidget(QLabel("Derivative presentation"),2,0)
        smooth_grid.addWidget(self.derivative_presentation,2,1)
        smooth_grid.addWidget(QLabel("SG window points"),2,2)
        smooth_grid.addWidget(self.derivative_window,2,3)
        self.derivative_presentation.currentTextChanged.connect(self._update_derivative_controls)
        self.derivative_window.valueChanged.connect(self._update_derivative_controls)
        self._update_derivative_controls()
        outer.addWidget(smooth_box)

        # Raman baseline correction (shown only in Raman mode).
        self.baseline_box = QGroupBox("Raman baseline correction")
        baseline_grid = QGridLayout(self.baseline_box)

        self.baseline_method = QComboBox()
        self.baseline_method.addItems(["None", "AsLS", "arPLS"])
        self.baseline_method.setCurrentText("AsLS")

        self.asls_lambda = ReliableDoubleSpinBox()
        self.asls_lambda.setRange(1.0, 1.0e12)
        self.asls_lambda.setDecimals(0)
        self.asls_lambda.setSingleStep(10000.0)
        self.asls_lambda.setValue(600.0)
        self.asls_lambda.setKeyboardTracking(False)

        self.asls_p = ReliableDoubleSpinBox()
        self.asls_p.setRange(0.0001, 0.5)
        self.asls_p.setDecimals(4)
        self.asls_p.setSingleStep(0.001)
        self.asls_p.setValue(0.1)
        self.asls_p.setKeyboardTracking(False)

        baseline_grid.addWidget(QLabel("Baseline correction"), 0, 0)
        baseline_grid.addWidget(self.baseline_method, 0, 1)
        baseline_grid.addWidget(QLabel("λ (smoothness)"), 0, 2)
        baseline_grid.addWidget(self.asls_lambda, 0, 3)
        baseline_grid.addWidget(QLabel("AsLS p (asymmetry)"), 0, 4)
        baseline_grid.addWidget(self.asls_p, 0, 5)

        self.baseline_method.currentTextChanged.connect(self._update_baseline_controls)
        self.baseline_method.currentTextChanged.connect(self._plot_preview)
        self.asls_lambda.valueChanged.connect(self._plot_preview)
        self.asls_p.valueChanged.connect(self._plot_preview)
        self._update_baseline_controls()

        self.baseline_box.setVisible(self.spectral_mode == "Raman")
        outer.addWidget(self.baseline_box)

        # Raman preview option. Hidden by default because the thick mean curve
        # can obscure the individual potential-dependent Raman spectra.
        self.show_mean_spectrum = QCheckBox("Show mean spectrum")
        self.show_mean_spectrum.setChecked(False if self.spectral_mode == "Raman" else True)
        self.show_mean_spectrum.setVisible(self.spectral_mode == "Raman")
        self.show_mean_spectrum.stateChanged.connect(self._plot_preview)
        outer.addWidget(self.show_mean_spectrum)
        self.show_cycle_mean = QCheckBox("Show cycle mean")
        self.show_cycle_mean.setChecked(False)
        self.show_cycle_mean.setToolTip("Show mean CV and mean selected spectral signals across all cycles.")
        outer.addWidget(self.show_cycle_mean)

        # Real preview + wavelength selection
        preview_box = QGroupBox("Spectrum preview / wavelength selection")
        preview_layout = QHBoxLayout(preview_box)

        self.figure = Figure(figsize=(7.8, 4.2), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlabel(self._spectral_x_label())
        self.ax.set_ylabel(self._spectral_y_label())
        self.ax.text(
            0.5, 0.5,
            "Load a spectra CSV to display the preview.\nClick directly on a spectral position to select it.",
            ha="center", va="center", transform=self.ax.transAxes, color="#556"
        )
        self.canvas.setMinimumHeight(410)
        self.canvas.mpl_connect("button_press_event", self._on_preview_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_preview_motion)
        self.canvas.mpl_connect("button_release_event", self._on_preview_release)

        preview_plot_layout = QVBoxLayout()
        self.preview_toolbar = NavigationToolbar(self.canvas, self)
        self.preview_toolbar.setToolTip(
            "Use Home to reset, Pan to move and Zoom to select a spectral region."
        )
        preview_plot_layout.addWidget(self.preview_toolbar)
        preview_plot_layout.addWidget(self.canvas)
        preview_layout.addLayout(preview_plot_layout, 3)

        sel = QVBoxLayout()
        if self.spectral_mode == "Raman":
            self.range_mode = QCheckBox("Select Raman peak ranges")
            self.range_mode.setToolTip("Two successive clicks define a Raman interval; its integrated intensity is evaluated versus potential.")
            self.range_mode.stateChanged.connect(self._cancel_pending_range)
            sel.addWidget(self.range_mode)

            # Ranges are shown in the upper box when range selection is used.
            sel.addWidget(QLabel("Selected Raman ranges"))
            self.selected_ranges_list = QListWidget()
            self.selected_ranges_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
            sel.addWidget(self.selected_ranges_list)
            remove_range_btn = QPushButton("Remove selected Raman range")
            remove_range_btn.clicked.connect(self._remove_selected_range)
            sel.addWidget(remove_range_btn)
            clear_ranges_btn = QPushButton("Clear Raman ranges")
            clear_ranges_btn.clicked.connect(self._clear_selected_ranges)
            sel.addWidget(clear_ranges_btn)

            # Individual Raman shifts are shown in the lower box.
            sel.addWidget(QLabel("Selected Raman shifts"))
        else:
            self.range_mode = None
            self.selected_ranges_list = None
            sel.addWidget(QLabel("Selected wavelengths"))

        self.selected = QListWidget()
        self.selected.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        sel.addWidget(self.selected)

        remove_btn = QPushButton("Remove selected Raman shift" if self.spectral_mode == "Raman" else "Remove selected wavelength")
        remove_btn.clicked.connect(self._remove_selected_wavelength)
        sel.addWidget(remove_btn)

        clear_btn = QPushButton("Clear selected Raman shifts" if self.spectral_mode == "Raman" else "Clear selected wavelengths")
        clear_btn.clicked.connect(self._clear_selected_wavelengths)
        sel.addWidget(clear_btn)
        preview_layout.addLayout(sel,1)
        outer.addWidget(preview_box)

        # Replot when spectral smoothing settings change
        self.spec_smooth.stateChanged.connect(self._plot_preview)
        self.spec_window.valueChanged.connect(self._plot_preview)
        self.wl_min.valueChanged.connect(self._plot_preview)
        self.wl_max.valueChanged.connect(self._plot_preview)

        # Visualization and export
        out_box = QGroupBox("Visualization and export")
        out = QHBoxLayout(out_box)
        self.plot_btn = QPushButton("Synchronize + Plot")
        self.plot_btn.clicked.connect(self._synchronize_and_plot)
        out.addWidget(self.plot_btn)
        self.export_excel_btn = QPushButton("Export Excel")
        self.export_png_btn = QPushButton("Export PNG (standard view)")
        self.export_html_btn = QPushButton("Export Interactive HTML")
        for b in (self.export_excel_btn, self.export_png_btn, self.export_html_btn):
            b.setEnabled(False)
            b.setToolTip("Run Synchronize + Plot first.")
            out.addWidget(b)
        self.export_excel_btn.clicked.connect(self._export_excel)
        self.export_png_btn.clicked.connect(self._export_png)
        self.export_html_btn.clicked.connect(self._export_html)
        outer.addWidget(out_box)

        note = QLabel(
            "CV/spectra synchronization, selected-position traces, Raman AsLS baseline correction, derivative presentation (None / Savitzky–Golay / PCHIP), interactive waterfall and Excel/PNG/HTML export are active."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #555; font-style: italic;")
        outer.addWidget(note)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setWidget(central)
        self.setCentralWidget(scroll)

        self.setStyleSheet("""
            QMainWindow { background: #f4f6f8; }
            QGroupBox { font-weight: 700; border: 1px solid #d8dee5; border-radius: 8px; margin-top: 10px; padding-top: 10px; background: white; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
            QPushButton { padding: 7px 11px; border-radius: 5px; background: #1f6feb; color: white; font-weight: 600; }
            QPushButton:disabled { background: #aeb7c2; color: #eef1f4; }
            QLineEdit, QSpinBox, QDoubleSpinBox { padding: 4px; }
        """)

    def _spectral_x_label(self):
        return "Raman shift / cm⁻¹" if self.spectral_mode == "Raman" else "Wavelength / nm"

    def _spectral_y_label(self):
        if self.spectral_mode == "Absorbance":
            return "Absorbance"
        if self.spectral_mode == "Fluorescence":
            return "Fluorescence / ECL intensity / a.u."
        return "Raman intensity / a.u."

    @staticmethod
    def _dspin(value, lo, hi, decimals, step=None):
        w = ReliableDoubleSpinBox()
        w.setRange(lo, hi)
        w.setDecimals(decimals)
        if step is not None:
            w.setSingleStep(step)
        w.setValue(value)
        w.setKeyboardTracking(False)
        return w

    def _load_cv_file(self):
        p, _ = QFileDialog.getOpenFileName(self, "Load CV CSV", "", "CSV files (*.csv);;All files (*.*)")
        if not p:
            return
        try:
            potential, currents = self._read_cv_csv(Path(p))
        except Exception as exc:
            QMessageBox.critical(self, "CV import", f"Could not read CV data:\n{exc}")
            return
        self.cv_path.setText(p)
        self.cv_potential = potential
        self.cv_currents = currents

    @staticmethod
    def _read_cv_csv(path: Path):
        raw = pd.read_csv(path, header=None, sep=None, engine="python")
        num = raw.apply(pd.to_numeric, errors="coerce")
        a = num.to_numpy(dtype=float)
        # Remove fully nonnumeric rows/columns (e.g. textual headers).
        a = a[~np.all(~np.isfinite(a), axis=1)]
        a = a[:, ~np.all(~np.isfinite(a), axis=0)]
        if a.shape[0] < 3 or a.shape[1] < 2:
            raise ValueError("Expected first column = potential and one or more current columns.")
        potential = a[:, 0]
        currents = a[:, 1:]
        good = np.isfinite(potential)
        potential = potential[good]
        currents = currents[good]
        # Interpolate occasional missing current values column-wise.
        for j in range(currents.shape[1]):
            y = currents[:, j]
            ok = np.isfinite(y)
            if ok.sum() >= 2 and not ok.all():
                currents[:, j] = np.interp(np.arange(len(y)), np.flatnonzero(ok), y[ok])
        if currents.shape[1] < 1:
            raise ValueError("No current column detected.")
        return potential.astype(float), currents.astype(float)

    def _load_spectra_file(self):
        p, _ = QFileDialog.getOpenFileName(self, "Load spectra CSV", "", "CSV files (*.csv);;All files (*.*)")
        if not p:
            return
        try:
            wavelengths, spectra, orientation = self._read_spectra_csv(Path(p))
        except Exception as exc:
            QMessageBox.critical(self, "Spectra import", f"Could not read spectral data:\n{exc}")
            return

        self.spec_path.setText(p)
        self.wavelengths = wavelengths
        self.spectra = spectra
        if orientation == "column":
            self.spec_format_label.setText(("Original Raman format detected: first column = Raman shift; following columns = individual spectra — transposed internally." if self.spectral_mode == "Raman" else "Original spectral format detected (spectral axis in first column) — transposed internally."))
        else:
            self.spec_format_label.setText("Transposed spectral format detected (spectral axis in first row).")
        self.wl_min.setValue(float(np.nanmin(wavelengths)))
        self.wl_max.setValue(float(np.nanmax(wavelengths)))
        self._clear_selected_wavelengths()
        self._plot_preview()

    def _read_spectra_csv(self, path: Path):
        # sep=None lets pandas detect comma/semicolon/tab exports.
        raw = pd.read_csv(path, header=None, sep=None, engine="python")
        num = raw.apply(pd.to_numeric, errors="coerce")
        a = num.to_numpy(dtype=float)

        if a.shape[0] < 2 or a.shape[1] < 2:
            raise ValueError("The file does not contain a two-dimensional spectral data matrix.")

        first_row = a[0, :]
        first_col = a[:, 0]

        def score_axis(v):
            v = v[np.isfinite(v)]
            if len(v) < 5:
                return -1
            d = np.diff(v)
            monotonic = max(np.mean(d > 0), np.mean(d < 0))
            unique = len(np.unique(v)) / len(v)
            return monotonic + 0.25 * unique

        row_score = score_axis(first_row)
        col_score = score_axis(first_col)

        # Raman spectrometers commonly export one spectral coordinate per row:
        # first column = Raman shift, following columns = successive spectra.
        # For a tall matrix with a monotonic first column, prefer this original
        # instrument format explicitly instead of relying only on heuristic scores.
        finite_col = first_col[np.isfinite(first_col)]
        col_monotonic = False
        if finite_col.size >= 5:
            dc = np.diff(finite_col)
            col_monotonic = bool(np.all(dc > 0) or np.all(dc < 0))

        raman_original_column_format = (
            self.spectral_mode == "Raman"
            and a.shape[0] > a.shape[1]
            and col_monotonic
        )

        if raman_original_column_format:
            x = first_col
            spectra = a[:, 1:].T
            orientation = "column"
        elif row_score >= col_score:
            x = first_row
            spectra = a[1:, :]
            orientation = "row"
        else:
            x = first_col
            spectra = a[:, 1:].T
            orientation = "column"

        valid_x = np.isfinite(x)
        x = x[valid_x]
        spectra = spectra[:, valid_x]

        # Remove completely empty spectra.
        spectra = spectra[~np.all(~np.isfinite(spectra), axis=1)]
        if len(x) < 5 or spectra.shape[0] < 1:
            raise ValueError("No usable wavelength/Raman-shift axis and spectra were detected.")

        # Fill occasional NaNs along a spectrum by linear interpolation.
        for i in range(spectra.shape[0]):
            y = spectra[i]
            good = np.isfinite(y)
            if good.sum() >= 2 and not good.all():
                spectra[i] = np.interp(np.arange(len(y)), np.flatnonzero(good), y[good])

        return x.astype(float), spectra.astype(float), orientation

    def _update_baseline_controls(self):
        if not hasattr(self, "baseline_method"):
            return
        method = self.baseline_method.currentText()
        raman = self.spectral_mode == "Raman"
        self.asls_lambda.setEnabled(raman and method in ("AsLS", "arPLS"))
        self.asls_p.setEnabled(raman and method == "AsLS")
        self.asls_p.setToolTip("Used only for AsLS. arPLS determines asymmetric weights automatically.")

    @staticmethod
    def _asls_baseline(y, lam=1e5, p=0.001, niter=10):
        """Asymmetric least-squares baseline (Eilers-style AsLS)."""
        y = np.asarray(y, dtype=float)
        n = len(y)
        if n < 3:
            return np.zeros_like(y)

        # Second-difference penalty matrix.
        D = sparse.diags(
            [np.ones(n - 2), -2.0 * np.ones(n - 2), np.ones(n - 2)],
            [0, 1, 2],
            shape=(n - 2, n),
            format="csc",
        )
        H = float(lam) * (D.T @ D)
        w = np.ones(n, dtype=float)

        for _ in range(int(niter)):
            W = sparse.spdiags(w, 0, n, n, format="csc")
            z = spsolve(W + H, w * y)
            w = np.where(y > z, float(p), 1.0 - float(p))

        return np.asarray(z, dtype=float)

    @staticmethod
    def _arpls_baseline(y, lam=1e5, ratio=1e-6, niter=50):
        """Adaptive iteratively reweighted penalized least-squares baseline."""
        y = np.asarray(y, dtype=float)
        n = len(y)
        if n < 3:
            return np.zeros_like(y)

        D = sparse.diags(
            [np.ones(n - 2), -2.0 * np.ones(n - 2), np.ones(n - 2)],
            [0, 1, 2],
            shape=(n - 2, n),
            format="csc",
        )
        H = float(lam) * (D.T @ D)
        w = np.ones(n, dtype=float)

        for _ in range(int(niter)):
            W = sparse.spdiags(w, 0, n, n, format="csc")
            z = spsolve(W + H, w * y)
            d = y - z
            dn = d[d < 0]
            if dn.size < 2:
                break
            m = float(np.mean(dn))
            sd = float(np.std(dn))
            if not np.isfinite(sd) or sd < 1e-15:
                break
            arg = 2.0 * (d - (2.0 * sd - m)) / sd
            arg = np.clip(arg, -60.0, 60.0)
            w_new = 1.0 / (1.0 + np.exp(arg))
            denom = max(np.linalg.norm(w), 1e-15)
            if np.linalg.norm(w_new - w) / denom < float(ratio):
                w = w_new
                break
            w = w_new

        W = sparse.spdiags(w, 0, n, n, format="csc")
        z = spsolve(W + H, w * y)
        return np.asarray(z, dtype=float)

    def _baseline_corrected_spectra(self):
        if self.spectra is None:
            return None
        data = np.asarray(self.spectra, dtype=float)

        if self.spectral_mode != "Raman":
            return data

        if not hasattr(self, "baseline_method") or self.baseline_method.currentText() == "None":
            return data

        lam = float(self.asls_lambda.value())
        p = float(self.asls_p.value())
        method = self.baseline_method.currentText()
        corrected = np.empty_like(data, dtype=float)

        # Each Raman spectrum receives its own independent baseline.
        for i in range(data.shape[0]):
            if method == "arPLS":
                baseline = self._arpls_baseline(data[i], lam=lam, ratio=1e-6, niter=50)
            else:
                baseline = self._asls_baseline(data[i], lam=lam, p=p, niter=10)
            corrected[i] = data[i] - baseline

        return corrected

    def _smoothed_spectra_for_preview(self):
        if self.spectra is None:
            return None

        # Processing order:
        # 1) Raman baseline correction (if selected)
        # 2) spectral smoothing (if selected)
        data = self._baseline_corrected_spectra()

        if not self.spec_smooth.isChecked():
            return data
        npts = data.shape[1]
        w = int(self.spec_window.value())
        if w % 2 == 0:
            w += 1
        max_odd = npts if npts % 2 == 1 else npts - 1
        w = min(w, max_odd)
        if w < 5:
            return data
        poly = min(3, w - 2)
        return savgol_filter(data, window_length=w, polyorder=poly, axis=1)

    def _cancel_pending_range(self, *args):
        self.pending_range_start = None
        self.drag_range_index = None
        self.drag_range_edge = None

    def _update_selected_ranges_list(self):
        if self.selected_ranges_list is None: return
        self.selected_ranges_list.clear()
        for lo, hi in self.selected_ranges:
            self.selected_ranges_list.addItem(f"{lo:.1f} – {hi:.1f} cm⁻¹")

    def _remove_selected_range(self):
        if self.selected_ranges_list is None: return
        row = self.selected_ranges_list.currentRow()
        if 0 <= row < len(self.selected_ranges):
            self.selected_ranges.pop(row)
            self._update_selected_ranges_list()
            self._plot_preview(preserve_view=True)

    def _clear_selected_ranges(self):
        self.selected_ranges=[]; self.pending_range_start=None
        self.drag_range_index=None; self.drag_range_edge=None
        self._update_selected_ranges_list()
        if self.wavelengths is not None: self._plot_preview(preserve_view=True)

    def _integrated_range_signal(self, spectra, lo, hi):
        lo,hi=sorted((float(lo),float(hi)))
        m=np.isfinite(self.wavelengths)&(self.wavelengths>=lo)&(self.wavelengths<=hi)
        if np.sum(m)<2: return np.full(spectra.shape[0],np.nan)
        x=np.asarray(self.wavelengths[m],float); y=np.asarray(spectra[:,m],float)
        order=np.argsort(x)
        trapezoid_func = getattr(np, "trapezoid", None)
        if trapezoid_func is not None:
            return trapezoid_func(y[:, order], x[order], axis=1)
        # Compatibility with older NumPy releases.
        return np.trapz(y[:, order], x[order], axis=1)

    def _plot_preview(self, preserve_view=False):
        if self.wavelengths is None or self.spectra is None:
            return
        data = self._smoothed_spectra_for_preview()

        old_xlim = self.ax.get_xlim() if preserve_view and self.ax.has_data() else None
        old_ylim = self.ax.get_ylim() if preserve_view and self.ax.has_data() else None
        self.ax.clear()
        # Raman: show every individual spectrum in the preview so the user can
        # directly judge the complete time/potential series. Other modes retain
        # the compact representative preview.
        if self.spectral_mode == "Raman":
            indices = np.arange(data.shape[0], dtype=int)
            line_alpha = 0.30
            line_width = 0.55
        else:
            n_show = min(12, data.shape[0])
            indices = np.unique(np.linspace(0, data.shape[0] - 1, n_show).astype(int))
            line_alpha = 0.55
            line_width = 0.8

        for idx in indices:
            self.ax.plot(self.wavelengths, data[idx], linewidth=line_width, alpha=line_alpha)

        show_mean = (
            self.spectral_mode != "Raman"
            or not hasattr(self, "show_mean_spectrum")
            or self.show_mean_spectrum.isChecked()
        )
        if show_mean:
            self.ax.plot(
                self.wavelengths,
                np.nanmean(data, axis=0),
                color="black",
                linewidth=1.8,
                label="Mean spectrum",
            )

        self.ax.set_xlabel(self._spectral_x_label())
        self.ax.set_ylabel(self._spectral_y_label())
        self.ax.grid(True, alpha=0.2)
        handles, labels = self.ax.get_legend_handles_labels()
        if handles:
            self.ax.legend(loc="best", fontsize=8)

        # Display exactly the spectral range requested in the acquisition
        # parameters. For Raman this is the entered Raman-shift interval.
        x1 = float(self.wl_min.value())
        x2 = float(self.wl_max.value())
        lo, hi = min(x1, x2), max(x1, x2)
        data_lo = float(np.nanmin(self.wavelengths))
        data_hi = float(np.nanmax(self.wavelengths))
        lo = max(lo, data_lo)
        hi = min(hi, data_hi)
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            self.ax.set_xlim(lo, hi)

            # Autoscale Y to spectra within the visible Raman-shift/spectral range.
            range_mask = np.isfinite(self.wavelengths) & (self.wavelengths >= lo) & (self.wavelengths <= hi)
            if np.any(range_mask):
                visible = data[:, range_mask]
                finite = visible[np.isfinite(visible)]
                if finite.size:
                    ymin = float(np.nanmin(finite))
                    ymax = float(np.nanmax(finite))
                    span = ymax - ymin
                    pad = 0.06 * span if span > 0 else max(abs(ymax), 1.0) * 0.06
                    self.ax.set_ylim(ymin - pad, ymax + pad)

        for i,(rlo,rhi) in enumerate(self.selected_ranges):
            color=self.selection_colors[i % len(self.selection_colors)]
            self.ax.axvspan(rlo,rhi,color=color,alpha=0.10)
            self.ax.axvline(rlo,linewidth=1.4,linestyle="--",color=color,alpha=0.8)
            self.ax.axvline(rhi,linewidth=1.4,linestyle="--",color=color,alpha=0.8)
        if self.pending_range_start is not None:
            self.ax.axvline(self.pending_range_start,linewidth=2,linestyle=":",color="#444444")
        self.selection_lines = []
        for i, x in enumerate(self.selected_wavelengths):
            color = self.selection_colors[i % len(self.selection_colors)]
            self.selection_lines.append(
                self.ax.axvline(x, linewidth=2.0, linestyle="--", color=color, alpha=0.95)
            )

        self.figure.tight_layout()
        if old_xlim is not None and old_ylim is not None:
            self.ax.set_xlim(old_xlim)
            self.ax.set_ylim(old_ylim)

        self.canvas.draw_idle()

    def _nearest_spectral_position(self, xdata):
        idx = int(np.nanargmin(np.abs(self.wavelengths - xdata)))
        return float(self.wavelengths[idx])

    def _update_selected_list(self):
        unit = "cm⁻¹" if self.spectral_mode == "Raman" else "nm"
        self.selected.clear()
        for x in self.selected_wavelengths:
            self.selected.addItem(f"{x:.1f} {unit}")

    def _on_preview_press(self, event):

        # Do not interpret Matplotlib Zoom/Pan gestures as Raman-shift selections.
        toolbar = getattr(self.canvas, "toolbar", None)
        if toolbar is not None and getattr(toolbar, "mode", ""):
            return
        if self.wavelengths is None or event.inaxes is not self.ax or event.xdata is None:
            return
        if event.button == 3:  # right click: undo previous zoom/pan step
            toolbar = getattr(self.canvas, "toolbar", None)
            if toolbar is not None:
                try:
                    toolbar.back()
                except Exception:
                    try:
                        toolbar.home()
                    except Exception:
                        pass
            return

        if event.button != 1:
            return

        if self.spectral_mode=="Raman" and self.range_mode is not None and self.range_mode.isChecked():
            # Existing range boundaries can be moved. Hit testing is in screen
            # pixels, so it remains precise when the spectrum is zoomed.
            click_x_px = float(event.x)
            nearest = None
            nearest_dist = float("inf")
            for ridx, (rlo, rhi) in enumerate(self.selected_ranges):
                for edge_name, edge_x in (("lo", rlo), ("hi", rhi)):
                    edge_px = float(
                        self.ax.transData.transform(
                            (edge_x, event.ydata if event.ydata is not None else 0.0)
                        )[0]
                    )
                    dist = abs(edge_px - click_x_px)
                    if dist < nearest_dist:
                        nearest_dist = dist
                        nearest = (ridx, edge_name)

            if nearest is not None and nearest_dist <= 9.0:
                self.drag_range_index, self.drag_range_edge = nearest
                self.pending_range_start = None
                return

            # Away from an existing boundary, two successive clicks define
            # a new Raman peak range.
            x=self._nearest_spectral_position(event.xdata)
            if self.pending_range_start is None:
                self.pending_range_start=x
            else:
                lo,hi=sorted((self.pending_range_start,x))
                if hi>lo:
                    self.selected_ranges.append((lo,hi))
                    self._update_selected_ranges_list()
                self.pending_range_start=None
            self._plot_preview(preserve_view=True)
            return

        # If the click is visually close to an existing marker, start dragging it.
        # Use SCREEN distance rather than a fraction of the full Raman range.
        # The old full-range threshold became far too large after zooming and
        # prevented insertion of a new marker between two nearby Raman peaks.
        if self.selected_wavelengths:
            click_x_px = float(event.x)
            marker_px = [
                float(self.ax.transData.transform((xmark, event.ydata if event.ydata is not None else 0.0))[0])
                for xmark in self.selected_wavelengths
            ]
            nearest = min(range(len(marker_px)), key=lambda i: abs(marker_px[i] - click_x_px))
            if abs(marker_px[nearest] - click_x_px) <= 8.0:
                self.drag_selection_index = nearest
                return

        # Otherwise add a new marker at the nearest actually measured spectral position.
        x = self._nearest_spectral_position(event.xdata)
        if any(abs(x - old) < 1e-9 for old in self.selected_wavelengths):
            return
        self.selected_wavelengths.append(x)
        self._update_selected_list()
        self._plot_preview(preserve_view=True)

    def _on_preview_motion(self, event):
        if self.drag_range_index is not None:
            if self.wavelengths is None or event.inaxes is not self.ax or event.xdata is None:
                return
            new_x = self._nearest_spectral_position(event.xdata)
            ridx = self.drag_range_index
            if 0 <= ridx < len(self.selected_ranges):
                lo, hi = self.selected_ranges[ridx]
                if self.drag_range_edge == "lo":
                    lo = min(new_x, hi)
                else:
                    hi = max(new_x, lo)
                if hi > lo:
                    self.selected_ranges[ridx] = (lo, hi)
                    self._update_selected_ranges_list()
                    self._plot_preview(preserve_view=True)
            return

        if self.drag_selection_index is None:
            return
        if self.wavelengths is None or event.inaxes is not self.ax or event.xdata is None:
            return
        new_x = self._nearest_spectral_position(event.xdata)
        i = self.drag_selection_index
        # Avoid two markers occupying the same spectral position.
        if any(j != i and abs(new_x - old) < 1e-9 for j, old in enumerate(self.selected_wavelengths)):
            return
        self.selected_wavelengths[i] = new_x
        if i < len(self.selection_lines):
            self.selection_lines[i].set_xdata([new_x, new_x])
        self._update_selected_list()
        self.canvas.draw_idle()

    def _leave_zoom_mode_after_release(self):
        """Make Matplotlib zoom a one-operation mode for easier peak picking."""
        toolbar = getattr(self.canvas, "toolbar", None)
        if toolbar is None:
            return
        mode = str(getattr(toolbar, "mode", ""))
        if "zoom" in mode.lower():
            try:
                toolbar.zoom()  # toggle Zoom off, preserving the new view limits
            except Exception:
                pass

    def _on_preview_release(self, event):
        if self.drag_range_index is not None:
            self.drag_range_index = None
            self.drag_range_edge = None
            self._update_selected_ranges_list()
            self._plot_preview(preserve_view=True)

        toolbar = getattr(self.canvas, "toolbar", None)
        zoom_was_active = toolbar is not None and "zoom" in str(getattr(toolbar, "mode", "")).lower()
        if self.drag_selection_index is not None:
            self.drag_selection_index = None
            # Moving a marker must keep the manually chosen zoom.
            self._plot_preview(preserve_view=True)
        if zoom_was_active:
            QTimer.singleShot(0, self._leave_zoom_mode_after_release)

    def _remove_selected_wavelength(self):
        """Remove only the item currently highlighted in the selection list."""
        row = self.selected.currentRow()
        if 0 <= row < len(self.selected_wavelengths):
            self.selected_wavelengths.pop(row)
            self.drag_selection_index = None
            self._update_selected_list()
            if self.wavelengths is not None:
                self._plot_preview(preserve_view=True)

    def _clear_selected_wavelengths(self):
        self.selected_wavelengths = []
        self.drag_selection_index = None
        self.selected.clear()
        if self.wavelengths is not None:
            self._plot_preview(preserve_view=True)

    def _validated_odd_window(self, requested: int, npts: int):
        w = int(requested)
        if w % 2 == 0:
            w += 1
        max_odd = npts if npts % 2 == 1 else npts - 1
        w = min(w, max_odd)
        return w

    def _smoothed_cv(self):
        data = self.cv_currents
        if data is None or not self.cv_smooth.isChecked():
            return data
        w = self._validated_odd_window(self.cv_window.value(), data.shape[0])
        if w < 5:
            return data
        poly = min(3, w - 2)
        return savgol_filter(data, window_length=w, polyorder=poly, axis=0)

    def _spectral_potential_mapping(self):
        if self.spectra is None:
            raise ValueError("No spectra loaded.")
        e0 = float(self.e_start.value())
        es = float(self.e_switch.value())
        rate = float(self.scan_rate.value())
        cycles = int(self.cycles.value())
        if rate <= 0:
            raise ValueError("Scan rate must be greater than zero.")
        span = abs(e0 - es)
        if span <= 0:
            raise ValueError("Start and switching potentials must be different.")
        half_time = span / rate
        full_time = 2.0 * half_time
        total_time = cycles * full_time
        t = np.linspace(0.0, total_time, self.spectra.shape[0])
        phase = np.mod(t, full_time)
        direction = np.sign(es - e0)
        if direction == 0:
            direction = -1.0
        e = np.where(
            phase <= half_time,
            e0 + direction * rate * phase,
            es - direction * rate * (phase - half_time),
        )
        e[-1] = e0
        cyc = np.clip(np.floor(t / full_time).astype(int) + 1, 1, cycles)
        return t, e, cyc

    def _waterfall_cycle_path(self, e_spec, scan_spec):
        """Return an unfolded x-coordinate for the 3D waterfall.

        Each complete CV cycle (+E -> -E -> +E, or the reverse) is placed
        consecutively along the 3D x-axis instead of plotting all cycles on
        top of one another. Tick labels still show the physical potential.
        """
        e_spec = np.asarray(e_spec, dtype=float)
        scan_spec = np.asarray(scan_spec, dtype=int)

        e0 = float(self.e_start.value())
        es = float(self.e_switch.value())
        span = abs(e0 - es)
        cycle_width = 2.0 * span

        path = np.zeros_like(e_spec, dtype=float)

        for s in range(1, int(self.cycles.value()) + 1):
            inds = np.flatnonzero(scan_spec == s)
            if len(inds) == 0:
                continue

            ev = e_spec[inds]
            # Find the switching point within this cycle.
            switch_local = int(np.nanargmin(np.abs(ev - es)))

            # Distance travelled from start potential to switching potential,
            # then back toward the start potential.
            first = np.abs(ev[:switch_local + 1] - e0)
            second = span + np.abs(ev[switch_local + 1:] - es)
            local_path = np.concatenate([first, second])

            path[inds] = (s - 1) * cycle_width + local_path

        # Repeated potential tick labels for each cycle.
        tickvals = []
        ticktext = []
        midpoint_potential = 0.5 * (e0 + es)

        for s in range(int(self.cycles.value())):
            base = s * cycle_width
            # Five ticks: start, midpoint, switch, midpoint, start.
            tickvals.extend([
                base,
                base + 0.5 * span,
                base + span,
                base + 1.5 * span,
                base + 2.0 * span,
            ])
            ticktext.extend([
                f"{e0:.1f}",
                f"{midpoint_potential:.1f}",
                f"{es:.1f}",
                f"{midpoint_potential:.1f}",
                f"{e0:.1f}",
            ])

        return path, tickvals, ticktext, cycle_width

    def _synchronize_and_plot(self):
        if self.cv_potential is None or self.cv_currents is None:
            QMessageBox.information(self, "Synchronize + Plot", "Please load a CV CSV first.")
            return
        if self.wavelengths is None or self.spectra is None:
            QMessageBox.information(self, "Synchronize + Plot", "Please load a spectra CSV first.")
            return
        has_point_selection = bool(self.selected_wavelengths)
        has_range_selection = (
            self.spectral_mode == "Raman"
            and hasattr(self, "selected_ranges")
            and bool(self.selected_ranges)
        )
        if not (has_point_selection or has_range_selection):
            QMessageBox.information(
                self, "Synchronize + Plot",
                "Please select at least one Raman shift or Raman peak range in the spectrum preview."
                if self.spectral_mode == "Raman"
                else "Please select at least one wavelength in the spectrum preview."
            )
            return
        try:
            fig = self._build_interactive_figure()
            tmp = Path(tempfile.gettempdir()) / "SpectroCV_synchronized_plot.html"
            fig.write_html(tmp, include_plotlyjs=True, full_html=True, config={
                "displaylogo": False, "scrollZoom": True, "responsive": True, "displayModeBar": True,
                "toImageButtonOptions": {"format": "png", "filename": "SpectroCV_current_view", "scale": 2}
            })
            self.last_plot_html = tmp
            for b in (self.export_excel_btn, self.export_png_btn, self.export_html_btn):
                b.setEnabled(True)
                b.setToolTip("")
            self.export_png_btn.setToolTip(
                "Saves the standardized PNG view. For a PNG of the currently rotated/zoomed "
                "3D view, use the camera icon in the interactive Plotly window."
            )
            webbrowser.open(tmp.as_uri())
        except Exception as exc:
            QMessageBox.critical(self, "Synchronize + Plot", f"Could not create plot:\n{exc}")

    def _check_export_ready(self):
        if self.cv_potential is None or self.cv_currents is None:
            QMessageBox.information(self, "Export", "Please load a CV CSV first.")
            return False
        if self.wavelengths is None or self.spectra is None:
            QMessageBox.information(self, "Export", "Please load a spectra CSV first.")
            return False
        has_point_selection = bool(self.selected_wavelengths)
        has_range_selection = (
            self.spectral_mode == "Raman"
            and hasattr(self, "selected_ranges")
            and bool(self.selected_ranges)
        )
        if not (has_point_selection or has_range_selection):
            QMessageBox.information(
                self, "Export",
                "Please select at least one Raman shift or Raman peak range first."
                if self.spectral_mode == "Raman"
                else "Please select at least one wavelength first."
            )
            return False
        return True

    def _export_html(self):
        if not self._check_export_ready():
            return
        p, _ = QFileDialog.getSaveFileName(
            self, "Export Interactive HTML", "SpectroCV_interactive.html", "HTML files (*.html)"
        )
        if not p:
            return
        try:
            if not p.lower().endswith(".html"):
                p += ".html"
            fig = self._build_interactive_figure()
            fig.write_html(p, include_plotlyjs=True, full_html=True, config={
                "displaylogo": False, "scrollZoom": True, "responsive": True, "displayModeBar": True,
                "toImageButtonOptions": {"format": "png", "filename": "SpectroCV_current_view", "scale": 2}
            })
            QMessageBox.information(self, "Export Interactive HTML", f"Saved:\n{p}")
        except Exception as exc:
            QMessageBox.critical(self, "Export Interactive HTML", f"Could not export HTML:\n{exc}")


    def _update_derivative_controls(self):
        mode = self.derivative_presentation.currentText() if hasattr(self, "derivative_presentation") else "PCHIP"
        # SG window is meaningful only for Savitzky-Golay smoothing.
        self.derivative_window.setEnabled(mode == "Savitzky–Golay")
        self.derivative_window.setToolTip(
            "Used for Savitzky–Golay smoothing. Larger windows produce stronger smoothing."
        )

    @staticmethod
    def _split_scan_branches(potential):
        """Return contiguous index arrays for monotonic branches of one CV scan."""
        potential = np.asarray(potential, dtype=float)
        n = len(potential)
        if n <= 1:
            return [np.arange(n, dtype=int)]
        d = np.diff(potential)
        signs = np.sign(d)
        # Fill zero direction from nearest previous/nonzero direction.
        for i in range(len(signs)):
            if signs[i] == 0:
                signs[i] = signs[i-1] if i > 0 and signs[i-1] != 0 else 1
        cuts = [0]
        for i in range(1, len(signs)):
            if signs[i] != signs[i-1]:
                cuts.append(i)
        cuts.append(n-1)
        branches = []
        for j in range(len(cuts)-1):
            a = cuts[j]
            b = cuts[j+1] + 1
            branch = np.arange(a, b, dtype=int)
            if len(branch) >= 2:
                branches.append(branch)
        return branches or [np.arange(n, dtype=int)]

    def _derivative_display_series(self, potential, values):
        """Prepare derivative values for plotting only.

        None: original adjacent-point derivative.
        Savitzky–Golay: smooth derivative values branch-wise.
        PCHIP: shape-preserving interpolation branch-wise with increased point density.
        """
        x = np.asarray(potential, dtype=float)
        y = np.asarray(values, dtype=float)
        mode = self.derivative_presentation.currentText()

        if mode == "None":
            mask = np.isfinite(x) & np.isfinite(y)
            return x[mask], y[mask]

        branches = self._split_scan_branches(x)
        x_out, y_out = [], []

        for branch in branches:
            xb = x[branch]
            yb = y[branch]
            mask = np.isfinite(xb) & np.isfinite(yb)
            xb, yb = xb[mask], yb[mask]
            if len(xb) < 2:
                continue

            if mode == "Savitzky–Golay":
                win = int(self.derivative_window.value())
                if win % 2 == 0:
                    win += 1
                max_odd = len(yb) if len(yb) % 2 == 1 else len(yb) - 1
                win = min(win, max_odd)
                if win >= 3:
                    poly = min(3, win - 1)
                    ys = savgol_filter(yb, window_length=win, polyorder=poly, mode="interp")
                else:
                    ys = yb
                xd = xb
            else:  # PCHIP
                # PCHIP requires increasing x. Reverse cathodic branches for interpolation,
                # then reverse the result back so acquisition order is preserved.
                reverse = xb[0] > xb[-1]
                xi = xb[::-1] if reverse else xb
                yi = yb[::-1] if reverse else yb

                # Remove duplicate x values, retaining acquisition-order values.
                keep = np.r_[True, np.diff(xi) > 1e-12]
                xi, yi = xi[keep], yi[keep]
                if len(xi) < 2:
                    continue

                # Fixed internal resolution for presentation only.
                # PCHIP density does not control smoothing strength, so exposing
                # it in the UI would be misleading.
                density = 10
                n_dense = (len(xi) - 1) * density + 1
                xd_inc = np.linspace(xi[0], xi[-1], n_dense)
                yd_inc = PchipInterpolator(xi, yi, extrapolate=False)(xd_inc)
                if reverse:
                    xd, ys = xd_inc[::-1], yd_inc[::-1]
                else:
                    xd, ys = xd_inc, yd_inc

            # NaN separator prevents Excel from visually bridging branches.
            if x_out:
                x_out.append(np.nan)
                y_out.append(np.nan)
            x_out.extend(np.asarray(xd, dtype=float).tolist())
            y_out.extend(np.asarray(ys, dtype=float).tolist())

        return np.asarray(x_out, dtype=float), np.asarray(y_out, dtype=float)

    def _potential_derivative(self, signal, potential, scan_ids):
        """Mathematical potential derivative dS/dE using adjacent spectral points.

        The potential vector must be the potential assigned to the spectra
        (Potential_signal_V), not the finer CV potential grid.
        """
        signal = np.asarray(signal, dtype=float)
        potential = np.asarray(potential, dtype=float)
        scan_ids = np.asarray(scan_ids)

        deriv = np.full(signal.shape, np.nan, dtype=float)
        if len(signal) < 2:
            return deriv

        for i in range(1, len(signal)):
            if scan_ids[i] != scan_ids[i - 1]:
                continue

            dE = potential[i] - potential[i - 1]
            dS = signal[i] - signal[i - 1]
            if not np.isfinite(dE) or not np.isfinite(dS) or abs(dE) < 1e-15:
                continue

            deriv[i] = dS / dE

        return deriv

    def _scan_direction_derivative(self, signal, potential, scan_ids):
        """Scan-direction signal change dS/|dE| using adjacent spectral points."""
        signal = np.asarray(signal, dtype=float)
        potential = np.asarray(potential, dtype=float)
        scan_ids = np.asarray(scan_ids)

        deriv = np.full(signal.shape, np.nan, dtype=float)
        if len(signal) < 2:
            return deriv

        for i in range(1, len(signal)):
            if scan_ids[i] != scan_ids[i - 1]:
                continue

            dE = potential[i] - potential[i - 1]
            dS = signal[i] - signal[i - 1]
            if not np.isfinite(dE) or not np.isfinite(dS) or abs(dE) < 1e-15:
                continue

            deriv[i] = dS / abs(dE)

        return deriv

    def _export_excel(self):
        if not self._check_export_ready():
            return
        p, _ = QFileDialog.getSaveFileName(
            self, "Export Excel", "SpectroCV_synchronized.xlsx", "Excel files (*.xlsx)"
        )
        if not p:
            return
        if not p.lower().endswith(".xlsx"):
            p += ".xlsx"
        try:
            cv_proc = self._smoothed_cv()
            spec_proc = self._smoothed_spectra_for_preview()
            t, e_spec, scan_spec = self._spectral_potential_mapping()

            # Parameters
            params = pd.DataFrame({
                "Parameter": [
                    "Spectral mode", "Start potential / V", "Switching potential / V",
                    "Scan rate / V s^-1", "Potential increment / V", "Number of cycles",
                    "Integration time / ms", "Averages", "Spectral range from",
                    "Spectral range to", "CV smoothing", "CV smoothing window / points",
                    "Spectral smoothing", "Spectral smoothing window / points",
                    "Raman baseline correction", "Baseline lambda", "AsLS p",
                    "Derivative presentation", "Derivative SG window / points",
                    "Selected spectral positions"
                ],
                "Value": [
                    self.spectral_mode, self.e_start.value(), self.e_switch.value(),
                    self.scan_rate.value(), self.increment.value(), self.cycles.value(),
                    self.integration.value(), self.averages.value(), self.wl_min.value(),
                    self.wl_max.value(), self.cv_smooth.isChecked(), self.cv_window.value(),
                    self.spec_smooth.isChecked(), self.spec_window.value(),
                    (self.baseline_method.currentText() if self.spectral_mode == "Raman" else "n/a"),
                    (self.asls_lambda.value() if self.spectral_mode == "Raman" else "n/a"),
                    (self.asls_p.value() if self.spectral_mode == "Raman" else "n/a"),
                    self.derivative_presentation.currentText(),
                    self.derivative_window.value(),
                    ", ".join(f"{x:.3f}" for x in self.selected_wavelengths)
                ]
            })

            # CV raw + processed.
            cv_data = {"Potential_V": self.cv_potential}
            for j in range(self.cv_currents.shape[1]):
                cv_data[f"Current_{j+1}_raw_A"] = self.cv_currents[:, j]
                cv_data[f"Current_{j+1}_processed_A"] = cv_proc[:, j]
            cv_df = pd.DataFrame(cv_data)

            # Synchronized selected spectral positions.
            sync_data = {
                "Spectrum_index": np.arange(1, len(t) + 1),
                "Time_s": t,
                "Scan": scan_spec,
                "Potential_V": e_spec,
            }
            unit = "cm-1" if self.spectral_mode == "Raman" else "nm"
            selected_actual = []
            for chosen in self.selected_wavelengths:
                idx = int(np.nanargmin(np.abs(self.wavelengths - chosen)))
                actual = float(self.wavelengths[idx])
                selected_actual.append((idx, actual))
                sync_data[f"Signal_{actual:.3f}_{unit}"] = spec_proc[:, idx]
            range_actual=[]
            if self.spectral_mode=="Raman":
                for rlo,rhi in self.selected_ranges:
                    rsig=self._integrated_range_signal(spec_proc,rlo,rhi)
                    rheader=f"Integrated_Raman_{rlo:.1f}-{rhi:.1f}_cm-1"
                    sync_data[rheader]=rsig
                    range_actual.append((rlo,rhi,rheader,rsig))
            sync_df = pd.DataFrame(sync_data)

            # Full processed spectral matrix. IMPORTANT: wavelength headers are written as NUMBERS,
            # not strings. This allows Excel XY charts to use the header row as the true x-axis.
            spec_df = pd.DataFrame(spec_proc, columns=[float(x) for x in self.wavelengths])
            spec_df.insert(0, "Potential_V", e_spec)
            spec_df.insert(0, "Scan", scan_spec)
            spec_df.insert(0, "Time_s", t)
            spec_df.insert(0, "Spectrum_index", np.arange(1, len(t) + 1))

            # Tables for the combined chart. CV currents are stored in microampere.
            combined_data = {"Potential_V": self.cv_potential}
            for j in range(cv_proc.shape[1]):
                combined_data[f"CV_scan_{j+1}_Current_uA"] = cv_proc[:, j] * 1e6
            combined_df = pd.DataFrame(combined_data)

            signal_chart_df = pd.DataFrame({"Potential_signal_V": e_spec})
            signal_headers = []

            # The derivative table deliberately keeps the spectral potential
            # next to the corresponding spectral signal. This prevents confusion
            # with the much finer CV potential grid.
            derivative_chart_df = pd.DataFrame({
                "Spectrum_index": np.arange(1, len(e_spec) + 1),
                "Scan": scan_spec,
                "Potential_signal_V": e_spec,
            })
            derivative_headers = []
            scan_derivative_headers = []
            derivative_signal_headers = []

            for idx, actual in selected_actual:
                header = f"{self._spectral_y_label()}_{actual:.3f}_{unit}"
                signal = spec_proc[:, idx]
                signal_chart_df[header] = signal
                signal_headers.append(header)

                sig_header = f"{self._spectral_y_label()}_{actual:.3f}_{unit}"
                dheader = f"d{self._spectral_y_label()}_dE_{actual:.3f}_{unit}_per_V"
                sdheader = f"d{self._spectral_y_label()}_dAbsE_{actual:.3f}_{unit}_per_V"
                derivative_chart_df[sig_header] = signal
                derivative_chart_df[dheader] = self._potential_derivative(signal, e_spec, scan_spec)
                derivative_chart_df[sdheader] = self._scan_direction_derivative(signal, e_spec, scan_spec)
                derivative_signal_headers.append(sig_header)
                derivative_headers.append(dheader)
                scan_derivative_headers.append(sdheader)

            for rlo,rhi,rheader,rsig in range_actual:
                signal_chart_df[rheader]=rsig
                signal_headers.append(rheader)

                # Derivatives of the integrated Raman peak area versus potential.
                # This makes range-only selections fully equivalent to selecting
                # an individual Raman shift for derivative analysis.
                sig_header = rheader
                dheader = f"dIntegrated_Raman_dE_{rlo:.1f}-{rhi:.1f}_cm-1_per_V"
                sdheader = f"dIntegrated_Raman_dAbsE_{rlo:.1f}-{rhi:.1f}_cm-1_per_V"

                derivative_chart_df[sig_header] = rsig
                derivative_chart_df[dheader] = self._potential_derivative(
                    rsig, e_spec, scan_spec
                )
                derivative_chart_df[sdheader] = self._scan_direction_derivative(
                    rsig, e_spec, scan_spec
                )

                derivative_signal_headers.append(sig_header)
                derivative_headers.append(dheader)
                scan_derivative_headers.append(sdheader)

            # Plot-only derivative helper data. Original derivative values remain unchanged
            # in derivative_chart_df; this table is used only by the visible Excel chart.
            derivative_display_data = {}
            max_display_len = 0
            display_pairs = []

            for k, sdheader in enumerate(scan_derivative_headers):
                for s in range(1, int(self.cycles.value()) + 1):
                    inds = np.flatnonzero(scan_spec == s)
                    if len(inds) == 0:
                        continue

                    xs = e_spec[inds]
                    ys = derivative_chart_df.loc[inds, sdheader].to_numpy(dtype=float)
                    xd, yd = self._derivative_display_series(xs, ys)

                    # X and Y from one curve must always have identical lengths.
                    n_pair = min(len(xd), len(yd))
                    xd = np.asarray(xd[:n_pair], dtype=float)
                    yd = np.asarray(yd[:n_pair], dtype=float)

                    key_x = f"E_w{k+1}_scan{s}"
                    key_y = f"D_w{k+1}_scan{s}"
                    display_pairs.append((key_x, key_y, xd, yd))
                    max_display_len = max(max_display_len, n_pair)

            # Different scans/wavelengths can contain different numbers of valid
            # derivative points. Pad every plot-only series to a common length
            # before constructing the DataFrame. This avoids pandas'
            # "All arrays must be of the same length" error while preserving
            # each X/Y pair exactly.
            for key_x, key_y, xd, yd in display_pairs:
                xpad = np.full(max_display_len, np.nan, dtype=float)
                ypad = np.full(max_display_len, np.nan, dtype=float)
                xpad[:len(xd)] = xd
                ypad[:len(yd)] = yd
                derivative_display_data[key_x] = xpad
                derivative_display_data[key_y] = ypad

            derivative_display_df = pd.DataFrame(derivative_display_data)

            # XlsxWriter is used intentionally here. It handles combined primary/secondary axes
            # and explicit XY-series references much more reliably than the previous chart code.
            # Optional mean-over-cycles table for Excel.
            mean_export_df = None
            if int(self.cycles.value()) > 1:
                mean_cols = {}

                # Mean CV on the common CV potential grid.
                # cv_proc has shape (n_potential_points, n_cycles) and is already
                # aligned point-by-point for all CV cycles. Average across cycles
                # (axis=1), then convert A -> µA.
                mean_cols["Potential_V_CV_mean"] = pd.Series(self.cv_potential)
                mean_cols["CV_mean_Current_uA"] = pd.Series(
                    np.nanmean(cv_proc, axis=1) * 1e6
                )

                # Mean selected Raman/wavelength traces.  Complete cycles are aligned
                # along their forward/reverse trajectory before averaging.
                # Reuse the synchronized arrays already calculated above in this export:
                # e_spec = potential assigned to each spectrum
                # scan_spec = cycle index of each spectrum
                # spec_proc = processed/baseline-corrected spectral matrix
                e_mean = e_spec
                scan_mean = scan_spec
                spectra_mean = spec_proc
                for selected_value in self.selected_wavelengths:
                    idx_mean = int(np.argmin(np.abs(self.wavelengths - selected_value)))
                    actual_mean = float(self.wavelengths[idx_mean])
                    emean, ymean = self._cycle_mean_trace(
                        e_mean, scan_mean, spectra_mean[:, idx_mean]
                    )
                    if len(emean):
                        if "Potential_V_signal_mean" not in mean_cols:
                            mean_cols["Potential_V_signal_mean"] = pd.Series(emean)
                        if self.spectral_mode == "Raman":
                            cname = f"Mean_Raman_{actual_mean:.3f}_cm-1_a.u."
                        else:
                            cname = f"Mean_signal_{actual_mean:.3f}_nm"
                        mean_cols[cname] = pd.Series(ymean)
                if self.spectral_mode=="Raman":
                    for rlo,rhi in self.selected_ranges:
                        rsig=self._integrated_range_signal(spectra_mean,rlo,rhi)
                        emean,ymean=self._cycle_mean_trace(e_mean,scan_mean,rsig)
                        if len(emean):
                            if "Potential_V_signal_mean" not in mean_cols:
                                mean_cols["Potential_V_signal_mean"]=pd.Series(emean)
                            mean_cols[f"Mean_Integrated_Raman_{rlo:.1f}-{rhi:.1f}_cm-1"]=pd.Series(ymean)

                mean_export_df = pd.DataFrame(mean_cols)

            with pd.ExcelWriter(p, engine="xlsxwriter") as writer:
                params.to_excel(writer, sheet_name="Parameters", index=False)
                cv_df.to_excel(writer, sheet_name="CV", index=False)
                sync_df.to_excel(writer, sheet_name="Selected signals", index=False)
                spec_df.to_excel(writer, sheet_name="Spectra synchronized", index=False)
                if mean_export_df is not None:
                    mean_export_df.to_excel(writer, sheet_name="Cycle means", index=False)
                combined_df.to_excel(writer, sheet_name="CV + selected signals", index=False, startcol=0)
                signal_start_col = combined_df.shape[1] + 1  # one blank separator column
                signal_chart_df.to_excel(
                    writer, sheet_name="CV + selected signals", index=False, startcol=signal_start_col
                )

                # Additional worksheet: CV + potential derivative of selected spectral signal(s).
                # The derivative table is placed first so Potential_signal_V is visibly paired
                # with the spectral signal and its derivative. CV helper data are kept separately
                # to the right and are used only for the combined chart.
                derivative_start_col = 0
                derivative_chart_df.to_excel(
                    writer, sheet_name="CV + signal derivatives", index=False, startcol=derivative_start_col
                )
                derivative_cv_start_col = derivative_chart_df.shape[1] + 2
                combined_df.to_excel(
                    writer, sheet_name="CV + signal derivatives", index=False, startcol=derivative_cv_start_col
                )
                derivative_display_start_col = derivative_cv_start_col + combined_df.shape[1] + 2
                derivative_display_df.to_excel(
                    writer, sheet_name="CV + signal derivatives", index=False,
                    startcol=derivative_display_start_col
                )

                wb = writer.book
                header_fmt = wb.add_format({
                    "bold": True, "bg_color": "#EAF2F8", "border": 1, "align": "center"
                })
                num_fmt = wb.add_format({"num_format": "0.000000"})

                # General worksheet formatting.
                for sheet_name, ws in writer.sheets.items():
                    ws.freeze_panes(1, 0)
                    ws.set_row(0, 20, header_fmt)
                    ws.set_column(0, 0, 18)
                    ws.set_column(1, 3, 15, num_fmt)

                writer.sheets["Parameters"].set_column(0, 0, 34)
                writer.sheets["Parameters"].set_column(1, 1, 24)
                writer.sheets["CV"].set_column(0, cv_df.shape[1]-1, 22, num_fmt)
                writer.sheets["Selected signals"].set_column(0, sync_df.shape[1]-1, 19, num_fmt)
                writer.sheets["Spectra synchronized"].set_column(0, 3, 17, num_fmt)
                writer.sheets["Spectra synchronized"].set_column(4, spec_df.shape[1]-1, 12, num_fmt)
                writer.sheets["CV + selected signals"].set_column(0, combined_df.shape[1]-1, 20, num_fmt)
                writer.sheets["CV + selected signals"].set_column(
                    signal_start_col, signal_start_col + signal_chart_df.shape[1]-1, 20, num_fmt
                )
                writer.sheets["CV + signal derivatives"].set_column(
                    derivative_start_col,
                    derivative_start_col + derivative_chart_df.shape[1]-1,
                    22, num_fmt
                )
                writer.sheets["CV + signal derivatives"].set_column(
                    derivative_cv_start_col,
                    derivative_cv_start_col + combined_df.shape[1]-1,
                    20, num_fmt
                )
                if derivative_display_df.shape[1] > 0:
                    writer.sheets["CV + signal derivatives"].set_column(
                        derivative_display_start_col,
                        derivative_display_start_col + derivative_display_df.shape[1]-1,
                        18, num_fmt
                    )

                e_min = float(min(self.e_start.value(), self.e_switch.value()))
                e_max = float(max(self.e_start.value(), self.e_switch.value()))
                colors = ["#7C3AED", "#F97316", "#0EA5E9", "#16A34A", "#DC2626", "#A855F7"]

                # 1) Selected signals: potential-dependent spectral signal(s).
                if signal_headers:
                    ws_sig = writer.sheets["Selected signals"]
                    sig_chart = wb.add_chart({"type": "scatter", "subtype": "smooth"})
                    dash_styles = ["solid", "dash", "dot", "dash_dot"]
                    for k, header in enumerate(signal_headers):
                        color = colors[k % len(colors)]
                        for s in range(1, int(self.cycles.value()) + 1):
                            inds = np.flatnonzero(scan_spec == s)
                            if len(inds) == 0:
                                continue
                            r1, r2 = int(inds[0]) + 1, int(inds[-1]) + 1
                            line_opts = {"color": color, "width": 2.0}
                            dash = dash_styles[(s-1) % len(dash_styles)]
                            if dash != "solid":
                                line_opts["dash_type"] = dash
                            sig_chart.add_series({
                                "name": f"{header} – scan {s}",
                                "categories": ["Selected signals", r1, 3, r2, 3],
                                "values": ["Selected signals", r1, 4+k, r2, 4+k],
                                "line": line_opts,
                                "marker": {"type": "none"},
                            })
                    yr = self._padded_range(sync_df.iloc[:, 4:].to_numpy(), fraction=0.06)
                    yopts = {"name": self._spectral_y_label(), "major_gridlines": {"visible": True}}
                    if yr is not None:
                        yopts.update({"min": yr[0], "max": yr[1]})
                    sig_chart.set_x_axis({
                        "name": "Potential / V", "min": e_min, "max": e_max,
                        "major_unit": 0.2, "crossing": e_min,
                    })
                    sig_chart.set_y_axis(yopts)
                    sig_chart.set_legend({"position": "bottom"})
                    sig_chart.set_size({"width": 880, "height": 500})
                    ws_sig.insert_chart("G2", sig_chart)

                # 2) Spectra synchronized: TRUE spectra, x = wavelength/Raman shift,
                # y = spectral intensity/absorbance. Each ROW is one spectrum.
                ws_spec = writer.sheets["Spectra synchronized"]
                spec_chart = wb.add_chart({"type": "scatter", "subtype": "smooth"})
                first_spec_col = 4
                last_spec_col = first_spec_col + len(self.wavelengths) - 1
                for i in range(len(spec_df)):
                    spec_chart.add_series({
                        "categories": ["Spectra synchronized", 0, first_spec_col, 0, last_spec_col],
                        "values": ["Spectra synchronized", i+1, first_spec_col, i+1, last_spec_col],
                        "line": {"width": 0.75},
                        "marker": {"type": "none"},
                    })
                x_label = "Raman shift / cm⁻¹" if self.spectral_mode == "Raman" else "Wavelength / nm"
                spec_chart.set_x_axis({
                    "name": x_label,
                    "min": float(np.nanmin(self.wavelengths)),
                    "max": float(np.nanmax(self.wavelengths)),
                    "major_gridlines": {"visible": True},
                })
                syr = self._padded_range(spec_proc, fraction=0.06)
                syopts = {"name": self._spectral_y_label(), "major_gridlines": {"visible": True}}
                if syr is not None:
                    syopts.update({"min": syr[0], "max": syr[1]})
                spec_chart.set_y_axis(syopts)
                spec_chart.set_legend({"none": True})
                spec_chart.set_size({"width": 900, "height": 520})
                ws_spec.insert_chart("F2", spec_chart)

                # 3) CV + selected signals: same potential range, primary current ordinate and
                # secondary spectral ordinate. CV = black, selected wavelength(s) = colored.
                ws_comb = writer.sheets["CV + selected signals"]
                cv_chart = wb.add_chart({"type": "scatter", "subtype": "smooth"})
                dash_styles = ["solid", "dash", "dot"]
                for j in range(cv_proc.shape[1]):
                    series_opts = {
                        "name": f"CV scan {j+1}",
                        "categories": ["CV + selected signals", 1, 0, len(combined_df), 0],
                        "values": ["CV + selected signals", 1, 1+j, len(combined_df), 1+j],
                        "line": {"color": "#000000", "width": 2.25},
                        "marker": {"type": "none"},
                    }
                    if j < len(dash_styles) and dash_styles[j] != "solid":
                        series_opts["line"]["dash_type"] = dash_styles[j]
                    cv_chart.add_series(series_opts)

                # Selected spectral traces use the same scan coding as the CV:
                # scan 1 solid, scan 2 dashed, scan 3 dotted (same color per wavelength).
                for k, header in enumerate(signal_headers):
                    color = colors[k % len(colors)]
                    for s in range(1, int(self.cycles.value()) + 1):
                        inds = np.flatnonzero(scan_spec == s)
                        if len(inds) == 0:
                            continue
                        r1, r2 = int(inds[0]) + 1, int(inds[-1]) + 1
                        line_opts = {"color": color, "width": 2.0}
                        dash = dash_styles[(s-1) % len(dash_styles)]
                        if dash != "solid":
                            line_opts["dash_type"] = dash
                        cv_chart.add_series({
                            "name": f"{header} – scan {s}",
                            "categories": [
                                "CV + selected signals", r1, signal_start_col,
                                r2, signal_start_col
                            ],
                            "values": [
                                "CV + selected signals", r1, signal_start_col+1+k,
                                r2, signal_start_col+1+k
                            ],
                            "y2_axis": True,
                            "line": line_opts,
                            "marker": {"type": "none"},
                        })

                if mean_export_df is not None and self.show_cycle_mean.isChecked():
                    mh=list(mean_export_df.columns); nr=len(mean_export_df)
                    cv_chart.add_series({"name":"Mean CV","categories":["Cycle means",1,0,nr,0],
                        "values":["Cycle means",1,1,nr,1],"line":{"color":"#000000","width":3.5},
                        "marker":{"type":"none"}})
                    if "Potential_V_signal_mean" in mh:
                        xc=mh.index("Potential_V_signal_mean")
                        mean_sig=[h for h in mh if h.startswith(("Mean_Raman_","Mean_signal_","Mean_Integrated_Raman_"))]
                        for kk,h in enumerate(mean_sig):
                            yc=mh.index(h)
                            cv_chart.add_series({"name":h,"categories":["Cycle means",1,xc,nr,xc],
                                "values":["Cycle means",1,yc,nr,yc],"y2_axis":True,
                                "line":{"color":colors[kk % len(colors)],"width":3.25},"marker":{"type":"none"}})

                if mean_export_df is not None:
                    ws_mean = writer.sheets["Cycle means"]
                    mean_chart = wb.add_chart({"type": "scatter", "subtype": "smooth"})
                    mh = list(mean_export_df.columns)
                    nr = len(mean_export_df)

                    # Mean CV on primary Y axis.
                    if "Potential_V_CV_mean" in mh and "CV_mean_Current_uA" in mh:
                        xcv = mh.index("Potential_V_CV_mean")
                        ycv = mh.index("CV_mean_Current_uA")
                        mean_chart.add_series({
                            "name": "Mean CV",
                            "categories": ["Cycle means", 1, xcv, nr, xcv],
                            "values": ["Cycle means", 1, ycv, nr, ycv],
                            "line": {"color": "#000000", "width": 3.0},
                            "marker": {"type": "none"},
                        })

                    # Mean Raman / spectral traces on secondary Y axis.
                    if "Potential_V_signal_mean" in mh:
                        xsig = mh.index("Potential_V_signal_mean")
                        mean_headers = [
                            h for h in mh
                            if h.startswith(("Mean_Raman_", "Mean_signal_", "Mean_Integrated_Raman_"))
                        ]
                        for kmean, h in enumerate(mean_headers):
                            ysig = mh.index(h)
                            mean_chart.add_series({
                                "name": h,
                                "categories": ["Cycle means", 1, xsig, nr, xsig],
                                "values": ["Cycle means", 1, ysig, nr, ysig],
                                "y2_axis": True,
                                "line": {"color": colors[kmean % len(colors)], "width": 2.5},
                                "marker": {"type": "none"},
                            })

                    mean_chart.set_x_axis({
                        "name": "Potential / V",
                        "min": e_min, "max": e_max, "major_unit": 0.2,
                        "num_format": "0.0",
                    })
                    mean_cv_vals = mean_export_df["CV_mean_Current_uA"].to_numpy(dtype=float)
                    mean_cv_range = self._padded_range(mean_cv_vals, fraction=0.06)
                    mean_current_axis = {"name": "Current / µA", "num_format": "0.00"}
                    if mean_cv_range is not None:
                        mean_current_axis.update({"min": mean_cv_range[0], "max": mean_cv_range[1]})
                    mean_chart.set_y_axis(mean_current_axis)
                    mean_chart.set_y2_axis({"name": self._spectral_y_label()})
                    mean_chart.set_legend({"position": "bottom"})
                    mean_chart.set_size({"width": 1050, "height": 610})
                    ws_mean.insert_chart("H2", mean_chart)

                current_range = self._padded_range(cv_proc * 1e6, fraction=0.06)
                current_axis = {
                    "name": "Current / µA",
                    "major_gridlines": {"visible": True},
                    "num_format": "0.00",
                }
                if current_range is not None:
                    current_axis.update({"min": current_range[0], "max": current_range[1]})

                signal_values = signal_chart_df.iloc[:, 1:].to_numpy() if signal_headers else np.array([0.0, 1.0])
                signal_range = self._padded_range(signal_values, fraction=0.06)
                signal_axis = {
                    "name": self._spectral_y_label(),
                    "major_gridlines": {"visible": False},
                    "num_format": "0.000",
                }
                if signal_range is not None:
                    signal_axis.update({"min": signal_range[0], "max": signal_range[1]})

                cv_chart.set_x_axis({
                    "name": "Potential / V",
                    "min": e_min, "max": e_max, "major_unit": 0.2,
                    "num_format": "0.0",
                })
                cv_chart.set_y_axis(current_axis)
                cv_chart.set_y2_axis(signal_axis)
                cv_chart.set_legend({"position": "bottom"})
                cv_chart.set_plotarea({"border": {"color": "#BFBFBF"}})
                cv_chart.set_chartarea({"border": {"none": True}})
                cv_chart.set_size({"width": 1050, "height": 610})
                ws_comb.insert_chart("A6", cv_chart)

                # 4) CV + potential derivative of selected spectral signal(s): current on the
                # primary ordinate, d(signal)/dE on the secondary ordinate.
                ws_der = writer.sheets["CV + signal derivatives"]
                der_chart = wb.add_chart({"type": "scatter", "subtype": "straight"})

                for j in range(cv_proc.shape[1]):
                    series_opts = {
                        "name": f"CV scan {j+1}",
                        "categories": [
                            "CV + signal derivatives", 1, derivative_cv_start_col,
                            len(combined_df), derivative_cv_start_col
                        ],
                        "values": [
                            "CV + signal derivatives", 1, derivative_cv_start_col + 1 + j,
                            len(combined_df), derivative_cv_start_col + 1 + j
                        ],
                        "line": {"color": "#000000", "width": 2.25},
                        "marker": {"type": "none"},
                    }
                    if j < len(dash_styles) and dash_styles[j] != "solid":
                        series_opts["line"]["dash_type"] = dash_styles[j]
                    der_chart.add_series(series_opts)

                # derivative table columns: Potential, Scan, derivative1, derivative2, ...
                for k, dheader in enumerate(derivative_headers):
                    color = colors[k % len(colors)]
                    for s in range(1, int(self.cycles.value()) + 1):
                        inds = np.flatnonzero(scan_spec == s)
                        if len(inds) == 0:
                            continue
                        r1, r2 = int(inds[0]) + 1, int(inds[-1]) + 1
                        line_opts = {"color": color, "width": 2.0}
                        dash = dash_styles[(s-1) % len(dash_styles)]
                        if dash != "solid":
                            line_opts["dash_type"] = dash
                        # Visible derivative uses the selected presentation mode
                        # (None / Savitzky-Golay / PCHIP) from a plot-only helper table.
                        scan_dheader = scan_derivative_headers[k]
                        helper_pair = 2 * (k * int(self.cycles.value()) + (s - 1))
                        helper_x_col = derivative_display_start_col + helper_pair
                        helper_y_col = helper_x_col + 1
                        helper_rows = len(derivative_display_df)
                        der_chart.add_series({
                            "name": f"{scan_dheader} – scan {s}",
                            "categories": [
                                "CV + signal derivatives", 1, helper_x_col,
                                helper_rows, helper_x_col
                            ],
                            "values": [
                                "CV + signal derivatives", 1, helper_y_col,
                                helper_rows, helper_y_col
                            ],
                            "y2_axis": True,
                            "line": line_opts,
                            "marker": {"type": "none"},
                        })

                deriv_values = (
                    derivative_display_df.iloc[:, 1::2].to_numpy()
                    if derivative_display_df.shape[1] >= 2 else np.array([0.0, 1.0])
                )
                deriv_range = self._padded_range(deriv_values, fraction=0.06)
                if self.spectral_mode == "Absorbance":
                    derivative_axis_name = "derived absorbance / 1/V"
                elif self.spectral_mode == "Fluorescence":
                    derivative_axis_name = "derived fluorescence / 1/V"
                else:
                    if self.spectral_mode == "Raman" and self.selected_ranges and not self.selected_wavelengths:
                        derivative_axis_name = "Scan-direction derivative of integrated Raman intensity / (a.u. cm⁻¹ V⁻¹)"
                    else:
                        derivative_axis_name = "derived Raman signal / 1/V"

                deriv_axis = {
                    "name": derivative_axis_name,
                    "major_gridlines": {"visible": False},
                    "num_format": "0.000",
                }
                if deriv_range is not None:
                    deriv_axis.update({"min": deriv_range[0], "max": deriv_range[1]})

                der_chart.set_x_axis({
                    "name": "Potential / V",
                    "min": e_min, "max": e_max, "major_unit": 0.2,
                    "num_format": "0.0",
                })
                der_chart.set_y_axis(current_axis)
                der_chart.set_y2_axis(deriv_axis)
                der_chart.set_legend({"position": "bottom"})
                der_chart.set_plotarea({"border": {"color": "#BFBFBF"}})
                der_chart.set_chartarea({"border": {"none": True}})
                der_chart.set_size({"width": 1050, "height": 610})
                ws_der.insert_chart("A6", der_chart)

            QMessageBox.information(self, "Export Excel", f"Saved:\n{p}")
        except Exception as exc:
            QMessageBox.critical(self, "Export Excel", f"Could not export Excel file:\n{exc}")

    def _export_png(self):
        if not self._check_export_ready():
            return
        p, _ = QFileDialog.getSaveFileName(self, "Export PNG", "SpectroCV_plot.png", "PNG files (*.png)")
        if not p:
            return
        if not p.lower().endswith(".png"):
            p += ".png"
        try:
            self._save_static_png(Path(p))
            QMessageBox.information(self, "Export PNG", f"Saved:\n{p}")
        except Exception as exc:
            QMessageBox.critical(self, "Export PNG", f"Could not export PNG:\n{exc}")

    @staticmethod
    def _padded_range(values, fraction=0.07, min_pad=None):
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return None
        lo = float(np.min(arr))
        hi = float(np.max(arr))
        span = hi - lo
        if span <= 0:
            base = max(abs(lo), 1.0)
            pad = 0.05 * base
        else:
            pad = fraction * span
        if min_pad is not None:
            pad = max(pad, float(min_pad))
        return [lo - pad, hi + pad]

    @staticmethod
    def _cycle_mean_trace(e_spec, scan_spec, values):
        """Mean of repeated cycles while preserving the complete forward/reverse path."""
        e_spec=np.asarray(e_spec,float); scan_spec=np.asarray(scan_spec,int); values=np.asarray(values,float)
        cs=[int(c) for c in np.unique(scan_spec)]
        lengths=[np.sum(scan_spec==c) for c in cs]
        if not lengths: return np.array([]),np.array([])
        n=max(3,int(round(np.mean(lengths)))); u=np.linspace(0,1,n)
        ee=[]; yy=[]
        for c in cs:
            m=(scan_spec==c)&np.isfinite(e_spec)&np.isfinite(values)
            ec=e_spec[m]; yc=values[m]
            if len(ec)<2: continue
            uc=np.linspace(0,1,len(ec))
            ee.append(np.interp(u,uc,ec)); yy.append(np.interp(u,uc,yc))
        if not yy:return np.array([]),np.array([])
        return np.nanmean(np.vstack(ee),axis=0),np.nanmean(np.vstack(yy),axis=0)

    def _save_static_png(self, path: Path):
        cv_currents = self._smoothed_cv()
        spectral = self._smoothed_spectra_for_preview()
        t, e_spec, scan_spec = self._spectral_potential_mapping()
        e_lo = min(float(self.e_start.value()), float(self.e_switch.value()))
        e_hi = max(float(self.e_start.value()), float(self.e_switch.value()))
        mask_cv = np.isfinite(self.cv_potential) & (self.cv_potential >= e_lo) & (self.cv_potential <= e_hi)

        fig = Figure(figsize=(12, 10), dpi=150)
        gs = fig.add_gridspec(2, 1, height_ratios=[0.30, 0.70])
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = ax1.twinx()
        ax3 = fig.add_subplot(gs[1, 0], projection="3d")

        dash_mpl = ["-", "--", ":", "-."]
        for j in range(cv_currents.shape[1]):
            y = cv_currents[:, j] * 1e6
            m = mask_cv & np.isfinite(y)
            ax1.plot(self.cv_potential[m], y[m], color="black", linewidth=2.0,
                     linestyle=dash_mpl[j % len(dash_mpl)], label=f"CV scan {j+1}")

        if self.show_cycle_mean.isChecked() and cv_currents.shape[1] > 1:
            ym=np.nanmean(cv_currents,axis=1)*1e6
            mm=mask_cv & np.isfinite(ym)
            ax1.plot(self.cv_potential[mm],ym[mm],color="black",linewidth=3.4,alpha=0.72,label="Mean CV")

        colors = ["#7C3AED", "#F97316", "#0EA5E9", "#16A34A", "#DC2626", "#A16207", "#DB2777", "#475569"]
        unit = "cm⁻¹" if self.spectral_mode == "Raman" else "nm"
        for k, chosen in enumerate(self.selected_wavelengths):
            idx = int(np.nanargmin(np.abs(self.wavelengths - chosen)))
            actual = float(self.wavelengths[idx])
            signal = spectral[:, idx]
            for s in range(1, int(self.cycles.value()) + 1):
                m = scan_spec == s
                ax2.plot(e_spec[m], signal[m], color=colors[k % len(colors)], linewidth=1.7,
                         linestyle=dash_mpl[(s-1) % len(dash_mpl)],
                         label=f"{actual:.1f} {unit} – scan {s}")
            if self.show_cycle_mean.isChecked() and int(self.cycles.value()) > 1:
                em,ym=self._cycle_mean_trace(e_spec,scan_spec,signal)
                if len(em):
                    ax2.plot(em,ym,color=colors[k % len(colors)],linewidth=3.2,alpha=0.78,
                             label=f"{actual:.1f} {unit} – mean")

        if self.spectral_mode=="Raman":
            for q,(rlo,rhi) in enumerate(self.selected_ranges):
                signal=self._integrated_range_signal(spectral,rlo,rhi)
                color=colors[(len(self.selected_wavelengths)+q)%len(colors)]
                for sc in range(1,int(self.cycles.value())+1):
                    m=scan_spec==sc
                    ax2.plot(e_spec[m],signal[m],color=color,linewidth=1.7,
                             linestyle=dash_mpl[(sc-1)%len(dash_mpl)],
                             label=f"{rlo:.1f}–{rhi:.1f} cm⁻¹ area – scan {sc}")
                if self.show_cycle_mean.isChecked() and int(self.cycles.value())>1:
                    em,ym=self._cycle_mean_trace(e_spec,scan_spec,signal)
                    if len(em): ax2.plot(em,ym,color=color,linewidth=3.2,alpha=.78,
                                        label=f"{rlo:.1f}–{rhi:.1f} cm⁻¹ area – mean")

        # Tight, data-driven axes. Keep the potential interval close to the entered CV range.
        e_span = max(e_hi - e_lo, 1e-9)
        e_pad = 0.025 * e_span
        ax1.set_xlim(e_lo - e_pad, e_hi + e_pad)
        current_values = []
        for j in range(cv_currents.shape[1]):
            y = cv_currents[:, j] * 1e6
            m = mask_cv & np.isfinite(y)
            current_values.extend(y[m].tolist())
        current_range = self._padded_range(current_values, fraction=0.06)
        if current_range:
            ax1.set_ylim(*current_range)
        signal_values = []
        for chosen in self.selected_wavelengths:
            idx = int(np.nanargmin(np.abs(self.wavelengths - chosen)))
            signal_values.extend(spectral[:, idx][np.isfinite(spectral[:, idx])].tolist())
        signal_range = self._padded_range(signal_values, fraction=0.06)
        if signal_range:
            ax2.set_ylim(*signal_range)
        ax1.set_xlabel("Potential / V")
        ax1.set_ylabel("Current / µA")
        ax2.set_ylabel(self._spectral_y_label())
        ax1.grid(True, alpha=0.2)
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, 1.17),
                   ncol=min(4, max(1, len(l1 + l2))), fontsize=7, frameon=False)

        x_min = float(self.wl_min.value())
        x_max = float(self.wl_max.value())
        lo, hi = min(x_min, x_max), max(x_min, x_max)
        wmask = np.isfinite(self.wavelengths) & (self.wavelengths >= lo) & (self.wavelengths <= hi)
        xaxis = self.wavelengths[wmask]
        spec_show = spectral[:, wmask]

        waterfall_path, waterfall_ticks, waterfall_ticktext, waterfall_cycle_width = \
            self._waterfall_cycle_path(e_spec, scan_spec)

        for i in range(spec_show.shape[0]):
            ax3.plot(np.full_like(xaxis, waterfall_path[i], dtype=float), xaxis, spec_show[i], linewidth=0.8)

        ax3.set_xlim(0, int(self.cycles.value()) * waterfall_cycle_width)
        ax3.set_xticks(waterfall_ticks)
        ax3.set_xticklabels(waterfall_ticktext, fontsize=7)
        ax3.set_xlabel("Potential / V  (Cycle 1 → Cycle 2 → ...)")
        ax3.set_ylabel(self._spectral_x_label())
        ax3.set_zlabel(self._spectral_y_label())
        ax3.view_init(elev=27, azim=-58)

        fig.subplots_adjust(hspace=0.35, top=0.92, bottom=0.06, left=0.08, right=0.92)
        fig.savefig(path, dpi=200, bbox_inches="tight")

    def _build_interactive_figure(self):
        cv_currents = self._smoothed_cv()
        spectral = self._smoothed_spectra_for_preview()
        t, e_spec, scan_spec = self._spectral_potential_mapping()

        # Restrict both displays to exactly the entered CV potential interval.
        e_lo = min(float(self.e_start.value()), float(self.e_switch.value()))
        e_hi = max(float(self.e_start.value()), float(self.e_switch.value()))
        mask_cv = np.isfinite(self.cv_potential) & (self.cv_potential >= e_lo) & (self.cv_potential <= e_hi)

        fig = make_subplots(
            rows=2, cols=1,
            specs=[[{"type": "xy", "secondary_y": True}], [{"type": "scene"}]],
            row_heights=[0.30, 0.70], vertical_spacing=0.07
        )

        # CV: always black; scans distinguished by line style.
        dash_styles = ["solid", "dash", "dot", "dashdot"]
        n_cv = cv_currents.shape[1]
        for j in range(n_cv):
            y = cv_currents[:, j] * 1e6
            m = mask_cv & np.isfinite(y)
            fig.add_trace(go.Scatter(
                x=self.cv_potential[m], y=y[m], mode="lines", name=f"CV scan {j+1}",
                line=dict(color="black", width=3.0, dash=dash_styles[j % len(dash_styles)]),
                hovertemplate=f"CV scan {j+1}<br>Potential: %{{x:.3f}} V<br>Current: %{{y:.2f}} µA<extra></extra>"
            ), row=1, col=1, secondary_y=False)

        if self.show_cycle_mean.isChecked() and n_cv > 1:
            ym=np.nanmean(cv_currents,axis=1)*1e6
            mm=mask_cv & np.isfinite(ym)
            fig.add_trace(go.Scatter(x=self.cv_potential[mm],y=ym[mm],mode="lines",name="Mean CV",
                line=dict(color="black",width=5.0,dash="solid"),opacity=0.72,
                hovertemplate="Mean CV<br>Potential: %{x:.3f} V<br>Current: %{y:.2f} µA<extra></extra>"),
                row=1,col=1,secondary_y=False)

        # One fixed color per selected wavelength; scan identity by dash style.
        colors = ["#7C3AED", "#F97316", "#0EA5E9", "#16A34A", "#DC2626", "#A16207", "#DB2777", "#475569"]
        unit = "cm⁻¹" if self.spectral_mode == "Raman" else "nm"
        for k, chosen in enumerate(self.selected_wavelengths):
            idx = int(np.nanargmin(np.abs(self.wavelengths - chosen)))
            actual = float(self.wavelengths[idx])
            signal = spectral[:, idx]
            color = colors[k % len(colors)]
            for s in range(1, int(self.cycles.value()) + 1):
                m = scan_spec == s
                fig.add_trace(go.Scatter(
                    x=e_spec[m], y=signal[m], mode="lines",
                    name=f"{actual:.1f} {unit} – scan {s}",
                    line=dict(color=color, width=2.5, dash=dash_styles[(s-1) % len(dash_styles)]),
                    hovertemplate=(f"{actual:.1f} {unit}, scan {s}<br>"
                                   "Potential: %{x:.3f} V<br>Signal: %{y:.4f}<extra></extra>")
                ), row=1, col=1, secondary_y=True)
            if self.show_cycle_mean.isChecked() and int(self.cycles.value()) > 1:
                em,ym=self._cycle_mean_trace(e_spec,scan_spec,signal)
                if len(em):
                    fig.add_trace(go.Scatter(x=em,y=ym,mode="lines",
                        name=f"{actual:.1f} {unit} – mean",
                        line=dict(color=color,width=4.5,dash="solid"),opacity=0.78,
                        hovertemplate=(f"{actual:.1f} {unit}, mean<br>Potential: %{{x:.3f}} V<br>"
                                       "Mean signal: %{y:.4f}<extra></extra>")),
                        row=1,col=1,secondary_y=True)

        if self.spectral_mode=="Raman":
            for q,(rlo,rhi) in enumerate(self.selected_ranges):
                signal=self._integrated_range_signal(spectral,rlo,rhi)
                color=colors[(len(self.selected_wavelengths)+q)%len(colors)]
                for sc in range(1,int(self.cycles.value())+1):
                    m=scan_spec==sc
                    fig.add_trace(go.Scatter(x=e_spec[m],y=signal[m],mode="lines",
                        name=f"{rlo:.1f}–{rhi:.1f} cm⁻¹ area – scan {sc}",
                        line=dict(color=color,width=2.5,dash=dash_styles[(sc-1)%len(dash_styles)])),
                        row=1,col=1,secondary_y=True)
                if self.show_cycle_mean.isChecked() and int(self.cycles.value())>1:
                    em,ym=self._cycle_mean_trace(e_spec,scan_spec,signal)
                    if len(em):
                        fig.add_trace(go.Scatter(x=em,y=ym,mode="lines",
                            name=f"{rlo:.1f}–{rhi:.1f} cm⁻¹ area – mean",
                            line=dict(color=color,width=4.5,dash="solid"),opacity=.78),
                            row=1,col=1,secondary_y=True)

        # Spectral waterfall. The CV cycles are unfolded consecutively so that
        # cycle 1, cycle 2, ... are all visible instead of lying on top of one another.
        x_min = float(self.wl_min.value())
        x_max = float(self.wl_max.value())
        lo, hi = min(x_min, x_max), max(x_min, x_max)
        wmask = np.isfinite(self.wavelengths) & (self.wavelengths >= lo) & (self.wavelengths <= hi)
        xaxis = self.wavelengths[wmask]
        spec_show = spectral[:, wmask]

        waterfall_path, waterfall_ticks, waterfall_ticktext, waterfall_cycle_width = \
            self._waterfall_cycle_path(e_spec, scan_spec)

        for i in range(spec_show.shape[0]):
            fig.add_trace(go.Scatter3d(
                x=np.full_like(xaxis, waterfall_path[i], dtype=float),
                y=xaxis, z=spec_show[i], mode="lines", line=dict(width=4), showlegend=False,
                hovertemplate=(f"Cycle {scan_spec[i]}<br>Time: {t[i]:.2f} s<br>"
                               f"Potential: {e_spec[i]:+.3f} V<br>"
                               f"{self._spectral_x_label()}: %{{y:.1f}}<br>"
                               f"{self._spectral_y_label()}: %{{z:.4f}}<extra></extra>")
            ), row=2, col=1)

        # Force compact axis ranges instead of Plotly's broad autorange.
        e_span = max(e_hi - e_lo, 1e-9)
        e_pad = 0.025 * e_span
        x_range = [e_lo - e_pad, e_hi + e_pad]

        current_values = []
        for j in range(n_cv):
            y = cv_currents[:, j] * 1e6
            m = mask_cv & np.isfinite(y)
            current_values.extend(y[m].tolist())
        current_range = self._padded_range(current_values, fraction=0.06)

        signal_values = []
        for chosen in self.selected_wavelengths:
            idx = int(np.nanargmin(np.abs(self.wavelengths - chosen)))
            sig = spectral[:, idx]
            signal_values.extend(sig[np.isfinite(sig)].tolist())
        signal_range = self._padded_range(signal_values, fraction=0.06)

        fig.update_xaxes(
            title_text="Potential / V", range=x_range, autorange=False, nticks=9, row=1, col=1
        )
        fig.update_yaxes(
            title_text="Current / µA", range=current_range, autorange=False if current_range else True,
            nticks=9, row=1, col=1, secondary_y=False
        )
        fig.update_yaxes(
            title_text=self._spectral_y_label(), range=signal_range, autorange=False if signal_range else True,
            nticks=9, row=1, col=1, secondary_y=True
        )
        fig.update_layout(
            height=1100, margin=dict(l=70, r=90, t=45, b=30),
            scene=dict(
                xaxis=dict(
                    title="Potential / V  (Cycle 1 → Cycle 2 → ...)",
                    range=[0, int(self.cycles.value()) * waterfall_cycle_width],
                    tickmode="array",
                    tickvals=waterfall_ticks,
                    ticktext=waterfall_ticktext,
                ),
                yaxis=dict(
                    title=self._spectral_x_label(),
                    range=[lo, hi],
                    autorange=False,
                ),
                zaxis=dict(title=self._spectral_y_label()),
                camera=dict(eye=dict(x=1.5, y=-1.5, z=1.15)),
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0.0)
        )
        return fig


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    dlg = SpectralModeDialog()
    if dlg.exec() != QDialog.Accepted:
        return
    win = MainWindow(dlg.selected_mode())
    win.show()
    if QApplication.instance() is app:
        app.exec()


if __name__ == "__main__":
    main()

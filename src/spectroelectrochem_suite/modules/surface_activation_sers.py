
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from PySide6.QtCore import Qt, QEvent
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout, QGridLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox,
    QPushButton, QScrollArea, QSpinBox, QDoubleSpinBox, QTabWidget, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.ticker import AutoMinorLocator

try:
    from scipy.signal import savgol_filter, find_peaks
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
except Exception:
    savgol_filter = None
    find_peaks = None
    sparse = None
    spsolve = None

MODULE_TITLE = "Electrochemical Surface Activation & SERS Analysis"
MODULE_SUBTITLE = "Au/Ag Surface Characterization | Raman Enhancement | Activation Analysis"


# ----------------------------
# File loading / preprocessing
# ----------------------------

def _detect_delimiter(path: str | Path) -> str:
    text = Path(path).read_text(encoding="utf-8-sig", errors="replace")[:8192]
    try:
        return csv.Sniffer().sniff(text, delimiters=";,\t,").delimiter
    except Exception:
        counts = {d: text.count(d) for d in (";", "\t", ",")}
        return max(counts, key=counts.get)


def _float(v):
    if v is None:
        return np.nan
    if isinstance(v, (int, float, np.number)):
        return float(v)
    s = str(v).strip().replace("\u2212", "-")
    if not s:
        return np.nan
    if "," in s and "." not in s:
        s = s.replace(",", ".")
    try:
        return float(s)
    except Exception:
        return np.nan


def load_numeric_table(path: str | Path) -> pd.DataFrame:
    """Robust numeric table reader for CSV/TXT with optional headers."""
    path = Path(path)
    sep = _detect_delimiter(path)
    raw = pd.read_csv(path, sep=sep, header=None, dtype=str, engine="python")
    num = raw.map(_float)
    # remove empty rows/columns
    num = num.loc[num.notna().any(axis=1), num.notna().any(axis=0)]
    if num.empty:
        raise ValueError("No numerical data found.")
    return num.reset_index(drop=True)


def load_raman(path: str | Path):
    """
    Raman format:
      col 1 = Raman shift / cm^-1
      col 2..n = intensity replicate(s)
    A non-numeric header row is tolerated.
    """
    df = load_numeric_table(path)
    if df.shape[1] < 2:
        raise ValueError("Raman file needs Raman shift in column 1 and at least one intensity column.")
    x = df.iloc[:, 0].to_numpy(float)
    y = df.iloc[:, 1:].to_numpy(float)
    good = np.isfinite(x) & np.any(np.isfinite(y), axis=1)
    x, y = x[good], y[good, :]
    if len(x) < 5:
        raise ValueError("Too few usable Raman points.")
    col_good = np.any(np.isfinite(y), axis=0)
    y = y[:, col_good]
    order = np.argsort(x)
    return x[order], y[order, :]


def load_cv(path: str | Path):
    """
    Activation CV format:
      col 1 = potential / V
      col 2..n = current / A (cycles or replicates)
    """
    df = load_numeric_table(path)
    if df.shape[1] < 2:
        raise ValueError("CV file needs potential in column 1 and current in column 2 or later.")
    e = df.iloc[:, 0].to_numpy(float)
    i = df.iloc[:, 1:].to_numpy(float)
    good = np.isfinite(e) & np.any(np.isfinite(i), axis=1)
    e, i = e[good], i[good, :]
    if len(e) < 5:
        raise ValueError("Too few usable CV points.")
    return e, i


def edge_linear_baseline(x, y, edge_fraction=0.08):
    n = len(x)
    k = max(2, int(edge_fraction * n))
    xx = np.r_[x[:k], x[-k:]]
    yy = np.r_[y[:k], y[-k:]]
    p = np.polyfit(xx, yy, 1)
    return np.polyval(p, x)


def poly_baseline(x, y, degree=3, lower_quantile=0.35, iterations=5):
    """Simple robust iterative polynomial baseline."""
    mask = np.isfinite(x) & np.isfinite(y)
    x0, y0 = x[mask], y[mask]
    if len(x0) < degree + 2:
        return np.zeros_like(y)
    keep = np.ones(len(x0), dtype=bool)
    p = np.polyfit(x0, y0, degree)
    for _ in range(iterations):
        p = np.polyfit(x0[keep], y0[keep], degree)
        base = np.polyval(p, x0)
        resid = y0 - base
        cutoff = np.quantile(resid, lower_quantile)
        keep = resid <= cutoff
        if keep.sum() < degree + 2:
            break
    return np.polyval(p, x)


def asls_baseline(y, lam=1e5, p=0.01, niter=10):
    """Asymmetric least-squares baseline (Eilers)."""
    y = np.asarray(y, dtype=float)
    if sparse is None or spsolve is None or len(y) < 3:
        return np.zeros_like(y)
    n = len(y)
    d = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n))
    penalty = lam * (d.T @ d)
    w = np.ones(n)
    z = y.copy()
    for _ in range(max(1, int(niter))):
        W = sparse.spdiags(w, 0, n, n)
        z = spsolve(W + penalty, w * y)
        w = p * (y > z) + (1.0 - p) * (y <= z)
    return np.asarray(z, float)


def arpls_baseline(y, lam=1e5, ratio=1e-6, niter=50):
    """Adaptive reweighted penalized least-squares baseline (arPLS)."""
    y = np.asarray(y, dtype=float)
    if sparse is None or spsolve is None or len(y) < 3:
        return np.zeros_like(y)
    n = len(y)
    d = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n))
    H = lam * (d.T @ d)
    w = np.ones(n)
    z = y.copy()
    for _ in range(max(1, int(niter))):
        W = sparse.spdiags(w, 0, n, n)
        z = spsolve(W + H, w * y)
        resid = y - z
        neg = resid[resid < 0]
        if len(neg) < 2:
            break
        m = float(np.mean(neg))
        sd = float(np.std(neg))
        if sd <= 1e-30:
            break
        expo = np.clip(2.0 * (resid - (2.0 * sd - m)) / sd, -60, 60)
        w_new = 1.0 / (1.0 + np.exp(expo))
        denom = np.linalg.norm(w)
        if denom > 0 and np.linalg.norm(w - w_new) / denom < ratio:
            w = w_new
            break
        w = w_new
    return np.asarray(z, float)


def preprocess_spectrum(x, y, baseline_mode="arPLS", smooth=True, window=11, polyorder=3,
                        baseline_lambda=1e5, asls_p=0.01):
    y = np.asarray(y, float)
    if baseline_mode == "None":
        base = np.zeros_like(y)
    elif baseline_mode == "Edge-linear":
        base = edge_linear_baseline(x, y)
    elif baseline_mode == "Polynomial":
        base = poly_baseline(x, y, degree=3)
    elif baseline_mode == "AsLS":
        base = asls_baseline(y, lam=float(baseline_lambda), p=float(asls_p), niter=12)
    else:  # arPLS recommended for Raman
        base = arpls_baseline(y, lam=float(baseline_lambda), ratio=1e-6, niter=50)
    corr = y - base

    if smooth and savgol_filter is not None and len(corr) >= 7:
        w = max(5, int(window))
        if w % 2 == 0:
            w += 1
        if w >= len(corr):
            w = len(corr) - 1 if len(corr) % 2 == 0 else len(corr)
        if w >= 5 and w > polyorder:
            corr = savgol_filter(corr, w, min(polyorder, w - 2), mode="interp")
    return corr, base


def interpolate_common(x1, y1, x2, y2):
    lo = max(np.nanmin(x1), np.nanmin(x2))
    hi = min(np.nanmax(x1), np.nanmax(x2))
    if hi <= lo:
        raise ValueError("Before/after Raman spectra do not overlap in Raman shift.")
    x = x1[(x1 >= lo) & (x1 <= hi)]
    if len(x) < 5:
        x = np.linspace(lo, hi, 500)
    return x, np.interp(x, x1, y1), np.interp(x, x2, y2)


def nearest_local_peak(x, y, center, half_width=15.0):
    mask = np.isfinite(x) & np.isfinite(y) & (x >= center - half_width) & (x <= center + half_width)
    if mask.sum() < 2:
        return np.nan, np.nan
    xx, yy = x[mask], y[mask]
    k = int(np.nanargmax(yy))
    return float(xx[k]), float(yy[k])


def safe_ratio(a, b):
    if not np.isfinite(a) or not np.isfinite(b) or abs(b) < 1e-30:
        return np.nan
    return float(a / b)


def positive_area(x, y):
    yp = np.clip(np.asarray(y, float), 0, None)
    return float(np.trapezoid(yp, x))


def activation_charge(e, currents, scan_rate):
    """
    Approximate charge passed during an activation CV.

    The potential path may reverse direction. Therefore integration is performed
    segment-wise with |dE| so forward and reverse scans cannot cancel.
    """
    if scan_rate <= 0:
        return np.nan, np.nan, np.nan
    e = np.asarray(e, float)
    mean_i = np.nanmean(currents, axis=1)
    if len(e) < 2:
        return np.nan, np.nan, np.nan
    de = np.abs(np.diff(e))
    i_mid = 0.5 * (mean_i[:-1] + mean_i[1:])
    q_abs = float(np.nansum(np.abs(i_mid) * de) / scan_rate)
    q_an = float(np.nansum(np.clip(i_mid, 0, None) * de) / scan_rate)
    q_cat = float(np.nansum(np.clip(-i_mid, 0, None) * de) / scan_rate)
    return q_abs, q_an, q_cat


# ----------------------------
# Core scientific analysis
# ----------------------------

def analyze_surface(before, after, settings, target_peaks=None):
    xb, yb = before
    xa, ya = after

    yb_mean_raw = np.nanmean(yb, axis=1)
    ya_mean_raw = np.nanmean(ya, axis=1)

    yb_corr, bb = preprocess_spectrum(
        xb, yb_mean_raw, settings["baseline"], settings["smooth"],
        settings["sg_window"], settings["sg_poly"], settings["baseline_lambda"], settings["asls_p"]
    )
    ya_corr, ba = preprocess_spectrum(
        xa, ya_mean_raw, settings["baseline"], settings["smooth"],
        settings["sg_window"], settings["sg_poly"], settings["baseline_lambda"], settings["asls_p"]
    )

    x, yb_i, ya_i = interpolate_common(xb, yb_corr, xa, ya_corr)
    diff = ya_i - yb_i

    area_before = positive_area(x, yb_i)
    area_after = positive_area(x, ya_i)
    global_enh = safe_ratio(area_after, area_before)

    if target_peaks is None or len(target_peaks) == 0:
        # Auto-detect from activated spectrum.
        if find_peaks is not None and len(x) > 10:
            height = np.nanmin(ya_i) + 0.12 * (np.nanmax(ya_i) - np.nanmin(ya_i))
            prom = 0.05 * max(np.nanmax(ya_i) - np.nanmin(ya_i), 1e-30)
            idx, props = find_peaks(ya_i, height=height, prominence=prom, distance=max(2, len(x)//100))
            if len(idx):
                order = np.argsort(ya_i[idx])[::-1][:8]
                target_peaks = sorted([float(x[idx[k]]) for k in order])
            else:
                target_peaks = []
        else:
            target_peaks = []

    peak_rows = []
    for center in target_peaks:
        pb, ib = nearest_local_peak(x, yb_i, center, settings["peak_half_width"])
        pa, ia = nearest_local_peak(x, ya_i, center, settings["peak_half_width"])
        peak_rows.append({
            "Target / cm-1": float(center),
            "Before peak / cm-1": pb,
            "After peak / cm-1": pa,
            "Shift Δν / cm-1": pa - pb if np.isfinite(pa) and np.isfinite(pb) else np.nan,
            "Intensity before": ib,
            "Intensity after": ia,
            "Apparent enhancement I_after/I_before": safe_ratio(ia, ib),
        })

    peak_df = pd.DataFrame(peak_rows)
    summary = {
        "Integrated positive Raman area before": area_before,
        "Integrated positive Raman area after": area_after,
        "Global apparent Raman enhancement": global_enh,
        "Max intensity before": float(np.nanmax(yb_i)),
        "Max intensity after": float(np.nanmax(ya_i)),
        "Max-intensity ratio": safe_ratio(float(np.nanmax(ya_i)), float(np.nanmax(yb_i))),
    }
    processed = {
        "x": x, "before": yb_i, "after": ya_i, "difference": diff,
        "x_before_raw": xb, "before_raw": yb_mean_raw, "baseline_before": bb,
        "x_after_raw": xa, "after_raw": ya_mean_raw, "baseline_after": ba,
    }
    return summary, peak_df, processed


# ----------------------------
# Matplotlib canvas
# ----------------------------

class PlotCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(10.8, 7.2), constrained_layout=True)
        super().__init__(self.fig)


    def _style_axes(self, axes):
        """Use clearer plot colors and more detailed x-axis graduations."""
        if not isinstance(axes, (list, tuple, np.ndarray)):
            axes = [axes]
        for ax in np.asarray(axes, dtype=object).ravel():
            if ax is None:
                continue
            ax.set_facecolor("#fbfdff")
            ax.grid(True, which="major", alpha=0.28, linewidth=0.7)
            ax.grid(True, which="minor", alpha=0.12, linewidth=0.5)
            ax.xaxis.set_minor_locator(AutoMinorLocator(2))
            ax.tick_params(axis="x", which="major", length=6, width=1.0, labelsize=9)
            ax.tick_params(axis="x", which="minor", length=3, width=0.8)
            ax.tick_params(axis="y", which="major", labelsize=9)

    def overview(self, results, cv_data=None, raman_range=None):
        """Combined overview: Raman, difference and CV for Au/Ag.

        The canvas is intentionally tall so the enclosing QScrollArea can be
        scrolled vertically without squeezing the individual plots.
        """
        cv_data = cv_data or {}
        self.fig.clear()
        axes = self.fig.subplots(3, 2)
        metals = ["Au", "Ag"]
        for col, metal in enumerate(metals):
            ax = axes[0, col]
            d = results.get(metal)
            if d:
                p = d["processed"]
                ax.plot(p["x"], p["before"], label="Before activation", linewidth=1.8, color="#1565C0")
                ax.plot(p["x"], p["after"], label="After activation", linewidth=1.8, color="#F57C00")
                ax.set_title(f"{metal}: Raman before / after")
                ax.set_xlabel("Raman shift / cm$^{-1}$")
                ax.set_ylabel("Baseline-corrected intensity / a.u.")
                if raman_range is not None:
                    lo, hi = raman_range
                    if hi > lo:
                        ax.set_xlim(lo, hi)
                ax.legend()
                ax.grid(alpha=0.2)

                ax2 = axes[1, col]
                ax2.axhline(0, linewidth=0.8)
                ax2.plot(p["x"], p["difference"], linewidth=1.7, color="#8E24AA")
                ax2.set_title(f"{metal}: Difference (after − before)")
                ax2.set_xlabel("Raman shift / cm$^{-1}$")
                ax2.set_ylabel("Δ intensity / a.u.")
                if raman_range is not None:
                    lo, hi = raman_range
                    if hi > lo:
                        ax2.set_xlim(lo, hi)
                ax2.grid(alpha=0.2)
            else:
                ax.text(0.5, 0.5, f"No {metal} Raman pair loaded", ha="center", va="center")
                axes[1, col].text(0.5, 0.5, "No Raman difference available", ha="center", va="center")

            ax3 = axes[2, col]
            shown = False
            for state, linestyle in [("before", "-"), ("after", "--")]:
                data = cv_data.get(f"{metal}_{state}")
                if data is None:
                    continue
                e, currents = data
                mean_i = np.nanmean(currents, axis=1)
                for j in range(currents.shape[1]):
                    ax3.plot(e, currents[:, j], linestyle=linestyle, linewidth=0.75, alpha=0.24)
                ax3.plot(e, mean_i, linestyle=linestyle, linewidth=2.0, label=state.capitalize())
                shown = True
            if shown:
                ax3.set_title(f"{metal}: CV before / after activation")
                ax3.set_xlabel("Potential / V")
                ax3.set_ylabel("Current / A")
                ax3.grid(alpha=0.2)
                ax3.legend()
            else:
                ax3.text(0.5, 0.5, f"No {metal} CVs loaded", ha="center", va="center")

        self._style_axes(self.figure.axes)
        self.draw_idle()

    def baseline_control(self, results, raman_range=None):
        self.fig.clear()
        axes = self.fig.subplots(2, 2)
        for row, metal in enumerate(("Au", "Ag")):
            d = results.get(metal)
            for col, state in enumerate(("before", "after")):
                ax = axes[row, col]
                if not d:
                    ax.text(0.5, 0.5, f"No {metal} data", ha="center", va="center")
                    continue
                p = d["processed"]
                if state == "before":
                    x = p["x_before_raw"]
                    raw = p["before_raw"]
                    baseline = p["baseline_before"]
                else:
                    x = p["x_after_raw"]
                    raw = p["after_raw"]
                    baseline = p["baseline_after"]
                ax.plot(x, raw, linewidth=1.2, label="Raw Raman")
                ax.plot(x, baseline, linewidth=1.8, label="Calculated baseline")
                ax.set_title(f"{metal} {state}: baseline control")
                ax.set_xlabel("Raman shift / cm$^{-1}$")
                ax.set_ylabel("Intensity / a.u.")
                if raman_range is not None:
                    lo, hi = raman_range
                    if hi > lo:
                        ax.set_xlim(lo, hi)
                ax.grid(alpha=0.2)
                ax.legend()
        self._style_axes(self.figure.axes)
        self.draw_idle()

    def peak_plot(self, results, raman_range=None):
        self.fig.clear()
        ax = self.fig.subplots(1, 1)
        shown = False
        for metal, marker in [("Au", "o"), ("Ag", "s")]:
            d = results.get(metal)
            if d is None or d["peaks"].empty:
                continue
            df = d["peaks"]
            good = np.isfinite(df["Target / cm-1"]) & np.isfinite(df["Apparent enhancement I_after/I_before"])
            if good.any():
                ax.plot(df.loc[good, "Target / cm-1"],
                        df.loc[good, "Apparent enhancement I_after/I_before"],
                        marker=marker, linewidth=1.4, label=metal)
                shown = True
        ax.set_xlabel("Raman shift / cm$^{-1}$")
        ax.set_ylabel("Apparent enhancement $I_{after}/I_{before}$")
        ax.set_title("Peak-specific Raman enhancement")
        if raman_range is not None:
            lo, hi = raman_range
            if hi > lo:
                ax.set_xlim(lo, hi)
        ax.grid(alpha=0.2)
        if shown:
            ax.legend()
        self._style_axes(self.figure.axes)
        self.draw_idle()

    def cv_plot(self, cv_data, scan_rates):
        self.fig.clear()
        axes = self.fig.subplots(1, 2)
        for ax, metal in zip(axes, ["Au", "Ag"]):
            shown = False
            for state, linestyle in [("before", "-"), ("after", "--")]:
                data = cv_data.get(f"{metal}_{state}")
                if data is None:
                    continue
                e, currents = data
                mean_i = np.nanmean(currents, axis=1)
                for j in range(currents.shape[1]):
                    ax.plot(e, currents[:, j], linestyle=linestyle, linewidth=0.8, alpha=0.28)
                ax.plot(e, mean_i, linestyle=linestyle, linewidth=2.2, label=state.capitalize())
                shown = True
            if shown:
                ax.set_title(f"{metal}: CV before / after activation")
                ax.set_xlabel("Potential / V")
                ax.set_ylabel("Current / A")
                ax.grid(alpha=0.2)
                ax.legend()
            else:
                ax.text(0.5, 0.5, f"No {metal} CVs loaded", ha="center", va="center")
        self._style_axes(self.figure.axes)
        self.draw_idle()


# ----------------------------
# Main window
# ----------------------------

class SurfaceActivationWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(MODULE_TITLE)
        self.resize(1480, 900)
        self._wheel_scroll_targets = {}
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #f7f9fc; }
            QGroupBox {
                font-weight: 600;
                border: 1px solid #b8c7d9;
                border-radius: 7px;
                margin-top: 9px;
                padding-top: 8px;
                background: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 1px 7px;
                color: #244a73;
                background: #e8f1fb;
                border-radius: 4px;
            }
            QPushButton {
                background: #eaf3fb;
                border: 1px solid #9ebbd5;
                border-radius: 5px;
                padding: 5px 9px;
            }
            QPushButton:hover { background: #d8eafb; }
            QPushButton:pressed { background: #c3def5; }
            QTabWidget::pane { border: 1px solid #b8c7d9; background: white; }
            QTabBar::tab {
                background: #e9eef5;
                padding: 7px 14px;
                margin-right: 2px;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
            }
            QTabBar::tab:selected { background: #cfe3f6; color: #183f67; font-weight: 600; }
            QDoubleSpinBox, QSpinBox, QComboBox, QLineEdit {
                background: white; border: 1px solid #b7c3d0; border-radius: 4px; padding: 3px;
            }
            QTableWidget { background: white; gridline-color: #d5dee8; }
        """)

        self.raman_paths = {
            "Au_before": None, "Au_after": None,
            "Ag_before": None, "Ag_after": None,
        }
        self.cv_paths = {
            "Au_before": None, "Au_after": None,
            "Ag_before": None, "Ag_after": None,
        }
        self.results = {}
        self.cv_data = {}

        root = QWidget()
        root_layout = QHBoxLayout(root)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.controls_scroll = scroll
        controls = QWidget()
        self.controls_layout = QVBoxLayout(controls)
        scroll.setWidget(controls)
        scroll.setMinimumWidth(410)
        scroll.setMaximumWidth(500)

        root_layout.addWidget(scroll)
        self.tabs = QTabWidget()
        root_layout.addWidget(self.tabs, 1)
        self.setCentralWidget(root)

        self._build_controls()
        # Allow the mouse wheel to scroll the complete left control panel even
        # when the pointer is over spin boxes, combo boxes, line edits or buttons.
        controls.installEventFilter(self)
        for child in controls.findChildren(QWidget):
            child.installEventFilter(self)
        self._build_tabs()

    def _build_controls(self):
        title = QLabel(f"<div style='padding:9px;background:#dcecf9;border-radius:7px;'><b style='color:#173f67;font-size:15px;'>{MODULE_TITLE}</b><br><span style='color:#4d6478'>{MODULE_SUBTITLE}</span></div>")
        title.setWordWrap(True)
        self.controls_layout.addWidget(title)

        range_box = QGroupBox("1. Raman axis and display range")
        range_form = QFormLayout(range_box)
        self.raman_axis_input = QComboBox()
        self.raman_axis_input.addItems(["Raman shift (cm⁻¹)", "Wavelength (nm)"])
        self.raman_axis_input.setCurrentText("Raman shift (cm⁻¹)")
        range_form.addRow("Input x-axis:", self.raman_axis_input)
        self.raman_min = QDoubleSpinBox()
        self.raman_min.setRange(-1000.0, 10000.0)
        self.raman_min.setDecimals(1)
        self.raman_min.setValue(0.0)
        self.raman_min.setSuffix(" cm⁻¹")
        self.raman_max = QDoubleSpinBox()
        self.raman_max.setRange(-1000.0, 10000.0)
        self.raman_max.setDecimals(1)
        self.raman_max.setValue(4000.0)
        self.raman_max.setSuffix(" cm⁻¹")
        range_form.addRow("From:", self.raman_min)
        range_form.addRow("To:", self.raman_max)
        self.controls_layout.addWidget(range_box)

        files_box = QGroupBox("2. Raman spectra")
        grid = QGridLayout(files_box)
        self.file_labels = {}
        row = 0
        for metal in ("Au", "Ag"):
            for state in ("before", "after"):
                key = f"{metal}_{state}"
                lab = QLabel(f"{metal} {state}")
                btn = QPushButton("Load CSV/TXT")
                name = QLabel("not loaded")
                name.setWordWrap(True)
                btn.clicked.connect(lambda _=False, k=key: self._load_raman(k))
                grid.addWidget(lab, row, 0)
                grid.addWidget(btn, row, 1)
                grid.addWidget(name, row, 2)
                self.file_labels[key] = name
                row += 1
        self.controls_layout.addWidget(files_box)

        cv_box = QGroupBox("3. CV before / after activation")
        cvgrid = QGridLayout(cv_box)
        self.cv_labels = {}
        self.scan_rate = {}
        row = 0
        for metal in ("Au", "Ag"):
            for state in ("before", "after"):
                key = f"{metal}_{state}"
                btn = QPushButton(f"Load {metal} {state} CV")
                lab = QLabel("not loaded")
                sr = QDoubleSpinBox()
                sr.setRange(0.000001, 100.0)
                sr.setDecimals(6)
                sr.setValue(0.050)
                sr.setSuffix(" V/s")
                btn.clicked.connect(lambda _=False, k=key: self._load_cv(k))
                cvgrid.addWidget(btn, row, 0)
                cvgrid.addWidget(lab, row, 1)
                cvgrid.addWidget(sr, row, 2)
                self.cv_labels[key] = lab
                self.scan_rate[key] = sr
                row += 1
        self.controls_layout.addWidget(cv_box)

        proc = QGroupBox("4. Raman processing")
        form = QFormLayout(proc)
        self.baseline = QComboBox()
        self.baseline.addItems(["arPLS", "AsLS", "Polynomial", "Edge-linear", "None"])
        self.baseline.setCurrentText("arPLS")
        self.smooth = QCheckBox("Savitzky–Golay smoothing")
        self.smooth.setChecked(True)
        self.sg_window = QSpinBox()
        self.sg_window.setRange(5, 101); self.sg_window.setSingleStep(2); self.sg_window.setValue(11)
        self.sg_poly = QSpinBox()
        self.sg_poly.setRange(2, 5); self.sg_poly.setValue(3)
        self.baseline_lambda = QDoubleSpinBox()
        self.baseline_lambda.setRange(1.0, 1.0e12); self.baseline_lambda.setDecimals(0); self.baseline_lambda.setValue(1000.0)
        self.baseline_lambda.setSingleStep(100.0)
        self.asls_p = QDoubleSpinBox()
        self.asls_p.setRange(0.0001, 0.5); self.asls_p.setDecimals(4); self.asls_p.setValue(0.01)
        self.peak_half_width = QDoubleSpinBox()
        self.peak_half_width.setRange(1.0, 100.0); self.peak_half_width.setValue(15.0); self.peak_half_width.setSuffix(" cm⁻¹")
        self.target_peaks = QLineEdit()
        self.target_peaks.setPlaceholderText("e.g. 1025, 1170, 1315; empty = auto")
        form.addRow("Baseline correction:", self.baseline)
        form.addRow("Baseline λ:", self.baseline_lambda)
        form.addRow("AsLS p:", self.asls_p)
        form.addRow("", self.smooth)
        form.addRow("SG window:", self.sg_window)
        form.addRow("SG polynomial:", self.sg_poly)
        form.addRow("Peak search ±:", self.peak_half_width)
        form.addRow("Target peaks / cm⁻¹:", self.target_peaks)
        self.controls_layout.addWidget(proc)

        meta = QGroupBox("5. Experiment metadata")
        mform = QFormLayout(meta)
        self.probe = QLineEdit("Ru(bpy)3 2+")
        self.electrolyte = QLineEdit("Na2SO4 (aq)")
        self.activation = QLineEdit("KCl (aq), electrochemical activation")
        self.laser = QLineEdit("785")
        self.laser.setToolTip("Required for automatic conversion when Raman x-axis input is Wavelength (nm). Kept as metadata for Raman-shift input.")
        self.conc = QLineEdit("")
        mform.addRow("Raman probe:", self.probe)
        mform.addRow("Measurement electrolyte:", self.electrolyte)
        mform.addRow("Activation:", self.activation)
        mform.addRow("Laser wavelength / nm:", self.laser)
        mform.addRow("Probe concentration:", self.conc)
        self.controls_layout.addWidget(meta)

        buttons = QGroupBox("6. Analysis / export")
        b = QVBoxLayout(buttons)
        run = QPushButton("Analyze Au / Ag surface activation")
        run.setMinimumHeight(42)
        run.clicked.connect(self.analyze)
        exp_xlsx = QPushButton("Export Excel")
        exp_xlsx.clicked.connect(self.export_excel)
        exp_png = QPushButton("Export current plot PNG")
        exp_png.clicked.connect(self.export_png)
        b.addWidget(run); b.addWidget(exp_xlsx); b.addWidget(exp_png)
        self.controls_layout.addWidget(buttons)
        self.controls_layout.addStretch(1)

    def _scrollable_plot_tab(self, canvas, minimum_height=950):
        """Create a plot tab that can also be scrolled with the mouse wheel over the canvas."""
        outer = QWidget()
        layout = QVBoxLayout(outer)
        layout.setContentsMargins(4, 4, 4, 4)
        toolbar = NavigationToolbar(canvas, self)
        # Make active tools (especially Zoom/Pan) visibly selected.
        toolbar.setStyleSheet("""
            QToolButton:checked {
                background: #7faed6;
                border: 2px solid #2f6594;
                border-radius: 4px;
            }
        """)
        reset_zoom = QPushButton("Reset zoom")
        reset_zoom.setToolTip("Restore the original plot view")
        reset_zoom.clicked.connect(toolbar.home)

        layout.addWidget(toolbar)

        reset_zoom.setText("↶  RESET ZOOM")
        reset_zoom.setMinimumSize(170, 42)
        reset_zoom.setMaximumWidth(190)
        reset_zoom.setStyleSheet("""
            QPushButton {
                background: #cfe3f6;
                border: 2px solid #4f86b5;
                border-radius: 6px;
                padding: 6px 14px;
                font-size: 14px;
                font-weight: 700;
                text-align: left;
            }
            QPushButton:hover { background: #b8d7f0; }
            QPushButton:pressed { background: #9fc7e8; }
        """)
        reset_row = QHBoxLayout()
        # Small left offset places RESET ZOOM approximately below the toolbar zoom icon.
        reset_row.addSpacing(135)
        reset_row.addWidget(reset_zoom)
        reset_row.addStretch(1)
        layout.addLayout(reset_row)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        holder = QWidget()
        holder_layout = QVBoxLayout(holder)
        holder_layout.setContentsMargins(0, 0, 0, 0)
        canvas.setMinimumHeight(minimum_height)
        holder.setMinimumHeight(minimum_height)
        holder_layout.addWidget(canvas)
        scroll.setWidget(holder)
        layout.addWidget(scroll, 1)

        # Mouse wheel over the Matplotlib canvas scrolls the enclosing plot area.
        # Mouse clicks and drags are untouched, so toolbar Zoom/Pan still work.
        self._wheel_scroll_targets[canvas] = scroll
        canvas.installEventFilter(self)
        return outer

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Wheel:
            # Left control panel: wheel scrolls the panel instead of changing
            # a spin-box/combo-box value under the mouse pointer.
            controls = getattr(self, "controls_scroll", None)
            if controls is not None and (obj is controls.widget() or controls.widget().isAncestorOf(obj)):
                bar = controls.verticalScrollBar()
                if bar.maximum() > bar.minimum():
                    delta = event.angleDelta().y()
                    step = int(-delta * 0.58)
                    if step == 0 and delta != 0:
                        step = -1 if delta > 0 else 1
                    bar.setValue(bar.value() + step)
                    return True

            # Plot tabs: wheel scrolls vertically while click/drag remains
            # available for Matplotlib Zoom/Pan.
            if obj in self._wheel_scroll_targets:
                scroll = self._wheel_scroll_targets[obj]
                bar = scroll.verticalScrollBar()
                if bar.maximum() > bar.minimum():
                    delta = event.angleDelta().y()
                    step = int(-delta * 0.58)
                    if step == 0 and delta != 0:
                        step = -1 if delta > 0 else 1
                    bar.setValue(bar.value() + step)
                    return True
        return super().eventFilter(obj, event)

    def _build_tabs(self):
        self.overview_canvas = PlotCanvas()
        self.tabs.addTab(self._scrollable_plot_tab(self.overview_canvas, 1250), "Raman + CV overview")

        self.baseline_canvas = PlotCanvas()
        self.tabs.addTab(self._scrollable_plot_tab(self.baseline_canvas, 1000), "Baseline control")

        self.peak_canvas = PlotCanvas()
        self.tabs.addTab(self._scrollable_plot_tab(self.peak_canvas, 850), "Enhancement")

        self.cv_canvas = PlotCanvas()
        self.tabs.addTab(self._scrollable_plot_tab(self.cv_canvas, 850), "Activation CV")

        self.table = QTableWidget()
        self.tabs.addTab(self.table, "Summary")

    def _load_raman(self, key):
        path, _ = QFileDialog.getOpenFileName(self, "Load Raman spectrum", "", "Data files (*.csv *.txt *.tsv);;All files (*.*)")
        if not path:
            return
        try:
            load_raman(path)
        except Exception as exc:
            QMessageBox.critical(self, "Raman file error", str(exc))
            return
        self.raman_paths[key] = path
        self.file_labels[key].setText(Path(path).name)

    def _load_cv(self, key):
        label = key.replace("_", " ")
        path, _ = QFileDialog.getOpenFileName(self, f"Load {label} CV", "", "Data files (*.csv *.txt *.tsv);;All files (*.*)")
        if not path:
            return
        try:
            load_cv(path)
        except Exception as exc:
            QMessageBox.critical(self, "CV file error", str(exc))
            return
        self.cv_paths[key] = path
        self.cv_labels[key].setText(Path(path).name)

    def _convert_raman_axis_if_needed(self, data):
        """Convert spectral wavelength in nm to Stokes Raman shift in cm^-1 when selected."""
        x, y = data
        if self.raman_axis_input.currentText().startswith("Wavelength"):
            try:
                laser_nm = float(self.laser.text().strip().replace(",", "."))
            except Exception:
                raise ValueError("Enter a valid laser wavelength in nm for wavelength-to-Raman-shift conversion.")
            if laser_nm <= 0:
                raise ValueError("Laser wavelength must be greater than 0 nm.")
            x = np.asarray(x, dtype=float)
            if np.any(x <= 0):
                raise ValueError("Spectral wavelengths must be greater than 0 nm.")
            # Stokes Raman shift: 10^7 * (1/lambda_laser - 1/lambda_signal)
            x = 1.0e7 * (1.0 / laser_nm - 1.0 / x)
            order = np.argsort(x)
            return x[order], np.asarray(y)[order, :]
        return data

    def _raman_range(self):
        lo = float(self.raman_min.value())
        hi = float(self.raman_max.value())
        if hi <= lo:
            raise ValueError("Raman range: 'To' must be greater than 'From'.")
        return lo, hi

    def _settings(self):
        return {
            "baseline": self.baseline.currentText(),
            "baseline_lambda": self.baseline_lambda.value(),
            "asls_p": self.asls_p.value(),
            "smooth": self.smooth.isChecked(),
            "sg_window": self.sg_window.value(),
            "sg_poly": self.sg_poly.value(),
            "peak_half_width": self.peak_half_width.value(),
        }

    def _parse_targets(self):
        txt = self.target_peaks.text().strip()
        if not txt:
            return []
        txt = txt.replace(";", ",")
        vals = []
        for part in txt.split(","):
            part = part.strip().replace(",", ".")
            if part:
                vals.append(float(part))
        return vals

    def analyze(self):
        self.results = {}
        self.cv_data = {}
        settings = self._settings()
        try:
            raman_range = self._raman_range()
        except ValueError as exc:
            QMessageBox.warning(self, "Raman range", str(exc))
            return
        targets = self._parse_targets()

        missing_pairs = []
        for metal in ("Au", "Ag"):
            bp = self.raman_paths[f"{metal}_before"]
            ap = self.raman_paths[f"{metal}_after"]
            if bp and ap:
                try:
                    before = self._convert_raman_axis_if_needed(load_raman(bp))
                    after = self._convert_raman_axis_if_needed(load_raman(ap))
                    summary, peaks, processed = analyze_surface(before, after, settings, targets)
                    self.results[metal] = {
                        "summary": summary, "peaks": peaks, "processed": processed,
                        "before_path": bp, "after_path": ap,
                    }
                except Exception as exc:
                    QMessageBox.critical(self, f"{metal} analysis error", str(exc))
                    return
            elif bp or ap:
                missing_pairs.append(metal)

            for state in ("before", "after"):
                cv_key = f"{metal}_{state}"
                if self.cv_paths[cv_key]:
                    try:
                        self.cv_data[cv_key] = load_cv(self.cv_paths[cv_key])
                    except Exception as exc:
                        QMessageBox.warning(self, f"{metal} {state} CV warning", str(exc))

        if not self.results:
            QMessageBox.information(self, "No complete Raman pair", "Load at least one complete before/after pair for Au or Ag.")
            return
        if missing_pairs:
            QMessageBox.warning(self, "Incomplete pair", "A before/after Raman file is missing for: " + ", ".join(missing_pairs))

        self.overview_canvas.overview(self.results, self.cv_data, raman_range)
        self.baseline_canvas.baseline_control(self.results, raman_range)
        self.peak_canvas.peak_plot(self.results, raman_range)
        self.cv_canvas.cv_plot(self.cv_data, {k: self.scan_rate[k].value() for k in self.scan_rate})
        self._fill_summary()

    def _summary_rows(self):
        rows = []
        for metal in ("Au", "Ag"):
            if metal not in self.results:
                continue
            s = self.results[metal]["summary"]
            rows.extend([
                [metal, "Global apparent Raman enhancement", s["Global apparent Raman enhancement"], "dimensionless"],
                [metal, "Integrated Raman area before", s["Integrated positive Raman area before"], "a.u.·cm⁻¹"],
                [metal, "Integrated Raman area after", s["Integrated positive Raman area after"], "a.u.·cm⁻¹"],
                [metal, "Maximum intensity ratio", s["Max-intensity ratio"], "dimensionless"],
            ])
            charges = {}
            for state in ("before", "after"):
                key = f"{metal}_{state}"
                if key in self.cv_data:
                    charges[state] = activation_charge(
                        self.cv_data[key][0], self.cv_data[key][1], self.scan_rate[key].value()
                    )
                    q_abs, q_an, q_cat = charges[state]
                    rows.extend([
                        [metal, f"CV {state}: absolute charge", q_abs, "C"],
                        [metal, f"CV {state}: anodic charge", q_an, "C"],
                        [metal, f"CV {state}: cathodic charge", q_cat, "C"],
                    ])
            if "before" in charges and "after" in charges:
                qb = charges["before"][0]
                qa = charges["after"][0]
                rows.append([metal, "CV charge ratio after/before", safe_ratio(qa, qb), "dimensionless"])
        return rows

    def _fill_summary(self):
        rows = self._summary_rows()
        self.table.setRowCount(len(rows))
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Surface", "Parameter", "Value", "Unit"])
        for r, row in enumerate(rows):
            for c, v in enumerate(row):
                if isinstance(v, float):
                    text = "" if not np.isfinite(v) else f"{v:.6g}"
                else:
                    text = str(v)
                self.table.setItem(r, c, QTableWidgetItem(text))
        self.table.resizeColumnsToContents()

    def export_excel(self):
        if not self.results:
            QMessageBox.information(self, "Nothing to export", "Run the analysis first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export analysis", "Surface_Activation_SERS.xlsx", "Excel (*.xlsx)")
        if not path:
            return
        if not path.lower().endswith(".xlsx"):
            path += ".xlsx"

        meta = pd.DataFrame([
            ["Module", MODULE_TITLE],
            ["Probe", self.probe.text()],
            ["Measurement electrolyte", self.electrolyte.text()],
            ["Activation", self.activation.text()],
            ["Raman input x-axis", self.raman_axis_input.currentText()],
            ["Laser wavelength / nm", self.laser.text()],
            ["Probe concentration", self.conc.text()],
            ["Raman display range from / cm-1", self.raman_min.value()],
            ["Raman display range to / cm-1", self.raman_max.value()],
            ["Baseline", self.baseline.currentText()],
            ["Baseline lambda", self.baseline_lambda.value()],
            ["AsLS p", self.asls_p.value()],
            ["Smoothing", self.smooth.isChecked()],
        ], columns=["Parameter", "Value"])

        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            meta.to_excel(writer, sheet_name="Metadata", index=False)
            pd.DataFrame(self._summary_rows(), columns=["Surface", "Parameter", "Value", "Unit"]).to_excel(
                writer, sheet_name="Summary", index=False
            )
            for metal, d in self.results.items():
                p = d["processed"]
                pd.DataFrame({
                    "Raman shift / cm-1": p["x"],
                    "Before corrected": p["before"],
                    "After corrected": p["after"],
                    "Difference after-before": p["difference"],
                }).to_excel(writer, sheet_name=f"{metal}_Raman", index=False)
                d["peaks"].to_excel(writer, sheet_name=f"{metal}_Peaks", index=False)
                for state in ("before", "after"):
                    key = f"{metal}_{state}"
                    if key in self.cv_data:
                        e, cur = self.cv_data[key]
                        data = {"Potential / V": e}
                        for j in range(cur.shape[1]):
                            data[f"Current {j+1} / A"] = cur[:, j]
                        pd.DataFrame(data).to_excel(writer, sheet_name=f"{metal}_CV_{state}", index=False)

        QMessageBox.information(self, "Export complete", f"Saved:\n{path}")

    def export_png(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export current plot", "Surface_Activation_SERS.png", "PNG (*.png)")
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"
        idx = self.tabs.currentIndex()
        canvas = {
            0: self.overview_canvas,
            1: self.baseline_canvas,
            2: self.peak_canvas,
            3: self.cv_canvas,
        }.get(idx, self.overview_canvas)
        canvas.fig.savefig(path, dpi=300, bbox_inches="tight")
        QMessageBox.information(self, "Export complete", f"Saved:\n{path}")


def start(parent=None):
    window = SurfaceActivationWindow()
    if parent is not None:
        try:
            window.setParent(parent, Qt.Window)
        except Exception:
            pass
    window.show()
    return window


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    w = SurfaceActivationWindow()
    w.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

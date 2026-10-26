from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox, QPushButton, QPlainTextEdit,
    QLayout, QScrollArea, QSplitter, QTabWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

MODULE_TITLE = "Stripping Voltammetry Analysis – SWV | DPV"
METHODS = ("SWV", "DPV")
METHOD_LONG = {
    "SWV": "Square Wave Voltammetry (SWV)",
    "DPV": "Differential Pulse Voltammetry (DPV)",
}


def _detect_delimiter(path: str | Path) -> str:
    text = Path(path).read_text(encoding="utf-8-sig", errors="replace")[:4096]
    try:
        return csv.Sniffer().sniff(text, delimiters=";,\t,").delimiter
    except Exception:
        counts = {d: text.count(d) for d in (";", "\t", ",")}
        return max(counts, key=counts.get)


def _to_float(value):
    if value is None:
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)
    s = str(value).strip().replace("\u2212", "-")
    if not s:
        return np.nan
    if "," in s and "." not in s:
        s = s.replace(",", ".")
    return float(s)


def load_stripping_csv(path: str | Path):
    """Suite stripping format: first column potential, first row concentrations."""
    path = Path(path)
    sep = _detect_delimiter(path)
    raw = pd.read_csv(path, sep=sep, header=None, dtype=str, engine="python")
    if raw.shape[0] < 3 or raw.shape[1] < 2:
        raise ValueError("CSV must contain a concentration header row, a potential column and at least one current column.")

    conc = np.array([_to_float(v) for v in raw.iloc[0, 1:]], dtype=float)
    potential = np.array([_to_float(v) for v in raw.iloc[1:, 0]], dtype=float)
    # pandas 2.1+ compatible; avoids deprecated/removed DataFrame.applymap().
    current = raw.iloc[1:, 1:].map(_to_float).to_numpy(dtype=float)

    valid_rows = np.isfinite(potential) & np.any(np.isfinite(current), axis=1)
    potential = potential[valid_rows]
    current = current[valid_rows, :]

    valid_cols = np.isfinite(conc) & np.any(np.isfinite(current), axis=0)
    conc = conc[valid_cols]
    current = current[:, valid_cols]

    if len(potential) < 3 or len(conc) < 1:
        raise ValueError("No usable numerical stripping data were found.")

    order = np.argsort(potential)
    potential = potential[order]
    current = current[order, :]
    return potential, conc, current, str(raw.iloc[0, 0]), sep


def load_unknown_csv(path: str | Path):
    """Unknown-sample format: first column potential; remaining columns are replicate current traces.

    A first header row is tolerated and ignored for concentration because the concentration is unknown.
    """
    path = Path(path)
    sep = _detect_delimiter(path)
    raw = pd.read_csv(path, sep=sep, header=None, dtype=str, engine="python")
    if raw.shape[0] < 3 or raw.shape[1] < 2:
        raise ValueError("Unknown CSV must contain a potential column and at least one current column.")

    # Detect whether row 0 is a header by testing the first cell as potential.
    try:
        first = _to_float(raw.iloc[0, 0])
    except Exception:
        first = np.nan
    start = 0 if np.isfinite(first) else 1
    potential = np.array([_to_float(v) for v in raw.iloc[start:, 0]], dtype=float)
    currents = raw.iloc[start:, 1:].map(_to_float).to_numpy(dtype=float)
    valid_rows = np.isfinite(potential) & np.any(np.isfinite(currents), axis=1)
    potential = potential[valid_rows]
    currents = currents[valid_rows, :]
    valid_cols = np.any(np.isfinite(currents), axis=0)
    currents = currents[:, valid_cols]
    if len(potential) < 3 or currents.shape[1] < 1:
        raise ValueError("No usable numerical unknown-sample data were found.")
    order = np.argsort(potential)
    return potential[order], currents[order, :], sep


def analyze_unknown(potential, currents, e1, e2, slope, intercept, absolute_area=True):
    if not np.isfinite(slope) or abs(slope) < 1e-30:
        raise ValueError("A valid calibration slope is required before calculating an unknown concentration.")
    rows = []
    areas = []
    for j in range(currents.shape[1]):
        xx, yy, base, corr, signed_area = local_linear_baseline(potential, currents[:, j], e1, e2)
        area = abs(signed_area) if absolute_area else signed_area
        areas.append(area)
        rows.append({
            "Replicate": j + 1,
            "Signed integrated peak area / A V": signed_area,
            "Integrated peak area / A V": area,
        })
    mean_area = float(np.nanmean(areas))
    sd_area = float(np.nanstd(areas, ddof=1)) if len(areas) > 1 else 0.0
    concentration = float((mean_area - intercept) / slope)
    return pd.DataFrame(rows), mean_area, sd_area, concentration


def local_linear_baseline(x, y, x1, x2):
    lo, hi = sorted((float(x1), float(x2)))
    if hi <= lo:
        raise ValueError("E2 must be larger than E1.")
    mask = (x >= lo) & (x <= hi) & np.isfinite(y)
    xx = x[mask]
    yy = y[mask]
    if len(xx) < 2:
        raise ValueError("Integration window contains fewer than two points.")
    y1 = np.interp(lo, x, y)
    y2 = np.interp(hi, x, y)
    baseline = y1 + (y2 - y1) * (xx - lo) / (hi - lo)
    corr = yy - baseline
    area = float(np.trapezoid(corr, xx))
    return xx, yy, baseline, corr, area


def analyze_dataset(potential, concentrations, currents, e1, e2, absolute_area=True, manual_peaks=None):
    """Calculate cursor-defined peak areas and optional manually selected peak currents.

    Important: the integration cursors E1/E2 affect only the integrated peak area.
    Peak potential/current are taken directly from the raw voltammogram at the
    manually clicked potential and are therefore independent of E1/E2. If no
    manual peak has been selected for a concentration, peak potential/current
    remain NaN and are excluded from peak-current calibration.
    """
    rows, areas = [], []
    for j, c in enumerate(concentrations):
        y = currents[:, j]
        _xx, _yy, _base, _corr, signed_area = local_linear_baseline(potential, y, e1, e2)
        manual_x = None if not manual_peaks else manual_peaks.get(j)
        if manual_x is not None and np.isfinite(manual_x):
            peak_potential = float(np.clip(float(manual_x), float(np.nanmin(potential)), float(np.nanmax(potential))))
            peak_current = float(np.interp(peak_potential, potential, y))
        else:
            peak_potential = np.nan
            peak_current = np.nan
        area = abs(signed_area) if absolute_area else signed_area
        areas.append(area)
        rows.append({
            "Concentration": float(c),
            "Peak potential / V": peak_potential,
            "Manual peak current / A": peak_current,
            "Signed integrated peak area / A V": signed_area,
            "Integrated peak area / A V": area,
        })

    x = np.asarray(concentrations, dtype=float)
    y = np.asarray(areas, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() >= 2 and np.ptp(x[valid]) > 0:
        slope, intercept = np.polyfit(x[valid], y[valid], 1)
        pred = slope * x[valid] + intercept
        ss_res = np.sum((y[valid] - pred) ** 2)
        ss_tot = np.sum((y[valid] - np.mean(y[valid])) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    else:
        slope = intercept = r2 = np.nan
    return pd.DataFrame(rows), float(slope), float(intercept), float(r2)


def linear_fit_stats(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]; y = y[valid]
    if len(x) < 2 or np.ptp(x) <= 0:
        return {"slope": np.nan, "intercept": np.nan, "r2": np.nan, "sigma": np.nan, "lod": np.nan, "loq": np.nan}
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope*x + intercept
    resid = y-pred
    ss_res=float(np.sum(resid**2)); ss_tot=float(np.sum((y-np.mean(y))**2))
    r2=1.0-ss_res/ss_tot if ss_tot>0 else 1.0
    # Residual standard deviation (standard error of regression); requires >=3 standards.
    sigma=float(np.sqrt(ss_res/(len(x)-2))) if len(x)>=3 else np.nan
    lod=3.3*sigma/abs(slope) if np.isfinite(sigma) and abs(slope)>1e-30 else np.nan
    loq=10.0*sigma/abs(slope) if np.isfinite(sigma) and abs(slope)>1e-30 else np.nan
    return {"slope":float(slope), "intercept":float(intercept), "r2":float(r2), "sigma":sigma, "lod":float(lod), "loq":float(loq)}


def _initial_window(potential, currents):
    mean_y = np.nanmean(currents, axis=1)
    n = len(potential)
    detr = mean_y - np.linspace(mean_y[0], mean_y[-1], n)
    k = int(np.nanargmax(np.abs(detr)))
    half = max(5, int(0.10 * n))
    i1, i2 = max(0, k - half), min(n - 1, k + half)
    e1, e2 = float(potential[i1]), float(potential[i2])
    if e1 == e2:
        e1, e2 = map(float, np.quantile(potential, [0.3, 0.7]))
    return sorted((e1, e2))


class ComparisonCanvas(FigureCanvas):
    """Two-row preview: SWV and DPV, each with voltammograms and calibration."""
    def __init__(self):
        self.fig = Figure(figsize=(11.5, 8), constrained_layout=True)
        super().__init__(self.fig)
        self.axes = {}
        self.datasets = {}
        self.results = {}
        self.fits = {}
        self.peak_fits = {}
        self.detection_limits = {}
        self.unknown_results = {}
        self.windows = {m: (None, None) for m in METHODS}
        self.unit = "mol/L"
        self.dragging = None
        self.cursor_lines = {}
        self.manual_peaks = {m: {} for m in METHODS}
        self.peak_markers = {m: [] for m in METHODS}
        self.on_window_changed = None
        self.on_peak_changed = None
        self.mpl_connect("button_press_event", self._press)
        self.mpl_connect("motion_notify_event", self._move)
        self.mpl_connect("button_release_event", self._release)
        self.mpl_connect("scroll_event", self._scroll_zoom)
        self.redraw()

    @staticmethod
    def _dataset_key(ds):
        """Stable identity for deciding whether a zoom view may be reused.

        A newly loaded data set must always start with Matplotlib autoscaling.
        Only redraws of the *same* SWV/DPV file (for example after manual peak
        picking) are allowed to restore the current zoom/pan limits.
        """
        if ds is None:
            return None
        try:
            path = ds.get("path")
            if path is not None:
                return str(Path(path).resolve())
        except Exception:
            pass
        # Fallback for programmatically supplied data sets without a path.
        try:
            return (id(ds.get("potential")), id(ds.get("currents")))
        except Exception:
            return id(ds)

    def set_state(self, datasets, results, fits, windows, unit, peak_fits=None, detection_limits=None, manual_peaks=None):
        old_keys = {m: self._dataset_key(self.datasets.get(m)) for m in METHODS}
        new_keys = {m: self._dataset_key(datasets.get(m)) for m in METHODS}
        preserve_methods = {
            m for m in METHODS
            if old_keys.get(m) is not None and old_keys.get(m) == new_keys.get(m)
        }

        # Keep a private shallow copy.  The main window mutates its datasets
        # dictionary when a second method (e.g. DPV after SWV) is loaded.
        # If the canvas stores the very same dict object, old_keys already see
        # the newly loaded file before set_state() can compare old vs. new.
        # That made the still-empty DPV axes (0..1 / 0..1) look like a valid
        # zoom state and restore those limits over the real DPV data.
        self.datasets = dict(datasets)
        self.results = results
        self.fits = fits
        self.peak_fits = peak_fits or {}
        self.detection_limits = detection_limits or {}
        self.manual_peaks = manual_peaks or {m: {} for m in METHODS}
        self.windows = {m: tuple(windows.get(m, (None, None))) for m in METHODS}
        self.unit = unit
        self.redraw(preserve_methods=preserve_methods)

    def redraw(self, preserve_methods=None):
        # Preserve the user's current zoom/pan view only when the same data set
        # is being redrawn.  The previous implementation also captured the
        # default empty-axes limits (0..1) before the first CSV was loaded and
        # restored them on top of real data, which could make SWV appear blank
        # or DPV show only a small clipped fragment.
        preserve_methods = set(preserve_methods or ())
        saved_views = {}
        for method, axes_pair in self.axes.items():
            if method not in preserve_methods:
                continue
            if axes_pair and axes_pair[0] is not None:
                ax_curve = axes_pair[0]
                try:
                    xlim = tuple(ax_curve.get_xlim())
                    ylim = tuple(ax_curve.get_ylim())
                    # Guard against invalid/collapsed limits.
                    if all(np.isfinite(xlim)) and all(np.isfinite(ylim)) and xlim[0] != xlim[1] and ylim[0] != ylim[1]:
                        saved_views[method] = (xlim, ylim)
                except Exception:
                    pass

        self.fig.clear()
        self.axes = {}
        self.cursor_lines = {}
        for row, method in enumerate(METHODS):
            ax_curve = self.fig.add_subplot(2, 2, row * 2 + 1)
            ax_cal = self.fig.add_subplot(2, 2, row * 2 + 2)
            self.axes[method] = (ax_curve, ax_cal)
            ds = self.datasets.get(method)
            if ds is None:
                ax_curve.text(0.5, 0.5, f"Load {method} CSV", ha="center", va="center", transform=ax_curve.transAxes)
                ax_cal.text(0.5, 0.5, f"{method} calibration", ha="center", va="center", transform=ax_cal.transAxes)
                ax_curve.set_axis_off(); ax_cal.set_axis_off()
                continue

            potential, conc, currents = ds["potential"], ds["conc"], ds["currents"]
            palette = ["#1c7ed6", "#0ca678", "#f08c00", "#e03131", "#7048e8", "#1098ad", "#c2255c", "#5c940d", "#d9480f"]
            for j, c in enumerate(conc):
                ax_curve.plot(potential, currents[:, j] * 1e6, lw=1.5, color=palette[j % len(palette)], label=f"{c:g}")
            ax_curve.set_xlabel("Potential / V")
            ax_curve.set_ylabel("Current / µA", color="#495057")
            ax_curve.tick_params(axis="y", labelcolor="#495057")
            ax_curve.set_title(f"{method} – stripping voltammograms (drag red cursors)")
            ax_curve.grid(True, alpha=0.25)
            e1, e2 = self.windows.get(method, (None, None))
            l1 = ax_curve.axvline(e1, ls="--", lw=1.8, color="red")
            l2 = ax_curve.axvline(e2, ls="--", lw=1.8, color="red")
            self.cursor_lines[method] = (l1, l2)
            # Mark manually selected peak positions. A click selects the nearest trace.
            for j, px in self.manual_peaks.get(method, {}).items():
                if 0 <= int(j) < currents.shape[1] and np.isfinite(px):
                    py = float(np.interp(px, potential, currents[:, int(j)]) * 1e6)
                    ax_curve.plot([px], [py], marker="o", ms=7, mfc="yellow", mec=palette[int(j) % len(palette)], mew=1.8, zorder=8)
            if len(conc) <= 10:
                ax_curve.legend(title=f"c / {self.unit}", fontsize=6.5, loc="best")

            # Restore the independently chosen SWV/DPV zoom window after the
            # curve and peak markers have been rebuilt. This keeps zoom active
            # while several manual peak points are selected one after another.
            if method in saved_views:
                try:
                    (xlim, ylim) = saved_views[method]
                    ax_curve.set_xlim(xlim)
                    ax_curve.set_ylim(ylim)
                except Exception:
                    pass

            table = self.results.get(method)
            fit = self.fits.get(method, (np.nan, np.nan, np.nan))
            if table is not None and not table.empty:
                x = table["Concentration"].to_numpy(float)
                y = table["Integrated peak area / A V"].to_numpy(float) * 1e6
                method_color = "#1c7ed6" if method == "SWV" else "#e8590c"
                ax_cal.scatter(x, y, s=44, color=method_color, edgecolors="white", linewidths=0.6, label="Calibration")
                slope, intercept, r2 = fit
                if np.isfinite(slope):
                    xx = np.linspace(np.min(x), np.max(x), 200)
                    yy = (slope * xx + intercept) * 1e6
                    ax_cal.plot(xx, yy, lw=2.0, color=method_color, label="Linear fit")
                    det = self.detection_limits.get(method, {})
                    lod = det.get("lod", np.nan); loq = det.get("loq", np.nan)
                    extra = ""
                    if np.isfinite(lod) and np.isfinite(loq):
                        extra = f"\nLOD = {lod:.4g}, LOQ = {loq:.4g} {self.unit}"
                    ax_cal.text(0.04, 0.96,
                                f"Area = {slope*1e6:.4g}·c + {intercept*1e6:.4g}\nR² = {r2:.5f}" + extra,
                                transform=ax_cal.transAxes, va="top",
                                bbox=dict(boxstyle="round", fc="white", ec="0.75", alpha=0.9))
                # Independent peak-current calibration on a secondary y axis.
                # Peak-current calibration is strictly manual and independent of
                # the integration window. Only clicked peak points are shown.
                pf = self.peak_fits.get(method, {})
                yp_all = np.abs(table["Manual peak current / A"].to_numpy(float))*1e6
                manual_mask = np.isfinite(yp_all)
                if np.any(manual_mask):
                    axp = ax_cal.twinx()
                    axp.scatter(x[manual_mask], yp_all[manual_mask], s=34, marker="s", facecolors="none", edgecolors="#6f42c1", linewidths=1.4, label="Manual |peak current|")
                    if pf and np.isfinite(pf.get("slope", np.nan)) and np.count_nonzero(manual_mask) >= 2:
                        xxp=np.linspace(np.min(x[manual_mask]), np.max(x[manual_mask]), 200)
                        axp.plot(xxp, (pf["slope"]*xxp+pf["intercept"])*1e6, ls="--", lw=1.7, color="#6f42c1")
                        ax_cal.text(0.04, 0.70, f"Manual |Ip| fit: R² = {pf['r2']:.5f}", transform=ax_cal.transAxes, va="top", color="#5f3dc4")
                    axp.set_ylabel("Manual |peak current| / µA", color="#6f42c1")
                    axp.tick_params(axis="y", labelcolor="#6f42c1")
            ures = self.unknown_results.get(method)
            if ures is not None and np.isfinite(ures.get("concentration", np.nan)):
                uc = float(ures["concentration"]); ua = float(ures["mean_area"]) * 1e6
                ax_cal.scatter([uc], [ua], s=95, marker="D", color="#d6336c", edgecolors="black", linewidths=0.8, zorder=6, label="Unknown")
                ax_cal.axhline(ua, ls=":", lw=1.2, color="#d6336c", alpha=0.8)
                ax_cal.axvline(uc, ls=":", lw=1.2, color="#d6336c", alpha=0.8)
                ax_cal.annotate(f"unknown c = {uc:.4g} {self.unit}", (uc, ua), xytext=(8, 10), textcoords="offset points", color="#a61e4d", fontsize=8, fontweight="bold")
                ax_cal.legend(fontsize=7, loc="best")

            ax_cal.set_xlabel(f"Concentration / {self.unit}")
            method_color = "#1c7ed6" if method == "SWV" else "#e8590c"
            ax_cal.set_ylabel("Integrated peak area / µA·V", color=method_color)
            ax_cal.tick_params(axis="y", labelcolor=method_color)
            ax_cal.set_title(f"{method} – integrated peak area vs. concentration")
            ax_cal.grid(True, alpha=0.25)
        self.draw_idle()

    def _method_for_axis(self, ax):
        for m, (a_curve, _a_cal) in self.axes.items():
            if ax is a_curve:
                return m
        return None

    def _nearest_cursor(self, method, x):
        ds = self.datasets.get(method)
        e1, e2 = self.windows.get(method, (None, None))
        if ds is None or e1 is None or e2 is None:
            return None
        span = max(np.ptp(ds["potential"]), 1e-9)
        d1, d2 = abs(x - e1), abs(x - e2)
        if min(d1, d2) <= 0.03 * span:
            return 1 if d1 <= d2 else 2
        return None

    def _common_range(self):
        loaded = [d for d in self.datasets.values() if d is not None]
        lo = max(float(np.min(d["potential"])) for d in loaded)
        hi = min(float(np.max(d["potential"])) for d in loaded)
        if hi <= lo:  # no overlap: use overall range; individual analyses will validate
            lo = min(float(np.min(d["potential"])) for d in loaded)
            hi = max(float(np.max(d["potential"])) for d in loaded)
        return lo, hi

    def _toolbar_active(self):
        """Return True while the Matplotlib pan/zoom toolbar owns mouse clicks."""
        tb = getattr(self, "toolbar", None)
        mode = getattr(tb, "mode", "") if tb is not None else ""
        return bool(mode)

    def _scroll_zoom(self, event):
        """Mouse-wheel zoom for the SWV/DPV voltammogram axes only.

        The axis under the pointer is zoomed independently, centered on the mouse
        position. This does not change integration limits or manual peak points.
        """
        method = self._method_for_axis(event.inaxes)
        if method is None or event.xdata is None or event.ydata is None:
            return
        ax = event.inaxes
        base_scale = 1.25
        scale = 1.0 / base_scale if event.button == "up" else base_scale
        x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
        xc, yc = float(event.xdata), float(event.ydata)
        new_x0 = xc - (xc - x0) * scale
        new_x1 = xc + (x1 - xc) * scale
        new_y0 = yc - (yc - y0) * scale
        new_y1 = yc + (y1 - yc) * scale
        ax.set_xlim(new_x0, new_x1); ax.set_ylim(new_y0, new_y1)
        self.draw_idle()

    def _press(self, event):
        method = self._method_for_axis(event.inaxes)

        # Right mouse button resets the voltammogram under the pointer.
        if event.button == 3 and method is not None:
            ax = event.inaxes
            ds = self.datasets.get(method)
            if ds is not None:
                potential = np.asarray(ds["potential"], dtype=float)
                currents_uA = np.asarray(ds["currents"], dtype=float) * 1e6
                finite_x = potential[np.isfinite(potential)]
                finite_y = currents_uA[np.isfinite(currents_uA)]
                if finite_x.size and finite_y.size:
                    xmin, xmax = float(np.min(finite_x)), float(np.max(finite_x))
                    ymin, ymax = float(np.min(finite_y)), float(np.max(finite_y))
                    dx, dy = xmax - xmin, ymax - ymin
                    xpad = 0.05 * dx if dx > 0 else max(abs(xmin) * 0.05, 0.05)
                    ypad = 0.05 * dy if dy > 0 else max(abs(ymin) * 0.05, 0.05)
                    ax.set_xlim(xmin - xpad, xmax + xpad)
                    ax.set_ylim(ymin - ypad, ymax + ypad)
                    self.draw_idle()
            return

        if self._toolbar_active():
            return
        if method and event.xdata is not None and event.button == 1:
            which = self._nearest_cursor(method, event.xdata)
            if which:
                self.dragging = (method, which)
            else:
                ds = self.datasets.get(method)
                if ds is not None:
                    x = float(event.xdata)
                    ys = np.array([np.interp(x, ds["potential"], ds["currents"][:, j]) * 1e6 for j in range(ds["currents"].shape[1])])
                    if event.ydata is not None and len(ys):
                        j = int(np.nanargmin(np.abs(ys - float(event.ydata))))
                        self.manual_peaks.setdefault(method, {})[j] = x
                        if callable(self.on_peak_changed):
                            self.on_peak_changed(method, j, x)

    def _move(self, event):
        method = self._method_for_axis(event.inaxes)
        if self.dragging is None or event.xdata is None or method is None:
            return
        drag_method, drag_which = self.dragging
        if method != drag_method:
            return
        ds = self.datasets.get(method)
        if ds is None:
            return
        lo, hi = float(np.min(ds["potential"])), float(np.max(ds["potential"]))
        val = float(np.clip(event.xdata, lo, hi))
        e1, e2 = self.windows[method]
        if drag_which == 1:
            e1 = min(val, e2 - 1e-9)
        else:
            e2 = max(val, e1 + 1e-9)
        self.windows[method] = (e1, e2)
        l1, l2 = self.cursor_lines[method]
        l1.set_xdata([e1, e1]); l2.set_xdata([e2, e2])
        self.draw_idle()

    def _release(self, event):
        method = self._method_for_axis(event.inaxes)
        if self.dragging is not None:
            drag_method, _ = self.dragging
            self.dragging = None
            if callable(self.on_window_changed):
                e1, e2 = self.windows[drag_method]
                self.on_window_changed(drag_method, e1, e2)


class StrippingWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(MODULE_TITLE)
        self.resize(1480, 900)
        self.datasets = {m: None for m in METHODS}
        self.results = {m: None for m in METHODS}
        self.fits = {m: (np.nan, np.nan, np.nan) for m in METHODS}
        self.peak_fits = {m: {} for m in METHODS}
        self.detection_limits = {m: {} for m in METHODS}
        self.unknown = {m: None for m in METHODS}
        self.unknown_results = {m: None for m in METHODS}
        self.manual_peaks = {m: {} for m in METHODS}
        self.temp_html = None
        self._build_ui()

    def _build_ui(self):
        root = QWidget(); self.setCentralWidget(root)
        main = QVBoxLayout(root)
        title = QLabel(MODULE_TITLE)
        title.setStyleSheet("font-size: 23px; font-weight: 700; color: #18324a;")
        subtitle = QLabel("Load SWV and/or DPV data. CSV format: first column = potential; first row = concentration. Both methods are shown in preview, waterfall, PNG and Excel export.")
        subtitle.setWordWrap(True)
        main.addWidget(title); main.addWidget(subtitle)

        splitter = QSplitter(Qt.Horizontal); main.addWidget(splitter, 1)
        left = QWidget(); left.setMinimumWidth(500); left.setMaximumWidth(650)
        lv = QVBoxLayout(left)
        # Keep the full control panel at its natural minimum height so the QScrollArea
        # provides vertical scrolling instead of compressing/clipping the lower export controls.
        lv.setSizeConstraint(QLayout.SetMinimumSize)

        box = QGroupBox("Input – SWV and DPV")
        box.setStyleSheet("QGroupBox {font-weight:700; color:#174a7e; border:1px solid #9ec5e5; border-radius:6px; margin-top:8px; padding-top:8px;} QGroupBox::title {subcontrol-origin: margin; left:10px; padding:0 4px;}")
        form = QFormLayout(box)
        self.file_edits = {}
        for method in METHODS:
            edit = QLineEdit(); edit.setReadOnly(True); self.file_edits[method] = edit
            btn = QPushButton(f"Load {method}…")
            btn.clicked.connect(lambda _checked=False, m=method: self.load_csv(m))
            row = QWidget(); rh = QHBoxLayout(row); rh.setContentsMargins(0,0,0,0); rh.addWidget(edit,1); rh.addWidget(btn)
            form.addRow(f"{method} file:", row)
        self.unknown_edits = {}
        for method in METHODS:
            edit = QLineEdit(); edit.setReadOnly(True); self.unknown_edits[method] = edit
            btn = QPushButton(f"Load unknown {method}…")
            btn.setStyleSheet("background:#7b2cbf; color:white; font-weight:600; padding:5px;")
            btn.clicked.connect(lambda _checked=False, m=method: self.load_unknown(m))
            row = QWidget(); rh = QHBoxLayout(row); rh.setContentsMargins(0,0,0,0); rh.addWidget(edit,1); rh.addWidget(btn)
            form.addRow(f"Unknown {method} (optional):", row)
        self.unit = QComboBox(); self.unit.addItems(["mol/L", "mmol/L", "µmol/L", "mg/L", "user-defined"])
        self.unit.currentTextChanged.connect(self.recalculate)
        form.addRow("Concentration unit:", self.unit)
        lv.addWidget(box)

        ibox = QGroupBox("Peak integration – independent cursor windows")
        ibox.setStyleSheet("QGroupBox {font-weight:700; color:#6a1b9a; border:1px solid #c9a7df; border-radius:6px; margin-top:8px; padding-top:8px;} QGroupBox::title {subcontrol-origin: margin; left:10px; padding:0 4px;}")
        iform = QFormLayout(ibox)
        self.window_edits = {}
        for method in METHODS:
            e1e = QLineEdit(); e2e = QLineEdit()
            self.window_edits[method] = (e1e, e2e)
            e1e.editingFinished.connect(lambda m=method: self.recalculate_from_fields(m))
            e2e.editingFinished.connect(lambda m=method: self.recalculate_from_fields(m))
            row = QWidget(); rh = QHBoxLayout(row); rh.setContentsMargins(0,0,0,0)
            rh.addWidget(QLabel("E1:")); rh.addWidget(e1e); rh.addWidget(QLabel("E2:")); rh.addWidget(e2e)
            iform.addRow(f"{method} / V:", row)
        self.abs_area = QCheckBox("Use absolute integrated peak area")
        self.abs_area.setChecked(True); self.abs_area.toggled.connect(self.recalculate)
        info = QLabel("SWV and DPV have independent E1–E2 integration windows. Drag the two red cursors separately in each preview plot. Click directly on each desired peak to define peak potential/current manually for the nearest concentration trace (yellow marker). Peak currents are read from the raw curve and do NOT depend on E1–E2. The peak-current calibration appears only after manual peak points have been selected. Local baseline for integration only: straight line between E1 and E2 for each concentration trace.")
        info.setWordWrap(True)
        iform.addRow(self.abs_area); iform.addRow(info)
        reset_row = QWidget(); rr = QHBoxLayout(reset_row); rr.setContentsMargins(0,0,0,0)
        for method in METHODS:
            b = QPushButton(f"Reset manual {method} peaks")
            b.clicked.connect(lambda _checked=False, m=method: self.reset_manual_peaks(m))
            rr.addWidget(b)
        iform.addRow("Manual peaks:", reset_row)
        lv.addWidget(ibox)

        # Optional experimental metadata. Three side-by-side columns: Common | SWV | DPV.
        # These entries document the stripping procedure but intentionally do not
        # alter the numerical analysis.
        self.proc_box = QGroupBox("Measurement procedure (optional)")
        self.proc_box.setCheckable(True)
        self.proc_box.setChecked(False)
        self.proc_box.setStyleSheet("QGroupBox {font-weight:700; color:#2b8a3e; border:1px solid #8fd19e; border-radius:6px; margin-top:8px; padding-top:8px;} QGroupBox::title {subcontrol-origin: margin; left:10px; padding:0 4px;}")
        self.proc_box.setMinimumHeight(430)
        pv = QVBoxLayout(self.proc_box)

        columns = QWidget(); ch = QHBoxLayout(columns); ch.setContentsMargins(0,0,0,0); ch.setSpacing(8)
        self.proc_common = {}
        self.proc_method = {m:{} for m in METHODS}

        def add_proc_column(title_text, fields, store, accent):
            gb = QGroupBox(title_text)
            gb.setStyleSheet(f"QGroupBox {{font-weight:700; color:{accent}; border:1px solid #cfd8dc; border-radius:5px; margin-top:6px; padding-top:7px;}} QGroupBox::title {{subcontrol-origin: margin; left:8px; padding:0 3px;}}")
            vv = QVBoxLayout(gb); vv.setSpacing(3)
            for key, label, placeholder in fields:
                lab = QLabel(label); lab.setStyleSheet("font-size:10px; color:#343a40;")
                e = QLineEdit(); e.setPlaceholderText(placeholder); e.setMinimumWidth(125)
                store[key] = e
                vv.addWidget(lab); vv.addWidget(e)
            vv.addStretch(1)
            ch.addWidget(gb, 1)

        add_proc_column("Common", [
            ("analyte", "Analyte", "e.g. Cu²⁺"),
            ("electrode", "Working electrode", "e.g. Bi-film electrode"),
            ("electrolyte", "Supporting electrolyte", "composition / concentration"),
            ("reference", "Reference electrode", "e.g. Ag/AgCl (3 M KCl)"),
            ("deposition_potential", "Deposition potential / V", "optional"),
            ("deposition_time", "Deposition time / s", "optional"),
            ("equilibration_time", "Equilibration/rest time / s", "optional"),
        ], self.proc_common, "#2b8a3e")
        add_proc_column("SWV", [
            ("frequency", "Frequency / Hz", "optional"),
            ("amplitude", "Amplitude / V", "optional"),
            ("step", "Potential step / V", "optional"),
        ], self.proc_method["SWV"], "#1c7ed6")
        add_proc_column("DPV", [
            ("pulse_amplitude", "Pulse amplitude / V", "optional"),
            ("pulse_duration", "Pulse duration / s", "optional"),
            ("step", "Potential step / V", "optional"),
        ], self.proc_method["DPV"], "#e8590c")
        pv.addWidget(columns)

        self.proc_notes = QPlainTextEdit(); self.proc_notes.setPlaceholderText("Notes / measurement procedure (optional)")
        self.proc_notes.setMinimumHeight(70); self.proc_notes.setMaximumHeight(90)
        pv.addWidget(self.proc_notes)
        proc_hint=QLabel("Documentation only – these entries do not affect peak integration or calibration.")
        proc_hint.setWordWrap(True); proc_hint.setStyleSheet("color:#52734d; font-size:10px;")
        pv.addWidget(proc_hint)
        lv.addWidget(self.proc_box)

        ubox = QGroupBox("Unknown concentration")
        ubox.setStyleSheet("QGroupBox {font-weight:700; color:#a61e4d; border:1px solid #e8a5bc; border-radius:6px; margin-top:8px; padding-top:8px;} QGroupBox::title {subcontrol-origin: margin; left:10px; padding:0 4px;}")
        uv = QVBoxLayout(ubox)
        self.unknown_labels = {}
        for method in METHODS:
            lab = QLabel(f"{method}: no unknown sample loaded")
            lab.setWordWrap(True)
            lab.setStyleSheet("padding:5px; background:#fff0f6; border-radius:4px; color:#8a1746;")
            self.unknown_labels[method] = lab
            uv.addWidget(lab)
        lv.addWidget(ubox)

        self.waterfall_btn = QPushButton("Open interactive 3D waterfall (SWV + DPV)")
        self.waterfall_btn.clicked.connect(self.open_waterfall); self.waterfall_btn.setEnabled(False)
        self.export_html_btn = QPushButton("Export 3D waterfall HTML…")
        self.export_html_btn.clicked.connect(self.export_waterfall); self.export_html_btn.setEnabled(False)
        self.export_excel_btn = QPushButton("Export SWV + DPV results to Excel…")
        self.export_excel_btn.clicked.connect(self.export_excel); self.export_excel_btn.setEnabled(False)
        self.export_png_btn = QPushButton("Export SWV + DPV preview PNG…")
        self.export_png_btn.clicked.connect(self.export_png); self.export_png_btn.setEnabled(False)
        button_colors = ["#0b7285", "#1971c2", "#2f9e44", "#f08c00"]
        for b, color in zip([self.waterfall_btn, self.export_html_btn, self.export_excel_btn, self.export_png_btn], button_colors):
            b.setStyleSheet(f"background:{color}; color:white; font-weight:600; padding:7px; border-radius:4px;")
            lv.addWidget(b)
        lv.addStretch()
        self.status = QLabel("Load SWV and/or DPV CSV file(s). For direct comparison, load both.")
        self.status.setWordWrap(True); lv.addWidget(self.status)
        left_scroll = QScrollArea(); left_scroll.setWidgetResizable(True); left_scroll.setFrameShape(QScrollArea.NoFrame)
        left_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        left_scroll.setWidget(left)
        splitter.addWidget(left_scroll)

        tabs = QTabWidget()
        analysis_tab = QWidget(); av = QVBoxLayout(analysis_tab)
        self.canvas = ComparisonCanvas(); self.canvas.on_window_changed = self.window_changed; self.canvas.on_peak_changed = self.peak_changed
        av.addWidget(self.canvas, 1)
        self.preview_toolbar = NavigationToolbar(self.canvas, analysis_tab)
        self.preview_toolbar.setToolTip("Zoom/pan the SWV and DPV plots independently. Mouse wheel zooms; right-click in an SWV/DPV plot resets that plot to the full data range. The magnifier and hand tools remain available.")
        av.addWidget(self.preview_toolbar)
        zoom_hint = QLabel("Zoom: mouse wheel over an SWV/DPV voltammogram. Right-click in the plot resets its zoom to the complete data range. Magnifier/pan tools remain available. Zooming does not change integration limits or manual peak selections.")
        zoom_hint.setWordWrap(True)
        zoom_hint.setStyleSheet("color:#495057; font-size:11px; padding:2px 4px;")
        av.addWidget(zoom_hint)
        tabs.addTab(analysis_tab, "SWV + DPV preview")

        table_tab = QWidget(); tv = QVBoxLayout(table_tab)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["Method", "Concentration", "Peak potential / V", "Manual peak current / µA", "Signed area / µA·V", "Area / µA·V"])
        self.table.horizontalHeader().setStretchLastSection(True)
        tv.addWidget(self.table)
        tabs.addTab(table_tab, "Combined results table")
        splitter.addWidget(tabs); splitter.setStretchFactor(1, 1)

    def _loaded_methods(self):
        return [m for m in METHODS if self.datasets[m] is not None]

    def _buttons_enabled(self):
        enabled = bool(self._loaded_methods())
        for b in [self.waterfall_btn, self.export_html_btn, self.export_excel_btn, self.export_png_btn]:
            b.setEnabled(enabled)

    def _common_potential_range(self):
        loaded = [self.datasets[m] for m in self._loaded_methods()]
        lo = max(float(np.min(d["potential"])) for d in loaded)
        hi = min(float(np.max(d["potential"])) for d in loaded)
        if hi <= lo:
            raise ValueError("SWV and DPV potential ranges do not overlap; a shared integration window cannot be used.")
        return lo, hi

    def load_csv(self, method):
        path, _ = QFileDialog.getOpenFileName(self, f"Load {method} stripping-voltammetry CSV", "", "CSV files (*.csv);;All files (*)")
        if not path:
            return
        try:
            potential, conc, currents, header, sep = load_stripping_csv(path)
            self.datasets[method] = {
                "path": Path(path), "potential": potential, "conc": conc, "currents": currents,
                "header": header, "delimiter": sep,
            }
            self.file_edits[method].setText(path)

            # Initialize an independent integration window for this method.
            e1_edit, e2_edit = self.window_edits[method]
            if not e1_edit.text().strip() or not e2_edit.text().strip():
                e1, e2 = _initial_window(potential, currents)
            else:
                e1 = float(e1_edit.text().replace(",", ".")); e2 = float(e2_edit.text().replace(",", "."))
                e1, e2 = sorted((e1, e2))
            lo, hi = float(np.min(potential)), float(np.max(potential))
            e1 = max(e1, lo); e2 = min(e2, hi)
            if e2 <= e1:
                e1, e2 = _initial_window(potential, currents)
            e1_edit.setText(f"{e1:.6g}"); e2_edit.setText(f"{e2:.6g}")
            self._buttons_enabled(); self.recalculate()
            loaded = ", ".join(self._loaded_methods())
            self.status.setText(f"Loaded {method}: {len(potential)} potential points × {len(conc)} concentrations. Active methods: {loaded}.")
        except Exception as e:
            QMessageBox.critical(self, f"{method} CSV import", str(e))

    def load_unknown(self, method):
        if self.datasets.get(method) is None:
            QMessageBox.information(self, "Unknown sample", f"Load the {method} calibration CSV first.")
            return
        path, _ = QFileDialog.getOpenFileName(self, f"Load unknown {method} CSV", "", "CSV files (*.csv);;All files (*)")
        if not path:
            return
        try:
            potential, currents, sep = load_unknown_csv(path)
            self.unknown[method] = {"path": Path(path), "potential": potential, "currents": currents, "delimiter": sep}
            self.unknown_edits[method].setText(path)
            self.recalculate()
            self.status.setText(f"Unknown {method} sample loaded ({currents.shape[1]} replicate curve(s)).")
        except Exception as e:
            QMessageBox.critical(self, f"Unknown {method} CSV import", str(e))

    def recalculate_from_fields(self, method=None):
        if not self._loaded_methods():
            return
        methods = [method] if method else self._loaded_methods()
        try:
            for m in methods:
                if self.datasets.get(m) is None:
                    continue
                e1_edit, e2_edit = self.window_edits[m]
                e1 = float(e1_edit.text().replace(",", ".")); e2 = float(e2_edit.text().replace(",", "."))
                e1, e2 = sorted((e1, e2))
                p = self.datasets[m]["potential"]
                lo, hi = float(np.min(p)), float(np.max(p))
                if e1 < lo or e2 > hi:
                    raise ValueError(f"{m} integration window must lie within {lo:.6g} to {hi:.6g} V.")
                e1_edit.setText(f"{e1:.6g}"); e2_edit.setText(f"{e2:.6g}")
            self.recalculate()
        except Exception as e:
            QMessageBox.warning(self, "Integration window", str(e))

    def window_changed(self, method, e1, e2):
        e1_edit, e2_edit = self.window_edits[method]
        e1_edit.setText(f"{e1:.6g}"); e2_edit.setText(f"{e2:.6g}")
        self.recalculate()

    def reset_manual_peaks(self, method):
        self.manual_peaks[method] = {}
        self.recalculate()

    def peak_changed(self, method, trace_index, potential):
        self.manual_peaks.setdefault(method, {})[int(trace_index)] = float(potential)
        self.recalculate()

    def recalculate(self):
        loaded = self._loaded_methods()
        if not loaded:
            return
        try:
            windows = {}
            for method in METHODS:
                ds = self.datasets[method]
                if ds is None:
                    self.results[method] = None
                    self.fits[method] = (np.nan, np.nan, np.nan)
                    self.peak_fits[method] = {}
                    self.detection_limits[method] = {}
                    self.unknown_results[method] = None
                    continue
                e1_edit, e2_edit = self.window_edits[method]
                e1 = float(e1_edit.text().replace(",", ".")); e2 = float(e2_edit.text().replace(",", "."))
                e1, e2 = sorted((e1, e2))
                lo, hi = float(np.min(ds["potential"])), float(np.max(ds["potential"]))
                e1 = max(e1, lo); e2 = min(e2, hi)
                if e2 <= e1:
                    raise ValueError(f"No valid {method} integration window remains.")
                windows[method] = (e1, e2)
                e1_edit.setText(f"{e1:.6g}"); e2_edit.setText(f"{e2:.6g}")
                self.results[method], slope, intercept, r2 = analyze_dataset(
                    ds["potential"], ds["conc"], ds["currents"], e1, e2, self.abs_area.isChecked(), self.manual_peaks.get(method, {}))
                self.fits[method] = (slope, intercept, r2)
                tab = self.results[method]
                self.peak_fits[method] = linear_fit_stats(tab["Concentration"], np.abs(tab["Manual peak current / A"]))
                self.detection_limits[method] = linear_fit_stats(tab["Concentration"], tab["Integrated peak area / A V"])
                u = self.unknown.get(method)
                if u is not None:
                    utab, umean, usd, uc = analyze_unknown(u["potential"], u["currents"], e1, e2, slope, intercept, self.abs_area.isChecked())
                    self.unknown_results[method] = {"table": utab, "mean_area": umean, "sd_area": usd, "concentration": uc}
                else:
                    self.unknown_results[method] = None
            for method in METHODS:
                ures = self.unknown_results.get(method)
                if ures is None:
                    self.unknown_labels[method].setText(f"{method}: no unknown sample loaded")
                else:
                    cmin=float(np.nanmin(self.datasets[method]["conc"])); cmax=float(np.nanmax(self.datasets[method]["conc"]))
                    uc=float(ures["concentration"])
                    warning = "  ⚠ EXTRAPOLATION outside calibration range" if (uc < cmin or uc > cmax) else ""
                    if uc < 0:
                        warning += "  ⚠ negative calculated concentration"
                    self.unknown_labels[method].setText(
                        f"{method}: mean area = {ures['mean_area']*1e6:.6g} µA·V  →  unknown concentration = {uc:.6g} {self.unit.currentText()}{warning}"
                    )
                    self.unknown_labels[method].setStyleSheet("padding:5px; background:#fff3bf; border-radius:4px; color:#9c2f00; font-weight:600;" if warning else "padding:5px; background:#fff0f6; border-radius:4px; color:#8a1746;")
            self.canvas.unknown_results = self.unknown_results
            self.canvas.set_state(self.datasets, self.results, self.fits, windows, self.unit.currentText(), self.peak_fits, self.detection_limits, self.manual_peaks)
            self.populate_table()
            status_parts = []
            for m in loaded:
                r2 = self.fits[m][2]
                e1, e2 = windows[m]
                det=self.detection_limits.get(m,{})
                dl = f", LOD={det.get('lod',np.nan):.3g}, LOQ={det.get('loq',np.nan):.3g}" if np.isfinite(det.get("lod",np.nan)) else ""
                status_parts.append((f"{m}: {e1:.5g}–{e2:.5g} V, R²={r2:.5f}" + dl) if np.isfinite(r2) else f"{m}: {e1:.5g}–{e2:.5g} V, R²=n/a")
            self.status.setText("Independent integration windows; " + "; ".join(status_parts))
        except Exception as e:
            self.status.setText(f"Analysis error: {e}")

    def populate_table(self):
        rows = []
        for method in METHODS:
            table = self.results.get(method)
            if table is None:
                continue
            for _, row in table.iterrows():
                rows.append((method, row))
        self.table.setRowCount(len(rows))
        for i, (method, row) in enumerate(rows):
            vals = [method, row["Concentration"], row["Peak potential / V"],
                    row["Manual peak current / A"] * 1e6,
                    row["Signed integrated peak area / A V"] * 1e6,
                    row["Integrated peak area / A V"] * 1e6]
            for j, v in enumerate(vals):
                text = str(v) if j == 0 else f"{float(v):.7g}"
                self.table.setItem(i, j, QTableWidgetItem(text))

    def _waterfall_figure(self):
        loaded = self._loaded_methods()
        if not loaded:
            raise ValueError("No data loaded.")
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        if len(loaded) == 2:
            fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]],
                                subplot_titles=("SWV", "DPV"), horizontal_spacing=0.03)
            for col, method in enumerate(METHODS, start=1):
                ds = self.datasets[method]
                for j, c in enumerate(ds["conc"]):
                    fig.add_trace(go.Scatter3d(
                        x=ds["potential"], y=np.full_like(ds["potential"], c, dtype=float), z=ds["currents"][:, j] * 1e6,
                        mode="lines", name=f"{method}: {c:g} {self.unit.currentText()}", line=dict(width=5, color=["#1c7ed6","#0ca678","#f08c00","#e03131","#7048e8","#1098ad","#c2255c"][j % 7]),
                        legendgroup=method), row=1, col=col)
                u = self.unknown.get(method)
                ures = self.unknown_results.get(method)
                if u is not None and ures is not None:
                    uc = float(ures["concentration"])
                    mean_u = np.nanmean(u["currents"], axis=1)
                    fig.add_trace(go.Scatter3d(
                        x=u["potential"], y=np.full_like(u["potential"], uc, dtype=float), z=mean_u * 1e6,
                        mode="lines", name=f"{method}: unknown → {uc:.4g} {self.unit.currentText()}",
                        line=dict(width=8, color="#d6336c", dash="dash"), legendgroup=f"{method}-unknown"), row=1, col=col)
            scene_cfg = dict(xaxis_title="Potential / V", yaxis_title=f"Concentration / {self.unit.currentText()}", zaxis_title="Current / µA")
            fig.update_layout(scene=scene_cfg, scene2=scene_cfg)
        else:
            method = loaded[0]; ds = self.datasets[method]
            fig = go.Figure()
            for j, c in enumerate(ds["conc"]):
                fig.add_trace(go.Scatter3d(
                    x=ds["potential"], y=np.full_like(ds["potential"], c, dtype=float), z=ds["currents"][:, j] * 1e6,
                    mode="lines", name=f"{method}: {c:g} {self.unit.currentText()}", line=dict(width=5, color=["#1c7ed6","#0ca678","#f08c00","#e03131","#7048e8","#1098ad","#c2255c"][j % 7])))
            u = self.unknown.get(method)
            ures = self.unknown_results.get(method)
            if u is not None and ures is not None:
                uc = float(ures["concentration"]); mean_u = np.nanmean(u["currents"], axis=1)
                fig.add_trace(go.Scatter3d(
                    x=u["potential"], y=np.full_like(u["potential"], uc, dtype=float), z=mean_u * 1e6,
                    mode="lines", name=f"{method}: unknown → {uc:.4g} {self.unit.currentText()}",
                    line=dict(width=8, color="#d6336c", dash="dash")))
            fig.update_layout(scene=dict(xaxis_title="Potential / V", yaxis_title=f"Concentration / {self.unit.currentText()}", zaxis_title="Current / µA"))
        fig.update_layout(title="SWV and DPV – interactive stripping-voltammetry waterfall",
                          margin=dict(l=0, r=0, t=60, b=0), legend=dict(title="Method / concentration"))
        return fig

    def open_waterfall(self):
        try:
            fig = self._waterfall_figure()
            p = Path(tempfile.gettempdir()) / "SEC_Stripping_SWV_DPV_Waterfall.html"
            fig.write_html(p, include_plotlyjs="cdn")
            self.temp_html = p
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(p)))
        except Exception as e:
            QMessageBox.critical(self, "3D waterfall", str(e))

    def export_waterfall(self):
        p, _ = QFileDialog.getSaveFileName(self, "Export interactive SWV + DPV waterfall", "Stripping_SWV_DPV_Waterfall.html", "HTML (*.html)")
        if not p:
            return
        try:
            self._waterfall_figure().write_html(p, include_plotlyjs="cdn")
            self.status.setText(f"Waterfall exported: {p}")
        except Exception as e:
            QMessageBox.critical(self, "HTML export", str(e))

    def export_png(self):
        p, _ = QFileDialog.getSaveFileName(self, "Export SWV + DPV preview", "Stripping_SWV_DPV_Analysis.png", "PNG (*.png)")
        if not p:
            return
        try:
            self.canvas.fig.savefig(p, dpi=300, bbox_inches="tight")
            self.status.setText(f"PNG exported: {p}")
        except Exception as e:
            QMessageBox.critical(self, "PNG export", str(e))

    def _procedure_metadata(self, method):
        """Return optional experimental metadata for Excel export."""
        if not self.proc_box.isChecked():
            return []
        common_labels = {
            "analyte":"Analyte", "electrode":"Working electrode",
            "electrolyte":"Supporting electrolyte", "reference":"Reference electrode",
            "deposition_potential":"Deposition potential / V",
            "deposition_time":"Deposition time / s",
            "equilibration_time":"Equilibration/rest time / s",
        }
        method_labels = {
            "SWV":{"frequency":"SWV frequency / Hz", "amplitude":"SWV amplitude / V", "step":"SWV potential step / V"},
            "DPV":{"pulse_amplitude":"DPV pulse amplitude / V", "pulse_duration":"DPV pulse duration / s", "step":"DPV potential step / V"},
        }
        rows=[]
        for key,label in common_labels.items():
            value=self.proc_common[key].text().strip()
            if value: rows.append((label,value))
        for key,label in method_labels[method].items():
            value=self.proc_method[method][key].text().strip()
            if value: rows.append((label,value))
        notes=self.proc_notes.toPlainText().strip()
        if notes: rows.append(("Notes / measurement procedure",notes))
        return rows

    def export_excel(self):
        p, _ = QFileDialog.getSaveFileName(self, "Export SWV + DPV stripping analysis", "Stripping_SWV_DPV_Analysis.xlsx", "Excel (*.xlsx)")
        if not p:
            return
        try:
            unit = self.unit.currentText()
            combined_rows = []
            for method in METHODS:
                table = self.results.get(method)
                if table is not None:
                    t = table.copy(); t.insert(0, "Method", method); combined_rows.append(t)
            combined = pd.concat(combined_rows, ignore_index=True) if combined_rows else pd.DataFrame()

            with pd.ExcelWriter(p, engine="xlsxwriter") as writer:
                for method in METHODS:
                    ds = self.datasets[method]
                    if ds is None:
                        continue
                    raw = pd.DataFrame({"Potential / V": ds["potential"]})
                    for j, c in enumerate(ds["conc"]):
                        raw[f"Current / A; c={c:g} {unit}"] = ds["currents"][:, j]
                    raw.to_excel(writer, sheet_name=f"{method} Raw data", index=False)
                    self.results[method].to_excel(writer, sheet_name=f"{method} Peak analysis", index=False)
                    ures = self.unknown_results.get(method)
                    if ures is not None:
                        ures["table"].to_excel(writer, sheet_name=f"{method} Unknown", index=False)
                        pd.DataFrame({
                            "Parameter": ["Mean integrated area / A V", "SD integrated area / A V", f"Calculated concentration / {unit}"],
                            "Value": [ures["mean_area"], ures["sd_area"], ures["concentration"]]
                        }).to_excel(writer, sheet_name=f"{method} Unknown", startrow=len(ures["table"])+3, index=False)
                    slope, intercept, r2 = self.fits[method]
                    e1_edit, e2_edit = self.window_edits[method]
                    e1, e2 = sorted((float(e1_edit.text().replace(",", ".")), float(e2_edit.text().replace(",", "."))))
                    meta = pd.DataFrame({
                        "Parameter": ["Method", "Source file", "Concentration unit", "E1 / V", "E2 / V", "Baseline", "Absolute area",
                                      "Calibration slope / (A V)/concentration", "Calibration intercept / A V", "Calibration R2",
                                      f"LOD / {unit}", f"LOQ / {unit}", "LOD/LOQ sigma definition",
                                      "Manual peak-current calibration slope / A/concentration", "Manual peak-current calibration intercept / A", "Manual peak-current calibration R2",
                                      "Unknown sample file", f"Unknown concentration / {unit}", "Unknown extrapolation warning"],
                        "Value": [METHOD_LONG[method], str(ds["path"]), unit, e1, e2, "Linear between E1 and E2",
                                  self.abs_area.isChecked(), slope, intercept, r2,
                                  self.detection_limits.get(method,{}).get("lod",np.nan), self.detection_limits.get(method,{}).get("loq",np.nan), "Residual standard deviation of linear area calibration (n-2 degrees of freedom)",
                                  self.peak_fits.get(method,{}).get("slope",np.nan), self.peak_fits.get(method,{}).get("intercept",np.nan), self.peak_fits.get(method,{}).get("r2",np.nan),
                                  str(self.unknown[method]["path"]) if self.unknown.get(method) is not None else "",
                                  self.unknown_results[method]["concentration"] if self.unknown_results.get(method) is not None else np.nan,
                                  ("EXTRAPOLATION" if self.unknown_results.get(method) is not None and (self.unknown_results[method]["concentration"] < float(np.nanmin(ds["conc"])) or self.unknown_results[method]["concentration"] > float(np.nanmax(ds["conc"]))) else "")]
                    })
                    proc_rows = self._procedure_metadata(method)
                    if proc_rows:
                        proc_df = pd.DataFrame(proc_rows, columns=["Parameter", "Value"])
                        meta = pd.concat([meta, pd.DataFrame({"Parameter":["--- Measurement procedure (optional) ---"], "Value":[""]}), proc_df], ignore_index=True)
                    meta.to_excel(writer, sheet_name=f"{method} Metadata", index=False)

                    # Dedicated calibration sheet including the optional unknown point.
                    cal = pd.DataFrame({
                        f"Concentration / {unit}": self.results[method]["Concentration"].to_numpy(float),
                        "Integrated peak area / A V": self.results[method]["Integrated peak area / A V"].to_numpy(float),
                    })
                    cal.to_excel(writer, sheet_name=f"{method} Calibration", index=False)
                    wb = writer.book
                    ws_cal = writer.sheets[f"{method} Calibration"]
                    ncal = len(cal)
                    chart_cal = wb.add_chart({"type": "scatter", "subtype": "straight_with_markers"})
                    chart_cal.add_series({
                        "name": f"{method} calibration",
                        "categories": [f"{method} Calibration", 1, 0, ncal, 0],
                        "values": [f"{method} Calibration", 1, 1, ncal, 1],
                        "marker": {"type": "circle", "size": 7, "border": {"color": "#ffffff"}, "fill": {"color": "#1c7ed6" if method == "SWV" else "#e8590c"}},
                        "line": {"none": True},
                        "trendline": {"type": "linear", "display_equation": True, "display_r_squared": True, "line": {"color": "#364fc7"}},
                    })
                    if ures is not None:
                        ur = ncal + 3
                        ws_cal.write(ur, 0, "Unknown calculated concentration")
                        ws_cal.write_number(ur, 1, float(ures["concentration"]))
                        ws_cal.write(ur + 1, 0, "Unknown mean integrated peak area / A V")
                        ws_cal.write_number(ur + 1, 1, float(ures["mean_area"]))
                        # Put x/y values into a compact pair for charting.
                        ws_cal.write(ur + 3, 0, "Unknown concentration")
                        ws_cal.write(ur + 3, 1, "Unknown area")
                        ws_cal.write_number(ur + 4, 0, float(ures["concentration"]))
                        ws_cal.write_number(ur + 4, 1, float(ures["mean_area"]))
                        chart_cal.add_series({
                            "name": "Unknown",
                            "categories": [f"{method} Calibration", ur + 4, 0, ur + 4, 0],
                            "values": [f"{method} Calibration", ur + 4, 1, ur + 4, 1],
                            "marker": {"type": "diamond", "size": 9, "border": {"color": "#000000"}, "fill": {"color": "#d6336c"}},
                            "line": {"none": True},
                        })
                    chart_cal.set_title({"name": f"{method}: integrated peak area vs concentration"})
                    chart_cal.set_x_axis({"name": f"Concentration / {unit}"})
                    chart_cal.set_y_axis({"name": "Integrated peak area / A V"})
                    chart_cal.set_legend({"position": "bottom"})
                    ws_cal.insert_chart("D2", chart_cal, {"x_scale": 1.55, "y_scale": 1.35})

                if not combined.empty:
                    combined.to_excel(writer, sheet_name="Combined comparison", index=False)
                    wb = writer.book
                    ws = writer.sheets["Combined comparison"]
                    chart = wb.add_chart({"type": "scatter", "subtype": "straight_with_markers"})
                    start = 1
                    for method in METHODS:
                        table = self.results.get(method)
                        if table is None:
                            continue
                        n = len(table)
                        # Combined columns: Method=0, Concentration=1, ..., Integrated area=5
                        chart.add_series({
                            "name": method,
                            "categories": ["Combined comparison", start, 1, start + n - 1, 1],
                            "values": ["Combined comparison", start, 5, start + n - 1, 5],
                            "marker": {"type": "circle"},
                            "trendline": {"type": "linear", "display_equation": True, "display_r_squared": True},
                        })
                        start += n
                    chart.set_title({"name": "SWV and DPV: integrated peak area vs concentration"})
                    chart.set_x_axis({"name": f"Concentration / {unit}"})
                    chart.set_y_axis({"name": "Integrated peak area / A V"})
                    ws.insert_chart("H2", chart, {"x_scale": 1.55, "y_scale": 1.35})
            self.status.setText(f"Excel exported: {p}")
        except Exception as e:
            QMessageBox.critical(self, "Excel export", str(e))


def main():
    app = QApplication.instance() or QApplication([])
    w = StrippingWindow(); w.show(); app.exec()


if __name__ == "__main__":
    main()

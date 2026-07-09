"""
SERS/Raman Voltammogram GUI v4.2.1

A small Windows desktop program for Raman/SERS voltammogram data.

CSV structure:
- First row: potentials
- First column: wavenumbers
- Remaining matrix: Raman intensities

Main features:
- graphical file selection
- wavenumber-range selection
- optional Savitzky-Golay smoothing
- optional moving-average smoothing
- raw / clip99 / log1p / clip99+log1p scaling
- Excel export
- interactive HTML export:
    - 3D surface
    - heatmap
    - contour plot
    - rotatable waterfall plot
    - raw vs smoothed comparison waterfall

Required packages:
    py -m pip install pandas numpy plotly openpyxl scipy
"""

from pathlib import Path
import sys
import subprocess
import re
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import pandas as pd


def ensure_package(import_name, pip_name=None):
    if pip_name is None:
        pip_name = import_name
    try:
        return __import__(import_name)
    except ModuleNotFoundError:
        answer = messagebox.askyesno(
            "Missing package",
            f"The package '{pip_name}' is missing.\n\nInstall it now?"
        )
        if answer:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
            return __import__(import_name)
        raise SystemExit(f"Missing package: {pip_name}")


def ensure_plotly():
    ensure_package("plotly", "plotly")
    import plotly.graph_objects as go
    return go


def ensure_scipy():
    ensure_package("scipy", "scipy")
    from scipy.signal import savgol_filter
    return savgol_filter


def safe_float_text(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if text == "":
        return np.nan
    text = text.replace(",", ".")
    return pd.to_numeric(text, errors="coerce")


def safe_name_number(x):
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", f"{x:g}")


def read_raman_csv(csv_path: Path):
    encodings_to_try = ["utf-8-sig", "cp1252", "latin1", "utf-16", "utf-16le", "utf-16be"]
    last_error = None

    for enc in encodings_to_try:
        try:
            df = pd.read_csv(
                csv_path,
                sep=None,
                engine="python",
                header=None,
                dtype=str,
                encoding=enc
            )
            encoding_used = enc
            break
        except UnicodeDecodeError as err:
            last_error = err
    else:
        raise UnicodeDecodeError(
            "unknown", b"", 0, 1,
            "The CSV file could not be decoded with common encodings."
        ) from last_error

    df = df.map(safe_float_text)
    df = df.dropna(how="all").dropna(axis=1, how="all")

    potentials = df.iloc[0, 1:].to_numpy(dtype=float)
    wavenumbers = df.iloc[1:, 0].to_numpy(dtype=float)
    intensities = df.iloc[1:, 1:].to_numpy(dtype=float)

    valid_pot = ~np.isnan(potentials)
    valid_wn = ~np.isnan(wavenumbers)

    potentials = potentials[valid_pot]
    wavenumbers = wavenumbers[valid_wn]
    intensities = intensities[valid_wn, :][:, valid_pot]

    if intensities.shape != (len(wavenumbers), len(potentials)):
        raise ValueError(
            "Data dimensions do not match:\n"
            f"Intensity matrix: {intensities.shape}\n"
            f"Number of wavenumbers: {len(wavenumbers)}\n"
            f"Number of potentials: {len(potentials)}"
        )

    # Sort axes
    pot_order = np.argsort(potentials)
    potentials = potentials[pot_order]
    intensities = intensities[:, pot_order]

    wn_order = np.argsort(wavenumbers)
    wavenumbers = wavenumbers[wn_order]
    intensities = intensities[wn_order, :]

    return potentials, wavenumbers, intensities, encoding_used


def moving_average_smoothing(intensities, window=7):
    if window < 3:
        return intensities.copy()
    if window % 2 == 0:
        window += 1

    kernel = np.ones(window, dtype=float) / window
    smoothed = np.empty_like(intensities, dtype=float)

    for j in range(intensities.shape[1]):
        smoothed[:, j] = np.convolve(intensities[:, j], kernel, mode="same")

    return smoothed


def savitzky_golay_smoothing(intensities, window_length=11, polyorder=3):
    savgol_filter = ensure_scipy()

    n = intensities.shape[0]

    if n < 5:
        return intensities.copy()

    # window must be odd and smaller/equal number of points
    window_length = int(window_length)
    polyorder = int(polyorder)

    if window_length % 2 == 0:
        window_length += 1

    if window_length > n:
        window_length = n if n % 2 == 1 else n - 1

    if window_length <= polyorder:
        window_length = polyorder + 2
        if window_length % 2 == 0:
            window_length += 1

    if window_length > n:
        window_length = n if n % 2 == 1 else n - 1

    if window_length < 5 or window_length <= polyorder:
        return intensities.copy()

    return savgol_filter(
        intensities,
        window_length=window_length,
        polyorder=polyorder,
        axis=0,
        mode="interp"
    )


def scale_intensity(intensities, scaling_mode):
    z = intensities.astype(float).copy()
    description = "Smoothed Raman intensity / a.u."
    suffix = "raw"

    if scaling_mode in {"clip99", "clip99_log1p"}:
        upper = np.nanpercentile(z, 99)
        z = np.clip(z, None, upper)
        description = "Smoothed intensity clipped at 99th percentile"
        suffix = "clip99"

    if scaling_mode in {"log1p", "clip99_log1p"}:
        z_min = np.nanmin(z)
        if z_min < 0:
            z = z - z_min
        z = np.log1p(z)
        if scaling_mode == "log1p":
            description = "log1p smoothed intensity"
            suffix = "log1p"
        else:
            description = "Smoothed intensity clipped at 99th percentile and log1p-scaled"
            suffix = "clip99_log1p"

    return z, description, suffix


def create_long_format_table(potentials, wavenumbers, intensities) -> pd.DataFrame:
    pot_grid, wn_grid = np.meshgrid(potentials, wavenumbers)
    return pd.DataFrame({
        "Wavenumber / cm^-1": wn_grid.ravel(),
        "Potential / V": pot_grid.ravel(),
        "Raman intensity / a.u.": intensities.ravel()
    })



def create_waterfall_tables(potentials, wavenumbers, values, offset):
    """
    Excel-ready waterfall data for Raman/SERS voltammograms.

    Waterfall_Shifted_Values contains the values used for a classical
    waterfall representation:
        shifted = plotted_value - minimum(plotted_spectrum) + spectrum_index * offset

    Waterfall_Unshifted_Values contains the plotted values before vertical shifting.
    Waterfall_Offsets documents subtracted minima and added offsets.
    """
    shifted = {"Wavenumber / cm^-1": wavenumbers}
    unshifted = {"Wavenumber / cm^-1": wavenumbers}
    offsets = []

    for j, pot in enumerate(potentials):
        y = values[:, j].astype(float)
        ymin = float(np.nanmin(y))
        vertical_offset = float(j * offset)
        col_name = f"{pot:g} V"

        unshifted[col_name] = y
        shifted[col_name] = y - ymin + vertical_offset

        offsets.append({
            "Column": col_name,
            "Potential / V": pot,
            "Spectrum index": j,
            "Minimum subtracted": ymin,
            "Vertical offset added": vertical_offset,
            "Formula": "shifted = plotted value - minimum + vertical offset"
        })

    return pd.DataFrame(shifted), pd.DataFrame(unshifted), pd.DataFrame(offsets)


def export_to_excel(csv_path, out_dir, potentials, wavenumbers,
                    raw_intensities, smoothed_intensities, plotted_intensities,
                    wn_start, wn_final, smoothing_description, scaling_description,
                    waterfall_shifted_df=None, waterfall_unshifted_df=None, waterfall_offsets_df=None):
    excel_path = out_dir / (
        f"{csv_path.stem}_wavenumber_{safe_name_number(wn_start)}_to_"
        f"{safe_name_number(wn_final)}_processed.xlsx"
    )

    raw_matrix_df = pd.DataFrame(raw_intensities, index=wavenumbers, columns=potentials)
    raw_matrix_df.index.name = "Wavenumber / cm^-1"
    raw_matrix_df.columns.name = "Potential / V"

    smoothed_matrix_df = pd.DataFrame(smoothed_intensities, index=wavenumbers, columns=potentials)
    smoothed_matrix_df.index.name = "Wavenumber / cm^-1"
    smoothed_matrix_df.columns.name = "Potential / V"

    plotted_matrix_df = pd.DataFrame(plotted_intensities, index=wavenumbers, columns=potentials)
    plotted_matrix_df.index.name = "Wavenumber / cm^-1"
    plotted_matrix_df.columns.name = "Potential / V"

    metadata_df = pd.DataFrame({
        "Property": [
            "Source file", "Selected wavenumber start / cm^-1", "Selected wavenumber final / cm^-1",
            "Potential minimum / V", "Potential maximum / V", "Number of potentials",
            "Number of selected wavenumbers", "Intensity matrix rows", "Intensity matrix columns",
            "Smoothing used", "Scaling used for plots"
        ],
        "Value": [
            str(csv_path), wn_start, wn_final,
            float(np.nanmin(potentials)), float(np.nanmax(potentials)),
            len(potentials), len(wavenumbers),
            raw_intensities.shape[0], raw_intensities.shape[1],
            smoothing_description, scaling_description
        ]
    })

    try:
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            metadata_df.to_excel(writer, sheet_name="Metadata", index=False)
            raw_matrix_df.to_excel(writer, sheet_name="Raw_Intensity_Matrix")
            smoothed_matrix_df.to_excel(writer, sheet_name="Smoothed_Intensity_Matrix")
            plotted_matrix_df.to_excel(writer, sheet_name="Plotted_Intensity_Matrix")
            create_long_format_table(potentials, wavenumbers, raw_intensities).to_excel(
                writer, sheet_name="Raw_Long_Format", index=False
            )
            create_long_format_table(potentials, wavenumbers, smoothed_intensities).to_excel(
                writer, sheet_name="Smoothed_Long_Format", index=False
            )
            create_long_format_table(potentials, wavenumbers, plotted_intensities).to_excel(
                writer, sheet_name="Plotted_Long_Format", index=False
            )
    except ModuleNotFoundError:
        ensure_package("openpyxl", "openpyxl")
        return export_to_excel(
            csv_path, out_dir, potentials, wavenumbers,
            raw_intensities, smoothed_intensities, plotted_intensities,
            wn_start, wn_final, smoothing_description, scaling_description
        )

    return excel_path



def wavenumber_axis_settings(wn_start, wn_final):
    start_tick = 100 * np.floor(min(wn_start, wn_final) / 100)
    return dict(title="Wavenumber / cm^-1", tickmode="linear", tick0=start_tick, dtick=100)


def intensity_axis_settings(z_label, z_range):
    settings = dict(title=z_label)
    if z_range is not None:
        settings["range"] = [z_range[0], z_range[1]]
    return settings


def intensity_2d_axis_settings(z_label, z_range):
    settings = dict(title=z_label)
    if z_range is not None:
        settings["range"] = [z_range[0], z_range[1]]
    return settings


def add_z_axis_buttons(fig, z_plot, z_range):
    zmin = float(np.nanmin(z_plot))
    zmax = float(np.nanmax(z_plot))
    manual = [float(z_range[0]), float(z_range[1])] if z_range is not None else [zmin, zmax]
    p99 = float(np.nanpercentile(z_plot, 99))
    p95 = float(np.nanpercentile(z_plot, 95))

    fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            direction="right",
            x=0.02,
            y=1.08,
            xanchor="left",
            yanchor="top",
            buttons=[
                dict(label="Auto Z", method="relayout", args=[{"scene.zaxis.autorange": True}]),
                dict(label="Manual Z", method="relayout", args=[{"scene.zaxis.range": manual, "scene.zaxis.autorange": False}]),
                dict(label="0 to 99%", method="relayout", args=[{"scene.zaxis.range": [max(0, zmin), p99], "scene.zaxis.autorange": False}]),
                dict(label="0 to 95%", method="relayout", args=[{"scene.zaxis.range": [max(0, zmin), p95], "scene.zaxis.autorange": False}]),
            ]
        )]
    )


def plot_surface(go, csv_path, out_dir, potentials, wavenumbers, z_plot, wn_start, wn_final, suffix, z_label, smooth_suffix, z_range=None, *args, **kwargs):
    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=potentials, y=wavenumbers, z=z_plot,
        colorscale="Turbo",
        colorbar=dict(title=z_label),
        contours={"z": {"show": True, "usecolormap": True, "project_z": True}}
    ))
    fig.update_layout(
        title=f"Interactive 3D Raman Voltammogram ({wn_start:g}-{wn_final:g} cm^-1)",
        scene=dict(
            xaxis=dict(title="Potential / V"),
            yaxis=wavenumber_axis_settings(wn_start, wn_final),
            zaxis=intensity_axis_settings(z_label, z_range),
            camera=dict(eye=dict(x=1.55, y=-1.75, z=1.10)),
            aspectmode="manual",
            aspectratio=dict(x=1.4, y=1.1, z=0.75)
        ),
        width=1150, height=820, margin=dict(l=0, r=0, b=0, t=55)
    )
    html_path = out_dir / (
        f"{csv_path.stem}_wavenumber_{safe_name_number(wn_start)}_to_"
        f"{safe_name_number(wn_final)}_surface_{smooth_suffix}_{suffix}.html"
    )
    add_z_axis_buttons(fig, z_plot, z_range)
    fig.write_html(html_path, include_plotlyjs="cdn")
    return fig, html_path


def plot_heatmap(go, csv_path, out_dir, potentials, wavenumbers, z_plot, wn_start, wn_final, suffix, z_label, smooth_suffix, z_range=None, *args, **kwargs):
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        x=potentials, y=wavenumbers, z=z_plot,
        colorscale="Turbo",
        colorbar=dict(title=z_label)
    ))
    fig.update_layout(
        title=f"Raman Voltammogram Heatmap ({wn_start:g}-{wn_final:g} cm^-1)",
        xaxis_title="Potential / V", yaxis=wavenumber_axis_settings(wn_start, wn_final),
        width=1050, height=760, margin=dict(l=70, r=20, b=60, t=70)
    )
    html_path = out_dir / (
        f"{csv_path.stem}_wavenumber_{safe_name_number(wn_start)}_to_"
        f"{safe_name_number(wn_final)}_heatmap_{smooth_suffix}_{suffix}.html"
    )
    fig.write_html(html_path, include_plotlyjs="cdn")
    return fig, html_path


def plot_contour(go, csv_path, out_dir, potentials, wavenumbers, z_plot, wn_start, wn_final, suffix, z_label, smooth_suffix, z_range=None, *args, **kwargs):
    fig = go.Figure()
    fig.add_trace(go.Contour(
        x=potentials, y=wavenumbers, z=z_plot,
        colorscale="Turbo",
        contours=dict(coloring="heatmap", showlabels=True, labelfont=dict(size=10)),
        colorbar=dict(title=z_label)
    ))
    fig.update_layout(
        title=f"Raman Voltammogram Contour Plot ({wn_start:g}-{wn_final:g} cm^-1)",
        xaxis_title="Potential / V", yaxis=wavenumber_axis_settings(wn_start, wn_final),
        width=1050, height=760, margin=dict(l=70, r=20, b=60, t=70)
    )
    html_path = out_dir / (
        f"{csv_path.stem}_wavenumber_{safe_name_number(wn_start)}_to_"
        f"{safe_name_number(wn_final)}_contour_{smooth_suffix}_{suffix}.html"
    )
    fig.write_html(html_path, include_plotlyjs="cdn")
    return fig, html_path


def plot_waterfall(go, csv_path, out_dir, potentials, wavenumbers, z_plot, wn_start, wn_final, suffix, z_label, smooth_suffix, z_range=None, waterfall_offset=0.0, *args, **kwargs):
    fig = go.Figure()
    max_lines = 120
    if len(potentials) > max_lines:
        indices = np.linspace(0, len(potentials) - 1, max_lines, dtype=int)
    else:
        indices = np.arange(len(potentials))

    for idx in indices:
        pot = potentials[idx]
        fig.add_trace(go.Scatter3d(
            x=wavenumbers,
            y=np.full_like(wavenumbers, pot, dtype=float),
            z=z_plot[:, idx] - np.nanmin(z_plot[:, idx]) + idx * waterfall_offset,
            mode="lines",
            line=dict(width=3),
            name=f"{pot:g} V",
            showlegend=False,
            hovertemplate=(
                "Wavenumber: %{x:.2f} cm^-1<br>"
                "Potential: %{y:.4g} V<br>"
                "Intensity: %{z:.4g}<extra></extra>"
            )
        ))

    fig.update_layout(
        title=f"Rotatable Raman Waterfall Plot ({wn_start:g}-{wn_final:g} cm^-1)",
        scene=dict(
            xaxis=wavenumber_axis_settings(wn_start, wn_final),
            yaxis=dict(title="Potential / V"),
            zaxis=intensity_axis_settings(z_label, z_range),
            camera=dict(eye=dict(x=1.55, y=-1.9, z=1.1)),
            aspectmode="manual",
            aspectratio=dict(x=1.7, y=1.1, z=0.85)
        ),
        width=1150, height=820, margin=dict(l=0, r=0, b=0, t=55)
    )
    html_path = out_dir / (
        f"{csv_path.stem}_wavenumber_{safe_name_number(wn_start)}_to_"
        f"{safe_name_number(wn_final)}_waterfall_{smooth_suffix}_{suffix}.html"
    )
    add_z_axis_buttons(fig, z_plot, z_range)
    fig.write_html(html_path, include_plotlyjs="cdn")
    return fig, html_path


def plot_comparison_waterfall(go, csv_path, out_dir, potentials, wavenumbers,
                              raw_intensities, smoothed_intensities,
                              wn_start, wn_final, smooth_suffix, z_range=None, *args, **kwargs):
    """
    Comparison plot with a small potential-axis offset:
    grey = raw, blue = smoothed. This makes the smoothing effect visible.
    """
    fig = go.Figure()

    max_lines = 50
    if len(potentials) > max_lines:
        indices = np.linspace(0, len(potentials) - 1, max_lines, dtype=int)
    else:
        indices = np.arange(len(potentials))

    if len(potentials) > 1:
        dy = 0.18 * np.nanmedian(np.abs(np.diff(np.sort(potentials))))
        if not np.isfinite(dy) or dy == 0:
            dy = 0.01
    else:
        dy = 0.01

    for idx in indices:
        pot = potentials[idx]

        fig.add_trace(go.Scatter3d(
            x=wavenumbers,
            y=np.full_like(wavenumbers, pot - dy, dtype=float),
            z=raw_intensities[:, idx],
            mode="lines",
            line=dict(width=2, color="rgba(120,120,120,0.45)"),
            name=f"Raw {pot:g} V",
            showlegend=False,
            hovertemplate="Raw<br>Wavenumber: %{x:.2f}<br>Potential: %{y:.4g}<br>Intensity: %{z:.4g}<extra></extra>"
        ))

        fig.add_trace(go.Scatter3d(
            x=wavenumbers,
            y=np.full_like(wavenumbers, pot + dy, dtype=float),
            z=smoothed_intensities[:, idx],
            mode="lines",
            line=dict(width=5, color="rgba(0,60,220,0.95)"),
            name=f"Smoothed {pot:g} V",
            showlegend=False,
            hovertemplate="Smoothed<br>Wavenumber: %{x:.2f}<br>Potential: %{y:.4g}<br>Intensity: %{z:.4g}<extra></extra>"
        ))

    fig.update_layout(
        title=f"Raw vs Smoothed Raman Waterfall ({wn_start:g}-{wn_final:g} cm^-1)",
        scene=dict(
            xaxis=wavenumber_axis_settings(wn_start, wn_final),
            yaxis=dict(title="Potential / V; grey=raw, blue=smoothed"),
            zaxis=intensity_axis_settings("Raman intensity / a.u.", z_range),
            camera=dict(eye=dict(x=1.55, y=-1.9, z=1.1)),
            aspectmode="manual",
            aspectratio=dict(x=1.7, y=1.1, z=0.85)
        ),
        width=1150, height=820, margin=dict(l=0, r=0, b=0, t=55)
    )

    html_path = out_dir / (
        f"{csv_path.stem}_wavenumber_{safe_name_number(wn_start)}_to_"
        f"{safe_name_number(wn_final)}_raw_vs_smoothed_waterfall_{smooth_suffix}.html"
    )
    add_z_axis_buttons(fig, smoothed_intensities, z_range)
    fig.write_html(html_path, include_plotlyjs="cdn")
    return fig, html_path


def plot_middle_spectrum_comparison(go, csv_path, out_dir, potentials, wavenumbers,
                                    raw_intensities, smoothed_intensities,
                                    wn_start, wn_final, smooth_suffix, z_range=None, *args, **kwargs):
    """2D control plot for one representative spectrum."""
    idx = len(potentials) // 2
    pot = potentials[idx]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=wavenumbers,
        y=raw_intensities[:, idx],
        mode="lines",
        name="Raw spectrum",
        line=dict(width=1, color="rgba(120,120,120,0.65)")
    ))
    fig.add_trace(go.Scatter(
        x=wavenumbers,
        y=smoothed_intensities[:, idx],
        mode="lines",
        name="Smoothed spectrum",
        line=dict(width=3, color="rgba(0,60,220,0.95)")
    ))

    fig.update_layout(
        title=f"Raw vs Smoothed Single Raman Spectrum at {pot:g} V",
        xaxis=wavenumber_axis_settings(wn_start, wn_final),
        yaxis=intensity_2d_axis_settings("Raman intensity / a.u.", z_range),
        width=1050,
        height=650,
        margin=dict(l=70, r=30, b=60, t=70)
    )

    html_path = out_dir / (
        f"{csv_path.stem}_wavenumber_{safe_name_number(wn_start)}_to_"
        f"{safe_name_number(wn_final)}_single_spectrum_raw_vs_smoothed_{smooth_suffix}.html"
    )
    fig.write_html(html_path, include_plotlyjs="cdn")
    return fig, html_path


class RamanGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SERS/Raman Voltammogram GUI v4.2.1")
        self.root.geometry("760x720")

        self.csv_path = tk.StringVar()
        self.output_dir = tk.StringVar()

        self.wn_start = tk.StringVar()
        self.wn_final = tk.StringVar()

        self.smoothing_method = tk.StringVar(value="savgol")
        self.savgol_window = tk.StringVar(value="31")
        self.savgol_poly = tk.StringVar(value="3")
        self.moving_average_window = tk.StringVar(value="7")

        self.scaling = tk.StringVar(value="raw")

        self.auto_intensity_axis = tk.BooleanVar(value=True)
        self.intensity_min = tk.StringVar(value="")
        self.intensity_max = tk.StringVar(value="")

        self.create_surface = tk.BooleanVar(value=True)
        self.create_heatmap = tk.BooleanVar(value=True)
        self.create_contour = tk.BooleanVar(value=True)
        self.create_waterfall = tk.BooleanVar(value=True)
        self.create_comparison = tk.BooleanVar(value=True)
        self.create_single_comparison = tk.BooleanVar(value=True)
        self.waterfall_offset = tk.StringVar(value="100")

        self.create_widgets()

    def create_widgets(self):
        pad = {"padx": 8, "pady": 5}

        frm_file = ttk.LabelFrame(self.root, text="Input and output")
        frm_file.pack(fill="x", **pad)

        ttk.Label(frm_file, text="CSV file:").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(frm_file, textvariable=self.csv_path, width=72).grid(row=0, column=1, **pad)
        ttk.Button(frm_file, text="Browse", command=self.browse_csv).grid(row=0, column=2, **pad)

        ttk.Label(frm_file, text="Output folder:").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(frm_file, textvariable=self.output_dir, width=72).grid(row=1, column=1, **pad)
        ttk.Button(frm_file, text="Browse", command=self.browse_output_dir).grid(row=1, column=2, **pad)

        frm_range = ttk.LabelFrame(self.root, text="Wavenumber range")
        frm_range.pack(fill="x", **pad)

        ttk.Label(frm_range, text="Start / cm^-1:").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(frm_range, textvariable=self.wn_start, width=16).grid(row=0, column=1, sticky="w", **pad)
        ttk.Label(frm_range, text="Final / cm^-1:").grid(row=0, column=2, sticky="w", **pad)
        ttk.Entry(frm_range, textvariable=self.wn_final, width=16).grid(row=0, column=3, sticky="w", **pad)
        ttk.Button(frm_range, text="Read CSV range", command=self.read_range_from_csv).grid(row=0, column=4, **pad)

        frm_smooth = ttk.LabelFrame(self.root, text="Spectral smoothing")
        frm_smooth.pack(fill="x", **pad)

        ttk.Radiobutton(frm_smooth, text="No smoothing", variable=self.smoothing_method, value="none").grid(row=0, column=0, sticky="w", **pad)
        ttk.Radiobutton(frm_smooth, text="Moving average", variable=self.smoothing_method, value="moving").grid(row=1, column=0, sticky="w", **pad)
        ttk.Label(frm_smooth, text="Window:").grid(row=1, column=1, sticky="e", **pad)
        ttk.Entry(frm_smooth, textvariable=self.moving_average_window, width=8).grid(row=1, column=2, sticky="w", **pad)

        ttk.Radiobutton(frm_smooth, text="Savitzky-Golay, stronger smoothing", variable=self.smoothing_method, value="savgol").grid(row=2, column=0, sticky="w", **pad)
        ttk.Label(frm_smooth, text="Window:").grid(row=2, column=1, sticky="e", **pad)
        ttk.Entry(frm_smooth, textvariable=self.savgol_window, width=8).grid(row=2, column=2, sticky="w", **pad)
        ttk.Label(frm_smooth, text="Polynomial order:").grid(row=2, column=3, sticky="e", **pad)
        ttk.Entry(frm_smooth, textvariable=self.savgol_poly, width=8).grid(row=2, column=4, sticky="w", **pad)

        frm_scaling = ttk.LabelFrame(self.root, text="Intensity scaling for plots")
        frm_scaling.pack(fill="x", **pad)

        ttk.Radiobutton(frm_scaling, text="Raw", variable=self.scaling, value="raw").grid(row=0, column=0, sticky="w", **pad)
        ttk.Radiobutton(frm_scaling, text="Clip at 99th percentile", variable=self.scaling, value="clip99").grid(row=0, column=1, sticky="w", **pad)
        ttk.Radiobutton(frm_scaling, text="log1p", variable=self.scaling, value="log1p").grid(row=0, column=2, sticky="w", **pad)
        ttk.Radiobutton(frm_scaling, text="Clip 99% + log1p", variable=self.scaling, value="clip99_log1p").grid(row=0, column=3, sticky="w", **pad)

        frm_intensity = ttk.LabelFrame(self.root, text="Intensity axis")
        frm_intensity.pack(fill="x", **pad)

        ttk.Checkbutton(frm_intensity, text="Automatic intensity axis", variable=self.auto_intensity_axis).grid(row=0, column=0, sticky="w", **pad)
        ttk.Label(frm_intensity, text="Minimum:").grid(row=0, column=1, sticky="e", **pad)
        ttk.Entry(frm_intensity, textvariable=self.intensity_min, width=12).grid(row=0, column=2, sticky="w", **pad)
        ttk.Label(frm_intensity, text="Maximum:").grid(row=0, column=3, sticky="e", **pad)
        ttk.Entry(frm_intensity, textvariable=self.intensity_max, width=12).grid(row=0, column=4, sticky="w", **pad)

        ttk.Label(frm_intensity, text="Waterfall vertical offset:").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(frm_intensity, textvariable=self.waterfall_offset, width=12).grid(row=1, column=1, sticky="w", **pad)

        frm_plots = ttk.LabelFrame(self.root, text="Plots to create")
        frm_plots.pack(fill="x", **pad)

        ttk.Checkbutton(frm_plots, text="3D Surface", variable=self.create_surface).grid(row=0, column=0, sticky="w", **pad)
        ttk.Checkbutton(frm_plots, text="Heatmap", variable=self.create_heatmap).grid(row=0, column=1, sticky="w", **pad)
        ttk.Checkbutton(frm_plots, text="Contour", variable=self.create_contour).grid(row=0, column=2, sticky="w", **pad)
        ttk.Checkbutton(frm_plots, text="Waterfall", variable=self.create_waterfall).grid(row=0, column=3, sticky="w", **pad)
        ttk.Checkbutton(frm_plots, text="Raw vs smoothed waterfall", variable=self.create_comparison).grid(row=0, column=4, sticky="w", **pad)
        ttk.Checkbutton(frm_plots, text="Raw vs smoothed single spectrum", variable=self.create_single_comparison).grid(row=1, column=0, sticky="w", **pad)

        frm_run = ttk.Frame(self.root)
        frm_run.pack(fill="x", **pad)
        ttk.Button(frm_run, text="Create Excel and HTML files", command=self.run_analysis).pack(side="left", padx=8, pady=10)

        self.log = tk.Text(self.root, height=16, wrap="word")
        self.log.pack(fill="both", expand=True, padx=8, pady=8)

    def write_log(self, text):
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.root.update_idletasks()

    def browse_csv(self):
        filename = filedialog.askopenfilename(
            title="Select Raman/SERS CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.csv_path.set(filename)
            p = Path(filename)
            self.output_dir.set(str(p.parent))
            try:
                self.read_range_from_csv()
            except Exception as e:
                self.write_log(f"Could not read range automatically: {e}")

    def browse_output_dir(self):
        dirname = filedialog.askdirectory(title="Select output folder")
        if dirname:
            self.output_dir.set(dirname)

    def read_range_from_csv(self):
        csv = Path(self.csv_path.get().strip().strip('"'))
        if not csv.exists():
            messagebox.showerror("Error", "Please select a valid CSV file first.")
            return

        potentials, wavenumbers, intensities, encoding = read_raman_csv(csv)
        self.wn_start.set(f"{float(np.nanmin(wavenumbers)):g}")
        self.wn_final.set(f"{float(np.nanmax(wavenumbers)):g}")
        self.write_log(f"CSV loaded for range check: {csv.name}")
        self.write_log(f"Encoding: {encoding}")
        self.write_log(f"Available wavenumber range: {self.wn_start.get()} to {self.wn_final.get()} cm^-1")
        self.write_log(f"Potential range: {float(np.nanmin(potentials)):g} to {float(np.nanmax(potentials)):g} V")
        self.write_log(f"Matrix: {intensities.shape[0]} wavenumbers x {intensities.shape[1]} potentials")

    def apply_smoothing(self, raw):
        method = self.smoothing_method.get()

        if method == "none":
            return raw.copy(), "No smoothing", "nosmooth"

        if method == "moving":
            window = int(self.moving_average_window.get())
            return moving_average_smoothing(raw, window), f"Moving average smoothing, {window} points", f"movavg{window}"

        window = int(self.savgol_window.get())
        poly = int(self.savgol_poly.get())
        return savitzky_golay_smoothing(raw, window, poly), f"Savitzky-Golay smoothing, {window} points, polynomial order {poly}", f"savgol{window}p{poly}"

    def run_analysis(self):
        try:
            csv = Path(self.csv_path.get().strip().strip('"'))
            if not csv.exists():
                messagebox.showerror("Error", "Please select a valid CSV file.")
                return

            out_dir = Path(self.output_dir.get().strip().strip('"')) if self.output_dir.get().strip() else csv.parent
            out_dir.mkdir(parents=True, exist_ok=True)

            wn_start = float(self.wn_start.get().replace(",", "."))
            wn_final = float(self.wn_final.get().replace(",", "."))
            if wn_start > wn_final:
                wn_start, wn_final = wn_final, wn_start

            self.write_log("Reading CSV file ...")
            potentials, wavenumbers, intensities, encoding = read_raman_csv(csv)
            self.write_log(f"Encoding used: {encoding}")

            mask = (wavenumbers >= wn_start) & (wavenumbers <= wn_final)
            if not np.any(mask):
                messagebox.showerror("Error", "No data points in selected wavenumber range.")
                return

            wavenumbers_selected = wavenumbers[mask]
            raw_selected = intensities[mask, :]

            self.write_log("Applying smoothing ...")
            smoothed, smoothing_description, smooth_suffix = self.apply_smoothing(raw_selected)
            self.write_log(f"Smoothing: {smoothing_description}")

            z_plot, z_label, scaling_suffix = scale_intensity(smoothed, self.scaling.get())
            self.write_log(f"Scaling: {z_label}")

            waterfall_offset = float(self.waterfall_offset.get().strip().replace(",", ".")) if self.waterfall_offset.get().strip() else 0.0
            self.write_log(f"Waterfall vertical offset: {waterfall_offset:g}")

            waterfall_shifted_df, waterfall_unshifted_df, waterfall_offsets_df = create_waterfall_tables(
                potentials, wavenumbers_selected, z_plot, waterfall_offset
            )

            z_range = None
            if not self.auto_intensity_axis.get():
                zmin_text = self.intensity_min.get().strip().replace(",", ".")
                zmax_text = self.intensity_max.get().strip().replace(",", ".")
                if zmin_text and zmax_text:
                    zmin = float(zmin_text)
                    zmax = float(zmax_text)
                    if zmin > zmax:
                        zmin, zmax = zmax, zmin
                    z_range = (zmin, zmax)
                    self.write_log(f"Manual intensity axis: {zmin:g} to {zmax:g}")
                else:
                    self.write_log("Manual intensity axis selected, but min/max are incomplete. Automatic axis will be used.")

            self.write_log("Exporting Excel file ...")
            excel_path = export_to_excel(
                csv, out_dir, potentials, wavenumbers_selected,
                raw_selected, smoothed, z_plot,
                wn_start, wn_final, smoothing_description, z_label,
                waterfall_shifted_df, waterfall_unshifted_df, waterfall_offsets_df
            )
            self.write_log(f"Excel: {excel_path}")

            go = ensure_plotly()
            created = []

            if self.create_surface.get():
                self.write_log("Creating 3D surface ...")
                created.append(("3D Surface", *plot_surface(
                    go, csv, out_dir, potentials, wavenumbers_selected, z_plot,
                    wn_start, wn_final, scaling_suffix, z_label, smooth_suffix, z_range, waterfall_offset
                )))

            if self.create_heatmap.get():
                self.write_log("Creating heatmap ...")
                created.append(("Heatmap", *plot_heatmap(
                    go, csv, out_dir, potentials, wavenumbers_selected, z_plot,
                    wn_start, wn_final, scaling_suffix, z_label, smooth_suffix, z_range
                )))

            if self.create_contour.get():
                self.write_log("Creating contour plot ...")
                created.append(("Contour", *plot_contour(
                    go, csv, out_dir, potentials, wavenumbers_selected, z_plot,
                    wn_start, wn_final, scaling_suffix, z_label, smooth_suffix, z_range
                )))

            if self.create_waterfall.get():
                self.write_log("Creating waterfall plot ...")
                created.append(("Waterfall", *plot_waterfall(
                    go, csv, out_dir, potentials, wavenumbers_selected, z_plot,
                    wn_start, wn_final, scaling_suffix, z_label, smooth_suffix, z_range
                )))

            if self.create_comparison.get():
                self.write_log("Creating raw-vs-smoothed waterfall plot ...")
                created.append(("Raw vs Smoothed Waterfall", *plot_comparison_waterfall(
                    go, csv, out_dir, potentials, wavenumbers_selected,
                    raw_selected, smoothed, wn_start, wn_final, smooth_suffix, z_range
                )))

            if self.create_single_comparison.get():
                self.write_log("Creating raw-vs-smoothed single-spectrum control plot ...")
                created.append(("Raw vs Smoothed Single Spectrum", *plot_middle_spectrum_comparison(
                    go, csv, out_dir, potentials, wavenumbers_selected,
                    raw_selected, smoothed, wn_start, wn_final, smooth_suffix, z_range
                )))

            for name, fig, path in created:
                self.write_log(f"{name}: {path}")

            self.write_log("Finished.")
            messagebox.showinfo("Finished", f"Files created in:\n{out_dir}")

            if created:
                created[0][1].show()

        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.write_log(f"ERROR: {e}")


def main():
    root = tk.Tk()
    app = RamanGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

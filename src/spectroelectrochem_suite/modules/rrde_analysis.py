#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RRDE-Analysis GUI – Version 5.4.7 Preview

CSV-Struktur:
    Spalte 1: Potential
    danach je Rotation rate zwei Spalten:
        Disk current, Ring current

Erzeugt einen Ergebnisordner mit:
    - Excel-Report
    - Raw data-CSV
    - smoothed Daten
    - 2D-Grafiken
    - getrennten 3D-HTML-Darstellungen
    - gemeinsamer 3D-HTML-Darstellung mit automatisch sichtbarem Ring current
"""

from __future__ import annotations

import csv
import os
import re
import subprocess
import sys
import traceback
import webbrowser
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter
import plotly.graph_objects as go
from plotly.subplots import make_subplots


APP_TITLE = "SpectroElectroChem Suite – RRDE Analysis"
VERSION = "5.4.7-preview"


@dataclass
class RRDEData:
    potential: np.ndarray
    rotations: List[float]
    disk: List[np.ndarray]
    ring: List[np.ndarray]
    potential_label: str


def natural_float(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip().replace(",", ".")
    if not text or text.lower().startswith("unnamed"):
        return None
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text)
    return float(match.group(0)) if match else None


def detect_delimiter(path: Path) -> str:
    sample = path.read_text(encoding="utf-8-sig", errors="replace")[:12000]
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t").delimiter
    except csv.Error:
        return max([",", ";", "\t"], key=sample.count)


def read_csv_robust(path: Path) -> pd.DataFrame:
    delimiter = detect_delimiter(path)
    errors = []
    for decimal in (".", ","):
        try:
            df = pd.read_csv(
                path,
                sep=delimiter,
                decimal=decimal,
                encoding="utf-8-sig",
                engine="python",
                skip_blank_lines=True,
            )
            df = df.dropna(axis=1, how="all")
            if df.shape[1] >= 3:
                return df
        except Exception as exc:
            errors.append(str(exc))
    raise ValueError("CSV file konnte nicht gelesen werden: " + " | ".join(errors))


def parse_rrde_csv(path: Path) -> RRDEData:
    df = read_csv_robust(path)
    potential_label = str(df.columns[0]).strip() or "Potential / V"
    potential_all = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(potential_all)

    if valid.sum() < 3:
        raise ValueError("Die erste Spalte enthält keine ausreichenden potential values.")

    potential = potential_all[valid]
    rotations, disk, ring = [], [], []

    for col in range(1, df.shape[1] - 1, 2):
        rpm = natural_float(df.columns[col])
        if rpm is None:
            rpm = float(len(rotations) + 1)

        d = pd.to_numeric(df.iloc[:, col], errors="coerce").to_numpy(dtype=float)[valid]
        r = pd.to_numeric(df.iloc[:, col + 1], errors="coerce").to_numpy(dtype=float)[valid]

        if np.isfinite(d).sum() >= 3 and np.isfinite(r).sum() >= 3:
            rotations.append(float(rpm))
            disk.append(d)
            ring.append(r)

    if not rotations:
        raise ValueError(
            "Keine Disk-/Ring-Spaltenpaare erkannt. Nach der Potentialspalte "
            "müssen je Rotation rate zwei Spalten folgen."
        )

    order = np.argsort(rotations)
    return RRDEData(
        potential=potential,
        rotations=[rotations[i] for i in order],
        disk=[disk[i] for i in order],
        ring=[ring[i] for i in order],
        potential_label=potential_label,
    )


def parse_rrde_manual(
    path: Path,
    potential_column: str,
    pairs: Sequence[Tuple[float, str, str]],
) -> RRDEData:
    """Liest RRDE-Daten ausschließlich anhand einer bestätigten manuellen Zuordnung."""
    df = read_csv_robust(path)

    if potential_column not in df.columns:
        raise ValueError(f"Die gewählte Potentialspalte '{potential_column}' wurde nicht gefunden.")

    potential_all = pd.to_numeric(df[potential_column], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(potential_all)
    if valid.sum() < 3:
        raise ValueError("Die gewählte Potentialspalte enthält zu wenige numerische Werte.")

    if not pairs:
        raise ValueError("Es wurde kein Disk-/Ring-Spaltenpaar zugeordnet.")

    potential = potential_all[valid]
    rotations: List[float] = []
    disk: List[np.ndarray] = []
    ring: List[np.ndarray] = []

    used_columns = {potential_column}
    for rpm, disk_column, ring_column in pairs:
        if disk_column not in df.columns:
            raise ValueError(f"Die gewählte Diskspalte '{disk_column}' wurde nicht gefunden.")
        if ring_column not in df.columns:
            raise ValueError(f"Die gewählte Ringspalte '{ring_column}' wurde nicht gefunden.")
        if disk_column == ring_column:
            raise ValueError(f"Disk- und Ringspalte dürfen bei {rpm:g} U/min nicht identisch sein.")
        if disk_column in used_columns or ring_column in used_columns:
            raise ValueError(
                f"Eine Spalte wurde mehrfach zugeordnet: {disk_column} oder {ring_column}."
            )
        used_columns.add(disk_column)
        used_columns.add(ring_column)

        d = pd.to_numeric(df[disk_column], errors="coerce").to_numpy(dtype=float)[valid]
        r = pd.to_numeric(df[ring_column], errors="coerce").to_numpy(dtype=float)[valid]

        if np.isfinite(d).sum() < 3:
            raise ValueError(f"Diskspalte '{disk_column}' enthält zu wenige numerische Werte.")
        if np.isfinite(r).sum() < 3:
            raise ValueError(f"Ringspalte '{ring_column}' enthält zu wenige numerische Werte.")

        rotations.append(float(rpm))
        disk.append(d)
        ring.append(r)

    order = np.argsort(rotations)
    return RRDEData(
        potential=potential,
        rotations=[rotations[i] for i in order],
        disk=[disk[i] for i in order],
        ring=[ring[i] for i in order],
        potential_label=str(potential_column),
    )


def interpolate_nans(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    good = np.isfinite(values)
    if good.sum() < 2:
        return values.copy()
    result = values.copy()
    x = np.arange(len(values))
    result[~good] = np.interp(x[~good], x[good], values[good])
    return result


def adjusted_savgol(n: int, window: int, poly: int) -> Tuple[int, int]:
    window = max(3, int(window))
    if window % 2 == 0:
        window += 1
    if window > n:
        window = n if n % 2 else n - 1
    poly = max(1, min(int(poly), window - 1))
    return window, poly


def smooth(values: np.ndarray, enabled: bool, window: int, poly: int) -> np.ndarray:
    values = interpolate_nans(values)
    if not enabled:
        return values
    w, p = adjusted_savgol(len(values), window, poly)
    return savgol_filter(values, w, p, mode="interp")


def format_rpm(rpm: float) -> str:
    return str(int(round(rpm))) if abs(rpm - round(rpm)) < 1e-9 else f"{rpm:g}"


def safe_filename(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9ÄÖÜäöüß._-]+", "_", text.strip())
    return text.strip("_.") or "RRDE_Messung"


def open_file(path: Path) -> None:
    if sys.platform.startswith("win"):
        os.startfile(str(path))  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


def choose_ring_display_scale(
    disk_curves: Sequence[np.ndarray],
    ring_curves: Sequence[np.ndarray],
) -> float:
    """Automatische lineare Skalierung nur für die gemeinsame 3D-Darstellung."""
    disk_abs = np.concatenate([np.abs(np.asarray(v, dtype=float)) for v in disk_curves])
    ring_abs = np.concatenate([np.abs(np.asarray(v, dtype=float)) for v in ring_curves])

    disk_level = np.nanpercentile(disk_abs, 95)
    ring_level = np.nanpercentile(ring_abs, 95)

    if not np.isfinite(ring_level) or ring_level <= 0:
        return 1.0
    factor = disk_level / ring_level if disk_level > 0 else 1.0

    # Auf eine gut lesbare 1-2-5-Dekade runden.
    exponent = np.floor(np.log10(factor))
    mantissa = factor / (10 ** exponent)
    rounded = 1 if mantissa < 1.5 else 2 if mantissa < 3.5 else 5 if mantissa < 7.5 else 10
    return float(max(1.0, rounded * 10 ** exponent))


def compensate_ring_background(
    potential: np.ndarray,
    curves: Sequence[np.ndarray],
    enabled: bool,
    method: str,
    potential_from: float,
    potential_to: float,
    manual_offset: float = 0.0,
) -> Tuple[List[np.ndarray], List[float]]:
    """Kompensiert den konstanten Untergrund der Ringströme.

    method == "range_mean":
        Für jede Kurve wird ihr eigener Mittelwert im gewählten
        Potentialbereich abgezogen.

    method == "manual":
        Derselbe manuell eingegebene Offset wird von allen Kurven abgezogen.

    Alle Werte müssen bereits in derselben Anzeigeeinheit vorliegen.
    """
    arrays = [np.asarray(v, dtype=float).copy() for v in curves]
    if not enabled:
        return arrays, [0.0] * len(arrays)

    if method == "manual":
        if not np.isfinite(manual_offset):
            raise ValueError("Der manuelle Ring current-Offset ist keine gültige Zahl.")
        offsets = [float(manual_offset)] * len(arrays)
        return [v - manual_offset for v in arrays], offsets

    lo, hi = sorted((float(potential_from), float(potential_to)))
    mask = np.isfinite(potential) & (potential >= lo) & (potential <= hi)
    if mask.sum() < 2:
        raise ValueError(
            "Im gewählten Potentialbereich wurden weniger als zwei Messpunkte gefunden. "
            "Bitte den Bereich für die Ring-Untergrundkompensation anpassen."
        )

    compensated, offsets = [], []
    for curve in arrays:
        offset = float(np.nanmean(curve[mask]))
        if not np.isfinite(offset):
            raise ValueError("Der Ring current-Untergrund konnte nicht bestimmt werden.")
        compensated.append(curve - offset)
        offsets.append(offset)
    return compensated, offsets


def subtract_rrde_background_measurement(
    sample: RRDEData,
    background: RRDEData,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Subtract a complete N2/background RRDE measurement from an O2 measurement.

    Disk and ring currents are matched by rotation rate and subtracted point by
    point. To avoid mixing forward and reverse scan branches, the potential grids
    must have the same length and agree within 1 mV.
    """
    sample_p = np.asarray(sample.potential, dtype=float)
    bg_p = np.asarray(background.potential, dtype=float)
    if sample_p.shape != bg_p.shape or not np.allclose(sample_p, bg_p, atol=1e-3, rtol=0.0, equal_nan=True):
        raise ValueError(
            "Die Hintergrundmessung besitzt nicht dasselbe Potentialraster wie die O2-Messung. "
            "Bitte dieselben Scanparameter, dieselbe Punktzahl und dieselbe Scanrichtung verwenden."
        )

    bg_rot = np.asarray(background.rotations, dtype=float)
    disk_corrected: List[np.ndarray] = []
    ring_corrected: List[np.ndarray] = []
    used = set()
    for rpm, disk, ring in zip(sample.rotations, sample.disk, sample.ring):
        if bg_rot.size == 0:
            raise ValueError("Die Hintergrundmessung enthält keine Rotationskurven.")
        idx = int(np.argmin(np.abs(bg_rot - float(rpm))))
        tolerance = max(1.0, 0.01 * abs(float(rpm)))
        if abs(float(bg_rot[idx]) - float(rpm)) > tolerance:
            raise ValueError(
                f"Für {format_rpm(rpm)} U/min wurde in der Hintergrundmessung keine passende Rotation gefunden."
            )
        if idx in used:
            raise ValueError("Eine Rotationskurve der Hintergrundmessung würde mehrfach verwendet.")
        used.add(idx)
        disk_bg = np.asarray(background.disk[idx], dtype=float)
        ring_bg = np.asarray(background.ring[idx], dtype=float)
        disk_corrected.append(np.asarray(disk, dtype=float) - disk_bg)
        ring_corrected.append(np.asarray(ring, dtype=float) - ring_bg)
    return disk_corrected, ring_corrected


def make_2d_plot(
    potential, rotations, raw_curves, smoothed_curves,
    title, ylabel, png_path, pdf_path, show_raw, reverse_x
):
    fig, ax = plt.subplots(figsize=(10.5, 7.0), constrained_layout=True)
    for rpm, raw, sm in zip(rotations, raw_curves, smoothed_curves):
        if show_raw:
            ax.plot(potential, raw, linewidth=0.7, alpha=0.25)
        ax.plot(potential, sm, linewidth=1.6, label=f"{format_rpm(rpm)} U/min")
    ax.set_title(title)
    ax.set_xlabel("Potential / V")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(title="Rotation", fontsize=8, ncol=2)
    if reverse_x:
        ax.invert_xaxis()
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)


def make_2d_html(
    potential, rotations, raw_curves, smoothed_curves,
    title, ylabel, html_path, show_raw, reverse_x
):
    fig = go.Figure()
    for rpm, raw, sm in zip(rotations, raw_curves, smoothed_curves):
        label = f"{format_rpm(rpm)} U/min"
        if show_raw:
            fig.add_trace(go.Scatter(
                x=potential, y=raw, mode="lines",
                opacity=0.22, line=dict(width=1),
                name=f"{label} roh", showlegend=False,
                hovertemplate="E=%{x:.4f} V<br>I=%{y:.6g}<extra>Raw data</extra>",
            ))
        fig.add_trace(go.Scatter(
            x=potential, y=sm, mode="lines",
            line=dict(width=2), name=label,
            hovertemplate="E=%{x:.4f} V<br>I=%{y:.6g}<extra>"+label+"</extra>",
        ))
    fig.update_layout(
        title=title, template="plotly_white",
        xaxis_title="Potential / V", yaxis_title=ylabel,
        legend_title="Rotation",
    )
    if reverse_x:
        fig.update_xaxes(autorange="reversed")
    fig.write_html(html_path, include_plotlyjs=True, full_html=True)



def make_combined_2d_plot(
    potential, rotations,
    disk_raw, disk_smooth,
    ring_raw, ring_smooth,
    disk_unit_label, ring_unit_label,
    title, png_path, pdf_path,
    show_raw, reverse_x
):
    """Disk und Ring gemeinsam in 2D mit echter sekundärer y-Achse."""
    fig, ax_disk = plt.subplots(figsize=(11.5, 7.2), constrained_layout=True)
    ax_ring = ax_disk.twinx()

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for idx, (rpm, dr, ds, rr, rs) in enumerate(
        zip(rotations, disk_raw, disk_smooth, ring_raw, ring_smooth)
    ):
        color = colors[idx % len(colors)]
        label = f"{format_rpm(rpm)} U/min"

        if show_raw:
            ax_disk.plot(
                potential, dr, linewidth=0.7, alpha=0.18, color=color
            )
            ax_ring.plot(
                potential, rr, linewidth=0.7, alpha=0.14,
                color=color, linestyle="--"
            )

        ax_disk.plot(
            potential, ds, linewidth=1.7, color=color,
            label=f"Disk {label}"
        )
        ax_ring.plot(
            potential, rs, linewidth=1.4, color=color,
            linestyle="--", label=f"Ring {label}"
        )

    ax_disk.set_title(title)
    ax_disk.set_xlabel("Potential / V")
    ax_disk.set_ylabel(f"Disk current / {disk_unit_label}")
    ax_ring.set_ylabel(f"Ring current / {ring_unit_label}")
    ax_disk.grid(alpha=0.25)

    if reverse_x:
        ax_disk.invert_xaxis()

    disk_handles, disk_labels = ax_disk.get_legend_handles_labels()
    ring_handles, ring_labels = ax_ring.get_legend_handles_labels()
    ax_disk.legend(
        disk_handles + ring_handles,
        disk_labels + ring_labels,
        fontsize=7,
        ncol=2,
        loc="best",
        title="Elektrode und Rotation",
    )

    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)


def make_combined_2d_html(
    potential, rotations,
    disk_raw, disk_smooth,
    ring_raw, ring_smooth,
    disk_unit_label, ring_unit_label,
    title, html_path,
    show_raw, reverse_x
):
    """Interaktive gemeinsame 2D-Darstellung mit Plotly-Sekundärachse."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    for rpm, dr, ds, rr, rs in zip(
        rotations, disk_raw, disk_smooth, ring_raw, ring_smooth
    ):
        label = f"{format_rpm(rpm)} U/min"

        if show_raw:
            fig.add_trace(
                go.Scatter(
                    x=potential, y=dr, mode="lines",
                    opacity=0.16, line=dict(width=1),
                    name=f"Disk roh – {label}",
                    legendgroup=label,
                    showlegend=False,
                    hovertemplate=(
                        "Disk roh<br>E=%{x:.4f} V"
                        "<br>I_D=%{y:.6g} " + disk_unit_label +
                        "<extra></extra>"
                    ),
                ),
                secondary_y=False,
            )
            fig.add_trace(
                go.Scatter(
                    x=potential, y=rr, mode="lines",
                    opacity=0.14, line=dict(width=1, dash="dot"),
                    name=f"Ring roh – {label}",
                    legendgroup=label,
                    showlegend=False,
                    hovertemplate=(
                        "Ring roh<br>E=%{x:.4f} V"
                        "<br>I_R=%{y:.6g} " + ring_unit_label +
                        "<extra></extra>"
                    ),
                ),
                secondary_y=True,
            )

        fig.add_trace(
            go.Scatter(
                x=potential, y=ds, mode="lines",
                line=dict(width=2),
                name=f"Disk – {label}",
                legendgroup=label,
                hovertemplate=(
                    "Disk<br>E=%{x:.4f} V"
                    "<br>I_D=%{y:.6g} " + disk_unit_label +
                    "<extra>" + label + "</extra>"
                ),
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=potential, y=rs, mode="lines",
                line=dict(width=2, dash="dash"),
                name=f"Ring – {label}",
                legendgroup=label,
                hovertemplate=(
                    "Ring<br>E=%{x:.4f} V"
                    "<br>I_R=%{y:.6g} " + ring_unit_label +
                    "<extra>" + label + "</extra>"
                ),
            ),
            secondary_y=True,
        )

    fig.update_layout(
        title=title,
        template="plotly_white",
        hovermode="closest",
        legend_title="Elektrode und Rotation",
        width=1350,
        height=760,
    )
    fig.update_xaxes(title_text="Potential / V")
    if reverse_x:
        fig.update_xaxes(autorange="reversed")

    fig.update_yaxes(
        title_text=f"Disk current / {disk_unit_label}",
        secondary_y=False,
    )
    fig.update_yaxes(
        title_text=f"Ring current / {ring_unit_label}",
        secondary_y=True,
    )

    fig.write_html(html_path, include_plotlyjs=True, full_html=True)

def make_separate_3d(
    potential, rotations, disk_raw, disk_smooth, ring_raw, ring_smooth,
    unit_label, html_path, show_raw, reverse_x
):
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("Diskströme", "Ringströme"),
        horizontal_spacing=0.04,
    )
    for rpm, dr, ds, rr, rs in zip(rotations, disk_raw, disk_smooth, ring_raw, ring_smooth):
        y = np.full_like(potential, rpm)
        name = f"{format_rpm(rpm)} U/min"
        if show_raw:
            fig.add_trace(go.Scatter3d(
                x=potential, y=y, z=dr, mode="lines",
                opacity=0.15, line=dict(width=2), showlegend=False,
                hovertemplate="E=%{x:.4f} V<br>n=%{y:g}<br>I_D=%{z:.6g}<extra>roh</extra>",
            ), row=1, col=1)
            fig.add_trace(go.Scatter3d(
                x=potential, y=y, z=rr, mode="lines",
                opacity=0.15, line=dict(width=2), showlegend=False,
                hovertemplate="E=%{x:.4f} V<br>n=%{y:g}<br>I_R=%{z:.6g}<extra>roh</extra>",
            ), row=1, col=2)
        fig.add_trace(go.Scatter3d(
            x=potential, y=y, z=ds, mode="lines",
            line=dict(width=5), name=name,
        ), row=1, col=1)
        fig.add_trace(go.Scatter3d(
            x=potential, y=y, z=rs, mode="lines",
            line=dict(width=5), name=name, showlegend=False,
        ), row=1, col=2)

    autorange = "reversed" if reverse_x else True
    # Die beiden Teilgrafiken erhalten bewusst unterschiedliche,
    # eindeutige Ordinatenbeschriftungen.
    if " bzw. " in unit_label:
        disk_unit, ring_unit = unit_label.split(" bzw. ", 1)
    else:
        disk_unit = ring_unit = unit_label

    scene_disk = dict(
        xaxis=dict(title="Potential / V", autorange=autorange),
        yaxis=dict(title="Rotation / U min⁻¹"),
        zaxis=dict(title=f"Disk current / {disk_unit}"),
        aspectmode="auto",
    )
    scene_ring = dict(
        xaxis=dict(title="Potential / V", autorange=autorange),
        yaxis=dict(title="Rotation / U min⁻¹"),
        zaxis=dict(title=f"Ring current / {ring_unit}"),
        aspectmode="auto",
    )
    fig.update_layout(
        title="RRDE-Messungen – getrennte 3D-Darstellung",
        template="plotly_white", scene=scene_disk, scene2=scene_ring,
        width=1450, height=780, legend_title="Rotation",
        margin=dict(l=10, r=10, t=80, b=10),
    )
    fig.write_html(html_path, include_plotlyjs=True, full_html=True)


def make_combined_3d(
    potential, rotations, disk_raw, disk_smooth, ring_raw, ring_smooth,
    disk_unit_label, ring_unit_label, html_path, show_raw, reverse_x,
    manual_factor: Optional[float] = None,
):
    """Gemeinsame 3D-Darstellung mit gegenüberliegender Ring currentskala."""
    auto_factor = choose_ring_display_scale(disk_smooth, ring_smooth)
    factor = float(manual_factor) if manual_factor is not None else auto_factor
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError("Der manuelle Ring current-Skalierungsfaktor muss größer als 0 sein.")

    fig = go.Figure()
    kinds = []

    for rpm, dr, ds, rr, rs in zip(rotations, disk_raw, disk_smooth, ring_raw, ring_smooth):
        y = np.full_like(potential, rpm, dtype=float)
        rpm_text = f"{format_rpm(rpm)} U/min"
        if show_raw:
            fig.add_trace(go.Scatter3d(
                x=potential, y=y, z=dr, mode="lines", line=dict(width=2), opacity=0.10,
                name=f"Disk roh – {rpm_text}", showlegend=False, customdata=dr,
                hovertemplate="Disk roh<br>E=%{x:.4f} V<br>n=%{y:g} U/min"
                              "<br>I<sub>D</sub>=%{customdata:.6g} "+disk_unit_label+"<extra></extra>",
            )); kinds.append("disk")
            fig.add_trace(go.Scatter3d(
                x=potential, y=y, z=rr*factor, mode="lines",
                line=dict(width=2, dash="dot"), opacity=0.10,
                name=f"Ring roh – {rpm_text}", showlegend=False, customdata=rr,
                hovertemplate="Ring roh<br>E=%{x:.4f} V<br>n=%{y:g} U/min"
                              "<br>I<sub>R</sub>=%{customdata:.6g} "+ring_unit_label+"<extra></extra>",
            )); kinds.append("ring")
        fig.add_trace(go.Scatter3d(
            x=potential, y=y, z=ds, mode="lines", line=dict(width=6),
            name=f"Disk – {rpm_text}", customdata=ds,
            hovertemplate="Disk<br>E=%{x:.4f} V<br>n=%{y:g} U/min"
                          "<br>I<sub>D</sub>=%{customdata:.6g} "+disk_unit_label+"<extra></extra>",
        )); kinds.append("disk")
        fig.add_trace(go.Scatter3d(
            x=potential, y=y, z=rs*factor, mode="lines+markers",
            line=dict(width=4, dash="dot"), marker=dict(size=2),
            name=f"Ring – {rpm_text}", customdata=rs,
            hovertemplate="Ring<br>E=%{x:.4f} V<br>n=%{y:g} U/min"
                          "<br>I<sub>R</sub>=%{customdata:.6g} "+ring_unit_label+
                          f"<br>Darstellung: ×{factor:g}<extra></extra>",
        )); kinds.append("ring")

    all_z=np.concatenate([np.asarray(v,float) for v in disk_smooth]+[np.asarray(v,float)*factor for v in ring_smooth])
    zmin=float(np.nanmin(all_z)); zmax=float(np.nanmax(all_z))
    if not np.isfinite(zmin) or not np.isfinite(zmax) or zmin==zmax: zmin,zmax=-1.0,1.0
    zpad=0.05*(zmax-zmin); zmin_plot,zmax_plot=zmin-zpad,zmax+zpad
    pmin,pmax=float(np.nanmin(potential)),float(np.nanmax(potential)); pspan=max(abs(pmax-pmin),1e-6)
    rmin,rmax=float(min(rotations)),float(max(rotations)); rspan=max(abs(rmax-rmin),1.0)

    # Gegenüberliegende Seite der Szene.
    ring_axis_x = pmax + 0.14*pspan if not reverse_x else pmin - 0.14*pspan
    ring_axis_y = rmax + 0.10*rspan
    tick_z=np.linspace(zmin,zmax,6); tick_labels=[f"{v/factor:.3g}" for v in tick_z]
    axis_color="#C00000"; disk_color="#17365D"

    fig.add_trace(go.Scatter3d(x=[ring_axis_x,ring_axis_x], y=[ring_axis_y,ring_axis_y], z=[zmin,zmax],
        mode="lines", line=dict(width=7,color=axis_color), showlegend=False, hoverinfo="skip",
        name=f"Ring currentskala / {ring_unit_label}")); kinds.append("axis")
    tx=[]; ty=[]; tz=[]
    for z in tick_z:
        tx += [ring_axis_x-0.028*pspan, ring_axis_x, None]; ty += [ring_axis_y,ring_axis_y,None]; tz += [float(z),float(z),None]
    fig.add_trace(go.Scatter3d(x=tx,y=ty,z=tz,mode="lines",line=dict(width=5,color=axis_color),showlegend=False,hoverinfo="skip")); kinds.append("axis")
    fig.add_trace(go.Scatter3d(x=[ring_axis_x+0.04*pspan]*len(tick_z),y=[ring_axis_y]*len(tick_z),z=tick_z,
        mode="text",text=tick_labels,textposition="middle right",textfont=dict(size=12,color=axis_color),showlegend=False,hoverinfo="skip")); kinds.append("axis")
    # Die beiden Strom-Achsentitel werden als 3D-Szenenannotationen direkt
    # an den oberen Enden der zugehörigen Achsen verankert. Anders als
    # Paper-Annotationen bewegen sie sich beim Drehen und Zoomen mit der Szene
    # und wirken dadurch sichtbar an die Achsen angebunden.
    disk_title = f"<b>Disk current / {disk_unit_label}</b>"
    ring_title = f"<b>Ring current / {ring_unit_label}</b>"
    disk_axis_x = pmin - 0.04 * pspan if not reverse_x else pmax + 0.04 * pspan
    disk_axis_y = rmin - 0.05 * rspan
    title_z = zmax_plot + 0.06 * (zmax_plot - zmin_plot)

    both=[True]*len(kinds); only_disk=[k in ("disk","axis") for k in kinds]; only_ring=[k in ("ring","axis") for k in kinds]
    x_range=([pmin-0.04*pspan,pmax+0.32*pspan] if not reverse_x else [pmax+0.04*pspan,pmin-0.32*pspan])
    mode_text="manuell" if manual_factor is not None else "automatisch"
    fig.update_layout(
        title=("RRDE – Disk- und Ringströme gemeinsam"+f"<br><sup>Ring current-Skalierung: {mode_text}, Faktor {factor:g}</sup>"),
        template="plotly_white",
        scene=dict(
            domain=dict(x=[0.03, 0.95], y=[0.00, 0.88]),
            xaxis=dict(title="Potential / V",range=x_range),
            yaxis=dict(title="Rotation / U min⁻¹",range=[rmin-0.05*rspan,rmax+0.18*rspan]),
            zaxis=dict(title=dict(text=""),
                       range=[zmin_plot,title_z],exponentformat="none",showexponent="none",tickformat=".0f",
                       tickfont=dict(color=disk_color),linecolor=disk_color,gridcolor="#D9E2F3"),
            aspectmode="auto",
            annotations=[
                dict(
                    x=disk_axis_x, y=disk_axis_y, z=title_z,
                    text=disk_title,
                    showarrow=False,
                    xanchor="left", yanchor="bottom",
                    font=dict(size=15, color=disk_color),
                ),
                dict(
                    x=ring_axis_x, y=ring_axis_y, z=title_z,
                    text=ring_title,
                    showarrow=False,
                    xanchor="right", yanchor="bottom",
                    font=dict(size=15, color=axis_color),
                ),
            ]),
        width=1500,height=860,
        legend=dict(
            title="Rotation",
            orientation="h",
            x=0.46, y=1.02,
            xanchor="center", yanchor="bottom",
        ),
        margin=dict(l=20,r=30,t=115,b=70),
        updatemenus=[dict(type="buttons",direction="right",x=0.02,y=1.08,xanchor="left",yanchor="top",
            buttons=[dict(label="Disk + Ring",method="update",args=[{"visible":both}]),
                     dict(label="Nur Disk",method="update",args=[{"visible":only_disk}]),
                     dict(label="Nur Ring",method="update",args=[{"visible":only_ring}])])],
        annotations=[
            dict(
                text=(
                    f"Disk current: {disk_unit_label}; Ring current: {ring_unit_label}; "
                    f"Ring räumlich ×{factor:g}."
                ),
                x=0.49, y=-0.08,
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=11),
            ),
        ])
    fig.write_html(html_path,include_plotlyjs=True,full_html=True)
    return factor, auto_factor


def save_processed_csv(
    path, potential, rotations, disk_raw, disk_smooth,
    ring_raw, ring_smooth, ring_compensated=None, disk_original=None, ring_original=None
):
    data = {"Potential / V": potential}
    if ring_compensated is None:
        ring_compensated = ring_smooth
    if disk_original is None:
        disk_original = disk_raw
    if ring_original is None:
        ring_original = ring_raw
    for rpm, dr0, rr0, dr, ds, rr, rs, rc in zip(
        rotations, disk_original, ring_original, disk_raw, disk_smooth, ring_raw, ring_smooth, ring_compensated
    ):
        key = format_rpm(rpm)
        data[f"Disk {key} rpm raw original / A"] = dr0
        data[f"Ring {key} rpm raw original / A"] = rr0
        data[f"Disk {key} rpm raw corrected / A"] = dr
        data[f"Disk {key} rpm smooth corrected / A"] = ds
        data[f"Ring {key} rpm raw corrected / A"] = rr
        data[f"Ring {key} rpm smooth uncompensated / A"] = rs
        data[f"Ring {key} rpm smooth compensated / A"] = rc
    pd.DataFrame(data).to_csv(path, index=False)


def save_excel_report(
    path: Path,
    source_file: Path,
    potential: np.ndarray,
    rotations: Sequence[float],
    disk_raw: Sequence[np.ndarray],
    disk_smooth: Sequence[np.ndarray],
    ring_raw: Sequence[np.ndarray],
    ring_smooth: Sequence[np.ndarray],
    window: int,
    poly: int,
    ring_factor: float,
    background_enabled: bool = False,
    background_method: str = "none",
    background_description: str = "Keine",
) -> None:
    """
    Direkter Excel-Export über XlsxWriter.

    Tabellenblätter:
      Übersicht
      Raw data
      Disk_geglättet
      Ring_geglättet
      Disk_Ring_gemeinsam
      Diagramme
    """
    try:
        import xlsxwriter
    except ImportError as exc:
        raise RuntimeError(
            "Das Paket 'xlsxwriter' ist nicht installiert.\n"
            "Installation: py -m pip install xlsxwriter"
        ) from exc

    try:
        workbook = xlsxwriter.Workbook(str(path), {"nan_inf_to_errors": True})
        workbook.set_properties({
            "title": "RRDE-Analysis",
            "subject": "Disk- und Ringströme",
            "author": "RRDE-Analysis GUI",
            "comments": f"Erstellt mit RRDE-Analysis Version {VERSION}",
        })

        fmt_title = workbook.add_format({
            "bold": True, "font_size": 18, "font_color": "white",
            "bg_color": "#17365D", "align": "left", "valign": "vcenter",
        })
        fmt_header = workbook.add_format({
            "bold": True, "font_color": "white", "bg_color": "#1F4E78",
            "border": 1, "align": "center", "valign": "vcenter",
        })
        fmt_sub = workbook.add_format({
            "bold": True, "font_color": "#17365D", "bg_color": "#D9EAF7",
            "border": 1,
        })
        fmt_text = workbook.add_format({"border": 1})
        fmt_potential = workbook.add_format({"num_format": "0.0000", "border": 1})
        fmt_scientific = workbook.add_format({"num_format": "0.000000E+00", "border": 1})
        fmt_disk_ma = workbook.add_format({"num_format": "0.0000", "border": 1})
        fmt_ring_ua = workbook.add_format({"num_format": "0.000", "border": 1})
        fmt_note = workbook.add_format({
            "font_color": "#7F6000", "bg_color": "#FFF2CC",
            "text_wrap": True, "border": 1,
        })

        # Übersicht
        ws = workbook.add_worksheet("Übersicht")
        ws.set_tab_color("#17365D")
        ws.merge_range("A1:F1", "RRDE-Analysis", fmt_title)
        ws.set_row(0, 28)
        ws.set_column("A:A", 38)
        ws.set_column("B:B", 72)
        ws.write_row("A3", ["Parameter", "Wert"], fmt_header)
        parameters = [
            ("Quelldatei", source_file.name),
            ("Erstellt", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            ("Potentialpunkte", len(potential)),
            ("Rotationen / U min⁻¹", ", ".join(format_rpm(r) for r in rotations)),
            ("Glättung", "Savitzky–Golay"),
            ("Fensterbreite / Punkte", window),
            ("Polynomgrad", poly),
            ("Ring-Skalierungsfaktor im gemeinsamen HTML", ring_factor),
            ("Ring-Untergrundkompensation", "Ja" if background_enabled else "Nein"),
            ("Methode/Bereich", background_description),
            ("Raw data", "Disk- und Ringströme in A"),
            ("Geglättete Daten", "Disk- und Ringströme in A"),
            ("Gemeinsames Datenblatt", "Disk current in mA; Ring current in µA"),
        ]
        for row, (key, value) in enumerate(parameters, start=3):
            ws.write(row, 0, key, fmt_sub)
            ws.write(row, 1, value, fmt_text)
        ws.write(
            15, 0,
            "Hinweis zur gemeinsamen 3D-Darstellung",
            fmt_sub,
        )
        ws.write(
            15, 1,
            "Die Ringströme werden nur im gemeinsamen HTML linear skaliert. "
            "Die Excel-Werte bleiben physikalisch unverändert und werden in "
            "geeigneten Einheiten ausgegeben.",
            fmt_note,
        )
        ws.freeze_panes(3, 0)

        def write_wide_sheet(name, disk_values=None, ring_values=None, both=False):
            sheet = workbook.add_worksheet(name)
            sheet.freeze_panes(1, 1)
            sheet.autofilter(0, 0, len(potential), 2 * len(rotations))
            sheet.set_column(0, 0, 15)

            headers = ["Potential / V"]
            for rpm in rotations:
                if both or disk_values is not None:
                    headers.append(f"Disk {format_rpm(rpm)} U/min / A")
                if both or ring_values is not None:
                    headers.append(f"Ring {format_rpm(rpm)} U/min / A")
            sheet.write_row(0, 0, headers, fmt_header)

            for i, e in enumerate(potential, start=1):
                sheet.write_number(i, 0, float(e), fmt_potential)

            col = 1
            for j in range(len(rotations)):
                if both or disk_values is not None:
                    values = disk_values[j] if disk_values is not None else disk_raw[j]
                    sheet.write_column(1, col, [float(v) for v in values], fmt_scientific)
                    sheet.set_column(col, col, 21)
                    col += 1
                if both or ring_values is not None:
                    values = ring_values[j] if ring_values is not None else ring_raw[j]
                    sheet.write_column(1, col, [float(v) for v in values], fmt_scientific)
                    sheet.set_column(col, col, 21)
                    col += 1
            return sheet

        raw_ws = write_wide_sheet("Raw data", disk_raw, ring_raw, both=True)
        disk_ws = write_wide_sheet("Disk_geglättet", disk_smooth, None, both=False)
        ring_ws = write_wide_sheet("Ring_geglättet_komp", None, ring_smooth, both=False)

        # Gemeinsames Blatt in passenden Einheiten
        comb = workbook.add_worksheet("Disk_Ring_gemeinsam")
        comb.freeze_panes(1, 1)
        comb.set_column(0, 0, 15)
        headers = ["Potential / V"]
        for rpm in rotations:
            headers += [
                f"Disk {format_rpm(rpm)} U/min / mA",
                f"Ring {format_rpm(rpm)} U/min / µA",
            ]
        comb.write_row(0, 0, headers, fmt_header)
        for i, e in enumerate(potential, start=1):
            comb.write_number(i, 0, float(e), fmt_potential)
        col = 1
        for j in range(len(rotations)):
            comb.write_column(1, col, [float(v * 1e3) for v in disk_smooth[j]], fmt_disk_ma)
            comb.set_column(col, col, 21)
            col += 1
            comb.write_column(1, col, [float(v * 1e6) for v in ring_smooth[j]], fmt_ring_ua)
            comb.set_column(col, col, 21)
            col += 1
        comb.autofilter(0, 0, len(potential), len(headers) - 1)

        # Diagrammblatt
        charts = workbook.add_worksheet("Diagramme")
        charts.set_tab_color("#70AD47")
        charts.merge_range("A1:N1", "RRDE-Diagramme – editierbare Excel-Diagramme", fmt_title)
        charts.set_row(0, 28)
        charts.set_column("A:N", 12)

        n = len(potential) + 1

        disk_chart = workbook.add_chart({"type": "scatter", "subtype": "straight"})
        for j, rpm in enumerate(rotations):
            disk_col = 1 + 2 * j
            disk_chart.add_series({
                "name": f"Disk {format_rpm(rpm)} U/min",
                "categories": ["Disk_Ring_gemeinsam", 1, 0, n - 1, 0],
                "values": ["Disk_Ring_gemeinsam", 1, disk_col, n - 1, disk_col],
                "line": {"width": 1.5},
            })
        disk_chart.set_title({"name": "Geglättete Diskströme"})
        disk_chart.set_x_axis({"name": "Potential / V", "major_gridlines": {"visible": True}})
        disk_chart.set_y_axis({"name": "Disk current / mA", "major_gridlines": {"visible": True}})
        disk_chart.set_legend({"position": "bottom"})
        disk_chart.set_size({"width": 760, "height": 440})
        charts.insert_chart("A3", disk_chart)

        ring_chart = workbook.add_chart({"type": "scatter", "subtype": "straight"})
        for j, rpm in enumerate(rotations):
            ring_col = 2 + 2 * j
            ring_chart.add_series({
                "name": f"Ring {format_rpm(rpm)} U/min",
                "categories": ["Disk_Ring_gemeinsam", 1, 0, n - 1, 0],
                "values": ["Disk_Ring_gemeinsam", 1, ring_col, n - 1, ring_col],
                "line": {"width": 1.5},
            })
        ring_chart.set_title({"name": "Geglättete Ringströme"})
        ring_chart.set_x_axis({"name": "Potential / V", "major_gridlines": {"visible": True}})
        ring_chart.set_y_axis({"name": "Ring current / µA", "major_gridlines": {"visible": True}})
        ring_chart.set_legend({"position": "bottom"})
        ring_chart.set_size({"width": 760, "height": 440})
        charts.insert_chart("A27", ring_chart)

        # Gemeinsames Excel-Diagramm mit einem einzigen Scatter-Chart.
        # XlsxWriter erzeugt bei Serien mit y2_axis=True innerhalb desselben
        # Diagramms ein zweites ScatterChart samt rechter Sekundärachse.
        combined_chart = workbook.add_chart({"type": "scatter", "subtype": "straight"})

        for j, rpm in enumerate(rotations):
            disk_col = 1 + 2 * j
            ring_col = 2 + 2 * j

            combined_chart.add_series({
                "name": f"Disk {format_rpm(rpm)} U/min",
                "categories": ["Disk_Ring_gemeinsam", 1, 0, n - 1, 0],
                "values": ["Disk_Ring_gemeinsam", 1, disk_col, n - 1, disk_col],
                "line": {"width": 1.3},
            })

            combined_chart.add_series({
                "name": f"Ring {format_rpm(rpm)} U/min",
                "categories": ["Disk_Ring_gemeinsam", 1, 0, n - 1, 0],
                "values": ["Disk_Ring_gemeinsam", 1, ring_col, n - 1, ring_col],
                "line": {"width": 1.0, "dash_type": "dash"},
                "y2_axis": True,
            })

        combined_chart.set_title({"name": "Disk- und Ringströme gemeinsam"})
        combined_chart.set_x_axis({
            "name": "Potential / V",
            "major_gridlines": {"visible": True},
        })
        combined_chart.set_y_axis({
            "name": "Disk current / mA",
            "major_gridlines": {"visible": True},
        })
        combined_chart.set_y2_axis({
            "name": "Ring current / µA",
            "major_gridlines": {"visible": False},
            "line": {"color": "#C00000", "width": 1.25},
            "num_font": {"color": "#C00000"},
            "name_font": {"color": "#C00000", "bold": True},
            "num_format": "0.000",
        })
        combined_chart.set_legend({"position": "bottom"})
        combined_chart.set_size({"width": 1040, "height": 540})
        charts.insert_chart("O3", combined_chart)

        workbook.close()

    except Exception:
        # Nicht als fehlendes Paket tarnen: tatsächlichen Fehler weiterreichen.
        try:
            workbook.close()
        except Exception:
            pass
        raise


def propose_nova_mapping(path: Path):
    """
    Erstellt die feste Metrohm-NOVA-Zuordnung.

    Struktur:
      Spalte 1: Potential
      Spalten 2, 4, 6, ...: Disk current
      Spalten 3, 5, 7, ...: Ring current

    Die Rotation rate steht im Header der jeweiligen Diskspalte. Ein leerer bzw.
    von pandas als 'Unnamed: ...' bezeichneter Ringheader erhält dieselbe
    Rotation rate wie die vorhergehende Diskspalte.
    """
    df = read_csv_robust(path)
    columns = [str(c) for c in df.columns]

    if len(columns) < 3:
        raise ValueError("Die NOVA-Datei enthält zu wenige Spalten für RRDE-Daten.")
    if (len(columns) - 1) % 2 != 0:
        raise ValueError(
            "Nach der Potentialspalte muss die NOVA-Datei eine gerade Anzahl "
            "von Stromspalten enthalten: jeweils Disk und Ring."
        )

    potential = columns[0]
    pairs = []
    display_rows = []
    display_headers = [f"Potential: {potential}"]

    pair_number = 0
    for disk_index in range(1, len(columns), 2):
        ring_index = disk_index + 1
        disk_column = columns[disk_index]
        ring_column = columns[ring_index]

        rpm = natural_float(disk_column)
        if rpm is None:
            raise ValueError(
                f"In der Diskspalte {disk_index + 1} ('{disk_column}') "
                "wurde keine Rotation rate gefunden."
            )

        pair_number += 1
        pairs.append((float(rpm), disk_column, ring_column))
        display_rows.append({
            "rotation": float(rpm),
            "disk_number": disk_index + 1,
            "ring_number": ring_index + 1,
            "disk_source": disk_column,
            "ring_source": ring_column,
            "disk_label": f"Disk {format_rpm(float(rpm))}",
            "ring_label": f"Ring {format_rpm(float(rpm))}",
        })
        display_headers.extend([
            f"Disk {format_rpm(float(rpm))}",
            f"Ring {format_rpm(float(rpm))}",
        ])

    if not pairs:
        raise ValueError("Es wurden keine Disk-/Ring-Paare gefunden.")

    return {
        "potential": potential,
        "pairs": pairs,
        "display_rows": display_rows,
        "display_headers": display_headers,
        "dataframe": df,
    }


FARADAY_CONSTANT = 96485.33212





REFERENCE_ELECTRODES_VS_SHE_25C = {
    "Ag/AgCl (3 M KCl)": 0.210,
    "Ag/AgCl (sat. KCl)": 0.197,
    "SCE": 0.241,
    "Hg/HgO (1 M NaOH)": 0.098,
    "Hg/HgSO4 (sat. K2SO4)": 0.640,
    "SHE": 0.0,
}


def calculate_equilibrium_potential(
    reaction: str,
    reference_electrode: str,
    ph: float,
    temperature_c: float = 25.0,
    user_equilibrium_vs_she: Optional[float] = None,
) -> dict:
    """Return equilibrium potentials on SHE, RHE and selected reference scales.

    The predefined aqueous reactions use their conventional thermodynamic
    potentials. Standard potentials and tabulated reference-electrode values
    are referenced to 25 °C; the pH conversion uses the entered temperature.
    """
    if not np.isfinite(ph) or not (0.0 <= ph <= 14.5):
        raise ValueError("Der pH-Wert muss zwischen 0 und 14,5 liegen.")
    if not np.isfinite(temperature_c) or temperature_c <= -273.15:
        raise ValueError("Die Temperatur ist ungültig.")
    if reference_electrode not in REFERENCE_ELECTRODES_VS_SHE_25C and reference_electrode != "RHE":
        raise ValueError("Bitte eine gültige Referenzelektrode auswählen.")

    temperature_k = temperature_c + 273.15
    nernst_per_ph = 2.303 * 8.31446261815324 * temperature_k / FARADAY_CONSTANT

    if reaction in ("Oxygen reduction (ORR)", "Oxygen evolution (OER)"):
        e_rhe = 1.229
        e_she = e_rhe - nernst_per_ph * ph
    elif reaction in ("Hydrogen evolution (HER)", "Hydrogen oxidation (HOR)"):
        e_rhe = 0.0
        e_she = -nernst_per_ph * ph
    elif reaction == "User-defined":
        if user_equilibrium_vs_she is None or not np.isfinite(user_equilibrium_vs_she):
            raise ValueError("Für eine benutzerdefinierte Reaktion ist E_eq vs SHE erforderlich.")
        e_she = float(user_equilibrium_vs_she)
        e_rhe = e_she + nernst_per_ph * ph
    else:
        raise ValueError("Bitte eine Reaktion auswählen.")

    if reference_electrode == "RHE":
        e_ref_vs_she = -nernst_per_ph * ph
    else:
        e_ref_vs_she = REFERENCE_ELECTRODES_VS_SHE_25C[reference_electrode]
    e_vs_reference = e_she - e_ref_vs_she

    warnings = []
    if abs(temperature_c - 25.0) > 0.2:
        warnings.append(
            "Standard- und Referenzelektrodenpotentiale sind für 25 °C tabelliert; "
            "nur die pH-Umrechnung wurde temperaturkorrigiert."
        )
    return {
        "reaction": reaction,
        "reference_electrode": reference_electrode,
        "ph": float(ph),
        "temperature_c": float(temperature_c),
        "nernst_per_ph_v": float(nernst_per_ph),
        "equilibrium_vs_she_v": float(e_she),
        "equilibrium_vs_rhe_v": float(e_rhe),
        "reference_vs_she_v": float(e_ref_vs_she),
        "equilibrium_vs_reference_v": float(e_vs_reference),
        "warnings": warnings,
    }


def calculate_tafel_analysis(
    potential: np.ndarray,
    rotations: Sequence[float],
    disk_curves_a: Sequence[np.ndarray],
    potential_from: float,
    potential_to: float,
    equilibrium_vs_reference_v: float,
    use_current_density: bool = True,
    area_cm2: Optional[float] = None,
    use_kl_corrected_current: bool = False,
    limiting_current_potential_v: Optional[float] = None,
    limiting_current_from_v: Optional[float] = None,
    limiting_current_to_v: Optional[float] = None,
    calculate_standard_rate_constant: bool = False,
    electron_number: Optional[float] = None,
    transfer_coefficient: Optional[float] = None,
    oxidized_concentration_mol_l: Optional[float] = None,
    reduced_concentration_mol_l: Optional[float] = None,
) -> dict:
    """Fit eta = a + b log10(|I| or |j|) for every rotation.

    If mass-transport correction is enabled, the kinetic current is calculated
    curve-by-curve from the Koutecky-Levich relation

        1/I = 1/I_k + 1/I_L,  I_k = I I_L / (I_L - I).

    I_L is preferably estimated as the median current in a user-selected
    diffusion-limited plateau range.  The older single-potential argument is
    retained for backward compatibility, but a range is more robust against
    noise and a locally sloping plateau.
    """
    xpot = np.asarray(potential, dtype=float)
    low, high = sorted((float(potential_from), float(potential_to)))
    range_mask = np.isfinite(xpot) & (xpot >= low) & (xpot <= high)
    if range_mask.sum() < 3:
        raise ValueError("Der gewählte Tafel-Bereich enthält weniger als drei Messpunkte.")
    if use_current_density and (area_cm2 is None or not np.isfinite(area_cm2) or area_cm2 <= 0):
        raise ValueError("Für die Stromdichte ist eine positive Elektrodenfläche erforderlich.")
    if len(rotations) != len(disk_curves_a):
        raise ValueError("Die Zahl der Rotationen und Diskstromkurven stimmt nicht überein.")
    if calculate_standard_rate_constant:
        if not use_current_density:
            raise ValueError("k⁰ kann nur aus der Austauschstromdichte j₀, nicht aus dem Gesamtstrom I₀, berechnet werden.")
        if electron_number is None or not np.isfinite(electron_number) or electron_number <= 0:
            raise ValueError("Für k⁰ ist eine positive Elektronenzahl n erforderlich.")
        if transfer_coefficient is None or not np.isfinite(transfer_coefficient) or not (0 < transfer_coefficient < 1):
            raise ValueError("Für k⁰ muss der Transferkoeffizient alpha zwischen 0 und 1 liegen.")
        if (oxidized_concentration_mol_l is None or not np.isfinite(oxidized_concentration_mol_l)
                or oxidized_concentration_mol_l <= 0):
            raise ValueError("Für k⁰ ist eine positive Konzentration c_O erforderlich.")
        if (reduced_concentration_mol_l is None or not np.isfinite(reduced_concentration_mol_l)
                or reduced_concentration_mol_l <= 0):
            raise ValueError("Für k⁰ ist eine positive Konzentration c_R erforderlich.")

    plateau_range = None
    if use_kl_corrected_current:
        if limiting_current_from_v is not None and limiting_current_to_v is not None:
            plateau_range = tuple(sorted((float(limiting_current_from_v), float(limiting_current_to_v))))
            if not all(np.isfinite(plateau_range)):
                raise ValueError("Der Grenzstrombereich enthält ungültige Werte.")
            if plateau_range[0] < np.nanmin(xpot) or plateau_range[1] > np.nanmax(xpot):
                raise ValueError("Der Grenzstrombereich liegt außerhalb des Messbereichs.")
            plateau_mask = np.isfinite(xpot) & (xpot >= plateau_range[0]) & (xpot <= plateau_range[1])
            if plateau_mask.sum() < 3:
                raise ValueError("Der Grenzstrombereich muss mindestens drei Messpunkte enthalten.")
        elif limiting_current_potential_v is not None and np.isfinite(limiting_current_potential_v):
            if limiting_current_potential_v < np.nanmin(xpot) or limiting_current_potential_v > np.nanmax(xpot):
                raise ValueError("Das Grenzstrom-Potential liegt außerhalb des Messbereichs.")
            plateau_mask = None
        else:
            raise ValueError("Für die Stofftransportkorrektur ist ein Grenzstrombereich erforderlich.")

    curves = []
    warnings = []
    eta_all = xpot - float(equilibrium_vs_reference_v)
    for rpm, current in zip(rotations, disk_curves_a):
        current = np.asarray(current, dtype=float)
        if current.shape != xpot.shape:
            raise ValueError(f"Potential und Diskstrom passen bei {rpm:g} U/min nicht zusammen.")

        limiting_current_a = None
        limiting_current_std_a = None
        limiting_plateau_rel_std = None
        limiting_plateau_drift_pct = None
        limiting_plateau_slope_a_per_v = None
        limiting_plateau_relative_slope_pct_per_v = None
        analysis_current = current.copy()
        correction_valid = np.ones(current.shape, dtype=bool)
        transport_fraction = np.full_like(current, np.nan, dtype=float)

        if use_kl_corrected_current:
            if plateau_range is not None:
                plateau_values = current[plateau_mask & np.isfinite(current)]
                if plateau_values.size < 3:
                    warnings.append(f"{format_rpm(rpm)} rpm: zu wenige Punkte im Grenzstrombereich; kein Fit.")
                    continue
                limiting_current_a = float(np.nanmedian(plateau_values))
                limiting_current_std_a = float(np.nanstd(plateau_values, ddof=1))
                limiting_plateau_rel_std = (
                    abs(limiting_current_std_a / limiting_current_a) if abs(limiting_current_a) > 0 else np.inf
                )
                # Relative end-to-end drift in the selected plateau range.
                pv = xpot[plateau_mask & np.isfinite(current)]
                cv = current[plateau_mask & np.isfinite(current)]
                order = np.argsort(pv)
                n_edge = max(1, min(5, len(cv) // 4))
                first = float(np.nanmedian(cv[order][:n_edge]))
                last = float(np.nanmedian(cv[order][-n_edge:]))
                limiting_plateau_drift_pct = 100.0 * abs(last - first) / max(abs(limiting_current_a), 1e-30)
                finite_plateau = np.isfinite(pv) & np.isfinite(cv)
                if finite_plateau.sum() >= 3 and np.ptp(pv[finite_plateau]) > 0:
                    limiting_plateau_slope_a_per_v = float(np.polyfit(pv[finite_plateau], cv[finite_plateau], 1)[0])
                    limiting_plateau_relative_slope_pct_per_v = 100.0 * abs(limiting_plateau_slope_a_per_v) / max(abs(limiting_current_a), 1e-30)
            else:
                limiting_current_a = _interp_current_at_potential(
                    xpot, current, float(limiting_current_potential_v)
                )

            if not np.isfinite(limiting_current_a) or abs(limiting_current_a) <= 1e-18:
                warnings.append(f"{format_rpm(rpm)} rpm: ungültiger Grenzstrom; kein Fit.")
                continue

            denominator = limiting_current_a - current
            threshold = max(abs(limiting_current_a) * 1e-6, 1e-18)
            transport_fraction = np.abs(current) / abs(limiting_current_a)
            correction_valid = (
                np.isfinite(current)
                & np.isfinite(denominator)
                & (np.abs(denominator) > threshold)
                & (current * limiting_current_a > 0)
                # I must remain below I_L in magnitude. Near I_L the calculated
                # I_k diverges and becomes dominated by small experimental errors.
                & (transport_fraction < 0.98)
            )
            analysis_current = np.full_like(current, np.nan, dtype=float)
            analysis_current[correction_valid] = (
                current[correction_valid] * limiting_current_a / denominator[correction_valid]
            )

        ordinate = analysis_current / float(area_cm2) if use_current_density else analysis_current
        valid = range_mask & correction_valid & np.isfinite(ordinate) & (np.abs(ordinate) > 1e-15)
        if valid.sum() < 3:
            warnings.append(f"{format_rpm(rpm)} rpm: zu wenige gültige Punkte; kein Fit.")
            continue

        log_abs = np.log10(np.abs(ordinate[valid]))
        eta = eta_all[valid]
        slope, intercept = np.polyfit(log_abs, eta, 1)
        fit = slope * log_abs + intercept
        ss_res = float(np.sum((eta - fit) ** 2))
        ss_tot = float(np.sum((eta - np.mean(eta)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
        tf_fit = transport_fraction[valid] if use_kl_corrected_current else np.full(valid.sum(), np.nan)
        exchange_value = float(10.0 ** (-intercept / slope))
        standard_rate_constant_cm_s = None
        if calculate_standard_rate_constant:
            c_o = float(oxidized_concentration_mol_l) / 1000.0  # mol cm^-3
            c_r = float(reduced_concentration_mol_l) / 1000.0   # mol cm^-3
            concentration_factor = c_o ** (1.0 - float(transfer_coefficient)) * c_r ** float(transfer_coefficient)
            standard_rate_constant_cm_s = exchange_value / (
                float(electron_number) * FARADAY_CONSTANT * concentration_factor
            )

        if use_kl_corrected_current:
            max_tf = float(np.nanmax(tf_fit))
            if max_tf > 0.80:
                warnings.append(
                    f"{format_rpm(rpm)} rpm: |I|/|I_L| erreicht {max_tf:.2f} im Tafel-Bereich; "
                    "die Korrektur ist nahe dem Grenzstrom sehr empfindlich."
                )
            if limiting_plateau_rel_std is not None and limiting_plateau_rel_std > 0.02:
                warnings.append(
                    f"{format_rpm(rpm)} rpm: Die relative Streuung im gewählten Diffusionsgrenzbereich beträgt "
                    f"{100*limiting_plateau_rel_std:.1f} %."
                )
            if limiting_plateau_drift_pct is not None:
                if limiting_plateau_drift_pct > 8.0:
                    warnings.append(
                        f"{format_rpm(rpm)} rpm: Der gewählte Bereich zeigt {limiting_plateau_drift_pct:.1f} % Drift und ist möglicherweise "
                        "nicht vollständig diffusionslimitiert. Prüfen Sie einen Bereich mit konstanterem Grenzstrom."
                    )
                elif limiting_plateau_drift_pct >= 3.0:
                    warnings.append(
                        f"{format_rpm(rpm)} rpm: Der gewählte Diffusionsgrenzbereich zeigt eine leichte Drift von "
                        f"{limiting_plateau_drift_pct:.1f} %."
                    )

        curves.append({
            "rpm": float(rpm),
            "potential_v": xpot[valid],
            "measured_current_a": current[valid],
            "current_a": analysis_current[valid],
            "limiting_current_a": float(limiting_current_a) if limiting_current_a is not None else None,
            "limiting_current_std_a": limiting_current_std_a,
            "limiting_plateau_rel_std": limiting_plateau_rel_std,
            "limiting_plateau_drift_pct": limiting_plateau_drift_pct,
            "limiting_plateau_slope_a_per_v": limiting_plateau_slope_a_per_v,
            "limiting_plateau_relative_slope_pct_per_v": limiting_plateau_relative_slope_pct_per_v,
            "transport_fraction": tf_fit,
            "ordinate": ordinate[valid],
            "log10_abs_ordinate": log_abs,
            "overpotential_v": eta,
            "fit_overpotential_v": fit,
            "slope_v_per_decade": float(slope),
            "slope_mv_per_decade": float(1000.0 * slope),
            "absolute_slope_mv_per_decade": float(abs(1000.0 * slope)),
            "intercept_v": float(intercept),
            "exchange_current_density_a_cm2": exchange_value if use_current_density else None,
            "exchange_current_a": exchange_value if not use_current_density else None,
            "standard_rate_constant_cm_s": standard_rate_constant_cm_s,
            "r2": float(r2),
            "n_points": int(valid.sum()),
            "max_transport_fraction": float(np.nanmax(tf_fit)) if use_kl_corrected_current else None,
            "median_transport_fraction": float(np.nanmedian(tf_fit)) if use_kl_corrected_current else None,
        })

    if not curves:
        raise ValueError("Für keine Rotationskurve konnte eine Tafel-Regression berechnet werden.")

    slopes = np.asarray([c["absolute_slope_mv_per_decade"] for c in curves], dtype=float)
    mean_slope = float(np.mean(slopes))
    std_slope = float(np.std(slopes, ddof=1)) if len(slopes) > 1 else 0.0
    return {
        "potential_range": (low, high),
        "equilibrium_vs_reference_v": float(equilibrium_vs_reference_v),
        "use_current_density": bool(use_current_density),
        "area_cm2": float(area_cm2) if area_cm2 is not None else None,
        "use_kl_corrected_current": bool(use_kl_corrected_current),
        "limiting_current_potential_v": (
            float(limiting_current_potential_v) if limiting_current_potential_v is not None else None
        ),
        "limiting_current_range_v": plateau_range,
        "mean_absolute_slope_mv_per_decade": mean_slope,
        "std_absolute_slope_mv_per_decade": std_slope,
        "n_fitted_curves": len(curves),
        "calculate_standard_rate_constant": bool(calculate_standard_rate_constant),
        "electron_number": float(electron_number) if electron_number is not None else None,
        "transfer_coefficient": float(transfer_coefficient) if transfer_coefficient is not None else None,
        "oxidized_concentration_mol_l": (float(oxidized_concentration_mol_l)
                                           if oxidized_concentration_mol_l is not None else None),
        "reduced_concentration_mol_l": (float(reduced_concentration_mol_l)
                                          if reduced_concentration_mol_l is not None else None),
        "mean_exchange_current_density_a_cm2": (
            float(np.mean([c["exchange_current_density_a_cm2"] for c in curves]))
            if use_current_density else None
        ),
        "mean_exchange_current_a": (
            float(np.mean([c["exchange_current_a"] for c in curves]))
            if not use_current_density else None
        ),
        "mean_standard_rate_constant_cm_s": (
            float(np.mean([c["standard_rate_constant_cm_s"] for c in curves]))
            if calculate_standard_rate_constant else None
        ),
        "curves": curves,
        "warnings": warnings,
    }
def save_tafel_outputs(prefix: Path, result: dict, electrochem: dict) -> None:
    """Write Tafel PNG, interactive HTML and Excel report."""
    current_symbol = "j_k" if result["use_kl_corrected_current"] and result["use_current_density"] else (
        "I_k" if result["use_kl_corrected_current"] else ("j" if result["use_current_density"] else "I")
    )
    xlabel = (f"log10(|{current_symbol}| / A cm^-2)" if result["use_current_density"]
              else f"log10(|{current_symbol}| / A)")
    mean_text = (
        f"Mean |slope| = {result['mean_absolute_slope_mv_per_decade']:.1f} ± "
        f"{result['std_absolute_slope_mv_per_decade']:.1f} mV dec⁻¹ "
        f"(n = {result['n_fitted_curves']})"
    )

    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    for curve in result["curves"]:
        name = f"{format_rpm(curve['rpm'])} rpm"
        ax.scatter(curve["log10_abs_ordinate"], curve["overpotential_v"], s=18, label=name)
        order = np.argsort(curve["log10_abs_ordinate"])
        ax.plot(curve["log10_abs_ordinate"][order], curve["fit_overpotential_v"][order])
    ax.set_xlabel(xlabel, fontweight="bold")
    ax.set_ylabel("Overpotential, eta / V", fontweight="bold")
    title = "Tafel analysis"
    if result["use_kl_corrected_current"]:
        title += " – mass-transport-corrected kinetic current"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Rotation rate", fontsize=8, loc="best")
    ax.text(0.01, 0.01, mean_text, transform=ax.transAxes, fontsize=9,
            va="bottom", bbox=dict(boxstyle="round", alpha=0.15))

    table_rows = [[
        format_rpm(c["rpm"]), f"{c['absolute_slope_mv_per_decade']:.1f}", f"{c['r2']:.4f}"
    ] for c in result["curves"]]
    table = ax.table(cellText=table_rows, colLabels=["rpm", "|slope| / mV dec⁻¹", "R²"],
                     cellLoc="center", bbox=[1.03, 0.12, 0.38, 0.78])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    fig.subplots_adjust(right=0.70, bottom=0.13)
    fig.savefig(prefix.with_name(prefix.name + "_Tafel.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)

    html = go.Figure()
    for curve in result["curves"]:
        name = f"{format_rpm(curve['rpm'])} rpm"
        hover = (f"{name}<br>|slope| = {curve['absolute_slope_mv_per_decade']:.2f} mV dec⁻¹"
                 f"<br>R² = {curve['r2']:.5f}<br>n = {curve['n_points']}")
        html.add_trace(go.Scatter(x=curve["log10_abs_ordinate"], y=curve["overpotential_v"],
                                  mode="markers", name=name + " data",
                                  hovertemplate=hover + "<extra></extra>"))
        order = np.argsort(curve["log10_abs_ordinate"])
        html.add_trace(go.Scatter(x=curve["log10_abs_ordinate"][order],
                                  y=curve["fit_overpotential_v"][order],
                                  mode="lines", name=name + " fit",
                                  hovertemplate=hover + "<extra></extra>"))
    html.update_layout(title=title, xaxis_title=xlabel,
                       yaxis_title="Overpotential, eta / V", width=1150, height=720,
                       annotations=[dict(text=mean_text, x=0.01, y=0.01, xref="paper", yref="paper",
                                         showarrow=False, bgcolor="rgba(255,255,255,0.75)")])
    html.write_html(prefix.with_name(prefix.name + "_Tafel.html"), include_plotlyjs=True)

    xlsx = prefix.with_name(prefix.name + "_Tafel.xlsx")
    with pd.ExcelWriter(xlsx, engine="xlsxwriter") as writer:
        summary = [
            ("General settings", ""),
            ("Reaction", electrochem["reaction"]),
            ("Reference electrode", electrochem["reference_electrode"]),
            ("pH", electrochem["ph"]),
            ("Temperature / °C", electrochem["temperature_c"]),
            ("Equilibrium potential vs SHE / V", electrochem["equilibrium_vs_she_v"]),
            ("Equilibrium potential vs RHE / V", electrochem["equilibrium_vs_rhe_v"]),
            ("Equilibrium potential vs selected reference / V", electrochem["equilibrium_vs_reference_v"]),
            ("Evaluation potential range / V", f"{result['potential_range'][0]:g} to {result['potential_range'][1]:g}"),
            ("Current representation", "Current density" if result["use_current_density"] else "Current"),
            ("Mass-transport-corrected kinetic current", "Yes" if result["use_kl_corrected_current"] else "No"),
            ("Limiting-current range / V", (
                f"{result['limiting_current_range_v'][0]:g} to {result['limiting_current_range_v'][1]:g}"
                if result["use_kl_corrected_current"] and result.get("limiting_current_range_v") else "not used"
            )),
            ("Electrode area / cm²", result["area_cm2"] if result["use_current_density"] else "not used"),
            ("", ""),
            ("Tafel analysis", ""),
            ("Mean absolute Tafel slope / mV decade^-1", result["mean_absolute_slope_mv_per_decade"]),
            ("Standard deviation / mV decade^-1", result["std_absolute_slope_mv_per_decade"]),
            ("Number of fitted rotation curves", result["n_fitted_curves"]),
            ("", ""),
            ("Exchange current", ""),
            ("Mean exchange current density j0 / A cm^-2",
             result["mean_exchange_current_density_a_cm2"] if result["use_current_density"] else "not calculated"),
            ("Mean exchange current I0 / A",
             result["mean_exchange_current_a"] if not result["use_current_density"] else "not calculated"),
            ("", ""),
            ("Butler–Volmer analysis", ""),
            ("Standard rate constant k0 calculated",
             "Yes" if result["calculate_standard_rate_constant"] else "No"),
            ("k0 applicability note", (
                "k0 is not calculated for ORR/OER/HER/HOR because the Butler–Volmer concentrations "
                "are not uniquely defined for these multistep electrocatalytic reactions."
                if electrochem["reaction"] in ("Oxygen reduction (ORR)", "Oxygen evolution (OER)",
                                                "Hydrogen evolution (HER)", "Hydrogen oxidation (HOR)")
                else "k0 is meaningful only when cO and cR represent a defined simple redox couple."
            )),
            ("Electron number n", result["electron_number"] if result["calculate_standard_rate_constant"] else "not used"),
            ("Transfer coefficient alpha", result["transfer_coefficient"] if result["calculate_standard_rate_constant"] else "not used"),
            ("Oxidized concentration c_O / mol L^-1",
             result["oxidized_concentration_mol_l"] if result["calculate_standard_rate_constant"] else "not used"),
            ("Reduced concentration c_R / mol L^-1",
             result["reduced_concentration_mol_l"] if result["calculate_standard_rate_constant"] else "not used"),
            ("Mean standard rate constant k0 / cm s⁻¹",
             result["mean_standard_rate_constant_cm_s"] if result["calculate_standard_rate_constant"] else "not calculated"),
            ("Butler–Volmer relation", "j₀ = n·F·k⁰·cO^(1−α)·cR^α"),
        ]
        for idx, warning in enumerate(electrochem.get("warnings", []) + result.get("warnings", []), start=1):
            summary.append((f"Warning {idx}", warning))
        pd.DataFrame(summary, columns=["Parameter", "Value"]).to_excel(writer, sheet_name="Summary", index=False)
        summary_sheet = writer.sheets["Summary"]
        section_format = writer.book.add_format({"bold": True, "bg_color": "#D9EAF7"})
        for excel_row, (parameter, _value) in enumerate(summary, start=1):
            if parameter in {"General settings", "Tafel analysis", "Exchange current", "Butler–Volmer analysis"}:
                summary_sheet.set_row(excel_row, None, section_format)
        fits = pd.DataFrame([{
            "Rotation rate / rpm": c["rpm"],
            "Tafel slope / mV decade^-1": c["slope_mv_per_decade"],
            "Absolute Tafel slope / mV decade^-1": c["absolute_slope_mv_per_decade"],
            "Intercept / V": c["intercept_v"],
            "Exchange current density j0 / A cm^-2": (
                c["exchange_current_density_a_cm2"] if result["use_current_density"] else np.nan),
            "Exchange current I0 / A": (c["exchange_current_a"] if not result["use_current_density"] else np.nan),
            "Standard rate constant k0 / cm s⁻¹": (
                c["standard_rate_constant_cm_s"] if result["calculate_standard_rate_constant"] else np.nan),
            "R²": c["r2"],
            "Number of points": c["n_points"],
            "Limiting current / A": c["limiting_current_a"] if result["use_kl_corrected_current"] else np.nan,
            "Limiting-current SD / A": c.get("limiting_current_std_a") if result["use_kl_corrected_current"] else np.nan,
            "Plateau drift / %": c.get("limiting_plateau_drift_pct") if result["use_kl_corrected_current"] else np.nan,
            "Median |I|/|I_L|": c.get("median_transport_fraction") if result["use_kl_corrected_current"] else np.nan,
            "Maximum |I|/|I_L|": c.get("max_transport_fraction") if result["use_kl_corrected_current"] else np.nan,
        } for c in result["curves"]])
        fits.to_excel(writer, sheet_name="Fits", index=False)
        for c in result["curves"]:
            sheet = f"{format_rpm(c['rpm'])} rpm"[:31]
            frame = {
                "Potential vs selected reference / V": c["potential_v"],
                "Measured disk current / A": c["measured_current_a"],
            }
            if result["use_kl_corrected_current"]:
                frame["Mass-transport-corrected kinetic current / A"] = c["current_a"]
                frame["Absolute measured-current fraction |I|/|I_L|"] = c["transport_fraction"]
            frame[("Current density / A cm^-2" if result["use_current_density"] else "Current / A")] = c["ordinate"]
            frame["log10 absolute current"] = c["log10_abs_ordinate"]
            frame["Overpotential / V"] = c["overpotential_v"]
            frame["Linear fit / V"] = c["fit_overpotential_v"]
            pd.DataFrame(frame).to_excel(writer, sheet_name=sheet, index=False)
        writer.sheets["Summary"].set_column("A:A", 50)
        writer.sheets["Summary"].set_column("B:B", 70)
        writer.sheets["Fits"].set_column(0, len(fits.columns)-1, 24)
def calculate_h2o2_analysis(
    potential: np.ndarray,
    rotations: Sequence[float],
    disk_curves_a: Sequence[np.ndarray],
    corrected_ring_curves_a: Sequence[np.ndarray],
    collection_efficiency: float,
    potential_from: Optional[float] = None,
    potential_to: Optional[float] = None,
    use_absolute_disk_current: bool = True,
) -> dict:
    """Calculate H2O2 yield and electron number n(E) for all rotations."""
    if not (0.0 < collection_efficiency <= 1.0):
        raise ValueError("Die Collection Efficiency N muss größer als 0 und höchstens 1 sein.")
    if len(rotations) != len(disk_curves_a) or len(rotations) != len(corrected_ring_curves_a):
        raise ValueError("Die Anzahl der Rotationen und Stromkurven stimmt nicht überein.")

    potential_array = np.asarray(potential, dtype=float)
    if potential_from is None or potential_to is None:
        range_mask = np.isfinite(potential_array)
        range_min = float(np.nanmin(potential_array))
        range_max = float(np.nanmax(potential_array))
    else:
        range_min, range_max = sorted((float(potential_from), float(potential_to)))
        range_mask = np.isfinite(potential_array) & (potential_array >= range_min) & (potential_array <= range_max)
        if not np.any(range_mask):
            raise ValueError("Im gewählten Potentialbereich liegen keine Messpunkte.")

    results = []
    warnings = []
    for rpm, disk, ring in zip(rotations, disk_curves_a, corrected_ring_curves_a):
        disk = np.asarray(disk, dtype=float)
        ring = np.asarray(ring, dtype=float)
        if disk.shape != ring.shape or disk.shape != np.asarray(potential).shape:
            raise ValueError(f"Disk-, Ring- und Potentialdaten passen bei {rpm:g} U/min nicht zusammen.")

        disk_for_calculation = np.abs(disk) if use_absolute_disk_current else disk
        ring_equiv = np.abs(ring) / collection_efficiency
        denominator = disk_for_calculation + ring_equiv
        valid = range_mask & np.isfinite(denominator) & (denominator > 1e-15)

        h2o2 = np.full_like(denominator, np.nan, dtype=float)
        n_e = np.full_like(denominator, np.nan, dtype=float)
        h2o2[valid] = 200.0 * ring_equiv[valid] / denominator[valid]
        n_e[valid] = 4.0 * disk_for_calculation[valid] / denominator[valid]

        finite_h = h2o2[np.isfinite(h2o2)]
        finite_n = n_e[np.isfinite(n_e)]
        if finite_h.size and (np.nanmin(finite_h) < -1e-9 or np.nanmax(finite_h) > 100.0 + 1e-9):
            warnings.append(f"{format_rpm(rpm)} rpm: H2O2 yield outside 0–100 %.")
        if finite_n.size and (np.nanmin(finite_n) < 2.0 - 1e-9 or np.nanmax(finite_n) > 4.0 + 1e-9):
            warnings.append(f"{format_rpm(rpm)} rpm: n(E) outside 2–4.")

        results.append({
            "rpm": float(rpm),
            "disk_a": disk,
            "ring_corrected_a": ring,
            "h2o2_yield_percent": h2o2,
            "n_electrons": n_e,
        })

    return {
        "potential": potential_array,
        "potential_range": (range_min, range_max),
        "collection_efficiency": float(collection_efficiency),
        "use_absolute_disk_current": bool(use_absolute_disk_current),
        "curves": results,
        "warnings": warnings,
    }


def save_h2o2_outputs(prefix: Path, result: dict, collection_source: str, background_description: str) -> None:
    """Write English PNG, interactive HTML and Excel outputs for H2O2 analysis."""
    potential = result["potential"]
    curves = result["curves"]

    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    for curve in curves:
        ax.plot(potential, curve["h2o2_yield_percent"], label=f"{format_rpm(curve['rpm'])} rpm")
    ax.axhline(0, linewidth=0.8, linestyle=":")
    ax.axhline(100, linewidth=0.8, linestyle=":")
    ax.set_xlabel("Potential / V", fontweight="bold")
    ax.set_ylabel("Hydrogen peroxide yield / %", fontweight="bold")
    ax.set_title("Hydrogen peroxide yield vs. potential")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Rotation rate")
    fig.tight_layout()
    fig.savefig(prefix.with_name(prefix.name + "_H2O2_yield.png"), dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    for curve in curves:
        ax.plot(potential, curve["n_electrons"], label=f"{format_rpm(curve['rpm'])} rpm")
    ax.axhline(2, linewidth=0.8, linestyle=":")
    ax.axhline(4, linewidth=0.8, linestyle=":")
    ax.set_xlabel("Potential / V", fontweight="bold")
    ax.set_ylabel("Number of electrons, n", fontweight="bold")
    ax.set_title("Number of electrons vs. potential")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Rotation rate")
    fig.tight_layout()
    fig.savefig(prefix.with_name(prefix.name + "_n_vs_potential.png"), dpi=220)
    plt.close(fig)

    html = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         subplot_titles=("Hydrogen peroxide yield", "Number of electrons"))
    for curve in curves:
        name = f"{format_rpm(curve['rpm'])} rpm"
        html.add_trace(go.Scatter(x=potential, y=curve["h2o2_yield_percent"], mode="lines", name=name), row=1, col=1)
        html.add_trace(go.Scatter(x=potential, y=curve["n_electrons"], mode="lines", name=name, showlegend=False), row=2, col=1)
    html.update_xaxes(title_text="Potential / V", row=2, col=1)
    html.update_yaxes(title_text="Hydrogen peroxide yield / %", row=1, col=1)
    html.update_yaxes(title_text="Number of electrons, n", row=2, col=1)
    html.update_layout(title="RRDE hydrogen peroxide analysis", height=850, width=1100)
    html.write_html(prefix.with_name(prefix.name + "_H2O2_analysis.html"), include_plotlyjs=True)

    xlsx_path = prefix.with_name(prefix.name + "_H2O2_analysis.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
        summary_rows = [
            ("Collection efficiency, N", result["collection_efficiency"]),
            ("Collection efficiency source", collection_source),
            ("Evaluation potential range / V", f"{result['potential_range'][0]:g} to {result['potential_range'][1]:g}"),
            ("Ring background correction", background_description),
            ("Use absolute cathodic disk currents", "Yes" if result.get("use_absolute_disk_current", True) else "No"),
            ("Formula for H2O2 yield", "200 (|IR|/N) / (|ID| + |IR|/N)"),
            ("Formula for n(E)", "4 |ID| / (|ID| + |IR|/N)"),
        ]
        if result["warnings"]:
            for idx, warning in enumerate(result["warnings"], start=1):
                summary_rows.append((f"Warning {idx}", warning))
        else:
            summary_rows.append(("Quality check", "No out-of-range values detected."))
        pd.DataFrame(summary_rows, columns=["Parameter", "Value"]).to_excel(
            writer, sheet_name="Summary", index=False
        )

        data = {"Potential / V": potential}
        for curve in curves:
            rpm = format_rpm(curve["rpm"])
            data[f"Disk current at {rpm} rpm / A"] = curve["disk_a"]
            data[f"Corrected ring current at {rpm} rpm / A"] = curve["ring_corrected_a"]
            data[f"H2O2 yield at {rpm} rpm / %"] = curve["h2o2_yield_percent"]
            data[f"Number of electrons at {rpm} rpm"] = curve["n_electrons"]
        df = pd.DataFrame(data)
        df.to_excel(writer, sheet_name="H2O2 analysis", index=False)
        writer.sheets["Summary"].set_column("A:A", 34)
        writer.sheets["Summary"].set_column("B:B", 70)
        writer.sheets["H2O2 analysis"].set_column(0, len(df.columns)-1, 24)


def _interp_current_at_potential(
    potential: np.ndarray,
    current: np.ndarray,
    target_potential: float,
) -> float:
    """Linear interpolation of a current curve at the selected potential."""
    x = np.asarray(potential, dtype=float)
    y = np.asarray(current, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    if x.size < 2:
        raise ValueError("Zu wenige gültige Punkte für die Potentialinterpolation.")
    order = np.argsort(x)
    x, y = x[order], y[order]
    if target_potential < x[0] or target_potential > x[-1]:
        raise ValueError(
            f"Das Analysisspotential {target_potential:g} V liegt außerhalb "
            f"des Messbereichs ({x[0]:g} bis {x[-1]:g} V)."
        )
    return float(np.interp(target_potential, x, y))


def calculate_levich_kl(
    potential: np.ndarray,
    rotations: Sequence[float],
    disk_curves_a: Sequence[np.ndarray],
    levich_potential: float,
    kl_potential: float,
    area_cm2: float,
    concentration_mol_l: float,
    viscosity_cm2_s: float,
    mode: str,
    n_value: Optional[float] = None,
    diffusion_cm2_s: Optional[float] = None,
    use_absolute_current: bool = True,
) -> dict:
    """Calculate Levich and Koutecký–Levich at two independent potentials."""
    if len(rotations) < 3:
        raise ValueError("Für Levich/Koutecký–Levich sind mindestens drei Rotation rateen nötig.")
    if area_cm2 <= 0 or concentration_mol_l <= 0 or viscosity_cm2_s <= 0:
        raise ValueError("Fläche, Konzentration und Viskosität müssen größer als null sein.")

    levich_currents_raw_a = np.array([
        _interp_current_at_potential(potential, curve, levich_potential)
        for curve in disk_curves_a
    ], dtype=float)
    kl_currents_raw_a = np.array([
        _interp_current_at_potential(potential, curve, kl_potential)
        for curve in disk_curves_a
    ], dtype=float)

    if use_absolute_current:
        levich_currents_fit_a = np.abs(levich_currents_raw_a)
        kl_currents_fit_a = np.abs(kl_currents_raw_a)
    else:
        levich_currents_fit_a = levich_currents_raw_a.copy()
        kl_currents_fit_a = kl_currents_raw_a.copy()

    if np.any(~np.isfinite(levich_currents_fit_a)):
        raise ValueError("Mindestens ein Levich-Disk current ist ungültig.")
    if np.any(~np.isfinite(kl_currents_fit_a)) or np.any(np.abs(kl_currents_fit_a) < 1e-15):
        raise ValueError("Mindestens ein KL-Disk current ist null oder ungültig.")

    rpm = np.asarray(rotations, dtype=float)
    omega = 2.0 * np.pi * rpm / 60.0
    sqrt_omega = np.sqrt(omega)
    inv_sqrt_omega = 1.0 / sqrt_omega

    levich_slope, levich_intercept = np.polyfit(sqrt_omega, levich_currents_fit_a, 1)
    levich_fit = levich_slope * sqrt_omega + levich_intercept
    ss_res_l = float(np.sum((levich_currents_fit_a - levich_fit) ** 2))
    ss_tot_l = float(np.sum((levich_currents_fit_a - np.mean(levich_currents_fit_a)) ** 2))
    levich_r2 = 1.0 - ss_res_l / ss_tot_l if ss_tot_l > 0 else 1.0

    inv_current = 1.0 / kl_currents_fit_a
    kl_slope, kl_intercept = np.polyfit(inv_sqrt_omega, inv_current, 1)
    kl_fit = kl_slope * inv_sqrt_omega + kl_intercept
    ss_res_k = float(np.sum((inv_current - kl_fit) ** 2))
    ss_tot_k = float(np.sum((inv_current - np.mean(inv_current)) ** 2))
    kl_r2 = 1.0 - ss_res_k / ss_tot_k if ss_tot_k > 0 else 1.0

    concentration_mol_cm3 = concentration_mol_l / 1000.0
    prefactor_without_n_d = (
        0.62 * FARADAY_CONSTANT * area_cm2 * concentration_mol_cm3
        * viscosity_cm2_s ** (-1.0 / 6.0)
    )

    result = {
        "levich_potential": levich_potential,
        "kl_potential": kl_potential,
        "rpm": rpm,
        "omega": omega,
        "sqrt_omega": sqrt_omega,
        "inv_sqrt_omega": inv_sqrt_omega,
        "levich_currents_raw_a": levich_currents_raw_a,
        "levich_currents_fit_a": levich_currents_fit_a,
        "kl_currents_raw_a": kl_currents_raw_a,
        "kl_currents_fit_a": kl_currents_fit_a,
        "inv_current": inv_current,
        "levich_slope": float(levich_slope),
        "levich_intercept": float(levich_intercept),
        "levich_r2": float(levich_r2),
        "levich_fit": levich_fit,
        "kl_slope": float(kl_slope),
        "kl_intercept": float(kl_intercept),
        "kl_r2": float(kl_r2),
        "kl_fit": kl_fit,
        "kinetic_current_a": float(1.0 / kl_intercept) if kl_intercept > 0 else np.nan,
        "mode": mode,
        "area_cm2": area_cm2,
        "concentration_mol_l": concentration_mol_l,
        "viscosity_cm2_s": viscosity_cm2_s,
        "use_absolute_current": use_absolute_current,
    }

    if mode == "D":
        if n_value is None or n_value <= 0:
            raise ValueError("Für die Bestimmung von D muss n größer als null sein.")
        base = levich_slope / (prefactor_without_n_d * n_value)
        if base <= 0:
            raise ValueError("Die Levich-Steigung ist nicht positiv; D kann so nicht berechnet werden.")
        result["n_value"] = float(n_value)
        result["diffusion_cm2_s"] = float(base ** 1.5)
    elif mode == "n":
        if diffusion_cm2_s is None or diffusion_cm2_s <= 0:
            raise ValueError("Für die Bestimmung von n muss D größer als null sein.")
        denom = prefactor_without_n_d * diffusion_cm2_s ** (2.0 / 3.0)
        result["diffusion_cm2_s"] = float(diffusion_cm2_s)
        result["n_value"] = float(levich_slope / denom)
    else:
        raise ValueError("Unbekannter Levich-Analysissmodus.")
    return result

def save_levich_kl_outputs(prefix: Path, result: dict) -> None:
    """Write Levich/KL PNG, HTML and Excel outputs."""
    # Static Levich plot
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.scatter(result["sqrt_omega"], result["levich_currents_fit_a"] * 1e3, label="Messpunkte")
    order = np.argsort(result["sqrt_omega"])
    ax.plot(result["sqrt_omega"][order], result["levich_fit"][order] * 1e3,
            linestyle="--", label="Lineare Regression")
    ax.set_xlabel(r"$\sqrt{\omega}$ / $(rad\,s^{-1})^{1/2}$")
    ax.set_ylabel(r"$|I_D|$ / mA")
    ax.set_title(f"Levich-Analysis bei E = {result['levich_potential']:.3f} V")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.text(0.98, 0.04, f"R² = {result['levich_r2']:.5f}", transform=ax.transAxes,
            ha="right", va="bottom")
    fig.tight_layout()
    fig.savefig(prefix.with_name(prefix.name + "_Levich.png"), dpi=220)
    plt.close(fig)

    # Static KL plot
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.scatter(result["inv_sqrt_omega"], result["inv_current"] / 1000.0, label="Messpunkte")
    order = np.argsort(result["inv_sqrt_omega"])
    ax.plot(result["inv_sqrt_omega"][order], result["kl_fit"][order] / 1000.0,
            linestyle="--", label="Lineare Regression")
    ax.set_xlabel(r"$1/\sqrt{\omega}$ / $(rad\,s^{-1})^{-1/2}$")
    ax.set_ylabel(r"$1/|I_D|$ / mA$^{-1}$")
    ax.set_title(f"Koutecký–Levich-Analysis bei E = {result['kl_potential']:.3f} V")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.text(0.98, 0.04, f"R² = {result['kl_r2']:.5f}", transform=ax.transAxes,
            ha="right", va="bottom")
    fig.tight_layout()
    fig.savefig(prefix.with_name(prefix.name + "_Koutecky_Levich.png"), dpi=220)
    plt.close(fig)

    # Interactive combined HTML
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Levich", "Koutecký–Levich"))
    fig.add_trace(go.Scatter(x=result["sqrt_omega"], y=result["levich_currents_fit_a"] * 1e3,
                             mode="markers", name="Levich-Messpunkte"), row=1, col=1)
    fig.add_trace(go.Scatter(x=result["sqrt_omega"][np.argsort(result["sqrt_omega"])],
                             y=result["levich_fit"][np.argsort(result["sqrt_omega"])] * 1e3,
                             mode="lines", name="Levich-Fit"), row=1, col=1)
    fig.add_trace(go.Scatter(x=result["inv_sqrt_omega"], y=result["inv_current"] / 1000.0,
                             mode="markers", name="KL-Messpunkte"), row=1, col=2)
    fig.add_trace(go.Scatter(x=result["inv_sqrt_omega"][np.argsort(result["inv_sqrt_omega"])],
                             y=result["kl_fit"][np.argsort(result["inv_sqrt_omega"])] / 1000.0,
                             mode="lines", name="KL-Fit"), row=1, col=2)
    fig.update_xaxes(title_text="√ω / (rad s⁻¹)¹ᐟ²", row=1, col=1)
    fig.update_yaxes(title_text="|I_D| / mA", row=1, col=1)
    fig.update_xaxes(title_text="1/√ω / (rad s⁻¹)⁻¹ᐟ²", row=1, col=2)
    fig.update_yaxes(title_text="1/|I_D| / mA⁻¹", row=1, col=2)
    fig.update_layout(title=(f"Levich bei E_L = {result['levich_potential']:.3f} V · "
                             f"Koutecký–Levich bei E_KL = {result['kl_potential']:.3f} V"),
                      width=1250, height=580)
    fig.write_html(prefix.with_name(prefix.name + "_Levich_KL.html"), include_plotlyjs=True)

    # Excel report
    xlsx_path = prefix.with_name(prefix.name + "_Levich_KL.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
        params = [
            ("Levich-Potential E_L / V", result["levich_potential"]),
            ("Koutecky-Levich-Potential E_KL / V", result["kl_potential"]),
            ("Elektrodenoberfläche / cm²", result["area_cm2"]),
            ("Konzentration / mol L⁻¹", result["concentration_mol_l"]),
            ("Kinematische Viskosität / cm² s⁻¹", result["viscosity_cm2_s"]),
            ("Modus", "D bestimmen" if result["mode"] == "D" else "n bestimmen"),
            ("Elektronenzahl n", result["n_value"]),
            ("Diffusionskoeffizient D / cm² s⁻¹", result["diffusion_cm2_s"]),
            ("Levich-Steigung / A (rad s⁻¹)^-1/2", result["levich_slope"]),
            ("Levich-Achsenabschnitt / A", result["levich_intercept"]),
            ("Levich R²", result["levich_r2"]),
            ("KL-Steigung / A⁻¹ (rad s⁻¹)^1/2", result["kl_slope"]),
            ("KL-Achsenabschnitt / A⁻¹", result["kl_intercept"]),
            ("KL R²", result["kl_r2"]),
            ("Kinetischer Strom I_k / A", result["kinetic_current_a"]),
        ]
        pd.DataFrame(params, columns=["Parameter", "Wert"]).to_excel(writer, sheet_name="Ergebnisse", index=False)
        data = pd.DataFrame({
            "rpm": result["rpm"],
            "omega_rad_s": result["omega"],
            "sqrt_omega": result["sqrt_omega"],
            "Levich_Disk current_roh_A": result["levich_currents_raw_a"],
            "Levich_Disk current_Analysis_A": result["levich_currents_fit_a"],
            "KL_Disk current_roh_A": result["kl_currents_raw_a"],
            "KL_Disk current_Analysis_A": result["kl_currents_fit_a"],
            "Levich_Fit_A": result["levich_fit"],
            "inv_sqrt_omega": result["inv_sqrt_omega"],
            "inv_KL_Disk current_1_A": result["inv_current"],
            "KL_Fit_1_A": result["kl_fit"],
        })
        data.to_excel(writer, sheet_name="Daten", index=False)
        wb = writer.book
        ws = writer.sheets["Ergebnisse"]
        ws.set_column("A:A", 42)
        ws.set_column("B:B", 18)
        wd = writer.sheets["Daten"]
        wd.set_column("A:K", 20)

        lev = wb.add_chart({"type": "scatter", "subtype": "straight_with_markers"})
        lev.add_series({"name": "Messpunkte", "categories": ["Daten", 1, 2, len(data), 2],
                        "values": ["Daten", 1, 4, len(data), 4], "marker": {"type": "circle"},
                        "line": {"none": True}})
        lev.add_series({"name": "Regression", "categories": ["Daten", 1, 2, len(data), 2],
                        "values": ["Daten", 1, 7, len(data), 7], "marker": {"type": "none"}})
        lev.set_title({"name": "Levich"}); lev.set_x_axis({"name": "sqrt(omega)"}); lev.set_y_axis({"name": "|I_D| / A"})
        wd.insert_chart("K2", lev, {"x_scale": 1.25, "y_scale": 1.25})

        kl = wb.add_chart({"type": "scatter", "subtype": "straight_with_markers"})
        kl.add_series({"name": "Messpunkte", "categories": ["Daten", 1, 10, len(data), 10],
                       "values": ["Daten", 1, 9, len(data), 9], "marker": {"type": "circle"},
                       "line": {"none": True}})
        kl.add_series({"name": "Regression", "categories": ["Daten", 1, 10, len(data), 10],
                       "values": ["Daten", 1, 10, len(data), 10], "marker": {"type": "none"}})
        kl.set_title({"name": "Koutecky-Levich"}); kl.set_x_axis({"name": "1/sqrt(omega)"}); kl.set_y_axis({"name": "1/|I_D| / 1/A"})
        wd.insert_chart("K20", kl, {"x_scale": 1.25, "y_scale": 1.25})


class NovaProposalDialog(tk.Toplevel):
    """Einfacher, bestätigungspflichtiger Importdialog für Metrohm NOVA."""

    def __init__(self, parent, path: Path, proposal):
        super().__init__(parent)
        self.title("Metrohm NOVA – RRDE-Import")
        self.geometry("900x690")
        self.minsize(800, 580)
        self.transient(parent)
        self.grab_set()

        self.path = path
        self.proposal = proposal
        self.result = None
        self.edit_requested = False

        main = ttk.Frame(self, padding=12)
        main.pack(fill="both", expand=True)

        ttk.Label(
            main,
            text="Metrohm NOVA – RRDE-Import",
            font=("Segoe UI", 17, "bold"),
        ).pack(anchor="w")

        ttk.Label(
            main,
            text=(
                "Die NOVA-Datei wird nach der festgelegten RRDE-Struktur gelesen: "
                "erste Spalte = Potential; danach jeweils Disk und Ring. "
                "Bitte kontrollieren Sie nur kurz die Übersicht und klicken Sie auf Importieren."
            ),
            wraplength=850,
        ).pack(anchor="w", pady=(0, 10))

        info = ttk.LabelFrame(main, text="Erkannte Struktur", padding=10)
        info.pack(fill="x", pady=4)
        ttk.Label(info, text=f"Datei: {path}", wraplength=820).pack(anchor="w")
        ttk.Label(
            info,
            text=f"Potential: {proposal['potential']}",
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w", pady=(5, 0))
        ttk.Label(
            info,
            text=(
                f"{len(proposal['pairs'])} Rotationsgeschwindigkeiten gefunden; "
                "Disk = erste Spalte jedes Paares, Ring = zweite Spalte."
            ),
        ).pack(anchor="w", pady=(3, 0))

        table_box = ttk.LabelFrame(
            main,
            text="Zuordnung – bitte kontrollieren",
            padding=8,
        )
        table_box.pack(fill="x", pady=5)

        tree = ttk.Treeview(
            table_box,
            columns=("rpm", "disk", "ring"),
            show="headings",
            height=min(12, max(4, len(proposal["display_rows"]))),
        )
        tree.heading("rpm", text="Rotation / U min⁻¹")
        tree.heading("disk", text="Disk")
        tree.heading("ring", text="Ring")
        tree.column("rpm", width=180, anchor="center")
        tree.column("disk", width=250, anchor="center")
        tree.column("ring", width=250, anchor="center")

        for row in proposal["display_rows"]:
            tree.insert(
                "",
                "end",
                values=(
                    format_rpm(row["rotation"]),
                    f"Spalte {row['disk_number']}",
                    f"Spalte {row['ring_number']}",
                ),
            )

        scrollbar = ttk.Scrollbar(table_box, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        tree.pack(side="left", fill="x", expand=True)
        scrollbar.pack(side="right", fill="y")

        preview_box = ttk.LabelFrame(
            main,
            text="Datenvorschau – verständliche Spaltennamen",
            padding=8,
        )
        preview_box.pack(fill="both", expand=True, pady=5)

        preview_columns = [f"c{i}" for i in range(len(proposal["display_headers"]))]
        preview_tree = ttk.Treeview(
            preview_box,
            columns=preview_columns,
            show="headings",
            height=5,
        )

        for key, label in zip(preview_columns, proposal["display_headers"]):
            preview_tree.heading(key, text=label)
            preview_tree.column(key, width=130, anchor="center", stretch=True)

        df = proposal["dataframe"]
        for _, row in df.head(5).iterrows():
            values = [row.iloc[i] if i < len(row) else "" for i in range(len(preview_columns))]
            preview_tree.insert("", "end", values=values)

        xscroll = ttk.Scrollbar(
            preview_box,
            orient="horizontal",
            command=preview_tree.xview,
        )
        preview_tree.configure(xscrollcommand=xscroll.set)
        preview_tree.pack(fill="both", expand=True)
        xscroll.pack(fill="x")

        ttk.Label(
            main,
            text=(
                "Leere NOVA-Überschriften werden nur in dieser Anzeige ersetzt: "
                "beispielsweise 'Ring 100' statt 'Unnamed'. Die Messwerte bleiben unverändert."
            ),
            wraplength=850,
        ).pack(anchor="w", pady=5)

        buttons = ttk.Frame(main)
        buttons.pack(fill="x", pady=(8, 0))

        ttk.Button(
            buttons,
            text="Abbrechen",
            command=self.destroy,
        ).pack(side="right")

        ttk.Button(
            buttons,
            text="Zuordnung bearbeiten …",
            command=self.edit,
        ).pack(side="right", padx=8)

        ttk.Button(
            buttons,
            text="Importieren",
            command=self.accept,
        ).pack(side="right")

        self.protocol("WM_DELETE_WINDOW", self.destroy)

    def accept(self):
        try:
            parse_rrde_manual(
                self.path,
                self.proposal["potential"],
                self.proposal["pairs"],
            )
        except Exception as exc:
            messagebox.showerror("NOVA-Import", str(exc), parent=self)
            return

        self.result = {
            "potential": self.proposal["potential"],
            "pairs": self.proposal["pairs"],
        }
        self.destroy()

    def edit(self):
        self.edit_requested = True
        self.destroy()


class ManualImportDialog(tk.Toplevel):
    """Dialog zur vollständig manuellen Zuordnung der RRDE-Spalten."""

    def __init__(self, parent, path: Path, existing=None):
        super().__init__(parent)
        self.title("RRDE-Spaltenzuordnung")
        self.geometry("920x680")
        self.minsize(820, 560)
        self.transient(parent)
        self.grab_set()

        self.path = path
        self.result = None
        self.df = read_csv_robust(path)
        self.columns = [str(c) for c in self.df.columns]
        self.row_widgets = []

        top = ttk.Frame(self, padding=12)
        top.pack(fill="both", expand=True)

        ttk.Label(
            top,
            text="Manuelle RRDE-Spaltenzuordnung",
            font=("Segoe UI", 16, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            top,
            text=(
                "Legen Sie die Potentialspalte sowie für jede Rotationsgeschwindigkeit "
                "die zugehörige Disk- und Ringspalte fest. Erst nach Bestätigung wird importiert."
            ),
            wraplength=860,
        ).pack(anchor="w", pady=(0, 10))

        file_box = ttk.LabelFrame(top, text="Datei", padding=8)
        file_box.pack(fill="x", pady=4)
        ttk.Label(file_box, text=str(path), wraplength=830).pack(anchor="w")

        potential_box = ttk.LabelFrame(top, text="Potential", padding=8)
        potential_box.pack(fill="x", pady=4)
        ttk.Label(potential_box, text="Potentialspalte:").pack(side="left")
        self.potential_var = tk.StringVar(
            value=(existing.get("potential") if existing else (self.columns[0] if self.columns else ""))
        )
        ttk.Combobox(
            potential_box,
            textvariable=self.potential_var,
            values=self.columns,
            state="readonly",
            width=55,
        ).pack(side="left", padx=8)

        pairs_box = ttk.LabelFrame(top, text="Disk-/Ring-Zuordnung", padding=8)
        pairs_box.pack(fill="both", expand=True, pady=4)

        toolbar = ttk.Frame(pairs_box)
        toolbar.pack(fill="x", pady=(0, 6))
        ttk.Button(toolbar, text="Zeile hinzufügen", command=self.add_row).pack(side="left")
        ttk.Button(toolbar, text="Letzte Zeile entfernen", command=self.remove_row).pack(side="left", padx=6)
        ttk.Button(
            toolbar,
            text="Spaltenfolge als Vorschlag eintragen",
            command=self.prepare_sequential_pairs,
        ).pack(side="left", padx=12)

        ttk.Label(
            toolbar,
            text="Der Vorschlag wird erst nach Ihrer Bestätigung verwendet.",
        ).pack(side="left", padx=6)

        canvas = tk.Canvas(pairs_box, highlightthickness=0)
        scrollbar = ttk.Scrollbar(pairs_box, orient="vertical", command=canvas.yview)
        self.rows_frame = ttk.Frame(canvas)
        self.rows_frame.bind(
            "<Configure>",
            lambda _e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.create_window((0, 0), window=self.rows_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        headers = ["Verwenden", "Rotation / U min⁻¹", "Diskspalte", "Ringspalte"]
        for col, label in enumerate(headers):
            ttk.Label(self.rows_frame, text=label, font=("Segoe UI", 9, "bold")).grid(
                row=0, column=col, sticky="w", padx=5, pady=4
            )
        self.rows_frame.columnconfigure(2, weight=1)
        self.rows_frame.columnconfigure(3, weight=1)

        existing_pairs = existing.get("pairs", []) if existing else []
        if existing_pairs:
            for rpm, disk_col, ring_col in existing_pairs:
                self.add_row(rpm, disk_col, ring_col, True)
        else:
            self.add_row()

        preview = ttk.LabelFrame(top, text="Dateivorschau – erste fünf Zeilen", padding=6)
        preview.pack(fill="x", pady=4)
        tree = ttk.Treeview(preview, columns=self.columns, show="headings", height=5)
        for col in self.columns:
            tree.heading(col, text=col)
            tree.column(col, width=125, stretch=True)
        for _, row in self.df.head(5).iterrows():
            tree.insert("", "end", values=[row.get(c, "") for c in self.columns])
        xscroll = ttk.Scrollbar(preview, orient="horizontal", command=tree.xview)
        tree.configure(xscrollcommand=xscroll.set)
        tree.pack(fill="x")
        xscroll.pack(fill="x")

        buttons = ttk.Frame(top)
        buttons.pack(fill="x", pady=(10, 0))
        ttk.Button(buttons, text="Abbrechen", command=self.destroy).pack(side="right")
        ttk.Button(
            buttons,
            text="Zuordnung bestätigen",
            command=self.confirm,
        ).pack(side="right", padx=8)

        self.protocol("WM_DELETE_WINDOW", self.destroy)

    def add_row(self, rpm="", disk_col="", ring_col="", enabled=True):
        row_index = len(self.row_widgets) + 1
        use_var = tk.BooleanVar(value=enabled)
        rpm_var = tk.StringVar(value=str(rpm))
        disk_var = tk.StringVar(value=str(disk_col))
        ring_var = tk.StringVar(value=str(ring_col))

        ttk.Checkbutton(self.rows_frame, variable=use_var).grid(
            row=row_index, column=0, padx=5, pady=3
        )
        ttk.Entry(self.rows_frame, textvariable=rpm_var, width=16).grid(
            row=row_index, column=1, padx=5, pady=3, sticky="ew"
        )
        ttk.Combobox(
            self.rows_frame, textvariable=disk_var,
            values=self.columns, state="readonly", width=34
        ).grid(row=row_index, column=2, padx=5, pady=3, sticky="ew")
        ttk.Combobox(
            self.rows_frame, textvariable=ring_var,
            values=self.columns, state="readonly", width=34
        ).grid(row=row_index, column=3, padx=5, pady=3, sticky="ew")

        self.row_widgets.append((use_var, rpm_var, disk_var, ring_var))

    def remove_row(self):
        if not self.row_widgets:
            return
        widgets = self.rows_frame.grid_slaves(row=len(self.row_widgets))
        for widget in widgets:
            widget.destroy()
        self.row_widgets.pop()

    def prepare_sequential_pairs(self):
        """Komfortfunktion; die Zuordnung bleibt sichtbar und muss bestätigt werden."""
        for row in list(range(len(self.row_widgets))):
            widgets = self.rows_frame.grid_slaves(row=row + 1)
            for widget in widgets:
                widget.destroy()
        self.row_widgets.clear()

        usable = [c for c in self.columns if c != self.potential_var.get()]
        for i in range(0, len(usable) - 1, 2):
            disk_col = usable[i]
            ring_col = usable[i + 1]
            rpm = natural_float(disk_col)
            rpm_text = "" if rpm is None else format_rpm(rpm)
            self.add_row(rpm_text, disk_col, ring_col, True)

    def confirm(self):
        potential = self.potential_var.get().strip()
        if not potential:
            messagebox.showerror("Spaltenzuordnung", "Bitte wählen Sie eine Potentialspalte.", parent=self)
            return

        pairs = []
        try:
            for use_var, rpm_var, disk_var, ring_var in self.row_widgets:
                if not use_var.get():
                    continue
                rpm_text = rpm_var.get().strip().replace(",", ".")
                disk_col = disk_var.get().strip()
                ring_col = ring_var.get().strip()
                if not rpm_text and not disk_col and not ring_col:
                    continue
                if not rpm_text:
                    raise ValueError("Bei einer verwendeten Zeile fehlt die Rotationsgeschwindigkeit.")
                rpm = float(rpm_text)
                if rpm <= 0:
                    raise ValueError("Rotationsgeschwindigkeiten müssen größer als 0 sein.")
                if not disk_col or not ring_col:
                    raise ValueError(f"Bei {rpm:g} U/min fehlen Disk- oder Ringspalte.")
                pairs.append((rpm, disk_col, ring_col))

            if not pairs:
                raise ValueError("Bitte ordnen Sie mindestens ein Disk-/Ring-Spaltenpaar zu.")

            # Validierung durch einen Testimport.
            parse_rrde_manual(self.path, potential, pairs)

        except Exception as exc:
            messagebox.showerror("Ungültige Zuordnung", str(exc), parent=self)
            return

        self.result = {"potential": potential, "pairs": pairs}
        self.destroy()


class RRDEApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"{APP_TITLE} v{VERSION}")
        self.geometry("1000x830")
        self.minsize(920, 760)

        self.csv_var = tk.StringVar()
        self.file_status_var = tk.StringVar(value="Noch keine CSV-Datei ausgewählt.")
        self.output_var = tk.StringVar()
        self.name_var = tk.StringVar(value="RRDE_Messung")
        self.smooth_var = tk.BooleanVar(value=True)
        self.window_var = tk.StringVar(value="15")
        self.poly_var = tk.StringVar(value="3")
        self.disk_unit_var = tk.StringVar(value="µA")
        self.ring_unit_var = tk.StringVar(value="µA")
        self.show_raw_var = tk.BooleanVar(value=True)
        self.reverse_x_var = tk.BooleanVar(value=False)
        self.open_result_var = tk.BooleanVar(value=True)
        self.ring_scale_mode_var = tk.StringVar(value="auto")
        self.ring_factor_var = tk.StringVar(value="20")
        self.ring_bg_enabled_var = tk.BooleanVar(value=False)
        self.ring_bg_method_var = tk.StringVar(value="range_mean")
        self.ring_bg_from_var = tk.StringVar(value="-1.70")
        self.ring_bg_to_var = tk.StringVar(value="-1.30")
        self.ring_bg_offset_var = tk.StringVar(value="0.0")
        self.background_csv_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Bereit.")
        self.lkl_enabled_var = tk.BooleanVar(value=False)
        self.lkl_mode_var = tk.StringVar(value="D")
        self.lkl_levich_potential_var = tk.StringVar(value="-0.90")
        self.lkl_kl_potential_var = tk.StringVar(value="-0.50")
        self.lkl_area_var = tk.StringVar(value="0.196")
        self.lkl_concentration_var = tk.StringVar(value="1.20E-3")
        self.lkl_viscosity_var = tk.StringVar(value="0.0100")
        self.lkl_n_var = tk.StringVar(value="4")
        self.lkl_d_var = tk.StringVar(value="2.00E-5")
        self.lkl_abs_var = tk.BooleanVar(value=True)
        self.h2o2_enabled_var = tk.BooleanVar(value=False)
        self.h2o2_abs_disk_var = tk.BooleanVar(value=True)
        self.collection_mode_var = tk.StringVar(value="manufacturer")
        self.collection_manufacturer_var = tk.StringVar(value="0.250")
        self.collection_manual_var = tk.StringVar(value="0.230")
        self.h2o2_potential_from_var = tk.StringVar(value="-0.80")
        self.h2o2_potential_to_var = tk.StringVar(value="0.20")
        self.tafel_enabled_var = tk.BooleanVar(value=False)
        self.tafel_reaction_var = tk.StringVar(value="")
        self.tafel_reference_var = tk.StringVar(value="")
        self.tafel_ph_var = tk.StringVar(value="")
        self.tafel_temperature_var = tk.StringVar(value="25.0")
        self.tafel_user_eq_var = tk.StringVar(value="")
        self.tafel_potential_from_var = tk.StringVar(value="-0.20")
        self.tafel_potential_to_var = tk.StringVar(value="0.20")
        self.tafel_current_mode_var = tk.StringVar(value="density")
        self.tafel_area_var = tk.StringVar(value="0.196")
        self.tafel_kl_corrected_var = tk.BooleanVar(value=False)
        self.tafel_limiting_from_var = tk.StringVar(value="-1.10")
        self.tafel_limiting_to_var = tk.StringVar(value="-0.90")
        self.tafel_k0_enabled_var = tk.BooleanVar(value=False)
        self.tafel_n_var = tk.StringVar(value="1")
        self.tafel_alpha_var = tk.StringVar(value="0.5")
        self.tafel_co_var = tk.StringVar(value="0.005")
        self.tafel_cr_var = tk.StringVar(value="0.005")
        self.tafel_eq_display_var = tk.StringVar(value="Select reaction, reference electrode and pH.")
        self.last_folder: Optional[Path] = None

        self.build_ui()

    def build_ui(self):
        # Fester unterer Aktionsbereich + scrollbarerer Inhaltsbereich.
        # Dadurch bleibt "Analysis starten" immer sichtbar.
        outer = ttk.Frame(self)
        outer.pack(fill="both", expand=True)

        action_bar = ttk.Frame(outer, padding=(12, 8))
        action_bar.pack(side="bottom", fill="x")

        self.start_btn = ttk.Button(
            action_bar,
            text="Analysis starten",
            command=self.run_analysis,
        )
        self.start_btn.pack(side="left")

        ttk.Button(
            action_bar,
            text="Letzten Ergebnisordner öffnen",
            command=self.open_last,
        ).pack(side="left", padx=8)

        self.progress = ttk.Progressbar(
            action_bar,
            mode="indeterminate",
            length=270,
        )
        self.progress.pack(side="right")

        content_holder = ttk.Frame(outer)
        content_holder.pack(side="top", fill="both", expand=True)

        canvas = tk.Canvas(content_holder, highlightthickness=0)
        vscroll = ttk.Scrollbar(
            content_holder,
            orient="vertical",
            command=canvas.yview,
        )
        canvas.configure(yscrollcommand=vscroll.set)

        vscroll.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        main = ttk.Frame(canvas, padding=12)
        canvas_window = canvas.create_window((0, 0), window=main, anchor="nw")

        def update_scrollregion(_event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def resize_inner(event):
            canvas.itemconfigure(canvas_window, width=event.width)

        main.bind("<Configure>", update_scrollregion)
        canvas.bind("<Configure>", resize_inner)

        # Mausradunterstützung für Windows/Linux.
        def on_mousewheel(event):
            if getattr(event, "delta", 0):
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            elif getattr(event, "num", None) == 4:
                canvas.yview_scroll(-1, "units")
            elif getattr(event, "num", None) == 5:
                canvas.yview_scroll(1, "units")

        canvas.bind_all("<MouseWheel>", on_mousewheel)
        canvas.bind_all("<Button-4>", on_mousewheel)
        canvas.bind_all("<Button-5>", on_mousewheel)

        ttk.Label(
            main,
            text="RRDE-Analysis",
            font=("Segoe UI", 20, "bold"),
        ).pack(anchor="w")

        ttk.Label(
            main,
            text=(
                "Disk- und Ringströme: Glättung, 2D, getrennte 3D-Darstellung, "
                "gemeinsame 3D-Darstellung, Excel-Report und optionale Levich/Koutecký–Levich-Analysis."
            ),
        ).pack(anchor="w", pady=(0, 10))

        inp = ttk.LabelFrame(main, text="Eingabe und Ausgabe", padding=10)
        inp.pack(fill="x", pady=4)
        inp.columnconfigure(1, weight=1)

        ttk.Label(inp, text="Messdatei (CSV):").grid(
            row=0, column=0, sticky="w", pady=4
        )
        ttk.Entry(inp, textvariable=self.csv_var).grid(
            row=0, column=1, sticky="ew", padx=8
        )
        ttk.Button(
            inp,
            text="Durchsuchen",
            command=self.browse_csv,
        ).grid(row=0, column=2)

        ttk.Label(inp, text="Dateistatus:").grid(
            row=1, column=0, sticky="nw", pady=4
        )
        ttk.Label(
            inp,
            textvariable=self.file_status_var,
            wraplength=680,
        ).grid(row=1, column=1, sticky="w", padx=8, pady=4)

        ttk.Button(
            inp,
            text="CSV-Struktur …",
            command=self.show_csv_structure,
        ).grid(row=1, column=2, sticky="n")

        ttk.Label(inp, text="Ausgabeordner:").grid(
            row=2, column=0, sticky="w", pady=4
        )
        ttk.Entry(inp, textvariable=self.output_var).grid(
            row=2, column=1, sticky="ew", padx=8
        )
        ttk.Button(
            inp,
            text="Durchsuchen",
            command=self.browse_output,
        ).grid(row=2, column=2)

        ttk.Label(inp, text="Messungsname:").grid(
            row=3, column=0, sticky="w", pady=4
        )
        ttk.Entry(
            inp,
            textvariable=self.name_var,
            width=45,
        ).grid(row=3, column=1, sticky="w", padx=8)

        # Zusätzlicher Startbereich im oberen sichtbaren Teil des Fensters.
        quick_start = ttk.Frame(main, padding=(0, 8))
        quick_start.pack(fill="x")

        self.quick_start_button = tk.Button(
            quick_start,
            text="▶  AUSWERTUNG STARTEN\n(oder Taste F5)",
            command=self.run_analysis,
            bg="#078A16",
            fg="white",
            activebackground="#056F12",
            activeforeground="white",
            font=("Segoe UI", 15, "bold"),
            relief="flat",
            cursor="hand2",
            padx=18,
            pady=10,
        )
        self.quick_start_button.pack(fill="x", expand=True)

        ttk.Label(
            quick_start,
            text="CSV file wird direkt ohne Prüfdialog ausgewertet.",
        ).pack(anchor="w", pady=(5, 0))

        smooth_box = ttk.LabelFrame(main, text="Glättung", padding=10)
        smooth_box.pack(fill="x", pady=4)

        ttk.Checkbutton(
            smooth_box,
            text="Savitzky–Golay-Glättung anwenden",
            variable=self.smooth_var,
        ).grid(row=0, column=0, columnspan=2, sticky="w")

        ttk.Label(
            smooth_box,
            text="Fensterbreite / Punkte:",
        ).grid(row=1, column=0, sticky="w", pady=5)

        ttk.Entry(
            smooth_box,
            textvariable=self.window_var,
            width=10,
        ).grid(row=1, column=1, sticky="w", padx=8)

        ttk.Label(
            smooth_box,
            text="Polynomgrad:",
        ).grid(row=1, column=2, sticky="w", padx=(30, 0))

        ttk.Entry(
            smooth_box,
            textvariable=self.poly_var,
            width=10,
        ).grid(row=1, column=3, sticky="w", padx=8)

        display = ttk.LabelFrame(main, text="Darstellung", padding=10)
        display.pack(fill="x", pady=4)

        ttk.Label(display, text="Disk current-Einheit:").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Combobox(
            display,
            textvariable=self.disk_unit_var,
            values=["A", "mA", "µA"],
            state="readonly",
            width=8,
        ).grid(row=0, column=1, sticky="w", padx=8)

        ttk.Label(display, text="Ring current-Einheit:").grid(
            row=0, column=2, sticky="w", padx=(25, 0)
        )
        ttk.Combobox(
            display,
            textvariable=self.ring_unit_var,
            values=["A", "mA", "µA", "nA"],
            state="readonly",
            width=8,
        ).grid(row=0, column=3, sticky="w", padx=8)

        ttk.Checkbutton(
            display,
            text="Rohkurven zusätzlich schwach darstellen",
            variable=self.show_raw_var,
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=6)

        ttk.Checkbutton(
            display,
            text="Potentialachse umkehren",
            variable=self.reverse_x_var,
        ).grid(row=1, column=2, sticky="w", padx=(25, 0))

        ttk.Checkbutton(
            display,
            text="Ergebnisordner nach Abschluss öffnen",
            variable=self.open_result_var,
        ).grid(row=2, column=0, columnspan=2, sticky="w")

        ttk.Label(
            display,
            text="Ring current-Skalierung im gemeinsamen 3D-Diagramm:",
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(10, 3))

        ttk.Radiobutton(
            display,
            text="Automatisch",
            variable=self.ring_scale_mode_var,
            value="auto",
            command=self.update_factor_state,
        ).grid(row=4, column=0, sticky="w")

        ttk.Radiobutton(
            display,
            text="Manuell",
            variable=self.ring_scale_mode_var,
            value="manual",
            command=self.update_factor_state,
        ).grid(row=4, column=1, sticky="w")

        ttk.Label(
            display,
            text="Multiplikationsfaktor:",
        ).grid(row=4, column=2, sticky="e", padx=(25, 4))

        self.ring_factor_entry = ttk.Entry(
            display,
            textvariable=self.ring_factor_var,
            width=10,
            state="disabled",
        )
        self.ring_factor_entry.grid(row=4, column=3, sticky="w")

        bg = ttk.LabelFrame(
            main,
            text="Untergrundkompensation",
            padding=10,
        )
        bg.pack(fill="x", pady=4)

        ttk.Checkbutton(
            bg,
            text="Untergrund kompensieren",
            variable=self.ring_bg_enabled_var,
            command=self.update_background_state,
        ).grid(row=0, column=0, columnspan=4, sticky="w")

        ttk.Radiobutton(
            bg,
            text="Mittelwert im Potentialbereich",
            variable=self.ring_bg_method_var,
            value="range_mean",
            command=self.update_background_state,
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(8, 4))

        ttk.Label(bg, text="Von / V:").grid(
            row=2, column=0, sticky="e", padx=(0, 4)
        )
        self.ring_bg_from_entry = ttk.Entry(
            bg,
            textvariable=self.ring_bg_from_var,
            width=10,
        )
        self.ring_bg_from_entry.grid(row=2, column=1, sticky="w")

        ttk.Label(bg, text="Bis / V:").grid(
            row=2, column=2, sticky="e", padx=(25, 4)
        )
        self.ring_bg_to_entry = ttk.Entry(
            bg,
            textvariable=self.ring_bg_to_var,
            width=10,
        )
        self.ring_bg_to_entry.grid(row=2, column=3, sticky="w")

        ttk.Radiobutton(
            bg,
            text="Manuellen konstanten Offset abziehen",
            variable=self.ring_bg_method_var,
            value="manual",
            command=self.update_background_state,
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(10, 4))

        ttk.Label(bg, text="Offset:").grid(
            row=4, column=0, sticky="e", padx=(0, 4)
        )
        self.ring_bg_offset_entry = ttk.Entry(
            bg,
            textvariable=self.ring_bg_offset_var,
            width=12,
        )
        self.ring_bg_offset_entry.grid(row=4, column=1, sticky="w")
        ttk.Label(
            bg,
            textvariable=self.ring_unit_var,
        ).grid(row=4, column=2, sticky="w", padx=(5, 0))

        ttk.Radiobutton(
            bg,
            text="Separate N₂-/Hintergrundmessung aus CSV abziehen (Disk + Ring)",
            variable=self.ring_bg_method_var,
            value="measurement_csv",
            command=self.update_background_state,
        ).grid(row=5, column=0, columnspan=4, sticky="w", pady=(10, 4))

        self.background_csv_entry = ttk.Entry(
            bg, textvariable=self.background_csv_var, width=72
        )
        self.background_csv_entry.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(0, 4))
        self.background_csv_button = ttk.Button(
            bg, text="CSV auswählen …", command=self.browse_background_csv
        )
        self.background_csv_button.grid(row=6, column=3, sticky="e", padx=(8, 0), pady=(0, 4))

        ttk.Label(
            bg,
            text=(
                "Bei einer separaten Hintergrundmessung werden Disk- und Ringstrom für jede "
                "Rotation punktweise abgezogen. Potentialraster, Scanrichtung und Rotationen "
                "müssen mit der O₂-Messung übereinstimmen. Bereich und manueller Offset "
                "korrigieren weiterhin nur den Ringstrom."
            ),
            wraplength=850,
        ).grid(row=7, column=0, columnspan=4, sticky="w", pady=(8, 0))

        h2o2 = ttk.LabelFrame(
            main,
            text="H₂O₂-Ausbeute und n(E)",
            padding=10,
        )
        h2o2.pack(fill="x", pady=4)

        ttk.Checkbutton(
            h2o2,
            text="H₂O₂-Ausbeute und Elektronenzahl n(E) zusätzlich berechnen",
            variable=self.h2o2_enabled_var,
            command=self.update_h2o2_state,
        ).grid(row=0, column=0, columnspan=5, sticky="w", pady=(0, 8))

        ttk.Label(h2o2, text="Collection Efficiency:").grid(row=1, column=0, sticky="w")
        self.collection_manufacturer_radio = ttk.Radiobutton(
            h2o2, text="Herstellerwert", variable=self.collection_mode_var,
            value="manufacturer", command=self.update_h2o2_state
        )
        self.collection_manufacturer_radio.grid(row=1, column=1, sticky="w", padx=6)
        self.collection_manufacturer_entry = ttk.Entry(
            h2o2, textvariable=self.collection_manufacturer_var, width=10
        )
        self.collection_manufacturer_entry.grid(row=1, column=2, sticky="w")

        self.collection_manual_radio = ttk.Radiobutton(
            h2o2, text="Experimenteller/manueller Wert", variable=self.collection_mode_var,
            value="manual", command=self.update_h2o2_state
        )
        self.collection_manual_radio.grid(row=2, column=1, sticky="w", padx=6, pady=(5, 0))
        self.collection_manual_entry = ttk.Entry(
            h2o2, textvariable=self.collection_manual_var, width=10
        )
        self.collection_manual_entry.grid(row=2, column=2, sticky="w", pady=(5, 0))

        ttk.Label(h2o2, text="Potentialbereich für H₂O₂ und n(E):").grid(
            row=3, column=0, sticky="w", pady=(10, 0)
        )
        ttk.Label(h2o2, text="von").grid(row=3, column=1, sticky="e", pady=(10, 0))
        self.h2o2_potential_from_entry = ttk.Entry(
            h2o2, textvariable=self.h2o2_potential_from_var, width=10
        )
        self.h2o2_potential_from_entry.grid(row=3, column=2, sticky="w", pady=(10, 0))
        ttk.Label(h2o2, text="V   bis").grid(row=3, column=3, sticky="e", pady=(10, 0))
        self.h2o2_potential_to_entry = ttk.Entry(
            h2o2, textvariable=self.h2o2_potential_to_var, width=10
        )
        self.h2o2_potential_to_entry.grid(row=3, column=4, sticky="w", pady=(10, 0))
        ttk.Label(h2o2, text="V").grid(row=3, column=5, sticky="w", pady=(10, 0))

        self.h2o2_use_plot_range_button = ttk.Button(
            h2o2, text="Aktuellen Plotbereich verwenden …",
            command=self.choose_h2o2_plot_range,
        )
        self.h2o2_use_plot_range_button.grid(
            row=4, column=1, columnspan=4, sticky="w", padx=6, pady=(7, 0)
        )

        self.h2o2_abs_disk_check = ttk.Checkbutton(
            h2o2,
            text="Use absolute values of cathodic disk currents for H₂O₂ yield and electron-transfer number",
            variable=self.h2o2_abs_disk_var,
        )
        self.h2o2_abs_disk_check.grid(row=5, column=0, columnspan=6, sticky="w", pady=(9, 0))
        ttk.Label(
            h2o2,
            text="Disk currents are linearly interpolated at the selected potentials before calculating the H₂O₂ yield and electron-transfer number.",
            wraplength=850,
        ).grid(row=6, column=0, columnspan=6, sticky="w", pady=(4, 0))

        ttk.Label(
            h2o2,
            text=("Zoomen Sie im Auswahlfenster auf den gewünschten ORR-Bereich und übernehmen "
                  "Sie anschließend den aktuell sichtbaren x-Achsenbereich. Außerhalb dieses "
                  "Bereichs werden H₂O₂-Ausbeute und n(E) nicht berechnet."),
            wraplength=850,
        ).grid(row=5, column=0, columnspan=6, sticky="w", pady=(8, 0))

        ttk.Label(
            h2o2,
            text=("Die vorhandene Ring-Untergrundkorrektur wird vor der Berechnung angewendet. "
                  "Alle Rotationsgeschwindigkeiten werden ausgewertet; Exporte sind auf Englisch."),
            wraplength=850,
        ).grid(row=6, column=0, columnspan=6, sticky="w", pady=(8, 0))

        tafel = ttk.LabelFrame(main, text="Tafel analysis (aqueous systems)", padding=10)
        tafel.pack(fill="x", pady=4)
        ttk.Checkbutton(
            tafel, text="Calculate Tafel analysis additionally",
            variable=self.tafel_enabled_var, command=self.update_tafel_state,
        ).grid(row=0, column=0, columnspan=6, sticky="w", pady=(0, 8))

        ttk.Label(tafel, text="Reaction *:").grid(row=1, column=0, sticky="e", pady=3)
        self.tafel_reaction_combo = ttk.Combobox(
            tafel, textvariable=self.tafel_reaction_var, state="readonly", width=31,
            values=["Oxygen reduction (ORR)", "Oxygen evolution (OER)",
                    "Hydrogen evolution (HER)", "Hydrogen oxidation (HOR)", "User-defined"],
        )
        self.tafel_reaction_combo.grid(row=1, column=1, columnspan=2, sticky="w", padx=6)
        self.tafel_reaction_combo.bind("<<ComboboxSelected>>", lambda _e: self.update_tafel_state())

        ttk.Label(tafel, text="Reference electrode *:").grid(row=2, column=0, sticky="e", pady=3)
        self.tafel_reference_combo = ttk.Combobox(
            tafel, textvariable=self.tafel_reference_var, state="readonly", width=31,
            values=list(REFERENCE_ELECTRODES_VS_SHE_25C.keys()) + ["RHE"],
        )
        self.tafel_reference_combo.grid(row=2, column=1, columnspan=2, sticky="w", padx=6)
        self.tafel_reference_combo.bind("<<ComboboxSelected>>", lambda _e: self.update_tafel_equilibrium_display())

        ttk.Label(tafel, text="pH *:").grid(row=3, column=0, sticky="e", pady=3)
        self.tafel_ph_entry = ttk.Entry(tafel, textvariable=self.tafel_ph_var, width=12)
        self.tafel_ph_entry.grid(row=3, column=1, sticky="w", padx=6)
        self.tafel_ph_entry.bind("<FocusOut>", lambda _e: self.update_tafel_equilibrium_display())
        ttk.Label(tafel, text="Temperature:").grid(row=3, column=2, sticky="e", padx=(20, 4))
        self.tafel_temperature_entry = ttk.Entry(tafel, textvariable=self.tafel_temperature_var, width=10)
        self.tafel_temperature_entry.grid(row=3, column=3, sticky="w")
        ttk.Label(tafel, text="°C").grid(row=3, column=4, sticky="w", padx=4)

        ttk.Label(tafel, text="For user-defined: E_eq vs SHE:").grid(row=4, column=0, sticky="e", pady=3)
        self.tafel_user_eq_entry = ttk.Entry(tafel, textvariable=self.tafel_user_eq_var, width=12)
        self.tafel_user_eq_entry.grid(row=4, column=1, sticky="w", padx=6)
        ttk.Label(tafel, text="V").grid(row=4, column=2, sticky="w")

        ttk.Label(tafel, text="Calculated equilibrium potential:").grid(row=5, column=0, sticky="ne", pady=(8, 3))
        ttk.Label(tafel, textvariable=self.tafel_eq_display_var, wraplength=650).grid(
            row=5, column=1, columnspan=5, sticky="w", padx=6, pady=(8, 3))

        ttk.Label(tafel, text="Tafel potential range:").grid(row=6, column=0, sticky="e", pady=(8, 3))
        ttk.Label(tafel, text="from").grid(row=6, column=1, sticky="e", pady=(8, 3))
        self.tafel_from_entry = ttk.Entry(tafel, textvariable=self.tafel_potential_from_var, width=10)
        self.tafel_from_entry.grid(row=6, column=2, sticky="w", pady=(8, 3))
        ttk.Label(tafel, text="V  to").grid(row=6, column=3, sticky="e", pady=(8, 3))
        self.tafel_to_entry = ttk.Entry(tafel, textvariable=self.tafel_potential_to_var, width=10)
        self.tafel_to_entry.grid(row=6, column=4, sticky="w", pady=(8, 3))
        ttk.Label(tafel, text="V").grid(row=6, column=5, sticky="w", pady=(8, 3))
        self.tafel_plot_range_button = ttk.Button(
            tafel, text="Use current plot range …", command=self.choose_tafel_plot_range)
        self.tafel_plot_range_button.grid(row=7, column=1, columnspan=3, sticky="w", padx=6, pady=4)

        ttk.Label(tafel, text="Current representation:").grid(row=8, column=0, sticky="e", pady=3)
        self.tafel_density_radio = ttk.Radiobutton(
            tafel, text="Current density", variable=self.tafel_current_mode_var,
            value="density", command=self.update_tafel_state)
        self.tafel_density_radio.grid(row=8, column=1, sticky="w", padx=6)
        self.tafel_current_radio = ttk.Radiobutton(
            tafel, text="Current", variable=self.tafel_current_mode_var,
            value="current", command=self.update_tafel_state)
        self.tafel_current_radio.grid(row=8, column=2, sticky="w", padx=6)
        ttk.Label(tafel, text="Electrode area:").grid(row=8, column=3, sticky="e", padx=(20, 4))
        self.tafel_area_entry = ttk.Entry(tafel, textvariable=self.tafel_area_var, width=10)
        self.tafel_area_entry.grid(row=8, column=4, sticky="w")
        ttk.Label(tafel, text="cm²").grid(row=8, column=5, sticky="w", padx=4)

        self.tafel_kl_check = ttk.Checkbutton(
            tafel, text="Use kinetic current (mass-transport corrected)",
            variable=self.tafel_kl_corrected_var, command=self.update_tafel_state)
        self.tafel_kl_check.grid(row=9, column=0, columnspan=3, sticky="w", pady=(6, 3))
        ttk.Label(
            tafel,
            text="Potential range of the diffusion-limited plateau used to determine Iₗ",
        ).grid(row=10, column=0, columnspan=8, sticky="w", pady=(5, 1))
        self.tafel_limiting_from_entry = ttk.Entry(
            tafel, textvariable=self.tafel_limiting_from_var, width=9)
        ttk.Label(tafel, text="From").grid(row=11, column=0, sticky="w")
        self.tafel_limiting_from_entry.grid(row=11, column=1, sticky="w", padx=(4, 0))
        ttk.Label(tafel, text="V").grid(row=11, column=2, sticky="w", padx=(2, 14))
        ttk.Label(tafel, text="To").grid(row=11, column=3, sticky="e", padx=(0, 4))
        self.tafel_limiting_to_entry = ttk.Entry(
            tafel, textvariable=self.tafel_limiting_to_var, width=9)
        self.tafel_limiting_to_entry.grid(row=11, column=4, sticky="w")
        ttk.Label(tafel, text="V").grid(row=11, column=5, sticky="w", padx=4)

        self.tafel_k0_check = ttk.Checkbutton(
            tafel, text="Calculate standard rate constant k⁰ (defined simple redox couple only)",
            variable=self.tafel_k0_enabled_var, command=self.update_tafel_state)
        self.tafel_k0_check.grid(row=12, column=0, columnspan=4, sticky="w", pady=(7, 3))
        ttk.Label(tafel, text="n:").grid(row=13, column=0, sticky="e")
        self.tafel_n_entry = ttk.Entry(tafel, textvariable=self.tafel_n_var, width=8)
        self.tafel_n_entry.grid(row=13, column=1, sticky="w")
        ttk.Label(tafel, text="alpha:").grid(row=13, column=2, sticky="e")
        self.tafel_alpha_entry = ttk.Entry(tafel, textvariable=self.tafel_alpha_var, width=8)
        self.tafel_alpha_entry.grid(row=13, column=3, sticky="w")
        ttk.Label(tafel, text="c_O / mol L⁻¹:").grid(row=13, column=4, sticky="e")
        self.tafel_co_entry = ttk.Entry(tafel, textvariable=self.tafel_co_var, width=10)
        self.tafel_co_entry.grid(row=13, column=5, sticky="w")
        ttk.Label(tafel, text="c_R / mol L⁻¹:").grid(row=13, column=6, sticky="e")
        self.tafel_cr_entry = ttk.Entry(tafel, textvariable=self.tafel_cr_var, width=10)
        self.tafel_cr_entry.grid(row=13, column=7, sticky="w")

        ttk.Label(
            tafel,
            text=("* Required. The program calculates E_eq from reaction, reference electrode and pH. "
                  "The regression uses eta versus log10 of the absolute disk current or current density. "
                  "Select only the linear kinetic Tafel region. Optionally, the program calculates the kinetic "
                  "current from 1/I = 1/I_k + 1/I_L. I_L is estimated as the median current in the selected "
                  "diffusion-limited plateau range. The report warns if this range is not sufficiently flat or "
                  "if the Tafel data lie too close to I_L. From the fitted intercept, j₀ is obtained at eta = 0. "
                  "For a user-defined simple redox couple, k⁰ can be calculated from j₀ = n·F·k⁰·cO^(1−α)·cR^α. "
                  "For ORR, OER, HER and HOR, k⁰ is not calculated because the concentrations are not uniquely defined."),
            wraplength=850,
        ).grid(row=14, column=0, columnspan=8, sticky="w", pady=(8, 0))

        lkl = ttk.LabelFrame(
            main,
            text="Levich / Koutecký–Levich (hydrodynamische Analysis)",
            padding=10,
        )
        lkl.pack(fill="x", pady=4)
        for col in (1, 4):
            lkl.columnconfigure(col, weight=1)

        ttk.Checkbutton(
            lkl,
            text="Levich/Koutecký–Levich zusätzlich berechnen",
            variable=self.lkl_enabled_var,
            command=self.update_lkl_state,
        ).grid(row=0, column=0, columnspan=6, sticky="w", pady=(0, 8))

        ttk.Label(lkl, text="Analysissmodus:").grid(row=1, column=0, sticky="w")
        self.lkl_mode_d = ttk.Radiobutton(
            lkl, text="Diffusionskoeffizient D bestimmen",
            variable=self.lkl_mode_var, value="D", command=self.update_lkl_state,
        )
        self.lkl_mode_d.grid(row=1, column=1, columnspan=2, sticky="w", padx=6)
        self.lkl_mode_n = ttk.Radiobutton(
            lkl, text="Elektronenzahl n bestimmen",
            variable=self.lkl_mode_var, value="n", command=self.update_lkl_state,
        )
        self.lkl_mode_n.grid(row=1, column=3, columnspan=2, sticky="w", padx=6)

        fields = [
            ("Levich-Potential E_L:", self.lkl_levich_potential_var, "V"),
            ("Koutecký–Levich-Potential E_KL:", self.lkl_kl_potential_var, "V"),
            ("Elektrodenoberfläche A:", self.lkl_area_var, "cm²"),
            ("Konzentration c:", self.lkl_concentration_var, "mol/L"),
            ("Kinematische Viskosität ν:", self.lkl_viscosity_var, "cm²/s"),
        ]
        self.lkl_common_entries = []
        for i, (label, var, unit) in enumerate(fields, start=2):
            ttk.Label(lkl, text=label).grid(row=i, column=0, sticky="e", pady=3)
            entry = ttk.Entry(lkl, textvariable=var, width=14)
            entry.grid(row=i, column=1, sticky="w", padx=6)
            ttk.Label(lkl, text=unit).grid(row=i, column=2, sticky="w")
            self.lkl_common_entries.append(entry)

        ttk.Label(lkl, text="Bei D-Bestimmung: n =").grid(row=2, column=3, sticky="e", padx=(25, 4))
        self.lkl_n_entry = ttk.Entry(lkl, textvariable=self.lkl_n_var, width=14)
        self.lkl_n_entry.grid(row=2, column=4, sticky="w")
        ttk.Label(lkl, text="dimensionslos").grid(row=2, column=5, sticky="w", padx=5)

        ttk.Label(lkl, text="Bei n-Bestimmung: D =").grid(row=3, column=3, sticky="e", padx=(25, 4))
        self.lkl_d_entry = ttk.Entry(lkl, textvariable=self.lkl_d_var, width=14)
        self.lkl_d_entry.grid(row=3, column=4, sticky="w")
        ttk.Label(lkl, text="cm²/s").grid(row=3, column=5, sticky="w", padx=5)

        self.lkl_abs_check = ttk.Checkbutton(
            lkl,
            text="Use absolute values of cathodic disk currents for Levich/Koutecký–Levich calculations",
            variable=self.lkl_abs_var,
        )
        self.lkl_abs_check.grid(row=7, column=0, columnspan=4, sticky="w", pady=(8, 0))
        ttk.Label(
            lkl,
            text=("Disk currents are linearly interpolated at the selected Levich and "
                  "Koutecký–Levich potentials before the regressions are calculated."),
        ).grid(row=8, column=0, columnspan=6, sticky="w", pady=(4, 0))

        export = ttk.LabelFrame(main, text="Ergebnisordner", padding=10)
        export.pack(fill="x", pady=4)

        ttk.Label(
            export,
            text=(
                "Excel-Report · Raw data-CSV · Disk/Ring 2D als PNG, PDF und HTML · "
                "gemeinsame 2D-Darstellung · getrennte 3D-HTML · gemeinsame 3D-HTML · optional H₂O₂/n(E), Tafel und Levich/KL"
            ),
            wraplength=900,
        ).pack(anchor="w")

        status = ttk.LabelFrame(main, text="Status", padding=8)
        status.pack(fill="x", pady=(4, 12))

        ttk.Label(
            status,
            textvariable=self.status_var,
            wraplength=920,
            justify="left",
        ).pack(anchor="nw", fill="x")

        self.update_factor_state()
        self.update_background_state()
        self.update_h2o2_state()
        self.update_tafel_state()
        self.update_lkl_state()

        # F5 als zusätzliche Startmöglichkeit.
        self.bind("<F5>", lambda _event: self.run_analysis())

    def update_h2o2_state(self):
        enabled = self.h2o2_enabled_var.get()
        radio_state = "normal" if enabled else "disabled"
        self.collection_manufacturer_radio.configure(state=radio_state)
        self.collection_manual_radio.configure(state=radio_state)
        manufacturer_state = "normal" if enabled and self.collection_mode_var.get() == "manufacturer" else "disabled"
        manual_state = "normal" if enabled and self.collection_mode_var.get() == "manual" else "disabled"
        self.collection_manufacturer_entry.configure(state=manufacturer_state)
        self.collection_manual_entry.configure(state=manual_state)
        range_state = "normal" if enabled else "disabled"
        self.h2o2_potential_from_entry.configure(state=range_state)
        self.h2o2_potential_to_entry.configure(state=range_state)
        self.h2o2_use_plot_range_button.configure(state=range_state)

    def _load_selected_rrde_data(self) -> RRDEData:
        source = Path(self.csv_var.get().strip())
        if not source.is_file():
            raise ValueError("Bitte zuerst eine vorhandene CSV-Messdatei auswählen.")
        return parse_rrde_csv(source)

    def _load_background_rrde_data(self, source: Path) -> RRDEData:
        """Load an N₂/background CSV using the same standard RRDE structure."""
        return parse_rrde_csv(source)

    def choose_h2o2_plot_range(self):
        try:
            data = self._load_selected_rrde_data()
        except Exception as exc:
            messagebox.showerror("Plotbereich", str(exc), parent=self)
            return

        dialog = tk.Toplevel(self)
        dialog.title("Potentialbereich aus aktuellem Plot übernehmen")
        dialog.geometry("940x700")
        dialog.transient(self)

        ttk.Label(
            dialog,
            text=("Zoomen oder verschieben Sie den Plot mit der Werkzeugleiste. "
                  "Klicken Sie danach auf „Aktuellen x-Achsenbereich übernehmen“."),
            padding=(10, 8), wraplength=900,
        ).pack(fill="x")

        fig = Figure(figsize=(8.5, 5.5), dpi=100)
        ax = fig.add_subplot(111)
        dscale = {"A": 1.0, "mA": 1e3, "µA": 1e6, "nA": 1e9}[self.disk_unit_var.get()]
        for rpm, curve in zip(data.rotations, data.disk):
            ax.plot(data.potential, np.asarray(curve, dtype=float) * dscale,
                    label=f"{format_rpm(rpm)} rpm")
        ax.set_xlabel("Potential / V", fontweight="bold")
        ax.set_ylabel(f"Disk current / {self.disk_unit_var.get()}", fontweight="bold")
        ax.set_title("Select evaluation potential range")
        ax.grid(True, alpha=0.3)
        ax.legend(title="Rotation rate", fontsize=8, ncol=2)
        if self.reverse_x_var.get():
            ax.invert_xaxis()
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=dialog)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=8)
        toolbar = NavigationToolbar2Tk(canvas, dialog, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(fill="x", padx=8)

        buttons = ttk.Frame(dialog, padding=10)
        buttons.pack(fill="x")

        def apply_range():
            left, right = ax.get_xlim()
            low, high = sorted((float(left), float(right)))
            self.h2o2_potential_from_var.set(f"{low:.6g}")
            self.h2o2_potential_to_var.set(f"{high:.6g}")
            self.status_var.set(
                f"Potentialbereich für H₂O₂/n(E) übernommen: {low:.6g} bis {high:.6g} V."
            )
            dialog.destroy()

        ttk.Button(
            buttons, text="Aktuellen x-Achsenbereich übernehmen", command=apply_range
        ).pack(side="left")
        ttk.Button(buttons, text="Abbrechen", command=dialog.destroy).pack(side="right")

    def choose_tafel_plot_range(self):
        try:
            data = self._load_selected_rrde_data()
        except Exception as exc:
            messagebox.showerror("Tafel plot range", str(exc), parent=self)
            return
        dialog = tk.Toplevel(self)
        dialog.title("Select Tafel potential range")
        dialog.geometry("940x700")
        dialog.transient(self)
        ttk.Label(dialog, text=("Zoom to the linear kinetic Tafel region and then adopt the visible x-axis range."),
                  padding=(10, 8), wraplength=900).pack(fill="x")
        fig = Figure(figsize=(8.5, 5.5), dpi=100)
        ax = fig.add_subplot(111)
        dscale = {"A": 1.0, "mA": 1e3, "µA": 1e6, "nA": 1e9}[self.disk_unit_var.get()]
        for rpm, curve in zip(data.rotations, data.disk):
            ax.plot(data.potential, np.asarray(curve, dtype=float) * dscale,
                    label=f"{format_rpm(rpm)} rpm")
        ax.set_xlabel("Potential / V", fontweight="bold")
        ax.set_ylabel(f"Disk current / {self.disk_unit_var.get()}", fontweight="bold")
        ax.set_title("Select linear Tafel region")
        ax.grid(True, alpha=0.3); ax.legend(title="Rotation rate", fontsize=8, ncol=2)
        if self.reverse_x_var.get(): ax.invert_xaxis()
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=dialog); canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=8)
        toolbar = NavigationToolbar2Tk(canvas, dialog, pack_toolbar=False); toolbar.update(); toolbar.pack(fill="x", padx=8)
        buttons = ttk.Frame(dialog, padding=10); buttons.pack(fill="x")
        def apply_range():
            low, high = sorted(map(float, ax.get_xlim()))
            self.tafel_potential_from_var.set(f"{low:.6g}")
            self.tafel_potential_to_var.set(f"{high:.6g}")
            self.status_var.set(f"Tafel potential range adopted: {low:.6g} to {high:.6g} V.")
            dialog.destroy()
        ttk.Button(buttons, text="Adopt current x-axis range", command=apply_range).pack(side="left")
        ttk.Button(buttons, text="Cancel", command=dialog.destroy).pack(side="right")

    def update_tafel_equilibrium_display(self):
        if not self.tafel_enabled_var.get():
            return
        try:
            reaction = self.tafel_reaction_var.get().strip()
            reference = self.tafel_reference_var.get().strip()
            if not reaction or not reference or not self.tafel_ph_var.get().strip():
                self.tafel_eq_display_var.set("Select reaction, reference electrode and enter pH.")
                return
            ph = float(self.tafel_ph_var.get().replace(",", "."))
            temperature = float(self.tafel_temperature_var.get().replace(",", "."))
            user_eq = None
            if reaction == "User-defined":
                user_eq = float(self.tafel_user_eq_var.get().replace(",", "."))
            result = calculate_equilibrium_potential(reaction, reference, ph, temperature, user_eq)
            self.tafel_eq_display_var.set(
                f"{result['equilibrium_vs_reference_v']:.4f} V vs {reference}; "
                f"{result['equilibrium_vs_she_v']:.4f} V vs SHE; "
                f"{result['equilibrium_vs_rhe_v']:.4f} V vs RHE"
            )
        except Exception as exc:
            self.tafel_eq_display_var.set(str(exc))

    def update_tafel_state(self):
        enabled = self.tafel_enabled_var.get()
        combo_state = "readonly" if enabled else "disabled"
        for widget in getattr(self, "tafel_reaction_combo", None), getattr(self, "tafel_reference_combo", None):
            if widget is not None: widget.configure(state=combo_state)
        common_state = "normal" if enabled else "disabled"
        for widget in [getattr(self, "tafel_ph_entry", None), getattr(self, "tafel_temperature_entry", None),
                       getattr(self, "tafel_from_entry", None), getattr(self, "tafel_to_entry", None),
                       getattr(self, "tafel_plot_range_button", None), getattr(self, "tafel_density_radio", None),
                       getattr(self, "tafel_current_radio", None), getattr(self, "tafel_kl_check", None)]:
            if widget is not None: widget.configure(state=common_state)
        simple_redox = enabled and self.tafel_reaction_var.get() == "User-defined"
        if not simple_redox and self.tafel_k0_enabled_var.get():
            self.tafel_k0_enabled_var.set(False)
        if hasattr(self, "tafel_k0_check"):
            self.tafel_k0_check.configure(state="normal" if simple_redox else "disabled")
        user_state = "normal" if enabled and self.tafel_reaction_var.get() == "User-defined" else "disabled"
        if hasattr(self, "tafel_user_eq_entry"): self.tafel_user_eq_entry.configure(state=user_state)
        area_state = "normal" if enabled and self.tafel_current_mode_var.get() == "density" else "disabled"
        if hasattr(self, "tafel_area_entry"): self.tafel_area_entry.configure(state=area_state)
        kl_state = "normal" if enabled and self.tafel_kl_corrected_var.get() else "disabled"
        for widget in [getattr(self, "tafel_limiting_from_entry", None),
                       getattr(self, "tafel_limiting_to_entry", None)]:
            if widget is not None:
                widget.configure(state=kl_state)
        k0_state = ("normal" if enabled and self.tafel_k0_enabled_var.get()
                    and self.tafel_current_mode_var.get() == "density" else "disabled")
        for widget in [getattr(self, "tafel_n_entry", None), getattr(self, "tafel_alpha_entry", None),
                       getattr(self, "tafel_co_entry", None), getattr(self, "tafel_cr_entry", None)]:
            if widget is not None:
                widget.configure(state=k0_state)
        self.update_tafel_equilibrium_display()

    def update_factor_state(self):
        state = "normal" if self.ring_scale_mode_var.get() == "manual" else "disabled"
        self.ring_factor_entry.configure(state=state)

    def update_background_state(self):
        enabled = self.ring_bg_enabled_var.get()
        method = self.ring_bg_method_var.get()
        range_state = "normal" if enabled and method == "range_mean" else "disabled"
        manual_state = "normal" if enabled and method == "manual" else "disabled"
        file_state = "normal" if enabled and method == "measurement_csv" else "disabled"
        self.ring_bg_from_entry.configure(state=range_state)
        self.ring_bg_to_entry.configure(state=range_state)
        self.ring_bg_offset_entry.configure(state=manual_state)
        self.background_csv_entry.configure(state=file_state)
        self.background_csv_button.configure(state=file_state)

    def browse_background_csv(self):
        filename = filedialog.askopenfilename(
            title="N₂-/Hintergrundmessung auswählen",
            filetypes=[("CSV-Dateien", "*.csv")],
        )
        if filename:
            self.background_csv_var.set(filename)

    def update_lkl_state(self):
        enabled = self.lkl_enabled_var.get()
        common_state = "normal" if enabled else "disabled"
        for entry in getattr(self, "lkl_common_entries", []):
            entry.configure(state=common_state)
        self.lkl_mode_d.configure(state=common_state)
        self.lkl_mode_n.configure(state=common_state)
        self.lkl_abs_check.configure(state=common_state)
        if not enabled:
            self.lkl_n_entry.configure(state="disabled")
            self.lkl_d_entry.configure(state="disabled")
        elif self.lkl_mode_var.get() == "D":
            self.lkl_n_entry.configure(state="normal")
            self.lkl_d_entry.configure(state="disabled")
        else:
            self.lkl_n_entry.configure(state="disabled")
            self.lkl_d_entry.configure(state="normal")

    def browse_csv(self):
        filename = filedialog.askopenfilename(
            title="RRDE-CSV-Messdatei auswählen",
            filetypes=[("CSV-Dateien", "*.csv")],
        )
        if filename:
            p = Path(filename)
            self.csv_var.set(str(p))
            self.output_var.set(str(p.parent))
            self.name_var.set(p.stem)
            try:
                data = parse_rrde_csv(p)
                rpms = ", ".join(format_rpm(r) for r in data.rotations)
                self.file_status_var.set(
                    f"CSV erkannt: {len(data.rotations)} Rotationen ({rpms} U/min); "
                    f"Potential-, Disk- und Ringspalten wurden eingelesen."
                )
            except Exception as exc:
                self.file_status_var.set(f"CSV ausgewählt, Strukturprüfung fehlgeschlagen: {exc}")

    def show_csv_structure(self):
        messagebox.showinfo(
            "Erforderliche RRDE-CSV-Struktur",
            "Erwartete Spaltenfolge:\n\n"
            "1. Spalte: Potential in V\n"
            "danach für jede Rotationsgeschwindigkeit genau zwei Spalten:\n"
            "  • Diskstrom\n"
            "  • Ringstrom\n\n"
            "Beispiel:\n"
            "Potential / V | 500 rpm Disk | 500 rpm Ring | 1000 rpm Disk | 1000 rpm Ring\n\n"
            "Die Rotationsgeschwindigkeit muss im Namen der jeweiligen Diskspalte stehen. "
            "Die N₂-Hintergrunddatei muss dieselbe Spaltenstruktur und dieselben Rotationen besitzen.",
            parent=self,
        )

    def browse_output(self):
        folder = filedialog.askdirectory(title="Ausgabeordner auswählen")
        if folder:
            self.output_var.set(folder)

    def open_last(self):
        if self.last_folder and self.last_folder.exists():
            open_file(self.last_folder)
        else:
            messagebox.showinfo("RRDE-Analysis", "Noch kein Ergebnisordner vorhanden.")

    def run_analysis(self):
        try:
            source = Path(self.csv_var.get().strip())
            if not source.is_file():
                raise ValueError("Bitte eine vorhandene CSV-Messdatei auswählen.")

            base_output = Path(self.output_var.get().strip() or source.parent)
            window = int(self.window_var.get())
            poly = int(self.poly_var.get())

            result_name = safe_filename(self.name_var.get())
            result_folder = base_output / f"{result_name}_Ergebnisse"
            result_folder.mkdir(parents=True, exist_ok=True)

            self.start_btn.config(state="disabled")
            self.quick_start_button.config(state="disabled")
            self.progress.start(12)
            self.status_var.set("CSV-Messdatei wird gelesen …")
            self.update_idletasks()

            data = self._load_selected_rrde_data()
            disk_raw_original_a = [interpolate_nans(v) for v in data.disk]
            ring_raw_original_a = [interpolate_nans(v) for v in data.ring]

            bg_enabled = self.ring_bg_enabled_var.get()
            bg_method = self.ring_bg_method_var.get()
            background_source = None
            if bg_enabled and bg_method == "measurement_csv":
                bg_path = Path(self.background_csv_var.get().strip())
                if not bg_path.is_file():
                    raise ValueError("Bitte eine vorhandene CSV-Datei mit der N₂-/Hintergrundmessung auswählen.")
                background_data = self._load_background_rrde_data(bg_path)
                disk_raw_a, ring_raw_a = subtract_rrde_background_measurement(data, background_data)
                background_source = bg_path
            else:
                disk_raw_a = [v.copy() for v in disk_raw_original_a]
                ring_raw_a = [v.copy() for v in ring_raw_original_a]

            disk_sm_a = [smooth(v, self.smooth_var.get(), window, poly) for v in disk_raw_a]
            ring_sm_a = [smooth(v, self.smooth_var.get(), window, poly) for v in ring_raw_a]

            scale_map = {"A": 1.0, "mA": 1e3, "µA": 1e6, "nA": 1e9}
            dscale = scale_map[self.disk_unit_var.get()]
            rscale = scale_map[self.ring_unit_var.get()]
            disk_raw = [v * dscale for v in disk_raw_a]
            disk_sm = [v * dscale for v in disk_sm_a]
            ring_raw_uncomp = [v * rscale for v in ring_raw_a]
            ring_sm_uncomp = [v * rscale for v in ring_sm_a]

            bg_from = float(self.ring_bg_from_var.get().replace(",", "."))
            bg_to = float(self.ring_bg_to_var.get().replace(",", "."))
            bg_manual = float(self.ring_bg_offset_var.get().replace(",", "."))

            constant_bg_enabled = bg_enabled and bg_method in {"range_mean", "manual"}
            ring_raw, ring_offsets_raw = compensate_ring_background(
                data.potential, ring_raw_uncomp, constant_bg_enabled, bg_method,
                bg_from, bg_to, bg_manual
            )
            ring_sm, ring_offsets = compensate_ring_background(
                data.potential, ring_sm_uncomp, constant_bg_enabled, bg_method,
                bg_from, bg_to, bg_manual
            )

            # Für CSV und Excel wieder in Ampere zurückrechnen.
            ring_sm_comp_a = [v / rscale for v in ring_sm]

            if not bg_enabled:
                bg_description = "Keine"
            elif bg_method == "measurement_csv":
                bg_description = f"Separate Messung (Disk + Ring): {background_source.name}"
            elif bg_method == "manual":
                bg_description = (
                    f"Manueller Offset: {bg_manual:g} {self.ring_unit_var.get()}"
                )
            else:
                offsets_text = ", ".join(f"{v:.4g}" for v in ring_offsets)
                bg_description = (
                    f"Kurvenindividueller Mittelwert von {bg_from:g} bis {bg_to:g} V; "
                    f"Offsets in {self.ring_unit_var.get()}: {offsets_text}"
                )

            prefix = result_folder / result_name

            h2o2_result = None
            if self.h2o2_enabled_var.get():
                self.status_var.set("H₂O₂-Ausbeute und n(E) werden berechnet …")
                self.update_idletasks()
                if self.collection_mode_var.get() == "manufacturer":
                    collection_efficiency = float(self.collection_manufacturer_var.get().replace(",", "."))
                    collection_source = "Manufacturer specification"
                else:
                    collection_efficiency = float(self.collection_manual_var.get().replace(",", "."))
                    collection_source = "Experimental/manual value"
                h2o2_potential_from = float(self.h2o2_potential_from_var.get().replace(",", "."))
                h2o2_potential_to = float(self.h2o2_potential_to_var.get().replace(",", "."))
                h2o2_result = calculate_h2o2_analysis(
                    data.potential, data.rotations, disk_sm_a, ring_sm_comp_a,
                    collection_efficiency, h2o2_potential_from, h2o2_potential_to,
                    self.h2o2_abs_disk_var.get()
                )
                if not bg_enabled:
                    h2o2_background_description = "None"
                elif bg_method == "measurement_csv":
                    h2o2_background_description = f"Separate measurement, disk + ring: {background_source.name}"
                elif bg_method == "manual":
                    h2o2_background_description = (
                        f"Constant offset: {bg_manual:g} {self.ring_unit_var.get()}"
                    )
                else:
                    offsets_text_en = ", ".join(f"{v:.4g}" for v in ring_offsets)
                    h2o2_background_description = (
                        f"Curve-specific mean from {bg_from:g} to {bg_to:g} V; "
                        f"offsets in {self.ring_unit_var.get()}: {offsets_text_en}"
                    )
                save_h2o2_outputs(
                    prefix, h2o2_result, collection_source, h2o2_background_description
                )

            tafel_result = None
            tafel_electrochem = None
            if self.tafel_enabled_var.get():
                self.status_var.set("Tafel analysis is being calculated …")
                self.update_idletasks()
                reaction = self.tafel_reaction_var.get().strip()
                reference = self.tafel_reference_var.get().strip()
                if not reaction:
                    raise ValueError("Tafel analysis: reaction is required.")
                if not reference:
                    raise ValueError("Tafel analysis: reference electrode is required.")
                if not self.tafel_ph_var.get().strip():
                    raise ValueError("Tafel analysis: pH is required.")
                ph = float(self.tafel_ph_var.get().replace(",", "."))
                temperature = float(self.tafel_temperature_var.get().replace(",", "."))
                user_eq = None
                if reaction == "User-defined":
                    if not self.tafel_user_eq_var.get().strip():
                        raise ValueError("Tafel analysis: E_eq vs SHE is required for a user-defined reaction.")
                    user_eq = float(self.tafel_user_eq_var.get().replace(",", "."))
                tafel_electrochem = calculate_equilibrium_potential(
                    reaction, reference, ph, temperature, user_eq)
                tafel_from = float(self.tafel_potential_from_var.get().replace(",", "."))
                tafel_to = float(self.tafel_potential_to_var.get().replace(",", "."))
                use_density = self.tafel_current_mode_var.get() == "density"
                area = float(self.tafel_area_var.get().replace(",", ".")) if use_density else None
                use_kl_corrected = self.tafel_kl_corrected_var.get()
                limiting_from = (
                    float(self.tafel_limiting_from_var.get().replace(",", "."))
                    if use_kl_corrected else None
                )
                limiting_to = (
                    float(self.tafel_limiting_to_var.get().replace(",", "."))
                    if use_kl_corrected else None
                )
                calculate_k0 = self.tafel_k0_enabled_var.get()
                if calculate_k0 and reaction != "User-defined":
                    raise ValueError(
                        "Tafel analysis: k⁰ is only available for a user-defined simple redox couple. "
                        "For ORR, OER, HER and HOR, the Butler–Volmer concentrations are not uniquely defined."
                    )
                if calculate_k0 and not use_density:
                    raise ValueError("Tafel analysis: k⁰ requires current density and a valid electrode area.")
                n_e = float(self.tafel_n_var.get().replace(",", ".")) if calculate_k0 else None
                alpha = float(self.tafel_alpha_var.get().replace(",", ".")) if calculate_k0 else None
                c_o = float(self.tafel_co_var.get().replace(",", ".")) if calculate_k0 else None
                c_r = float(self.tafel_cr_var.get().replace(",", ".")) if calculate_k0 else None
                tafel_result = calculate_tafel_analysis(
                    data.potential, data.rotations, disk_sm_a, tafel_from, tafel_to,
                    tafel_electrochem["equilibrium_vs_reference_v"], use_density, area,
                    use_kl_corrected, None, limiting_from, limiting_to,
                    calculate_k0, n_e, alpha, c_o, c_r)
                save_tafel_outputs(prefix, tafel_result, tafel_electrochem)

            lkl_result = None
            if self.lkl_enabled_var.get():
                self.status_var.set("Levich/Koutecký–Levich wird berechnet …")
                self.update_idletasks()
                levich_e = float(self.lkl_levich_potential_var.get().replace(",", "."))
                kl_e = float(self.lkl_kl_potential_var.get().replace(",", "."))
                area = float(self.lkl_area_var.get().replace(",", "."))
                concentration = float(self.lkl_concentration_var.get().replace(",", "."))
                viscosity = float(self.lkl_viscosity_var.get().replace(",", "."))
                mode = self.lkl_mode_var.get()
                n_value = float(self.lkl_n_var.get().replace(",", ".")) if mode == "D" else None
                d_value = float(self.lkl_d_var.get().replace(",", ".")) if mode == "n" else None
                lkl_result = calculate_levich_kl(
                    data.potential,
                    data.rotations,
                    disk_sm_a,
                    levich_e,
                    kl_e,
                    area,
                    concentration,
                    viscosity,
                    mode,
                    n_value=n_value,
                    diffusion_cm2_s=d_value,
                    use_absolute_current=self.lkl_abs_var.get(),
                )
                save_levich_kl_outputs(prefix, lkl_result)

            self.status_var.set("2D- und HTML-Diagramme werden erzeugt …")
            self.update_idletasks()

            make_2d_plot(
                data.potential, data.rotations, disk_raw, disk_sm,
                f"{result_name} – Diskströme",
                f"Disk current / {self.disk_unit_var.get()}",
                prefix.with_name(prefix.name + "_Disk_2D.png"),
                prefix.with_name(prefix.name + "_Disk_2D.pdf"),
                self.show_raw_var.get(), self.reverse_x_var.get(),
            )
            make_2d_plot(
                data.potential, data.rotations, ring_raw, ring_sm,
                f"{result_name} – Ringströme" + (" (untergrundkompensiert)" if bg_enabled else ""),
                f"Ring current / {self.ring_unit_var.get()}",
                prefix.with_name(prefix.name + "_Ring_2D.png"),
                prefix.with_name(prefix.name + "_Ring_2D.pdf"),
                self.show_raw_var.get(), self.reverse_x_var.get(),
            )
            make_2d_html(
                data.potential, data.rotations, disk_raw, disk_sm,
                f"{result_name} – Diskströme",
                f"Disk current / {self.disk_unit_var.get()}",
                prefix.with_name(prefix.name + "_Disk_2D_interaktiv.html"),
                self.show_raw_var.get(), self.reverse_x_var.get(),
            )
            make_2d_html(
                data.potential, data.rotations, ring_raw, ring_sm,
                f"{result_name} – Ringströme" + (" (untergrundkompensiert)" if bg_enabled else ""),
                f"Ring current / {self.ring_unit_var.get()}",
                prefix.with_name(prefix.name + "_Ring_2D_interaktiv.html"),
                self.show_raw_var.get(), self.reverse_x_var.get(),
            )

            make_combined_2d_plot(
                data.potential, data.rotations,
                disk_raw, disk_sm, ring_raw, ring_sm,
                self.disk_unit_var.get(), self.ring_unit_var.get(),
                f"{result_name} – Disk- und Ringströme gemeinsam"
                + (" (Ring untergrundkompensiert)" if bg_enabled else ""),
                prefix.with_name(prefix.name + "_Disk_Ring_2D_gemeinsam.png"),
                prefix.with_name(prefix.name + "_Disk_Ring_2D_gemeinsam.pdf"),
                self.show_raw_var.get(), self.reverse_x_var.get(),
            )
            make_combined_2d_html(
                data.potential, data.rotations,
                disk_raw, disk_sm, ring_raw, ring_sm,
                self.disk_unit_var.get(), self.ring_unit_var.get(),
                f"{result_name} – Disk- und Ringströme gemeinsam"
                + (" (Ring untergrundkompensiert)" if bg_enabled else ""),
                prefix.with_name(prefix.name + "_Disk_Ring_2D_gemeinsam_interaktiv.html"),
                self.show_raw_var.get(), self.reverse_x_var.get(),
            )

            make_separate_3d(
                data.potential, data.rotations,
                disk_raw, disk_sm, ring_raw, ring_sm,
                f"{self.disk_unit_var.get()} bzw. {self.ring_unit_var.get()}",
                prefix.with_name(prefix.name + "_RRDE_3D_getrennt.html"),
                self.show_raw_var.get(), self.reverse_x_var.get(),
            )

            manual_factor = None
            if self.ring_scale_mode_var.get() == "manual":
                manual_factor = float(self.ring_factor_var.get().replace(",", "."))
                if manual_factor <= 0:
                    raise ValueError("Der manuelle Ring current-Skalierungsfaktor muss größer als 0 sein.")

            ring_factor, automatic_factor = make_combined_3d(
                data.potential, data.rotations,
                disk_raw, disk_sm, ring_raw, ring_sm,
                self.disk_unit_var.get(), self.ring_unit_var.get(),
                prefix.with_name(prefix.name + "_RRDE_3D_gemeinsam.html"),
                self.show_raw_var.get(), self.reverse_x_var.get(),
                manual_factor=manual_factor,
            )

            self.status_var.set("Excel-Report wird geschrieben …")
            self.update_idletasks()

            save_processed_csv(
                prefix.with_name(prefix.name + "_RRDE_ausgewertet.csv"),
                data.potential, data.rotations,
                disk_raw_a, disk_sm_a, ring_raw_a, ring_sm_a, ring_sm_comp_a,
                disk_original=disk_raw_original_a, ring_original=ring_raw_original_a,
            )

            save_excel_report(
                prefix.with_name(prefix.name + "_RRDE_Report.xlsx"),
                source, data.potential, data.rotations,
                disk_raw_original_a, disk_sm_a, ring_raw_original_a, ring_sm_comp_a,
                window, poly, ring_factor,
                background_enabled=bg_enabled,
                background_method=bg_method,
                background_description=bg_description,
            )

            self.last_folder = result_folder
            rotations = ", ".join(format_rpm(r) for r in data.rotations)
            self.status_var.set(
                "Analysis erfolgreich abgeschlossen.\n"
                f"Rotationen: {rotations} U/min\n"
                f"Ergebnisordner: {result_folder}\n"
                f"Ring-Skalierungsfaktor im gemeinsamen 3D-HTML: {ring_factor:g} "
                f"({'manuell' if manual_factor is not None else 'automatisch'})\n"
                f"Ring-Untergrund: {bg_description}"
                + (
                    "\nH₂O₂/n(E): N = " + f"{h2o2_result['collection_efficiency']:.4g}; "
                    + ("keine Plausibilitätswarnungen" if not h2o2_result["warnings"] else f"{len(h2o2_result['warnings'])} Warnung(en)")
                    if h2o2_result is not None else ""
                )
                + (
                    "\nLevich/KL: " + f"E_L={lkl_result['levich_potential']:.3f} V, "
                    + f"E_KL={lkl_result['kl_potential']:.3f} V, "
                    + "D = " + f"{lkl_result['diffusion_cm2_s']:.4g} cm²/s, "
                    + f"n = {lkl_result['n_value']:.4g}, "
                    + f"R²(L) = {lkl_result['levich_r2']:.5f}, "
                    + f"R²(KL) = {lkl_result['kl_r2']:.5f}"
                    if lkl_result is not None else ""
                )
            )

            if self.open_result_var.get():
                open_file(result_folder)

            messagebox.showinfo(
                "RRDE-Analysis",
                "Die Analysis wurde erfolgreich abgeschlossen.\n\n"
                f"Ergebnisordner:\n{result_folder}",
            )

        except Exception as exc:
            details = traceback.format_exc()
            self.status_var.set(f"Fehler: {exc}")
            messagebox.showerror(
                "Fehler bei der RRDE-Analysis",
                f"{exc}\n\nTechnische Details:\n{details[-2500:]}",
            )
        finally:
            self.progress.stop()
            self.start_btn.config(state="normal")
            self.quick_start_button.config(state="normal")


def main():
    app = RRDEApp()
    if len(sys.argv) > 1:
        p = Path(sys.argv[1])
        if p.is_file():
            app.csv_var.set(str(p.resolve()))
            app.output_var.set(str(p.resolve().parent))
            app.name_var.set(p.stem)
    app.mainloop()


if __name__ == "__main__":
    main()

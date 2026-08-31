from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QFileDialog, QMessageBox, QDoubleSpinBox, QSpinBox,
    QGroupBox, QComboBox, QLineEdit, QTableWidget, QTableWidgetItem,
    QHeaderView, QDialog, QScrollArea, QCheckBox
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


def _read_numeric_csv(path):
    """Read common numeric CSV/TXT exports with decimal comma or decimal point."""
    best = None
    for sep in [None, ";", ",", "\t"]:
        try:
            df = pd.read_csv(path, sep=sep, engine="python", header=None, comment="#")
            # Preserve possible first-row labels separately, then coerce numeric body.
            numeric = df.apply(
                lambda col: pd.to_numeric(
                    col.astype(str).str.strip().str.replace(",", ".", regex=False),
                    errors="coerce"
                )
            )
            numeric = numeric.dropna(axis=0, how="all").dropna(axis=1, how="all")
            if numeric.shape[0] >= 3 and numeric.shape[1] >= 2:
                if best is None or numeric.size > best.size:
                    best = numeric.reset_index(drop=True)
        except Exception:
            pass
    if best is None:
        raise ValueError("Could not read a numeric CSV/TXT table.")
    return best


def _phase_deg(z):
    return np.degrees(np.angle(z))


def _circuit_impedance(freq, model, p):
    """Return complex impedance for the selected equivalent circuit."""
    f = np.asarray(freq, dtype=float)
    w = 2.0 * np.pi * f
    jw = 1j * w

    if model == "Rs + (Rct || Cdl)":
        Rs, Rct, Cdl = p
        Zpar = 1.0 / (1.0 / Rct + jw * Cdl)
        return Rs + Zpar

    if model == "Rs + (Rct || CPE)":
        Rs, Rct, Q, alpha = p
        Ycpe = Q * (jw ** alpha)
        Zpar = 1.0 / (1.0 / Rct + Ycpe)
        return Rs + Zpar

    if model == "Rs + (Rct || Cdl) + W":
        Rs, Rct, Cdl, sigma = p
        Zpar = 1.0 / (1.0 / Rct + jw * Cdl)
        Zw = sigma * (1.0 - 1j) / np.sqrt(w)
        return Rs + Zpar + Zw

    if model == "Rs + (Rct || CPE) + W":
        Rs, Rct, Q, alpha, sigma = p
        Ycpe = Q * (jw ** alpha)
        Zpar = 1.0 / (1.0 / Rct + Ycpe)
        Zw = sigma * (1.0 - 1j) / np.sqrt(w)
        return Rs + Zpar + Zw

    if model == "Rs + (R1 || CPE1) + (R2 || CPE2)":
        Rs, R1, Q1, alpha1, R2, Q2, alpha2 = p
        Z1 = 1.0 / (1.0 / R1 + Q1 * (jw ** alpha1))
        Z2 = 1.0 / (1.0 / R2 + Q2 * (jw ** alpha2))
        return Rs + Z1 + Z2

    if model == "Rs + (R1 || CPE1) + (R2 || CPE2) + W":
        Rs, R1, Q1, alpha1, R2, Q2, alpha2, sigma = p
        Z1 = 1.0 / (1.0 / R1 + Q1 * (jw ** alpha1))
        Z2 = 1.0 / (1.0 / R2 + Q2 * (jw ** alpha2))
        Zw = sigma * (1.0 - 1j) / np.sqrt(w)
        return Rs + Z1 + Z2 + Zw

    if model == "Rs + (Rct || CPE) + Wo":
        Rs, Rct, Q, alpha, sigma, tau = p
        Zpar = 1.0 / (1.0 / Rct + Q * (jw ** alpha))
        # finite-length Warburg, open/reflecting boundary
        gamma = np.sqrt(jw * tau)
        Zw = sigma * np.sqrt(tau) / np.sqrt(jw * tau) / np.tanh(gamma)
        return Rs + Zpar + Zw

    if model == "Rs + (Rct || CPE) + Ws":
        Rs, Rct, Q, alpha, sigma, tau = p
        Zpar = 1.0 / (1.0 / Rct + Q * (jw ** alpha))
        # finite-length Warburg, short/transmissive boundary
        gamma = np.sqrt(jw * tau)
        Zw = sigma * np.sqrt(tau) / np.sqrt(jw * tau) * np.tanh(gamma)
        return Rs + Zpar + Zw

    if model == "Rs + (Rfilm || CPEfilm) + (Rct || CPEdl)":
        Rs, Rfilm, Qfilm, afilm, Rct, Qdl, adl = p
        Zfilm = 1.0 / (1.0 / Rfilm + Qfilm * (jw ** afilm))
        Zct = 1.0 / (1.0 / Rct + Qdl * (jw ** adl))
        return Rs + Zfilm + Zct

    if model == "Rs + (Rfilm || CPEfilm) + (Rct || CPEdl) + W":
        Rs, Rfilm, Qfilm, afilm, Rct, Qdl, adl, sigma = p
        Zfilm = 1.0 / (1.0 / Rfilm + Qfilm * (jw ** afilm))
        Zct = 1.0 / (1.0 / Rct + Qdl * (jw ** adl))
        Zw = sigma * (1.0 - 1j) / np.sqrt(w)
        return Rs + Zfilm + Zct + Zw

    if model == "Rs + (R1 || C1) + (R2 || C2)":
        Rs, R1, C1, R2, C2 = p
        Z1 = 1.0 / (1.0 / R1 + jw * C1)
        Z2 = 1.0 / (1.0 / R2 + jw * C2)
        return Rs + Z1 + Z2

    if model == "Rs + (R1 || C1) + (R2 || C2) + W":
        Rs, R1, C1, R2, C2, sigma = p
        Z1 = 1.0 / (1.0 / R1 + jw * C1)
        Z2 = 1.0 / (1.0 / R2 + jw * C2)
        Zw = sigma * (1.0 - 1j) / np.sqrt(w)
        return Rs + Z1 + Z2 + Zw

    if model == "Rs + (R1 || C1) + (R2 || CPE2)":
        Rs, R1, C1, R2, Q2, alpha2 = p
        Z1 = 1.0 / (1.0 / R1 + jw * C1)
        Z2 = 1.0 / (1.0 / R2 + Q2 * (jw ** alpha2))
        return Rs + Z1 + Z2

    if model == "Rs + (R1 || CPE1) + (R2 || C2)":
        Rs, R1, Q1, alpha1, R2, C2 = p
        Z1 = 1.0 / (1.0 / R1 + Q1 * (jw ** alpha1))
        Z2 = 1.0 / (1.0 / R2 + jw * C2)
        return Rs + Z1 + Z2

    if model == "Rs + (R1 || C1) + (R2 || CPE2) + W":
        Rs, R1, C1, R2, Q2, alpha2, sigma = p
        Z1 = 1.0 / (1.0 / R1 + jw * C1)
        Z2 = 1.0 / (1.0 / R2 + Q2 * (jw ** alpha2))
        Zw = sigma * (1.0 - 1j) / np.sqrt(w)
        return Rs + Z1 + Z2 + Zw

    if model == "Rs + (R1 || CPE1) + (R2 || C2) + W":
        Rs, R1, Q1, alpha1, R2, C2, sigma = p
        Z1 = 1.0 / (1.0 / R1 + Q1 * (jw ** alpha1))
        Z2 = 1.0 / (1.0 / R2 + jw * C2)
        Zw = sigma * (1.0 - 1j) / np.sqrt(w)
        return Rs + Z1 + Z2 + Zw

    if model == "Rs + L + (Rct || Cdl)":
        Rs, L, Rct, Cdl = p
        return Rs + jw*L + 1.0/(1.0/Rct + jw*Cdl)

    if model == "Rs + L + (Rct || CPE)":
        Rs, L, Rct, Q, alpha = p
        return Rs + jw*L + 1.0/(1.0/Rct + Q*(jw**alpha))

    if model == "Rs + L + (Rct || CPE) + W":
        Rs, L, Rct, Q, alpha, sigma = p
        Zct = 1.0/(1.0/Rct + Q*(jw**alpha))
        return Rs + jw*L + Zct + sigma*(1.0-1j)/np.sqrt(w)

    if model == "Rs + L + (Rct || CPE) + Wo":
        Rs, L, Rct, Q, alpha, Rw, tauw = p
        Zct = 1.0/(1.0/Rct + Q*(jw**alpha))
        x=np.sqrt(jw*tauw)
        return Rs + jw*L + Zct + Rw/(x*np.tanh(x))

    if model == "Rs + L + (Rct || CPE) + Ws":
        Rs, L, Rct, Q, alpha, Rw, tauw = p
        Zct = 1.0/(1.0/Rct + Q*(jw**alpha))
        x=np.sqrt(jw*tauw)
        return Rs + jw*L + Zct + Rw*np.tanh(x)/x

    if model == "Rs + L + (R1 || C1) + (R2 || C2)":
        Rs, L, R1, C1, R2, C2 = p
        Z1=1.0/(1.0/R1+jw*C1); Z2=1.0/(1.0/R2+jw*C2)
        return Rs+jw*L+Z1+Z2

    if model == "Rs + L + (R1 || CPE1) + (R2 || CPE2)":
        Rs, L, R1, Q1, a1, R2, Q2, a2 = p
        Z1=1.0/(1.0/R1+Q1*(jw**a1)); Z2=1.0/(1.0/R2+Q2*(jw**a2))
        return Rs+jw*L+Z1+Z2

    if model == "Rs + L + (R1 || C1) + (R2 || CPE2)":
        Rs, L, R1, C1, R2, Q2, a2 = p
        Z1=1.0/(1.0/R1+jw*C1); Z2=1.0/(1.0/R2+Q2*(jw**a2))
        return Rs+jw*L+Z1+Z2

    if model == "Rs + L + (R1 || CPE1) + (R2 || C2)":
        Rs, L, R1, Q1, a1, R2, C2 = p
        Z1=1.0/(1.0/R1+Q1*(jw**a1)); Z2=1.0/(1.0/R2+jw*C2)
        return Rs+jw*L+Z1+Z2

    raise ValueError(f"Unknown circuit model: {model}")


def _parameter_names(model):
    return {
        "Rs + (Rct || Cdl)": ["Rs / Ω", "Rct / Ω", "Cdl / F"],
        "Rs + (Rct || CPE)": ["Rs / Ω", "Rct / Ω", "Q / S·s^α", "α"],
        "Rs + (Rct || Cdl) + W": ["Rs / Ω", "Rct / Ω", "Cdl / F", "σW / Ω·s^-1/2"],
        "Rs + (Rct || CPE) + W": ["Rs / Ω", "Rct / Ω", "Q / S·s^α", "α", "σW / Ω·s^-1/2"],
        "Rs + (R1 || CPE1) + (R2 || CPE2)": [
            "Rs / Ω", "R1 / Ω", "Q1 / S·s^α1", "α1",
            "R2 / Ω", "Q2 / S·s^α2", "α2"
        ],
        "Rs + (R1 || CPE1) + (R2 || CPE2) + W": [
            "Rs / Ω", "R1 / Ω", "Q1 / S·s^α1", "α1",
            "R2 / Ω", "Q2 / S·s^α2", "α2", "σW / Ω·s^-1/2"
        ],
        "Rs + (Rct || CPE) + Wo": [
            "Rs / Ω", "Rct / Ω", "Q / S·s^α", "α",
            "σW / Ω·s^-1/2", "τW / s"
        ],
        "Rs + (Rct || CPE) + Ws": [
            "Rs / Ω", "Rct / Ω", "Q / S·s^α", "α",
            "σW / Ω·s^-1/2", "τW / s"
        ],
        "Rs + (Rfilm || CPEfilm) + (Rct || CPEdl)": [
            "Rs / Ω", "Rfilm / Ω", "Qfilm / S·s^αfilm", "αfilm",
            "Rct / Ω", "Qdl / S·s^αdl", "αdl"
        ],
        "Rs + (Rfilm || CPEfilm) + (Rct || CPEdl) + W": [
            "Rs / Ω", "Rfilm / Ω", "Qfilm / S·s^αfilm", "αfilm",
            "Rct / Ω", "Qdl / S·s^αdl", "αdl", "σW / Ω·s^-1/2"
        ],
        "Rs + (R1 || C1) + (R2 || C2)": ["Rs / Ω","R1 / Ω","C1 / F","R2 / Ω","C2 / F"],
        "Rs + (R1 || C1) + (R2 || C2) + W": ["Rs / Ω","R1 / Ω","C1 / F","R2 / Ω","C2 / F","σW / Ω·s^-1/2"],
        "Rs + (R1 || C1) + (R2 || CPE2)": ["Rs / Ω","R1 / Ω","C1 / F","R2 / Ω","Q2 / S·s^α2","α2"],
        "Rs + (R1 || CPE1) + (R2 || C2)": ["Rs / Ω","R1 / Ω","Q1 / S·s^α1","α1","R2 / Ω","C2 / F"],
        "Rs + (R1 || C1) + (R2 || CPE2) + W": ["Rs / Ω","R1 / Ω","C1 / F","R2 / Ω","Q2 / S·s^α2","α2","σW / Ω·s^-1/2"],
        "Rs + (R1 || CPE1) + (R2 || C2) + W": ["Rs / Ω","R1 / Ω","Q1 / S·s^α1","α1","R2 / Ω","C2 / F","σW / Ω·s^-1/2"],
        "Rs + L + (Rct || Cdl)": ["Rs / Ω","L / H","Rct / Ω","Cdl / F"],
        "Rs + L + (Rct || CPE)": ["Rs / Ω","L / H","Rct / Ω","Q / S·s^α","α"],
        "Rs + L + (Rct || CPE) + W": ["Rs / Ω","L / H","Rct / Ω","Q / S·s^α","α","σW / Ω·s^-1/2"],
        "Rs + L + (Rct || CPE) + Wo": ["Rs / Ω","L / H","Rct / Ω","Q / S·s^α","α","Rw / Ω","τw / s"],
        "Rs + L + (Rct || CPE) + Ws": ["Rs / Ω","L / H","Rct / Ω","Q / S·s^α","α","Rw / Ω","τw / s"],
        "Rs + L + (R1 || C1) + (R2 || C2)": ["Rs / Ω","L / H","R1 / Ω","C1 / F","R2 / Ω","C2 / F"],
        "Rs + L + (R1 || CPE1) + (R2 || CPE2)": ["Rs / Ω","L / H","R1 / Ω","Q1 / S·s^α1","α1","R2 / Ω","Q2 / S·s^α2","α2"],
        "Rs + L + (R1 || C1) + (R2 || CPE2)": ["Rs / Ω","L / H","R1 / Ω","C1 / F","R2 / Ω","Q2 / S·s^α2","α2"],
        "Rs + L + (R1 || CPE1) + (R2 || C2)": ["Rs / Ω","L / H","R1 / Ω","Q1 / S·s^α1","α1","R2 / Ω","C2 / F"],
    }[model]



def _parameter_meaning(name):
    key = name.split(" / ")[0]
    meanings = {
        "Rs": "Solution/series resistance",
        "Rct": "Charge-transfer resistance",
        "Cdl": "Double-layer capacitance",
        "C1": "Ideal capacitance of electrochemical process 1",
        "C2": "Ideal capacitance of electrochemical process 2",
        "L": "Series inductance (instrument/cabling or genuine inductive response)",
        "Q": "CPE magnitude (equals capacitance only when α = 1)",
        "α": "CPE exponent; 1 = ideal capacitor",
        "R1": "Resistance of electrochemical process 1",
        "Q1": "CPE magnitude of process 1",
        "α1": "CPE exponent of process 1",
        "R2": "Resistance of electrochemical process 2",
        "Q2": "CPE magnitude of process 2",
        "α2": "CPE exponent of process 2",
        "Rfilm": "Film/coating resistance",
        "Qfilm": "Film/coating CPE magnitude",
        "αfilm": "Film/coating CPE exponent",
        "Qdl": "Double-layer CPE magnitude",
        "αdl": "Double-layer CPE exponent",
        "σW": "Warburg coefficient",
        "τW": "Finite-diffusion characteristic time",
    }
    return meanings.get(key, "")


def _cpe_interpretations(names, values):
    out = []
    vals = dict(zip([n.split(" / ")[0] for n in names], values))
    pairs = [("Q", "α"), ("Q1", "α1"), ("Q2", "α2"),
             ("Qfilm", "αfilm"), ("Qdl", "αdl")]
    for qn, an in pairs:
        if qn in vals and an in vals:
            q = float(vals[qn]); a = float(vals[an])
            if a >= 0.97:
                out.append(
                    f"{an} = {a:.3f}: nearly ideal capacitive behavior; "
                    f"{qn} may be interpreted approximately as C ≈ {q*1e6:.3g} µF."
                )
            elif a >= 0.85:
                out.append(
                    f"{an} = {a:.3f}: mildly non-ideal capacitive behavior; "
                    f"{qn} is a CPE parameter and should not be treated directly as capacitance."
                )
            else:
                out.append(
                    f"{an} = {a:.3f}: pronounced non-ideal/distributed behavior; "
                    f"{qn} is not a capacitance."
                )
    return out


def _effective_capacitance_rows(model, names, values):
    vals = {n.split(" / ")[0]: float(v) for n, v in zip(names, values)}
    rows = []
    for cname, meaning in [
        ("Cdl", "Ideal double-layer capacitance (direct fit parameter)"),
        ("C1", "Ideal capacitance of electrochemical process 1 (direct fit parameter)"),
        ("C2", "Ideal capacitance of electrochemical process 2 (direct fit parameter)"),
    ]:
        if cname in vals and np.isfinite(vals[cname]) and vals[cname] > 0:
            rows.append((cname, vals[cname], "F", meaning))

    branches = [
        ("Rct","Q","α","Ceff,dl","Charge-transfer/double-layer branch"),
        ("R1","Q1","α1","Ceff,1","Electrochemical process 1"),
        ("R2","Q2","α2","Ceff,2","Electrochemical process 2"),
        ("Rfilm","Qfilm","αfilm","Ceff,film","Film/coating branch"),
        ("Rct","Qdl","αdl","Ceff,dl","Charge-transfer/double-layer branch"),
    ]
    seen=set()
    for rn, qn, an, cn, meaning in branches:
        if (rn,qn,an) in seen:
            continue
        seen.add((rn,qn,an))
        if rn in vals and qn in vals and an in vals:
            R,Q,a=vals[rn],vals[qn],vals[an]
            if not (R>0 and Q>0 and np.isfinite(a) and a>=0.50):
                continue
            ceff=(Q*(R**(1.0-a)))**(1.0/a)
            if not np.isfinite(ceff) or ceff <= 0 or ceff > 1.0:
                continue
            if a >= 0.97:
                method="Q approximately equals C because α ≈ 1"
            elif a >= 0.80:
                method="Hsu–Mansfeld effective capacitance; use with moderate confidence"
            else:
                method="Hsu–Mansfeld effective capacitance; model-dependent, use with caution"
            rows.append((cn,ceff,"F",f"{meaning}; {method}"))
    return rows

def _format_capacitance(c):
    a=abs(c)
    if a >= 1e-3: return f"{c*1e3:.5g} mF"
    if a >= 1e-6: return f"{c*1e6:.5g} µF"
    if a >= 1e-9: return f"{c*1e9:.5g} nF"
    return f"{c:.5E} F"

class EISWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EIS Analysis – Nyquist | Bode | Equivalent Circuit Fitting")
        self.resize(1350, 900)

        self.nyquist = None
        self.bode = None
        self.nyquist_path = None
        self.bode_path = None
        self.last_fit = None

        self._build()

    def _build(self):
        central = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(central)
        self.setCentralWidget(scroll)
        outer = QVBoxLayout(central)
        outer.setSpacing(10)

        self.setStyleSheet("""
            QGroupBox {
                font-weight:700; color:#17365D;
                border:1px solid #B8C7D9; border-radius:7px;
                margin-top:10px; padding-top:10px;
            }
            QGroupBox::title {
                subcontrol-origin:margin; left:10px; padding:0 5px;
            }
            QPushButton {
                padding:7px 11px; border-radius:5px;
                background:#2F75B5; color:white; font-weight:600;
            }
            QPushButton:hover { background:#3F86C6; }
            QPushButton#loadButton { background:#2F75B5; }
            QPushButton#previewButton { background:#2E8B57; }
            QPushButton#fitButton { background:#D9822B; }
            QPushButton#diagnosticButton { background:#7A5CC7; }
            QPushButton#exportButton { background:#58636F; }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                background:white; border:1px solid #BFC7D0;
                border-radius:4px; padding:3px 5px;
            }
            QHeaderView::section {
                background:#DCEAF7; color:#17365D;
                font-weight:700; padding:5px; border:1px solid #C3D5E6;
            }
        """)

        title = QLabel("EIS Analysis – Nyquist | Bode | Equivalent Circuit Fitting")
        title.setStyleSheet("font-size:20px;font-weight:700;color:#17365D;")
        outer.addWidget(title)

        subtitle = QLabel(
            "Nyquist and Bode CSV import, preview, six Randles-type "
            "equivalent circuits, complex nonlinear fitting, residuals and Excel export."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color:#4A5568;")
        outer.addWidget(subtitle)

        inp = QGroupBox("1. Input data")
        inp.setStyleSheet("QGroupBox { background:#F5FAFF; }")
        ig = QGridLayout(inp)

        self.nyq_label = QLabel("No file")
        self.bode_label = QLabel("No file")

        b1 = QPushButton("Load Nyquist CSV")
        b1.setObjectName("loadButton")
        b1.clicked.connect(self.load_nyquist)
        b2 = QPushButton("Load Bode CSV")
        b2.setObjectName("loadButton")
        b2.clicked.connect(self.load_bode)

        ig.addWidget(b1, 0, 0)
        ig.addWidget(self.nyq_label, 0, 1)
        ig.addWidget(b2, 1, 0)
        ig.addWidget(self.bode_label, 1, 1)

        self.nyq_sign = QComboBox()
        self.nyq_sign.addItems(["CSV contains Z''", "CSV contains -Z''"])
        self.nyq_sign.setCurrentText("CSV contains -Z''")
        ig.addWidget(QLabel("Nyquist imaginary column"), 0, 2)
        ig.addWidget(self.nyq_sign, 0, 3)

        self.phase_sign = QComboBox()
        self.phase_sign.addItems(["Phase in degrees", "-Phase in degrees"])
        self.phase_sign.setCurrentText("Phase in degrees")
        ig.addWidget(QLabel("Bode phase column"), 1, 2)
        ig.addWidget(self.phase_sign, 1, 3)

        outer.addWidget(inp)

        mapping = QGroupBox("2. Column mapping")
        mapping.setStyleSheet("QGroupBox { background:#F7FBF4; }")
        mg = QGridLayout(mapping)

        self.nyq_f_col = QComboBox()
        self.nyq_re_col = QComboBox()
        self.nyq_im_col = QComboBox()
        self.bode_f_col = QComboBox()
        self.bode_mag_col = QComboBox()
        self.bode_phase_col = QComboBox()

        for c in (
            self.nyq_f_col, self.nyq_re_col, self.nyq_im_col,
            self.bode_f_col, self.bode_mag_col, self.bode_phase_col
        ):
            c.setMinimumWidth(90)

        labels = [
            ("Nyquist frequency", self.nyq_f_col, 0, 0),
            ("Nyquist Z'", self.nyq_re_col, 0, 2),
            ("Nyquist Z'' / -Z''", self.nyq_im_col, 0, 4),
            ("Bode frequency", self.bode_f_col, 1, 0),
            ("Bode |Z|", self.bode_mag_col, 1, 2),
            ("Bode phase", self.bode_phase_col, 1, 4),
        ]
        for text, widget, row, col in labels:
            mg.addWidget(QLabel(text), row, col)
            mg.addWidget(widget, row, col + 1)

        note = QLabel(
            "If the Nyquist export has no frequency column, choose 'none' for Nyquist frequency. "
            "The program will then use the Bode frequencies when the row counts match."
        )
        note.setWordWrap(True)
        mg.addWidget(note, 2, 0, 1, 6)
        outer.addWidget(mapping)

        circuit = QGroupBox("3. Equivalent circuit")
        circuit.setStyleSheet("QGroupBox { background:#FFF9F0; }")
        cg = QGridLayout(circuit)

        self.model = QComboBox()
        self.model.addItems([
            "Rs + (Rct || Cdl)",
            "Rs + (Rct || CPE)",
            "Rs + (Rct || Cdl) + W",
            "Rs + (Rct || CPE) + W",
            "Rs + (R1 || CPE1) + (R2 || CPE2)",
            "Rs + (R1 || CPE1) + (R2 || CPE2) + W",
            "Rs + (Rct || CPE) + Wo",
            "Rs + (Rct || CPE) + Ws",
            "Rs + (Rfilm || CPEfilm) + (Rct || CPEdl)",
            "Rs + (Rfilm || CPEfilm) + (Rct || CPEdl) + W",
            "Rs + (R1 || C1) + (R2 || C2)",
            "Rs + (R1 || C1) + (R2 || C2) + W",
            "Rs + (R1 || C1) + (R2 || CPE2)",
            "Rs + (R1 || CPE1) + (R2 || C2)",
            "Rs + (R1 || C1) + (R2 || CPE2) + W",
            "Rs + (R1 || CPE1) + (R2 || C2) + W",
            "Rs + L + (Rct || Cdl)",
            "Rs + L + (Rct || CPE)",
            "Rs + L + (Rct || CPE) + W",
            "Rs + L + (Rct || CPE) + Wo",
            "Rs + L + (Rct || CPE) + Ws",
            "Rs + L + (R1 || C1) + (R2 || C2)",
            "Rs + L + (R1 || CPE1) + (R2 || CPE2)",
            "Rs + L + (R1 || C1) + (R2 || CPE2)",
            "Rs + L + (R1 || CPE1) + (R2 || C2)",
        ])
        self.model.currentTextChanged.connect(self._refresh_parameter_table)
        cg.addWidget(QLabel("Circuit model"), 0, 0)
        cg.addWidget(self.model, 0, 1, 1, 3)

        self.weighting = QComboBox()
        self.weighting.addItems(["1 / |Z| weighting", "Unweighted"])
        cg.addWidget(QLabel("Fit weighting"), 0, 4)
        cg.addWidget(self.weighting, 0, 5)

        weighting_help = QLabel(
            "Fit weighting determines how strongly each impedance point contributes to the least-squares fit. "
            "1 / |Z| weighting reduces the dominance of high-impedance data points and gives greater emphasis "
            "to relative rather than purely absolute impedance deviations."
        )
        weighting_help.setWordWrap(True)
        weighting_help.setStyleSheet("color:#444; padding:2px 4px 5px 4px;")
        cg.addWidget(weighting_help)

        self.param_table = QTableWidget(0, 4)
        self.param_table.setHorizontalHeaderLabels(["Parameter", "Meaning", "Initial value", "Fit value"])
        self.param_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.param_table.setMaximumHeight(180)
        cg.addWidget(self.param_table, 1, 0, 1, 6)

        outer.addWidget(circuit)

        optional = QGroupBox("4. Optional electrochemical parameters for later Warburg-D analysis")
        optional.setStyleSheet("QGroupBox { background:#F8F4FF; }")
        og = QGridLayout(optional)
        self.area = QDoubleSpinBox()
        self.area.setDecimals(6)
        self.area.setRange(1e-9, 1e6)
        self.area.setValue(0.785)
        self.conc = QLineEdit("1e-3")
        self.ne = QSpinBox()
        self.ne.setRange(1, 20)
        self.ne.setValue(1)
        self.temp = QDoubleSpinBox()
        self.temp.setDecimals(2)
        self.temp.setRange(1, 2000)
        self.temp.setValue(298.15)
        for k, (lab, w) in enumerate([
            ("Area A / cm²", self.area),
            ("Concentration / mol L⁻¹", self.conc),
            ("n / electrons", self.ne),
            ("Temperature / K", self.temp),
        ]):
            og.addWidget(QLabel(lab), 0, 2*k)
            og.addWidget(w, 0, 2*k + 1)
        outer.addWidget(optional)

        actions = QHBoxLayout()
        for txt, fn in [
            ("Preview Nyquist + Bode", self.preview),
            ("Fit equivalent circuit", self.fit),
            ("Compare circuit models", self.compare_models),
            ("Kramers-Kronig check", self.kramers_kronig_check),
            ("Fit stability analysis", self.fit_stability_analysis),
            ("Show residuals", self.show_residuals),
            ("Export Excel", self.export_excel),
        ]:
            b = QPushButton(txt)
            if txt == "Preview Nyquist + Bode": b.setObjectName("previewButton")
            elif txt in ("Fit equivalent circuit", "Compare circuit models"): b.setObjectName("fitButton")
            elif txt == "Export Excel": b.setObjectName("exportButton")
            else: b.setObjectName("diagnosticButton")
            b.clicked.connect(fn)
            actions.addWidget(b)
        outer.addLayout(actions)

        self.status = QLabel("Ready.")
        self.status.setStyleSheet("padding:8px;background:#F3F6F9;border:1px solid #CCD6E0;")
        outer.addWidget(self.status)

        self._refresh_parameter_table()

    def _fill_combo(self, combo, ncols, allow_none=False, default=None):
        combo.blockSignals(True)
        combo.clear()
        if allow_none:
            combo.addItem("none", -1)
        for i in range(ncols):
            combo.addItem(f"column {i+1}", i)
        if default is not None:
            idx = combo.findData(default)
            if idx >= 0:
                combo.setCurrentIndex(idx)
        combo.blockSignals(False)

    def load_nyquist(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Load Nyquist CSV", "", "CSV/TXT (*.csv *.txt);;All files (*)"
        )
        if not p:
            return
        try:
            self.nyquist = _read_numeric_csv(p)
            self.nyquist_path = p
            self.nyq_label.setText(Path(p).name)
            n = self.nyquist.shape[1]

            # Common exports: Zre, -Zim OR f, Zre, -Zim.
            self._fill_combo(self.nyq_f_col, n, allow_none=True, default=(2 if n >= 3 else -1))
            self._fill_combo(self.nyq_re_col, n, default=0)
            self._fill_combo(self.nyq_im_col, n, default=(1 if n >= 2 else 0))
            # The user's EIS export format is Z', -Z'', frequency.
            if n >= 3:
                self.nyq_sign.setCurrentText("CSV contains -Z''")
            self.status.setText(
                f"Nyquist loaded: {self.nyquist.shape[0]} rows × {n} columns."
            )
        except Exception as e:
            QMessageBox.critical(self, "Nyquist import", str(e))

    def load_bode(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Load Bode CSV", "", "CSV/TXT (*.csv *.txt);;All files (*)"
        )
        if not p:
            return
        try:
            self.bode = _read_numeric_csv(p)
            self.bode_path = p
            self.bode_label.setText(Path(p).name)
            n = self.bode.shape[1]
            self._fill_combo(self.bode_f_col, n, default=0)
            self._fill_combo(self.bode_mag_col, n, allow_none=True, default=(-1 if n == 2 else 1))
            self._fill_combo(self.bode_phase_col, n, default=(1 if n == 2 else (2 if n >= 3 else 0)))
            self.status.setText(
                f"Bode loaded: {self.bode.shape[0]} rows × {n} columns."
            )
        except Exception as e:
            QMessageBox.critical(self, "Bode import", str(e))

    def _col(self, df, combo):
        idx = int(combo.currentData())
        if idx < 0:
            return None
        return df.iloc[:, idx].to_numpy(dtype=float)

    def _nyquist_arrays(self):
        if self.nyquist is None:
            raise ValueError("Load Nyquist CSV first.")
        f = self._col(self.nyquist, self.nyq_f_col)
        zr = self._col(self.nyquist, self.nyq_re_col)
        zi_in = self._col(self.nyquist, self.nyq_im_col)

        if self.nyq_sign.currentText() == "CSV contains -Z''":
            zi = -zi_in
        else:
            zi = zi_in

        good = np.isfinite(zr) & np.isfinite(zi)
        if f is not None:
            good &= np.isfinite(f) & (f > 0)
            f = f[good]
        return f, zr[good], zi[good]

    def _bode_arrays(self):
        if self.bode is None:
            raise ValueError("Load Bode CSV first.")
        f = self._col(self.bode, self.bode_f_col)
        phase = self._col(self.bode, self.bode_phase_col)
        if self.phase_sign.currentText() == "-Phase in degrees":
            phase = -phase
        good = np.isfinite(f) & (f > 0) & np.isfinite(phase)
        f = f[good]
        phase = phase[good]

        mag = None
        mag_col = self._col(self.bode, self.bode_mag_col)
        if mag_col is not None:
            mag = mag_col[good]
        return f, mag, phase

    def _complex_dataset(self):
        f_n, zr, zi = self._nyquist_arrays()
        if f_n is None:
            if self.bode is None:
                raise ValueError(
                    "Nyquist CSV has no frequency column. Load Bode CSV so its frequencies can be used."
                )
            f_b, _, _ = self._bode_arrays()
            if len(f_b) != len(zr):
                raise ValueError(
                    "Nyquist has no frequency column and its row count does not match the Bode row count."
                )
            f_n = f_b

        z = zr + 1j * zi
        good = np.isfinite(f_n) & (f_n > 0) & np.isfinite(z.real) & np.isfinite(z.imag)
        return f_n[good], z[good]

    def _refresh_parameter_table(self):
        model = self.model.currentText()
        names = _parameter_names(model)
        defaults = {
            "Rs / Ω": "10",
            "Rct / Ω": "100",
            "Cdl / F": "1e-5",
            "Q / S·s^α": "1e-5",
            "α": "0.9",
            "R1 / Ω": "50",
            "Q1 / S·s^α1": "1e-5",
            "α1": "0.9",
            "R2 / Ω": "50",
            "C1 / F": "1e-6",
            "C2 / F": "1e-5",
            "L / H": "1e-6",
            "Q2 / S·s^α2": "1e-4",
            "α2": "0.9",
            "Rfilm / Ω": "30",
            "Qfilm / S·s^αfilm": "1e-6",
            "αfilm": "0.9",
            "Qdl / S·s^αdl": "1e-5",
            "αdl": "0.9",
            "τW / s": "1",
            "σW / Ω·s^-1/2": "10",
        }
        old = {}
        for r in range(self.param_table.rowCount()):
            name_item = self.param_table.item(r, 0)
            init_item = self.param_table.item(r, 2)
            if name_item and init_item:
                old[name_item.text()] = init_item.text()

        self.param_table.setRowCount(len(names))
        for r, name in enumerate(names):
            self.param_table.setItem(r, 0, QTableWidgetItem(name))
            self.param_table.setItem(r, 1, QTableWidgetItem(_parameter_meaning(name)))
            self.param_table.setItem(r, 2, QTableWidgetItem(old.get(name, defaults[name])))
            self.param_table.setItem(r, 3, QTableWidgetItem(""))

    def _initial_params(self):
        vals = []
        for r in range(self.param_table.rowCount()):
            text = self.param_table.item(r, 2).text().strip().replace(",", ".")
            vals.append(float(text))
        return np.asarray(vals, dtype=float)

    def _bounds(self):
        names = _parameter_names(self.model.currentText())
        lo, hi = [], []
        for n in names:
            if n in ("α", "α1", "α2", "αfilm", "αdl"):
                lo.append(0.2)
                hi.append(1.0)
            else:
                lo.append(1e-15)
                hi.append(np.inf)
        return np.asarray(lo), np.asarray(hi)

    def _estimate_initials(self, f, z):
        """Data-driven starting values for one- and two-time-constant Randles models."""
        zr = z.real
        zi = z.imag
        Rs = max(float(np.nanmin(zr)), 1e-9)
        Rtotal = max(float(np.nanmax(zr) - Rs), 1e-6)

        idx = int(np.nanargmax(-zi))
        f_peak = max(float(f[idx]), 1e-12)
        C = max(1.0 / (2.0 * np.pi * f_peak * Rtotal), 1e-12)

        # Two-time-constant starts: split total polarization resistance and place
        # one characteristic time above and one below the dominant arc frequency.
        R1 = max(0.45 * Rtotal, 1e-6)
        R2 = max(0.55 * Rtotal, 1e-6)
        f1 = max(f_peak * 8.0, 1e-12)
        f2 = max(f_peak / 8.0, 1e-12)
        Q1 = max(1.0 / (2.0 * np.pi * f1 * R1), 1e-12)
        Q2 = max(1.0 / (2.0 * np.pi * f2 * R2), 1e-12)

        sigma0 = max(Rtotal * np.sqrt(2*np.pi*np.nanmin(f)) * 0.05, 1e-9)

        model = self.model.currentText()
        if model == "Rs + (Rct || Cdl)":
            return np.array([Rs, Rtotal, C])
        if model == "Rs + (Rct || CPE)":
            return np.array([Rs, Rtotal, C, 0.90])
        if model == "Rs + (Rct || Cdl) + W":
            return np.array([Rs, Rtotal, C, sigma0])
        if model == "Rs + (Rct || CPE) + W":
            return np.array([Rs, Rtotal, C, 0.90, sigma0])
        if model == "Rs + (R1 || CPE1) + (R2 || CPE2)":
            return np.array([Rs, R1, Q1, 0.90, R2, Q2, 0.90])
        if model == "Rs + (R1 || CPE1) + (R2 || CPE2) + W":
            return np.array([Rs, R1, Q1, 0.90, R2, Q2, 0.90, sigma0])
        if model in ("Rs + (Rct || CPE) + Wo", "Rs + (Rct || CPE) + Ws"):
            tau0 = max(1.0 / (2.0 * np.pi * max(np.nanmin(f), 1e-12)), 1e-6)
            return np.array([Rs, Rtotal, C, 0.90, sigma0, tau0])
        if model == "Rs + (Rfilm || CPEfilm) + (Rct || CPEdl)":
            return np.array([Rs, R1, Q1, 0.90, R2, Q2, 0.90])
        if model == "Rs + (Rfilm || CPEfilm) + (Rct || CPEdl) + W":
            return np.array([Rs, R1, Q1, 0.90, R2, Q2, 0.90, sigma0])
        if model == "Rs + (R1 || C1) + (R2 || C2)":
            return np.array([Rs, R1, Q1, R2, Q2])
        if model == "Rs + (R1 || C1) + (R2 || C2) + W":
            return np.array([Rs, R1, Q1, R2, Q2, sigma0])
        if model == "Rs + (R1 || C1) + (R2 || CPE2)":
            return np.array([Rs, R1, Q1, R2, Q2, 0.90])
        if model == "Rs + (R1 || CPE1) + (R2 || C2)":
            return np.array([Rs, R1, Q1, 0.90, R2, Q2])
        if model == "Rs + (R1 || C1) + (R2 || CPE2) + W":
            return np.array([Rs, R1, Q1, R2, Q2, 0.90, sigma0])
        if model == "Rs + (R1 || CPE1) + (R2 || C2) + W":
            return np.array([Rs, R1, Q1, 0.90, R2, Q2, sigma0])
        L0 = 1e-6
        if model == "Rs + L + (Rct || Cdl)":
            return np.array([Rs,L0,R1,Q1])
        if model == "Rs + L + (Rct || CPE)":
            return np.array([Rs,L0,R1,Q1,0.90])
        if model == "Rs + L + (Rct || CPE) + W":
            return np.array([Rs,L0,R1,Q1,0.90,sigma0])
        if model in ("Rs + L + (Rct || CPE) + Wo","Rs + L + (Rct || CPE) + Ws"):
            return np.array([Rs,L0,R1,Q1,0.90,max(R1,1.0),1.0])
        if model == "Rs + L + (R1 || C1) + (R2 || C2)":
            return np.array([Rs,L0,R1,Q1,R2,Q2])
        if model == "Rs + L + (R1 || CPE1) + (R2 || CPE2)":
            return np.array([Rs,L0,R1,Q1,0.90,R2,Q2,0.90])
        if model == "Rs + L + (R1 || C1) + (R2 || CPE2)":
            return np.array([Rs,L0,R1,Q1,R2,Q2,0.90])
        if model == "Rs + L + (R1 || CPE1) + (R2 || C2)":
            return np.array([Rs,L0,R1,Q1,0.90,R2,Q2])
        raise ValueError(f"Unknown model: {model}")

    def _set_initial_table(self, p):
        for r, val in enumerate(p):
            self.param_table.setItem(r, 3, QTableWidgetItem(f"{val:.6E}"))

    def preview(self):
        try:
            f_n, zr, zi = self._nyquist_arrays()
            f_b, mag, phase = self._bode_arrays()
            # If Bode CSV has no |Z| column, derive magnitude from Nyquist Z' and Z''.
            f_mag = f_n if f_n is not None else None
            mag_from_nyq = np.sqrt(zr**2 + zi**2)

            d = QDialog(self)
            d.setWindowTitle("EIS preview")
            d.resize(1250, 820)
            lay = QVBoxLayout(d)

            fig = Figure(figsize=(11, 7), tight_layout=True)
            canvas = FigureCanvas(fig)
            lay.addWidget(NavigationToolbar(canvas, d))
            lay.addWidget(canvas)

            ax1 = fig.add_subplot(131)
            ax2 = fig.add_subplot(132)
            ax3 = fig.add_subplot(133)

            ax1.plot(zr, -zi, "o-")
            ax1.set_xlabel("Z' / Ω")
            ax1.set_ylabel("-Z'' / Ω")
            ax1.set_title("Nyquist")
            ax1.grid(alpha=.25)
            ax1.set_aspect("equal", adjustable="datalim")

            if mag is None:
                if f_mag is None:
                    raise ValueError("Cannot derive |Z| because the Nyquist file has no frequency column.")
                order_mag = np.argsort(f_mag)
                ax2.semilogx(f_mag[order_mag], mag_from_nyq[order_mag], "o-")
            else:
                order_mag = np.argsort(f_b)
                ax2.semilogx(f_b[order_mag], mag[order_mag], "o-")
            order = np.argsort(f_b)
            ax2.set_xlabel("Frequency / Hz")
            ax2.set_ylabel("|Z| / Ω")
            ax2.set_title("Bode magnitude")
            ax2.grid(alpha=.25)

            ax3.semilogx(f_b[order], phase[order], "o-")
            ax3.set_xlabel("Frequency / Hz")
            ax3.set_ylabel("Phase / °")
            ax3.set_title("Bode phase")
            ax3.grid(alpha=.25)

            canvas.draw()
            d.exec()
        except Exception as e:
            QMessageBox.critical(self, "EIS preview", str(e))

    def _fit_model_core(self, model, f, z_meas, initial_override=None):
        """Fit one circuit model and return fit statistics without changing the UI."""
        current_model = self.model.currentText()
        self.model.blockSignals(True)
        self.model.setCurrentText(model)
        self.model.blockSignals(False)
        try:
            p0 = np.asarray(initial_override, float) if initial_override is not None else self._estimate_initials(f, z_meas)
            names = _parameter_names(model)
            lo, hi = [], []
            for name in names:
                if name in ("α", "α1", "α2", "αfilm", "αdl"):
                    lo.append(0.2); hi.append(1.0)
                else:
                    lo.append(1e-15); hi.append(np.inf)
            lo = np.asarray(lo, dtype=float)
            hi = np.asarray(hi, dtype=float)

            def residual(p):
                zf = _circuit_impedance(f, model, p)
                dr = zf.real - z_meas.real
                di = zf.imag - z_meas.imag
                if self.weighting.currentText() == "1 / |Z| weighting":
                    wt = 1.0 / np.maximum(np.abs(z_meas), 1e-12)
                    dr = dr * wt
                    di = di * wt
                return np.concatenate([dr, di])

            result = least_squares(
                residual, p0, bounds=(lo, hi), max_nfev=30000,
                xtol=1e-12, ftol=1e-12, gtol=1e-12
            )
            pfit = result.x
            zfit = _circuit_impedance(f, model, pfit)
            raw = np.concatenate([zfit.real-z_meas.real, zfit.imag-z_meas.imag])
            rss = float(np.sum(raw**2))
            nobs = 2 * len(f)
            k = len(pfit)
            rmse = float(np.sqrt(rss / max(nobs-k, 1)))
            aic = float(nobs * np.log(max(rss/nobs, 1e-300)) + 2*k)
            aicc = (
                float(aic + (2*k*(k+1))/(nobs-k-1))
                if nobs > k + 1 else np.inf
            )
            bic = float(nobs * np.log(max(rss/nobs, 1e-300)) + k*np.log(nobs))
            return {
                "model": model, "params": pfit, "param_names": names,
                "z_fit": zfit, "rss": rss, "rmse": rmse,
                "aic": aic, "aicc": aicc, "bic": bic,
                "success": bool(result.success), "message": result.message,
            }
        finally:
            self.model.blockSignals(True)
            self.model.setCurrentText(current_model)
            self.model.blockSignals(False)

    def fit(self):
        try:
            f, z_meas = self._complex_dataset()
            model = self.model.currentText()

            # Put data-driven guesses into the table on the first fit attempt.
            p_guess = self._estimate_initials(f, z_meas)
            self._set_initial_table(p_guess)
            p0 = self._initial_params()
            bounds = self._bounds()

            def residual(p):
                zf = _circuit_impedance(f, model, p)
                dr = zf.real - z_meas.real
                di = zf.imag - z_meas.imag
                if self.weighting.currentText() == "1 / |Z| weighting":
                    w = 1.0 / np.maximum(np.abs(z_meas), 1e-12)
                    dr = dr * w
                    di = di * w
                return np.concatenate([dr, di])

            result = least_squares(
                residual, p0, bounds=bounds, max_nfev=20000,
                xtol=1e-12, ftol=1e-12, gtol=1e-12
            )
            pfit = result.x
            zfit = _circuit_impedance(f, model, pfit)
            raw_res = np.concatenate([zfit.real-z_meas.real, zfit.imag-z_meas.imag])
            rss = float(np.sum(raw_res**2))
            n = 2 * len(f)
            k = len(pfit)
            rmse = float(np.sqrt(rss / max(n-k, 1)))
            aic = float(n * np.log(max(rss/n, 1e-300)) + 2*k)

            for r, val in enumerate(pfit):
                self.param_table.setItem(r, 3, QTableWidgetItem(f"{val:.6E}"))

            self.last_fit = {
                "model": model,
                "freq": f,
                "z_meas": z_meas,
                "z_fit": zfit,
                "params": pfit,
                "param_names": _parameter_names(model),
                "success": bool(result.success),
                "message": result.message,
                "rss": rss,
                "rmse": rmse,
                "aic": aic,
            }

            d = QDialog(self)
            d.setWindowTitle(f"EIS fit – {model}")
            d.resize(1300, 880)
            lay = QVBoxLayout(d)

            head = QLabel(
                f"Model: {model} | success: {result.success} | RMSE = {rmse:.4E} Ω | AIC = {aic:.3f}"
            )
            head.setStyleSheet("font-weight:700;padding:7px;background:#EEF6FF;border:1px solid #9CC4E4;")
            lay.addWidget(head)

            fig = Figure(figsize=(11, 7), tight_layout=True)
            canvas = FigureCanvas(fig)
            lay.addWidget(NavigationToolbar(canvas, d))
            lay.addWidget(canvas)

            ax1 = fig.add_subplot(131)
            ax2 = fig.add_subplot(132)
            ax3 = fig.add_subplot(133)

            ax1.plot(z_meas.real, -z_meas.imag, "o", label="Measured")
            ax1.plot(zfit.real, -zfit.imag, "-", lw=2, label="Fit")
            ax1.set_xlabel("Z' / Ω")
            ax1.set_ylabel("-Z'' / Ω")
            ax1.set_title("Nyquist")
            ax1.grid(alpha=.25)
            ax1.set_aspect("equal", adjustable="datalim")
            ax1.legend()

            order = np.argsort(f)
            mag_m = np.abs(z_meas)
            mag_f = np.abs(zfit)
            ph_m = _phase_deg(z_meas)
            ph_f = _phase_deg(zfit)

            ax2.semilogx(f[order], mag_m[order], "o", label="Measured")
            ax2.semilogx(f[order], mag_f[order], "-", lw=2, label="Fit")
            ax2.set_xlabel("Frequency / Hz")
            ax2.set_ylabel("|Z| / Ω")
            ax2.set_title("Bode magnitude")
            ax2.grid(alpha=.25)
            ax2.legend()

            ax3.semilogx(f[order], ph_m[order], "o", label="Measured")
            ax3.semilogx(f[order], ph_f[order], "-", lw=2, label="Fit")
            ax3.set_xlabel("Frequency / Hz")
            ax3.set_ylabel("Phase / °")
            ax3.set_title("Bode phase")
            ax3.grid(alpha=.25)
            ax3.legend()

            table = QTableWidget(len(pfit), 3)
            table.setHorizontalHeaderLabels(["Parameter", "Meaning", "Fit value"])
            table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            for r, (name, val) in enumerate(zip(_parameter_names(model), pfit)):
                table.setItem(r, 0, QTableWidgetItem(name))
                table.setItem(r, 1, QTableWidgetItem(_parameter_meaning(name)))
                table.setItem(r, 2, QTableWidgetItem(f"{val:.6E}"))
            table.setMaximumHeight(210)
            lay.addWidget(table)

            cpe_notes = _cpe_interpretations(_parameter_names(model), pfit)
            if cpe_notes:
                cpe_box = QLabel("CPE interpretation:\n" + "\n".join("• " + x for x in cpe_notes))
                cpe_box.setWordWrap(True)
                cpe_box.setStyleSheet(
                    "padding:8px;background:#F6F2FF;border:1px solid #B9A7E8;font-weight:600;"
                )
                lay.addWidget(cpe_box)

            canvas.draw()
            d.exec()

            self.status.setText(
                f"EIS fit complete: {model}; RMSE={rmse:.4E} Ω; AIC={aic:.2f}"
            )

        except Exception as e:
            QMessageBox.critical(self, "Equivalent-circuit fit", str(e))

    def compare_models(self):
        try:
            f, z_meas = self._complex_dataset()
            models = [
                "Rs + (Rct || Cdl)",
                "Rs + (Rct || CPE)",
                "Rs + (Rct || Cdl) + W",
                "Rs + (Rct || CPE) + W",
                "Rs + (R1 || CPE1) + (R2 || CPE2)",
                "Rs + (R1 || CPE1) + (R2 || CPE2) + W",
                "Rs + (Rct || CPE) + Wo",
                "Rs + (Rct || CPE) + Ws",
                "Rs + (Rfilm || CPEfilm) + (Rct || CPEdl)",
                "Rs + (Rfilm || CPEfilm) + (Rct || CPEdl) + W",
                "Rs + (R1 || C1) + (R2 || C2)",
                "Rs + (R1 || C1) + (R2 || C2) + W",
                "Rs + (R1 || C1) + (R2 || CPE2)",
                "Rs + (R1 || CPE1) + (R2 || C2)",
                "Rs + (R1 || C1) + (R2 || CPE2) + W",
                "Rs + (R1 || CPE1) + (R2 || C2) + W",
                "Rs + L + (Rct || Cdl)",
                "Rs + L + (Rct || CPE)",
                "Rs + L + (Rct || CPE) + W",
                "Rs + L + (Rct || CPE) + Wo",
                "Rs + L + (Rct || CPE) + Ws",
                "Rs + L + (R1 || C1) + (R2 || C2)",
                "Rs + L + (R1 || CPE1) + (R2 || CPE2)",
                "Rs + L + (R1 || C1) + (R2 || CPE2)",
                "Rs + L + (R1 || CPE1) + (R2 || C2)",
            ]

            results = []
            for model in models:
                try:
                    results.append(self._fit_model_core(model, f, z_meas))
                except Exception as exc:
                    results.append({
                        "model": model, "params": np.array([]), "param_names": [],
                        "z_fit": None, "rss": np.nan, "rmse": np.nan,
                        "aic": np.nan, "aicc": np.nan, "bic": np.nan,
                        "success": False, "message": str(exc),
                    })

            valid = [r for r in results if r["success"] and np.isfinite(r["aicc"])]
            if not valid:
                raise ValueError("None of the circuit models could be fitted successfully.")

            best = min(valid, key=lambda r: r["aicc"])
            best_aicc = best["aicc"]
            for r in results:
                r["delta_aicc"] = (
                    r["aicc"] - best_aicc if np.isfinite(r["aicc"]) else np.nan
                )

            ordered = sorted(
                results,
                key=lambda r: r["aicc"] if np.isfinite(r["aicc"]) else np.inf
            )

            # Rank by AICc with ties for numerically identical AICc.
            last_aicc = None
            last_rank = 0
            for idx, r in enumerate(ordered, start=1):
                if last_aicc is None or not np.isfinite(r["aicc"]) or abs(r["aicc"] - last_aicc) > 1e-9:
                    last_rank = idx
                    last_aicc = r["aicc"]
                r["rank"] = last_rank

            self.model_comparison = ordered
            self.best_model_comparison = best

            d = QDialog(self)
            d.setWindowTitle("EIS equivalent-circuit model comparison")
            d.resize(1450, 940)
            lay = QVBoxLayout(d)

            expl = QLabel(
                "<b>Rank</b> = AICc-based model position (1 = best-supported model; identical AICc values share a rank).  "
                "<b>RMSE</b> = root mean square fit error in Ω; lower is better.  "
                "<b>AIC</b> = Akaike information criterion; balances fit quality and model complexity.  "
                "<b>AICc</b> = AIC corrected for finite sample size; primary ranking criterion here.  "
                "<b>ΔAICc</b> = difference from the best model (0 = best; < 2 ≈ similarly supported; > 10 = weak support).  "
                "<b>BIC</b> = Bayesian information criterion; penalizes additional parameters more strongly than AIC."
            )
            expl.setWordWrap(True)
            expl.setStyleSheet(
                "font-weight:500;padding:9px;background:#F4F8FC;border:1px solid #B7C8D8;"
            )
            lay.addWidget(expl)

            displayed = QLabel("")
            displayed.setStyleSheet(
                "font-size:14px;font-weight:700;padding:7px;background:#F3FAF3;border:1px solid #9BC79B;"
            )
            lay.addWidget(displayed)

            fig = Figure(figsize=(11, 6), tight_layout=True)
            canvas = FigureCanvas(fig)
            lay.addWidget(NavigationToolbar(canvas, d))
            lay.addWidget(canvas, 1)

            ax1 = fig.add_subplot(131)
            ax2 = fig.add_subplot(132)
            ax3 = fig.add_subplot(133)

            table = QTableWidget(len(ordered), 8)
            table.setHorizontalHeaderLabels([
                "Rank", "Model", "Parameters", "RMSE / Ω", "AIC", "AICc", "ΔAICc", "BIC"
            ])
            header = table.horizontalHeader()
            header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
            table.setColumnWidth(0, 58)
            table.setColumnWidth(1, 600)
            table.setColumnWidth(2, 78)
            table.setColumnWidth(3, 112)
            table.setColumnWidth(4, 88)
            table.setColumnWidth(5, 88)
            table.setColumnWidth(6, 92)
            table.setColumnWidth(7, 88)
            header.setStretchLastSection(True)
            table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
            table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)

            for row, r in enumerate(ordered):
                values = [
                    str(r["rank"]),
                    r["model"],
                    str(len(r["params"])),
                    "" if not np.isfinite(r["rmse"]) else f'{r["rmse"]:.5E}',
                    "" if not np.isfinite(r["aic"]) else f'{r["aic"]:.3f}',
                    "" if not np.isfinite(r["aicc"]) else f'{r["aicc"]:.3f}',
                    "" if not np.isfinite(r["delta_aicc"]) else f'{r["delta_aicc"]:.3f}',
                    "" if not np.isfinite(r["bic"]) else f'{r["bic"]:.3f}',
                ]
                for col, value in enumerate(values):
                    table.setItem(row, col, QTableWidgetItem(value))
            table.setMaximumHeight(280)
            lay.addWidget(table)

            bottom = QHBoxLayout()
            use_selected = QPushButton("Use selected model in main window")
            circuit_btn = QPushButton("Show circuit block diagram")
            cap_btn = QPushButton("Capacitance analysis")
            close_btn = QPushButton("Close")
            bottom.addWidget(use_selected)
            bottom.addWidget(circuit_btn)
            bottom.addWidget(cap_btn)
            bottom.addStretch(1)
            bottom.addWidget(close_btn)
            lay.addLayout(bottom)

            state = {"selected": best}

            def draw_result(r):
                if r is None or r.get("z_fit") is None:
                    return

                state["selected"] = r
                zfit = r["z_fit"]
                order = np.argsort(f)

                ax1.clear()
                ax2.clear()
                ax3.clear()

                ax1.plot(z_meas.real, -z_meas.imag, "o", label="Measured")
                ax1.plot(zfit.real, -zfit.imag, "-", lw=2, label="Fit")
                ax1.set_xlabel("Z' / Ω")
                ax1.set_ylabel("-Z'' / Ω")
                ax1.set_title("Nyquist")
                ax1.grid(alpha=.25)
                ax1.set_aspect("equal", adjustable="datalim")
                ax1.legend()

                ax2.semilogx(f[order], np.abs(z_meas)[order], "o", label="Measured")
                ax2.semilogx(f[order], np.abs(zfit)[order], "-", lw=2, label="Fit")
                ax2.set_xlabel("Frequency / Hz")
                ax2.set_ylabel("|Z| / Ω")
                ax2.set_title("Bode magnitude")
                ax2.grid(alpha=.25)
                ax2.legend()

                ax3.semilogx(f[order], _phase_deg(z_meas)[order], "o", label="Measured")
                ax3.semilogx(f[order], _phase_deg(zfit)[order], "-", lw=2, label="Fit")
                ax3.set_xlabel("Frequency / Hz")
                ax3.set_ylabel("Phase / °")
                ax3.set_title("Bode phase")
                ax3.grid(alpha=.25)
                ax3.legend()

                da = r.get("delta_aicc", np.nan)
                displayed.setText(
                    f"Displayed model: {r['model']} — Rank {r['rank']}, "
                    f"RMSE = {r['rmse']:.4E} Ω, AICc = {r['aicc']:.3f}, "
                    f"ΔAICc = {da:.3f}"
                )
                canvas.draw_idle()

            def row_selected(row, col):
                if 0 <= row < len(ordered):
                    draw_result(ordered[row])

            def use_current():
                r = state["selected"]
                if r is None or r.get("z_fit") is None:
                    return
                self.model.setCurrentText(r["model"])
                self._refresh_parameter_table()
                for i, val in enumerate(r["params"]):
                    self.param_table.setItem(i, 2, QTableWidgetItem(f"{val:.6E}"))
                    self.param_table.setItem(i, 3, QTableWidgetItem(f"{val:.6E}"))
                d.accept()

            def show_current_circuit():
                r = state["selected"]
                if r is not None:
                    self._show_circuit_diagram(r["model"], r.get("params"), r.get("param_names"))

            def show_current_capacitance():
                r = state["selected"]
                if r is not None:
                    self._show_capacitance_analysis(r["model"], r.get("params"), r.get("param_names"))

            table.cellClicked.connect(row_selected)
            use_selected.clicked.connect(use_current)
            circuit_btn.clicked.connect(show_current_circuit)
            cap_btn.clicked.connect(show_current_capacitance)
            close_btn.clicked.connect(d.accept)

            # Select and display the statistically best model initially.
            best_row = next(i for i, r in enumerate(ordered) if r is best)
            table.selectRow(best_row)
            draw_result(best)

            canvas.draw()
            d.exec()

            self.status.setText(
                f"Model comparison complete. Best AICc model: {best['model']} "
                f"(RMSE={best['rmse']:.4E} Ω; AICc={best['aicc']:.2f})."
            )

        except Exception as e:
            QMessageBox.critical(self, "Model comparison", str(e))

    def _show_capacitance_analysis(self, model, params=None, param_names=None):
        if params is None or param_names is None:
            QMessageBox.information(self, "Capacitance analysis", "No fitted parameters are available.")
            return
        rows = _effective_capacitance_rows(model, param_names, params)
        if not rows:
            QMessageBox.information(
                self, "Capacitance analysis",
                "The selected circuit does not contain an ideal capacitance or an R || CPE branch "
                "from which an effective capacitance can be estimated."
            )
            return

        d=QDialog(self); d.setWindowTitle("Capacitance analysis"); d.resize(920,430)
        lay=QVBoxLayout(d)
        info=QLabel(
            "<b>Capacitance analysis</b><br>"
            "For an ideal capacitor, the fitted C value is reported directly. "
            "For a parallel R || CPE branch, an effective capacitance is estimated from "
            "<b>C<sub>eff</sub> = [Q·R<sup>(1−α)</sup>]<sup>1/α</sup></b> "
            "(Hsu–Mansfeld characteristic-frequency relation). "
            "When α ≈ 1, Q approaches an ordinary capacitance. "
            "For strongly non-ideal CPEs, Ceff is a model-dependent effective quantity rather than a unique physical capacitance."
        )
        info.setWordWrap(True)
        info.setStyleSheet("padding:9px;background:#F4F8FC;border:1px solid #B7C8D8;")
        lay.addWidget(info)

        t=QTableWidget(len(rows),4)
        t.setHorizontalHeaderLabels(["Quantity","Value","Unit","Meaning / method"])
        t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        for r,(name,val,unit,meaning) in enumerate(rows):
            t.setItem(r,0,QTableWidgetItem(name))
            t.setItem(r,1,QTableWidgetItem(_format_capacitance(val).split()[0]))
            t.setItem(r,2,QTableWidgetItem(_format_capacitance(val).split()[1]))
            t.setItem(r,3,QTableWidgetItem(meaning))
        lay.addWidget(t)

        note=QLabel(
            "<b>Interpretation:</b> Capacitance values can help assign time constants. "
            "For example, a double-layer process and a thin film/coating process may occupy different capacitance ranges, "
            "but assignment should always be supported by the electrochemical system and electrode geometry."
        )
        note.setWordWrap(True); note.setStyleSheet("padding:8px;background:#FFF9E8;border:1px solid #D8C786;")
        lay.addWidget(note)
        d.exec()

    def _show_circuit_diagram(self, model, params=None, param_names=None):
        """Draw a compact schematic block diagram for the selected equivalent circuit."""
        d = QDialog(self)
        d.setWindowTitle("Equivalent-circuit block diagram")
        d.resize(1050, 430)
        lay = QVBoxLayout(d)

        title = QLabel(f"<b>{model}</b>")
        title.setStyleSheet("font-size:15px;padding:6px;")
        lay.addWidget(title)

        fig = Figure(figsize=(10, 3.2), tight_layout=True)
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.set_axis_off()
        ax.set_xlim(0, 12); ax.set_ylim(0, 5)

        def wire(x1,y1,x2,y2):
            ax.plot([x1,x2],[y1,y2], color="black", lw=1.8)
        def box(x,y,w,h,text):
            from matplotlib.patches import FancyBboxPatch
            p=FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.05",
                             fill=False,edgecolor="black",linewidth=1.6)
            ax.add_patch(p); ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=10)
        def parallel_block(x, top, bottom, label_top, label_bottom, width=2.4):
            wire(x, bottom, x, top)
            wire(x+width, bottom, x+width, top)
            box(x+0.45, top-0.35, width-0.9, 0.7, label_top)
            box(x+0.45, bottom-0.35, width-0.9, 0.7, label_bottom)
            wire(x,top,x+0.45,top); wire(x+width-0.45,top,x+width,top)
            wire(x,bottom,x+0.45,bottom); wire(x+width-0.45,bottom,x+width,bottom)

        y=2.5
        wire(0.4,y,0.9,y); box(0.9,y-0.4,1.2,0.8,"Rs")
        if " + L + " in model:
            wire(2.1,y,2.35,y); box(2.35,y-0.4,1.2,0.8,"L"); wire(3.55,y,3.9,y)
            x=3.9
        else:
            wire(2.1,y,2.5,y)
            x=2.5

        if "Rfilm" in model:
            parallel_block(x,3.4,1.6,"Rfilm","CPEfilm"); x+=2.8
            wire(x-0.4,y,x,y)
            parallel_block(x,3.4,1.6,"Rct","CPEdl"); x+=2.8
        elif "R1" in model:
            lower1 = "C1" if "(R1 || C1)" in model else "CPE1"
            lower2 = "C2" if "(R2 || C2)" in model else "CPE2"
            parallel_block(x,3.4,1.6,"R1",lower1); x+=2.8
            wire(x-0.4,y,x,y)
            parallel_block(x,3.4,1.6,"R2",lower2); x+=2.8
        else:
            lower = "Cdl" if "Cdl" in model else "CPE"
            parallel_block(x,3.4,1.6,"Rct",lower); x+=2.8

        if "+ Wo" in model:
            wire(x-0.4,y,x,y); box(x,y-0.4,1.6,0.8,"Wo"); x+=1.9
        elif "+ Ws" in model:
            wire(x-0.4,y,x,y); box(x,y-0.4,1.6,0.8,"Ws"); x+=1.9
        elif "+ W" in model:
            wire(x-0.4,y,x,y); box(x,y-0.4,1.6,0.8,"W"); x+=1.9

        wire(x-0.4,y,min(x+0.5,11.5),y)
        ax.text(0.4,4.55,
                "R = resistance   |   L = inductance   |   C = ideal capacitor   |   CPE = constant phase element   |   "
                "W = semi-infinite Warburg   |   Wo/Ws = finite-length Warburg",
                fontsize=10, va="center")

        lay.addWidget(canvas)
        if params is not None and param_names is not None:
            txt_parts = []
            for n, v in zip(param_names, params):
                if " / " in n:
                    pname, unit = n.split(" / ", 1)
                    txt_parts.append(f"{pname} = {float(v):.4g} {unit}")
                else:
                    txt_parts.append(f"{n} = {float(v):.4g}")
            txt = "   ".join(txt_parts)
            lab=QLabel(txt); lab.setWordWrap(True)
            lab.setStyleSheet("padding:7px;background:#F4F8FC;border:1px solid #B7C8D8;")
            lay.addWidget(lab)
            caps = _effective_capacitance_rows(model, param_names, params)
            if caps:
                ctxt = "   |   ".join(f"{name} = {_format_capacitance(val)}" for name,val,unit,meaning in caps)
                clab=QLabel("<b>Capacitance:</b> " + ctxt)
                clab.setWordWrap(True)
                clab.setStyleSheet("padding:7px;background:#F3FAF3;border:1px solid #9BC79B;")
                lay.addWidget(clab)
        canvas.draw()
        d.exec()

    def kramers_kronig_check(self):
        """Practical linear Kramers-Kronig consistency check using a non-negative RC basis."""
        try:
            f, z = self._complex_dataset()
            mask = np.isfinite(f) & np.isfinite(z.real) & np.isfinite(z.imag) & (f > 0)
            f=np.asarray(f[mask],float); z=np.asarray(z[mask],complex)
            if len(f) < 8:
                raise ValueError("At least 8 valid frequency points are required.")

            order=np.argsort(f); f=f[order]; z=z[order]
            w=2*np.pi*f
            # Log-spaced relaxation-time basis spanning beyond measured frequency window.
            n_tau=max(12,min(50,len(f)))
            tau=np.logspace(np.log10(1/(2*np.pi*f.max()))-1,
                            np.log10(1/(2*np.pi*f.min()))+1,n_tau)
            basis=1.0/(1.0+1j*w[:,None]*tau[None,:])

            # Linear least squares: Rs + sum Rk/(1+j*w*tauk) + j*w*L.
            # This is a practical linear KK consistency representation.
            A_real=np.column_stack([np.ones(len(f)), basis.real, np.zeros(len(f))])
            A_imag=np.column_stack([np.zeros(len(f)), basis.imag, w])
            A=np.vstack([A_real,A_imag])
            b=np.concatenate([z.real,z.imag])
            coef, *_ = np.linalg.lstsq(A,b,rcond=None)
            zkk=coef[0] + basis.dot(coef[1:-1]) + 1j*w*coef[-1]

            scale=max(float(np.sqrt(np.mean(np.abs(z)**2))),1e-12)
            rms=float(np.sqrt(np.mean(np.abs(zkk-z)**2)))
            rel=100*rms/scale
            if rel <= 2:
                verdict="Good"
            elif rel <= 5:
                verdict="Acceptable"
            else:
                verdict="Questionable"

            d=QDialog(self); d.setWindowTitle("Kramers-Kronig consistency check"); d.resize(1300,820)
            lay=QVBoxLayout(d)
            info=QLabel(
                f"<b>Kramers-Kronig consistency: {verdict}</b> — relative complex RMS deviation = {rel:.2f} %.<br>"
                "This is a practical linear KK consistency check. Small, structureless residuals support "
                "causality/time-invariance consistency; systematic residuals can indicate drift, non-linearity, "
                "insufficient frequency range, or other measurement inconsistencies."
            )
            info.setWordWrap(True); info.setStyleSheet(
                "padding:9px;background:#F3FAF3;border:1px solid #9BC79B;font-weight:600;")
            lay.addWidget(info)

            fig=Figure(figsize=(11,6),tight_layout=True); canvas=FigureCanvas(fig)
            lay.addWidget(NavigationToolbar(canvas,d)); lay.addWidget(canvas,1)
            ax1=fig.add_subplot(221); ax2=fig.add_subplot(222)
            ax3=fig.add_subplot(223); ax4=fig.add_subplot(224)
            ax1.plot(z.real,-z.imag,"o",label="Measured"); ax1.plot(zkk.real,-zkk.imag,"-",label="KK reconstruction")
            ax1.set_xlabel("Z' / Ω"); ax1.set_ylabel("-Z'' / Ω"); ax1.set_title("Nyquist"); ax1.grid(alpha=.25); ax1.legend()
            ax2.semilogx(f,np.abs(z),"o",label="Measured"); ax2.semilogx(f,np.abs(zkk),"-",label="KK reconstruction")
            ax2.set_xlabel("Frequency / Hz"); ax2.set_ylabel("|Z| / Ω"); ax2.set_title("Bode magnitude"); ax2.grid(alpha=.25); ax2.legend()
            rr=zkk.real-z.real; ri=zkk.imag-z.imag
            ax3.semilogx(f,rr,"o-"); ax3.axhline(0,color="black",lw=1)
            ax3.set_xlabel("Frequency / Hz"); ax3.set_ylabel("ΔZ' / Ω"); ax3.set_title("Real residual"); ax3.grid(alpha=.25)
            ax4.semilogx(f,ri,"o-"); ax4.axhline(0,color="black",lw=1)
            ax4.set_xlabel("Frequency / Hz"); ax4.set_ylabel("ΔZ'' / Ω"); ax4.set_title("Imaginary residual"); ax4.grid(alpha=.25)
            canvas.draw(); d.exec()
            self.status.setText(f"Kramers-Kronig consistency: {verdict}; relative RMS deviation = {rel:.2f} %")
        except Exception as e:
            QMessageBox.critical(self,"Kramers-Kronig check",str(e))

    def fit_stability_analysis(self):
        """Repeated multi-start fitting to assess convergence and parameter identifiability."""
        try:
            f, z = self._complex_dataset()
            model = self.model.currentText()
            names = _parameter_names(model)
            p0 = self._estimate_initials(f, z)
            rng = np.random.default_rng(20260824)

            n_runs = 30
            runs = []
            for k in range(n_runs):
                # Log-normal perturbation keeps positive physical parameters positive.
                trial = np.asarray(p0, float).copy()
                for i, name in enumerate(names):
                    short = name.split(" / ")[0]
                    if short.startswith("α"):
                        trial[i] = np.clip(trial[i] + rng.normal(0, 0.08), 0.15, 1.0)
                    else:
                        trial[i] = max(trial[i] * np.exp(rng.normal(0, 0.85)), 1e-15)
                try:
                    r = self._fit_model_core(model, f, z, initial_override=trial)
                    if r.get("success") and np.all(np.isfinite(r["params"])) and np.isfinite(r["rmse"]):
                        runs.append(r)
                except Exception:
                    pass

            if len(runs) < 3:
                raise ValueError(
                    f"Only {len(runs)} of {n_runs} multi-start fits converged. "
                    "The fit is unstable or the selected circuit is poorly identifiable."
                )

            P=np.vstack([r["params"] for r in runs])
            rmse=np.array([r["rmse"] for r in runs],float)
            best_i=int(np.argmin(rmse))
            best=runs[best_i]

            med=np.median(P,axis=0)
            mean=np.mean(P,axis=0)
            sd=np.std(P,axis=0,ddof=1)
            cv=np.where(np.abs(mean)>1e-30,100*sd/np.abs(mean),np.nan)

            # Robust spread around the best basin: runs within 2% of best RMSE.
            good = rmse <= max(rmse.min()*1.02, rmse.min()+1e-12)
            Pg=P[good]
            if len(Pg)>=3:
                gmean=np.mean(Pg,axis=0); gsd=np.std(Pg,axis=0,ddof=1)
                gcv=np.where(np.abs(gmean)>1e-30,100*gsd/np.abs(gmean),np.nan)
            else:
                gcv=cv

            worst_cv=float(np.nanmax(gcv)) if np.any(np.isfinite(gcv)) else np.inf
            conv=100*len(runs)/n_runs
            basin=100*np.sum(good)/len(runs)

            # Separate numerical fit-quality stability from parameter identifiability.
            # This avoids calling a fit "unstable" when repeated fits reach the same
            # objective minimum but individual equivalent-circuit parameters vary.
            if conv >= 90 and basin >= 80:
                fit_quality = "stable"
            elif conv >= 70 and basin >= 50:
                fit_quality = "moderately stable"
            else:
                fit_quality = "unstable"

            if worst_cv < 10:
                identifiability = "good"
            elif worst_cv < 30:
                identifiability = "moderate"
            else:
                identifiability = "poor"

            verdict = f"Fit quality: {fit_quality} — parameter identifiability: {identifiability}"

            d=QDialog(self); d.setWindowTitle("Fit stability analysis"); d.resize(1280,820)
            lay=QVBoxLayout(d)
            info=QLabel(
                f"<b>{verdict}</b> — {len(runs)}/{n_runs} fits converged ({conv:.0f} %); "
                f"{np.sum(good)}/{len(runs)} converged fits lie within 2 % of the best RMSE ({basin:.0f} %).<br>"
                "The selected equivalent circuit is repeatedly fitted from randomly perturbed starting values. "
                "Fit-quality stability describes whether the same RMSE minimum is reached repeatedly; parameter "
                "identifiability describes whether essentially the same circuit parameters are obtained. "
                "A fit can therefore have stable fit quality but poor parameter identifiability."
            )
            info.setWordWrap(True)
            info.setStyleSheet("padding:9px;background:#F3FAF3;border:1px solid #9BC79B;font-weight:600;")
            lay.addWidget(info)

            fig=Figure(figsize=(10,4.4),tight_layout=True); canvas=FigureCanvas(fig)
            lay.addWidget(NavigationToolbar(canvas,d)); lay.addWidget(canvas,1)
            ax=fig.add_subplot(121)
            ax.plot(np.arange(1,len(rmse)+1),rmse,"o")
            ax.axhline(rmse.min(),ls="--",lw=1.2,label="Best RMSE")
            ax.set_xlabel("Converged multi-start fit")
            ax.set_ylabel("RMSE / Ω")
            ax.set_title("Fit-quality stability")
            ax.grid(alpha=.25); ax.legend()

            ax2=fig.add_subplot(122)
            finite_cv=np.where(np.isfinite(gcv),gcv,0)
            ax2.bar(np.arange(len(names)),finite_cv)
            ax2.set_xticks(np.arange(len(names)))
            ax2.set_xticklabels([n.split(" / ")[0] for n in names],rotation=45,ha="right")
            ax2.set_ylabel("CV within best basin / %")
            ax2.set_title("Parameter stability")
            ax2.axhline(10,ls="--",lw=1)
            ax2.axhline(30,ls=":",lw=1)
            ax2.grid(axis="y",alpha=.25)

            table=QTableWidget(len(names),6)
            table.setHorizontalHeaderLabels(["Parameter","Best fit","Mean","SD","CV / %","Assessment"])
            table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            for i,n in enumerate(names):
                short=n.split(" / ")[0]
                unit=n.split(" / ",1)[1] if " / " in n else ""
                assess = "Stable" if gcv[i] < 10 else ("Moderate" if gcv[i] < 30 else "Poorly identified")
                vals=[
                    short,
                    f"{best['params'][i]:.5E}" + (f" {unit}" if unit else ""),
                    f"{mean[i]:.5E}" + (f" {unit}" if unit else ""),
                    f"{sd[i]:.3E}" + (f" {unit}" if unit else ""),
                    f"{gcv[i]:.2f}" if np.isfinite(gcv[i]) else "",
                    assess
                ]
                for j,v in enumerate(vals):
                    table.setItem(i,j,QTableWidgetItem(v))
            table.setMaximumHeight(280)
            lay.addWidget(table)

            note=QLabel(
                "<b>Interpretation:</b> CV is the coefficient of variation of parameters among fits in the best-RMSE basin. "
                "As a practical diagnostic, CV < 10 % is labelled stable, 10–30 % moderate, and > 30 % poorly identified. "
                "These are diagnostic thresholds, not universal electrochemical laws. "
                "A low RMSE alone does not guarantee uniquely determined circuit parameters."
            )
            note.setWordWrap(True)
            note.setStyleSheet("padding:8px;background:#FFF9E8;border:1px solid #D8C786;")
            lay.addWidget(note)
            canvas.draw(); d.exec()
            self.status.setText(
                f"Fit stability: {verdict}; {len(runs)}/{n_runs} converged; "
                f"best RMSE={rmse.min():.4E} Ω."
            )
        except TypeError as e:
            QMessageBox.critical(
                self,"Fit stability analysis",
                "The internal fitting routine does not yet accept multi-start initial values. "
                "Please use the updated module file.\\n\\n"+str(e)
            )
        except Exception as e:
            QMessageBox.critical(self,"Fit stability analysis",str(e))

    def show_residuals(self):
        try:
            if self.last_fit is None:
                raise ValueError("Run an equivalent-circuit fit first.")

            f = self.last_fit["freq"]
            zm = self.last_fit["z_meas"]
            zf = self.last_fit["z_fit"]
            dr = zm.real - zf.real
            di = zm.imag - zf.imag
            order = np.argsort(f)

            d = QDialog(self)
            d.setWindowTitle("EIS fit residuals")
            d.resize(1050, 760)
            lay = QVBoxLayout(d)

            fig = Figure(figsize=(9, 6), tight_layout=True)
            canvas = FigureCanvas(fig)
            lay.addWidget(NavigationToolbar(canvas, d))
            lay.addWidget(canvas)

            ax1 = fig.add_subplot(211)
            ax2 = fig.add_subplot(212)
            ax1.semilogx(f[order], dr[order], "o-")
            ax1.axhline(0, lw=1)
            ax1.set_ylabel("Z' measured - fit / Ω")
            ax1.grid(alpha=.25)

            ax2.semilogx(f[order], di[order], "o-")
            ax2.axhline(0, lw=1)
            ax2.set_xlabel("Frequency / Hz")
            ax2.set_ylabel("Z'' measured - fit / Ω")
            ax2.grid(alpha=.25)

            canvas.draw()
            d.exec()

        except Exception as e:
            QMessageBox.critical(self, "Residuals", str(e))

    def export_excel(self):
        try:
            p, _ = QFileDialog.getSaveFileName(
                self, "Export EIS Excel", "EIS_Analysis.xlsx", "Excel (*.xlsx)"
            )
            if not p:
                return
            if not p.lower().endswith(".xlsx"):
                p += ".xlsx"

            with pd.ExcelWriter(p, engine="xlsxwriter") as writer:
                wb = writer.book

                if self.nyquist is not None:
                    f, zr, zi = self._nyquist_arrays()
                    dny = {"Z_real_ohm": zr, "Z_imag_ohm": zi, "minus_Z_imag_ohm": -zi}
                    if f is not None:
                        dny = {"Frequency_Hz": f, **dny}
                    ny_df = pd.DataFrame(dny)
                    ny_df.to_excel(writer, sheet_name="Nyquist", index=False)
                    ws = writer.sheets["Nyquist"]
                    n = len(ny_df)
                    xcol = 1 if f is not None else 0
                    ycol = xcol + 2
                    chart = wb.add_chart({"type":"scatter","subtype":"smooth_with_markers"})
                    chart.add_series({
                        "name":"Nyquist",
                        "categories":["Nyquist",1,xcol,n,xcol],
                        "values":["Nyquist",1,ycol,n,ycol],
                    })
                    chart.set_title({"name":"Nyquist plot"})
                    chart.set_x_axis({"name":"Z' / ohm"})
                    chart.set_y_axis({"name":"-Z'' / ohm"})
                    chart.set_legend({"none":True})
                    ws.insert_chart("F2", chart, {"x_scale":1.25,"y_scale":1.15})

                if self.bode is not None:
                    f, mag, phase = self._bode_arrays()
                    if mag is None:
                        # Bode export contains frequency + phase only; keep phase on its native grid.
                        bd = pd.DataFrame({"Frequency_Hz": f, "Phase_deg": phase})
                    else:
                        bd = pd.DataFrame({"Frequency_Hz": f, "Z_magnitude_ohm": mag, "Phase_deg": phase})
                    bd.to_excel(writer, sheet_name="Bode", index=False)
                    ws = writer.sheets["Bode"]
                    n = len(bd)
                    if mag is not None:
                        ch1 = wb.add_chart({"type":"scatter","subtype":"smooth_with_markers"})
                        ch1.add_series({
                            "name":"|Z|",
                            "categories":["Bode",1,0,n,0],
                            "values":["Bode",1,1,n,1],
                        })
                        ch1.set_title({"name":"Bode magnitude"})
                        ch1.set_x_axis({"name":"Frequency / Hz","log_base":10})
                        ch1.set_y_axis({"name":"|Z| / ohm"})
                        ch1.set_legend({"none":True})
                        ws.insert_chart("E2", ch1, {"x_scale":1.25,"y_scale":1.15})

                    ch2 = wb.add_chart({"type":"scatter","subtype":"smooth_with_markers"})
                    ch2.add_series({
                        "name":"Phase",
                        "categories":["Bode",1,0,n,0],
                        "values":["Bode",1,(2 if mag is not None else 1),n,(2 if mag is not None else 1)],
                    })
                    ch2.set_title({"name":"Bode phase"})
                    ch2.set_x_axis({"name":"Frequency / Hz","log_base":10})
                    ch2.set_y_axis({"name":"Phase / deg"})
                    ch2.set_legend({"none":True})
                    ws.insert_chart("E20", ch2, {"x_scale":1.25,"y_scale":1.15})

                if hasattr(self, "model_comparison"):
                    rows = []
                    for r in sorted(
                        self.model_comparison,
                        key=lambda x: x["aicc"] if np.isfinite(x["aicc"]) else np.inf
                    ):
                        rows.append({
                            "Model": r["model"],
                            "Number_of_parameters": len(r["params"]),
                            "RMSE_ohm": r["rmse"],
                            "RSS": r["rss"],
                            "AIC": r["aic"],
                            "AICc": r["aicc"],
                            "Delta_AICc": r.get("delta_aicc", np.nan),
                            "BIC": r["bic"],
                            "Success": r["success"],
                        })
                    cmp_df = pd.DataFrame(rows)
                    cmp_df.to_excel(writer, sheet_name="Model comparison", index=False)
                    ws_cmp = writer.sheets["Model comparison"]
                    if len(cmp_df):
                        chart = wb.add_chart({"type":"column"})
                        chart.add_series({
                            "name":"ΔAICc",
                            "categories":["Model comparison",1,0,len(cmp_df),0],
                            "values":["Model comparison",1,6,len(cmp_df),6],
                        })
                        chart.set_title({"name":"Equivalent-circuit model comparison"})
                        chart.set_x_axis({"name":"Circuit model"})
                        chart.set_y_axis({"name":"ΔAICc"})
                        chart.set_legend({"none":True})
                        ws_cmp.insert_chart("K2", chart, {"x_scale":1.4,"y_scale":1.2})

                if self.last_fit is not None:
                    lf = self.last_fit
                    f = lf["freq"]
                    zm = lf["z_meas"]
                    zf = lf["z_fit"]
                    fit_df = pd.DataFrame({
                        "Frequency_Hz": f,
                        "Zreal_measured_ohm": zm.real,
                        "Zimag_measured_ohm": zm.imag,
                        "minus_Zimag_measured_ohm": -zm.imag,
                        "Zreal_fit_ohm": zf.real,
                        "Zimag_fit_ohm": zf.imag,
                        "minus_Zimag_fit_ohm": -zf.imag,
                        "Zmag_measured_ohm": np.abs(zm),
                        "Zmag_fit_ohm": np.abs(zf),
                        "Phase_measured_deg": _phase_deg(zm),
                        "Phase_fit_deg": _phase_deg(zf),
                        "Residual_Zreal_ohm": zm.real-zf.real,
                        "Residual_Zimag_ohm": zm.imag-zf.imag,
                    })
                    fit_df.to_excel(writer, sheet_name="Fit", index=False)
                    ws = writer.sheets["Fit"]
                    n = len(fit_df)

                    params = pd.DataFrame({
                        "Parameter": lf["param_names"],
                        "Meaning": [_parameter_meaning(x) for x in lf["param_names"]],
                        "Fit_value": lf["params"],
                    })
                    cap_rows = _effective_capacitance_rows(
                        lf["model"], lf["param_names"], lf["params"]
                    )
                    if cap_rows:
                        extra = pd.DataFrame({
                            "Parameter": [x[0] for x in cap_rows],
                            "Meaning": [x[3] for x in cap_rows],
                            "Fit_value": [x[1] for x in cap_rows],
                        })
                        params = pd.concat([params, extra], ignore_index=True)
                    params.to_excel(writer, sheet_name="Fit parameters", index=False)
                    wsp = writer.sheets["Fit parameters"]
                    wsp.write("D2","Model")
                    wsp.write("E2",lf["model"])
                    wsp.write("D3","RMSE / ohm")
                    wsp.write_number("E3",float(lf["rmse"]))
                    wsp.write("D4","AIC")
                    wsp.write_number("E4",float(lf["aic"]))

                    ny = wb.add_chart({"type":"scatter","subtype":"smooth_with_markers"})
                    ny.add_series({
                        "name":"Measured",
                        "categories":["Fit",1,1,n,1],
                        "values":["Fit",1,3,n,3],
                    })
                    ny.add_series({
                        "name":"Fit",
                        "categories":["Fit",1,4,n,4],
                        "values":["Fit",1,6,n,6],
                    })
                    ny.set_title({"name":"Nyquist: measured vs fit"})
                    ny.set_x_axis({"name":"Z' / ohm"})
                    ny.set_y_axis({"name":"-Z'' / ohm"})
                    ws.insert_chart("O2",ny,{"x_scale":1.25,"y_scale":1.15})

                    bm = wb.add_chart({"type":"scatter","subtype":"smooth_with_markers"})
                    bm.add_series({
                        "name":"Measured",
                        "categories":["Fit",1,0,n,0],
                        "values":["Fit",1,7,n,7],
                    })
                    bm.add_series({
                        "name":"Fit",
                        "categories":["Fit",1,0,n,0],
                        "values":["Fit",1,8,n,8],
                    })
                    bm.set_title({"name":"Bode magnitude: measured vs fit"})
                    bm.set_x_axis({"name":"Frequency / Hz","log_base":10})
                    bm.set_y_axis({"name":"|Z| / ohm"})
                    ws.insert_chart("O20",bm,{"x_scale":1.25,"y_scale":1.15})

                    bp = wb.add_chart({"type":"scatter","subtype":"smooth_with_markers"})
                    bp.add_series({
                        "name":"Measured",
                        "categories":["Fit",1,0,n,0],
                        "values":["Fit",1,9,n,9],
                    })
                    bp.add_series({
                        "name":"Fit",
                        "categories":["Fit",1,0,n,0],
                        "values":["Fit",1,10,n,10],
                    })
                    bp.set_title({"name":"Bode phase: measured vs fit"})
                    bp.set_x_axis({"name":"Frequency / Hz","log_base":10})
                    bp.set_y_axis({"name":"Phase / deg"})
                    ws.insert_chart("O38",bp,{"x_scale":1.25,"y_scale":1.15})

            self.status.setText(f"Excel exported: {p}")

        except Exception as e:
            QMessageBox.critical(self, "Excel export", str(e))


def main():
    app = QApplication.instance() or QApplication([])
    w = EISWindow()
    w.show()
    app.exec()


if __name__ == "__main__":
    main()

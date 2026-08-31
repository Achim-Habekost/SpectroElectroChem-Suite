from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import linregress

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QFileDialog, QMessageBox, QDoubleSpinBox, QSpinBox,
    QGroupBox, QCheckBox, QComboBox, QLineEdit, QTableWidget, QTableWidgetItem,
    QHeaderView, QDialog, QScrollArea
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector

F = 96485.33212
R = 8.314462618


def _read_numeric_csv(path):
    # Robust import for comma/semicolon/tab files and decimal comma/dot.
    last = None
    for sep in [None, ";", ",", "\t"]:
        try:
            df = pd.read_csv(path, sep=sep, engine="python", header=None, comment="#")
            df = df.apply(lambda col: pd.to_numeric(
                col.astype(str).str.strip().str.replace(",", ".", regex=False),
                errors="coerce"))
            df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
            if df.shape[0] >= 3 and df.shape[1] >= 2:
                return df.reset_index(drop=True)
        except Exception as e:
            last = e
    raise ValueError(f"Could not read numeric CSV: {last}")


def _smooth(y, enabled, window, poly):
    y = np.asarray(y, dtype=float)
    if not enabled or len(y) < 5:
        return y
    w = int(window)
    if w % 2 == 0:
        w += 1
    w = min(w, len(y) if len(y) % 2 else len(y)-1)
    p = min(int(poly), max(1, w-1))
    if w <= p or w < 3:
        return y
    return savgol_filter(y, w, p)


def _linfit(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[m], np.asarray(y)[m]
    if len(x) < 2:
        raise ValueError("At least two valid points are required.")
    r = linregress(x, y)
    return r.slope, r.intercept, r.rvalue**2



def _pick_range(x, y, title, xlabel, ylabel):
    import matplotlib.pyplot as plt
    x=np.asarray(x,float); y=np.asarray(y,float)
    ok=np.isfinite(x)&np.isfinite(y); x=x[ok]; y=y[ok]
    fig,ax=plt.subplots(figsize=(9,6)); ax.plot(x,y)
    ax.set_title(title+" — click two limits"); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.grid(True,alpha=.25)
    pts=plt.ginput(2,timeout=-1); plt.close(fig)
    if len(pts)!=2: return None
    return tuple(sorted((float(pts[0][0]),float(pts[1][0]))))

class DiffusionWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # Interactive analysis ranges; None = full range / automatic peak search.
        self.cottrell_fit_range = None
        self.anson_fit_range = None
        self.rs_anodic_range = None
        self.rs_cathodic_range = None
        self.rs_manual_peaks = []
        self.rs_peak_mode_by_scan = []

        # Interactive fit/peak ranges. None means use the full available range.
        self.setWindowTitle("Diffusion Coefficient Analysis – Cottrell | Anson | Randles–Ševčík")
        self.resize(1320, 900)
        self.ca = self.cc = self.cv = None
        self.ca_path = self.cc_path = self.cv_path = None
        self._build()

    def _build(self):
        c = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(c)
        self.setCentralWidget(scroll)
        outer = QVBoxLayout(c)
        outer.setSpacing(10)

        self.setStyleSheet("""
            QGroupBox {
                font-weight: 700;
                color: #17365D;
                border: 1px solid #B8C7D9;
                border-radius: 7px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QPushButton { padding: 6px 10px; border-radius: 5px; background: #2F75B5; color: white; font-weight: 600; }
            QPushButton:hover { background: #3F86C6; }
            QPushButton#inputButton { background:#2F75B5; }
            QPushButton#cottrellButton { background:#2E8B57; }
            QPushButton#ansonButton { background:#D9822B; }
            QPushButton#randlesButton { background:#7A5CC7; }
            QPushButton#advancedButton { background:#3F7F8F; }
            QPushButton#exportButton { background:#58636F; }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox { background: white; border: 1px solid #BFC7D0; border-radius: 4px; padding: 3px 5px; }
            QHeaderView::section { background: #DCEAF7; color: #17365D; font-weight: 700; padding: 5px; border: 1px solid #C3D5E6; }
        """)

        title = QLabel("Diffusion Coefficient Analysis – Cottrell | Anson | Randles–Ševčík")
        title.setStyleSheet("font-size:20px;font-weight:700;color:#17365D;")
        outer.addWidget(title)
        unit_note = QLabel("All calculations are performed internally in A, C, mol cm⁻³, cm², V s⁻¹ and K.")
        unit_note.setStyleSheet("color:#4A5568;")
        outer.addWidget(unit_note)

        self.ca_current_unit = QComboBox()
        self.ca_current_unit.addItems(["A", "mA", "µA", "nA"])
        self.ca_current_unit.setCurrentText("µA")
        self.cc_charge_unit = QComboBox()
        self.cc_charge_unit.addItems(["C", "mC", "µC", "nC"])
        self.cc_charge_unit.setCurrentText("µC")
        self.cv_current_unit = QComboBox()
        self.cv_current_unit.addItems(["A", "mA", "µA", "nA"])
        self.cv_current_unit.setCurrentText("µA")

        files = QGroupBox("1. Input data")
        files.setStyleSheet("QGroupBox { background:#F5FAFF; }")
        g = QGridLayout(files)
        self.ca_lab = QLabel("No file")
        self.cc_lab = QLabel("No file")
        self.cv_lab = QLabel("No file")
        for row, (txt, lab, fn) in enumerate([
            ("Chronoamperometry CSV", self.ca_lab, self.load_ca),
            ("Chronocoulometry / charge chronometry CSV", self.cc_lab, self.load_cc),
            ("Cyclic voltammetry CSV (multiple scan rates)", self.cv_lab, self.load_cv),
        ]):
            b=QPushButton(f"Load {txt}"); b.setObjectName("inputButton"); b.clicked.connect(fn)
            g.addWidget(b,row,0); g.addWidget(lab,row,1)

        # Unit/power selectors belong directly to the corresponding input file.
        g.addWidget(QLabel("Current unit"), 0, 2)
        g.addWidget(self.ca_current_unit, 0, 3)
        g.addWidget(QLabel("Charge unit"), 1, 2)
        g.addWidget(self.cc_charge_unit, 1, 3)
        g.addWidget(QLabel("Current unit"), 2, 2)
        g.addWidget(self.cv_current_unit, 2, 3)
        outer.addWidget(files)

        par = QGroupBox("2. Electrochemical parameters")
        par.setStyleSheet("QGroupBox { background:#F7FBF4; }")
        pg = QGridLayout(par)
        self.conc = QLineEdit("1e-3")
        self.z = QSpinBox(); self.z.setRange(1,20); self.z.setValue(1)
        self.area = QDoubleSpinBox(); self.area.setDecimals(6); self.area.setRange(1e-6,1e5); self.area.setValue(0.785)
        self.temp = QDoubleSpinBox(); self.temp.setDecimals(2); self.temp.setRange(1,2000); self.temp.setValue(298.15)

        

        params = [
            ("Concentration / mol L⁻¹", self.conc),
            ("z (number of electrons)", self.z),
            ("Electrode area A / cm²", self.area),
            ("Temperature / K", self.temp),
        ]
        for i, (lab, w) in enumerate(params):
            row, col = divmod(i, 3)
            pg.addWidget(QLabel(lab), row, col*2)
            pg.addWidget(w, row, col*2+1)
        outer.addWidget(par)

        sm = QGroupBox("3. Independent smoothing (Savitzky–Golay)")
        sm.setStyleSheet("QGroupBox { background:#FFF9F0; }")
        sg=QGridLayout(sm)
        self.smooth_controls={}
        for row,name in enumerate(["Chronoamperometry","Chronocoulometry","CV"]):
            en=QCheckBox("Smooth"); win=QSpinBox(); win.setRange(3,999); win.setSingleStep(2); win.setValue(11)
            pol=QComboBox(); pol.addItems(["2", "3", "4", "5"]); pol.setCurrentText("2")
            sg.addWidget(QLabel(name),row,0); sg.addWidget(en,row,1); sg.addWidget(QLabel("Window"),row,2)
            sg.addWidget(win,row,3); sg.addWidget(QLabel("Polynomial order"),row,4); sg.addWidget(pol,row,5)
            self.smooth_controls[name]=(en,win,pol)
        outer.addWidget(sm)

        cvbox=QGroupBox("4. Randles–Ševčík scan rates")
        cvbox.setStyleSheet("QGroupBox { background:#F8F4FF; }")
        cvg=QVBoxLayout(cvbox)
        scan_row = QHBoxLayout()
        scan_row.addWidget(QLabel("Number of CV scans, n"))
        self.n_scans = QSpinBox(); self.n_scans.setRange(1,100); self.n_scans.setValue(5)
        self.n_scans.valueChanged.connect(self._rebuild_scan_rate_rows)
        scan_row.addWidget(self.n_scans)
        scan_row.addSpacing(20)
        scan_row.addWidget(QLabel("Enter one scan rate for each CV current trace."))
        scan_row.addStretch(1)
        cvg.addLayout(scan_row)
        self.rate_table=QTableWidget(0,2)
        self.rate_table.setHorizontalHeaderLabels(["CV trace","Scan rate / V s⁻¹"])
        self.rate_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.rate_table.setMinimumHeight(150)
        self.rate_table.setMaximumHeight(230)
        cvg.addWidget(self.rate_table)
        outer.addWidget(cvbox)
        self._rebuild_scan_rate_rows()

        actions=QHBoxLayout()
        for txt,fn in [("Cottrell analysis",self.cottrell),("Anson analysis",self.anson),
                       ("Randles–Ševčík analysis",self.randles),
                       ("Transport mechanism",self.transport_mechanism),
                       ("Nicholson analysis",self.nicholson),
                       ("Laviron analysis",self.laviron),
                       ("Summary",self.summary),("Export Excel",self.export_excel)]:
            b=QPushButton(txt)
            if txt == "Cottrell analysis": b.setObjectName("cottrellButton")
            elif txt == "Anson analysis": b.setObjectName("ansonButton")
            elif txt == "Randles–Ševčík analysis": b.setObjectName("randlesButton")
            elif txt == "Export Excel": b.setObjectName("exportButton")
            else: b.setObjectName("advancedButton")
            b.clicked.connect(fn); actions.addWidget(b)
        outer.addLayout(actions)

        self.status=QLabel("Ready.")
        self.status.setStyleSheet("padding:8px;background:#F3F6F9;border:1px solid #CCD6E0;")
        outer.addWidget(self.status)

    def _params(self):
        try:
            c=float(self.conc.text().strip().replace(",","."))
        except Exception:
            raise ValueError("Invalid concentration.")
        # mol/L -> mol/cm3
        return c/1000.0, int(self.z.value()), float(self.area.value()), float(self.temp.value())

    def load_ca(self):
        p,_=QFileDialog.getOpenFileName(self,"Load chronoamperometry CSV","","CSV/TXT (*.csv *.txt);;All files (*)")
        if p:
            self.ca=_read_numeric_csv(p); self.ca_path=p; self.ca_lab.setText(Path(p).name)
            self.status.setText(f"Chronoamperometry loaded: {self.ca.shape[0]} rows, {self.ca.shape[1]} columns.")

    def load_cc(self):
        p,_=QFileDialog.getOpenFileName(self,"Load chronocoulometry CSV","","CSV/TXT (*.csv *.txt);;All files (*)")
        if p:
            self.cc=_read_numeric_csv(p); self.cc_path=p; self.cc_lab.setText(Path(p).name)
            self.status.setText(f"Chronocoulometry loaded: {self.cc.shape[0]} rows, {self.cc.shape[1]} columns.")

    def _rebuild_scan_rate_rows(self, *args):
        n = int(self.n_scans.value())

        # Preserve existing entries by row index.
        old_values = []
        for r in range(self.rate_table.rowCount()):
            w = self.rate_table.cellWidget(r, 1)
            if isinstance(w, QLineEdit):
                old_values.append(w.text().strip())
            else:
                item = self.rate_table.item(r, 1)
                old_values.append(item.text().strip() if item is not None else "")

        self.rate_table.setRowCount(0)
        self.rate_table.setRowCount(n)

        for i in range(n):
            self.rate_table.setItem(i, 0, QTableWidgetItem(f"CV {i+1}"))
            editor = QLineEdit()
            editor.setPlaceholderText("e.g. 0.02")
            if i < len(old_values) and old_values[i]:
                editor.setText(old_values[i])
            self.rate_table.setCellWidget(i, 1, editor)

            # Keyboard-friendly scan-rate entry:
            # Return/Enter accepts the current value and moves to the next row.
            def _advance_to_next_row(row=i):
                if row < self.rate_table.rowCount() - 1:
                    next_editor = self.rate_table.cellWidget(row + 1, 1)
                    if isinstance(next_editor, QLineEdit):
                        next_editor.setFocus()
                        next_editor.selectAll()

            editor.returnPressed.connect(_advance_to_next_row)

    def load_cv(self):
        p,_=QFileDialog.getOpenFileName(self,"Load CV CSV","","CSV/TXT (*.csv *.txt);;All files (*)")
        if p:
            self.cv=_read_numeric_csv(p); self.cv_path=p; self.cv_lab.setText(Path(p).name)
            n=max(1,self.cv.shape[1]-1)
            self.n_scans.setValue(n)
            self.rs_manual_peaks = [dict(epa=np.nan, ipa=np.nan, epc=np.nan, ipc=np.nan) for _ in range(n)]
            self.rs_peak_mode_by_scan = ["Both peaks"] * n
            self._rebuild_scan_rate_rows()
            self.status.setText(f"CV loaded: {n} current trace(s). Enter {n} scan rate(s).")

    def _ca_current_factor(self):
        return {"A": 1.0, "mA": 1e-3, "µA": 1e-6, "nA": 1e-9}[self.ca_current_unit.currentText()]

    def _cv_current_factor(self):
        return {"A": 1.0, "mA": 1e-3, "µA": 1e-6, "nA": 1e-9}[self.cv_current_unit.currentText()]

    def _charge_factor(self):
        return {"C": 1.0, "mC": 1e-3, "µC": 1e-6, "nC": 1e-9}[self.cc_charge_unit.currentText()]

    def _ca_arrays(self):
        if self.ca is None:
            raise ValueError("Load chronoamperometry CSV first.")
        t = self.ca.iloc[:, 0].to_numpy(float)
        i = self.ca.iloc[:, 1].to_numpy(float) * self._ca_current_factor()
        enabled, window_widget, poly_widget = self.smooth_controls["Chronoamperometry"]
        poly_order = int(poly_widget.currentText())
        return t, _smooth(i, enabled.isChecked(), window_widget.value(), poly_order)

    def _cc_arrays(self):
        if self.cc is None:
            raise ValueError("Load chronocoulometry CSV first.")
        t = self.cc.iloc[:, 0].to_numpy(float)
        q = self.cc.iloc[:, 1].to_numpy(float) * self._charge_factor()
        enabled, window_widget, poly_widget = self.smooth_controls["Chronocoulometry"]
        poly_order = int(poly_widget.currentText())
        return t, _smooth(q, enabled.isChecked(), window_widget.value(), poly_order)

    def _dialog_plot(self,title,plots):
        d=QDialog(self); d.setWindowTitle(title); d.resize(1050,760)
        l=QVBoxLayout(d); fig=Figure(figsize=(9,6),tight_layout=True); canvas=FigureCanvas(fig)
        l.addWidget(NavigationToolbar(canvas,d)); l.addWidget(canvas)
        ax=fig.add_subplot(111)
        plots(ax)
        canvas.draw(); d.exec()

    def select_cottrell_range(self):
        try:
            t,i=self._ca_arrays(); ok=np.isfinite(t)&(t>0)&np.isfinite(i)
            self.cottrell_fit_range=_pick_range(1/np.sqrt(t[ok]),i[ok],"Cottrell fit range","t^-1/2 / s^-1/2","Current / A")
        except Exception as e: QMessageBox.critical(self,"Cottrell fit range",str(e))

    def select_anson_range(self):
        try:
            t,q=self._cc_arrays(); ok=np.isfinite(t)&(t>=0)&np.isfinite(q)
            self.anson_fit_range=_pick_range(np.sqrt(t[ok]),q[ok],"Anson fit range","t^1/2 / s^1/2","Charge / C")
        except Exception as e: QMessageBox.critical(self,"Anson fit range",str(e))

    def _pick_rs(self,title):
        if self.cv is None: raise ValueError("Load CV CSV first.")
        E=self.cv.iloc[:,0].to_numpy(float); en,w,pol=self.smooth_controls["CV"]; po=int(pol.currentText())
        ys=[_smooth(self.cv.iloc[:,j].to_numpy(float)*self._cv_current_factor(),en.isChecked(),w.value(),po) for j in range(1,self.cv.shape[1])]
        return _pick_range(E,np.nanmean(np.vstack(ys),axis=0),title,"Potential / V","Current / A")

    def select_rs_anodic_range(self):
        try: self.rs_anodic_range=self._pick_rs("RS anodic peak range")
        except Exception as e: QMessageBox.critical(self,"RS anodic peak range",str(e))

    def select_rs_cathodic_range(self):
        try: self.rs_cathodic_range=self._pick_rs("RS cathodic peak range")
        except Exception as e: QMessageBox.critical(self,"RS cathodic peak range",str(e))

    def cottrell(self):
        try:
            t,i=self._ca_arrays(); c,z,A,T=self._params()
            good=(t>0)&np.isfinite(t)&np.isfinite(i); x=1/np.sqrt(t[good]); y=i[good]
            d=QDialog(self); d.setWindowTitle("Cottrell analysis — select linear fit range"); d.resize(1100,800)
            lay=QVBoxLayout(d); info=QLabel("Drag horizontally across the desired linear region. Afterwards drag either edge, or drag the whole selected region, to fine-tune it. D and R² update after each move.")
            info.setStyleSheet("font-weight:600;padding:6px;background:#EEF6FF;border:1px solid #9CC4E4;"); lay.addWidget(info)
            fig=Figure(figsize=(9,6),tight_layout=True); canvas=FigureCanvas(fig); lay.addWidget(NavigationToolbar(canvas,d)); lay.addWidget(canvas)
            ax=fig.add_subplot(111); ax.scatter(x,y,s=12,label="Data"); ax.set_xlabel("t⁻¹ᐟ² / s⁻¹ᐟ²"); ax.set_ylabel("Current / A"); ax.grid(alpha=.25)
            fitline,=ax.plot([],[],lw=2,label="Linear fit"); shade=[None]
            def apply_range(lo,hi):
                lo,hi=sorted((float(lo),float(hi))); mask=(x>=lo)&(x<=hi)
                if mask.sum()<3:return
                slope,inter,r2=_linfit(x[mask],y[mask]); D=np.pi*(abs(slope)/(z*F*A*c))**2
                self.cottrell_fit_range=(lo,hi); self.last_cottrell=(D,r2,slope,inter)
                xx=np.linspace(lo,hi,300); fitline.set_data(xx,slope*xx+inter); fitline.set_label(f"Selected-range fit, R²={r2:.5f}")
                if shade[0] is not None: shade[0].remove()
                shade[0]=ax.axvspan(lo,hi,alpha=.12)
                ax.set_title(f"Cottrell analysis   D = {D:.4E} cm² s⁻¹   R² = {r2:.5f}"); ax.legend(); canvas.draw_idle()
                self.status.setText(f"Cottrell: D = {D:.4E} cm² s⁻¹; R² = {r2:.5f}; selected range {lo:.4g}–{hi:.4g}")
            selector=SpanSelector(ax,apply_range,'horizontal',useblit=True,interactive=True,drag_from_anywhere=True,props=dict(alpha=.18),handle_props=dict(alpha=.9))
            lo,hi=self.cottrell_fit_range if self.cottrell_fit_range else (float(np.min(x)),float(np.max(x))); selector.extents=(lo,hi); apply_range(lo,hi)
            row=QHBoxLayout(); full=QPushButton("Use full range"); full.clicked.connect(lambda: apply_range(float(np.min(x)),float(np.max(x)))); close=QPushButton("Close / keep selected range"); close.clicked.connect(d.accept); row.addWidget(full); row.addStretch(1); row.addWidget(close); lay.addLayout(row)
            d._selector=selector; d.exec()
        except Exception as e: QMessageBox.critical(self,"Cottrell",str(e))

    def anson(self):
        try:
            t,q=self._cc_arrays(); c,z,A,T=self._params()
            good=(t>=0)&np.isfinite(t)&np.isfinite(q); x=np.sqrt(t[good]); y=q[good]
            d=QDialog(self); d.setWindowTitle("Anson analysis — select linear fit range"); d.resize(1100,800)
            lay=QVBoxLayout(d); info=QLabel("Drag horizontally across the desired linear region. Afterwards drag either edge, or drag the whole selected region, to fine-tune it. D, R² and Q₀ update after each move.")
            info.setStyleSheet("font-weight:600;padding:6px;background:#FFF4E5;border:1px solid #E7B96B;"); lay.addWidget(info)
            fig=Figure(figsize=(9,6),tight_layout=True); canvas=FigureCanvas(fig); lay.addWidget(NavigationToolbar(canvas,d)); lay.addWidget(canvas)
            ax=fig.add_subplot(111); ax.scatter(x,y,s=12,label="Data"); ax.set_xlabel("t¹ᐟ² / s¹ᐟ²"); ax.set_ylabel("Charge / C"); ax.grid(alpha=.25)
            fitline,=ax.plot([],[],lw=2,label="Linear fit"); shade=[None]
            def apply_range(lo,hi):
                lo,hi=sorted((float(lo),float(hi))); mask=(x>=lo)&(x<=hi)
                if mask.sum()<3:return
                slope,inter,r2=_linfit(x[mask],y[mask]); D=np.pi*(abs(slope)/(2*z*F*A*c))**2
                self.anson_fit_range=(lo,hi); self.last_anson=(D,r2,slope,inter)
                xx=np.linspace(lo,hi,300); fitline.set_data(xx,slope*xx+inter); fitline.set_label(f"Selected-range fit, R²={r2:.5f}")
                if shade[0] is not None: shade[0].remove()
                shade[0]=ax.axvspan(lo,hi,alpha=.12)
                ax.set_title(f"Anson analysis   D = {D:.4E} cm² s⁻¹   R² = {r2:.5f}   Q₀ = {inter:.4E} C"); ax.legend(); canvas.draw_idle()
                self.status.setText(f"Anson: D = {D:.4E} cm² s⁻¹; R² = {r2:.5f}; Q₀ = {inter:.4E} C; selected range {lo:.4g}–{hi:.4g}")
            selector=SpanSelector(ax,apply_range,'horizontal',useblit=True,interactive=True,drag_from_anywhere=True,props=dict(alpha=.18),handle_props=dict(alpha=.9))
            lo,hi=self.anson_fit_range if self.anson_fit_range else (float(np.min(x)),float(np.max(x))); selector.extents=(lo,hi); apply_range(lo,hi)
            row=QHBoxLayout(); full=QPushButton("Use full range"); full.clicked.connect(lambda: apply_range(float(np.min(x)),float(np.max(x)))); close=QPushButton("Close / keep selected range"); close.clicked.connect(d.accept); row.addWidget(full); row.addStretch(1); row.addWidget(close); lay.addLayout(row)
            d._selector=selector; d.exec()
        except Exception as e: QMessageBox.critical(self,"Anson",str(e))

    def _rates(self):
        values = []
        for r in range(self.rate_table.rowCount()):
            widget = self.rate_table.cellWidget(r, 1)
            if isinstance(widget, QLineEdit):
                text = widget.text().strip()
            else:
                item = self.rate_table.item(r, 1)
                text = item.text().strip() if item is not None else ""

            text = text.replace(",", ".")
            if not text:
                raise ValueError(
                    f"Scan rate in row {r+1} is empty. "
                    "Please enter one scan rate for every CV trace."
                )
            try:
                value = float(text)
            except ValueError:
                raise ValueError(
                    f"Invalid scan rate in row {r+1}: '{text}'. "
                    "Use values such as 0.02, 0.05 or 0.1 V s⁻¹."
                )
            if value <= 0:
                raise ValueError(
                    f"Scan rate in row {r+1} must be greater than zero."
                )
            values.append(value)

        return np.asarray(values, dtype=float)

    def _cv_peaks(self):
        if self.cv is None:
            raise ValueError("Load CV CSV first.")

        rates = self._rates()
        ntraces = self.cv.shape[1] - 1
        if len(rates) != ntraces:
            raise ValueError(
                f"Number of scan rates ({len(rates)}) does not match number of CV current traces ({ntraces})."
            )

        if len(getattr(self, "rs_manual_peaks", [])) == ntraces:
            epa = np.array([p.get("epa", np.nan) for p in self.rs_manual_peaks], float)
            ipa = np.array([p.get("ipa", np.nan) for p in self.rs_manual_peaks], float)
            epc = np.array([p.get("epc", np.nan) for p in self.rs_manual_peaks], float)
            ipc = np.array([p.get("ipc", np.nan) for p in self.rs_manual_peaks], float)

            # If the user has manually selected at least one peak, preserve the manual dataset.
            if np.any(np.isfinite(ipa)) or np.any(np.isfinite(ipc)):
                return rates, ipa, ipc, epa, epc

        # Fallback automatic max/min only before any manual selection exists.
        E = self.cv.iloc[:, 0].to_numpy(float)
        enabled, window_widget, poly_widget = self.smooth_controls["CV"]
        poly_order = int(poly_widget.currentText())

        ipa, ipc, epa, epc = [], [], [], []
        for j in range(1, self.cv.shape[1]):
            raw_current = self.cv.iloc[:, j].to_numpy(float) * self._cv_current_factor()
            y = _smooth(raw_current, enabled.isChecked(), window_widget.value(), poly_order)
            good = np.isfinite(E) & np.isfinite(y)
            ee, yy = E[good], y[good]
            if len(yy) == 0:
                raise ValueError(f"CV trace {j} contains no valid numeric data.")
            ia = int(np.nanargmax(yy))
            ic = int(np.nanargmin(yy))
            ipa.append(yy[ia]); ipc.append(yy[ic])
            epa.append(ee[ia]); epc.append(ee[ic])

        return rates, np.asarray(ipa), np.asarray(ipc), np.asarray(epa), np.asarray(epc)

    def randles(self):
        try:
            if self.cv is None:
                raise ValueError("Load CV CSV first.")

            c, z, A, T = self._params()
            rates = self._rates()
            E = self.cv.iloc[:, 0].to_numpy(float)
            ntraces = self.cv.shape[1] - 1

            if len(rates) != ntraces:
                raise ValueError(
                    f"Number of scan rates ({len(rates)}) does not match number of CV traces ({ntraces})."
                )

            enabled, window_widget, poly_widget = self.smooth_controls["CV"]
            poly_order = int(poly_widget.currentText())
            curves = []
            for j in range(1, self.cv.shape[1]):
                raw = self.cv.iloc[:, j].to_numpy(float) * self._cv_current_factor()
                curves.append(_smooth(raw, enabled.isChecked(), window_widget.value(), poly_order))
            curves = np.asarray(curves, dtype=float)

            # Persistent per-scan peak storage.
            if len(getattr(self, "rs_manual_peaks", [])) != ntraces:
                self.rs_manual_peaks = [
                    dict(epa=np.nan, ipa=np.nan, epc=np.nan, ipc=np.nan)
                    for _ in range(ntraces)
                ]
            if len(getattr(self, "rs_peak_mode_by_scan", [])) != ntraces:
                self.rs_peak_mode_by_scan = ["Both peaks"] * ntraces

            d = QDialog(self)
            d.setWindowTitle("Randles–Ševčík analysis — manual peak selection for each CV")
            d.resize(1320, 900)
            lay = QVBoxLayout(d)

            info = QLabel(
                "Select one CV scan at a time. Choose whether it contains both peaks, only an anodic peak, "
                "only a cathodic peak, or no usable peak. Then click 'Mark anodic peak' or "
                "'Mark cathodic peak' and click directly on the desired point in the CV."
            )
            info.setWordWrap(True)
            info.setStyleSheet(
                "font-weight:600;padding:7px;background:#F2EEFF;border:1px solid #B9A7E8;"
            )
            lay.addWidget(info)

            top = QHBoxLayout()
            top.addWidget(QLabel("Displayed CV scan"))
            scan_combo = QComboBox()
            for k, rate in enumerate(rates):
                scan_combo.addItem(f"CV {k+1} — {rate:g} V s⁻¹", k)
            top.addWidget(scan_combo)

            top.addSpacing(16)
            top.addWidget(QLabel("Peak availability"))
            availability = QComboBox()
            availability.addItems(["Both peaks", "Anodic only", "Cathodic only", "No usable peak"])
            top.addWidget(availability)

            mark_a = QPushButton("Mark anodic peak")
            mark_c = QPushButton("Mark cathodic peak")
            clear_scan = QPushButton("Clear peaks in this scan")
            top.addWidget(mark_a)
            top.addWidget(mark_c)
            top.addWidget(clear_scan)
            top.addStretch(1)
            lay.addLayout(top)

            fig = Figure(figsize=(11, 6.5), tight_layout=True)
            canvas = FigureCanvas(fig)
            lay.addWidget(NavigationToolbar(canvas, d))
            lay.addWidget(canvas, 1)
            axcv = fig.add_subplot(121)
            axrs = fig.add_subplot(122)

            table = QTableWidget(ntraces, 7)
            table.setHorizontalHeaderLabels([
                "Scan", "Scan rate / V s⁻¹", "Peak availability",
                "Epa / V", "ipa / A", "Epc / V", "ipc / A"
            ])
            table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            table.setMaximumHeight(220)
            lay.addWidget(table)

            bottom = QHBoxLayout()
            previous_btn = QPushButton("Previous scan")
            next_btn = QPushButton("Next scan")
            close_btn = QPushButton("Close / keep peak selections")
            bottom.addWidget(previous_btn)
            bottom.addWidget(next_btn)
            bottom.addStretch(1)
            bottom.addWidget(close_btn)
            lay.addLayout(bottom)

            state = {"mark": None}
            marker_artists = []

            def nearest_index(x_click, y_click, ee, yy):
                good = np.isfinite(ee) & np.isfinite(yy)
                idx_all = np.flatnonzero(good)
                if len(idx_all) == 0:
                    raise ValueError("This CV contains no valid data.")
                ex = ee[good]
                iy = yy[good]
                xspan = max(float(np.nanmax(ex) - np.nanmin(ex)), 1e-12)
                yspan = max(float(np.nanmax(iy) - np.nanmin(iy)), 1e-12)
                dist2 = ((ex - x_click) / xspan) ** 2 + ((iy - y_click) / yspan) ** 2
                return int(idx_all[int(np.nanargmin(dist2))])

            def set_availability_for_scan(k, value):
                self.rs_peak_mode_by_scan[k] = value
                p = self.rs_manual_peaks[k]
                if value == "Anodic only":
                    p["epc"] = p["ipc"] = np.nan
                elif value == "Cathodic only":
                    p["epa"] = p["ipa"] = np.nan
                elif value == "No usable peak":
                    p["epa"] = p["ipa"] = p["epc"] = p["ipc"] = np.nan

            def update_table():
                for k in range(ntraces):
                    p = self.rs_manual_peaks[k]
                    vals = [
                        f"CV {k+1}",
                        f"{rates[k]:.6g}",
                        self.rs_peak_mode_by_scan[k],
                        "" if not np.isfinite(p["epa"]) else f'{p["epa"]:.6g}',
                        "" if not np.isfinite(p["ipa"]) else f'{p["ipa"]:.6E}',
                        "" if not np.isfinite(p["epc"]) else f'{p["epc"]:.6g}',
                        "" if not np.isfinite(p["ipc"]) else f'{p["ipc"]:.6E}',
                    ]
                    for col, val in enumerate(vals):
                        table.setItem(k, col, QTableWidgetItem(val))

            def calculate_rs():
                axrs.clear()
                pref = 0.4463 * z * F * A * c
                x_all = np.sqrt(rates)

                epa = np.array([p["epa"] for p in self.rs_manual_peaks], float)
                ipa = np.array([p["ipa"] for p in self.rs_manual_peaks], float)
                epc = np.array([p["epc"] for p in self.rs_manual_peaks], float)
                ipc = np.array([p["ipc"] for p in self.rs_manual_peaks], float)

                Da = Dc = r2a = r2c = np.nan
                anodic_ok = np.isfinite(ipa)
                cathodic_ok = np.isfinite(ipc)

                if np.count_nonzero(anodic_ok) >= 2:
                    ma, ba, r2a = _linfit(x_all[anodic_ok], np.abs(ipa[anodic_ok]))
                    Da = (ma / pref) ** 2 * (R * T / (z * F))
                    xx = np.linspace(np.nanmin(x_all[anodic_ok]), np.nanmax(x_all[anodic_ok]), 200)
                    axrs.scatter(x_all[anodic_ok], np.abs(ipa[anodic_ok]), label="Anodic |ip|")
                    axrs.plot(xx, ma * xx + ba, label=f"Anodic fit R²={r2a:.5f}")

                if np.count_nonzero(cathodic_ok) >= 2:
                    mc, bc, r2c = _linfit(x_all[cathodic_ok], np.abs(ipc[cathodic_ok]))
                    Dc = (mc / pref) ** 2 * (R * T / (z * F))
                    xx = np.linspace(np.nanmin(x_all[cathodic_ok]), np.nanmax(x_all[cathodic_ok]), 200)
                    axrs.scatter(x_all[cathodic_ok], np.abs(ipc[cathodic_ok]), label="Cathodic |ip|")
                    axrs.plot(xx, mc * xx + bc, label=f"Cathodic fit R²={r2c:.5f}")

                axrs.set_xlabel("√scan rate / (V s⁻¹)¹ᐟ²")
                axrs.set_ylabel("|Peak current| / A")
                axrs.grid(alpha=.25)

                title_parts = []
                if np.isfinite(Da):
                    title_parts.append(f"D anodic={Da:.4E}")
                if np.isfinite(Dc):
                    title_parts.append(f"D cathodic={Dc:.4E}")
                axrs.set_title("; ".join(title_parts) + (" cm² s⁻¹" if title_parts else "Select at least two peaks of one type"))
                if axrs.has_data():
                    axrs.legend()

                self.last_rs = (Da, Dc, r2a, r2c, epa, epc, ipa, ipc, rates)
                self.rs_manual_table = pd.DataFrame({
                    "Scan": np.arange(1, ntraces + 1),
                    "Scan_rate_V_s": rates,
                    "Peak_availability": self.rs_peak_mode_by_scan,
                    "Epa_V": epa,
                    "ipa_A": ipa,
                    "Epc_V": epc,
                    "ipc_A": ipc,
                    "Delta_Ep_V": epa - epc,
                    "abs_ipa_over_ipc": np.abs(ipa / ipc),
                })

            def redraw_cv():
                for art in marker_artists:
                    try:
                        art.remove()
                    except Exception:
                        pass
                marker_artists.clear()

                axcv.clear()
                k = int(scan_combo.currentData())
                yy = curves[k]
                axcv.plot(E, yy, lw=1.6, label=f"CV {k+1}, {rates[k]:g} V s⁻¹")
                axcv.set_xlabel("Potential / V")
                axcv.set_ylabel("Current / A")
                axcv.grid(alpha=.25)
                axcv.legend()

                p = self.rs_manual_peaks[k]
                if np.isfinite(p["epa"]) and np.isfinite(p["ipa"]):
                    marker_artists.append(axcv.plot(p["epa"], p["ipa"], "o", ms=9, label="Anodic peak")[0])
                    axcv.annotate("anodic", (p["epa"], p["ipa"]), xytext=(7, 7), textcoords="offset points")
                if np.isfinite(p["epc"]) and np.isfinite(p["ipc"]):
                    marker_artists.append(axcv.plot(p["epc"], p["ipc"], "o", ms=9, label="Cathodic peak")[0])
                    axcv.annotate("cathodic", (p["epc"], p["ipc"]), xytext=(7, -14), textcoords="offset points")

                availability.blockSignals(True)
                availability.setCurrentText(self.rs_peak_mode_by_scan[k])
                availability.blockSignals(False)

                mode_now = self.rs_peak_mode_by_scan[k]
                mark_a.setEnabled(mode_now in ("Both peaks", "Anodic only"))
                mark_c.setEnabled(mode_now in ("Both peaks", "Cathodic only"))

                calculate_rs()
                update_table()
                canvas.draw_idle()

            def on_availability_changed(text):
                k = int(scan_combo.currentData())
                set_availability_for_scan(k, text)
                redraw_cv()

            def on_click(event):
                if event.inaxes is not axcv or state["mark"] is None:
                    return
                if event.xdata is None or event.ydata is None:
                    return
                k = int(scan_combo.currentData())
                yy = curves[k]
                idx = nearest_index(float(event.xdata), float(event.ydata), E, yy)
                p = self.rs_manual_peaks[k]

                if state["mark"] == "anodic":
                    p["epa"] = float(E[idx])
                    p["ipa"] = float(yy[idx])
                elif state["mark"] == "cathodic":
                    p["epc"] = float(E[idx])
                    p["ipc"] = float(yy[idx])

                state["mark"] = None
                info.setText(
                    "Peak stored. Choose another peak button, change the scan, or continue to the next scan."
                )
                redraw_cv()

            def begin_mark(kind):
                k = int(scan_combo.currentData())
                mode_now = self.rs_peak_mode_by_scan[k]
                if kind == "anodic" and mode_now not in ("Both peaks", "Anodic only"):
                    return
                if kind == "cathodic" and mode_now not in ("Both peaks", "Cathodic only"):
                    return
                state["mark"] = kind
                info.setText(
                    f"CV {k+1}: click directly on the desired {kind.upper()} peak. "
                    "The nearest measured CV point will be stored."
                )

            def clear_current():
                k = int(scan_combo.currentData())
                self.rs_manual_peaks[k] = dict(epa=np.nan, ipa=np.nan, epc=np.nan, ipc=np.nan)
                redraw_cv()

            def previous_scan():
                scan_combo.setCurrentIndex(max(0, scan_combo.currentIndex() - 1))

            def next_scan():
                scan_combo.setCurrentIndex(min(scan_combo.count() - 1, scan_combo.currentIndex() + 1))

            scan_combo.currentIndexChanged.connect(redraw_cv)
            availability.currentTextChanged.connect(on_availability_changed)
            mark_a.clicked.connect(lambda: begin_mark("anodic"))
            mark_c.clicked.connect(lambda: begin_mark("cathodic"))
            clear_scan.clicked.connect(clear_current)
            previous_btn.clicked.connect(previous_scan)
            next_btn.clicked.connect(next_scan)
            close_btn.clicked.connect(d.accept)
            canvas.mpl_connect("button_press_event", on_click)

            update_table()
            redraw_cv()
            canvas.draw()
            d.exec()

            # Final status text after closing the dialog.
            if hasattr(self, "last_rs"):
                Da, Dc, r2a, r2c = self.last_rs[:4]
                parts = []
                if np.isfinite(Da):
                    parts.append(f"D anodic={Da:.4E} cm² s⁻¹ (R²={r2a:.5f})")
                if np.isfinite(Dc):
                    parts.append(f"D cathodic={Dc:.4E} cm² s⁻¹ (R²={r2c:.5f})")
                self.status.setText("Randles–Ševčík: " + ("; ".join(parts) if parts else "no complete peak series yet"))

        except Exception as e:
            QMessageBox.critical(self, "Randles–Ševčík", str(e))

    def transport_mechanism(self):
        """Compare diffusion-controlled and surface-confined scan-rate dependences."""
        try:
            if not hasattr(self, "last_rs"):
                raise ValueError(
                    "Run Randles–Ševčík first and manually select the CV peak currents."
                )

            rates = np.asarray(self.last_rs[8], dtype=float)
            ipa = np.asarray(self.last_rs[6], dtype=float)
            ipc = np.asarray(self.last_rs[7], dtype=float)

            def analyse_branch(currents, label):
                valid = np.isfinite(rates) & np.isfinite(currents) & (rates > 0) & (np.abs(currents) > 0)
                v = rates[valid]
                ip = np.abs(currents[valid])
                if len(v) < 3:
                    return None

                m_sqrt, b_sqrt, r2_sqrt = _linfit(np.sqrt(v), ip)
                m_v, b_v, r2_v = _linfit(v, ip)
                b_exp, log_intercept, r2_log = _linfit(np.log10(v), np.log10(ip))

                if abs(b_exp - 0.5) <= 0.15:
                    interpretation = "Predominantly diffusion-controlled"
                    recommendation = "Randles–Ševčík / Nicholson are mechanistically appropriate."
                elif abs(b_exp - 1.0) <= 0.15:
                    interpretation = "Predominantly surface-confined"
                    recommendation = "Laviron may be appropriate if the redox species is truly immobilized/adsorbed."
                else:
                    interpretation = "Mixed or ambiguous control"
                    recommendation = "Do not assign a single transport model from scan-rate dependence alone."

                return {
                    "label": label,
                    "v": v,
                    "ip": ip,
                    "m_sqrt": m_sqrt,
                    "intercept_sqrt": b_sqrt,
                    "r2_sqrt": r2_sqrt,
                    "m_v": m_v,
                    "intercept_v": b_v,
                    "r2_v": r2_v,
                    "b": b_exp,
                    "log_intercept": log_intercept,
                    "r2_log": r2_log,
                    "interpretation": interpretation,
                    "recommendation": recommendation,
                }

            anodic = analyse_branch(ipa, "Anodic")
            cathodic = analyse_branch(ipc, "Cathodic")
            results = [r for r in (anodic, cathodic) if r is not None]
            if not results:
                raise ValueError(
                    "At least three valid peak currents of one polarity are required."
                )

            # Overall interpretation only if available branches agree.
            interpretations = [r["interpretation"] for r in results]
            if len(set(interpretations)) == 1:
                overall = interpretations[0]
            else:
                overall = "Anodic and cathodic branches do not give the same classification"

            self.last_transport = {
                "anodic": anodic,
                "cathodic": cathodic,
                "overall": overall,
            }

            d = QDialog(self)
            d.setWindowTitle("Transport mechanism analysis")
            d.resize(1350, 900)
            lay = QVBoxLayout(d)

            note = QLabel(
                "Mechanism diagnostic from scan-rate dependence of the manually selected peak currents. "
                "For an ideal dissolved diffusion-controlled species, |ip| ∝ ν^1/2 and the log–log slope b ≈ 0.5. "
                "For an ideal surface-confined species, |ip| ∝ ν and b ≈ 1.0. "
                "Intermediate b values indicate mixed or ambiguous control."
            )
            note.setWordWrap(True)
            note.setStyleSheet(
                "font-weight:600;padding:7px;background:#EEF7FF;border:1px solid #8DB8D9;"
            )
            lay.addWidget(note)

            fig = Figure(figsize=(12, 6.5), tight_layout=True)
            canvas = FigureCanvas(fig)
            lay.addWidget(NavigationToolbar(canvas, d))
            lay.addWidget(canvas, 1)

            ax1 = fig.add_subplot(131)
            ax2 = fig.add_subplot(132)
            ax3 = fig.add_subplot(133)

            current_scale = 1.0e-5
            for r in results:
                v = r["v"]
                ip = r["ip"]

                x1 = np.sqrt(v)
                ax1.scatter(x1, ip/current_scale, label=f'{r["label"]} data')
                xx1 = np.linspace(np.min(x1), np.max(x1), 200)
                ax1.plot(
                    xx1,
                    (r["m_sqrt"] * xx1 + r["intercept_sqrt"])/current_scale,
                    label=f'{r["label"]} fit R²={r["r2_sqrt"]:.4f}'
                )

                ax2.scatter(v, ip/current_scale, label=f'{r["label"]} data')
                xx2 = np.linspace(np.min(v), np.max(v), 200)
                ax2.plot(
                    xx2,
                    (r["m_v"] * xx2 + r["intercept_v"])/current_scale,
                    label=f'{r["label"]} fit R²={r["r2_v"]:.4f}'
                )

                lv = np.log10(v)
                lip = np.log10(ip)
                ax3.scatter(lv, lip, label=f'{r["label"]} data')
                xx3 = np.linspace(np.min(lv), np.max(lv), 200)
                ax3.plot(
                    xx3,
                    r["b"] * xx3 + r["log_intercept"],
                    label=f'{r["label"]}: b={r["b"]:.3f}, R²={r["r2_log"]:.4f}'
                )

            ax1.set_xlabel("√scan rate / (V s⁻¹)¹ᐟ²")
            ax1.set_ylabel(r"|Peak current| / ($10^{-5}$ A)")
            ax1.set_title("Diffusion diagnostic")
            ax1.grid(alpha=.25)
            ax1.legend(fontsize=8)

            ax2.set_xlabel("Scan rate / V s⁻¹")
            ax2.set_ylabel(r"|Peak current| / ($10^{-5}$ A)")
            ax2.set_title("Surface-confined diagnostic")
            ax2.grid(alpha=.25)
            ax2.legend(fontsize=8)

            ax3.set_xlabel("log₁₀(scan rate)")
            ax3.set_ylabel("log₁₀(|peak current|)")
            ax3.set_title("Power-law exponent b")
            ax3.grid(alpha=.25)
            ax3.legend(fontsize=8)

            table = QTableWidget(len(results), 7)
            table.setHorizontalHeaderLabels([
                "Branch", "b", "R² log-log", "R² vs √ν", "R² vs ν",
                "Interpretation", "Recommendation"
            ])
            table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            for i, r in enumerate(results):
                vals = [
                    r["label"],
                    f'{r["b"]:.4f}',
                    f'{r["r2_log"]:.5f}',
                    f'{r["r2_sqrt"]:.5f}',
                    f'{r["r2_v"]:.5f}',
                    r["interpretation"],
                    r["recommendation"],
                ]
                for j, val in enumerate(vals):
                    table.setItem(i, j, QTableWidgetItem(val))
            table.setMaximumHeight(190)
            lay.addWidget(table)

            b_parts = []
            if anodic is not None:
                b_parts.append(f"bₐ = {anodic['b']:.3f}")
            if cathodic is not None:
                b_parts.append(f"b꜀ = {cathodic['b']:.3f}")
            b_text = ", ".join(b_parts)
            result_text = f"{overall} ({b_text})" if b_text else overall

            result_label = QLabel(result_text)
            try:
                from PySide6.QtCore import Qt as _Qt
                result_label.setAlignment(_Qt.AlignmentFlag.AlignCenter)
            except Exception:
                pass
            result_label.setStyleSheet(
                "font-size:15px;font-weight:700;padding:9px;background:#F5F5F5;border:1px solid #CCCCCC;"
            )
            lay.addWidget(result_label)

            canvas.draw()
            d.exec()

            status_parts = [
                f'{r["label"]} b={r["b"]:.3f} ({r["interpretation"]})'
                for r in results
            ]
            self.status.setText("Transport mechanism: " + "; ".join(status_parts))

        except Exception as e:
            QMessageBox.critical(self, "Transport mechanism", str(e))

    def _nicholson_diffusion_source(self):
        """Prefer transient-method D values because reversible Randles-Sevcik D is unreliable for quasireversible systems."""
        if hasattr(self, "last_cottrell") and np.isfinite(self.last_cottrell[0]):
            return float(self.last_cottrell[0]), "Cottrell"
        if hasattr(self, "last_anson") and np.isfinite(self.last_anson[0]):
            return float(self.last_anson[0]), "Anson"
        if hasattr(self, "last_rs"):
            vals = [v for v in (self.last_rs[0], self.last_rs[1]) if np.isfinite(v)]
            if vals:
                return float(np.mean(vals)), "mean Randles–Ševčík (fallback)"
        raise ValueError(
            "Nicholson requires a diffusion coefficient. Run Cottrell or Anson analysis first "
            "(preferred), or Randles–Ševčík as a fallback."
        )

    def nicholson(self):
        """Nicholson/Lavagnini analysis of heterogeneous electron-transfer rate constant k0."""
        try:
            if not hasattr(self, "last_rs"):
                raise ValueError(
                    "Run Randles–Ševčík first and manually mark anodic and cathodic peaks."
                )

            c, z, A, T = self._params()
            rates = np.asarray(self.last_rs[8], dtype=float)
            epa = np.asarray(self.last_rs[4], dtype=float)
            epc = np.asarray(self.last_rs[5], dtype=float)
            D, D_source = self._nicholson_diffusion_source()

            dep_mV = np.abs(epa - epc) * 1000.0
            ndep_mV = z * dep_mV
            valid = (
                np.isfinite(rates) & np.isfinite(epa) & np.isfinite(epc) &
                (rates > 0) & (ndep_mV > 63.0) & (ndep_mV < 212.0)
            )

            # Lavagnini approximation of Nicholson's working curve.
            psi = np.full_like(dep_mV, np.nan, dtype=float)
            psi[valid] = (
                -0.6288 + 0.0021 * ndep_mV[valid]
            ) / (
                1.0 - 0.017 * ndep_mV[valid]
            )

            # k0 = psi * sqrt(pi D n F v / RT)
            k0 = np.full_like(dep_mV, np.nan, dtype=float)
            goodpsi = valid & np.isfinite(psi) & (psi > 0)
            k0[goodpsi] = psi[goodpsi] * np.sqrt(
                np.pi * D * z * F * rates[goodpsi] / (R * T)
            )

            self.last_nicholson = {
                "rates": rates, "Epa": epa, "Epc": epc,
                "DeltaEp_mV": dep_mV, "nDeltaEp_mV": ndep_mV,
                "psi": psi, "k0_cm_s": k0, "valid": goodpsi,
                "D_cm2_s": D, "D_source": D_source,
            }

            d = QDialog(self)
            d.setWindowTitle("Nicholson analysis")
            d.resize(1150, 800)
            lay = QVBoxLayout(d)

            note = QLabel(
                "Nicholson analysis: The anodic and cathodic peak potentials (Epa and Epc) "
                "were selected manually. Their peak separation ΔEp is converted to the Nicholson "
                "kinetic parameter ψ using the Lavagnini approximation. This approximation is "
                "applied only when 63 < n·ΔEp < 212 mV. "
                f"For calculating k⁰, the diffusion coefficient from the {D_source} analysis is used: "
                f"D = {D:.4E} cm² s⁻¹."
            )
            note.setWordWrap(True)
            note.setStyleSheet(
                "font-weight:600;padding:7px;background:#EEF8F0;border:1px solid #93C89B;"
            )
            lay.addWidget(note)

            fig = Figure(figsize=(10, 6), tight_layout=True)
            canvas = FigureCanvas(fig)
            lay.addWidget(NavigationToolbar(canvas, d))
            lay.addWidget(canvas)

            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)

            ax1.plot(np.log10(rates[np.isfinite(dep_mV)]), dep_mV[np.isfinite(dep_mV)], "o-")
            ax1.set_xlabel("log₁₀(scan rate / V s⁻¹)")
            ax1.set_ylabel("ΔEp / mV")
            ax1.set_title("Peak separation")
            ax1.grid(alpha=.25)

            if np.any(goodpsi):
                ax2.plot(rates[goodpsi], k0[goodpsi], "o")
                mean_k0 = float(np.nanmean(k0[goodpsi]))
                sd_k0 = float(np.nanstd(k0[goodpsi], ddof=1)) if np.count_nonzero(goodpsi) > 1 else np.nan
                ax2.axhline(mean_k0, ls="--", label=f"Mean k⁰ = {mean_k0:.3E} cm s⁻¹")
                ax2.legend()
                self.status.setText(
                    f"Nicholson: k⁰ = {mean_k0:.4E} cm s⁻¹ "
                    f"(D from {D_source}; {np.count_nonzero(goodpsi)} valid scans)"
                )
            else:
                mean_k0 = sd_k0 = np.nan
                ax2.text(
                    .5, .5, "No scan satisfies\n63 < nΔEp < 212 mV",
                    ha="center", va="center", transform=ax2.transAxes
                )

            ax2.set_xlabel("Scan rate / V s⁻¹")
            ax2.set_ylabel("k⁰ / cm s⁻¹")
            ax2.set_title("Nicholson heterogeneous rate constant")
            ax2.grid(alpha=.25)

            self.last_nicholson["mean_k0_cm_s"] = mean_k0
            self.last_nicholson["sd_k0_cm_s"] = sd_k0

            table = QTableWidget(len(rates), 6)
            table.setHorizontalHeaderLabels([
                "Scan rate / V s⁻¹", "ΔEp / mV", "nΔEp / mV",
                "ψ", "k⁰ / cm s⁻¹", "Nicholson valid?"
            ])
            table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            for i in range(len(rates)):
                vals = [
                    f"{rates[i]:.6g}",
                    "" if not np.isfinite(dep_mV[i]) else f"{dep_mV[i]:.3f}",
                    "" if not np.isfinite(ndep_mV[i]) else f"{ndep_mV[i]:.3f}",
                    "" if not np.isfinite(psi[i]) else f"{psi[i]:.6g}",
                    "" if not np.isfinite(k0[i]) else f"{k0[i]:.6E}",
                    "Yes" if goodpsi[i] else "No",
                ]
                for j, val in enumerate(vals):
                    table.setItem(i, j, QTableWidgetItem(val))
            table.setMaximumHeight(230)
            lay.addWidget(table)

            canvas.draw()
            d.exec()

        except Exception as e:
            QMessageBox.critical(self, "Nicholson", str(e))

    def laviron(self):
        """Laviron high-scan-rate analysis. Primarily intended for surface-confined/diffusionless redox systems."""
        try:
            if not hasattr(self, "last_rs"):
                raise ValueError(
                    "Run Randles–Ševčík first and manually mark anodic/cathodic peaks."
                )

            c, z, A, T = self._params()
            rates = np.asarray(self.last_rs[8], dtype=float)
            epa = np.asarray(self.last_rs[4], dtype=float)
            epc = np.asarray(self.last_rs[5], dtype=float)

            both = np.isfinite(rates) & np.isfinite(epa) & np.isfinite(epc) & (rates > 0)
            if np.count_nonzero(both) < 3:
                raise ValueError(
                    "Laviron analysis requires at least three scans with both anodic and cathodic peaks."
                )

            rates_b = rates[both]
            epa_b = epa[both]
            epc_b = epc[both]
            logv = np.log10(rates_b)

            # Formal-potential estimate from the slowest scan.
            slow = int(np.argmin(rates_b))
            E0 = 0.5 * (epa_b[slow] + epc_b[slow])

            # Use points with sufficiently large peak displacement from E0 when possible.
            eta_a = epa_b - E0
            eta_c = epc_b - E0
            high_a = np.abs(eta_a) >= 0.100
            high_c = np.abs(eta_c) >= 0.100
            if np.count_nonzero(high_a) < 2:
                high_a = np.ones_like(logv, dtype=bool)
            if np.count_nonzero(high_c) < 2:
                high_c = np.ones_like(logv, dtype=bool)

            ma, ba, r2a = _linfit(logv[high_a], epa_b[high_a])
            mc, bc, r2c = _linfit(logv[high_c], epc_b[high_c])

            # Laviron slopes (V per decade):
            # anodic: +2.303RT / [(1-alpha)nF]
            # cathodic: -2.303RT / [alpha nF]
            alpha_from_c = -2.303 * R * T / (z * F * mc) if mc < 0 else np.nan
            one_minus_alpha_from_a = 2.303 * R * T / (z * F * ma) if ma > 0 else np.nan
            alpha_from_a = 1.0 - one_minus_alpha_from_a if np.isfinite(one_minus_alpha_from_a) else np.nan

            alpha_candidates = [
                a for a in (alpha_from_c, alpha_from_a)
                if np.isfinite(a) and 0 < a < 1
            ]
            alpha = float(np.mean(alpha_candidates)) if alpha_candidates else np.nan

            dep = np.abs(epa_b - epc_b)
            ks = np.full_like(rates_b, np.nan, dtype=float)

            # Classical Laviron expression for surface-confined systems.
            if np.isfinite(alpha) and 0 < alpha < 1:
                logks = (
                    alpha * np.log10(1.0 - alpha)
                    + (1.0 - alpha) * np.log10(alpha)
                    - np.log10(R * T / (z * F * rates_b))
                    - alpha * (1.0 - alpha) * z * F * dep / (2.303 * R * T)
                )
                ks = 10.0 ** logks

            self.last_laviron = {
                "rates": rates_b,
                "log10_rates": logv,
                "Epa": epa_b, "Epc": epc_b, "DeltaEp_V": dep,
                "E0_est_V": E0,
                "slope_anodic_V_dec": ma,
                "slope_cathodic_V_dec": mc,
                "R2_anodic": r2a, "R2_cathodic": r2c,
                "alpha_from_anodic": alpha_from_a,
                "alpha_from_cathodic": alpha_from_c,
                "alpha": alpha,
                "ks_s-1": ks,
                "mean_ks_s-1": float(np.nanmean(ks)) if np.any(np.isfinite(ks)) else np.nan,
            }

            d = QDialog(self)
            d.setWindowTitle("Laviron analysis")
            d.resize(1150, 820)
            lay = QVBoxLayout(d)

            note = QLabel(
                "Important: the classical Laviron treatment implemented here is primarily a "
                "surface-confined/diffusionless high-scan-rate model. Use it for freely diffusing "
                "solution species only with appropriate mechanistic justification. "
                "E⁰′ is estimated from (Epa+Epc)/2 at the slowest scan."
            )
            note.setWordWrap(True)
            note.setStyleSheet(
                "font-weight:600;padding:7px;background:#FFF5E8;border:1px solid #D9A85A;"
            )
            lay.addWidget(note)

            fig = Figure(figsize=(10, 6), tight_layout=True)
            canvas = FigureCanvas(fig)
            lay.addWidget(NavigationToolbar(canvas, d))
            lay.addWidget(canvas)
            ax = fig.add_subplot(111)

            ax.scatter(logv, epa_b, label="Epa")
            ax.scatter(logv, epc_b, label="Epc")
            xa = np.linspace(np.min(logv[high_a]), np.max(logv[high_a]), 200)
            xc = np.linspace(np.min(logv[high_c]), np.max(logv[high_c]), 200)
            ax.plot(xa, ma*xa + ba, label=f"Anodic fit R²={r2a:.5f}")
            ax.plot(xc, mc*xc + bc, label=f"Cathodic fit R²={r2c:.5f}")
            ax.axhline(E0, ls="--", label=f"E⁰′ estimate = {E0:.4f} V")
            ax.set_xlabel("log₁₀(scan rate / V s⁻¹)")
            ax.set_ylabel("Peak potential / V")
            ax.set_title(
                f"Laviron: α ≈ {alpha:.3f}" if np.isfinite(alpha)
                else "Laviron: α could not be determined reliably"
            )
            ax.grid(alpha=.25)
            ax.legend()

            result = QLabel()
            if np.isfinite(alpha):
                mean_ks = self.last_laviron["mean_ks_s-1"]
                result.setText(
                    f"α from cathodic branch = {alpha_from_c:.4f} | "
                    f"α from anodic branch = {alpha_from_a:.4f} | "
                    f"mean α = {alpha:.4f} | "
                    f"mean apparent kₛ = {mean_ks:.4E} s⁻¹"
                )
                self.status.setText(
                    f"Laviron: α={alpha:.4f}; apparent kₛ={mean_ks:.4E} s⁻¹"
                )
            else:
                result.setText("Laviron α could not be determined from the current peak-potential slopes.")
            result.setStyleSheet("font-weight:700;padding:7px;background:#F7F7F7;")
            lay.addWidget(result)

            table = QTableWidget(len(rates_b), 5)
            table.setHorizontalHeaderLabels([
                "Scan rate / V s⁻¹", "Epa / V", "Epc / V",
                "ΔEp / V", "apparent kₛ / s⁻¹"
            ])
            table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            for i in range(len(rates_b)):
                vals = [
                    f"{rates_b[i]:.6g}", f"{epa_b[i]:.6g}", f"{epc_b[i]:.6g}",
                    f"{dep[i]:.6g}",
                    "" if not np.isfinite(ks[i]) else f"{ks[i]:.6E}",
                ]
                for j, val in enumerate(vals):
                    table.setItem(i, j, QTableWidgetItem(val))
            table.setMaximumHeight(220)
            lay.addWidget(table)

            canvas.draw()
            d.exec()

        except Exception as e:
            QMessageBox.critical(self, "Laviron", str(e))

    def summary(self):
        rows = []
        if hasattr(self, "last_cottrell"):
            rows.append(("Cottrell", self.last_cottrell[0], self.last_cottrell[1]))
        if hasattr(self, "last_anson"):
            rows.append(("Anson", self.last_anson[0], self.last_anson[1]))
        if hasattr(self, "last_rs"):
            Da, Dc, r2a, r2c = self.last_rs[:4]
            if np.isfinite(Da):
                rows.append(("Randles–Ševčík anodic", Da, r2a))
            if np.isfinite(Dc):
                rows.append(("Randles–Ševčík cathodic", Dc, r2c))

        if not rows and not hasattr(self, "last_nicholson") and not hasattr(self, "last_laviron"):
            QMessageBox.information(self, "Summary", "Run at least one analysis first.")
            return

        txt = "\n".join(
            f"{name}: D = {D:.5E} cm² s⁻¹; R² = {r2:.5f}"
            for name, D, r2 in rows
        )
        if hasattr(self, "last_transport"):
            tr = self.last_transport
            txt += "\n\nTransport mechanism:"
            for key in ("anodic", "cathodic"):
                r = tr.get(key)
                if r is not None:
                    txt += (
                        f"\n{r['label']}: b = {r['b']:.3f}; "
                        f"R²(log-log) = {r['r2_log']:.4f}; "
                        f"{r['interpretation']}"
                    )
            txt += f"\nOverall: {tr['overall']}"

        if hasattr(self, "last_nicholson"):
            nk = self.last_nicholson
            if np.isfinite(nk.get("mean_k0_cm_s", np.nan)):
                txt += (
                    f"\n\nNicholson: mean k⁰ = {nk['mean_k0_cm_s']:.5E} cm s⁻¹"
                    f"\nD source: {nk['D_source']}"
                )
        if hasattr(self, "last_laviron"):
            lv = self.last_laviron
            if np.isfinite(lv.get("alpha", np.nan)):
                txt += (
                    f"\n\nLaviron (surface-confined model): α = {lv['alpha']:.4f}"
                    f"\napparent mean kₛ = {lv['mean_ks_s-1']:.5E} s⁻¹"
                )
        QMessageBox.information(self, "Diffusion / kinetics summary", txt)

    def export_excel(self):
        try:
            p,_=QFileDialog.getSaveFileName(self,"Export Excel","Diffusion_Coefficient_Analysis.xlsx","Excel (*.xlsx)")
            if not p:return
            if not p.lower().endswith(".xlsx"):p+=".xlsx"
            with pd.ExcelWriter(p,engine="xlsxwriter") as writer:
                wb=writer.book
                summary=[]
                if self.ca is not None:
                    t,i=self._ca_arrays(); x=np.where(t>0,1/np.sqrt(t),np.nan)
                    pd.DataFrame({"Time_s":t,"Current_A":i,"t^-1/2":x}).to_excel(writer,sheet_name="Cottrell",index=False)
                    ws = writer.sheets["Cottrell"]
                    nrows = len(t)
                    if nrows:
                        chart = wb.add_chart({"type": "scatter", "subtype": "smooth_with_markers"})
                        chart.add_series({
                            "name": "Cottrell data",
                            "categories": ["Cottrell", 1, 2, nrows, 2],
                            "values": ["Cottrell", 1, 1, nrows, 1],
                        })
                        chart.set_title({"name": "Cottrell analysis"})
                        chart.set_x_axis({"name": "t^-1/2 / s^-1/2"})
                        chart.set_y_axis({"name": "Current / A"})
                        chart.set_legend({"none": True})
                        ws.insert_chart("E2", chart, {"x_scale": 1.25, "y_scale": 1.15})
                if self.cc is not None:
                    t,q=self._cc_arrays()
                    sqrt_t = np.sqrt(np.clip(t,0,None))
                    pd.DataFrame({"Time_s":t,"Charge_C":q,"sqrt_t":sqrt_t}).to_excel(writer,sheet_name="Anson",index=False)
                    ws = writer.sheets["Anson"]
                    nrows = len(t)
                    if nrows:
                        chart = wb.add_chart({"type": "scatter", "subtype": "smooth_with_markers"})
                        chart.add_series({
                            "name": "Anson data",
                            "categories": ["Anson", 1, 2, nrows, 2],
                            "values": ["Anson", 1, 1, nrows, 1],
                        })
                        chart.set_title({"name": "Anson analysis"})
                        chart.set_x_axis({"name": "t^1/2 / s^1/2"})
                        chart.set_y_axis({"name": "Charge / C"})
                        chart.set_legend({"none": True})
                        ws.insert_chart("E2", chart, {"x_scale": 1.25, "y_scale": 1.15})
                if self.cv is not None:
                    rates,ipa,ipc,epa,epc=self._cv_peaks()
                    peak_modes = getattr(self, "rs_peak_mode_by_scan", ["Automatic"] * len(rates))
                    if len(peak_modes) != len(rates):
                        peak_modes = ["Automatic"] * len(rates)
                    pd.DataFrame({
                        "Scan": np.arange(1, len(rates)+1),
                        "Scan_rate_V_s": rates,
                        "sqrt_scan_rate": np.sqrt(rates),
                        "Peak_availability": peak_modes,
                        "ipa_A": ipa,
                        "Epa_V": epa,
                        "ipc_A": ipc,
                        "Epc_V": epc,
                        "Delta_Ep_V": epa-epc,
                        "abs_ipa_over_ipc": np.abs(ipa/ipc)
                    }).to_excel(writer, sheet_name="Randles-Sevcik", index=False)
                    ws = writer.sheets["Randles-Sevcik"]
                    nrows = len(rates)
                    if nrows:
                        chart = wb.add_chart({"type": "scatter", "subtype": "smooth_with_markers"})
                        chart.add_series({
                            "name": "Anodic peak current",
                            "categories": ["Randles-Sevcik", 1, 2, nrows, 2],
                            "values": ["Randles-Sevcik", 1, 4, nrows, 4],
                        })
                        chart.add_series({
                            "name": "Cathodic peak current",
                            "categories": ["Randles-Sevcik", 1, 2, nrows, 2],
                            "values": ["Randles-Sevcik", 1, 6, nrows, 6],
                        })
                        chart.set_title({"name": "Randles-Sevcik analysis"})
                        chart.set_x_axis({"name": "sqrt(scan rate / V s^-1)"})
                        chart.set_y_axis({"name": "Peak current / A"})
                        ws.insert_chart("L2", chart, {"x_scale": 1.30, "y_scale": 1.20})
                if hasattr(self,"last_cottrell"): summary.append(["Cottrell",self.last_cottrell[0],self.last_cottrell[1]])
                if hasattr(self,"last_anson"): summary.append(["Anson",self.last_anson[0],self.last_anson[1]])
                if hasattr(self,"last_rs"):
                    Da, Dc, r2a, r2c = self.last_rs[:4]
                    if np.isfinite(Da):
                        summary.append(["Randles-Sevcik anodic", Da, r2a])
                    if np.isfinite(Dc):
                        summary.append(["Randles-Sevcik cathodic", Dc, r2c])
                if hasattr(self, "last_transport"):
                    tr = self.last_transport
                    rates = np.asarray(self.last_rs[8], dtype=float)
                    ipa = np.asarray(self.last_rs[6], dtype=float)
                    ipc = np.asarray(self.last_rs[7], dtype=float)

                    transport_df = pd.DataFrame({
                        "Scan_rate_V_s": rates,
                        "sqrt_scan_rate": np.sqrt(rates),
                        "log10_scan_rate": np.where(rates > 0, np.log10(rates), np.nan),
                        "abs_ipa_A": np.abs(ipa),
                        "log10_abs_ipa": np.where(np.isfinite(ipa) & (np.abs(ipa) > 0), np.log10(np.abs(ipa)), np.nan),
                        "abs_ipc_A": np.abs(ipc),
                        "log10_abs_ipc": np.where(np.isfinite(ipc) & (np.abs(ipc) > 0), np.log10(np.abs(ipc)), np.nan),
                    })
                    transport_df.to_excel(writer, sheet_name="Transport mechanism", index=False)
                    ws = writer.sheets["Transport mechanism"]

                    summary_rows = []
                    for key in ("anodic", "cathodic"):
                        r = tr.get(key)
                        if r is not None:
                            summary_rows.append([
                                r["label"], r["b"], r["r2_log"], r["r2_sqrt"], r["r2_v"],
                                r["interpretation"], r["recommendation"]
                            ])
                    if summary_rows:
                        pd.DataFrame(
                            summary_rows,
                            columns=["Branch","b","R2_loglog","R2_vs_sqrt_v","R2_vs_v","Interpretation","Recommendation"]
                        ).to_excel(writer, sheet_name="Transport mechanism", index=False, startrow=len(transport_df)+3)

                    nrows = len(rates)
                    if nrows:
                        ch1 = wb.add_chart({"type":"scatter","subtype":"smooth_with_markers"})
                        ch1.add_series({
                            "name":"Anodic |ip|",
                            "categories":["Transport mechanism",1,1,nrows,1],
                            "values":["Transport mechanism",1,3,nrows,3]
                        })
                        ch1.add_series({
                            "name":"Cathodic |ip|",
                            "categories":["Transport mechanism",1,1,nrows,1],
                            "values":["Transport mechanism",1,5,nrows,5]
                        })
                        ch1.set_title({"name":"Peak current vs sqrt(scan rate)"})
                        ch1.set_x_axis({"name":"sqrt(scan rate / V s^-1)"})
                        ch1.set_y_axis({"name":"|Peak current| / A"})
                        ws.insert_chart("I2", ch1, {"x_scale":1.25,"y_scale":1.15})

                        ch2 = wb.add_chart({"type":"scatter","subtype":"smooth_with_markers"})
                        ch2.add_series({
                            "name":"Anodic |ip|",
                            "categories":["Transport mechanism",1,0,nrows,0],
                            "values":["Transport mechanism",1,3,nrows,3]
                        })
                        ch2.add_series({
                            "name":"Cathodic |ip|",
                            "categories":["Transport mechanism",1,0,nrows,0],
                            "values":["Transport mechanism",1,5,nrows,5]
                        })
                        ch2.set_title({"name":"Peak current vs scan rate"})
                        ch2.set_x_axis({"name":"Scan rate / V s^-1"})
                        ch2.set_y_axis({"name":"|Peak current| / A"})
                        ws.insert_chart("I20", ch2, {"x_scale":1.25,"y_scale":1.15})

                        ch3 = wb.add_chart({"type":"scatter","subtype":"smooth_with_markers"})
                        ch3.add_series({
                            "name":"Anodic log-log",
                            "categories":["Transport mechanism",1,2,nrows,2],
                            "values":["Transport mechanism",1,4,nrows,4]
                        })
                        ch3.add_series({
                            "name":"Cathodic log-log",
                            "categories":["Transport mechanism",1,2,nrows,2],
                            "values":["Transport mechanism",1,6,nrows,6]
                        })
                        ch3.set_title({"name":"log(|ip|) vs log(scan rate)"})
                        ch3.set_x_axis({"name":"log10(scan rate)"})
                        ch3.set_y_axis({"name":"log10(|Peak current|)"})
                        ws.insert_chart("I38", ch3, {"x_scale":1.25,"y_scale":1.15})

                if hasattr(self, "last_nicholson"):
                    nk = self.last_nicholson
                    pd.DataFrame({
                        "Scan_rate_V_s": nk["rates"],
                        "Epa_V": nk["Epa"],
                        "Epc_V": nk["Epc"],
                        "DeltaEp_mV": nk["DeltaEp_mV"],
                        "nDeltaEp_mV": nk["nDeltaEp_mV"],
                        "psi": nk["psi"],
                        "k0_cm_s": nk["k0_cm_s"],
                        "Nicholson_valid": nk["valid"],
                    }).to_excel(writer, sheet_name="Nicholson", index=False)
                    ws = writer.sheets["Nicholson"]
                    ws.write("J2", "D source")
                    ws.write("K2", nk["D_source"])
                    ws.write("J3", "D / cm² s⁻¹")
                    ws.write_number("K3", float(nk["D_cm2_s"]))
                    if np.isfinite(nk.get("mean_k0_cm_s", np.nan)):
                        ws.write("J4", "Mean k0 / cm s⁻¹")
                        ws.write_number("K4", float(nk["mean_k0_cm_s"]))

                    nrows = len(nk["rates"])
                    if nrows:
                        # Excel equivalent of "scatter with smooth/interpolated lines and markers".
                        ch1 = writer.book.add_chart({"type": "scatter", "subtype": "smooth_with_markers"})
                        ch1.add_series({
                            "name": "Peak separation",
                            "categories": ["Nicholson", 1, 0, nrows, 0],
                            "values": ["Nicholson", 1, 3, nrows, 3],
                        })
                        ch1.set_title({"name": "Peak separation"})
                        ch1.set_x_axis({"name": "Scan rate / V s^-1"})
                        ch1.set_y_axis({"name": "Delta Ep / mV"})
                        ch1.set_legend({"none": True})
                        ws.insert_chart("J7", ch1, {"x_scale": 1.25, "y_scale": 1.15})

                        ch2 = writer.book.add_chart({"type": "scatter", "subtype": "smooth_with_markers"})
                        ch2.add_series({
                            "name": "Nicholson k0",
                            "categories": ["Nicholson", 1, 0, nrows, 0],
                            "values": ["Nicholson", 1, 6, nrows, 6],
                        })
                        ch2.set_title({"name": "Nicholson heterogeneous rate constant"})
                        ch2.set_x_axis({"name": "Scan rate / V s^-1"})
                        ch2.set_y_axis({"name": "k0 / cm s^-1"})
                        ch2.set_legend({"none": True})
                        ws.insert_chart("J23", ch2, {"x_scale": 1.25, "y_scale": 1.15})

                if hasattr(self, "last_laviron"):
                    lv = self.last_laviron
                    pd.DataFrame({
                        "Scan_rate_V_s": lv["rates"],
                        "log10_scan_rate": lv["log10_rates"],
                        "Epa_V": lv["Epa"],
                        "Epc_V": lv["Epc"],
                        "DeltaEp_V": lv["DeltaEp_V"],
                        "apparent_ks_s-1": lv["ks_s-1"],
                    }).to_excel(writer, sheet_name="Laviron", index=False)
                    ws = writer.sheets["Laviron"]
                    ws.write("H2", "E0 estimate / V")
                    ws.write_number("I2", float(lv["E0_est_V"]))
                    ws.write("H3", "alpha")
                    if np.isfinite(lv["alpha"]):
                        ws.write_number("I3", float(lv["alpha"]))
                    ws.write("H4", "R2 anodic")
                    ws.write_number("I4", float(lv["R2_anodic"]))
                    ws.write("H5", "R2 cathodic")
                    ws.write_number("I5", float(lv["R2_cathodic"]))
                    ws.write("H6", "Mean apparent ks / s^-1")
                    if np.isfinite(lv["mean_ks_s-1"]):
                        ws.write_number("I6", float(lv["mean_ks_s-1"]))

                    nrows = len(lv["rates"])
                    if nrows:
                        ch = writer.book.add_chart({"type": "scatter", "subtype": "smooth_with_markers"})
                        ch.add_series({
                            "name": "Epa",
                            "categories": ["Laviron", 1, 1, nrows, 1],
                            "values": ["Laviron", 1, 2, nrows, 2],
                        })
                        ch.add_series({
                            "name": "Epc",
                            "categories": ["Laviron", 1, 1, nrows, 1],
                            "values": ["Laviron", 1, 3, nrows, 3],
                        })
                        ch.set_title({"name": "Laviron peak-potential analysis"})
                        ch.set_x_axis({"name": "log10(scan rate / V s^-1)"})
                        ch.set_y_axis({"name": "Peak potential / V"})
                        ws.insert_chart("H8", ch, {"x_scale": 1.35, "y_scale": 1.25})

                        ch2 = writer.book.add_chart({"type": "scatter", "subtype": "smooth_with_markers"})
                        ch2.add_series({
                            "name": "Apparent ks",
                            "categories": ["Laviron", 1, 0, nrows, 0],
                            "values": ["Laviron", 1, 5, nrows, 5],
                        })
                        ch2.set_title({"name": "Laviron apparent rate constant"})
                        ch2.set_x_axis({"name": "Scan rate / V s^-1"})
                        ch2.set_y_axis({"name": "Apparent ks / s^-1"})
                        ch2.set_legend({"none": True})
                        ws.insert_chart("H25", ch2, {"x_scale": 1.35, "y_scale": 1.25})

                if summary:
                    pd.DataFrame(summary,columns=["Method","D_cm2_s","R2"]).to_excel(writer,sheet_name="Summary",index=False)
            self.status.setText(f"Excel exported: {p}")
        except Exception as e: QMessageBox.critical(self,"Excel export",str(e))


def main():
    app=QApplication.instance() or QApplication([])
    w=DiffusionWindow(); w.show()
    app.exec()


if __name__=="__main__":
    main()

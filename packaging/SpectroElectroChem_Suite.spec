# PyInstaller spec for SpectroElectroChem Suite v3.0

from pathlib import Path
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = Path(SPECPATH).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

datas = [
    (str(ROOT / "src" / "spectroelectrochem_suite" / "plugins" / "plugins.json"),
     "spectroelectrochem_suite/plugins"),
    (str(ROOT / "docs" / "User_Manual.pdf"), "docs"),
    (str(ROOT / "docs" / "index.html"), "docs"),
    (str(ROOT / "run_plugin.py"), "."),
]
datas += collect_data_files("plotly")
datas += collect_data_files("PySide6")

a = Analysis(
    [str(ROOT / "main.py")],
    pathex=[str(SRC)],
    binaries=[],
    datas=datas,
    hiddenimports=
    collect_submodules("spectroelectrochem_suite")
    + [
        "pybaselines",
        "scipy",
        "plotly",
        "openpyxl",
        "PySide6",
        "matplotlib.backends.backend_pdf",
        "matplotlib.backends.backend_agg",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="SpectroElectroChem_Suite",
    icon=r"..\sec_logo.ico",
    debug=False,
    strip=False,
    upx=True,
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    name="SpectroElectroChem_Suite",
)

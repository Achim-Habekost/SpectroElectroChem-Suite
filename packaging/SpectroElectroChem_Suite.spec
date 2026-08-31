# PyInstaller spec for SpectroElectroChem Suite v6.0.0

from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files

ROOT = Path.cwd()

datas = [
    ("src/spectroelectrochem_suite/plugins/plugins.json", "spectroelectrochem_suite/plugins"),
    ("docs/User_Manual.pdf", "docs"),
    ("docs/index.html", "docs"),
    ("run_plugin.py", "."),
]
datas += collect_data_files("plotly")
datas += collect_data_files("PySide6")

a = Analysis(
    ["main.py"],
    pathex=["."],
    binaries=[],
    datas=datas,
    hiddenimports=[
        "spectroelectrochem_suite.modules.raman_spectrum_analysis",
        "spectroelectrochem_suite.modules.sers_raman_voltammogram",
        "spectroelectrochem_suite.modules.absorpto_fluoro_voltammogram",
        "spectroelectrochem_suite.modules.rrde_analysis",
        "spectroelectrochem_suite.modules.spectro_cv_synchronization",
        "spectroelectrochem_suite.modules.ecl_integrated_signal",
        "spectroelectrochem_suite.modules.diffusion_coefficient_analysis",
        "spectroelectrochem_suite.modules.eis_analysis",
        "spectroelectrochem_suite.modules.stripping_voltammetry",
        "spectroelectrochem_suite.modules.surface_activation_sers",
        "pybaselines",
        "scipy",
        "plotly",
        "openpyxl",
        "PySide6",
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
    icon=str(ROOT / "sec_logo.ico"),
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

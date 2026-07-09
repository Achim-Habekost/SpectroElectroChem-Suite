# PyInstaller spec for SpectroElectroChem Suite v3.0

from PyInstaller.utils.hooks import collect_data_files

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

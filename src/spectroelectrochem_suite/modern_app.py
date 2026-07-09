from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


APP_TITLE = "SpectroElectroChem Suite"


def app_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def start_plugin(plugin_id: str) -> None:
    runner = app_root() / "run_plugin.py"
    creationflags = subprocess.CREATE_NEW_CONSOLE if sys.platform.startswith("win") else 0
    subprocess.Popen([sys.executable, str(runner), plugin_id], cwd=str(app_root()), creationflags=creationflags)


def load_plugins():
    path = Path(__file__).resolve().parent / "plugins" / "plugins.json"
    return json.loads(path.read_text(encoding="utf-8"))["plugins"]


try:
    from PySide6.QtCore import QUrl
    from PySide6.QtGui import QAction, QDesktopServices
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QGridLayout,
        QLabel, QPushButton, QFrame, QToolBar, QStatusBar, QMessageBox,
        QFileDialog, QTextEdit
    )
except ModuleNotFoundError:
    from spectroelectrochem_suite.app import main
    main()
    raise SystemExit


class ModuleCard(QFrame):
    def __init__(self, plugin):
        super().__init__()
        self.plugin = plugin
        self.setObjectName("moduleCard")
        layout = QVBoxLayout(self)

        title = QLabel(plugin["name"])
        title.setObjectName("cardTitle")
        desc = QLabel(plugin["description"])
        desc.setWordWrap(True)
        btn = QPushButton("Open module")
        btn.clicked.connect(lambda: start_plugin(plugin["id"]))

        layout.addWidget(title)
        layout.addWidget(desc)
        layout.addStretch()
        layout.addWidget(btn)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        from spectroelectrochem_suite import __version__
        self.version = __version__
        self.plugins = load_plugins()
        self.setWindowTitle(f"{APP_TITLE} v{self.version}")
        self.resize(1100, 720)
        self.apply_style()
        self.create_menu()
        self.create_toolbar()
        self.create_content()
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage(f"Ready - {APP_TITLE} v{self.version}")

    def apply_style(self):
        self.setStyleSheet("""
        QMainWindow { background: #f4f6f8; }
        QLabel#mainTitle { font-size: 28px; font-weight: 700; color: #18324a; }
        QLabel#subtitle { font-size: 13px; color: #4a5a68; }
        QLabel#cardTitle { font-size: 16px; font-weight: 700; color: #18324a; }
        QFrame#moduleCard { background: white; border: 1px solid #d8dee5; border-radius: 10px; padding: 12px; }
        QPushButton { padding: 8px 12px; border-radius: 6px; background: #1f6feb; color: white; font-weight: 600; }
        QPushButton:hover { background: #1158c7; }
        QToolBar { background: #e9eef4; spacing: 6px; }
        """)

    def create_menu(self):
        menu = self.menuBar()
        file_menu = menu.addMenu("File")
        for plugin in self.plugins:
            act = QAction(f"Open {plugin['name']}", self)
            act.triggered.connect(lambda checked=False, p=plugin: start_plugin(p["id"]))
            file_menu.addAction(act)
        file_menu.addSeparator()
        exit_act = QAction("Exit", self)
        exit_act.triggered.connect(self.close)
        file_menu.addAction(exit_act)

        project_menu = menu.addMenu("Project")
        new_project = QAction("Create project folder", self)
        new_project.triggered.connect(self.create_project_folder)
        project_menu.addAction(new_project)

        help_menu = menu.addMenu("Help")
        manual = QAction("Open PDF manual", self)
        manual.triggered.connect(self.open_manual)
        help_menu.addAction(manual)
        help_page = QAction("Open local help page", self)
        help_page.triggered.connect(self.open_help_page)
        help_menu.addAction(help_page)
        about = QAction("About", self)
        about.triggered.connect(self.about)
        help_menu.addAction(about)

    def create_toolbar(self):
        toolbar = QToolBar("Main toolbar")
        self.addToolBar(toolbar)
        for plugin in self.plugins:
            act = QAction(plugin["name"], self)
            act.triggered.connect(lambda checked=False, p=plugin: start_plugin(p["id"]))
            toolbar.addAction(act)

    def create_content(self):
        central = QWidget()
        layout = QVBoxLayout(central)

        title = QLabel(APP_TITLE)
        title.setObjectName("mainTitle")
        subtitle = QLabel("Professional software for Raman, SERS, absorption and fluorescence spectro-electrochemical data.")
        subtitle.setObjectName("subtitle")
        subtitle.setWordWrap(True)

        layout.addWidget(title)
        layout.addWidget(subtitle)

        grid = QGridLayout()
        for i, plugin in enumerate(self.plugins):
            grid.addWidget(ModuleCard(plugin), i // 2, i % 2)
        layout.addLayout(grid)

        info = QTextEdit()
        info.setReadOnly(True)
        info.setMaximumHeight(140)
        info.setText(
            "Version 3.0 introduces a modern Qt/PySide6 main window, a plugin-ready architecture, "
            "project-folder creation, installer configuration, GitHub Actions, and documentation. "
            "The established scientific analysis modules are preserved."
        )
        layout.addWidget(info)
        self.setCentralWidget(central)

    def create_project_folder(self):
        base = QFileDialog.getExistingDirectory(self, "Choose location for project folder")
        if not base:
            return
        folder = Path(base) / "SpectroElectroChem_Project"
        folder.mkdir(exist_ok=True)
        for sub in ["data", "results", "figures", "exports", "notes"]:
            (folder / sub).mkdir(exist_ok=True)
        (folder / "project.json").write_text(json.dumps({"software": APP_TITLE, "version": self.version}, indent=2), encoding="utf-8")
        QMessageBox.information(self, "Project created", f"Project folder created:\n{folder}")

    def open_manual(self):
        manual = app_root() / "docs" / "User_Manual.pdf"
        if manual.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(manual)))
        else:
            QMessageBox.information(self, "Manual", "User_Manual.pdf was not found.")

    def open_help_page(self):
        page = app_root() / "docs" / "index.html"
        if page.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(page)))
        else:
            QMessageBox.information(self, "Help", "index.html was not found.")

    def about(self):
        QMessageBox.information(
            self, "About",
            f"{APP_TITLE} v{self.version}\n\nAuthor: Prof. Dr. Achim Habekost\n\n"
            "Parts of the source code were developed with the assistance of OpenAI ChatGPT "
            "and were subsequently validated, modified and extended by the author."
        )


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

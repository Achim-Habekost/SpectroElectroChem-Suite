from __future__ import annotations

import subprocess
import sys
import tkinter as tk
import webbrowser
from pathlib import Path
from tkinter import ttk, messagebox

from spectroelectrochem_suite import __version__
from spectroelectrochem_suite.plugins.plugin_manager import load_plugins
from spectroelectrochem_suite import updater


APP_TITLE = "SpectroElectroChem Suite"


def app_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def module_runner(plugin_id: str) -> Path:
    return app_root() / "run_plugin.py"


def start_plugin(plugin_id: str):
    runner = module_runner(plugin_id)
    if not runner.exists():
        # Development mode fallback
        runner = Path(__file__).resolve().parents[2] / "run_plugin.py"

    creationflags = subprocess.CREATE_NEW_CONSOLE if sys.platform.startswith("win") else 0
    subprocess.Popen(
        [sys.executable, str(runner), plugin_id],
        cwd=str(app_root()),
        creationflags=creationflags
    )


def open_local_manual():
    candidates = [
        app_root() / "docs" / "User_Manual.pdf",
        Path(__file__).resolve().parents[2] / "docs" / "User_Manual.pdf",
    ]
    for p in candidates:
        if p.exists():
            webbrowser.open(p.as_uri())
            return
    messagebox.showinfo("Manual", "User_Manual.pdf was not found in the docs folder.")


def open_online_help():
    webbrowser.open("https://github.com/YOUR_GITHUB_NAME/SpectroElectroChem-Suite")


class SpectroElectroChemApp:
    def __init__(self, root):
        self.root = root
        self.root.title(f"{APP_TITLE} v{__version__}")
        self.root.geometry("920x620")
        self.root.minsize(820, 560)
        self.plugins = load_plugins()

        self._style()
        self._menu()
        self._toolbar()
        self._content()
        self._status(f"Ready - {APP_TITLE} v{__version__}")

    def _style(self):
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Title.TLabel", font=("Segoe UI", 22, "bold"))
        style.configure("Subtitle.TLabel", font=("Segoe UI", 10))
        style.configure("Module.TButton", font=("Segoe UI", 10, "bold"), padding=12)
        style.configure("Card.TFrame", relief="solid", borderwidth=1)

    def _menu(self):
        menu = tk.Menu(self.root)

        file_menu = tk.Menu(menu, tearoff=False)
        for plugin in self.plugins:
            file_menu.add_command(label=f"Open {plugin.name}", command=lambda p=plugin: start_plugin(p.id))
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.destroy)
        menu.add_cascade(label="File", menu=file_menu)

        tools_menu = tk.Menu(menu, tearoff=False)
        tools_menu.add_command(label="Check for updates", command=self.check_updates)
        tools_menu.add_command(label="Open download page", command=updater.open_download_page)
        menu.add_cascade(label="Tools", menu=tools_menu)

        help_menu = tk.Menu(menu, tearoff=False)
        help_menu.add_command(label="User manual (PDF)", command=open_local_manual)
        help_menu.add_command(label="Online help / GitHub", command=open_online_help)
        help_menu.add_separator()
        help_menu.add_command(label="About", command=self.about)
        menu.add_cascade(label="Help", menu=help_menu)

        self.root.config(menu=menu)

    def _toolbar(self):
        bar = ttk.Frame(self.root, padding=(10, 8))
        bar.pack(fill="x")
        for plugin in self.plugins:
            ttk.Button(bar, text=plugin.name, command=lambda p=plugin: start_plugin(p.id)).pack(side="left", padx=4)
        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=8)
        ttk.Button(bar, text="Manual", command=open_local_manual).pack(side="left", padx=4)
        ttk.Button(bar, text="Updates", command=self.check_updates).pack(side="left", padx=4)

    def _content(self):
        outer = ttk.Frame(self.root, padding=20)
        outer.pack(fill="both", expand=True)

        ttk.Label(outer, text=APP_TITLE, style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            outer,
            text="Integrated software for Raman, SERS, absorption and fluorescence spectro-electrochemical data.",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(2, 18))

        grid = ttk.Frame(outer)
        grid.pack(fill="both", expand=True)

        for i, plugin in enumerate(self.plugins):
            card = ttk.Frame(grid, padding=16, style="Card.TFrame")
            card.grid(row=i // 2, column=i % 2, padx=10, pady=10, sticky="nsew")

            ttk.Label(card, text=plugin.name, font=("Segoe UI", 13, "bold")).pack(anchor="w")
            ttk.Label(card, text=plugin.description, wraplength=360, justify="left").pack(anchor="w", pady=(6, 14))
            ttk.Button(card, text="Open module", style="Module.TButton", command=lambda p=plugin: start_plugin(p.id)).pack(anchor="w")

        for c in range(2):
            grid.columnconfigure(c, weight=1)

        info = ttk.Label(
            outer,
            text="Plugin-ready architecture: future modules such as IR or UV/Vis spectro-electrochemistry can be added through the plugin registry.",
            foreground="gray",
            wraplength=820,
            justify="left"
        )
        info.pack(anchor="w", pady=(12, 0))

        self.status = ttk.Label(self.root, relief="sunken", anchor="w", padding=(8, 3))
        self.status.pack(fill="x", side="bottom")

    def _status(self, text):
        if hasattr(self, "status"):
            self.status.config(text=text)

    def check_updates(self):
        available, current, latest, msg = updater.check_for_updates()
        if available:
            if messagebox.askyesno("Update available", f"Current version: {current}\nLatest version: {latest}\n\nOpen download page?"):
                updater.open_download_page()
        else:
            messagebox.showinfo("Updates", str(msg))

    def about(self):
        messagebox.showinfo(
            "About",
            f"{APP_TITLE} v{__version__}\n\n"
            "Author: Prof. Dr. Achim Habekost\n\n"
            "Parts of the source code were developed with the assistance of OpenAI ChatGPT "
            "and were subsequently validated, modified and extended by the author."
        )


def main():
    root = tk.Tk()
    SpectroElectroChemApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

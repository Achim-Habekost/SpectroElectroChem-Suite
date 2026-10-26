from __future__ import annotations

import subprocess
import sys
import tkinter as tk
import webbrowser
from pathlib import Path
from tkinter import ttk, messagebox

from spectroelectrochem_suite import __version__
from spectroelectrochem_suite.plugins.plugin_manager import load_plugins, launch_plugin
from spectroelectrochem_suite import updater


APP_TITLE = "SpectroElectroChem Suite"


def app_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def module_runner(plugin_id: str) -> Path:
    return app_root() / "run_plugin.py"


def start_plugin(plugin_id: str):
    """Launch a module in a separate process in source and frozen builds."""
    if getattr(sys, "frozen", False):
        # A PyInstaller executable cannot interpret a .py file as Python.
        # Relaunch this executable in dedicated plugin mode instead.
        command = [sys.executable, "--plugin", plugin_id]
    else:
        runner = module_runner(plugin_id)
        command = [sys.executable, str(runner), plugin_id]

    creationflags = subprocess.CREATE_NEW_CONSOLE if sys.platform.startswith("win") else 0
    try:
        subprocess.Popen(command, cwd=str(app_root()), creationflags=creationflags)
    except OSError as exc:
        messagebox.showerror("Module start failed", str(exc))


def open_local_manual():
    candidates = [
        # PyInstaller onedir keeps bundled data under _internal (sys._MEIPASS).
        Path(getattr(sys, "_MEIPASS", app_root())) / "docs" / "User_Manual.pdf",
        app_root() / "_internal" / "docs" / "User_Manual.pdf",
        app_root() / "docs" / "User_Manual.pdf",
    ]
    if not getattr(sys, "frozen", False):
        candidates.append(Path(__file__).resolve().parents[2] / "docs" / "User_Manual.pdf")
    for p in candidates:
        if p.exists():
            webbrowser.open(p.as_uri())
            return
    messagebox.showinfo("Manual", "User_Manual.pdf was not found in the docs folder.")


def open_online_help():
    webbrowser.open("https://github.com/Achim-Habekost/SpectroElectroChem-Suite")


class SpectroElectroChemApp:
    def __init__(self, root):
        self.root = root
        self.root.title(f"{APP_TITLE} v{__version__}")
        self.root.geometry("1060x650")
        self.root.minsize(780, 520)
        self.plugins = load_plugins()

        self._style()
        self._menu()
        self._toolbar()
        self._content()
        self._status(f"Ready - {APP_TITLE} v{__version__}")

    def _style(self):
        self.root.configure(background="#f4f7fb")
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Main.TFrame", background="#f4f7fb")
        # Muted module-group colours: blue, green, violet, and orange.
        self.card_colors = (
            ("#E7F0FA", "#204F80"),
            ("#E7F4ED", "#246344"),
            ("#F0EAF8", "#694B8D"),
            ("#FFF1E2", "#945B23"),
        )
        for index, (background, foreground) in enumerate(self.card_colors):
            style.configure(f"Card{index}.TFrame", background=background, relief="solid", borderwidth=1)
            style.configure(f"CardTitle{index}.TLabel", background=background,
                            foreground=foreground, font=("Segoe UI", 10, "bold"))
            style.configure(f"CardDesc{index}.TLabel", background=background,
                            foreground="#435466", font=("Segoe UI", 8))
        style.configure("Title.TLabel", background="#f4f7fb", foreground="#15365a", font=("Segoe UI", 19, "bold"))
        style.configure("Subtitle.TLabel", background="#f4f7fb", foreground="#45586d", font=("Segoe UI", 9))
        style.configure("Module.TButton", font=("Segoe UI", 9, "bold"), padding=(9, 6),
                        background="#D6E6F8", foreground="#173F70", bordercolor="#9EB8D5")
        style.map("Module.TButton",
                  background=[("pressed", "#A9C7E8"), ("active", "#C2D9F2")],
                  foreground=[("disabled", "#8192A5")])
        style.configure("Toolbar.TFrame", background="#f4f7fb")

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
        # Modules are available in the compact grid; no overflowing button bar.
        bar = ttk.Frame(self.root, style="Toolbar.TFrame", padding=(12, 6))
        bar.pack(fill="x")
        ttk.Button(bar, text="User manual", command=open_local_manual).pack(side="right", padx=4)
        ttk.Button(bar, text="Updates", command=self.check_updates).pack(side="right", padx=4)

    def _content(self):
        outer = ttk.Frame(self.root, style="Main.TFrame", padding=(16, 8, 16, 12))
        outer.pack(fill="both", expand=True)

        ttk.Label(outer, text=APP_TITLE, style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            outer,
            text="Raman, SERS, absorption and fluorescence spectro-electrochemical data analysis",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(0, 12))

        # A scrollable grid is a fallback on small displays or large text scaling.
        holder = ttk.Frame(outer, style="Main.TFrame")
        holder.pack(fill="both", expand=True)
        canvas = tk.Canvas(holder, background="#f4f7fb", highlightthickness=0, borderwidth=0)
        scrollbar = ttk.Scrollbar(holder, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        grid = ttk.Frame(canvas, style="Main.TFrame")
        grid_window = canvas.create_window((0, 0), window=grid, anchor="nw")
        grid.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(grid_window, width=e.width))

        for i, plugin in enumerate(self.plugins):
            # Group related modules by their function, not their grid position.
            # 0: Raman/SERS, 1: absorption/CV, 2: ECL/EIS, 3: other analyses.
            group = (0, 0, 1, 3, 1, 2, 3, 2, 3, 0)[i] if i < 10 else 3
            card = ttk.Frame(grid, style=f"Card{group}.TFrame", padding=(12, 9))
            card.grid(row=i // 2, column=i % 2, padx=5, pady=5, sticky="nsew")
            card.columnconfigure(0, weight=1)
            ttk.Label(card, text=plugin.name, style=f"CardTitle{group}.TLabel", wraplength=310).grid(
                row=0, column=0, sticky="w", padx=(0, 8))
            ttk.Button(card, text="Open", style="Module.TButton",
                       command=lambda p=plugin: start_plugin(p.id)).grid(
                row=0, column=1, rowspan=2, sticky="e")
            ttk.Label(card, text=plugin.description, style=f"CardDesc{group}.TLabel",
                      wraplength=360, justify="left").grid(
                row=1, column=0, sticky="w", pady=(3, 0), padx=(0, 8))

        grid.columnconfigure(0, weight=1, uniform="modules")
        grid.columnconfigure(1, weight=1, uniform="modules")

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
    if len(sys.argv) >= 2 and sys.argv[1] == "--plugin":
        if len(sys.argv) != 3:
            raise SystemExit("Usage: SpectroElectroChem_Suite.exe --plugin <plugin_id>")
        plugin_id = sys.argv[2]
        plugins = {plugin.id: plugin for plugin in load_plugins()}
        if plugin_id not in plugins:
            raise SystemExit(f"Unknown plugin: {plugin_id}")
        launch_plugin(plugins[plugin_id])
    else:
        main()

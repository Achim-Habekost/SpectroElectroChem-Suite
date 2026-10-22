from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "--plugin":
        from PySide6.QtCore import Qt

        Qt.KeyboardModifier.__int__ = lambda self: self.value
        Qt.ToolBarArea.__int__ = lambda self: self.value

        from spectroelectrochem_suite.plugins.plugin_manager import (
            load_plugins,
            launch_plugin,
        )

        plugin_id = sys.argv[2]
        plugins = {p.id: p for p in load_plugins()}

        if plugin_id not in plugins:
            raise SystemExit("Unknown plugin: " + plugin_id)

        launch_plugin(plugins[plugin_id])

    else:
        from spectroelectrochem_suite.modern_app import main as start_gui
        start_gui()


if __name__ == "__main__":
    main()
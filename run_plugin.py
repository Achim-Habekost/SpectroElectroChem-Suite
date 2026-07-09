from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spectroelectrochem_suite.plugins.plugin_manager import load_plugins, launch_plugin


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python run_plugin.py <plugin_id>")

    plugin_id = sys.argv[1]
    plugins = {p.id: p for p in load_plugins()}

    if plugin_id not in plugins:
        raise SystemExit(f"Unknown plugin: {plugin_id}")

    launch_plugin(plugins[plugin_id])


if __name__ == "__main__":
    main()

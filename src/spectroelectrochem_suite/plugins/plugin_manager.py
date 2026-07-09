from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class Plugin:
    id: str
    name: str
    description: str
    module: str


def plugin_registry_path() -> Path:
    return Path(__file__).resolve().parent / "plugins.json"


def load_plugins() -> List[Plugin]:
    data = json.loads(plugin_registry_path().read_text(encoding="utf-8"))
    return [Plugin(**item) for item in data.get("plugins", [])]


def launch_plugin(plugin: Plugin) -> None:
    mod = importlib.import_module(plugin.module)
    if hasattr(mod, "main"):
        mod.main()
    else:
        raise RuntimeError(f"Plugin {plugin.name} has no main() function.")

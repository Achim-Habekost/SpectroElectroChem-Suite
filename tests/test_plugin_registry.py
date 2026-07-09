import json
from pathlib import Path


def test_plugin_registry_exists():
    path = Path("src/spectroelectrochem_suite/plugins/plugins.json")
    assert path.exists()


def test_plugin_registry_contains_plugins():
    data = json.loads(Path("src/spectroelectrochem_suite/plugins/plugins.json").read_text(encoding="utf-8"))
    assert "plugins" in data
    assert len(data["plugins"]) >= 3
    for plugin in data["plugins"]:
        assert "id" in plugin
        assert "name" in plugin
        assert "module" in plugin

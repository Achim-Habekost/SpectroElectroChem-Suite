import json
from pathlib import Path


PLUGIN_REGISTRY = Path(
    "src/spectroelectrochem_suite/plugins/plugins.json"
)


def test_plugin_registry_exists():
    assert PLUGIN_REGISTRY.exists()


def test_plugin_registry_contains_plugins():
    data = json.loads(PLUGIN_REGISTRY.read_text(encoding="utf-8"))

    assert "plugins" in data
    assert len(data["plugins"]) >= 4

    for plugin in data["plugins"]:
        assert "id" in plugin
        assert "name" in plugin
        assert "module" in plugin


def test_rrde_plugin_registered():
    data = json.loads(PLUGIN_REGISTRY.read_text(encoding="utf-8"))

    plugin_ids = {plugin["id"] for plugin in data["plugins"]}

    assert "rrde_analysis" in plugin_ids
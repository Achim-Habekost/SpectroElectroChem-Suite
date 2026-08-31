from __future__ import annotations

import json
import webbrowser
from urllib.request import urlopen
from urllib.error import URLError, HTTPError

from . import __version__


# Official GitHub release endpoints.
GITHUB_RELEASES_API = "https://api.github.com/repos/Achim-Habekost/SpectroElectroChem-Suite/releases/latest"
GITHUB_RELEASES_PAGE = "https://github.com/Achim-Habekost/SpectroElectroChem-Suite/releases"


def get_latest_release(timeout: int = 5):
    try:
        with urlopen(GITHUB_RELEASES_API, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (URLError, HTTPError, TimeoutError, json.JSONDecodeError):
        return None


def check_for_updates():
    release = get_latest_release()
    if release is None:
        return False, __version__, None, "Could not contact the release server."

    tag = str(release.get("tag_name", "")).lstrip("v")
    if tag and tag != __version__:
        return True, __version__, tag, release.get("html_url", GITHUB_RELEASES_PAGE)

    return False, __version__, tag or __version__, "You are using the current version."


def open_download_page():
    webbrowser.open(GITHUB_RELEASES_PAGE)

from __future__ import annotations

import json
import webbrowser
from urllib.request import Request, urlopen
import ssl
import sys
from urllib.error import URLError, HTTPError

from . import __version__


# Official GitHub release endpoints.
GITHUB_RELEASES_API = "https://api.github.com/repos/Achim-Habekost/SpectroElectroChem-Suite/releases/latest"
GITHUB_RELEASES_PAGE = "https://github.com/Achim-Habekost/SpectroElectroChem-Suite/releases"


def get_latest_release(timeout: int = 10):
    """Return (release, error_message); preserve the actual network error."""
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "SpectroElectroChem-Suite-update-checker",
    }
    try:
        # certifi is packaged with the app's Python dependencies when available.
        # It helps frozen applications that cannot locate a CA certificate store.
        try:
            import certifi
            context = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            context = ssl.create_default_context()
        request = Request(GITHUB_RELEASES_API, headers=headers)
        with urlopen(request, timeout=timeout, context=context) as response:
            return json.loads(response.read().decode("utf-8")), None
    except (URLError, HTTPError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
        return None, str(exc)


def _version_tuple(version):
    """Compare simple release versions such as v6.3.0 and 6.2.0."""
    import re
    parts = re.findall(r"\d+", str(version).lstrip("vV").split("-")[0])
    return tuple(int(part) for part in parts) if parts else ()


def check_for_updates():
    release, error = get_latest_release()
    if release is None:
        return False, __version__, None, (
            "Update check failed: " + (error or "Unknown network error")
            + "\nYou can check releases manually under Tools > Open download page."
        )

    tag = str(release.get("tag_name", "")).lstrip("vV")
    if not tag:
        return False, __version__, None, "The release server returned no version tag."
    if _version_tuple(tag) > _version_tuple(__version__):
        return True, __version__, tag, release.get("html_url", GITHUB_RELEASES_PAGE)
    return False, __version__, tag, "You are using the current version or a newer development version."


def open_download_page():
    webbrowser.open(GITHUB_RELEASES_PAGE)

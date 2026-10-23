"""Run Dwarfium's browser edition next to this app, so the Telescope tab can show it.

Dwarfium (https://github.com/stevejcl/dwarfium, MIT) replicates the DWARFLAB app:
goto, tracking, joystick, focus, calibration, live view, live stacking and
schedules. Its standalone Windows package serves the full interface at
http://127.0.0.1:8000, which the Telescope tab embeds.

Installing downloads Dwarfium-Win.zip from the project's official release page
(linked from the Dwarfium README) and only extracts it if its SHA-256 matches
the digest GitHub publishes for that release asset.
"""
from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

from . import config

RELEASE_REPO = "acocalypso/dwarfium"  # where the Dwarfium README's Windows downloads are published
ASSET_NAME = "Dwarfium-Win.zip"
DEFAULT_URL = "http://127.0.0.1:8000"
ALLOWED_HOSTS = {"api.github.com", "github.com", "objects.githubusercontent.com",
                 "release-assets.githubusercontent.com"}
MAX_BYTES = 800 * 1024 * 1024


class DwarfiumError(Exception):
    pass


def web_url() -> str:
    return (config.load().get("dwarfium_web_url") or DEFAULT_URL).rstrip("/")


def install_root() -> Path:
    d = config.data_dir() / "dwarfium"
    d.mkdir(exist_ok=True)
    return d


def installed() -> dict | None:
    custom = config.load().get("dwarfium_standalone_dir")
    candidates = [Path(custom)] if custom else sorted(install_root().glob("*/"), reverse=True)
    for d in candidates:
        launcher = next(iter(d.rglob("launch-server&tools.bat")), None) if d.is_dir() else None
        if launcher:
            return {"folder": str(launcher.parent), "version": d.name, "launcher": str(launcher)}
    return None


def running() -> bool:
    try:
        with urllib.request.urlopen(web_url() + "/", timeout=2) as r:
            return r.status == 200
    except OSError:
        return False


def status() -> dict:
    return {"url": web_url(), "running": running(), "installed": installed()}


def _get(url: str, limit: int) -> bytes:
    parts = urllib.parse.urlparse(url)
    if parts.scheme != "https" or parts.hostname not in ALLOWED_HOSTS:
        raise DwarfiumError("Dwarfium is only downloaded over HTTPS from GitHub.")
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json",
                                               "User-Agent": "DWARF-Overview"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            final = urllib.parse.urlparse(r.geturl())
            if final.hostname not in ALLOWED_HOSTS:
                raise DwarfiumError("The download was redirected away from GitHub; stopped.")
            data = r.read(limit + 1)
    except OSError as e:
        raise DwarfiumError(f"Download failed ({e}).")
    if len(data) > limit:
        raise DwarfiumError("The download is unexpectedly large; stopped.")
    return data


def latest_release() -> dict:
    releases = json.loads(_get(f"https://api.github.com/repos/{RELEASE_REPO}/releases?per_page=15", 2_000_000))
    for rel in releases:
        if rel.get("prerelease") or rel.get("draft"):
            continue
        for asset in rel.get("assets", []):
            if asset.get("name") == ASSET_NAME:
                return {"tag": rel["tag_name"], "url": asset["browser_download_url"], "size": asset.get("size"),
                        "digest": asset.get("digest") or "", "published": rel.get("published_at")}
    raise DwarfiumError(f"No {ASSET_NAME} found in the latest Dwarfium releases.")


def install() -> dict:
    if sys.platform != "win32":
        raise DwarfiumError("The one-click install is for Windows. On other systems use the Dwarfium desktop app.")
    rel = latest_release()
    if not rel["digest"].startswith("sha256:"):
        raise DwarfiumError("GitHub didn't publish a checksum for this download, so it wasn't installed. "
                            "Install Dwarfium manually from its release page instead.")
    data = _get(rel["url"], MAX_BYTES)
    if hashlib.sha256(data).hexdigest() != rel["digest"].split(":", 1)[1].lower():
        raise DwarfiumError("The download doesn't match GitHub's checksum, so it wasn't installed.")
    dest = install_root() / rel["tag"]
    executables = []
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for info in z.infolist():
            name = PurePosixPath(info.filename)
            if name.is_absolute() or ".." in name.parts or ":" in info.filename:
                raise DwarfiumError(f"Unexpected path in the Dwarfium package: {info.filename}")
            if name.suffix.lower() in (".exe", ".dll", ".bat", ".cmd", ".ps1"):
                executables.append(info.filename)
        z.extractall(dest)
    if not installed():
        raise DwarfiumError("The package didn't contain Dwarfium's launcher.")
    return {"version": rel["tag"], "folder": str(dest), "sha256": rel["digest"], "programs": executables}


def start() -> dict:
    inst = installed()
    if not inst:
        raise DwarfiumError("Dwarfium's browser edition isn't installed yet.")
    if running():
        return {"already_running": True}
    launcher = Path(inst["launcher"])
    if sys.platform == "win32":
        # The launcher's name contains "&", so it must stay inside quotes for cmd.
        subprocess.Popen(f'cmd /c ""{launcher}""', cwd=str(launcher.parent),
                         creationflags=subprocess.CREATE_NEW_CONSOLE)
    else:
        subprocess.Popen(["sh", str(launcher.with_suffix(".sh"))], cwd=str(launcher.parent), start_new_session=True)
    return {"started": True}

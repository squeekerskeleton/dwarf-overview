"""Run Dwarfium's browser edition next to this app, so the Telescope tab can show it.

Dwarfium (https://github.com/stevejcl/dwarfium, MIT) replicates the DWARFLAB app:
goto, tracking, joystick, focus, calibration, live view, live stacking and
schedules. Its standalone Windows package serves the full interface at
http://127.0.0.1:8000, which the Telescope tab embeds.

Installing downloads Dwarfium-Win.zip from the project's official release page
(linked from the Dwarfium README) and only extracts it if its SHA-256 matches
the digest GitHub publishes for that release asset.

Autofill: Dwarfium keeps its connection settings in its own page storage, which
this app can't write to directly. So a tiny page is placed next to Dwarfium's
files; opened with the DWARF's address, site and time zone after the "#", it
saves them where Dwarfium looks on start-up and opens Dwarfium's connect page.
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
HELPER_NAME = "dwarf-overview-connect.html"
HELPER_HTML = """<!doctype html>
<meta charset="utf-8"><title>Connecting to your DWARF</title>
<body style="background:#000;color:#ccc;font:14px system-ui">Filling in your DWARF's details&hellip;
<script>
(function () {
  var p = new URLSearchParams(location.hash.slice(1)), num = function (k, lim) {
    var v = parseFloat(p.get(k)); return isFinite(v) && Math.abs(v) <= lim ? String(v) : null; };
  var ip = p.get("ip") || "", m = /^(\\d{1,3})\\.(\\d{1,3})\\.(\\d{1,3})\\.(\\d{1,3})$/.exec(ip);
  var ok = m && m.slice(1).every(function (o) { return +o < 256; }) &&
    (m[1] == "10" || (m[1] == "192" && m[2] == "168") || (m[1] == "172" && +m[2] >= 16 && +m[2] <= 31) || (m[1] == "169" && m[2] == "254"));
  if (ok) localStorage.setItem("IPDwarf", ip);
  var lat = num("lat", 90), lon = num("lon", 180), tz = p.get("tz") || "";
  if (lat !== null && lon !== null) { localStorage.setItem("latitude", lat); localStorage.setItem("longitude", lon); }
  if (/^[A-Za-z_]+(\\/[A-Za-z0-9_+-]+)*$/.test(tz)) localStorage.setItem("timezone", tz);
  location.replace("/setup-scope");
})();
</script>
"""


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


def _web_roots() -> list[Path]:
    """Folders Dwarfium's web server serves (the ones holding its index.html and _next)."""
    inst = installed()
    if not inst:
        return []
    base = Path(inst["folder"])
    return [p.parent for p in base.rglob("index.html") if (p.parent / "_next").is_dir()]


def prepare_autofill() -> bool:
    roots = _web_roots()
    for root in roots:
        target = root / HELPER_NAME
        if not target.exists() or target.read_text(encoding="utf-8") != HELPER_HTML:
            target.write_text(HELPER_HTML, encoding="utf-8")
    return bool(roots)


def start() -> dict:
    inst = installed()
    if not inst:
        raise DwarfiumError("Dwarfium's browser edition isn't installed yet.")
    if running():
        return {"already_running": True}
    prepare_autofill()
    launcher = Path(inst["launcher"])
    if sys.platform == "win32":
        # The launcher's name contains "&", so it must stay inside quotes for cmd.
        subprocess.Popen(f'cmd /c ""{launcher}""', cwd=str(launcher.parent),
                         creationflags=subprocess.CREATE_NEW_CONSOLE)
    else:
        subprocess.Popen(["sh", str(launcher.with_suffix(".sh"))], cwd=str(launcher.parent), start_new_session=True)
    return {"started": True}

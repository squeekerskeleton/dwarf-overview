"""Built-in updater.

New versions are published as a zip plus a small manifest (latest.json) in a
GitHub repository. The app reads the manifest, shows what changed, and only
installs when you click Install. Before anything is replaced it checks:
  - the download comes over HTTPS from GitHub,
  - its SHA-256 matches the manifest,
  - every file in the zip is an expected type and stays inside the app folder.
The current version is backed up first, so it can be restored with one click.
Your settings, API key and history live elsewhere and are never touched.
"""
from __future__ import annotations

import hashlib
import io
import json
import shutil
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path, PurePosixPath

from . import __version__, config

# Filled in once the update repository exists; can also be set in Settings.
DEFAULT_UPDATE_URL = "https://raw.githubusercontent.com/squeekerskeleton/dwarf-overview/main/latest.json"

APP_DIR = Path(__file__).resolve().parent.parent
ZIP_PREFIX = "DWARF Overview/"
ALLOWED_HOSTS = {"github.com", "raw.githubusercontent.com", "objects.githubusercontent.com",
                 "codeload.github.com", "release-assets.githubusercontent.com"}
ALLOWED_EXTS = {".py", ".html", ".css", ".js", ".json", ".txt", ".md", ".bat", ".command", ".sh", ".png", ".ico", ".svg"}
MAX_BYTES = 25 * 1024 * 1024
KEEP_BACKUPS = 5


class UpdateError(Exception):
    pass


def _vtuple(v: str) -> tuple:
    return tuple(int(p) if p.isdigit() else 0 for p in v.lstrip("v").split("."))


def _manifest_url() -> str:
    return (config.load().get("update_url") or DEFAULT_UPDATE_URL).strip()


def _fetch(url: str, limit: int) -> bytes:
    parts = urllib.parse.urlparse(url)
    if parts.scheme != "https" or parts.hostname not in ALLOWED_HOSTS:
        raise UpdateError("Updates are only downloaded over HTTPS from GitHub.")
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            # Redirects (e.g. to GitHub's asset storage) must stay on GitHub too.
            final = urllib.parse.urlparse(r.geturl())
            if final.scheme != "https" or final.hostname not in ALLOWED_HOSTS:
                raise UpdateError("The update download was redirected away from GitHub; stopped.")
            data = r.read(limit + 1)
    except OSError as e:
        raise UpdateError(f"Couldn't download the update ({e}).")
    if len(data) > limit:
        raise UpdateError("The update file is unexpectedly large; stopped.")
    return data


def check() -> dict:
    url = _manifest_url()
    if not url:
        return {"current": __version__, "configured": False}
    try:
        manifest = json.loads(_fetch(url, 256 * 1024))
    except ValueError:
        raise UpdateError("The update information couldn't be read.")
    latest = str(manifest.get("version", ""))
    return {"current": __version__, "configured": True, "latest": latest,
            "available": bool(latest) and _vtuple(latest) > _vtuple(__version__),
            "notes": manifest.get("notes", ""), "manifest": manifest}


def _backup_dir() -> Path:
    d = config.data_dir() / "backups"
    d.mkdir(exist_ok=True)
    return d


def _app_files():
    for p in APP_DIR.rglob("*"):
        rel = p.relative_to(APP_DIR)
        if p.is_file() and rel.parts[0] not in (".venv", ".git") and "__pycache__" not in rel.parts:
            yield p, rel


def _backup() -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = _backup_dir() / f"DWARF-Overview-{__version__}-{stamp}.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for p, rel in _app_files():
            z.write(p, ZIP_PREFIX + rel.as_posix())
    old = sorted(_backup_dir().glob("DWARF-Overview-*.zip"), key=lambda p: p.stat().st_mtime)
    for extra in old[:-KEEP_BACKUPS]:
        extra.unlink(missing_ok=True)
    return path


def _validated_members(z: zipfile.ZipFile) -> list[tuple[zipfile.ZipInfo, PurePosixPath]]:
    members = []
    for info in z.infolist():
        if info.is_dir():
            continue
        name = PurePosixPath(info.filename)
        if not info.filename.startswith(ZIP_PREFIX) or ".." in name.parts or name.is_absolute():
            raise UpdateError(f"Unexpected path in update: {info.filename}")
        rel = PurePosixPath(*name.parts[1:])
        if rel.parts and rel.parts[0] in (".venv", ".git"):
            raise UpdateError(f"Unexpected path in update: {info.filename}")
        if rel.suffix.lower() not in ALLOWED_EXTS:
            raise UpdateError(f"Unexpected file type in update: {rel}")
        members.append((info, rel))
    if not any(str(rel) == "dwarf_overview/__init__.py" for _, rel in members):
        raise UpdateError("The update doesn't look like DWARF Overview; stopped.")
    return members


def _install_zip(data: bytes) -> list[str]:
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        members = _validated_members(z)
        staging = config.data_dir() / "update-staging"
        shutil.rmtree(staging, ignore_errors=True)
        for info, rel in members:
            out = staging / Path(*rel.parts)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(z.read(info))
    # Replace the app's code; anything the new version no longer ships is removed.
    shutil.rmtree(APP_DIR / "dwarf_overview", ignore_errors=True)
    for p in staging.rglob("*"):
        if p.is_file():
            dest = APP_DIR / p.relative_to(staging)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
    shutil.rmtree(staging, ignore_errors=True)
    return [str(rel) for _, rel in members]


def install() -> dict:
    info = check()
    if not info.get("available"):
        raise UpdateError("You already have the latest version.")
    m = info["manifest"]
    expected = str(m.get("sha256", "")).lower()
    if len(expected) != 64:
        raise UpdateError("The update has no checksum, so it won't be installed.")
    data = _fetch(str(m.get("zip_url", "")), MAX_BYTES)
    if hashlib.sha256(data).hexdigest() != expected:
        raise UpdateError("The download doesn't match its checksum, so it wasn't installed.")
    backup = _backup()
    files = _install_zip(data)
    return {"installed": info["latest"], "backup": str(backup), "files": len(files)}


def backups() -> list[str]:
    return [p.name for p in sorted(_backup_dir().glob("DWARF-Overview-*.zip"), key=lambda p: p.stat().st_mtime,
                                   reverse=True)]


def restore(name: str) -> dict:
    path = _backup_dir() / Path(name).name
    if not path.is_file():
        raise UpdateError("That backup wasn't found.")
    data = path.read_bytes()
    _backup()
    _install_zip(data)
    return {"restored": name}

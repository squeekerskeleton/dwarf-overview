"""Pull sessions straight off the DWARF 3 over Wi-Fi.

The telescope serves its storage as a plain web directory listing at
http://<dwarf-ip>/DWARF3/Astronomy/ (this is how Dwarfium's session browser
reads it too). Each session folder holds shotsInfo.json, the stacked FITS and
JPG, and optionally the individual subs. Files are copied into a local library
folder and never deleted from the telescope.
"""
from __future__ import annotations

import ipaddress
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

from . import config

ROOTS = {"astro": "/DWARF3/Astronomy/", "dwarf2": "/sdcard/DWARF_II/Astronomy/"}
SKIP_FOLDERS = re.compile(r"dwarf_dark|solving_failed|cali_frame", re.I)
WANTED = re.compile(r"(shotsInfo\.json|stacked.*\.(fits?|tiff?|jpe?g|png)|.*\.(ser|mp4|avi))$", re.I)
LINK = re.compile(r'href="([^"?#]+)"[^>]*>([^<]+)</a>\s*(\d{2}-[A-Za-z]{3}-\d{4} \d{2}:\d{2})?\s*(\S+)?', re.I)


class DwarfError(Exception):
    pass


def _base() -> str:
    ip = (config.load().get("dwarf_ip") or "").strip()
    if not ip:
        raise DwarfError("Set the DWARF's IP address in Settings first.")
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        raise DwarfError("The DWARF address must be an IP address like 192.168.1.42.")
    # Only ever talk to a device on the local network.
    if not (addr.is_private or addr.is_link_local):
        raise DwarfError("That IP isn't on a local network. Use the address the DWARFLAB app or Dwarfium shows.")
    return f"http://{ip}"


def _get(url: str, timeout: float = 10) -> bytes:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.read()
    except OSError as e:
        raise DwarfError(f"Couldn't reach the DWARF ({e}). Is the laptop on the same Wi-Fi as the telescope?")


def _listing(path: str) -> list[dict]:
    html = _get(_base() + urllib.parse.quote(path)).decode("utf-8", "replace")
    out = []
    for href, text, date, size in LINK.findall(html):
        name = urllib.parse.unquote(href).rstrip("/")
        if name in ("..", ".") or name.startswith("/") or "/" in name or "\\" in name:
            continue
        out.append({"name": name, "dir": href.endswith("/"), "date": date or "", "size": size or ""})
    return out


def list_sessions() -> list[dict]:
    last_error = None
    for root in ROOTS.values():
        try:
            entries = _listing(root)
        except DwarfError as e:
            last_error = e
            continue
        sessions = [{"name": e["name"], "date": e["date"], "root": root} for e in entries
                    if e["dir"] and not SKIP_FOLDERS.search(e["name"])]
        sessions.sort(key=lambda s: s["name"], reverse=True)
        return sessions
    raise last_error or DwarfError("No astronomy folder found on the DWARF.")


def session_files(root: str, name: str) -> dict:
    if root not in ROOTS.values() or "/" in name or ".." in name:
        raise DwarfError("Bad session name.")
    entries = [e for e in _listing(f"{root}{name}/") if not e["dir"]]
    info = {}
    if any(e["name"] == "shotsInfo.json" for e in entries):
        try:
            info = json.loads(_get(_base() + urllib.parse.quote(f"{root}{name}/shotsInfo.json")))
        except (DwarfError, ValueError):
            info = {}
    return {"files": entries, "shot_info": info,
            "suggested": [e["name"] for e in entries if WANTED.match(e["name"])]}


def library_dir() -> Path:
    custom = config.load().get("library_folder")
    d = Path(custom).expanduser() if custom else Path.home() / "Pictures" / "DWARF Library"
    d.mkdir(parents=True, exist_ok=True)
    return d


def download(root: str, name: str, files: list[str]) -> dict:
    if root not in ROOTS.values() or "/" in name or ".." in name:
        raise DwarfError("Bad session name.")
    dest = library_dir() / re.sub(r'[<>:"|?*\\/]', "_", name)
    dest.mkdir(parents=True, exist_ok=True)
    saved = []
    for f in files:
        if "/" in f or "\\" in f or f in ("..", "."):
            continue
        target = dest / f
        tmp = target.with_suffix(target.suffix + ".part")
        try:
            with urllib.request.urlopen(_base() + urllib.parse.quote(f"{root}{name}/{f}"), timeout=30) as r, \
                    open(tmp, "wb") as out:
                while chunk := r.read(1 << 20):
                    out.write(chunk)
        except OSError as e:
            tmp.unlink(missing_ok=True)
            raise DwarfError(f"Download of {f} failed: {e}")
        tmp.replace(target)
        saved.append(str(target))
    return {"folder": str(dest), "saved": saved}

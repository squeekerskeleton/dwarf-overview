"""Projects: folders for finished photos, with notes and an overview of each photo.

Each project is a real folder (by default in Pictures\\DWARF Projects) that
finished photos are copied into. Project notes, photo notes and each photo's
overview are kept in the app's history database.

A photo's overview comes from its own file plus the DWARF's shotsInfo.json in
the session folder it came from: where it was taken (site, and the sky
coordinates), when, its size, how many frames were stacked, and how long the
capture took (first to last sub, or frames times exposure when the subs
aren't there).
"""
from __future__ import annotations

import json
import re
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path

from . import config, history

SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (id INTEGER PRIMARY KEY, name TEXT NOT NULL, folder TEXT NOT NULL,
  notes TEXT DEFAULT '', created TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS project_photos (id INTEGER PRIMARY KEY, project_id INTEGER NOT NULL, file TEXT NOT NULL,
  source TEXT, title TEXT DEFAULT '', notes TEXT DEFAULT '', overview TEXT, preview_b64 TEXT, added TEXT NOT NULL);
"""


class ProjectError(Exception):
    pass


def _db() -> sqlite3.Connection:
    con = history._conn()
    con.executescript(SCHEMA)
    return con


def root() -> Path:
    custom = config.load().get("projects_folder")
    d = Path(custom).expanduser() if custom else Path.home() / "Pictures" / "DWARF Projects"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _safe(name: str) -> str:
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(" .")
    if not name:
        raise ProjectError("Give the project a name.")
    return name[:80]


def list_projects() -> list[dict]:
    with _db() as con:
        rows = con.execute("SELECT p.id, p.name, p.folder, p.created, COUNT(f.id) FROM projects p "
                           "LEFT JOIN project_photos f ON f.project_id = p.id GROUP BY p.id ORDER BY p.created DESC").fetchall()
    return [{"id": r[0], "name": r[1], "folder": r[2], "created": r[3], "photos": r[4]} for r in rows]


def create(name: str) -> dict:
    name = _safe(name)
    folder = root() / name
    folder.mkdir(parents=True, exist_ok=True)
    with _db() as con:
        cur = con.execute("INSERT INTO projects (name, folder, created) VALUES (?, ?, ?)",
                          (name, str(folder), datetime.now().isoformat(timespec="seconds")))
    return get(cur.lastrowid)


def get(pid: int) -> dict:
    with _db() as con:
        p = con.execute("SELECT id, name, folder, notes, created FROM projects WHERE id = ?", (pid,)).fetchone()
        if not p:
            raise ProjectError("That project wasn't found.")
        photos = con.execute("SELECT id, file, source, title, notes, overview, preview_b64, added FROM project_photos "
                             "WHERE project_id = ? ORDER BY added", (pid,)).fetchall()
    return {"id": p[0], "name": p[1], "folder": p[2], "notes": p[3], "created": p[4],
            "photos": [{"id": r[0], "file": r[1], "source": r[2], "title": r[3], "notes": r[4],
                        "overview": json.loads(r[5] or "{}"), "preview_b64": r[6], "added": r[7],
                        "exists": Path(r[1]).exists()} for r in photos]}


def save_notes(pid: int, notes: str) -> dict:
    with _db() as con:
        con.execute("UPDATE projects SET notes = ? WHERE id = ?", (notes, pid))
    return {"saved": True}


def save_photo(photo_id: int, title: str | None, notes: str | None) -> dict:
    with _db() as con:
        if title is not None:
            con.execute("UPDATE project_photos SET title = ? WHERE id = ?", (title, photo_id))
        if notes is not None:
            con.execute("UPDATE project_photos SET notes = ? WHERE id = ?", (notes, photo_id))
    return {"saved": True}


def remove_photo(photo_id: int) -> dict:
    """Takes the photo off the project's list; the copied file stays in the folder."""
    with _db() as con:
        con.execute("DELETE FROM project_photos WHERE id = ?", (photo_id,))
    return {"removed": True}


# ---------------------------------------------------------------- the photo overview

def _shot_info(folder: Path) -> dict:
    for name in ("shotsInfo.json", "shotsinfo.json"):
        try:
            return json.loads((folder / name).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
    return {}


def _pick(info: dict, *keys):
    lower = {k.lower(): v for k, v in info.items()}
    for k in keys:
        v = lower.get(k.lower())
        if v not in (None, ""):
            return v
    return None


def _num(v) -> float | None:
    try:
        return float(str(v).rstrip("sS"))
    except (TypeError, ValueError):
        return None


def _when_from_name(name: str) -> datetime | None:
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})", name)
    if m:
        try:
            return datetime(*map(int, m.groups()))
        except ValueError:
            return None
    return None


def _stamp(name: str) -> datetime | None:
    m = re.search(r"(\d{8})-(\d{6})(\d{3})?", name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    except ValueError:
        return None


def _fits_date_obs(p: Path) -> datetime | None:
    try:
        from astropy.io import fits
        v = fits.getheader(str(p)).get("DATE-OBS")
        return datetime.fromisoformat(str(v)[:19]) if v else None
    except Exception:
        return None


def _dimensions(p: Path) -> tuple[int, int] | None:
    """Width and height from the file's header, without loading the pixels."""
    try:
        if p.suffix.lower() in (".fits", ".fit", ".fts"):
            from astropy.io import fits
            h = fits.getheader(str(p))
            return int(h["NAXIS1"]), int(h["NAXIS2"])
        if p.suffix.lower() in (".tif", ".tiff"):
            import tifffile
            with tifffile.TiffFile(str(p)) as tf:
                shape = tf.pages[0].shape
            return (int(shape[1]), int(shape[0])) if shape[0] > 4 else (int(shape[2]), int(shape[1]))
        from PIL import Image
        with Image.open(p) as im:
            return im.size
    except Exception:
        return None


def overview(file: str, source_folder: str | None = None) -> dict:
    p = Path(file)
    src = Path(source_folder) if source_folder else p.parent
    # A photo already copied into a project carries its session's info alongside as <name>.shotsInfo.json.
    try:
        info = json.loads(p.with_name(p.stem + ".shotsInfo.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        info = _shot_info(src)
    out = {"file_size_mb": round(p.stat().st_size / 1e6, 1) if p.exists() else None}
    dims = _dimensions(p)
    if dims:
        out["width"], out["height"] = dims
    exp = _num(_pick(info, "exp", "exposure", "expTime"))
    frames = _num(_pick(info, "shotsStacked", "stackedCount", "shotsTaken", "frames"))
    out.update({
        "target": _pick(info, "target", "targetName", "object"),
        "exposure_s": exp, "gain": _pick(info, "gain"), "filter": _pick(info, "ir", "filter", "irName"),
        "frames_stacked": int(frames) if frames else None,
        "frames_taken": _pick(info, "shotsTaken"),
        "ra": _pick(info, "ra", "RA"), "dec": _pick(info, "dec", "DEC"),
    })
    # Where: the site in Settings (the DWARF doesn't record GPS in shotsInfo), plus sky coordinates.
    cfg = config.load()
    if cfg.get("site_lat") not in ("", None) and cfg.get("site_lon") not in ("", None):
        out["site"] = {"lat": cfg["site_lat"], "lon": cfg["site_lon"], "name": cfg.get("site_name") or ""}
    lat, lon = _pick(info, "latitude", "lat"), _pick(info, "longitude", "lon")
    if lat is not None and lon is not None:
        out["site"] = {"lat": lat, "lon": lon, "name": cfg.get("site_name") or ""}
    # When, and how long. The DWARF names files with their start time (..._20261008-001207479...) and
    # writes DATE-OBS (end of the last exposure) into the stacked FITS; subs carry their own times too.
    start = _stamp(p.name) or _when_from_name(src.name)
    end = _fits_date_obs(p) if p.suffix.lower() in (".fits", ".fit", ".fts") else None
    sub_times = sorted(t for t in (_stamp(f.name) for f in src.glob("*.fit*")
                                   if not f.name.lower().startswith("stacked")) if t)
    if sub_times:
        start = min(start, sub_times[0]) if start else sub_times[0]
        end = max(end, sub_times[-1]) if end else sub_times[-1]
    when = start
    if start and end and end > start:
        out["elapsed_min"] = round((end - start).total_seconds() / 60, 1)
        out["elapsed_source"] = "first exposure to last"
    elif exp and frames:
        out["elapsed_min"] = round(exp * frames / 60, 1)
        out["elapsed_source"] = "frames x exposure (integration time)"
    if exp and frames:
        out["integration_min"] = round(exp * frames / 60, 1)
    if not when and p.exists():
        when = datetime.fromtimestamp(p.stat().st_mtime)
    out["date"] = when.isoformat(timespec="minutes") if when else None
    return out


def add_photo(pid: int, file: str, title: str = "", preview_b64: str | None = None, copy: bool = True) -> dict:
    proj = get(pid)
    src = Path(file)
    if not src.is_file():
        raise ProjectError("That file wasn't found.")
    ov = overview(str(src))
    dest = src
    if copy:
        dest = Path(proj["folder"]) / src.name
        if dest.exists() and dest.resolve() != src.resolve():
            dest = dest.with_name(f"{dest.stem}_{datetime.now():%Y%m%d%H%M%S}{dest.suffix}")
        if dest.resolve() != src.resolve():
            shutil.copy2(src, dest)
        for extra in ("shotsInfo.json",):
            if (src.parent / extra).exists() and not (Path(proj["folder"]) / f"{src.stem}.{extra}").exists():
                shutil.copy2(src.parent / extra, Path(proj["folder"]) / f"{src.stem}.{extra}")
    with _db() as con:
        cur = con.execute("INSERT INTO project_photos (project_id, file, source, title, overview, preview_b64, added) "
                          "VALUES (?, ?, ?, ?, ?, ?, ?)",
                          (pid, str(dest), str(src), title or ov.get("target") or src.stem, json.dumps(ov), preview_b64,
                           datetime.now().isoformat(timespec="seconds")))
    return {"id": cur.lastrowid, "file": str(dest), "overview": ov}

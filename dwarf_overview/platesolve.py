"""Work out what the photo shows by plate-solving the star field.

ASTAP (free, https://www.hnsky.org/astap.htm) matches the stars in the image
against its star database and returns exactly where in the sky the frame
points. The frame's corners are then checked against a bundled deep-sky
catalog (a subset of OpenNGC) to name the subject and everything else in view.

If ASTAP isn't installed or can't solve the frame, the coordinates or target
name the DWARF wrote into shotsInfo.json are used instead, and the result says so.
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

from . import config

DWARF3_ARCSEC_PER_PX = 206.265 * 2.0 / 150.0


class SolveError(Exception):
    pass


@lru_cache(maxsize=1)
def catalog() -> list[dict]:
    raw = json.loads((Path(__file__).parent / "data" / "deep_sky.json").read_text(encoding="utf-8"))
    cols = raw["columns"]
    return [dict(zip(cols, row)) for row in raw["rows"]]


def display_name(o: dict) -> str:
    label = f"M{o['messier']}" if o["messier"] else o["name"]
    if o["common"]:
        label += f" ({o['common']})"
    elif o["messier"]:
        label += f" ({o['name']})"
    return label


def find_astap() -> str | None:
    custom = config.load().get("astap_path")
    if custom:
        return custom if Path(custom).exists() else None
    if sys.platform == "win32":
        bases = [Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "astap",
                 Path(r"C:\Program Files (x86)\astap"), Path(r"C:\astap")]
        for base in bases:
            for exe in ("astap_cli.exe", "astap.exe"):
                if (base / exe).exists():
                    return str(base / exe)
        return None
    if sys.platform == "darwin":
        p = Path("/Applications/ASTAP.app/Contents/MacOS/astap")
        return str(p) if p.exists() else None
    return shutil.which("astap_cli") or shutil.which("astap")


# ---------------------------------------------------------------- hints from shotsInfo.json

def _parse_angle(v, hours: bool) -> float | None:
    if v is None or v == "":
        return None
    if isinstance(v, (int, float)):
        return float(v) * (15.0 if hours and abs(v) <= 24 else 1.0)
    s = str(v).strip()
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", s)
    if not nums:
        return None
    sign = -1.0 if s.startswith("-") else 1.0
    parts = [abs(float(n)) for n in nums[:3]]
    val = parts[0] + (parts[1] / 60 if len(parts) > 1 else 0) + (parts[2] / 3600 if len(parts) > 2 else 0)
    if hours and (len(parts) > 1 or "h" in s.lower() or val <= 24):
        val *= 15.0
    return sign * val


def hint_from_shot_info(info: dict) -> dict:
    lower = {k.lower(): v for k, v in (info or {}).items()}
    ra = next((lower[k] for k in ("ra", "ra_str", "rastr", "target_ra", "ra2000") if k in lower), None)
    dec = next((lower[k] for k in ("dec", "dec_str", "decstr", "target_dec", "dec2000") if k in lower), None)
    target = next((str(lower[k]) for k in ("target", "targetname", "object") if lower.get(k)), None)
    out = {"target": target}
    ra_deg, dec_deg = _parse_angle(ra, True), _parse_angle(dec, False)
    if ra_deg is not None and dec_deg is not None and 0 <= ra_deg < 360 and -90 <= dec_deg <= 90:
        out.update(ra=ra_deg, dec=dec_deg)
    return out


# ---------------------------------------------------------------- solving

def _read_ini(path: Path) -> dict:
    vals = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            vals[k.strip().upper()] = v.strip().strip("'")
    return vals


def solve(image_path: str, width: int, height: int, hint: dict) -> dict:
    exe = find_astap()
    if not exe:
        raise SolveError("ASTAP isn't installed.")
    work = config.data_dir() / "platesolve"
    work.mkdir(exist_ok=True)
    out_base = work / Path(image_path).stem
    for ext in (".ini", ".wcs"):
        out_base.with_suffix(ext).unlink(missing_ok=True)
    args = [exe, "-f", image_path, "-o", str(out_base), "-z", "0"]
    native = (width, height) in {(3840, 2160), (2160, 3840)}
    if native:
        args += ["-fov", f"{height * DWARF3_ARCSEC_PER_PX / 3600:.3f}"]
    if "ra" in hint:
        args += ["-ra", f"{hint['ra'] / 15:.5f}", "-spd", f"{hint['dec'] + 90:.5f}", "-r", "15"]
    else:
        args += ["-r", "180"]
    try:
        subprocess.run(args, capture_output=True, timeout=180,
                       creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
    except (OSError, subprocess.SubprocessError) as e:
        raise SolveError(f"ASTAP didn't run ({e}).")
    ini = out_base.with_suffix(".ini")
    if not ini.exists():
        raise SolveError("ASTAP produced no result.")
    v = _read_ini(ini)
    if v.get("PLTSOLVD", "F").upper() != "T":
        raise SolveError("ASTAP couldn't match the stars in this image.")
    return wcs_from_values(v, width, height)


def wcs_from_values(v: dict, width: int, height: int) -> dict:
    f = lambda k, d=0.0: float(v.get(k, d) or d)
    if "CD1_1" in v:
        cd = [[f("CD1_1"), f("CD1_2")], [f("CD2_1"), f("CD2_2")]]
    else:
        rot = math.radians(f("CROTA2"))
        c1, c2 = f("CDELT1"), f("CDELT2")
        cd = [[c1 * math.cos(rot), -c2 * math.sin(rot)], [c1 * math.sin(rot), c2 * math.cos(rot)]]
    return {"ra": f("CRVAL1"), "dec": f("CRVAL2"), "crpix": [f("CRPIX1", width / 2 + 0.5), f("CRPIX2", height / 2 + 0.5)],
            "cd": cd, "width": width, "height": height,
            "scale_arcsec": round(3600 * math.sqrt(abs(cd[0][0] * cd[1][1] - cd[0][1] * cd[1][0])), 3)}


def _world_to_pixel(wcs: dict, ra: float, dec: float) -> tuple[float, float] | None:
    """Gnomonic (TAN) projection, the standard plate-solve model. Returns FITS-style 1-based pixels."""
    a0, d0, a, d = map(math.radians, (wcs["ra"], wcs["dec"], ra, dec))
    cos_c = math.sin(d0) * math.sin(d) + math.cos(d0) * math.cos(d) * math.cos(a - a0)
    if cos_c <= 0:
        return None
    xi = math.degrees(math.cos(d) * math.sin(a - a0) / cos_c)
    eta = math.degrees((math.cos(d0) * math.sin(d) - math.sin(d0) * math.cos(d) * math.cos(a - a0)) / cos_c)
    (c11, c12), (c21, c22) = wcs["cd"]
    det = c11 * c22 - c12 * c21
    if det == 0:
        return None
    dx = (c22 * xi - c12 * eta) / det
    dy = (-c21 * xi + c11 * eta) / det
    return wcs["crpix"][0] + dx, wcs["crpix"][1] + dy


def _where(x: float, y: float, w: int, h: int, fits_y_up: bool) -> str:
    fx, fy = x / w, y / h
    if fits_y_up:
        fy = 1 - fy
    horiz = "left" if fx < 1 / 3 else "right" if fx > 2 / 3 else ""
    vert = "top" if fy < 1 / 3 else "bottom" if fy > 2 / 3 else ""
    return f"{vert} {horiz}".strip() or "centre"


def objects_in_field(wcs: dict, fits_y_up: bool = True) -> list[dict]:
    w, h = wcs["width"], wcs["height"]
    radius = wcs["scale_arcsec"] * math.hypot(w, h) / 2 / 3600 + 1.0
    found = []
    for o in catalog():
        # Cheap angular-distance prefilter before projecting.
        cosd = math.sin(math.radians(wcs["dec"])) * math.sin(math.radians(o["dec"])) + \
            math.cos(math.radians(wcs["dec"])) * math.cos(math.radians(o["dec"])) * math.cos(math.radians(o["ra"] - wcs["ra"]))
        if math.degrees(math.acos(max(-1.0, min(1.0, cosd)))) > radius:
            continue
        px = _world_to_pixel(wcs, o["ra"], o["dec"])
        if not px:
            continue
        x, y = px
        margin = (o["size_arcmin"] or 0) * 60 / wcs["scale_arcsec"] / 2
        if -margin <= x <= w + margin and -margin <= y <= h + margin:
            inside = 0 <= x <= w and 0 <= y <= h
            off_centre = math.hypot(x - w / 2, y - h / 2) / math.hypot(w / 2, h / 2)
            found.append({"name": display_name(o), "type": o["type"], "mag": o["mag"], "size_arcmin": o["size_arcmin"],
                          "where": _where(x, y, w, h, fits_y_up) if inside else "extends into the frame from outside",
                          "x": round(x), "y": round(y), "_off": off_centre, "_o": o})
    # Most prominent first: big and bright objects near the middle of the frame.
    def prominence(f):
        o = f["_o"]
        size = min(o["size_arcmin"] or 0.5, 200)
        mag = o["mag"] if o["mag"] is not None else 11
        return (math.log10(size + 1) * 2 - mag * 0.3 - f["_off"] * 2 + (1.0 if o["messier"] else 0) + (0.5 if o["common"] else 0))
    found.sort(key=prominence, reverse=True)
    for f in found:
        f.pop("_off"); f.pop("_o")
    return found


def nearest_object(ra: float, dec: float, max_deg: float = 1.5) -> dict | None:
    best, best_d = None, max_deg
    for o in catalog():
        cosd = math.sin(math.radians(dec)) * math.sin(math.radians(o["dec"])) + \
            math.cos(math.radians(dec)) * math.cos(math.radians(o["dec"])) * math.cos(math.radians(o["ra"] - ra))
        d = math.degrees(math.acos(max(-1.0, min(1.0, cosd))))
        # Prefer prominent objects when several are close.
        d_eff = d - (0.3 if o["messier"] else 0) - (0.1 if o["common"] else 0)
        if d_eff < best_d:
            best, best_d = o, d_eff
    return best


def lookup_name(target: str) -> dict | None:
    t = re.sub(r"[\s_-]+", "", target or "").lower()
    m = re.match(r"^m(?:essier)?(\d+)$", t)
    for o in catalog():
        if m and o["messier"] == m.group(1):
            return o
        if t and (o["name"].replace(" ", "").lower() == t or (o["common"] or "").replace(" ", "").lower() == t):
            return o
    return None


def identify(image_path: str, width: int, height: int, shot_info: dict, fits_y_up: bool = True) -> dict:
    hint = hint_from_shot_info(shot_info)
    try:
        wcs = solve(image_path, width, height, hint)
        objs = objects_in_field(wcs, fits_y_up)
        return {"method": "plate-solved from the star field (ASTAP)", "solved": True,
                "center_ra": round(wcs["ra"], 4), "center_dec": round(wcs["dec"], 4),
                "scale_arcsec_per_px": wcs["scale_arcsec"],
                "subject": objs[0]["name"] if objs else "No catalogued deep-sky object in the frame",
                "subject_type": objs[0]["type"] if objs else "", "in_field": objs[:15]}
    except SolveError as e:
        reason = str(e)
    if "ra" in hint:
        o = nearest_object(hint["ra"], hint["dec"])
        if o:
            return {"method": f"from the DWARF's pointing coordinates ({reason})", "solved": False,
                    "subject": display_name(o), "subject_type": o["type"], "in_field": []}
    if hint.get("target"):
        o = lookup_name(hint["target"])
        return {"method": f"from the DWARF's target name ({reason})", "solved": False,
                "subject": display_name(o) if o else hint["target"], "subject_type": o["type"] if o else "",
                "in_field": []}
    return {"method": reason, "solved": False, "subject": None, "subject_type": "", "in_field": []}

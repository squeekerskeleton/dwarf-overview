"""Find the DWARF's own master dark for a session, so Auto-apply can calibrate the raw frames with it.

The DWARF 3 keeps stacked darks in Astronomy/CALI_FRAME (dark/cam_0 is the telephoto camera), named like
dark_exp_15.000000_gain_60_bin_1_38C_stack_10.png: exposure, gain, binning (1 = 4K), sensor temperature and how
many darks were averaged. Auto-apply picks the one with the session's exposure, gain and binning and the nearest
temperature, from the DWARF over USB-C or Wi-Fi (or a copy in the library), and only uses it after checking that
its hot pixels line up with the light frames and that its levels match. Otherwise the stack runs without a dark,
as before; Winsorized rejection still removes most hot pixels.
"""
from __future__ import annotations

import re
import urllib.parse
import urllib.request
from pathlib import Path

from . import dwarf, scope

DARK = re.compile(r"dark_exp_([\d.]+)_gain_(\d+)_bin_(\d)_(-?\d+)C(?:_stack_(\d+))?\.(png|fits?|tiff?)$", re.I)
SESSION = re.compile(r"EXP_([\d.]+)_GAIN_(\d+)", re.I)


def _parse(name: str) -> dict | None:
    m = DARK.search(name)
    if not m:
        return None
    return {"exp": float(m.group(1)), "gain": int(m.group(2)), "bin": int(m.group(3)), "temp": int(m.group(4)),
            "stack": int(m.group(5) or 1)}


def _candidates() -> list[dict]:
    """Every dark file the app can see: {name, where, source, ...parsed}."""
    out = []
    local_roots = [Path(u) / "CALI_FRAME" for u in scope.usb_folders()] + [dwarf.library_dir() / "CALI_FRAME"]
    for root in local_roots:
        if not root.is_dir():
            continue
        for f in root.rglob("*"):
            p = _parse(f.name)
            if p and f.is_file() and "cam_1" not in str(f).lower():  # cam_1 is the wide-angle camera
                out.append({**p, "name": f.name, "where": str(f), "source": "USB-C" if "CALI_FRAME" in str(root) else "library"})
    if out:
        return out
    for root in dwarf.ROOTS.values():
        try:
            out += _walk_wifi(f"{root}CALI_FRAME/", 0)
        except (dwarf.DwarfError, OSError):
            continue
        if out:
            break
    return out


def _walk_wifi(path: str, depth: int) -> list[dict]:
    found = []
    for e in dwarf._listing(path):
        if e["dir"] and depth < 3 and e["name"].lower() != "cam_1":
            found += _walk_wifi(f"{path}{e['name']}/", depth + 1)
        elif (p := _parse(e["name"])):
            found.append({**p, "name": e["name"], "where": path + e["name"], "source": "Wi-Fi"})
    return found


def pick(session: str, shot_temp: float | None, binning: int = 1) -> dict | None:
    m = SESSION.search(session)
    if not m:
        return None
    exp, gain = float(m.group(1)), int(m.group(2))
    match = [c for c in _candidates() if abs(c["exp"] - exp) < 0.01 and c["gain"] == gain and c["bin"] == binning]
    if not match:
        return None
    t = shot_temp if shot_temp is not None else sum(c["temp"] for c in match) / len(match)
    return min(match, key=lambda c: (abs(c["temp"] - t), -c["stack"]))


SESSION_TIME = re.compile(r"(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})")
DARK_SESSION = re.compile(r"tele.*?exp_([\d.]+)_gain_(\d+)", re.I)
RAW_DARK = re.compile(r"\.(fits?|fts)$", re.I)


def _when(name: str):
    import datetime
    m = SESSION_TIME.search(name)
    if not m:
        return None
    try:
        return datetime.datetime(*map(int, m.groups()))
    except ValueError:
        return None


def pick_session_darks(session: str) -> dict | None:
    """Darks the DWARF shot that night (Astronomy/DWARF_DARK/tele_exp_<s>_gain_<g>_..._<date>): raw frames of
    the same exposure and gain taken just before the lights, at the same temperature. On the 2026-11-04 Veil
    their master was cleaner than CALI_FRAME's (one more frame registered per panel, 3% less noise).
    Returns {name, frames: [{name, where, source}], hours} for the set closest in time (within 24 h), or None."""
    m, t0 = SESSION.search(session), _when(session)
    if not m or not t0:
        return None
    exp, gain = float(m.group(1)), int(m.group(2))
    sets = []

    def consider(name, files):
        dm, t = DARK_SESSION.search(name), _when(name)
        if not dm or not t or abs(float(dm.group(1)) - exp) > 0.01 or int(dm.group(2)) != gain or "bin_2" in name.lower():
            return
        hours = abs((t0 - t).total_seconds()) / 3600
        frames = [f for f in files if RAW_DARK.search(f["name"]) and not f["name"].lower().startswith(("stacked", "failed"))]
        if hours <= 24 and len(frames) >= 3:
            sets.append({"name": name, "frames": frames, "hours": hours})

    roots = [Path(u) / "DWARF_DARK" for u in scope.usb_folders()] + [dwarf.library_dir() / "DWARF_DARK"]
    for root in roots:
        if root.is_dir():
            for d in root.iterdir():
                if d.is_dir():
                    consider(d.name, [{"name": f.name, "where": str(f), "source": "USB-C"} for f in d.iterdir() if f.is_file()])
    if not sets:
        for root in dwarf.ROOTS.values():
            try:
                for e in dwarf._listing(f"{root}DWARF_DARK/"):
                    if e["dir"] and DARK_SESSION.search(e["name"]) and _when(e["name"]):
                        path = f"{root}DWARF_DARK/{e['name']}/"
                        consider(e["name"], [{"name": x["name"], "where": path + x["name"], "source": "Wi-Fi"}
                                             for x in dwarf._listing(path) if not x["dir"]])
            except (dwarf.DwarfError, OSError):
                continue
            if sets:
                break
    return min(sets, key=lambda s: s["hours"]) if sets else None


def build_session_master(c: dict, cache: Path) -> Path:
    """Average the night's raw darks with 3-sigma clipping (Siril's rej 3 3, no normalisation) into one FITS."""
    import numpy as np
    from astropy.io import fits
    out = cache / f"{c['name']}_master.fits"
    if out.exists():
        return out
    folder = cache / c["name"]
    stack = []
    for f in c["frames"]:
        local = fetch(f, folder)
        stack.append(np.asarray(fits.getdata(local), dtype="float32"))
    if len({a.shape for a in stack}) != 1:
        raise ValueError("the night's darks aren't all the same size")
    cube = np.stack(stack)
    med = np.median(cube, axis=0)
    sig = 1.4826 * np.median(np.abs(cube - med), axis=0)
    keep = np.abs(cube - med) <= 3 * np.maximum(sig, 1.0)
    master = (cube * keep).sum(axis=0) / np.maximum(keep.sum(axis=0), 1)
    cache.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU(master.astype("float32")).writeto(out, overwrite=True)
    return out


def fetch(c: dict, cache: Path) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / c["name"]
    if target.exists():
        return target
    if c["source"] == "Wi-Fi":
        with urllib.request.urlopen(dwarf._base() + urllib.parse.quote(c["where"]), timeout=60) as r:
            data = r.read()
        target.write_bytes(data)
    else:
        target.write_bytes(Path(c["where"]).read_bytes())
    return target


def _read(path: Path):
    import numpy as np
    if path.suffix.lower() in (".fit", ".fits"):
        from astropy.io import fits
        return np.asarray(fits.getdata(path), dtype="float32")
    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("unreadable")
    return img.astype("float32") if img.ndim == 2 else None


def prepare(dark_file: Path, light_file: Path, out: Path) -> str:
    """Check the dark against a light frame and write it as a FITS master dark Siril can use.
    Returns a note; raises ValueError (with the reason) if it doesn't fit."""
    import numpy as np
    from astropy.io import fits
    dark = _read(dark_file)
    light = np.asarray(fits.getdata(light_file), dtype="float32")
    if dark is None or light.ndim != 2 or dark.shape != light.shape:
        raise ValueError(f"its size {None if dark is None else dark.shape} doesn't match the frames {light.shape}")
    d_med, l_med = float(np.median(dark)), float(np.median(light))
    if d_med > l_med:
        raise ValueError("it's brighter than the frames' background, so its levels don't match")
    mad = float(np.median(np.abs(dark - d_med))) or 1.0
    hot = dark > d_med + 25 * mad
    if hot.sum() < 20:
        raise ValueError("it has too few hot pixels to line up with the frames")
    if hot.sum() > 5000:  # keep the hottest few thousand
        thr = np.partition(dark[hot], -5000)[-5000]
        hot = dark >= thr
    l_mad = float(np.median(np.abs(light - l_med))) or 1.0

    def score(d):
        return float(np.median(light[d]) - l_med)
    flipped = np.flipud(hot)
    s, s_flip = score(hot), score(flipped)
    use_flip = s_flip > s
    best = max(s, s_flip)
    if best < 3 * l_mad:
        raise ValueError("its hot pixels don't line up with the frames'")
    amp_dark = float(np.median(dark[hot]) - d_med)
    ratio = best / max(1.0, amp_dark)
    if not 0.25 < ratio < 4:
        raise ValueError(f"its levels are {ratio:.1f}x off from the frames'")
    data = np.flipud(dark) if use_flip else dark
    # On a truly dark sky the frames' background sits at the dark's level (2026-11-04, 60 s gain 90: 187 against
    # 185), and Siril's integer calibration clips everything below it at 0: half the sky, so frames kept a handful
    # of stars and wouldn't register. A pedestal taken off the dark (about 4x the frames' noise, never below the
    # dark's own floor) keeps light minus dark positive; the constant goes with the background later.
    sigma = 1.4826 * l_mad
    ped = 0.0
    if l_med - d_med < 5 * sigma:
        ped = float(min(4 * sigma, max(0.0, np.percentile(data, 1))))
    fits.PrimaryHDU(np.clip(np.asarray(data, dtype="float32") - ped, 0, 65535).astype("uint16")).writeto(out, overwrite=True)
    return (f"hot pixels line up{' (flipped)' if use_flip else ''}"
            + (f"; a {ped:.0f} ADU pedestal keeps the very dark sky from clipping at zero" if ped else ""))

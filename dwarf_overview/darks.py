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
    fits.PrimaryHDU(np.clip(data, 0, 65535).astype("uint16")).writeto(out, overwrite=True)
    return f"hot pixels line up{' (flipped)' if use_flip else ''}"

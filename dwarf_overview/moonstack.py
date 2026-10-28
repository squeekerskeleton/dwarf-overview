"""Moon stack: gather every Moon shot of one night and turn the best of them into one sharp 4K image.

Sources, found on their own:
- the DWARF's raw Moon sessions (Astronomy/DWARF_RAW_TELE_Moon_EXP_..._<date>; 16-bit Bayer FITS, 1920x1080),
- its Moon videos (Videos/*.mp4; H.265, mostly 3840x2160),
from the library, the DWARF over USB-C, or the DWARF over Wi-Fi (copied first), grouped by the date they were
taken. Shots more than 45 minutes from the night's biggest cluster are left out: the lit edge and the Moon's tilt
change too much to combine.

The stack is lucky imaging, the way AutoStakkert works, done here so it needs no clicking:
1. every frame gets a sharpness score (Laplacian energy on the disc) and only the best ~10% are kept;
2. each kept frame is aligned to the sharpest one: rotation and shift for the whole frame (the DWARF's alt-az
   mount turns the field), then a grid of surface patches each find their own small shift, so seeing ripples are
   undone patch by patch;
3. the aligned frames are averaged in 32-bit;
4. Richardson-Lucy deconvolution and a gentle multi-scale sharpen bring the detail back, with the limb protected
   from ringing;
5. the Moon is centred in a 3840x2160 frame and saved as 16-bit TIFF, PNG, JPG and FITS.

By default every Moon video and raw session of the night is stacked together: 4K video frames as they are, HD
video and raw FITS frames (1920x1080) at 2x, each source ranked against itself, tones matched to the reference
frame, and native 4K frames weighted highest. Videos only or raw sessions only can be picked instead.
"""
from __future__ import annotations

import json
import re
import shutil
import threading
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

from . import dwarf, finishing, scope, tools

OUT_W, OUT_H = 3840, 2160
KEEP = 0.05            # share of each source's frames kept (5% beat 10% and 20% on Kip's 2026-04-30 videos)
MOON_SAT = 0.8         # mineral-colour boost: 3x and 1.5x showed teal/pink false colour to Kip (2026-10-28); DWARF's own Moon images are near grey
MOON_MEDIAN = 0.40     # disc median after the brightness curve (0.60 looked blown out to Kip; 0.38 showed maria noise)
RL_ITERS = 30          # Richardson-Lucy iterations (2026-10-23 after the lighter denoise: 15/20/25/30 gave detail/noise 10.5/11.1/11.5/11.8; 40 rimmed crater walls)
MAX_KEEP, MIN_KEEP = 1000, 50
REF_N = 8              # the alignment reference is the mean of this many of the sharpest frames
ADD_MIN = 1.0          # a further source joins only if the combined stack loses no detail/noise (Kip: "if it degrades resolution, do not stack together"; 0.98 let a raw session in on 2026-10-23)
SOFT = 0.8             # a source whose median frame is below this share of the best same-size source's is left out
CLUSTER_MIN = 45       # minutes either side of the night's main cluster
STEPS = ["Find the night's Moon shots", "Score every frame's sharpness", "Align and stack the best frames",
         "Sharpen", "Centre in a 4K frame and save"]
WEIGHTS = [5, 40, 45, 6, 4]
WHEN = re.compile(r"(20\d{2})[-_]?(\d{2})[-_]?(\d{2})[-_T ]?(\d{2})[-_:]?(\d{2})[-_:]?(\d{2})")


class MoonError(Exception):
    pass


def _when(name: str, fallback: float | None = None) -> datetime | None:
    if m := WHEN.search(name):
        try:
            return datetime(*map(int, m.groups()))
        except ValueError:
            pass
    return datetime.fromtimestamp(fallback) if fallback else None


def _is_moon_session(name: str) -> bool:
    return name.upper().startswith("DWARF_RAW") and "_MOON_" in name.upper()


# ---------------------------------------------------------------- finding the shots

def _local_items() -> list[dict]:
    items = []
    astro = [Path(u) for u in scope.usb_folders()]
    lib = dwarf.library_dir()
    for base, source in [(a, "DWARF (USB-C)") for a in astro] + [(lib, "library")]:
        try:
            dirs = [d for d in base.iterdir() if d.is_dir()]
            dirs += [c for d in dirs if not _is_moon_session(d.name) for c in d.iterdir() if c.is_dir()]
        except OSError:
            continue
        for d in dirs:
            if _is_moon_session(d.name):
                frames = [f for f in d.iterdir() if f.suffix.lower() in (".fits", ".fit")
                          and not f.name.lower().startswith(("failed", "stacked"))]
                if frames:
                    items.append({"kind": "fits", "name": d.name, "where": str(d), "source": source,
                                  "when": _when(d.name, d.stat().st_mtime), "frames": len(frames)})
    video_dirs = [(a.parent / "Videos", "DWARF (USB-C)") for a in astro] + [(lib, "library")]
    for vd, source in video_dirs:
        if not vd.is_dir():
            continue
        for f in list(vd.glob("*.mp4")) + list(vd.glob("*/*.mp4")):
            if "thumbnail" in str(f).lower():
                continue
            items.append({"kind": "video", "name": f.name, "where": str(f), "source": source,
                          "when": _when(f.name, f.stat().st_mtime), "size": f.stat().st_size})
    return items


def _wifi_items() -> list[dict]:
    items = []
    try:
        base = dwarf._base()
    except dwarf.DwarfError:
        return items
    for root in dwarf.ROOTS.values():
        try:
            entries = dwarf._listing(root)
        except (dwarf.DwarfError, OSError):
            continue
        for e in entries:
            if e["dir"] and _is_moon_session(e["name"]):
                items.append({"kind": "fits", "name": e["name"], "where": f"{root}{e['name']}/", "source": "DWARF (Wi-Fi)",
                              "when": _when(e["name"]), "frames": None})
        videos = root.replace("Astronomy/", "Videos/")
        try:
            for e in dwarf._listing(videos):
                if not e["dir"] and e["name"].lower().endswith(".mp4"):
                    items.append({"kind": "video", "name": e["name"], "where": videos + e["name"],
                                  "source": "DWARF (Wi-Fi)", "when": _when(e["name"]), "size": None})
        except (dwarf.DwarfError, OSError):
            pass
        break
    return items


def _all_items() -> list[dict]:
    items = _local_items()
    seen = {i["name"] for i in items}
    if not any(i["source"] == "DWARF (USB-C)" for i in items):
        items += [i for i in _wifi_items() if i["name"] not in seen]
    # the same session in the library and on the DWARF: keep the local copy
    best: dict[str, dict] = {}
    rank = {"library": 0, "DWARF (USB-C)": 1, "DWARF (Wi-Fi)": 2}
    for i in items:
        if i["name"] not in best or rank[i["source"]] < rank[best[i["name"]]["source"]]:
            best[i["name"]] = i
    return [i for i in best.values() if i["when"]]


def dates() -> list[dict]:
    """Nights with Moon shots, newest first."""
    by: dict[str, dict] = {}
    for i in _all_items():
        d = by.setdefault(i["when"].strftime("%Y-%m-%d"), {"date": i["when"].strftime("%Y-%m-%d"), "sessions": 0,
                                                           "frames": 0, "videos": 0, "sources": set(), "first": i["when"],
                                                           "last": i["when"]})
        d["sources"].add(i["source"])
        d["first"], d["last"] = min(d["first"], i["when"]), max(d["last"], i["when"])
        if i["kind"] == "fits":
            d["sessions"] += 1
            d["frames"] += i["frames"] or 0
        else:
            d["videos"] += 1
        d.setdefault("items", []).append({"name": i["name"], "kind": i["kind"], "time": i["when"].strftime("%H:%M:%S"),
                                          "frames": i.get("frames"), "size": i.get("size"), "source": i["source"]})
    out = []
    for d in sorted(by.values(), key=lambda d: d["date"], reverse=True):
        out.append({**d, "sources": sorted(d["sources"]), "first": d["first"].strftime("%H:%M"),
                    "last": d["last"].strftime("%H:%M"), "items": sorted(d.get("items", []), key=lambda i: i["time"])})
    return out


def _cluster(items: list[dict]) -> tuple[list[dict], list[dict]]:
    """The biggest group of shots within CLUSTER_MIN of each other; the rest are left out."""
    items = sorted(items, key=lambda i: i["when"])
    best, best_n = [], 0
    for a in items:
        group = [b for b in items if abs((b["when"] - a["when"]).total_seconds()) <= CLUSTER_MIN * 60]
        n = sum((b.get("frames") or 100) for b in group)
        if n > best_n:
            best, best_n = group, n
    return best, [i for i in items if i not in best]


# ---------------------------------------------------------------- reading frames

def _debayer(raw, code_cache: dict):
    """16-bit RGGB to float RGB. OpenCV names Bayer patterns differently from FITS, so pick the variant that
    gives the Moon its slightly warm colour (the wrong one turns it blue)."""
    import cv2
    import numpy as np
    raw16 = raw.astype("uint16")
    if "code" not in code_cache:
        a = cv2.cvtColor(raw16, cv2.COLOR_BayerBG2RGB).astype("float32")
        b = cv2.cvtColor(raw16, cv2.COLOR_BayerRG2RGB).astype("float32")
        bright = a.mean(axis=2) > np.percentile(a.mean(axis=2), 80)
        code_cache["code"] = cv2.COLOR_BayerBG2RGB if a[bright][:, 0].mean() >= a[bright][:, 2].mean() else cv2.COLOR_BayerRG2RGB
        return (a if code_cache["code"] == cv2.COLOR_BayerBG2RGB else b) / 65535.0
    return cv2.cvtColor(raw16, code_cache["code"]).astype("float32") / 65535.0


def _fits_frames(item: dict, work: Path, j) -> list[Path]:
    """The session's frames as local files. Sessions on the DWARF (USB-C or Wi-Fi) are copied into the work folder
    first, so unplugging it mid-run (2026-10-28) can't cut a later pass short."""
    if item["source"] == "DWARF (USB-C)":
        dest = work / "frames" / item["name"]
        src = sorted(f for f in Path(item["where"]).iterdir() if f.suffix.lower() in (".fits", ".fit")
                     and not f.name.lower().startswith(("failed", "stacked")))
        dest.mkdir(parents=True, exist_ok=True)
        for f in src:
            if not (dest / f.name).exists() or (dest / f.name).stat().st_size != f.stat().st_size:
                j.wait_if_paused()
                shutil.copy2(f, dest / (f.name + ".part"))
                (dest / (f.name + ".part")).replace(dest / f.name)
        return sorted(dest / f.name for f in src)
    if item["source"] != "DWARF (Wi-Fi)":
        return sorted(f for f in Path(item["where"]).iterdir() if f.suffix.lower() in (".fits", ".fit")
                      and not f.name.lower().startswith(("failed", "stacked")))
    dest = work / "frames" / item["name"]
    dest.mkdir(parents=True, exist_ok=True)
    base = dwarf._base()
    names = [e["name"] for e in dwarf._listing(item["where"]) if not e["dir"]
             and e["name"].lower().endswith((".fits", ".fit")) and not e["name"].lower().startswith(("failed", "stacked"))]
    for k, n in enumerate(names):
        if not (dest / n).exists():
            j.wait_if_paused()
            with urllib.request.urlopen(base + urllib.parse.quote(item["where"] + n), timeout=60) as r:
                (dest / (n + ".part")).write_bytes(r.read())
            (dest / (n + ".part")).replace(dest / n)
        if k % 20 == 0:
            j.log(f"Copied {k + 1} of {len(names)} frames of {item['name']} over Wi-Fi")
    return sorted(dest.glob("*.fits")) + sorted(dest.glob("*.fit"))


def _video_path(item: dict, work: Path, j) -> Path:
    if item["source"] == "DWARF (USB-C)":  # copied once, then every pass reads the local copy
        src, dest = Path(item["where"]), work / "frames" / item["name"]
        if not dest.exists() or dest.stat().st_size != src.stat().st_size:
            dest.parent.mkdir(parents=True, exist_ok=True)
            j.log(f"Copying {item['name']} from the DWARF…")
            shutil.copy2(src, str(dest) + ".part")
            Path(str(dest) + ".part").replace(dest)
        return dest
    if item["source"] != "DWARF (Wi-Fi)":
        return Path(item["where"])
    dest = work / "frames" / item["name"]
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        j.log(f"Copying {item['name']} over Wi-Fi…")
        with urllib.request.urlopen(dwarf._base() + urllib.parse.quote(item["where"]), timeout=60) as r, \
                open(str(dest) + ".part", "wb") as o:
            shutil.copyfileobj(r, o, 1 << 20)
        Path(str(dest) + ".part").replace(dest)
    return dest


def _video_frames(path: Path, width: int | None = None):
    """(index, float RGB 0..1) for every frame; with width, shrunk to it first (scoring: 4K float frames made
    scoring 14,600 frames take an hour on 2026-10-23)."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    try:
        k = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if width and fr.shape[1] > width:
                fr = cv2.resize(fr, (width, max(1, round(fr.shape[0] * width / fr.shape[1]))), interpolation=cv2.INTER_AREA)
            yield k, cv2.cvtColor(fr, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
            k += 1
    finally:
        cap.release()


def _video_info(path: Path) -> tuple[int, int, int]:
    import cv2
    cap = cv2.VideoCapture(str(path))
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()


# ---------------------------------------------------------------- measuring and aligning

def _gray_small(img, width: int = 960):
    import cv2
    g = img.mean(axis=2) if img.ndim == 3 else img
    s = width / g.shape[1]
    return cv2.resize(g, (width, max(1, round(g.shape[0] * s))), interpolation=cv2.INTER_AREA), s


def _sky_level(g) -> float:
    """The sky's brightness, from a band along the frame's edge (the Moon is never stacked touching it). Dusk and
    moonlit-haze skies sit well above black: 2026-10-23's was a quarter of full scale."""
    import numpy as np
    h, w = g.shape[:2]
    b = max(2, min(h, w) // 25)
    band = np.concatenate([g[:b].ravel(), g[-b:].ravel(), g[:, :b].ravel(), g[:, -b:].ravel()])
    return float(np.median(band))


def _disc_mask(g):
    """The lit Moon, measured above the sky: a fixed share of the brightest level put the whole frame inside the
    'disc' when the sky was bright (2026-10-23 at dusk)."""
    import numpy as np
    sky = _sky_level(g)
    return g > sky + 0.35 * (float(np.percentile(g, 99.5)) - sky)


def _disc(g):
    """Mask of the lit Moon in a small grey frame, and how bright it is."""
    import numpy as np
    from .lunar import _otsu
    g = np.clip(g - _sky_level(g), 0, None)
    norm = g / max(1e-6, float(np.percentile(g, 99.9)))
    thr = max(_otsu(np.clip(norm, 0, 1)), 0.08)
    m = norm > thr
    return m, float(g[m].mean()) if m.any() else 0.0


def _score(img) -> tuple[float, float, bool, float]:
    """Sharpness (Laplacian energy inside the disc, relative to brightness), brightness, whether the disc touches
    the edge of the frame, and the disc's radius as a share of the frame width."""
    import cv2
    import numpy as np
    g, _ = _gray_small(img)
    m, bright = _disc(g)
    if m.mean() < 0.005 or m.mean() > 0.7:  # no Moon, or glare/cloud filling the frame
        return 0.0, bright, True, 0.0
    inner = cv2.erode(m.astype("uint8"), np.ones((7, 7), np.uint8)).astype(bool)
    lap = cv2.Laplacian(g, cv2.CV_32F)
    s = float(np.sqrt(np.mean(lap[inner] ** 2)) / max(bright, 1e-4)) if inner.any() else 0.0
    edge = bool(m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any())
    return s, bright, edge, float(np.sqrt(m.sum() / np.pi) / g.shape[1])


def _global_warp(ref_small, img_small, scale: float):
    """Rotation + shift that lines img up with the reference (full-resolution 2x3 matrix), or None."""
    import cv2
    import numpy as np
    shift, _ = cv2.phaseCorrelate(ref_small.astype("float32"), img_small.astype("float32"))
    warp = np.array([[1, 0, -shift[0]], [0, 1, -shift[1]]], dtype="float32")
    try:
        _, warp = cv2.findTransformECC(ref_small.astype("float32"), img_small.astype("float32"), warp,
                                       cv2.MOTION_EUCLIDEAN,
                                       (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5), None, 5)
    except cv2.error:
        pass
    warp = warp.copy()
    warp[:, 2] /= scale
    return warp


def _ncc(a, b, mask) -> float:
    """Normalised cross-correlation of two grey images over a mask."""
    import numpy as np
    x, y = a[mask].astype("float64"), b[mask].astype("float64")
    x, y = x - x.mean(), y - y.mean()
    return float((x * y).sum() / max(1e-12, np.sqrt((x * x).sum() * (y * y).sum())))


def _local_map(ref_g, img_g, mask, patch: int = 128, step: int = 64, max_shift: float = 8.0):
    """Per-patch shifts across the disc (seeing ripples), smoothed into a full-resolution remap field."""
    import cv2
    import numpy as np
    h, w = ref_g.shape
    ys = list(range(0, h - patch + 1, step))
    xs = list(range(0, w - patch + 1, step))
    gy = np.zeros((len(ys), len(xs)), "float32")
    gx = np.zeros_like(gy)
    ok = np.zeros(gy.shape, bool)
    win = cv2.createHanningWindow((patch, patch), cv2.CV_32F)
    for a, y in enumerate(ys):
        for b, x in enumerate(xs):
            if mask[y + patch // 2, x + patch // 2] == 0:
                continue
            r = ref_g[y:y + patch, x:x + patch]
            if r.std() < 1e-3:
                continue
            # Copies: OpenCV applies the window to its inputs in place, which would darken the reference.
            (dx, dy), resp = cv2.phaseCorrelate(r.copy(), img_g[y:y + patch, x:x + patch].copy(), win)
            if resp > 0.1 and abs(dx) < max_shift and abs(dy) < max_shift:
                gx[a, b], gy[a, b], ok[a, b] = dx, dy, True
    if ok.sum() < 4:
        return None
    for g in (gx, gy):
        g[~ok] = float(np.median(g[ok]))
        g[:] = cv2.GaussianBlur(g, (3, 3), 0.8)
    fx = cv2.resize(gx, (w, h), interpolation=cv2.INTER_CUBIC)
    fy = cv2.resize(gy, (w, h), interpolation=cv2.INTER_CUBIC)
    X, Y = np.meshgrid(np.arange(w, dtype="float32"), np.arange(h, dtype="float32"))
    return X + fx, Y + fy


def _sharpness(img, mask=None) -> float:
    """RMS Laplacian inside the disc (eroded), relative to its brightness, at the image's own scale."""
    import cv2
    import numpy as np
    g = img.mean(axis=2).astype("float32") if img.ndim == 3 else img.astype("float32")
    if mask is None:
        mask = _disc_mask(g)
    inner = cv2.erode(mask.astype("uint8"), np.ones((25, 25), np.uint8)).astype(bool)
    if inner.sum() < 1000:
        return 0.0
    lap = cv2.Laplacian(cv2.GaussianBlur(g, (0, 0), 0.7), cv2.CV_32F)
    # relative to the disc's brightness above the sky: a bright dusk sky made 2026-10-23's videos look softer
    return float(np.sqrt(np.mean(lap[inner] ** 2)) / max(1e-6, float(g[inner].mean()) - _sky_level(g)))


def _detail_noise(img) -> float:
    """Surface detail over noise at a common 650 px disc: the RMS of a σ1-σ3 band-pass on the inner disc, over the
    MAD of the finest (σ<1) residual on the smoothest tenth of it (the maria). Plain sharpness rewards noise: on
    2026-10-23 the noisier 17:53 video scored sharper, yet adding it cut the stack's detail/noise from 14.8 to 11.3."""
    import cv2
    import numpy as np
    g = img.mean(axis=2).astype("float32") if img.ndim == 3 else img.astype("float32")
    g = np.clip(g - _sky_level(g), 0, None)
    mask = _disc_mask(g)
    ys, xs = np.nonzero(mask)
    if ys.size < 1000:
        return 0.0
    s = 650 / max(1, max(ys.max() - ys.min(), xs.max() - xs.min()))
    g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask.astype("uint8"), g.shape[::-1], interpolation=cv2.INTER_NEAREST)
    inner = cv2.erode(mask, np.ones((25, 25), np.uint8)).astype(bool)
    if inner.sum() < 1000:
        return 0.0
    b1, b3 = cv2.GaussianBlur(g, (0, 0), 1.0), cv2.GaussianBlur(g, (0, 0), 3.0)
    band = float(np.sqrt(np.mean((b1 - b3)[inner] ** 2)))
    fine = g - b1
    calm = cv2.GaussianBlur(np.abs(b1 - b3), (0, 0), 8)
    maria = inner & (calm <= np.percentile(calm[inner], 10))
    noise = 1.4826 * float(np.median(np.abs(fine[maria] - np.median(fine[maria])))) if maria.sum() > 100 else 0.0
    return band / max(1e-6, noise)


def _fit_to(img, s: float, shape: tuple):
    """A frame resized by s (its Moon to the reference's size) and placed on a frame of the given shape, cropped
    around the Moon or padded with the edge's own sky."""
    import cv2
    if s != 1.0:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
    H, W = shape
    h, w = img.shape[:2]
    if h > H or w > W:
        g, k = _gray_small(img)
        m = _disc(g)[0]
        cy, cx = ((float(v.mean()) / k for v in m.nonzero()) if m.any() else (h / 2, w / 2))
        y0 = int(min(max(0, round(cy - H / 2)), max(0, h - H)))
        x0 = int(min(max(0, round(cx - W / 2)), max(0, w - W)))
        img = img[y0:y0 + H, x0:x0 + W]
        h, w = img.shape[:2]
    if (h, w) != (H, W):
        t, l = (H - h) // 2, (W - w) // 2
        img = cv2.copyMakeBorder(img, t, H - h - t, l, W - w - l, cv2.BORDER_REPLICATE)
    return img


def _read(each_frame, si: int, keys: list) -> dict:
    """The given frames of one source, read in one pass (stopping once all are found)."""
    want, out = set(keys), {}
    for s_, k, img in each_frame({si}):
        if k in want:
            out[k] = img
            if len(out) == len(want):
                break
    if len(out) < len(want):
        raise MoonError(f"Only {len(out)} of {len(want)} frames could be read back from one of the shots: was the DWARF "
                        "unplugged or the file moved during the run? Copy the night into the library and start again.")
    return out


def _screen(j, sources: list, ranked: dict, each_frame, target: tuple, trial: int = 20) -> tuple[list, str]:
    """Which sources to stack: never one that costs resolution (Kip: "if it degrades resolution, do not stack
    together"). A source whose median frame is well below the sharpest source of the same size is left out (on
    2026-04-30 one video's 0.032 against 0.055; stacking it in cut the sharpness from 0.102 to 0.075). Then each
    source's best frames are stacked on their own at the common 4K scale, and, sharpest first, each next source
    joins only if the combined stack keeps ADD_MIN of the detail-to-noise ratio (_detail_noise). Returns (source
    indexes, note)."""
    import cv2
    import numpy as np
    left, notes = [], []
    med = {si: float(np.median([f[0] for f in fr])) for si, fr in ranked.items()}
    for si in list(ranked):
        # Only frames scored at the same shrink compare (0.17.11 grouped 4K with 1080p, and the 1080p raw session's
        # 2x-shrunk scores beat both 4K videos' 4x-shrunk ones, leaving the best video out); mixing is the trial's job.
        same = [o for o in ranked if sources[o]["shape"][1] == sources[si]["shape"][1]]
        top = max(med[o] for o in same)
        if med[si] < SOFT * top:
            left.append(si)
            notes.append(f"{sources[si]['name']} left out (softer: median sharpness {med[si]:.3f} against {top:.3f})")
    cand = [si for si in ranked if si not in left]
    if len(cand) < 2:
        return cand, "; ".join(notes)
    j.log("Testing whether each video and session adds detail or blurs it…")
    stacks = {}
    for si in cand:
        imgs = [_fit_to(i, sources[si]["s"], target) for i in _read(each_frame, si, [f[2] for f in ranked[si][:trial]]).values()]
        if not imgs:
            continue
        r_small, sc = _gray_small(imgs[0])
        acc = imgs[0].astype("float64")
        for img in imgs[1:]:
            if img.shape != imgs[0].shape:
                img = cv2.resize(img, imgs[0].shape[1::-1], interpolation=cv2.INTER_CUBIC)
            acc += np.clip(cv2.warpAffine(img, _global_warp(r_small, _gray_small(img)[0], sc), imgs[0].shape[1::-1],
                                          flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE), 0, None)
        stacks[si] = (acc / len(imgs)).astype("float32")
    order = sorted(stacks, key=lambda si: -_detail_noise(stacks[si]))
    first = stacks[order[0]]
    f_small, f_sc = _gray_small(first)
    f_g = first.mean(axis=2).astype("float32")
    f_mask = cv2.resize(_disc(f_small)[0].astype("uint8"), f_g.shape[::-1], interpolation=cv2.INTER_NEAREST).astype(bool)
    keep = getattr(j, "keep", KEEP)

    def stacked(si):  # the frames this source would give the real stack
        return float(min(len(ranked[si]), max(MIN_KEEP, round(len(ranked[si]) * keep))))
    used = [order[0]]
    acc, wsum = first.astype("float64") * stacked(order[0]), stacked(order[0])
    cur = _detail_noise(first)
    for si in order[1:]:
        img = stacks[si]
        if img.shape != first.shape:
            img = cv2.resize(img, first.shape[1::-1], interpolation=cv2.INTER_CUBIC)
        img = np.clip(cv2.warpAffine(img, _global_warp(f_small, _gray_small(img)[0], f_sc), first.shape[1::-1],
                                     flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE), 0, None)
        img = _tone_match(img, f_g, f_mask)
        n = stacked(si)
        trial_s = _detail_noise(((acc + img * n) / (wsum + n)).astype("float32"))
        if trial_s > ADD_MIN * cur:  # a tie doesn't join
            used.append(si)
            notes.append(f"{sources[si]['name']} added (detail/noise {cur:.1f} to {trial_s:.1f})")
            acc, wsum, cur = acc + img * n, wsum + n, trial_s
        else:
            notes.append(f"{sources[si]['name']} left out (it would lower the detail: detail/noise {trial_s:.1f} against {cur:.1f})")
    for si in cand:
        if si not in stacks:  # couldn't be tested: it might cost resolution, so it stays out
            notes.append(f"{sources[si]['name']} left out (its frames couldn't be tested)")
    return sorted(used), "; ".join(notes)


# ---------------------------------------------------------------- the job

def date_of(path: str) -> str:
    """The night a Moon photo, frame or video belongs to (YYYY-MM-DD)."""
    p = Path(path)
    for name in (p.name, p.parent.name):
        if w := _when(name):
            return w.strftime("%Y-%m-%d")
    if p.suffix.lower() in (".fits", ".fit"):
        try:
            from astropy.io import fits as pf
            if w := _when(str(pf.getheader(p).get("DATE-OBS", "")).replace(":", "")):
                return w.strftime("%Y-%m-%d")
        except (OSError, ValueError):
            pass
    try:
        return datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d")
    except OSError:
        raise MoonError("Couldn't tell which night this was taken.")


def night(date: str) -> dict:
    """What a night has, for the Assess tab's Moon card."""
    return next((d for d in dates() if d["date"] == date), {"date": date, "sessions": 0, "frames": 0, "videos": 0,
                                                           "sources": [], "first": "", "last": ""})


def _picked(path: str) -> dict | None:
    """A Moon video or raw frame the person opened themselves, possibly from a folder the app doesn't search."""
    p = Path(path)
    if not p.is_file():
        return None
    from .lunar import VIDEO_EXTS
    when = _when(p.name, p.stat().st_mtime)
    if p.suffix.lower() in VIDEO_EXTS:
        return {"kind": "video", "name": p.name, "where": str(p), "source": "picked", "when": when, "size": p.stat().st_size}
    if p.suffix.lower() in (".fits", ".fit") and _is_moon_session(p.parent.name):
        frames = [f for f in p.parent.iterdir() if f.suffix.lower() in (".fits", ".fit")
                  and not f.name.lower().startswith(("failed", "stacked"))]
        return {"kind": "fits", "name": p.parent.name, "where": str(p.parent), "source": "picked",
                "when": _when(p.parent.name, p.stat().st_mtime), "frames": len(frames)}
    return None


def start(date: str, source: str = "all", include: str | None = None, exclude: list[str] | None = None,
          keep: float | None = None) -> str:
    """Stack a night. include: a file the person opened (always used, even from outside the library and the
    DWARF); exclude: names of videos or sessions they unticked; keep: share of each source's frames to keep
    (default KEEP), for tuning runs."""
    from .autoprocess import _Job, work_root
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date or ""):
        raise MoonError("Pick a date.")
    items = [i for i in _all_items() if i["when"].strftime("%Y-%m-%d") == date and i["name"] not in (exclude or [])]
    extra = _picked(include) if include else None
    if extra and extra["name"] not in {i["name"] for i in items}:
        extra["must"] = True
        items.append(extra)
    elif extra:
        for i in items:
            if i["name"] == extra["name"]:
                i["must"] = True
    if not items:
        raise MoonError(f"No Moon shots found for {date}. Plug the DWARF in over USB-C, connect it over Wi-Fi, "
                        "or copy its Moon sessions and videos into the library.")
    work = work_root() / f"Moon_{date}"
    work.mkdir(parents=True, exist_ok=True)
    j = _Job(STEPS)
    j.weights = list(WEIGHTS)
    j.keep = float(min(0.5, max(0.01, keep))) if keep else KEEP
    threading.Thread(target=_run, args=(j, items, source, work, date), daemon=True).start()
    return j.id


def _set_aside(folder: Path, keep: int = 3) -> None:
    """Move an earlier run's result_moon_4k files into earlier/<time>/ instead of letting the new run write over
    them (2026-10-28: tuning runs replaced Kip's best Moon). The last `keep` earlier results are kept."""
    old = sorted(folder.glob("result_moon_4k.*")) if folder.is_dir() else []
    if not old:
        return
    stamp = datetime.fromtimestamp(max(f.stat().st_mtime for f in old)).strftime("%Y-%m-%d_%H%M%S")
    dest = folder / "earlier" / stamp
    dest.mkdir(parents=True, exist_ok=True)
    for f in old:
        f.replace(dest / f.name)
    sets = sorted(d for d in (folder / "earlier").iterdir() if d.is_dir())
    for d in sets[:-keep]:
        shutil.rmtree(d, ignore_errors=True)


def _run(j, items, source, work, date) -> None:
    try:
        out = _pipeline(j, items, source, work, date)
        if not j.stopped():
            from .autoprocess import archive, finished_root
            root = finished_root()
            if root:
                _set_aside(root / work.name)
            out = archive(j, work, out)
            j.job.update(state="done", output=out)
    except Exception as e:
        if not j.stopped():
            for s in j.job["steps"]:
                if s["state"] == "running":
                    s["state"] = "failed"
            j.job.update(state="failed", error=str(e))


def _tone_match(img, ref_l, mask):
    """Give a frame the reference's tones on the disc (quantile matching). Raw FITS frames are linear and the
    videos are gamma-encoded, and exposures differ, so this is what lets them be averaged together. Cubic warps
    leave slightly negative pixels at the limb; 0.16.0 divided by them and blew the stack up (gains to 1e9)."""
    import numpy as np
    img = np.clip(img, 0, None)
    lum = img.mean(axis=2)
    sel = mask & (lum > 0.02)
    if sel.sum() < 1000:
        return img
    q = np.linspace(0.5, 99.8, 64)
    a, b = np.percentile(lum[sel], q), np.percentile(ref_l[sel], q)
    a = np.maximum.accumulate(a + np.arange(64) * 1e-7)  # strictly increasing for interp
    new = np.interp(lum, a, b)
    lo = lum < a[0]  # darker than the disc (sky, shadowed edge): scale down towards black, never clamp up
    new[lo] = lum[lo] * (b[0] / max(1e-6, a[0]))
    hi = lum > a[-1]  # beyond the matched range: carry on with the last slope
    if hi.any():
        slope = float(np.clip((b[-1] - b[-8]) / max(1.0 / 255, a[-1] - a[-8]), 0.5, 1.5))
        new[hi] = b[-1] + (lum[hi] - a[-1]) * slope
    gain = np.where(lum > 1e-3, new / np.maximum(lum, 1e-3), 1.0)  # never divide by about zero
    return img * np.clip(gain, 0, 4)[..., None]


def _pipeline(j, items, source, work: Path, date: str) -> dict:
    import numpy as np
    import cv2

    # 1. which shots: by default every Moon video and raw session from the night, together
    j.step(0, "running")
    keep, left = _cluster(items)
    for i in [i for i in left if i.get("must")]:  # what the person picked is always stacked
        keep.append(i)
        left.remove(i)
    videos = [i for i in keep if i["kind"] == "video"]
    fits = [i for i in keep if i["kind"] == "fits"]
    vinfo = {}
    for v in videos:
        try:
            vinfo[v["name"]] = _video_info(_video_path(v, work, j))
        except Exception as e:
            j.log(f"Skipping {v['name']}: {e}")
    videos = [v for v in videos if v["name"] in vinfo and vinfo[v["name"]][2] > 0]
    if source == "video":
        fits = []
    elif source == "fits":
        videos = []
    if not videos and not fits:
        raise MoonError("None of that night's shots can be stacked here (no Moon videos or raw Moon sessions).")
    # Everything is brought to the scale of the reference source's Moon, measured (below), not assumed from the frame
    # size: the DWARF 3's 1080p video and raw frames are a crop at the same scale as its 4K video, and the 2x
    # upscaling assumed until 0.17.10 doubled their Moon (the ghost of 2026-10-23, and every HD/raw frame rejected).
    sources = []
    for v in videos:
        w, h, n = vinfo[v["name"]]
        sources.append({**v, "up": 1, "s": 1.0, "shape": (h, w), "n": n, "weight": 1.0 if w >= 3000 else 0.6})
    for f in fits:
        sources.append({**f, "up": 1, "s": 1.0, "shape": None, "n": f.get("frames") or 120, "weight": 0.7})
    total = sum(s["n"] for s in sources)
    parts = []
    nv, nf = sum(s["n"] for s in sources if s["kind"] == "video"), sum(s["n"] for s in sources if s["kind"] == "fits")
    if nv:
        parts.append(f"{sum(1 for s in sources if s['kind'] == 'video')} video(s), {nv} frames")
    if nf:
        parts.append(f"{sum(1 for s in sources if s['kind'] == 'fits')} raw session(s), about {nf} frames")
    note = (f"{' and '.join(parts)}, from {min(s['when'] for s in sources):%H:%M} to {max(s['when'] for s in sources):%H:%M}, "
            "each tested before being stacked together at 4K")
    if left:
        note += f"; left out {len(left)} shot(s) taken more than {CLUSTER_MIN} min from the rest"
    j.step(0, "done", note + ".")

    bayer: dict = {}

    def each_frame(only=None, width=None):
        for si, s in enumerate(sources):
            if only is not None and si not in only:
                continue
            if s["kind"] == "video":
                for k, img in _video_frames(_video_path(s, work, j), width):
                    yield si, k, img
            else:
                from astropy.io import fits as pf
                for f in _fits_frames(s, work, j):
                    try:
                        yield si, f.name, _debayer(np.asarray(pf.getdata(f)), bayer)
                    except (OSError, ValueError, TypeError):
                        continue

    # 2. score every frame
    j.step(1, "running")
    frames = []  # (score, source, frame key, brightness)
    radius: dict = {}  # source -> [(score, disc radius as a share of the frame width)]
    n = 0
    for si, k, img in each_frame(width=960):  # _score works at 960 px wide anyway
        j.wait_if_paused()
        if j.stopped():
            raise MoonError("Stopped.")
        s, bright, edge, rel = _score(img)
        if sources[si]["shape"] is None:
            sources[si]["shape"] = img.shape[:2]
        if s > 0 and not edge:
            frames.append((s, si, k, bright))
            radius.setdefault(si, []).append((s, rel))
        n += 1
        if n % 50 == 0:
            j.part(1, n / max(1, total))
    if len(frames) < 5:
        raise MoonError("Fewer than 5 usable Moon frames (the disc was missing, cut off by the frame edge or lost in glare).")
    # Each source is ranked against itself (resolution, exposure and gain differ), and the best 5% of each kept.
    ranked = {si: sorted((f for f in frames if f[1] == si), key=lambda f: -f[0]) for si in sorted({f[1] for f in frames})}
    # The Moon's size in each source (median of its 20 sharpest frames, in native pixels); every source is resized
    # to the size in the widest, sharpest source.
    r_nat = {si: float(np.median([r for _, r in sorted(radius[si], key=lambda t: -t[0])[:20]])) * sources[si]["shape"][1]
             for si in ranked}
    ref_si = max(ranked, key=lambda si: (sources[si]["shape"][1], float(np.median([f[0] for f in ranked[si]]))))
    for si in ranked:
        sc_ = r_nat[ref_si] / max(1.0, r_nat[si])
        sources[si]["s"] = 1.0 if abs(sc_ - 1) < 0.03 else sc_
    target = tuple(sources[ref_si]["shape"])
    used, why = _screen(j, sources, ranked, each_frame, target)
    sized = [f"{sources[si]['name']} x{sources[si]['s']:.2f}" for si in used if sources[si]["s"] != 1.0]
    if sized:
        why = (why + "; " if why else "") + "Moon size matched to the reference: " + ", ".join(sized)
    best = []
    for si in used:
        mine = ranked[si]
        best += mine[:min(len(mine), max(MIN_KEEP, round(len(mine) * getattr(j, "keep", KEEP))))]
    best = sorted(best, key=lambda f: -f[0] * sources[f[1]]["weight"])[:MAX_KEEP]
    j.step(1, "done", f"Scored {n} frames; keeping the sharpest {len(best)}, about {getattr(j, 'keep', KEEP):.0%} of each video and session"
                      + (f"; {why}" if why else "") + ".")

    # 3. align and stack, all on the sharpest frame from the best source
    j.step(2, "running")
    ref_f = max(best, key=lambda f: (f[1] == ref_si, f[0]))
    # The reference is the mean of the sharpest few frames of that source, lined up on the sharpest: a single
    # frame's noise pulled the patch alignment around.
    top = sorted((f for f in best if f[1] == ref_f[1]), key=lambda f: -f[0])[:REF_N]
    refs = _read(each_frame, ref_f[1], [f[2] for f in top])
    ref = _fit_to(refs.pop(ref_f[2]), sources[ref_f[1]]["s"], target)
    ref_small, scale = _gray_small(ref)
    if refs:
        acc_r = ref.astype("float64")
        for img in refs.values():
            img = _fit_to(img, sources[ref_f[1]]["s"], target)
            acc_r += np.clip(cv2.warpAffine(img, _global_warp(ref_small, _gray_small(img)[0], scale), ref.shape[1::-1],
                                            flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE), 0, None)
        ref = (acc_r / (len(refs) + 1)).astype("float32")
        ref_small, scale = _gray_small(ref)
    ref_g = ref.mean(axis=2).astype("float32")
    h, w = ref_g.shape
    ref_mask = cv2.resize(_disc(ref_small)[0].astype("uint8"), (w, h), interpolation=cv2.INTER_NEAREST)
    disc_mask = ref_mask.astype(bool)
    rm = _disc(ref_small)[0]
    ry, rx = (float(v.mean()) for v in np.nonzero(rm))
    r_d = 2 * np.sqrt(rm.sum() / np.pi)
    ref_lo = cv2.GaussianBlur(ref_small, (0, 0), 2)
    rejected = 0
    per = {si: [0, 0] for si in {f[1] for f in best}}  # stacked, left out
    wanted = {(f[1], f[2]) for f in best}
    score_of = {(f[1], f[2]): f[0] for f in best}
    acc = np.zeros_like(ref, dtype="float64")
    wsum, done, by_kind = 0.0, 0, {"video": 0, "fits": 0}
    for si, k, img in each_frame({f[1] for f in best}):
        if (si, k) not in wanted:
            continue
        j.wait_if_paused()
        if j.stopped():
            raise MoonError("Stopped.")
        src = sources[si]
        img = _fit_to(img, src["s"], (h, w))
        small, _ = _gray_small(img)
        warp = _global_warp(ref_small, small, scale)
        img = np.clip(cv2.warpAffine(img, warp, (w, h), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
                                     borderMode=cv2.BORDER_REPLICATE), 0, None)
        # A frame that didn't line up (its disc off the reference's, or its surface unlike it) is dropped: on
        # 2026-10-23 a few left a ghost Moon in the sky.
        if si != ref_f[1]:  # frames of the reference's own source share its exposure and gamma
            img = _tone_match(img, ref_g, disc_mask)  # before the check: linear raw frames looked unlike the video
        a_small = _gray_small(img)[0]
        am = _disc(a_small)[0]
        if am.sum() < 50 or np.hypot(*(np.array([v.mean() for v in np.nonzero(am)]) - (ry, rx))) > max(2.0, 0.01 * r_d) \
                or _ncc(ref_lo, cv2.GaussianBlur(a_small, (0, 0), 2), rm) < 0.9:
            rejected += 1
            per[si][1] += 1
            continue
        lm = _local_map(ref_g, img.mean(axis=2).astype("float32"), ref_mask)
        if lm is not None:
            img = np.clip(cv2.remap(img, lm[0], lm[1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE), 0, None)
        wgt = src["weight"] * score_of[(si, k)]  # sharper frames count for more
        acc += img * wgt
        wsum += wgt
        done += 1
        per[si][0] += 1
        by_kind[src["kind"]] += 1
        j.part(2, done / len(best))
        if done % 25 == 0:
            j.log(f"Aligned and stacked {done} of {len(best)}")
    if done < 5:
        raise MoonError("Fewer than 5 frames lined up with the reference; nothing to stack.")
    stack = (acc / max(1e-6, wsum)).astype("float32")
    mix = " and ".join(x for x in (f"{by_kind['video']} video frames" if by_kind["video"] else "",
                                   f"{by_kind['fits']} raw frames" if by_kind["fits"] else "") if x)
    j.step(2, "done", f"Stacked {done} frames ({mix}), sharper frames weighted higher: rotation and shift per frame onto "
                      f"a reference averaged from the {len(top)} sharpest, tones matched, then 128 px surface patches "
                      "aligned one by one"
                      + (f"; {rejected} frames that didn't line up were left out" if rejected else "")
                      + (" (" + "; ".join(f"{sources[si]['name']}: {a} stacked, {b} left out" for si, (a, b) in sorted(per.items()))
                         + ")." if len(per) > 1 or rejected else "."))

    # 4. sharpen
    j.step(3, "running")
    sharp, snote = _sharpen(stack, linear=False, up=sources[ref_f[1]]["up"])
    j.step(3, "done", snote)

    # 5. 4K frame
    j.step(4, "running")
    canvas = _centre(sharp)
    stem = work / "result_moon_4k"
    _set_aside(work)
    saved = finishing.save_final(canvas, stem)
    (work / "moon_stack.json").write_text(json.dumps({"date": date, "source": source, "frames_used": done,
                                                      "shots": [s["name"] for s in sources]}, indent=1), encoding="utf-8")
    j.outputs["final"] = saved["tif"]
    out = {"final": saved["tif"], "folder": str(work), "frames": done, "files": saved, "mode": "moon"}
    extra = ""
    try:  # measure and log the run, so the next Moon can say whether it beat the best so far
        from . import __version__, history
        metrics = _moon_metrics(canvas)
        out["run_id"] = history.add_run("moon", f"Moon {date}", "moon", str(work), __version__, metrics)
        extra = "; " + history.compare_note("moon", "moon", metrics, out["run_id"])
    except Exception as e:
        j.log(f"Couldn't log the run ({e}).")
    j.step(4, "done", f"Saved {OUT_W}×{OUT_H}: {saved['tif']} (also PNG, JPG and FITS){extra}.")
    return out


def _moon_metrics(img) -> dict:
    """Fine-detail sharpness on the disc at a common 650 px diameter (so 4K and HD results compare), noise on the
    flattest part of the disc, and clipping: what the 2026-04-30 rounds were judged by."""
    import cv2
    import numpy as np
    g = img.mean(axis=2).astype("float32")
    mask = _disc_mask(g)
    ys, xs = np.nonzero(mask)
    if ys.size < 1000:
        return {}
    d = max(ys.max() - ys.min(), xs.max() - xs.min())
    s = 650 / max(1, d)
    small = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    smask = cv2.resize(mask.astype("uint8"), small.shape[::-1], interpolation=cv2.INTER_NEAREST).astype(bool)
    hp = small - cv2.GaussianBlur(small, (0, 0), 1.5)
    inner = cv2.erode(smask.astype("uint8"), np.ones((25, 25), np.uint8)).astype(bool)
    local = cv2.GaussianBlur(np.abs(hp), (0, 0), 8)
    flat = inner & (local <= np.percentile(local[inner], 10)) if inner.any() else inner
    noise = 1.4826 * float(np.median(np.abs(hp[flat] - np.median(hp[flat])))) if flat.sum() > 100 else 0.0
    sharp = _sharpness(small, smask)
    return {"sharpness": round(sharp, 4), "noise": round(noise, 5), "detail_noise": round(_detail_noise(img), 2),
            "clipped_pct": round(float((g[mask] >= 0.999).mean() * 100), 3)}


def _remove_sky(x):
    """Take a bright or coloured sky (dusk, haze) off each channel: a smooth surface fitted to the sky around the
    disc, subtracted, so the sky ends black and its colour doesn't tint the Moon (2026-10-23: blue-grey sky at a
    quarter of full scale, orange Moon)."""
    import cv2
    import numpy as np
    lum = x.mean(axis=2)
    disc = _disc_mask(lum)
    h, w = lum.shape
    grow = max(15, int(0.04 * max(h, w))) | 1
    sky = ~cv2.dilate(disc.astype("uint8"), np.ones((grow, grow), np.uint8)).astype(bool)
    step = max(1, min(h, w) // 200)
    ys, xs = np.mgrid[0:h:step, 0:w:step]
    sel = sky[::step, ::step]
    if sel.sum() < 200:
        return x
    yy, xx = ys[sel] / h - 0.5, xs[sel] / w - 0.5
    A = np.stack([np.ones_like(yy), xx, yy, xx * xx, yy * yy, xx * yy], 1)
    Y, X = np.mgrid[0:h, 0:w]
    Y, X = Y / h - 0.5, X / w - 0.5
    out = x.copy()
    for c in range(3):
        v = x[..., c][::step, ::step][sel]
        keep = v <= np.percentile(v, 98)  # stray light and the odd star don't pull the fit
        coef = np.linalg.lstsq(A[keep], v[keep], rcond=None)[0]
        model = coef[0] + coef[1] * X + coef[2] * Y + coef[3] * X * X + coef[4] * Y * Y + coef[5] * X * Y
        out[..., c] = x[..., c] - model.astype("float32")
    return np.clip(out, 0, None).astype("float32")


def _align_channels(x):
    """Low in the sky the air spreads the Moon's colours a little (atmospheric dispersion): red and blue images sit
    a fraction of a pixel apart, which shows as cyan and pink fringes on bright edges (2026-10-23, Moon low at dusk).
    Red and blue are shifted onto green, as RegiStax's RGB align does."""
    import cv2
    import numpy as np
    g = x[..., 1].astype("float32")
    win = cv2.createHanningWindow(g.shape[::-1], cv2.CV_32F)
    out = x.copy()
    notes = []
    for c in (0, 2):
        (dx, dy), resp = cv2.phaseCorrelate(g.copy(), x[..., c].astype("float32").copy(), win)
        if resp > 0.2 and 0.05 < np.hypot(dx, dy) < 4:
            m = np.float32([[1, 0, -dx], [0, 1, -dy]])
            out[..., c] = cv2.warpAffine(x[..., c], m, g.shape[::-1], flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            notes.append(f"{'red' if c == 0 else 'blue'} {np.hypot(dx, dy):.2f} px")
    _align_channels.last = notes
    return np.clip(out, 0, None)


def _sharpen(img, linear: bool, up: int):
    """The finish tuned on Kip's 2026-04-30 videos (3.1x the fine detail of Kip's AutoStakkert stack at a better
    detail-to-noise ratio, no ringing) and reworked on the 2026-10-23 dusk video after Kip found it blown out: levels
    set, white balance on the highlands, an edge-preserving brightness denoise, Richardson-Lucy (30 iterations,
    Gaussian PSF), three-scale detail gains, the limb protected, natural mineral colour and a mid-tone brightness curve."""
    import cv2
    import numpy as np
    # Where a channel was near saturation in the video its colour isn't real (10-23: red clipped on 13% of the disc,
    # which turned the brightest craters cyan once colour was boosted).
    near_clip = cv2.GaussianBlur((np.clip(img, 0, None).max(axis=2) > 0.9).astype("float32"), (0, 0), 3)
    x = _align_channels(_remove_sky(np.clip(img, 0, None)))
    lum0 = x.mean(axis=2)
    lit = _disc_mask(lum0)
    if lit.sum() > 100:  # the Moon is close to grey: balance the channels on the bright highlands (the sensor is green-heavy)
        high = lit & (lum0 > np.percentile(lum0[lit], 60))
        means = np.array([x[..., c][high].mean() for c in range(3)])
        x = x * (means.mean() / np.maximum(means, 1e-6))
    sky = float(np.percentile(lum0[~lit], 50)) if (~lit).any() else 0.0
    xc = np.clip(x, 0, 1)  # a few stray spikes mustn't set the white point
    top = float(np.percentile(xc[lit], 99.9)) if lit.any() else float(np.percentile(xc, 99.95))
    x = np.clip((x - sky) / max(1e-6, top - sky), 0, 1.5).astype("float32")
    disc = _disc_mask(x.mean(axis=2)).astype("uint8")
    dmk = disc > 0
    if dmk.sum() < 100:
        dmk = np.ones(disc.shape, bool)
    # Sharpen the brightness only, taken from green (scaled to grey): green holds detail where red clipped.
    gray = cv2.cvtColor(np.clip(x, 0, None), cv2.COLOR_RGB2GRAY)
    kg = float(gray[dmk].mean() / max(1e-6, x[..., 1][dmk].mean()))
    lum = (x[..., 1] * kg).astype("float32")
    # The denoise follows the stack's measured noise (on the calmest 10% of the disc, from green): 0.015-0.02 suited
    # the 2026-10-23 video and 0.0225 the noisier 04-30 one, about 13x the noise. 0.03 on 10-23 smeared texture.
    res = lum - cv2.GaussianBlur(lum, (0, 0), 1.0 * up)
    e25 = 25 * up
    dmn = dmk & (cv2.erode(disc, np.ones((e25, e25), np.uint8)) > 0)
    sc, mad = 0.02, 0.0
    if dmn.sum() > 1000:
        flat = cv2.GaussianBlur(np.abs(lum - cv2.GaussianBlur(lum, (0, 0), 4.0 * up)), (0, 0), 8.0 * up)
        sel = dmn & (flat < np.percentile(flat[dmn], 10))
        r = res[sel]
        mad = float(1.4826 * np.median(np.abs(r - np.median(r))))
        sc = float(np.clip(13 * mad, 0.01, 0.04))
    den = cv2.bilateralFilter(lum, 0, sc, 1.5 * up)
    sigma = 1.2 * up
    est = den.copy()
    for _ in range(RL_ITERS):
        conv = cv2.GaussianBlur(est, (0, 0), sigma)
        est = est * cv2.GaussianBlur(den / np.maximum(conv, 1e-4), (0, 0), sigma)
    b1, b2, b3 = (cv2.GaussianBlur(est, (0, 0), s_ * up) for s_ in (1.0, 2.0, 4.0))
    sharp_l = b3 + 1.2 * (b2 - b3) + 1.5 * (b1 - b2) + 1.8 * (est - b1)
    # The limb and the dark sky stay as stacked: no ringing or halo outside the disc.
    er = 17 * up
    lm = cv2.GaussianBlur(cv2.erode(disc, np.ones((er, er), np.uint8)).astype("float32"), (0, 0), er / 2.2)
    sharp_l = np.clip(sharp_l * lm + lum * (1 - lm), 0, 2)
    # Mineral colour (Kip: "bring out the natural brownish blue tones"): large-scale colour ratios, with the part that
    # follows brightness or position across the disc (channel response, twilight scatter) fitted and removed, the
    # highlands set neutral, then boosted 3x on the disc only.
    s6 = 6.0 * up
    l6 = np.maximum(cv2.GaussianBlur(lum, (0, 0), s6), 1e-3)
    ch = cv2.GaussianBlur(x, (0, 0), s6) / l6[..., None]
    dm = dmn
    if dm.sum() > 1000:
        h, w = lum.shape
        yy, xx = np.mgrid[0:h, 0:w].astype("float32")
        xx = (xx - w / 2) / (w / 2)
        yy = (yy - h / 2) / (h / 2)
        lq = np.log(l6)
        terms = [lq, lq * lq, xx, yy, xx * xx, xx * yy, yy * yy]
        idx = np.flatnonzero(dm)[::7]  # a sample is plenty for 8 coefficients
        A = np.stack([np.ones(idx.size, "float32")] + [t.ravel()[idx] for t in terms], axis=1)
        for c in range(3):
            co = np.linalg.lstsq(A, ch[..., c].ravel()[idx], rcond=None)[0]
            fit = sum(co[i + 1] * t for i, t in enumerate(terms))
            ch[..., c] = ch[..., c] - fit + float(fit[dm].mean())
        ch = ch / np.maximum(ch.mean(axis=2, keepdims=True), 1e-4)
        hl = dm & (lum > np.percentile(lum[dm], 60))
        ch = ch / np.maximum(np.array([ch[..., c][hl].mean() for c in range(3)]), 1e-4)
        # Colour fades out over the outer 40 px: Kip marked teal and pink tints along the limb as false colour.
        e40 = 40 * up
        inner = cv2.GaussianBlur(cv2.erode(disc, np.ones((e40, e40), np.uint8)).astype("float32"), (0, 0), 12.0 * up)
        ch = 1 + MOON_SAT * (ch - 1) * ((1 - np.clip(3 * near_clip, 0, 1)) * inner)[..., None]
    else:
        ch = np.ones_like(x)
    out = np.clip(ch, 0, 3) * sharp_l[..., None]
    # Brightness curve: white point at the disc's 99.97th percentile, a gamma that puts the disc median at 0.40, a soft
    # shoulder from 0.82 and a local contrast lift (deeper maria, brighter rays). 0.17.2 sat at median 0.60
    # with 0.7% of the disc near white: Kip found it blown out.
    L = np.maximum(cv2.cvtColor(np.clip(out, 0, 4).astype("float32"), cv2.COLOR_RGB2GRAY), 1e-5)
    L1 = L / max(1e-5, float(np.percentile(L[dmk], 99.97)))
    med = float(np.median(L1[dmk]))
    gm = float(np.clip(np.log(MOON_MEDIAN) / np.log(med), 0.3, 3.0)) if 0 < med < 1 else 1.0
    L2 = L1 ** gm
    L2 = np.where(L2 > 0.82, 0.82 + 0.18 * np.tanh((L2 - 0.82) / 0.18), L2)
    L2 = np.clip(L2 + 0.35 * (L2 - cv2.GaussianBlur(L2, (0, 0), 50.0 * up)) * dmk, 0, 1)
    # A final crispening of the toned brightness, toward the look of DWARFLAB's own Moon images (crater rims and
    # rays; 2026-10-28: fine detail +25% with no ringing and the same detail/noise).
    L2 = np.clip(L2 + 0.3 * (L2 - cv2.GaussianBlur(L2, (0, 0), 0.7 * up)) * dmk, 0, 1)
    # True black (Kip, 2026-10-28: the sky beside the Moon held 1-2/255 of noise, "not a true black"). A black floor
    # measured well away from the Moon is taken off with a soft toe, then everything outside the Moon's outline (lit
    # part and any dark side, as one convex shape) is set to 0 with a soft edge.
    shape = None
    seen = (L2 > 0.03).astype("uint8")
    n_, lab, st, _ = cv2.connectedComponentsWithStats(seen, 8)
    if n_ > 1:
        big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        shape = np.zeros_like(seen)
        cv2.fillConvexPoly(shape, cv2.convexHull(cv2.findNonZero((lab == big).astype("uint8"))), 1)
        far = cv2.dilate(shape, np.ones((101 * up, 101 * up), np.uint8)) == 0
        bp = float(np.percentile(L2[far], 99.5)) if far.sum() > 1000 else 0.0
        if 0 < bp < 0.2:
            t = 3 * bp
            ln = np.where(L2 < t, np.clip(L2 - bp, 0, None) ** 2 / (2 * (t - bp)), L2 - bp - (t - bp) / 2)
            L2 = (ln / (1 - bp - (t - bp) / 2)).astype("float32")
    out = np.clip(out * (L2 / L)[..., None], 0, 1).astype("float32")
    if shape is not None:
        g7 = 7 * up
        shape = cv2.dilate(shape, np.ones((g7, g7), np.uint8)).astype("float32")
        out = out * cv2.GaussianBlur(shape, (0, 0), 1.5 * up)[..., None]
    rgb = getattr(_align_channels, "last", [])
    return out, ((f"colour channels lined up (atmospheric dispersion: {', '.join(rgb)}); " if rgb else "")
                 + "sky removed (it ends black); "
                 + f"white balance on the highlands, edge-preserving denoise (strength {sc:.3f}, from measured noise {mad:.4f}), Richardson-Lucy ({RL_ITERS} iterations, PSF σ {sigma:.1f}) "
                 "and three-scale detail gains (fine +80%, medium +50%, coarse +20%) on the brightness (from green); "
                 f"natural mineral colour kept faint ({MOON_SAT:g}x) (twilight and brightness-linked colour removed, highlands "
                 f"neutral, none where the video clipped); brightness curve to a disc median of {MOON_MEDIAN:g} with a soft "
                 "highlight shoulder and a gentle local contrast lift; the limb and sky are protected from halos and the sky is set to true black.")


def _centre(img):
    """The Moon in the middle of a 3840x2160 frame (fitted down if the stack is bigger)."""
    import cv2
    import numpy as np
    h, w = img.shape[:2]
    if w > OUT_W or h > OUT_H:
        s = min(OUT_W / w, OUT_H / h)
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
    lum = img.mean(axis=2)
    m = _disc_mask(lum)  # the disc, whatever its brightness and the sky's
    ys, xs = np.nonzero(m)
    cy, cx = (float(ys.mean()), float(xs.mean())) if ys.size else (h / 2, w / 2)
    canvas = np.zeros((OUT_H, OUT_W, 3), "float32")
    oy, ox = round(OUT_H / 2 - cy), round(OUT_W / 2 - cx)
    y0, x0 = max(0, oy), max(0, ox)
    y1, x1 = min(OUT_H, oy + h), min(OUT_W, ox + w)
    canvas[y0:y1, x0:x1] = img[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
    return canvas

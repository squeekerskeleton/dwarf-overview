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
MAX_KEEP, MIN_KEEP = 1000, 50
REF_N = 8              # the alignment reference is the mean of this many of the sharpest frames
ADD_MIN = 0.98         # a further source joins only if the combined stack keeps this share of the sharpness
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
    """The session's frames as local files (Wi-Fi sessions are copied into the work folder first)."""
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


def _video_frames(path: Path):
    """(index, float RGB 0..1) for every frame."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    try:
        k = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
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


def _disc(g):
    """Mask of the lit Moon in a small grey frame, and how bright it is."""
    import numpy as np
    from .lunar import _otsu
    norm = g / max(1e-6, float(np.percentile(g, 99.9)))
    thr = max(_otsu(np.clip(norm, 0, 1)), 0.08)
    m = norm > thr
    return m, float(g[m].mean()) if m.any() else 0.0


def _score(img) -> tuple[float, float, bool]:
    """Sharpness (Laplacian energy inside the disc, relative to brightness), brightness, and whether the disc
    touches the edge of the frame."""
    import cv2
    import numpy as np
    g, _ = _gray_small(img)
    m, bright = _disc(g)
    if m.mean() < 0.005 or m.mean() > 0.7:  # no Moon, or glare/cloud filling the frame
        return 0.0, bright, True
    inner = cv2.erode(m.astype("uint8"), np.ones((7, 7), np.uint8)).astype(bool)
    lap = cv2.Laplacian(g, cv2.CV_32F)
    s = float(np.sqrt(np.mean(lap[inner] ** 2)) / max(bright, 1e-4)) if inner.any() else 0.0
    edge = bool(m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any())
    return s, bright, edge


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
        mask = g > np.percentile(g, 99.5) * 0.25
    inner = cv2.erode(mask.astype("uint8"), np.ones((25, 25), np.uint8)).astype(bool)
    if inner.sum() < 1000:
        return 0.0
    lap = cv2.Laplacian(cv2.GaussianBlur(g, (0, 0), 0.7), cv2.CV_32F)
    return float(np.sqrt(np.mean(lap[inner] ** 2)) / max(1e-6, float(g[inner].mean())))


def _read(each_frame, si: int, keys: list) -> dict:
    """The given frames of one source, read in one pass (stopping once all are found)."""
    want, out = set(keys), {}
    for s_, k, img in each_frame({si}):
        if k in want:
            out[k] = img
            if len(out) == len(want):
                break
    return out


def _screen(j, sources: list, ranked: dict, each_frame, trial: int = 20) -> tuple[list, str]:
    """Which sources to stack: never one that costs resolution (Kip: "if it degrades resolution, do not stack
    together"). A source whose median frame is well below the sharpest source of the same size is left out (on
    2026-04-30 one video's 0.032 against 0.055; stacking it in cut the sharpness from 0.102 to 0.075). Then each
    source's best frames are stacked on their own at the common 4K scale, and, sharpest first, each next source
    joins only if the combined stack keeps ADD_MIN of the sharpness. Returns (source indexes, note)."""
    import cv2
    import numpy as np
    left, notes = [], []
    med = {si: float(np.median([f[0] for f in fr])) for si, fr in ranked.items()}
    for si in list(ranked):
        same = [o for o in ranked if sources[o]["up"] == sources[si]["up"]]
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
        imgs = list(_read(each_frame, si, [f[2] for f in ranked[si][:trial]]).values())
        if sources[si]["up"] > 1:
            imgs = [cv2.resize(i, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC) for i in imgs]
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
    order = sorted(stacks, key=lambda si: -_sharpness(stacks[si]))
    first = stacks[order[0]]
    f_small, f_sc = _gray_small(first)
    f_g = first.mean(axis=2).astype("float32")
    f_mask = cv2.resize(_disc(f_small)[0].astype("uint8"), f_g.shape[::-1], interpolation=cv2.INTER_NEAREST).astype(bool)
    used = [order[0]]
    acc, wsum = first.astype("float64") * len(ranked[order[0]]), float(len(ranked[order[0]]))
    cur = _sharpness(first)
    for si in order[1:]:
        img = stacks[si]
        if img.shape != first.shape:
            img = cv2.resize(img, first.shape[1::-1], interpolation=cv2.INTER_CUBIC)
        img = np.clip(cv2.warpAffine(img, _global_warp(f_small, _gray_small(img)[0], f_sc), first.shape[1::-1],
                                     flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE), 0, None)
        img = _tone_match(img, f_g, f_mask)
        n = float(len(ranked[si]))
        trial_s = _sharpness(((acc + img * n) / (wsum + n)).astype("float32"))
        if trial_s >= ADD_MIN * cur:
            used.append(si)
            acc, wsum, cur = acc + img * n, wsum + n, trial_s
        else:
            notes.append(f"{sources[si]['name']} left out (it would blur the stack: sharpness {trial_s:.3f} against {cur:.3f})")
    for si in cand:
        if si not in stacks:
            used.append(si)  # couldn't be tested: keep the old behaviour
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


def start(date: str, source: str = "all", include: str | None = None, exclude: list[str] | None = None) -> str:
    """Stack a night. include: a file the person opened (always used, even from outside the library and the
    DWARF); exclude: names of videos or sessions they unticked."""
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
    threading.Thread(target=_run, args=(j, items, source, work, date), daemon=True).start()
    return j.id


def _run(j, items, source, work, date) -> None:
    try:
        out = _pipeline(j, items, source, work, date)
        if not j.stopped():
            from .autoprocess import archive
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
    # Everything is brought to 4K: 4K videos as they are, HD videos and raw FITS (1920x1080) at 2x.
    sources = []
    for v in videos:
        w, h, n = vinfo[v["name"]]
        sources.append({**v, "up": 1 if w >= 3000 else 2, "n": n, "weight": 1.0 if w >= 3000 else 0.6})
    for f in fits:
        sources.append({**f, "up": 2, "n": f.get("frames") or 120, "weight": 0.7})
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

    def each_frame(only=None):
        for si, s in enumerate(sources):
            if only is not None and si not in only:
                continue
            if s["kind"] == "video":
                for k, img in _video_frames(_video_path(s, work, j)):
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
    n = 0
    for si, k, img in each_frame():
        j.wait_if_paused()
        if j.stopped():
            raise MoonError("Stopped.")
        s, bright, edge = _score(img)
        if s > 0 and not edge:
            frames.append((s, si, k, bright))
        n += 1
        if n % 50 == 0:
            j.part(1, n / max(1, total))
    if len(frames) < 5:
        raise MoonError("Fewer than 5 usable Moon frames (the disc was missing, cut off by the frame edge or lost in glare).")
    # Each source is ranked against itself (resolution, exposure and gain differ), and the best 5% of each kept.
    ranked = {si: sorted((f for f in frames if f[1] == si), key=lambda f: -f[0]) for si in sorted({f[1] for f in frames})}
    used, why = _screen(j, sources, ranked, each_frame)
    best = []
    for si in used:
        mine = ranked[si]
        best += mine[:min(len(mine), max(MIN_KEEP, round(len(mine) * KEEP)))]
    best = sorted(best, key=lambda f: -f[0] * sources[f[1]]["weight"])[:MAX_KEEP]
    j.step(1, "done", f"Scored {n} frames; keeping the sharpest {len(best)}, about 5% of each video and session"
                      + (f"; {why}" if why else "") + ".")

    # 3. align and stack, all on the sharpest frame from the best source
    j.step(2, "running")
    ref_f = max(best, key=lambda f: (sources[f[1]]["up"] == 1, f[0]))
    # The reference is the mean of the sharpest few frames of that source, lined up on the sharpest: a single
    # frame's noise pulled the patch alignment around.
    top = sorted((f for f in best if f[1] == ref_f[1]), key=lambda f: -f[0])[:REF_N]
    refs = _read(each_frame, ref_f[1], [f[2] for f in top])
    ref = refs.pop(ref_f[2])
    if sources[ref_f[1]]["up"] > 1:
        ref = cv2.resize(ref, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    ref_small, scale = _gray_small(ref)
    if refs:
        acc_r = ref.astype("float64")
        for img in refs.values():
            if sources[ref_f[1]]["up"] > 1:
                img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            acc_r += np.clip(cv2.warpAffine(img, _global_warp(ref_small, _gray_small(img)[0], scale), ref.shape[1::-1],
                                            flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE), 0, None)
        ref = (acc_r / (len(refs) + 1)).astype("float32")
        ref_small, scale = _gray_small(ref)
    ref_g = ref.mean(axis=2).astype("float32")
    h, w = ref_g.shape
    ref_mask = cv2.resize(_disc(ref_small)[0].astype("uint8"), (w, h), interpolation=cv2.INTER_NEAREST)
    disc_mask = ref_mask.astype(bool)
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
        if src["up"] > 1:
            img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        if img.shape[:2] != (h, w):  # a different resolution than the reference: fit it
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
        small, _ = _gray_small(img)
        warp = _global_warp(ref_small, small, scale)
        img = np.clip(cv2.warpAffine(img, warp, (w, h), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
                                     borderMode=cv2.BORDER_REPLICATE), 0, None)
        if si != ref_f[1]:  # frames of the reference's own source share its exposure and gamma
            img = _tone_match(img, ref_g, disc_mask)
        lm = _local_map(ref_g, img.mean(axis=2).astype("float32"), ref_mask)
        if lm is not None:
            img = np.clip(cv2.remap(img, lm[0], lm[1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE), 0, None)
        wgt = src["weight"] * score_of[(si, k)]  # sharper frames count for more
        acc += img * wgt
        wsum += wgt
        done += 1
        by_kind[src["kind"]] += 1
        j.part(2, done / len(best))
        if done % 25 == 0:
            j.log(f"Aligned and stacked {done} of {len(best)}")
    stack = (acc / max(1e-6, wsum)).astype("float32")
    mix = " and ".join(x for x in (f"{by_kind['video']} video frames" if by_kind["video"] else "",
                                   f"{by_kind['fits']} raw frames" if by_kind["fits"] else "") if x)
    j.step(2, "done", f"Stacked {done} frames ({mix}), sharper frames weighted higher: rotation and shift per frame onto "
                      f"a reference averaged from the {len(top)} sharpest, tones matched, then 128 px surface patches "
                      "aligned one by one.")

    # 4. sharpen
    j.step(3, "running")
    sharp, snote = _sharpen(stack, linear=False, up=sources[ref_f[1]]["up"])
    j.step(3, "done", snote)

    # 5. 4K frame
    j.step(4, "running")
    canvas = _centre(sharp)
    stem = work / "result_moon_4k"
    saved = finishing.save_final(canvas, stem)
    (work / "moon_stack.json").write_text(json.dumps({"date": date, "source": source, "frames_used": done,
                                                      "shots": [s["name"] for s in sources]}, indent=1), encoding="utf-8")
    j.outputs["final"] = saved["tif"]
    j.step(4, "done", f"Saved {OUT_W}×{OUT_H}: {saved['tif']} (also PNG, JPG and FITS).")
    return {"final": saved["tif"], "folder": str(work), "frames": done, "files": saved, "mode": "moon"}


def _sharpen(img, linear: bool, up: int):
    """The finish tuned on Kip's 2026-04-30 videos (3.1x the fine detail of Kip's AutoStakkert stack at a better
    detail-to-noise ratio, no ringing): levels set, white balance on the highlands, an edge-preserving brightness
    denoise, Richardson-Lucy (15 iterations, Gaussian PSF), three-scale detail gains, the limb protected."""
    import cv2
    import numpy as np
    x = np.clip(img, 0, None)
    lum0 = x.mean(axis=2)
    lit = lum0 > np.percentile(lum0, 99.5) * 0.25
    if lit.sum() > 100:  # the Moon is close to grey: balance the channels on the bright highlands (the sensor is green-heavy)
        high = lit & (lum0 > np.percentile(lum0[lit], 60))
        means = np.array([x[..., c][high].mean() for c in range(3)])
        x = x * (means.mean() / np.maximum(means, 1e-6))
    sky = float(np.percentile(lum0[~lit], 50)) if (~lit).any() else 0.0
    xc = np.clip(x, 0, 1)  # a few stray spikes mustn't set the white point
    top = float(np.percentile(xc[lit], 99.9)) if lit.any() else float(np.percentile(xc, 99.95))
    x = np.clip((x - sky) / max(1e-6, top - sky), 0, 1.5).astype("float32")
    base = x.copy()
    lum = x.mean(axis=2).astype("float32")
    den = cv2.bilateralFilter(lum, 0, 0.03, 1.5 * up)
    x = x * (den / np.maximum(lum, 1e-5))[..., None]
    sigma = 1.2 * up
    est = x.copy()
    for _ in range(15):
        conv = cv2.GaussianBlur(est, (0, 0), sigma)
        est = est * cv2.GaussianBlur(x / np.maximum(conv, 1e-4), (0, 0), sigma)
    b1, b2, b3 = (cv2.GaussianBlur(est, (0, 0), s_ * up) for s_ in (1.0, 2.0, 4.0))
    out = b3 + 1.2 * (b2 - b3) + 1.5 * (b1 - b2) + 1.8 * (est - b1)
    # The limb and the dark sky stay as stacked: no ringing or halo outside the disc.
    er = 17 * up
    disc = (base.mean(axis=2) > np.percentile(base.mean(axis=2), 99.5) * 0.25).astype("uint8")
    lm = cv2.GaussianBlur(cv2.erode(disc, np.ones((er, er), np.uint8)).astype("float32"), (0, 0), er / 2.2)[..., None]
    out = out * lm + base * (1 - lm)
    # A soft shoulder instead of a hard clip: 0.16.1 clipped 0.77% of the disc (Tycho and the brightest rays).
    out = np.where(out > 0.9, 0.9 + 0.1 * np.tanh((out - 0.9) / 0.1), out)
    out = np.clip(out, 0, 1)
    return out, (f"white balance on the highlands, edge-preserving denoise, Richardson-Lucy (15 iterations, PSF σ {sigma:.1f}) "
                 "and three-scale detail gains (fine +80%, medium +50%, coarse +20%); the limb and sky are protected from halos.")


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
    m = lum > max(1e-4, float(np.percentile(lum, 99.5)) * 0.25)  # the disc, whatever its brightness
    ys, xs = np.nonzero(m)
    cy, cx = (float(ys.mean()), float(xs.mean())) if ys.size else (h / 2, w / 2)
    canvas = np.zeros((OUT_H, OUT_W, 3), "float32")
    oy, ox = round(OUT_H / 2 - cy), round(OUT_W / 2 - cx)
    y0, x0 = max(0, oy), max(0, ox)
    y1, x1 = min(OUT_H, oy + h), min(OUT_W, ox + w)
    canvas[y0:y1, x0:x1] = img[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
    return canvas

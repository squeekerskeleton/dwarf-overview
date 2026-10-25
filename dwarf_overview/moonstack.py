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

4K videos are preferred when there are enough frames: they're full sensor resolution. Raw FITS frames (half
resolution) are stacked at 2x so they still make a 4K image.
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
KEEP = 0.10            # share of frames kept
MAX_KEEP, MIN_KEEP = 500, 30
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
    out = []
    for d in sorted(by.values(), key=lambda d: d["date"], reverse=True):
        out.append({**d, "sources": sorted(d["sources"]), "first": d["first"].strftime("%H:%M"),
                    "last": d["last"].strftime("%H:%M")})
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


def _local_map(ref_g, img_g, mask, patch: int = 160, step: int = 128, max_shift: float = 6.0):
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
            (dx, dy), resp = cv2.phaseCorrelate(r, img_g[y:y + patch, x:x + patch], win)
            if resp > 0.15 and abs(dx) < max_shift and abs(dy) < max_shift:
                gx[a, b], gy[a, b], ok[a, b] = dx, dy, True
    if ok.sum() < 4:
        return None
    for g in (gx, gy):
        g[~ok] = float(np.median(g[ok]))
        g[:] = cv2.GaussianBlur(g, (3, 3), 0.8)
    fx = cv2.resize(gx, (w, h), interpolation=cv2.INTER_LINEAR)
    fy = cv2.resize(gy, (w, h), interpolation=cv2.INTER_LINEAR)
    X, Y = np.meshgrid(np.arange(w, dtype="float32"), np.arange(h, dtype="float32"))
    return X + fx, Y + fy


# ---------------------------------------------------------------- the job

def start(date: str, source: str = "auto") -> str:
    from .autoprocess import _Job, work_root
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date or ""):
        raise MoonError("Pick a date.")
    items = [i for i in _all_items() if i["when"].strftime("%Y-%m-%d") == date]
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
            j.job.update(state="done", output=out)
    except Exception as e:
        if not j.stopped():
            for s in j.job["steps"]:
                if s["state"] == "running":
                    s["state"] = "failed"
            j.job.update(state="failed", error=str(e))


def _pipeline(j, items, source, work: Path, date: str) -> dict:
    import numpy as np
    import cv2

    # 1. which shots
    j.step(0, "running")
    keep, left = _cluster(items)
    videos = [i for i in keep if i["kind"] == "video"]
    fits = [i for i in keep if i["kind"] == "fits"]
    vinfo = {}
    for v in videos:
        try:
            vinfo[v["name"]] = _video_info(_video_path(v, work, j))
        except Exception as e:
            j.log(f"Skipping {v['name']}: {e}")
    big = [v for v in videos if v["name"] in vinfo and vinfo[v["name"]][0] >= 3000]
    big_frames = sum(vinfo[v["name"]][2] for v in big)
    use = source if source in ("video", "fits") else ("video" if big_frames >= 300 else "fits" if fits else "video")
    if use == "video":
        chosen = big or [v for v in videos if v["name"] in vinfo]
        if chosen and not big:  # only HD videos: stack at their size, then fit to 4K
            res = max(vinfo[v["name"]][:2] for v in chosen)
            chosen = [v for v in chosen if vinfo[v["name"]][:2] == res]
    else:
        chosen = fits
    if not chosen:
        raise MoonError("None of that night's shots can be stacked here (no Moon videos or raw Moon sessions).")
    note = (f"{len(chosen)} {'video' if use == 'video' else 'raw session'}{'s' if len(chosen) > 1 else ''} from "
            f"{min(i['when'] for i in chosen):%H:%M} to {max(i['when'] for i in chosen):%H:%M}"
            + (f" ({big_frames} 4K video frames, full sensor resolution)" if use == "video" and big else
               " (raw 16-bit frames, stacked at 2x for 4K)" if use == "fits" else ""))
    if left:
        note += f"; left out {len(left)} shot(s) taken more than {CLUSTER_MIN} min from the rest"
    j.step(0, "done", note + ".")

    # 2. score every frame
    j.step(1, "running")
    bayer: dict = {}
    frames = []  # (score, source index, frame index or path, brightness)

    def each_frame():
        for si, it in enumerate(chosen):
            if use == "video":
                p = _video_path(it, work, j)
                for k, img in _video_frames(p):
                    yield si, k, img
            else:
                from astropy.io import fits as pf
                for f in _fits_frames(it, work, j):
                    try:
                        yield si, f, _debayer(np.asarray(pf.getdata(f)), bayer)
                    except (OSError, ValueError, TypeError):
                        continue

    total = sum(vinfo[i["name"]][2] for i in chosen) if use == "video" else sum(i.get("frames") or 120 for i in chosen)
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
    if use == "fits":  # sessions differ in exposure and gain, so rank each against its own median
        meds = {si: float(np.median([f[0] for f in frames if f[1] == si])) or 1.0 for si in {f[1] for f in frames}}
        frames = [(f[0] / meds[f[1]], *f[1:]) for f in frames]
    frames.sort(key=lambda f: -f[0])
    n_keep = int(min(MAX_KEEP, max(MIN_KEEP, len(frames) * KEEP)))
    best = frames[:n_keep]
    j.step(1, "done", f"Scored {n} frames; keeping the sharpest {len(best)} ({100 * len(best) / len(frames):.0f}% of the usable ones).")

    # 3. align and stack
    j.step(2, "running")
    up = 2 if use == "fits" else 1
    wanted = {(f[1], f[2]): f for f in best}
    ref_key = (best[0][1], best[0][2])
    ref = None
    for si, k, img in each_frame():
        if (si, k) == ref_key:
            ref = img
            break
    if up > 1:
        ref = cv2.resize(ref, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)
    ref_small, scale = _gray_small(ref)
    ref_g = ref.mean(axis=2).astype("float32")
    ref_mask_small, ref_bright = _disc(ref_small)
    ref_mask = cv2.resize(ref_mask_small.astype("uint8"), (ref_g.shape[1], ref_g.shape[0]), interpolation=cv2.INTER_NEAREST)
    acc = np.zeros_like(ref, dtype="float64")
    used, done = 0, 0
    for si, k, img in each_frame():
        if (si, k) not in wanted:
            continue
        j.wait_if_paused()
        if j.stopped():
            raise MoonError("Stopped.")
        f = wanted[(si, k)]
        if up > 1:
            img = cv2.resize(img, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)
        img = img * (ref_bright / max(1e-6, f[3]))  # even out exposure differences between sessions
        small, _ = _gray_small(img)
        warp = _global_warp(ref_small, small, scale)
        h, w = ref_g.shape
        img = cv2.warpAffine(img, warp, (w, h), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
        lm = _local_map(ref_g, img.mean(axis=2).astype("float32"), ref_mask)
        if lm is not None:
            img = cv2.remap(img, lm[0], lm[1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        acc += img
        used += 1
        done += 1
        j.part(2, done / len(best))
        if done % 25 == 0:
            j.log(f"Aligned and stacked {done} of {len(best)}")
    stack = (acc / max(1, used)).astype("float32")
    j.step(2, "done", f"Stacked {used} frames: rotation and shift per frame, then surface patches aligned one by one.")

    # 4. sharpen
    j.step(3, "running")
    sharp, snote = _sharpen(stack, linear=(use == "fits"), up=up)
    j.step(3, "done", snote)

    # 5. 4K frame
    j.step(4, "running")
    canvas = _centre(sharp)
    stem = work / "result_moon_4k"
    saved = finishing.save_final(canvas, stem)
    (work / "moon_stack.json").write_text(json.dumps({"date": date, "source": use, "frames_used": used,
                                                      "shots": [i["name"] for i in chosen]}, indent=1), encoding="utf-8")
    j.outputs["final"] = saved["tif"]
    j.step(4, "done", f"Saved {OUT_W}×{OUT_H}: {saved['tif']} (also PNG, JPG and FITS).")
    return {"final": saved["tif"], "folder": str(work), "frames": used, "files": saved, "mode": "moon"}


def _sharpen(img, linear: bool, up: int):
    """Richardson-Lucy with a Gaussian PSF, then a light multi-scale unsharp mask, limb protected."""
    import cv2
    import numpy as np
    x = np.clip(img, 0, None)
    lum0 = x.mean(axis=2)
    lit = lum0 > np.percentile(lum0, 99.5) * 0.25
    if lit.any():  # the Moon is close to grey: balance the channels on the lit disc (the sensor itself is green-heavy)
        means = np.array([x[..., c][lit].mean() for c in range(3)])
        x = x * (means.mean() / np.maximum(means, 1e-6))
    sky = float(np.percentile(lum0[~lit], 50)) if (~lit).any() else 0.0
    top = float(np.percentile(x[lit], 99.9)) if lit.any() else float(np.percentile(x, 99.95))
    x = np.clip((x - sky) / max(1e-6, top - sky), 0, None)
    sigma = 1.6 if up > 1 else 1.2
    est = x.copy()
    for _ in range(12):
        conv = cv2.GaussianBlur(est, (0, 0), sigma)
        est = est * cv2.GaussianBlur(x / np.maximum(conv, 1e-4), (0, 0), sigma)
    fine = est - cv2.GaussianBlur(est, (0, 0), 1.0)
    mid = cv2.GaussianBlur(est, (0, 0), 1.0) - cv2.GaussianBlur(est, (0, 0), 3.0)
    out = est + 0.6 * fine + 0.35 * mid
    # Keep the limb and the dark sky clean: no ringing or halo outside the disc.
    lum = x.mean(axis=2)
    disc = cv2.GaussianBlur((lum > 0.06).astype("float32"), (0, 0), 3)[..., None]
    out = out * disc + x * (1 - disc)
    lo = float(np.percentile(out.mean(axis=2), 2))
    out = np.clip((out - lo) / max(1e-6, float(np.percentile(out, 99.97)) - lo), 0, 1)
    return out, f"Richardson-Lucy (12 iterations, PSF σ {sigma}) and a light two-scale sharpen; the limb and sky are protected from halos."


def _centre(img):
    """The Moon in the middle of a 3840x2160 frame (fitted down if the stack is bigger)."""
    import cv2
    import numpy as np
    h, w = img.shape[:2]
    if w > OUT_W or h > OUT_H:
        s = min(OUT_W / w, OUT_H / h)
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
    m = img.mean(axis=2) > 0.1
    ys, xs = np.nonzero(m)
    cy, cx = (float(ys.mean()), float(xs.mean())) if ys.size else (h / 2, w / 2)
    canvas = np.zeros((OUT_H, OUT_W, 3), "float32")
    oy, ox = round(OUT_H / 2 - cy), round(OUT_W / 2 - cx)
    y0, x0 = max(0, oy), max(0, ox)
    y1, x1 = min(OUT_H, oy + h), min(OUT_W, ox + w)
    canvas[y0:y1, x0:x1] = img[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
    return canvas

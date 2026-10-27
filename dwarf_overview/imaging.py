"""Load DWARF 3 images and measure the things that decide how to process them.

Everything here runs locally and costs nothing. The numbers it produces are sent
to Claude alongside a preview, so the assessment is grounded in measurements
rather than only in how the picture looks.
"""
from __future__ import annotations

import base64
import io
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_EXTS = {".fits", ".fit", ".fts", ".tif", ".tiff", ".png", ".jpg", ".jpeg"}

# DWARF 3 telephoto: 150 mm focal length, 2.0 um pixels (Sony IMX678).
DWARF3_TELE_ARCSEC_PER_PX = 206.265 * 2.0 / 150.0
DWARF3_NATIVE_SIZES = {(3840, 2160), (2160, 3840)}

ANALYSIS_LONG_EDGE = 2048
PREVIEW_LONG_EDGE = 1568
CROP_SIZE = 768


@dataclass
class LoadedImage:
    path: Path
    data: np.ndarray  # float32, H x W x C, 0..1
    source_format: str
    bit_depth: str
    shot_info: dict = field(default_factory=dict)

    @property
    def is_color(self) -> bool:
        return self.data.shape[2] == 3


# ---------------------------------------------------------------- loading

def _normalize(arr: np.ndarray) -> tuple[np.ndarray, str]:
    if np.issubdtype(arr.dtype, np.integer):
        depth = f"{arr.dtype.itemsize * 8}-bit integer"
        out = arr.astype(np.float32) / float(np.iinfo(arr.dtype).max)
    else:
        depth = "32-bit float" if arr.dtype == np.float32 else f"{arr.dtype} float"
        out = arr.astype(np.float32)
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        lo, hi = float(out.min()), float(out.max())
        if lo < 0 or hi > 1.0:
            # Some FITS writers store raw ADU as float; rescale into 0..1.
            out = (out - min(lo, 0.0)) / max(hi - min(lo, 0.0), 1e-12)
    return np.clip(out, 0.0, 1.0), depth


def _to_hwc(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return arr[:, :, None]
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[2] not in (1, 3, 4):
        arr = np.moveaxis(arr, 0, -1)
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if arr.shape[2] == 1:
        return arr
    return arr[:, :, :3]


def find_shot_info(path: Path) -> dict:
    for folder in (path.parent, path.parent.parent):
        candidate = folder / "shotsInfo.json"
        if candidate.is_file():
            try:
                return json.loads(candidate.read_text(encoding="utf-8", errors="replace"))
            except (OSError, ValueError):
                return {}
    return {}


def load_image(path: str | Path) -> LoadedImage:
    path = Path(path).expanduser()
    ext = path.suffix.lower()
    if ext not in IMAGE_EXTS:
        raise ValueError(f"Unsupported file type {ext!r}. Use FITS, TIFF, PNG or JPG.")

    if ext in (".fits", ".fit", ".fts"):
        from astropy.io import fits

        with fits.open(path, memmap=False) as hdul:
            hdu = next((h for h in hdul if h.data is not None and h.data.ndim >= 2), None)
            if hdu is None:
                raise ValueError("FITS file has no image data.")
            raw = np.asarray(hdu.data)
            # FITS stores unsigned 16-bit as signed + BZERO; astropy applies it, giving float.
        fmt = "FITS"
    elif ext in (".tif", ".tiff"):
        import tifffile

        raw = tifffile.imread(path)
        fmt = "TIFF"
    else:
        with Image.open(path) as im:
            if im.mode in ("I;16", "I;16B", "I;16L", "I"):
                raw = np.asarray(im).astype(np.uint16)
            else:
                raw = np.asarray(im.convert("RGB"))
        fmt = ext.lstrip(".").upper().replace("JPEG", "JPG")

    data, depth = _normalize(_to_hwc(raw))
    return LoadedImage(path=path, data=data, source_format=fmt, bit_depth=depth,
                       shot_info=find_shot_info(path))


# ---------------------------------------------------------------- helpers

def _downsample(data: np.ndarray, long_edge: int) -> tuple[np.ndarray, int]:
    h, w = data.shape[:2]
    factor = max(1, math.ceil(max(h, w) / long_edge))
    if factor == 1:
        return data, 1
    h2, w2 = h // factor, w // factor
    trimmed = data[: h2 * factor, : w2 * factor]
    return trimmed.reshape(h2, factor, w2, factor, -1).mean(axis=(1, 3)), factor


def _box_blur3(img: np.ndarray) -> np.ndarray:
    p = np.pad(img, 1, mode="edge")
    return sum(p[dy:dy + img.shape[0], dx:dx + img.shape[1]] for dy in range(3) for dx in range(3)) / 9.0


def _mad_sigma(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    med = np.median(values)
    return float(1.4826 * np.median(np.abs(values - med)))


def _resize(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(Image.fromarray(img.astype(np.float32), mode="F").resize(size, Image.BILINEAR))


def _side_name(dx: float, dy: float) -> str:
    vert = "bottom" if dy > 0 else "top"
    horiz = "right" if dx > 0 else "left"
    ax, ay = abs(dx), abs(dy)
    if ax > 2 * ay:
        return horiz
    if ay > 2 * ax:
        return vert
    return f"{vert}-{horiz}"


# ---------------------------------------------------------------- metrics

def measure(img: LoadedImage) -> dict:
    small, factor = _downsample(img.data, ANALYSIS_LONG_EDGE)
    lum = small.mean(axis=2)
    h, w = lum.shape

    # Background: median of an 8x8 grid of tiles, then a plane fit for direction.
    grid = 8
    th, tw = h // grid, w // grid
    cells = np.array([[np.median(lum[r * th:(r + 1) * th, c * tw:(c + 1) * tw]) for c in range(grid)]
                      for r in range(grid)], dtype=np.float32)
    bg_level = float(np.median(cells))
    yy, xx = np.mgrid[0:grid, 0:grid]
    design = np.column_stack([xx.ravel() - (grid - 1) / 2, yy.ravel() - (grid - 1) / 2, np.ones(grid * grid)])
    (gx, gy, _), *_ = np.linalg.lstsq(design, cells.ravel(), rcond=None)
    bg_map = _resize(cells, (w, h))
    flat = lum - bg_map

    # Noise from the high-pass residual of background pixels.
    resid = lum - _box_blur3(lum)
    rough = _mad_sigma(flat)
    bg_mask = flat < 3 * max(rough, 1e-6)
    noise = _mad_sigma(resid[bg_mask]) / math.sqrt(8.0 / 9.0)
    noise = max(noise, 1e-6)

    spread = float(np.percentile(cells, 95) - np.percentile(cells, 5))
    gradient = {
        "spread_percent_of_background": round(100 * spread / max(bg_level, 1e-4), 1),
        "spread_in_noise_sigmas": round(spread / noise, 1),
        "brighter_toward": _side_name(gx, gy) if spread > 0 else "none",
        "tile_medians_8x8_relative": [[round(float(v), 3) for v in row] for row in cells / max(bg_level, 1e-6)],
    }

    # Stretch state and clipping.
    median_l = float(np.median(lum))
    p99 = float(np.percentile(lum, 99))
    full = img.data
    stretch = {
        "median_luminance": round(median_l, 4),
        "p99_luminance": round(p99, 4),
        "state": "linear (unstretched)" if median_l < 0.06 else ("stretched" if median_l > 0.12 else "lightly stretched"),
        "saturated_pixels_percent": round(100 * float(np.mean(full.max(axis=2) >= 0.999)), 3),
        "black_clipped_percent": round(100 * float(np.mean(full.max(axis=2) <= 0.0005)), 3),
    }

    # Colour: background and signal balance per channel.
    color = None
    if img.is_color:
        bg_px = small[bg_mask]
        bg_ch = np.median(bg_px, axis=0)
        sig_mask = flat > 10 * noise
        sig_ch = np.median(small[sig_mask], axis=0) if sig_mask.sum() > 200 else bg_ch
        g = max(float(bg_ch[1]), 1e-6)
        ratios = {"R/G": float(bg_ch[0] / g), "B/G": float(bg_ch[2] / g)}
        dev = {k: abs(v - 1) for k, v in ratios.items()}
        worst = max(dev, key=dev.get)
        if dev[worst] < 0.05:
            cast = "neutral"
        elif bg_ch[1] > bg_ch[0] and bg_ch[1] > bg_ch[2]:
            cast = "green"
        else:
            cast = ["red", "green", "blue"][int(np.argmax(bg_ch))]
        color = {
            "background_rgb": [round(float(v), 5) for v in bg_ch],
            "background_ratios": {k: round(v, 3) for k, v in ratios.items()},
            "signal_rgb": [round(float(v), 5) for v in sig_ch],
            "background_cast": cast,
        }

    stars = _measure_stars(flat, noise, factor, img)
    positions = stars.pop("_positions", [])
    spectral = _color_emission(small, bg_map, flat, noise, positions) if img.is_color else None

    signal = float(np.percentile(lum, 99.5)) - bg_level
    return {
        "image": {
            "file": img.path.name,
            "format": img.source_format,
            "bit_depth": img.bit_depth,
            "width": int(img.data.shape[1]),
            "height": int(img.data.shape[0]),
            "channels": int(img.data.shape[2]),
            "analysis_downsample": factor,
        },
        "background": {"level": round(bg_level, 5), "noise_sigma_at_analysis_scale": round(noise, 6),
                       "snr_proxy_p99_5": round(signal / noise, 1)},
        "gradient": gradient,
        "stretch": stretch,
        "color": color,
        "stars": stars,
        "color_emission": spectral,
    }


def _measure_stars(flat: np.ndarray, noise: float, factor: int, img: LoadedImage) -> dict:
    from numpy.lib.stride_tricks import sliding_window_view

    h, w = flat.shape
    r = 2
    padded = np.pad(flat, r, mode="constant", constant_values=-np.inf)
    local_max = sliding_window_view(padded, (2 * r + 1, 2 * r + 1)).max(axis=(2, 3))
    peaks = (flat == local_max) & (flat > 6 * noise)
    border = 10
    peaks[:border] = peaks[-border:] = False
    peaks[:, :border] = peaks[:, -border:] = False
    ys, xs = np.nonzero(peaks)
    count = int(len(ys))
    if count == 0:
        return {"detected": 0, "note": "No stars detected above 6 sigma."}

    vals = flat[ys, xs]
    peak_ceiling = 0.85 * float(flat.max())
    order = np.argsort(-vals)
    keep = [i for i in order if vals[i] < peak_ceiling][:400]

    # Star size is measured on the full-resolution image, not the downsampled one, against each star's own local
    # sky, from the pixels above half its peak. The old moment fit on the downsampled image let the wings and the
    # sky noise in and roughly doubled the DWARF's 3-5 px stars (it kept telling you to check focus).
    full = img.data.mean(axis=2) if img.data.ndim == 3 else img.data
    fh, fw = full.shape
    rad = max(6, 3 * factor)
    yy, xx = np.mgrid[-rad:rad + 1, -rad:rad + 1]
    ring = np.hypot(yy, xx) > rad - 1.5
    rows = []
    for i in keep:
        y0s, x0s = int(ys[i]), int(xs[i])
        fy, fx = y0s * factor + factor // 2, x0s * factor + factor // 2
        if fy < rad or fx < rad or fy >= fh - rad or fx >= fw - rad:
            continue
        cut = full[fy - rad:fy + rad + 1, fx - rad:fx + rad + 1]
        py, px = np.unravel_index(int(np.argmax(cut[rad - factor:rad + factor + 1, rad - factor:rad + factor + 1])),
                                  (2 * factor + 1, 2 * factor + 1))
        cy0, cx0 = fy - factor + int(py), fx - factor + int(px)
        if cy0 < rad or cx0 < rad or cy0 >= fh - rad or cx0 >= fw - rad:
            continue
        cut = full[cy0 - rad:cy0 + rad + 1, cx0 - rad:cx0 + rad + 1].astype(np.float64)
        sky = float(np.median(cut[ring]))
        amp = float(cut[rad, rad]) - sky
        if amp <= 0:
            continue
        above = (cut - sky) >= amp / 2
        # Only the blob joined to the peak (a neighbour star above half max would inflate it).
        import cv2
        _, lab = cv2.connectedComponents(above.astype(np.uint8), connectivity=8)
        core = lab == lab[rad, rad]
        area = float(core.sum())
        if area < 1 or core[0].any() or core[-1].any() or core[:, 0].any() or core[:, -1].any():
            continue
        fwhm = 2 * math.sqrt(area / math.pi)
        wgt = np.where(core, cut - sky, 0.0)
        tot = wgt.sum()
        cx, cy = (wgt * xx).sum() / tot, (wgt * yy).sum() / tot
        mxx = (wgt * (xx - cx) ** 2).sum() / tot
        myy = (wgt * (yy - cy) ** 2).sum() / tot
        mxy = (wgt * (xx - cx) * (yy - cy)).sum() / tot
        tr, det = mxx + myy, mxx * myy - mxy ** 2
        disc = math.sqrt(max(tr * tr / 4 - det, 0))
        l1, l2 = tr / 2 + disc, max(tr / 2 - disc, 1e-6)
        angle = (0.5 * math.degrees(math.atan2(2 * mxy, mxx - myy))) % 180.0
        rows.append((x0s, y0s, fwhm, min(3.0, math.sqrt(l1 / l2)), angle, float(flat[y0s, x0s])))

    if not rows:
        return {"detected": count, "note": "Stars found but all saturated or at the edge."}

    arr = np.array(rows)
    fwhm, elong, ang = arr[:, 2], arr[:, 3], arr[:, 4]
    cx_n, cy_n = (arr[:, 0] - w / 2) / (w / 2), (arr[:, 1] - h / 2) / (h / 2)
    radius = np.hypot(cx_n, cy_n)
    centre, edge = radius < 0.4, radius > 0.75

    def med(a):
        return round(float(np.median(a)), 2) if a.size else None

    result = {
        "detected": count,
        "measured": int(len(rows)),
        "fwhm_px_median": med(fwhm),
        "elongation_median": med(elong),
        "elongated_percent": round(100 * float(np.mean(elong > 1.3)), 1),
        "fwhm_center_px": med(fwhm[centre]),
        "fwhm_edge_px": med(fwhm[edge]),
        "elongation_edge": med(elong[edge]),
        "dominant_elongation_angle_deg_0_is_horizontal": med(ang[elong > 1.2]) if np.any(elong > 1.2) else None,
    }
    result["_positions"] = [(int(r[0]), int(r[1])) for r in rows]
    native = (img.data.shape[1], img.data.shape[0]) in DWARF3_NATIVE_SIZES
    if native and result["fwhm_px_median"]:
        result["fwhm_arcsec_median"] = round(result["fwhm_px_median"] * DWARF3_TELE_ARCSEC_PER_PX, 1)
    return result


def _color_emission(small: np.ndarray, bg_map: np.ndarray, flat: np.ndarray, noise: float,
                    positions: list[tuple[int, int]]) -> dict:
    """What the colour says about the light: emission nebula red (H-alpha), teal (OIII), or starlight continuum.

    The DWARF 3 records three broad colour channels, not a spectrum, so these are colour ratios that hint at
    which emission lines dominate. With the Duo-band filter, red is essentially H-alpha (656 nm) and
    green + blue essentially OIII (496/501 nm).
    """
    # Per-channel background from the darkest half of pixels, then the light above it.
    bg_rgb = np.array([np.percentile(small[:, :, c], 30) for c in range(3)], np.float32)
    excess = small - bg_rgb
    # Remove stars so the nebula or galaxy itself is measured.
    star_mask = np.zeros_like(flat, bool)
    for x, y in positions:
        star_mask[max(0, y - 4):y + 5, max(0, x - 4):x + 5] = True
    neb = (flat > 4 * noise) & ~star_mask
    result: dict = {"note": "Colour ratios from three broad RGB channels, not a true spectrum. "
                            "Uncalibrated until colour calibration (SPCC/PCC) has been run."}
    if neb.sum() > 500:
        sig = np.clip(excess[neb], 0, None).sum(axis=0)
        frac = sig / max(float(sig.sum()), 1e-9)
        r, g, b = (float(v) for v in frac)
        if r > 0.45 and r > 1.4 * g:
            kind = "red-dominated: H-alpha emission (ionised hydrogen)"
        elif (g + b) / 2 > 1.15 * r and b > 0.3:
            kind = "teal/blue-green: OIII emission or blue reflection"
        elif b > 0.4 and b > 1.2 * r:
            kind = "blue: reflection nebula or hot young stars"
        else:
            kind = "balanced/yellowish: broadband starlight (galaxy, cluster, or dust)"
        result.update({
            "extended_signal_pixels_percent": round(100 * float(neb.mean()), 2),
            "signal_rgb_fraction": [round(r, 3), round(g, 3), round(b, 3)],
            "halpha_vs_oiii_ratio": round(r / max((g + b) / 2, 1e-6), 2),
            "dominant_light": kind,
        })
    else:
        result["dominant_light"] = "not enough extended signal to measure (mostly stars or background)"

    # Star colours: normalise each channel by the median star so an uncalibrated cast cancels out,
    # then sort stars from blue (hot) to red (cool) by their blue-to-red ratio.
    if len(positions) >= 20:
        fluxes = []
        for x, y in positions:
            ap = excess[max(0, y - 2):y + 3, max(0, x - 2):x + 3].reshape(-1, 3).sum(axis=0)
            if (ap > 0).all():
                fluxes.append(ap)
        if len(fluxes) >= 20:
            f = np.array(fluxes)
            f = f / np.median(f, axis=0)
            br = f[:, 2] / f[:, 0]
            bins = {"blue (hot, O/B/A)": br > 1.3, "white (F)": (br > 1.08) & (br <= 1.3),
                    "yellow (G, Sun-like)": (br > 0.92) & (br <= 1.08), "orange (K)": (br > 0.7) & (br <= 0.92),
                    "red (cool, M)": br <= 0.7}
            result["star_colours_percent"] = {k: round(100 * float(v.mean()), 1) for k, v in bins.items()}
            result["star_colour_note"] = "Relative to the median star in this frame, so treat as a rough mix."
    return result


# ---------------------------------------------------------------- previews

def stretch_params(reference: np.ndarray) -> tuple[float, float] | None:
    """Linked screen-transfer stretch, like Siril's autostretch, for viewing only.

    Returns (shadow clip, midtone balance), or None when the data is already stretched.
    """
    lum = reference.mean(axis=2)
    med = float(np.median(lum))
    if med > 0.12:
        return None
    shadow = max(0.0, med - 2.8 * _mad_sigma(lum))
    m = (med - shadow) / max(1 - shadow, 1e-6)
    target = 0.22
    # Midtones transfer function solving mtf(m) = target.
    mid = m * (target - 1) / (2 * target * m - target - m) if m > 0 else 0.5
    return shadow, mid


def apply_stretch(data: np.ndarray, params: tuple[float, float] | None) -> np.ndarray:
    if params is None:
        return data
    shadow, mid = params
    norm = np.clip((data - shadow) / max(1 - shadow, 1e-6), 0, 1)
    return ((mid - 1) * norm) / ((2 * mid - 1) * norm - mid)


def _jpeg_b64(arr: np.ndarray, quality: int = 88) -> str:
    rgb = arr if arr.shape[2] == 3 else np.repeat(arr, 3, axis=2)
    im = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8), mode="RGB")
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


def make_previews(img: LoadedImage) -> dict:
    """Whole-frame preview plus a 100% centre crop so star shape and noise are visible."""
    small, _ = _downsample(img.data, PREVIEW_LONG_EDGE)
    params = stretch_params(small)
    h, w = img.data.shape[:2]
    s = min(CROP_SIZE, h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    crop = img.data[y0:y0 + s, x0:x0 + s]
    return {"overview_jpeg_b64": _jpeg_b64(apply_stretch(small, params)),
            "crop_jpeg_b64": _jpeg_b64(apply_stretch(crop, params)),
            "display_stretched": params is not None}


def analyze(path: str | Path, mode: str = "auto") -> dict:
    """Measure an image or lunar video. mode is "auto", "deepsky" or "moon"."""
    from . import lunar

    path = Path(path).expanduser()
    if path.suffix.lower() in lunar.VIDEO_EXTS:
        return {"path": str(path), "shot_info": find_shot_info(path), "mode": "moon", "kind": "video",
                **lunar.analyze_video(path)}
    img = load_image(path)
    reason = "chosen by hand"
    if mode == "auto":
        mode, reason = detect_mode(path, img)
    if mode == "moon":
        metrics, crop_center = lunar.measure_stack(img)
        previews = lunar.previews_for_stack(img, crop_center)
    else:
        metrics, previews = measure(img), make_previews(img)
    return {"path": str(img.path), "shot_info": img.shot_info, "mode": mode, "mode_reason": reason, "kind": "image",
            "metrics": metrics, **previews}


MOON_WORDS = ("moon", "lunar", "luna")


def detect_mode(path: Path, img: LoadedImage) -> tuple[str, str]:
    """Moon or deep sky, from the strongest evidence first: what the DWARF wrote about the shot, the file and
    folder names, then the picture itself (one dominant bright disc or crescent)."""
    from . import lunar
    info = {str(k).lower(): v for k, v in (img.shot_info or {}).items()}
    target = str(info.get("target") or info.get("targetname") or info.get("object") or "").strip()
    if target:
        if any(w in target.lower() for w in MOON_WORDS):
            return "moon", f"the DWARF recorded the target as {target}"
        return "deepsky", f"the DWARF recorded the target as {target}"
    names = " ".join(p.lower() for p in path.parts[-3:])
    if any(w in names for w in MOON_WORDS):
        return "moon", "the file or folder name says Moon"
    if (m := re.search(r"exp_([\d.]+)", names)) and float(m.group(1)) < 0.1:
        return "moon", f"the exposure was {m.group(1)} s, far too short for anything but the Moon or Sun"
    if "dwarf_raw" in names or path.stem.lower().startswith(("result_", "stacked")):
        return "deepsky", "it's a deep-sky session or Auto-apply result"
    small, _ = _downsample(img.data, 512)
    if lunar.looks_like_moon(small.mean(axis=2)):
        return "moon", "the picture shows one large bright disc"
    return "deepsky", "the picture has no Moon disc in it"

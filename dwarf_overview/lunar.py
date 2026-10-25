"""Moon mode: measure lunar stacks and rate the frames in lunar videos.

A stacked Moon image is judged on detail (sharpness), how much of the bright
limb is clipped, and whether earlier sharpening left dark halos along the limb.
A video is rated frame by frame, which tells you how many frames are worth
stacking in AutoStakkert and whether the Moon drifted out of the field.
"""
from __future__ import annotations

import math
import struct
from pathlib import Path

import numpy as np
from PIL import Image

from .imaging import LoadedImage, _box_blur3, _downsample, _jpeg_b64, _mad_sigma

VIDEO_EXTS = {".ser", ".mp4", ".avi", ".mov", ".mkv"}
SAMPLE_FRAMES = 240


# ---------------------------------------------------------------- shared measures

def _otsu(lum: np.ndarray) -> float:
    hist, edges = np.histogram(lum, bins=256, range=(0.0, 1.0))
    w = np.cumsum(hist).astype(np.float64)
    mu = np.cumsum(hist * edges[:-1])
    total, mu_t = w[-1], mu[-1]
    w1 = total - w
    valid = (w > 0) & (w1 > 0)
    between = np.zeros_like(w)
    between[valid] = (mu_t * w[valid] - mu[valid] * total) ** 2 / (w[valid] * w1[valid])
    return float(edges[int(np.argmax(between))])


def _laplacian(lum: np.ndarray) -> np.ndarray:
    p = np.pad(lum, 1, mode="edge")
    return 4 * lum - p[:-2, 1:-1] - p[2:, 1:-1] - p[1:-1, :-2] - p[1:-1, 2:]


def _erode(mask: np.ndarray, r: int) -> np.ndarray:
    out = mask.copy()
    for _ in range(r):
        p = np.pad(out, 1, constant_values=False)
        out = p[1:-1, 1:-1] & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    return out


def _dilate(mask: np.ndarray, r: int) -> np.ndarray:
    return ~_erode(~mask, r)


def sharpness(lum: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Laplacian energy relative to brightness, so a brighter frame doesn't count as sharper."""
    lap = _laplacian(lum)
    sel = lap[mask] if mask is not None and mask.any() else lap
    mean = float(lum[mask].mean()) if mask is not None and mask.any() else float(lum.mean())
    return float(np.sqrt(np.mean(sel ** 2)) / max(mean, 1e-4))


def looks_like_moon(lum: np.ndarray) -> bool:
    thr = _otsu(lum)
    frac = float(np.mean(lum > thr))
    if not (thr > 0.08 and frac > 0.04 and float(np.median(lum)) < thr):
        return False
    # The Moon is one solid bright disc: most bright pixels sit in a single blob. A bright nebula or a
    # washed-out deep-sky image is patchy, so it fails this.
    import cv2
    n, _, stats, _ = cv2.connectedComponentsWithStats((lum > thr).astype("uint8"), connectivity=8)
    if n < 2:
        return False
    areas = stats[1:, cv2.CC_STAT_AREA]
    # ...and a big one (the DWARF 3's Moon is about 600 px across a 3840 px frame, 2% of it).
    return float(areas.max()) / float(areas.sum()) > 0.8 and float(areas.max()) / lum.size > 0.005


def _detail_center(lum: np.ndarray, inner: np.ndarray, factor: int) -> tuple[float, float]:
    """Full-resolution (y, x) of the busiest detail, normally the terminator, for the 100% crop."""
    detail = np.abs(_laplacian(lum)) * inner
    k = 32
    hh, ww = lum.shape[0] // k, lum.shape[1] // k
    if not (hh and ww) or not inner.any():
        return lum.shape[0] * factor / 2, lum.shape[1] * factor / 2
    tiles = detail[: hh * k, : ww * k].reshape(hh, k, ww, k).mean(axis=(1, 3))
    ty, tx = np.unravel_index(int(np.argmax(tiles)), tiles.shape)
    return (ty + 0.5) * k * factor, (tx + 0.5) * k * factor


def _crop(data: np.ndarray, center, size: int) -> np.ndarray:
    h, w = data.shape[:2]
    s = min(size, h, w)
    y0 = int(min(max(center[0] - s / 2, 0), h - s))
    x0 = int(min(max(center[1] - s / 2, 0), w - s))
    return data[y0:y0 + s, x0:x0 + s]


# ---------------------------------------------------------------- stacked image

def measure_stack(img: LoadedImage) -> dict:
    small, factor = _downsample(img.data, 2048)
    lum = small.mean(axis=2)
    thr = _otsu(lum)
    moon = lum > max(thr * 0.6, 0.02)
    inner = _erode(moon, 6)
    ring_out = _dilate(moon, 4) & ~moon
    far = ~_dilate(moon, 25)
    h, w = lum.shape

    ys, xs = np.nonzero(moon)
    disk = {}
    if ys.size:
        span = max(ys.max() - ys.min(), xs.max() - xs.min()) + 1
        # The lit side always shows the full diameter along one axis, so the larger span approximates it.
        radius = span / 2
        disk = {
            "lit_fraction_of_frame_percent": round(100 * float(moon.mean()), 1),
            "approx_illuminated_percent": round(min(100.0, 100 * moon.sum() / (math.pi * radius ** 2)), 0),
            "disk_diameter_px": int(span * factor),
            "touches_frame_edge": bool(moon[0].any() or moon[-1].any() or moon[:, 0].any() or moon[:, -1].any()),
        }

    full_lum = img.data.mean(axis=2)
    full_moon = full_lum > max(thr * 0.6, 0.02)
    sat = img.data.max(axis=2) >= 0.995
    resid = lum - _box_blur3(lum)
    noise = _mad_sigma(resid[inner]) / math.sqrt(8 / 9) if inner.any() else _mad_sigma(resid)
    sky = float(np.median(lum[far])) if far.any() else 0.0
    halo = None
    if ring_out.any() and far.any():
        # Over-sharpening leaves a dark ring just outside the limb, darker than the open sky.
        dip = sky - float(np.percentile(lum[ring_out], 10))
        halo = round(dip / max(_mad_sigma(lum[far]), 1e-4), 1)

    color = None
    if img.is_color and inner.any():
        px = small[inner]
        chroma = (px.max(axis=1) - px.min(axis=1)) / np.maximum(px.mean(axis=1), 1e-4)
        ch = px.mean(axis=0)
        color = {"mean_rgb": [round(float(v), 4) for v in ch],
                 "saturation_index": round(float(np.median(chroma)), 3)}

    crop_center = _detail_center(lum, inner, factor)

    metrics = {
        "image": {"file": img.path.name, "format": img.source_format, "bit_depth": img.bit_depth,
                  "width": int(img.data.shape[1]), "height": int(img.data.shape[0]),
                  "channels": int(img.data.shape[2]), "analysis_downsample": factor},
        "disk": disk,
        "detail": {
            "sharpness_index": round(sharpness(lum, inner), 4),
            "noise_sigma_on_disk": round(noise, 5),
            "mean_disk_brightness": round(float(lum[moon].mean()) if moon.any() else 0.0, 4),
        },
        "exposure": {
            "saturated_disk_percent": round(100 * float(sat[full_moon].mean()) if full_moon.any() else 0.0, 3),
            "brightest_p99_9": round(float(np.percentile(full_lum[full_moon], 99.9)) if full_moon.any() else 0.0, 4),
            "sky_level": round(sky, 4),
        },
        "sharpening_halo_sigma": halo,
        "color": color,
    }
    return metrics, crop_center


def previews_for_stack(img: LoadedImage, crop_center) -> dict:
    small, _ = _downsample(img.data, 1568)
    return {"overview_jpeg_b64": _jpeg_b64(small), "crop_jpeg_b64": _jpeg_b64(_crop(img.data, crop_center, 768)),
            "display_stretched": False}


# ---------------------------------------------------------------- videos

class _SerReader:
    """Minimal reader for SER files, the usual capture format for planetary and lunar video."""

    def __init__(self, path: Path):
        self.f = open(path, "rb")
        hdr = self.f.read(178)
        if not hdr.startswith(b"LUCAM-RECORDER"):
            raise ValueError("Not a SER file.")
        color_id, self.w, self.h, depth, self.n = struct.unpack("<i4x4i", hdr[18:42])
        self.planes = 3 if color_id in (100, 101) else 1
        # The header's endianness flag is unreliable in practice; 16-bit SER data is little-endian.
        self.dtype = np.dtype("<u2") if depth > 8 else np.dtype("u1")
        self.frame_bytes = self.w * self.h * self.planes * self.dtype.itemsize
        self.color_id = color_id
        self.fps = None

    def read(self, i: int) -> np.ndarray:
        self.f.seek(178 + i * self.frame_bytes)
        buf = np.frombuffer(self.f.read(self.frame_bytes), dtype=self.dtype)
        frame = buf.reshape(self.h, self.w, self.planes).astype(np.float32)
        frame /= 65535.0 if self.dtype.itemsize == 2 else 255.0
        if self.color_id == 101:  # BGR order
            frame = frame[:, :, ::-1]
        if 8 <= self.color_id <= 19:  # raw Bayer: bin 2x2 so the colour pattern doesn't read as detail
            h2, w2 = self.h // 2, self.w // 2
            frame = frame[: h2 * 2, : w2 * 2].reshape(h2, 2, w2, 2, 1).mean(axis=(1, 3))
        return frame

    def close(self):
        self.f.close()


class _CvReader:
    def __init__(self, path: Path):
        try:
            import cv2
        except ImportError as e:
            raise ValueError("Reading MP4/AVI needs the opencv-python-headless package. "
                             "Run start-windows.bat again to install it.") from e
        self.cv2 = cv2
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise ValueError("Couldn't open this video.")
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = round(float(self.cap.get(cv2.CAP_PROP_FPS)), 2) or None

    def read(self, i: int) -> np.ndarray | None:
        self.cap.set(self.cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = self.cap.read()
        if not ok:
            return None
        return frame[:, :, ::-1].astype(np.float32) / 255.0

    def close(self):
        self.cap.release()


def analyze_video(path: Path) -> dict:
    reader = _SerReader(path) if path.suffix.lower() == ".ser" else _CvReader(path)
    try:
        n = reader.n
        if n <= 0:
            raise ValueError("The video has no frames.")
        idx = np.unique(np.linspace(0, n - 1, min(n, SAMPLE_FRAMES)).astype(int))
        scores, bright, cents, edge, sat, frames = [], [], [], [], [], {}
        for i in idx:
            fr = reader.read(int(i))
            if fr is None:
                continue
            small, _ = _downsample(fr, 640)
            lum = small.mean(axis=2)
            thr = _otsu(lum)
            m = lum > max(thr * 0.6, 0.02)
            inner = _erode(m, 3)
            scores.append(sharpness(lum, inner))
            bright.append(float(lum[m].mean()) if m.any() else 0.0)
            ys, xs = np.nonzero(m)
            cents.append((float(ys.mean()) / lum.shape[0], float(xs.mean()) / lum.shape[1]) if ys.size else (0.5, 0.5))
            edge.append(bool(m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any()))
            sat.append(float(np.mean(fr.max(axis=2) >= 0.995)))
            frames[int(i)] = fr
    finally:
        reader.close()
    if not scores:
        raise ValueError("Couldn't read any frames from this video.")

    s = np.array(scores)
    order = np.argsort(-s)
    keys = list(frames)
    best, worst = frames[keys[int(order[0])]], frames[keys[int(order[-1])]]
    cents = np.array(cents)
    p90 = float(np.percentile(s, 90))
    good = float(np.mean(s >= 0.85 * p90))
    metrics = {
        "video": {"file": path.name, "format": path.suffix.lstrip(".").upper(), "frames": int(n),
                  "sampled": int(len(s)), "width": int(reader.w), "height": int(reader.h), "fps": reader.fps},
        "frame_quality": {
            "sharpness_p10": round(float(np.percentile(s, 10)), 4),
            "sharpness_median": round(float(np.median(s)), 4),
            "sharpness_p90": round(p90, 4),
            "spread_p90_over_p10": round(p90 / max(float(np.percentile(s, 10)), 1e-6), 2),
            "frames_within_15pct_of_best_percent": round(100 * good, 1),
            "sharpness_trend_first_vs_last_third": round(float(np.median(s[-len(s) // 3:]) / max(np.median(s[:len(s) // 3]), 1e-6)), 2),
        },
        "stability": {
            "drift_percent_of_frame": round(100 * float(np.hypot(*(cents.max(axis=0) - cents.min(axis=0)))), 1),
            "frames_touching_edge_percent": round(100 * float(np.mean(edge)), 1),
            "brightness_variation_percent": round(100 * float(np.std(bright) / max(np.mean(bright), 1e-6)), 1),
        },
        "exposure": {"saturated_pixels_percent_median": round(100 * float(np.median(sat)), 3),
                     "mean_disk_brightness": round(float(np.mean(bright)), 4)},
    }
    small_best, f = _downsample(best, 1024)
    lum = small_best.mean(axis=2)
    center = _detail_center(lum, _erode(lum > max(_otsu(lum) * 0.6, 0.02), 3), f)
    a, b = _crop(best, center, 512), _crop(worst, center, 512)
    # Side by side at 100%: the same patch from the sharpest frame (left) and the softest (right).
    pair = np.concatenate([a, np.ones((a.shape[0], 6, a.shape[2]), np.float32), b], axis=1)
    small_best, _ = _downsample(best, 1568)
    return {"metrics": metrics, "overview_jpeg_b64": _jpeg_b64(small_best), "crop_jpeg_b64": _jpeg_b64(pair),
            "display_stretched": False, "preview_note": "Image 1 is the sharpest sampled frame. Image 2 is a 100% crop: "
            "the same patch from the sharpest frame (left) and the softest frame (right)."}

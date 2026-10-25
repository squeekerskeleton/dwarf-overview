"""Image steps Auto-apply does itself (numpy, OpenCV, astropy) when no outside program covers them.

- colour_balance: an offline colour calibration from the image's own stars, used when Siril's SPCC/PCC can't run;
- reduce_stars: shrinks stars, with StarNet's starless image when StarNet is installed, otherwise with a star mask
  and a morphological erosion (the classic star-reduction method);
- hoo: a Hα/OIII palette version for Duo-Band sessions (red is Hα, green and blue carry OIII);
- save_final: 16-bit TIFF, PNG, JPG and a FITS copy.
"""
from __future__ import annotations

from pathlib import Path


def _np():
    import numpy as np
    return np


def read_fits_rgb(path: Path):
    """(H, W, 3) float32 and the header."""
    np = _np()
    from astropy.io import fits
    with fits.open(path) as h:
        data, header = np.asarray(h[0].data, dtype="float32"), h[0].header.copy()
    if data.ndim == 3 and data.shape[0] == 3:
        data = np.moveaxis(data, 0, -1)
    elif data.ndim == 2:
        data = np.repeat(data[..., None], 3, axis=2)
    return data, header


def write_fits_rgb(path: Path, img, header=None) -> None:
    np = _np()
    from astropy.io import fits
    hdr = header.copy() if header is not None else None
    if hdr is not None:
        for k in ("BZERO", "BSCALE", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2", "NAXIS3"):
            hdr.remove(k, ignore_missing=True)
    fits.PrimaryHDU(np.moveaxis(img.astype("float32"), -1, 0), header=hdr).writeto(path, overwrite=True)


def read_tif(path: Path):
    np = _np()
    import tifffile
    img = tifffile.imread(str(path))
    if img.ndim == 3 and img.shape[0] in (3, 4) and img.shape[-1] not in (3, 4):
        img = np.moveaxis(img, 0, -1)
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=2)
    img = img[..., :3]
    scale = 65535.0 if img.dtype == np.uint16 else 255.0 if img.dtype == np.uint8 else 1.0
    return np.clip(img.astype("float32") / scale, 0, 1)


def _star_mask(lum, size: int = 9, k: float = 5.0, grow: int = 2):
    """Soft mask of small bright points (stars): white top-hat above the noise, grown and feathered."""
    np = _np()
    import cv2
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    tophat = cv2.morphologyEx(lum, cv2.MORPH_TOPHAT, kernel)
    noise = 1.4826 * float(np.median(np.abs(tophat - np.median(tophat)))) or 1e-4
    mask = (tophat > k * noise).astype("float32")
    if grow:
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1)))
    return np.clip(cv2.GaussianBlur(mask, (0, 0), 1.2), 0, 1)


def colour_balance(src: Path, dst: Path) -> str:
    """Neutral background, then white stars on average (linear data)."""
    np = _np()
    img, header = read_fits_rgb(src)
    med = np.median(img.reshape(-1, 3), axis=0)
    img = img - med + med.mean()
    lum = img.mean(axis=2)
    mask = _star_mask(lum, grow=0) > 0.5
    peak = np.percentile(lum, 99.95)
    mask &= lum < peak  # leave saturated cores out
    if mask.sum() < 200:
        write_fits_rgb(dst, np.clip(img, 0, None), header)
        return "neutralised the background (too few stars to balance the colour from)"
    flux = np.array([np.mean(img[..., c][mask] - med.mean()) for c in range(3)])
    gains = np.clip(flux[1] / np.maximum(flux, 1e-9), 0.5, 2.0)
    img = (img - med.mean()) * gains + med.mean()
    write_fits_rgb(dst, np.clip(img, 0, None), header)
    return f"neutralised the background and balanced the stars to white (R×{gains[0]:.2f}, B×{gains[2]:.2f})"


def reduce_stars(img, amount: float, starless=None):
    """Stretched RGB in 0..1. With a starless image (StarNet) the stars are separated and scaled down;
    otherwise they're eroded inside a star mask."""
    np = _np()
    import cv2
    if starless is not None:
        stars = np.clip(1 - (1 - img) / np.maximum(1 - starless, 1e-4), 0, 1)
        return np.clip(1 - (1 - starless) * (1 - stars * (1 - amount * 0.6)), 0, 1)
    mask = _star_mask(img.mean(axis=2))[..., None] * amount
    eroded = cv2.erode(img, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    if amount > 0.6:  # a second, gentler pass for the bigger stars
        eroded = 0.5 * eroded + 0.5 * cv2.erode(eroded, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    return np.clip(img * (1 - mask) + eroded * mask, 0, 1)


def hoo(img):
    """Duo-Band palette: Hα in red, OIII (the green and blue pixels) in green and blue, OIII lifted to match."""
    np = _np()
    ha, o3 = img[..., 0], (img[..., 1] + img[..., 2]) / 2
    bg_h, bg_o = np.median(ha), np.median(o3)
    sig_h = np.percentile(ha, 99.5) - bg_h
    sig_o = max(1e-4, np.percentile(o3, 99.5) - bg_o)
    o3 = np.clip(bg_h + (o3 - bg_o) * min(3.0, max(1.0, sig_h / sig_o)), 0, 1)
    return np.stack([ha, o3, o3], axis=-1)


def save_final(img, stem: Path, header=None) -> dict:
    np = _np()
    import tifffile
    from PIL import Image
    out = {}
    tif = stem.with_suffix(".tif")
    tifffile.imwrite(str(tif), (np.clip(img, 0, 1) * 65535 + 0.5).astype("uint16"), photometric="rgb")
    out["tif"] = str(tif)
    eight = Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype("uint8"), "RGB")
    eight.save(stem.with_suffix(".png"), optimize=True)
    eight.save(stem.with_suffix(".jpg"), quality=95)
    out["png"], out["jpg"] = str(stem.with_suffix(".png")), str(stem.with_suffix(".jpg"))
    # FITS rows go bottom-up, so flip to keep the same orientation as the TIFF in FITS viewers.
    write_fits_rgb(stem.with_suffix(".fits"), np.flipud(img), header)
    out["fits"] = str(stem.with_suffix(".fits"))
    return out


def _sky(c):
    """Sky level of one channel: the median of its darker half, so nebulosity doesn't pull it up."""
    np = _np()
    lo = c[c <= np.percentile(c, 50)]
    return float(np.median(lo)) if lo.size else float(np.median(c))


def neutralise(src: Path, dst: Path) -> str:
    """Linear data: shift each channel so their sky levels match (a neutral grey background)."""
    np = _np()
    img, header = read_fits_rgb(src)
    sky = np.array([_sky(img[..., c]) for c in range(3)])
    spread = float((sky.max() - sky.min()) / max(1e-9, sky.mean()))
    img = img - sky + sky.mean()
    write_fits_rgb(dst, np.clip(img, 0, None), header)
    return f"background neutralised (channels were {spread * 100:.0f}% apart)"


def chroma_denoise(src: Path, dst: Path, sigma: float) -> None:
    """Smooth only the colour, not the brightness: kills the blotchy colour mottling (on the DWARF's Duo-Band data
    the red pixels, a quarter of the sensor, carry all the Hα and are much noisier) while keeping detail."""
    np = _np()
    import cv2
    img, header = read_fits_rgb(src)
    lum = img.mean(axis=2, keepdims=True)
    chroma = img - lum
    chroma = np.stack([cv2.GaussianBlur(chroma[..., c], (0, 0), sigma) for c in range(3)], axis=-1)
    write_fits_rgb(dst, np.clip(lum + chroma, 0, None), header)


def _noise(c):
    np = _np()
    import cv2
    hp = c - cv2.GaussianBlur(c, (0, 0), 2)
    return 1.4826 * float(np.median(np.abs(hp - np.median(hp))))


def quality(img) -> dict:
    """Stretched RGB: how coloured the background is and how noisy each channel is."""
    np = _np()
    h, w = img.shape[:2]
    core = img[h // 8: h - h // 8, w // 8: w - w // 8]
    sky = np.array([_sky(core[..., c]) for c in range(3)])
    noise = [_noise(core[..., c]) for c in range(3)]
    return {"sky": sky.tolist(), "spread": float((sky.max() - sky.min()) / max(1e-6, sky.mean())), "noise": noise,
            "red_vs_green": noise[0] / max(1e-6, noise[1])}


def quality_gate(img) -> tuple:
    """Check the finished image and fix what can be fixed: a coloured background is neutralised and strong colour
    noise smoothed. Returns (image, notes)."""
    np = _np()
    import cv2
    notes = []
    q = quality(img)
    if q["spread"] > 0.05:
        sky = np.array(q["sky"])
        img = np.clip(img - sky + sky.mean(), 0, 1)
        notes.append(f"background was {q['spread'] * 100:.0f}% off neutral, now neutral")
    if q["red_vs_green"] > 2 or q["red_vs_green"] < 0.5:
        lum = img.mean(axis=2, keepdims=True)
        chroma = img - lum
        chroma = np.stack([cv2.GaussianBlur(chroma[..., c], (0, 0), 2.5) for c in range(3)], axis=-1)
        img = np.clip(lum + chroma, 0, 1)
        notes.append(f"colour noise was uneven (red {q['red_vs_green']:.1f}× green), smoothed")
    after = quality(img)
    notes.append(f"check: background {after['spread'] * 100:.1f}% off neutral, red noise {after['red_vs_green']:.1f}× green")
    return img, notes

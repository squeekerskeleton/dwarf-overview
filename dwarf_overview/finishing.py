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
    # Stars stay out of the blur both ways: their own colour is kept, and it isn't smeared into a coloured ring
    # around them (C 33 in 0.11.1 had those halos). Normalised blur: blur(chroma * sky) / blur(sky).
    peak = max(1e-9, float(np.percentile(lum, 99.9)))
    preview = np.arcsinh(np.clip(lum[..., 0] / peak, 0, None) * 200) / np.arcsinh(200)
    stars = star_mask(np.repeat(preview[..., None], 3, axis=2))
    keep = (1 - stars).astype("float32")
    norm = np.maximum(cv2.GaussianBlur(keep, (0, 0), sigma), 1e-3)
    smooth = np.stack([cv2.GaussianBlur(chroma[..., c] * keep, (0, 0), sigma) / norm for c in range(3)], axis=-1)
    chroma = stars[..., None] * chroma + keep[..., None] * smooth
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


def quality_gate(img, tol: float = 0.02) -> tuple:
    """Check the finished image and fix what can be fixed: a coloured background is neutralised and strong colour
    noise smoothed. Returns (image, notes)."""
    np = _np()
    import cv2
    notes = []
    q = quality(img)
    if q["spread"] > tol:
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


# Luminance weights for the DWARF 3's Sony IMX678 (its relative red/green/blue response), so the brightness the
# stretch works on matches what the sensor saw rather than a generic Rec.709 mix.
IMX678_LUMA = (0.34, 0.48, 0.18)


def colour_stretch(img, target_bg: float = 0.12, shadow_sigma: float = 5.5, protect: float = 0.6):
    """Linear, colour-calibrated RGB -> stretched RGB that keeps the calibrated colour.

    Siril's unlinked auto-stretch stretches each channel on its own, which throws away SPCC's white balance (Siril's
    own docs say not to use it after colour calibration); a linked one keeps the balance but washes colour out of
    everything bright. Here only the brightness is stretched, with an inverse hyperbolic sine (strong in the faint
    nebula, gentle on stars), and every pixel keeps its red:green:blue ratios. The stretch strength is solved so the
    sky lands at target_bg. Near the sky the colour is eased toward grey (protect), so background noise doesn't turn
    into colour speckle; stars and nebula keep their full colour.

    shadow_sigma sets the black point that many sky-noise sigmas below the sky. Grain (sky noise / sky level) comes
    out at about 1/shadow_sigma: 2.8 in 0.11-0.12.2 gave 0.42 on C 33 against 0.16 in the DWARF's own JPG.
    """
    np = _np()
    x = np.clip(np.asarray(img, dtype="float32"), 0, None)
    w = np.array(IMX678_LUMA, dtype="float32")
    lum = x @ w
    step = max(1, lum.size // 400_000)
    sample = lum.reshape(-1)[::step]
    med = float(np.median(sample))
    low = sample[sample <= med]
    sigma = 1.4826 * float(np.median(np.abs(low - np.median(low)))) if low.size else 0.0
    black = max(0.0, med - shadow_sigma * max(sigma, 1e-7))
    top = float(np.percentile(sample, 99.995))
    if lum.size:
        top = max(top, float(lum.max()) * 0.98)
    scale = 1.0 / max(1e-9, top - black)
    xs = np.clip((x - black) * scale, 0, None)
    ls = np.clip((lum - black) * scale, 0, 1)
    m = min(0.5, max(1e-6, (med - black) * scale))

    def curve(v, d):
        return np.arcsinh(d * v) / np.arcsinh(d)
    lo_d, hi_d = 0.0, 7.0  # solve log10(D) so the sky's median lands on target_bg
    for _ in range(50):
        mid = (lo_d + hi_d) / 2
        if curve(m, 10 ** mid) < target_bg:
            lo_d = mid
        else:
            hi_d = mid
    d = 10 ** ((lo_d + hi_d) / 2)
    new_l = curve(ls, d)
    ratio = new_l / np.maximum(ls, 1e-9)
    out = xs * ratio[..., None]
    # Keep hue where a colour channel would pass white: scale the pixel down rather than clip one channel.
    peak = out.max(axis=2, keepdims=True)
    out = np.where(peak > 1, out / np.maximum(peak, 1e-9), out)
    if protect > 0:
        grey = new_l[..., None]
        # Weight 1 at the sky level, fading to 0 by about three times the sky brightness.
        near_sky = np.clip(1 - (new_l - target_bg) / max(1e-6, 2 * target_bg), 0, 1)[..., None] * protect
        out = out * (1 - near_sky) + grey * near_sky
    return np.clip(out, 0, 1).astype("float32"), {"black": black, "D": float(d), "sky_in": med}


def star_mask(img, size: int = 9):
    """Soft 0..1 mask of the stars in a stretched image: compact bright points that a morphological opening with a
    disc a little wider than a star removes (a top-hat), well above the sky noise. Broad nebulosity stays out."""
    import cv2
    np = _np()
    lum = (img @ np.array(IMX678_LUMA, dtype="float32")).astype("float32")
    sky = float(np.median(lum))
    noise = 1.4826 * float(np.median(np.abs(lum - sky)))
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    top = lum - cv2.morphologyEx(lum, cv2.MORPH_OPEN, disc)
    # A filament survives an opening with a short line along its own direction; a star survives none of them.
    lines = []
    for k in (np.eye(size, dtype="uint8"), np.fliplr(np.eye(size, dtype="uint8")),
              np.ones((1, size), "uint8"), np.ones((size, 1), "uint8")):
        lines.append(cv2.morphologyEx(lum, cv2.MORPH_OPEN, k))
    point = lum - np.max(lines, axis=0)
    core = ((top > max(6 * noise, 0.06)) & (point > 0.6 * top)).astype("uint8")
    # Keep only compact, roundish blobs: thin nebula filaments also pass a top-hat, and they must keep their colour.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(core, connectivity=8)
    w, h, area = stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT], stats[:, cv2.CC_STAT_AREA]
    ok = (area <= 4 * size * size) & (np.maximum(w, h) <= 2.2 * np.maximum(1, np.minimum(w, h))) & \
         (area >= 0.35 * w * h)
    ok[0] = False
    core = ok[labels].astype("uint8")
    core = cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    return np.clip(cv2.GaussianBlur(core.astype("float32"), (0, 0), 1.5) * 1.5, 0, 1)


def cap_star_colour(img, mask, cap: float = 0.25):
    """Stars in a colour-keeping stretch and HOO turn into pure red or cyan blobs; limit their colour to a share
    of their brightness (a star keeps a tint, not a neon disc). Only inside the star mask."""
    np = _np()
    lum = img @ np.array(IMX678_LUMA, dtype="float32")
    chroma = img - lum[..., None]
    core = mask > 0.9
    if core.sum() > 50:
        # Stars as a group should average white: remove their shared tint (C 33's leaned teal).
        tint = np.median(chroma[core] / np.maximum(lum[core], 1e-6)[:, None], axis=0)
        chroma = chroma - mask[..., None] * tint * lum[..., None]
    size = np.abs(chroma).max(axis=2)
    limit = cap * np.maximum(lum, 1e-6)
    shrink = np.minimum(1, limit / np.maximum(size, 1e-9))
    k = 1 - mask * (1 - shrink)
    return np.clip(lum[..., None] + chroma * k[..., None], 0, 1).astype("float32")


def sky_smooth(img, stars=None, sigma: float = 1.2):
    """A gentle brightness-only smoothing of the faint sky after the stretch: full strength at the sky level,
    fading out by a few noise levels above it, never on stars. Nebula detail and stars stay sharp."""
    import cv2
    np = _np()
    w = np.array(IMX678_LUMA, dtype="float32")
    lum = (img @ w).astype("float32")
    sky = float(np.median(lum))
    noise = 1.4826 * float(np.median(np.abs(lum - sky)))
    weight = np.clip(1 - (lum - sky) / max(1e-6, 4 * noise), 0, 1)
    if stars is not None:
        weight = weight * (1 - stars)
    smooth = cv2.GaussianBlur(lum, (0, 0), sigma)
    new = lum + weight * (smooth - lum)
    return np.clip(img + (new - lum)[..., None], 0, 1).astype("float32")


def saturation(img, amount: float = 0.35, protect_bg: bool = True, stars=None):
    """Saturation boost in a hue-preserving way (scale the distance from each pixel's grey), left off the sky
    and, given a star mask, off the stars."""
    np = _np()
    lum = img @ np.array(IMX678_LUMA, dtype="float32")
    gain = np.ones(lum.shape, dtype="float32") * amount
    if protect_bg:
        sky = float(np.median(lum))
        noise = 1.4826 * float(np.median(np.abs(lum - sky)))
        gain = gain * np.clip((lum - sky - noise) / max(1e-6, 3 * noise), 0, 1)
    if stars is not None:
        gain = gain * (1 - stars)
    gain = 1 + gain[..., None]
    out = lum[..., None] + (img - lum[..., None]) * gain
    return np.clip(out, 0, 1).astype("float32")

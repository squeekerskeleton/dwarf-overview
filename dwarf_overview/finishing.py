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
    try:
        img = tifffile.imread(str(path))
    except Exception:  # StarNet2 always writes LZW TIFFs, which tifffile can't decode without imagecodecs
        import cv2
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise OSError(f"Couldn't read {path}")
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB if img.shape[2] == 4 else cv2.COLOR_BGR2RGB)
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
    import cv2
    ha, o3 = img[..., 0], (img[..., 1] + img[..., 2]) / 2
    bg_h, bg_o = _sky(ha), _sky(o3)

    def nebula(c):  # star-free brightness: stars would otherwise set the balance (0.12.7 had OIII under Hα)
        flat = cv2.medianBlur(cv2.medianBlur((np.clip(c, 0, 1) * 65535).astype("uint16"), 5), 5)
        return float(np.percentile(flat.astype("float32") / 65535, 99))
    sig_h = nebula(ha) - bg_h
    sig_o = max(1e-4, nebula(o3) - bg_o)
    # OIII a touch above Hα, as in the C 33 version people liked most.
    o3 = np.clip(bg_h + (o3 - bg_o) * min(3.0, max(1.0, 1.1 * sig_h / sig_o)), 0, 1)
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


def _sky_noise(lum) -> tuple:
    """Sky level and pixel noise of a brightness image. A plain median and MAD count a big nebula as sky and as
    noise (C 33 fills much of the frame), which kept 0.12.4-0.12.5's faint-nebula lift from doing anything."""
    return _sky(lum), max(1e-6, _noise(lum))


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


def colour_stretch(img, target_bg: float = 0.11, shadow_sigma: float = 4.0, protect: float = 0.6):
    """Linear, colour-calibrated RGB -> stretched RGB that keeps the calibrated colour.

    Siril's unlinked auto-stretch stretches each channel on its own, which throws away SPCC's white balance (Siril's
    own docs say not to use it after colour calibration); a linked one keeps the balance but washes colour out of
    everything bright. Here only the brightness is stretched, with an inverse hyperbolic sine (strong in the faint
    nebula, gentle on stars), and every pixel keeps its red:green:blue ratios. The stretch strength is solved so the
    sky lands at target_bg. Near the sky the colour is eased toward grey (protect), so background noise doesn't turn
    into colour speckle; stars and nebula keep their full colour.

    shadow_sigma sets the black point that many sky-noise sigmas below the sky: higher means less grain but a
    flatter image. On C 33, 2.8 gave grain (sky noise / sky) 0.41 and 5.5 plus full sky smoothing looked flat and
    plasticky at 0.05; the DWARF's own JPG is 0.16. 4.0 sits between, and lift_faint brings the contrast back.
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
    sky, noise = _sky_noise(lum)
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


def sky_smooth(img, stars=None, sigma: float = 1.2, amount: float = 1.0):
    """A gentle brightness-only smoothing of the faint sky after the stretch: full strength at the sky level,
    fading out by a few noise levels above it, never on stars. Nebula detail and stars stay sharp."""
    import cv2
    np = _np()
    w = np.array(IMX678_LUMA, dtype="float32")
    lum = (img @ w).astype("float32")
    sky, noise = _sky_noise(lum)
    weight = amount * np.clip(1 - (lum - sky) / max(1e-6, 4 * noise), 0, 1)
    if stars is not None:
        weight = weight * (1 - stars)
    smooth = cv2.GaussianBlur(lum, (0, 0), sigma)
    new = lum + weight * (smooth - lum)
    return np.clip(img + (new - lum)[..., None], 0, 1).astype("float32")


def sky_chroma_smooth(img, stars=None, sigma: float = 2.0):
    """Smooth the colour speckle of the faint sky, colour only (brightness untouched), fading out a few noise
    levels above the sky and never on stars. sky_smooth works on brightness only, so per-channel speckle set a
    floor on the grain: C 33 stayed at 0.16 where the DWARF's own image, with grey sky grain, measured 0.08."""
    import cv2
    np = _np()
    w = np.array(IMX678_LUMA, dtype="float32")
    lum = (img @ w).astype("float32")
    sky, noise = _sky_noise(lum)
    weight = np.clip(1 - (cv2.GaussianBlur(lum, (0, 0), 2.0) - sky) / max(1e-6, 4 * noise), 0, 1)
    if stars is not None:
        weight = weight * (1 - stars)
    chroma = img - lum[..., None]
    soft = np.stack([cv2.GaussianBlur(chroma[..., c], (0, 0), sigma) for c in range(3)], axis=-1)
    return np.clip(img + weight[..., None] * (soft - chroma), 0, 1).astype("float32")


def grain(img) -> float:
    """Sky noise ÷ sky level of a stretched image (C 33: 0.41 in 0.12.2, 0.05 in 0.12.3, DWARF's JPG 0.16)."""
    q = quality(img)
    return float(sum(q["noise"]) / max(1e-6, sum(q["sky"])))


def smooth_to_grain(img, stars=None, target: float = 0.18):
    """Smooth the faint sky only as far as a natural-looking grain (target = sky noise ÷ sky level). Returns
    (image, grain before, grain after). Already below the target: left as it is."""
    g0 = grain(img)
    if g0 <= target:
        return img, g0, g0
    img = sky_chroma_smooth(img, stars)
    if grain(img) <= target:
        return img, g0, grain(img)
    best = None
    for sigma in (1.0, 1.5, 2.0):
        lo, hi = 0.0, 1.0
        for _ in range(7):
            mid = (lo + hi) / 2
            if grain(sky_smooth(img, stars, sigma, mid)) > target:
                lo = mid
            else:
                hi = mid
        out = sky_smooth(img, stars, sigma, hi)
        g1 = grain(out)
        best = (out, g0, g1)
        if g1 <= target * 1.1:
            break
    return best


def nebula_level(img, stars=None) -> float:
    """How bright the subject's nebulosity is above the sky, without stars: the 99th percentile of a 7 px median-
    filtered brightness (stars gone), minus the sky. C 33 in 0.11.1 (the punchy one) was about 0.16."""
    import cv2
    np = _np()
    lum = (img @ np.array(IMX678_LUMA, dtype="float32")).astype("float32")
    sky, _ = _sky_noise(lum)
    if stars is not None:
        lum = lum * (1 - stars) + sky * stars
    flat = cv2.medianBlur((np.clip(lum, 0, 1) * 65535).astype("uint16"), 5)
    flat = cv2.medianBlur(flat, 5).astype("float32") / 65535
    return float(np.percentile(flat, 99)) - sky


def lift_to(img, stars=None, target: float = 0.16, max_gain: float = 3.0):
    """Lift the faint nebula until nebula_level reaches target (or max_gain). Returns (image, before, after, gain).
    A fixed gain didn't carry over between targets; this aims at a measured result."""
    before = nebula_level(img, stars)
    if before >= target:
        return img, before, before, 1.0
    lo, hi = 1.0, max_gain
    best = lift_faint(img, stars, hi)
    if nebula_level(best, stars) <= target:
        return best, before, nebula_level(best, stars), hi
    for _ in range(8):
        mid = (lo + hi) / 2
        if nebula_level(lift_faint(img, stars, mid), stars) < target:
            lo = mid
        else:
            hi = mid
    out = lift_faint(img, stars, hi)
    return out, before, nebula_level(out, stars), hi


def lift_faint(img, stars=None, gain: float = 1.6):
    """Contrast for the faint nebula without lifting the sky's noise: brightness a few noise levels above the sky
    is pushed up (from a little above the sky's smoothed noise, slope `gain`, easing to none at white), the sky itself and the stars untouched.
    Colour ratios are kept. 0.12.3's gentle stretch left C 33's filaments 40-60% dimmer than 0.11.1."""
    np = _np()
    w = np.array(IMX678_LUMA, dtype="float32")
    lum = (img @ w).astype("float32")
    sky, _ = _sky_noise(lum)
    t = np.clip(lum - sky, 0, None)
    span = max(1e-6, 1 - sky)
    curved = sky + t * gain / (1 + (gain - 1) * t / span)
    # Where the lift starts is decided on a blurred copy: on the grainy unsmoothed stretch, "8 pixel-noise levels
    # above the sky" was above C 33's filaments, so 0.12.4-0.12.6 lifted almost nothing whatever the gain.
    import cv2
    soft = cv2.GaussianBlur(lum, (0, 0), 2.0)
    soft_noise = max(1e-6, 1.4826 * float(np.median(np.abs(soft[lum <= np.percentile(lum, 50)] - sky))))
    ramp = np.clip((soft - sky - 2 * soft_noise) / max(1e-6, 4 * soft_noise), 0, 1)
    if stars is not None:
        ramp = ramp * (1 - stars)
    new = np.where(lum > sky, lum + ramp * (curved - lum), lum)
    # Scale each pixel's signal above the sky, not the whole pixel: scaling the whole of it also scaled its grey
    # sky pedestal, which washed C 33's faint Hα toward yellow (green/red 0.46 linear, 0.88 after the stretch).
    sky_rgb = np.array([_sky_level(img[..., c])[0] for c in range(3)], dtype="float32")
    ratio = np.where(lum > sky, (new - sky) / np.maximum(lum - sky, 1e-6), 1.0)
    out = sky_rgb + (img - sky_rgb) * ratio[..., None]
    peak = out.max(axis=2, keepdims=True)
    out = np.where(peak > 1, out / np.maximum(peak, 1e-9), out)
    return np.clip(out, 0, 1).astype("float32")


def saturation(img, amount: float = 0.35, protect_bg: bool = True, stars=None):
    """Saturation boost in a hue-preserving way (scale the distance from each pixel's grey), left off the sky
    and, given a star mask, off the stars."""
    np = _np()
    lum = img @ np.array(IMX678_LUMA, dtype="float32")
    gain = np.ones(lum.shape, dtype="float32") * amount
    if protect_bg:
        sky, noise = _sky_noise(lum)
        gain = gain * np.clip((lum - sky - noise) / max(1e-6, 3 * noise), 0, 1)
    if stars is not None:
        gain = gain * (1 - stars)
    gain = 1 + gain[..., None]
    out = lum[..., None] + (img - lum[..., None]) * gain
    return np.clip(out, 0, 1).astype("float32")


def _robust_mad(v) -> float:
    np = _np()
    return 1.4826 * float(np.median(np.abs(v - np.median(v)))) if v.size else 0.0


def _sky_level(a) -> tuple:
    """Sky level and noise of an image whose bright part (subject, stars) may cover up to half of it: read off
    the 12.5th and 25th percentiles, which only the sky reaches (for Gaussian noise they sit 1.15σ and 0.674σ
    below the mean). Returns (level, noise)."""
    np = _np()
    v = a.reshape(-1)[:: max(1, a.size // 500_000)]
    p12, p25 = np.percentile(v, [12.5, 25])
    sigma = max(1e-12, float(p25 - p12) / 0.476)
    return float(p25) + 0.674 * sigma, sigma


def _stars_linear(lum):
    """Star mask for linear data (found on a quick asinh preview)."""
    np = _np()
    peak = max(1e-9, float(np.percentile(lum, 99.9)))
    preview = np.arcsinh(np.clip(lum / peak, 0, None) * 200) / np.arcsinh(200)
    return star_mask(np.repeat(preview[..., None], 3, axis=2)) > 0.1


def _tile_grid(chan, mask, rows: int, cols: int):
    """Each tile's sky: the median of the darker half of its unmasked pixels (a nebula edge or a missed faint
    star pulls a plain median up; the darker half barely moves, and its constant offset cancels out)."""
    np = _np()
    h, w = chan.shape
    th, tw = h // rows, w // cols
    grid = np.full((rows, cols), np.nan, dtype="float32")
    for r in range(rows):
        for c in range(cols):
            m = mask[r * th:(r + 1) * th, c * tw:(c + 1) * tw]
            if (~m).mean() > 0.3:
                v = chan[r * th:(r + 1) * th, c * tw:(c + 1) * tw][~m]
                grid[r, c] = float(np.median(v[v <= np.median(v)]))
    return grid


def _fill_grid(grid):
    """Tiles covered by the subject take the average of their filled neighbours, ring by ring."""
    import warnings
    np = _np()
    rows, cols = grid.shape
    if np.isnan(grid).all():
        return None
    for _ in range(rows + cols):
        gaps = np.isnan(grid)
        if not gaps.any():
            break
        padded = np.pad(grid, 1, constant_values=np.nan)
        near = np.stack([padded[1 + dy:1 + dy + rows, 1 + dx:1 + dx + cols]
                         for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            fill = np.nanmean(near, axis=0)
        grid = np.where(gaps & ~np.isnan(fill), fill, grid)
    return np.nan_to_num(grid, nan=float(np.nanmedian(grid)))


def _background(img, cols: int = 16, rounds: int = 3, stretched: bool = False):
    """Model the background left after GraXpert, channel by channel, from sky-only tiles. The subject is found
    against the model itself and the model refitted without it, a few times over, so a smooth ramp across the
    frame (which a plain brightness test would take for nebula) is modelled, and the nebula's faint glow isn't.
    Returns (models, subject mask, star mask, rows, cols, clear-nebula mask)."""
    import cv2
    np = _np()
    lum = img @ np.array(IMX678_LUMA, dtype="float32")
    h, w = lum.shape
    cols = max(3, min(cols, w // 240))  # tiles of at least 240 px, so a nebula's glow isn't modelled as sky
    rows = max(2, round(cols * h / w))
    th, tw = h // rows, w // cols
    found_stars = (star_mask(img) > 0.1) if stretched else _stars_linear(lum)
    stars = cv2.dilate(found_stars.astype("uint8"), np.ones((5, 5), "uint8")).astype(bool)
    grow = max(7, w // 96) * 2 + 1  # 81 px on the DWARF's 3840 px frames, less on smaller images
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow, grow))
    # Anything plainly brighter than the sky on a smoothed copy is subject from the start (on a stretched image
    # the residual test alone can miss a small nebula, and the model would then eat it).
    soft = cv2.GaussianBlur(lum.astype("float32"), (0, 0), 6)
    s0, n0 = _sky_level(soft)
    bright = (soft > s0 + 5 * n0).astype("uint8")
    base = cv2.dilate(bright, k).astype(bool)
    if base.mean() > 0.6:  # a subject filling the frame: keep its core only
        base = bright.astype(bool)
    subject = base
    strong = base
    pixel_noise = max(1e-12, _noise(lum))
    hp = lum - cv2.GaussianBlur(lum, (0, 0), 2)
    models = None
    for _ in range(rounds):
        mask = stars | subject
        new = []
        for ch in range(3):
            grid = _fill_grid(_tile_grid(img[..., ch], mask, rows, cols))
            if grid is None:
                break
            # Smooth surface through the tile levels, built at an eighth of the size (a 120 px blur on the full
            # frame took over a second per channel), then scaled up.
            small = cv2.resize(grid, (max(2, w // 8), max(2, h // 8)), interpolation=cv2.INTER_CUBIC)
            # A light blur (a sixth of a tile): 0.13.1 blurred by half a tile, which flattened the edge tiles'
            # excess into the frame and left C 33's brighter edges in place.
            small = cv2.GaussianBlur(small, (0, 0), max(th, tw) / 48)
            new.append(cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR))
        if len(new) < 3:
            break  # too much of the frame masked: keep the last good model
        models = new
        lum_model = np.stack(models, axis=-1) @ np.array(IMX678_LUMA, dtype="float32")
        resid = cv2.GaussianBlur(lum - lum_model, (0, 0), 6)
        # The subject is what stands clearly above the model: a quarter of a pixel's noise on this smoothed
        # residual (well above its own noise, and above the model's small tile-scale ripples), raised if that
        # would cover most of the frame.
        level = max(3 * _robust_mad(resid[~mask]), 0.25 * pixel_noise)
        speck = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        for _ in range(8):
            found = cv2.morphologyEx((resid > level).astype("uint8"), cv2.MORPH_OPEN, speck)
            n, labels, stats, _ = cv2.connectedComponentsWithStats(found, connectivity=8)
            keep = stats[:, cv2.CC_STAT_AREA] >= 1500  # noise specks and leftover star haloes are smaller
            keep[0] = False
            found = keep[labels].astype("uint8")
            found = _drop_rotation_wedges(lum, found, hp)
            grown = cv2.dilate(found, k)
            if grown.mean() < 0.5:
                break
            level *= 1.4
        subject = grown.astype(bool) | base
        # Clear nebula: the subject patches that somewhere stand well above the model.
        n, labels, stats, _ = cv2.connectedComponentsWithStats(found, connectivity=8)
        peak = np.zeros(n, dtype="float32")
        np.maximum.at(peak, labels.reshape(-1), resid.reshape(-1))
        clear = peak > 3 * level
        clear[0] = False
        strong = cv2.dilate(clear[labels].astype("uint8"), k).astype(bool) | base
    return models, subject, stars, rows, cols, strong


def _drop_rotation_wedges(lum, subject, hp=None):
    """A patch at the frame's edge that's brighter than the sky because fewer frames cover it (field rotation)
    looks like nebulosity to a brightness test. Such a patch touches the border and is noisier than the open sky
    (fewer frames, more noise); a real nebula isn't. Those patches are taken out of the subject mask."""
    import cv2
    np = _np()
    if hp is None:
        hp = lum - cv2.GaussianBlur(lum, (0, 0), 2)
    clear = subject == 0
    if clear.sum() < 1000:
        return subject
    base = _robust_mad(hp[clear])
    n, labels, stats, _ = cv2.connectedComponentsWithStats(subject, connectivity=8)
    h, w = subject.shape
    near = 300  # rotation wedges hug the border; their faint outer part may stop short of it
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if not (x <= near or y <= near or x + bw >= w - near or y + bh >= h - near) or area > 0.25 * h * w:
            continue
        if _robust_mad(hp[labels == i]) > 1.2 * base:
            subject[labels == i] = 0
    return subject


def _spread(lum, mask, rows: int, cols: int, noise: float) -> float:
    """Background evenness: the range of the sky-only tile levels, in units of the noise of one tile's level
    (pixel noise ÷ √pixels). C 33's 0.12.11 natural ranged about +4 to -2 on a stretched, smoothed image; on
    linear data the same unevenness is far below one pixel's noise, so it has to be judged per tile."""
    np = _np()
    grid = _tile_grid(lum, mask, rows, cols)
    vals = grid[~np.isnan(grid)]
    if vals.size < 3:
        return 0.0
    h, w = lum.shape
    per_tile = max(1.0, (h // rows) * (w // cols) * float((~mask).mean()) / 2)
    return float((vals.max() - vals.min()) / max(1e-12, noise / per_tile ** 0.5))


def flatten_array(img, cols: int = 16, stretched: bool = False) -> dict:
    """Subtract what's left of the background (a ramp, a dark moat where GraXpert took the nebula's glow for
    sky, the brighter, redder edges and corners that fewer frames cover after field rotation), channel by
    channel, keeping each channel's mean. Used on the linear data and again on the stretched image, where any
    leftover shows most. Returns {"image", "before", "after", "subject", "stars"} (before/after: the sky's
    unevenness in tile-noise units, measured the same way on both)."""
    np = _np()
    w = np.array(IMX678_LUMA, dtype="float32")
    lum = img @ w
    models, subject, stars, rows, cols, strong = _background(img, cols, stretched=stretched)
    mask = subject | stars
    noise = max(1e-12, _noise(lum))
    before = _spread(lum, mask, rows, cols, noise)
    out, after = img, before
    if models is not None:
        hi = 1.0 if stretched else None
        flat = np.stack([img[..., c] - (models[c] - float(models[c].mean())) for c in range(3)], axis=-1)
        flat = np.clip(_edge_pass(flat, stars | strong), 0, hi)
        measured = _spread(flat @ w, mask, rows, cols, noise)
        if measured < before:
            out, after = flat.astype("float32"), measured
    return {"image": out, "before": before, "after": after, "subject": subject, "stars": stars}


def _edge_pass(img, mask, band: int = 600, tile: int = 80):
    """Field rotation brightens and tints the frame's edges and corners over a few hundred pixels: narrower than
    the main model's 240 px tiles, which flattened C 33's edges only partly. A finer model (80 px tiles), only in
    a band along the border, fading out by `band` px inward; only clear nebula is kept out of it, since a
    rotation corner itself looks like faint nebula to the main model."""
    import cv2
    np = _np()
    h, w = img.shape[:2]
    rows, cols = max(2, h // tile), max(2, w // tile)
    yy, xx = np.ogrid[:h, :w]
    dist = np.minimum(np.minimum(yy, h - 1 - yy), np.minimum(xx, w - 1 - xx)).astype("float32")
    fade = np.clip(1 - dist / band, 0, 1) ** 2
    out = img.copy()
    for ch in range(3):
        grid = _fill_grid(_tile_grid(img[..., ch], mask, rows, cols))
        if grid is None:
            return img
        inner = float(np.median(grid[rows // 4: rows - rows // 4, cols // 4: cols - cols // 4]))
        small = cv2.resize(grid - inner, (max(2, w // 8), max(2, h // 8)), interpolation=cv2.INTER_CUBIC)
        small = cv2.GaussianBlur(small, (0, 0), tile / 24)
        model = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        out[..., ch] = img[..., ch] - model * fade
    return out


def _flat_note(res: dict, where: str) -> str:
    if res["after"] < res["before"]:
        return (f"{where}: unevenness across the sky {res['before']:.0f} → {res['after']:.0f} tile-noise levels "
                f"(stars and the subject, {res['subject'].mean() * 100:.0f}% of the frame, left out of the model)")
    return ""


def flatten(src: Path, dst: Path, cols: int = 16, oiii: bool = False) -> dict:
    """Linear RGB after background extraction and colour calibration: flatten_array, then for Duo-Band give
    OIII its teal in natural colour (see _teal_oiii). Returns {"note", "before", "after", "colour"}."""
    img, header = read_fits_rgb(src)
    res = flatten_array(img, cols)
    out, colour = res["image"], ""
    if oiii:
        out, colour = _teal_oiii(out, res["subject"], res["stars"])
    write_fits_rgb(dst, out.astype("float32"), header)
    return {"note": _flat_note(res, "background flattened"), "before": res["before"], "after": res["after"],
            "colour": colour}


def flatten_display(img) -> tuple:
    """The same flattening on the stretched image (the faint-nebula lift also lifts any leftover edge glow, as on
    C 33 in 0.13.1). Returns (image, note)."""
    res = flatten_array(img, stretched=True)
    return res["image"], _flat_note(res, "final image's background evened")


def _teal_oiii(img, subject, stars):
    """Duo-Band natural colour. Through the dual-band filter the red pixels see Hα and the blue ones OIII; the
    green ones see OIII too but came out about 20% under blue after narrowband SPCC, so OIII read blue and the
    Veil purple. 0.13.0 scaled all green (OIII went green, Hα orange); 0.13.1 moved green toward blue over the
    whole subject mask, which caught sky mottling and left green blotches. Now green follows blue only where
    there's real OIII: blue clearly above the sky (5× its smoothed noise, off the stars) in patches big enough to
    be nebula and touching the subject, weighted by how much OIII dominates Hα there. The sky isn't touched."""
    import cv2
    np = _np()
    stars_wide = cv2.dilate(stars.astype("uint8"), np.ones((7, 7), "uint8")).astype(bool)
    sky_px = ~subject & ~stars_wide
    if sky_px.sum() < 10000:
        return img, ""
    level = [float(np.median(img[..., c][sky_px])) for c in range(3)]
    # Smoothed, star-free signal above the sky (stars replaced by the sky level before smoothing).
    sig = []
    for c in range(3):
        ch = np.where(stars_wide, level[c], img[..., c]).astype("float32")
        sig.append(cv2.GaussianBlur(ch, (0, 0), 3) - level[c])
    noise_b = max(1e-12, _robust_mad(sig[2][sky_px]))
    real = (sig[2] > 5 * noise_b).astype("uint8")
    n, labels, stats, _ = cv2.connectedComponentsWithStats(real, connectivity=8)
    touch = np.zeros(n, dtype=bool)
    touch[np.unique(labels[subject & (real > 0)])] = True
    keep = (stats[:, cv2.CC_STAT_AREA] >= 1500) & touch
    keep[0] = False
    real = keep[labels]
    if real.sum() < 2000:
        return img, ""
    dominance = np.clip(1.2 * sig[2] / np.maximum(np.clip(sig[0], 0, None) + np.clip(sig[2], 0, None), 1e-12), 0, 1)
    weight = dominance * cv2.GaussianBlur(real.astype("float32"), (0, 0), 15) * (~stars_wide)
    out = img.copy()
    # Green takes blue's signal above the sky (blue's offset from green's sky level removed), never the sky itself.
    out[..., 1] = img[..., 1] + weight * ((img[..., 2] - level[2] + level[1]) - img[..., 1])
    out = np.clip(out, 0, None)
    o_dom = (dominance > 0.8) & real
    h_dom = (sig[0] > 2 * np.clip(sig[2], 0, None)) & real
    def ratio(a, b, m):
        return float(np.sum(a[m]) / max(1e-12, np.sum(b[m]))) if m.sum() > 100 else float("nan")
    o_ratio = ratio(out[..., 1] - level[1], out[..., 2] - level[2], o_dom)
    h_ratio = ratio(out[..., 1] - level[1], out[..., 0] - level[0], h_dom)
    return out, (f"OIII made teal for natural colour on {real.mean() * 100:.0f}% of the frame where it's clearly "
                 f"there (green/blue on OIII {o_ratio:.2f}; green/red on Hα {h_ratio:.2f})")


def even_noise(img, stars=None, tile: int = 96) -> tuple:
    """Stretched RGB: field rotation leaves the frame's edges and corners with fewer frames, so they're noisier
    (C 33's bottom-left corner: 1.5× the centre). Map the noise in tiles across the sky and smooth brightness
    just enough where a tile is over 1.2× the typical noise to bring it back, never on stars or the subject.
    Returns (image, note)."""
    import cv2
    np = _np()
    lum = (img @ np.array(IMX678_LUMA, dtype="float32")).astype("float32")
    soft = cv2.GaussianBlur(lum, (0, 0), 6)
    soft_sky, soft_noise = _sky_level(soft)
    subject = (soft > soft_sky + 4 * soft_noise).astype("uint8")
    subject = _drop_rotation_wedges(lum, subject)
    subject = cv2.dilate(subject, np.ones((21, 21), "uint8")).astype(bool)
    mask = subject | ((stars > 0.1) if stars is not None else False)
    hp = lum - cv2.GaussianBlur(lum, (0, 0), 2)
    h, w = lum.shape
    rows, cols = max(1, h // tile), max(1, w // tile)
    grid = np.full((rows, cols), np.nan, dtype="float32")
    for r in range(rows):
        for c in range(cols):
            m = mask[r * tile:(r + 1) * tile, c * tile:(c + 1) * tile]
            if (~m).mean() > 0.5:
                grid[r, c] = _robust_mad(hp[r * tile:(r + 1) * tile, c * tile:(c + 1) * tile][~m])
    if np.isnan(grid).all():
        return img, ""
    ratio = np.nan_to_num(grid / max(float(np.nanmedian(grid)), 1e-12), nan=1.0)
    worst = float(ratio.max())
    if worst < 1.2:
        return img, ""
    # A 1.6 px blur keeps about a fifth of the pixel noise; mix in just enough of it to reach the typical noise.
    weight = np.where(ratio > 1.15, np.clip((1 - 1 / np.maximum(ratio, 1e-6)) / 0.8, 0, 1), 0).astype("float32")
    weight = cv2.resize(weight, (cols * tile, rows * tile), interpolation=cv2.INTER_NEAREST)
    weight = cv2.copyMakeBorder(weight, 0, h - rows * tile, 0, w - cols * tile, cv2.BORDER_REPLICATE)
    weight = cv2.GaussianBlur(weight, (0, 0), tile / 2) * (~mask)
    new = lum + weight * (cv2.GaussianBlur(lum, (0, 0), 1.6) - lum)
    out = np.clip(img + (new - lum)[..., None], 0, 1).astype("float32")
    share = float((weight > 0.2).mean() * 100)
    return out, (f"edges that fewer frames cover (field rotation; {share:.0f}% of the frame, up to {worst:.1f}× "
                 "noisier) smoothed to match the rest")


def keep_ha_red(img, stars=None, cap: float = 0.45) -> tuple:
    """Stretched Duo-Band natural colour: Hα should read red. If green on the Hα-dominant nebula is over half of
    red (C 33 in 0.13.1 reached 0.88 after the stretch, which looks yellow-orange), pull it back to `cap` × red,
    only there. Returns (image, note)."""
    import cv2
    np = _np()
    level = [_sky_level(img[..., c])[0] for c in range(3)]
    st = (stars > 0.1) if stars is not None else np.zeros(img.shape[:2], bool)
    sig = [cv2.GaussianBlur(np.where(st, level[c], img[..., c]).astype("float32"), (0, 0), 2) - level[c]
           for c in range(3)]
    noise_r = max(1e-9, _sky_level(sig[0])[1])
    ha = (sig[0] > 5 * noise_r) & (sig[0] > 2 * np.clip(sig[2], 0, None)) & ~st
    if ha.sum() < 2000:
        return img, ""
    ratio = float(np.sum(sig[1][ha]) / max(1e-9, np.sum(sig[0][ha])))
    if ratio <= 0.5:
        return img, ""
    weight = cv2.GaussianBlur(ha.astype("float32"), (0, 0), 4)
    g_sig, r_sig = img[..., 1] - level[1], img[..., 0] - level[0]
    excess = np.clip(g_sig - cap * np.clip(r_sig, 0, None), 0, None)
    out = img.copy()
    out[..., 1] = img[..., 1] - weight * excess
    out = np.clip(out, 0, 1).astype("float32")
    after = float(np.sum((out[..., 1] - level[1])[ha]) / max(1e-9, np.sum(r_sig[ha])))
    return out, f"Hα kept red (green/red on it {ratio:.2f} → {after:.2f})"


def sky_blotches(img, tile: int = 96) -> float:
    """Green or magenta mottling of the sky in a stretched image: the spread (standard deviation) of the star- and
    subject-free tile averages of green − (red + blue)/2, on the 0–1 scale. Below about 0.004 (one 8-bit level)
    it can't be seen; 0.13.1's green blotches on C 33 were well above."""
    import cv2
    np = _np()
    lum = img @ np.array(IMX678_LUMA, dtype="float32")
    soft = cv2.GaussianBlur(lum.astype("float32"), (0, 0), 6)
    s0, n0 = _sky_level(soft)
    mask = (soft > s0 + 4 * n0) | (star_mask(img) > 0.05)
    tint = img[..., 1] - (img[..., 0] + img[..., 2]) / 2
    h, w = lum.shape
    tile = max(24, min(tile, w // 16))
    vals = []
    for r in range(h // tile):
        for c in range(w // tile):
            m = ~mask[r * tile:(r + 1) * tile, c * tile:(c + 1) * tile]
            if m.mean() > 0.5:
                vals.append(float(np.mean(tint[r * tile:(r + 1) * tile, c * tile:(c + 1) * tile][m])))
    return float(np.std(vals)) if len(vals) >= 10 else 0.0


# --- Layered finishing (0.15.0) --------------------------------------------------------------------------------
# Worked out on Kip's C 33 against the DWARF's own picture of it, which beat 0.13.2 and 0.14.0 on contrast and sky
# smoothness. The nebula and the sky are finished on StarNet's starless layer, where stars can't be mistaken for
# nebula or noise, and the stars are put back afterwards with a screen blend. Prototype on C 33: contrast 69 natural,
# 79 HOO (DWARF 58, 0.14.0 36), sky noise 0.0030 (DWARF 0.0027, 0.14.0 0.0049), flatness 0.43 (DWARF 1.03).

# ha_warm 0.8 turned bright Hα orange-yellow and 0.6 was still orange; 0.5 with ha_blue 0.3 is the DWARF's brick
# red. For HOO, OIII scaled to 0.75 of Hα's strength is balanced (1.1 went cyan).
# star_gamma 1.5 at scale 1.15 gives tight but bright stars; a floor of 0.035 drops the red specks StarNet leaves
# as "stars"; sky_blur 3 at mix 0.8 took the sky noise from 0.0034 to 0.0030 without looking plastic.
LAYERED = dict(stretch=100.0, black_sigma=2.5, sky_target=0.085, neb_target=0.27, neb_sat=1.35, neb_dn=0.6,
               star_scale=1.15, star_sat=1.0, star_gamma=1.5, star_floor=0.035, sky_dn=14.0, sky_blur=3.0, sky_mix=0.8,
               sky_grey=0.6, ha_warm=0.5, ha_blue=0.3, hoo=0, o3_boost=0.75)


def _mad(x) -> float:
    np = _np()
    m = np.median(x)
    return 1.4826 * float(np.median(np.abs(x - m)))


def even_linear(img) -> tuple:
    """Linear RGB: the edges that fewer frames covered after field rotation are noisier; blend them toward a
    blurred copy just as far as their extra noise. Returns (image, share of the frame over 1.3x, worst ratio)."""
    import cv2
    np = _np()
    lum = img.mean(axis=2)
    hp = lum - cv2.GaussianBlur(lum, (0, 0), 2)
    ts = 64
    h, w = lum.shape
    nm = cv2.resize(np.abs(hp), (max(1, w // ts), max(1, h // ts)), interpolation=cv2.INTER_AREA)
    nm = cv2.medianBlur(nm.astype("float32"), 3)
    typ = max(1e-12, float(np.median(nm)))
    ratio = cv2.GaussianBlur(cv2.resize(nm / typ, (w, h), interpolation=cv2.INTER_LINEAR), (0, 0), 24)
    k = np.clip((ratio - 1.15) / 0.6, 0, 1)[..., None]  # none where normal, full where 1.75x noisier or more
    smooth = np.stack([cv2.GaussianBlur(img[..., c], (0, 0), 3.0) for c in range(3)], -1)
    return (img * (1 - k) + smooth * k).astype("float32"), float((ratio > 1.3).mean()), float(ratio.max())


def base_stretch(lin, stretch: float = 100.0, black_sigma: float = 2.5, sky: float = 0.068):
    """Linear RGB -> stretched RGB for StarNet and the layered finish: each channel's sky removed, an arcsinh
    stretch of the brightness that keeps every pixel's colour ratios, then a midtone transfer that puts the sky at
    `sky` (kept a little under the final target, as the nebula curve and the star screen add some)."""
    np = _np()
    sky0 = np.array([_sky(lin[..., c][::4, ::4]) for c in range(3)], dtype="float32")
    x = lin - sky0
    lum = x.mean(axis=2)
    sub = lum[::4, ::4]
    sig = _mad(sub[sub < np.percentile(lum, 50)])
    peak = float(np.percentile(lum, 99.99))
    bp = -black_sigma * sig
    ln = np.clip((lum - bp) / max(1e-9, peak - bp), 0, None)
    ls = np.arcsinh(stretch * ln) / np.arcsinh(stretch)
    ratio = np.where(lum - bp > 1e-7, ls / np.maximum(ln, 1e-7), 0)[..., None] / max(1e-9, peak - bp)
    rgb = np.clip((x - bp) * ratio, 0, 1)
    s0 = _sky(rgb.mean(axis=2)[::4, ::4])
    if 0 < s0 < 1 and s0 != sky:
        m = (sky - 1) * s0 / ((2 * sky - 1) * s0 - sky)  # the midtone that maps s0 to sky
        rgb = np.clip(((m - 1) * rgb) / ((2 * m - 1) * rgb - m), 0, 1)
    return rgb.astype("float32")


def starless_subject(starless):
    """Where the nebula is, from the starless layer: big structure well above the sky (sigma-12 blur, above sky +
    6 noise levels, 20000 px or more) that doesn't touch the frame's border, grown by 30 px. Edge glow touches the
    border and isn't subject; 0.13's mask counted it in (38-52% of C 33's frame against 8%), so it was never
    flattened. If nothing qualifies (a subject that fills the frame), big structure touching the border counts."""
    import cv2
    np = _np()
    bl = cv2.GaussianBlur(starless.mean(axis=2), (0, 0), 12)
    sub = bl[::4, ::4]
    sky, noise = _sky(sub), _mad(sub[sub < np.percentile(bl, 50)])
    cand = (bl > sky + 6 * noise).astype("uint8")
    n, lab, stats, _ = cv2.connectedComponentsWithStats(cand, 8)
    h, w = cand.shape
    keep, border = [], []
    for i in range(1, n):
        x0, y0, ww, hh, area = stats[i]
        if area < 20000:
            continue
        (border if x0 <= 2 or y0 <= 2 or x0 + ww >= w - 2 or y0 + hh >= h - 2 else keep).append(i)
    ids = keep or border
    subj = np.isin(lab, ids).astype("uint8") if ids else np.zeros_like(cand)
    return cv2.dilate(subj, np.ones((61, 61), np.uint8)).astype("float32")


def flatten_tiles(img, mask, ts: int = 96):
    """Stretched RGB: a background model from 96 px tiles (each the median of its darker 60% of sky pixels),
    tiles mostly covered by the subject filled in from their neighbours, smoothed and subtracted (the median level
    kept). Handles the rotation wedges in the corners, which a smooth polynomial can't."""
    import cv2
    np = _np()
    h, w = img.shape[:2]
    gh, gw = -(-h // ts), -(-w // ts)
    hole = np.zeros((gh, gw), np.uint8)
    grids = np.zeros((3, gh, gw), np.float32)
    for r in range(gh):
        for c in range(gw):
            tile = (slice(r * ts, min(h, (r + 1) * ts)), slice(c * ts, min(w, (c + 1) * ts)))
            m = mask[tile] < 0.5
            if m.mean() < 0.3:
                hole[r, c] = 1
                continue
            for ch in range(3):
                v = img[tile][..., ch][m]
                grids[ch, r, c] = np.median(v[v <= np.percentile(v, 60)])
    if hole.all():
        return img
    out = img.copy()
    for ch in range(3):
        g = grids[ch]
        lo, hi = float(g[hole == 0].min()), float(g[hole == 0].max())
        g8 = np.clip((g - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
        filled = cv2.inpaint(g8, hole, 3, cv2.INPAINT_TELEA).astype(np.float32) / 255 * (hi - lo) + lo
        filled = cv2.GaussianBlur(np.where(hole == 1, filled, g).astype(np.float32), (0, 0), 1.0)
        model = cv2.resize(filled, (gw * ts, gh * ts), interpolation=cv2.INTER_CUBIC)[:h, :w]
        model = cv2.GaussianBlur(model, (0, 0), ts / 3)
        out[..., ch] = img[..., ch] - model + float(np.median(filled))
    return np.clip(out, 0, 1).astype("float32")


def edge_band(img, band: int = 300):
    """Stretched RGB: the outermost lines of each side can still glow after flattening (C 33's top and left 40 px
    sat 0.126/0.136 against a 0.102 centre). Per-line sky medians in the outer `band` px, smoothed, and only the
    excess over the interior removed, fading to nothing at `band` px in."""
    import cv2
    np = _np()
    h, w = img.shape[:2]
    if h < 3 * band or w < 3 * band:
        band = max(8, min(h, w) // 6)
    out = img.copy()
    fade = np.linspace(1, 0, band) ** 0.5
    for c in range(3):
        ch = out[..., c]
        inner = float(np.median(ch[band:h - band, band:w - band][::4, ::4]))
        for side in ("top", "bottom", "left", "right"):
            lines = [ch[i] if side == "top" else ch[h - 1 - i] if side == "bottom" else ch[:, i] if side == "left"
                     else ch[:, w - 1 - i] for i in range(band)]
            prof = np.array([np.median(line[band:-band]) for line in lines], dtype="float32") - inner
            prof = cv2.GaussianBlur(prof.reshape(-1, 1), (0, 0), 6).ravel()
            prof = np.clip(prof, 0, None) * fade
            if side == "top":
                ch[:band] -= prof[:, None]
            elif side == "bottom":
                ch[h - band:] -= prof[::-1][:, None]
            elif side == "left":
                ch[:, :band] -= prof[None, :]
            else:
                ch[:, w - band:] -= prof[::-1][None, :]
    return np.clip(out, 0, 1).astype("float32")


def layered_finish(rgb, starless, p: dict | None = None) -> tuple:
    """Finish the nebula and sky on the starless layer, then screen the stars back. rgb is base_stretch's output
    and starless StarNet's version of it. Returns (final, finished starless, finished stars, notes)."""
    import cv2
    np = _np()
    p = {**LAYERED, **(p or {})}
    notes = []
    stars = np.clip(1 - (1 - rgb) / np.maximum(1 - starless, 1e-4), 0, 1)
    mask = starless_subject(starless)
    notes.append(f"subject {mask.mean() * 100:.0f}% of the frame (border glow left out)")
    sl = flatten_tiles(starless, mask)
    # Nebula curve towards a measured brightness above the sky, ramped in from the noise.
    lum = sl.mean(axis=2)
    ssky = _sky(lum[::4, ::4])
    snoise = _mad(cv2.GaussianBlur(lum, (0, 0), 2)[::4, ::4][lum[::4, ::4] < np.percentile(lum, 50)])
    small = cv2.resize(lum, (lum.shape[1] // 2, lum.shape[0] // 2), interpolation=cv2.INTER_AREA)
    neb = float(np.percentile(cv2.medianBlur((np.clip(small, 0, 1) * 255).astype(np.uint8), 5), 99) / 255 - ssky)
    gain = float(np.clip(p["neb_target"] / max(neb, 1e-4), 1.0, 3.5))
    above = np.clip(lum - ssky, 0, None)
    wgt = cv2.GaussianBlur(np.clip((above - 1.5 * snoise) / max(1e-9, 4 * snoise), 0, 1).astype("float32"), (0, 0), 3)
    new = ssky + above * (1 + (gain - 1) * wgt)
    new = np.minimum(new, 1 - (1 - np.clip(new, 0, 1)) * 0.98)
    sl = sl * (new / np.maximum(lum, 1e-5))[..., None]
    notes.append(f"nebula lifted ×{gain:.2f} (from {neb:.3f} above the sky)")
    # Brightness-only denoise of the sky (edge-preserving), outside the nebula.
    lum = sl.mean(axis=2).astype("float32")
    den = cv2.bilateralFilter(lum, 0, p["sky_dn"] * snoise * 3, p["sky_dn"])
    sl = sl * (1 + (1 - wgt)[..., None] * ((den / np.maximum(lum, 1e-5))[..., None] - 1))
    if p["neb_dn"] > 0:  # and a lighter one inside the nebula
        lum = sl.mean(axis=2).astype("float32")
        den = cv2.bilateralFilter(lum, 0, 6 * snoise * 3, 4)
        sl = sl * (1 + (p["neb_dn"] * wgt)[..., None] * ((den / np.maximum(lum, 1e-5))[..., None] - 1))
    # Saturation on the nebula only; Hα warmed to orange-red (green toward ha_warm x red, blue down toward ha_blue x red).
    # Saturation only on the subject: on synthetic data with faint large-scale colour mottling (what turned green
    # in 0.13.1) saturating everything above the noise made green blotches of the sky.
    swgt = wgt * cv2.GaussianBlur(mask, (0, 0), 20)
    lum = sl.mean(axis=2, keepdims=True)
    sl = lum + (sl - lum) * (1 + (p["neb_sat"] - 1) * swgt[..., None])
    # The sky's faint colour eased toward grey (what colour_stretch's protect did), so mottling can't show.
    sl = lum + (sl - lum) * (1 - p["sky_grey"] * (1 - wgt))[..., None]
    r, g, b = sl[..., 0], sl[..., 1], sl[..., 2]
    skyc = [_sky(sl[..., c][::4, ::4]) for c in range(3)]
    if p["hoo"]:  # HOO from the starless layer: red = Hα, green = blue = OIII, scaled toward Hα's strength
        o3 = (g - skyc[1] + b - skyc[2]) / 2
        neb = wgt > 0.5
        scale = 1.0
        if neb.sum() > 100:
            scale = float(np.percentile((r - skyc[0])[neb], 99) / max(np.percentile(o3[neb], 99), 1e-5)) * p["o3_boost"]
        o3 = o3 * (1 + (float(np.clip(scale, 1.0, 3.0)) - 1) * wgt)
        g, b = skyc[1] + o3, skyc[2] + o3
    else:
        ra, ga, ba = r - skyc[0], g - skyc[1], b - skyc[2]
        ha = (ra > 1.5 * ba) & (swgt > 0.3)
        g = np.where(ha & (ga < p["ha_warm"] * ra), g + 0.6 * (skyc[1] + p["ha_warm"] * ra - g), g)
        if p["ha_blue"] > 0:
            b = np.where(ha & (ba > p["ha_blue"] * ra), b - 0.6 * (ba - p["ha_blue"] * ra), b)
    sl = flatten_tiles(np.clip(np.stack([r, g, b], -1), 0, 1).astype("float32"), mask)  # the steps above bring edges back
    if p["sky_blur"] > 0:
        lb = sl.mean(axis=2)
        lbb = cv2.GaussianBlur(lb, (0, 0), p["sky_blur"])
        mix = (p["sky_mix"] * (1 - wgt))[..., None]
        sl = np.clip(sl * (1 + mix * ((lbb / np.maximum(lb, 1e-5))[..., None] - 1)), 0, 1)
    # Stars: a little tighter (gamma), scaled, colour toned down; StarNet's noise specks (worst in the low-coverage
    # edges) dropped by a floor; the stars' shared tint removed.
    stars = stars ** p["star_gamma"]
    slum = stars.mean(axis=2, keepdims=True)
    stars = np.clip(slum + (stars - slum) * p["star_sat"], 0, 1) * p["star_scale"]
    stars = np.clip((stars - p["star_floor"]) / (1 - p["star_floor"]), 0, 1)
    sm = stars.mean(axis=2) > 0.15
    if sm.sum() > 100:
        tint = np.median(stars[sm] - stars[sm].mean(axis=1, keepdims=True), axis=0)
        stars = np.clip(stars - tint * (stars.mean(axis=2, keepdims=True) > 0.02), 0, 1)
    final = 1 - (1 - sl) * (1 - stars)
    # The sky at the target, neutral, and no glow along the outermost lines.
    final = np.clip(final - _sky(final.mean(axis=2)[::4, ::4]) + p["sky_target"], 0, 1)
    fs = np.array([_sky(final[..., c][::4, ::4]) for c in range(3)])
    final = edge_band(np.clip(final - fs + fs.mean(), 0, 1).astype("float32"))
    sl = np.clip(sl - _sky(sl.mean(axis=2)[::4, ::4]) + p["sky_target"], 0, 1)
    return final.astype("float32"), sl.astype("float32"), stars.astype("float32"), notes

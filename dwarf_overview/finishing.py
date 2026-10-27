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
    ratio = new / np.maximum(lum, 1e-6)
    out = img * ratio[..., None]
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


def _subject_mask(lum, sky: float, dilate_px: int = 40):
    """Linear brightness -> True where the subject (nebula, galaxy) or a star is, so a background model or a
    coverage fix keeps off it: a heavily blurred copy more than a couple of its own noise levels above the sky,
    grown by dilate_px so the faint outer glow is left out too."""
    import cv2
    np = _np()
    soft = cv2.GaussianBlur(lum.astype("float32"), (0, 0), 6)
    low = soft[soft <= np.percentile(soft, 50)]
    noise = max(1e-9, 1.4826 * float(np.median(np.abs(low - np.median(low)))))
    subject = (soft > sky + 2.5 * noise).astype("uint8")
    subject = _drop_rotation_wedges(lum, subject)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
    subject = cv2.dilate(subject, k)
    peak = max(1e-9, float(np.percentile(lum, 99.9)))
    preview = np.arcsinh(np.clip(lum / peak, 0, None) * 200) / np.arcsinh(200)
    stars = star_mask(np.repeat(preview[..., None], 3, axis=2)) > 0.1
    return subject.astype(bool) | stars


def _drop_rotation_wedges(lum, subject):
    """A patch at the frame's edge that's brighter than the sky because fewer frames cover it (field rotation)
    looks like nebulosity to a brightness test. Such a patch touches the border and is noisier than the open sky
    (fewer frames, more noise); a real nebula isn't. Those patches are taken out of the subject mask, so the
    background model and the coverage smoothing can fix them."""
    import cv2
    np = _np()
    hp = lum - cv2.GaussianBlur(lum, (0, 0), 2)
    clear = subject == 0
    if clear.sum() < 1000:
        return subject
    base = 1.4826 * float(np.median(np.abs(hp[clear] - np.median(hp[clear]))))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(subject, connectivity=8)
    h, w = subject.shape
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if not (x == 0 or y == 0 or x + bw >= w or y + bh >= h):
            continue
        if area > 0.25 * h * w:
            continue
        v = hp[labels == i]
        noise = 1.4826 * float(np.median(np.abs(v - np.median(v))))
        if noise > 1.2 * base:
            subject[labels == i] = 0
    return subject


def _tile_spread(lum, mask, sky: float, noise: float, cols: int = 16) -> float:
    """Background evenness: the range of sky-only tile medians, in sky-noise units (the C 33 natural from 0.12.11
    spanned about 6: +4 on the left edge and in the rotation corner, -2 in the moat under the Veil)."""
    np = _np()
    h, w = lum.shape
    rows = max(1, round(cols * h / w))
    th, tw = h // rows, w // cols
    meds = []
    for r in range(rows):
        for c in range(cols):
            t, m = lum[r * th:(r + 1) * th, c * tw:(c + 1) * tw], mask[r * th:(r + 1) * th, c * tw:(c + 1) * tw]
            if (~m).mean() > 0.3:
                meds.append(float(np.median(t[~m])))
    return (max(meds) - min(meds)) / max(noise, 1e-9) if len(meds) > 2 else 0.0


def flatten(src: Path, dst: Path, cols: int = 16) -> dict:
    """Linear RGB after background extraction and colour calibration: model what's left of the background from
    sky-only tiles (stars and the subject masked and grown) and subtract it, channel by channel, keeping each
    channel's mean. This catches what GraXpert leaves: a ramp across the frame, a dark moat where it took the
    nebula's glow for sky, and the brighter, redder corners that fewer frames cover after field rotation."""
    import cv2
    np = _np()
    img, header = read_fits_rgb(src)
    lum = img @ np.array(IMX678_LUMA, dtype="float32")
    sky = _sky(lum)
    noise = max(1e-9, _noise(lum))  # pixel noise of the sky, so the spread reads in "noise levels"
    mask = _subject_mask(lum, sky)
    h, w = lum.shape
    rows = max(1, round(cols * h / w))
    th, tw = h // rows, w // cols
    before = _tile_spread(lum, mask, sky, noise, cols)
    if before <= 1.5:  # already even: nothing worth changing
        write_fits_rgb(dst, img, header)
        return {"note": "", "before": before, "after": before}
    out = img.copy()
    for ch in range(3):
        grid = np.full((rows, cols), np.nan, dtype="float32")
        for r in range(rows):
            for c in range(cols):
                t = img[r * th:(r + 1) * th, c * tw:(c + 1) * tw, ch]
                m = mask[r * th:(r + 1) * th, c * tw:(c + 1) * tw]
                if (~m).mean() > 0.3:
                    v = t[~m]
                    med = np.median(v)
                    mad = 1.4826 * np.median(np.abs(v - med)) + 1e-9
                    grid[r, c] = float(np.median(v[np.abs(v - med) < 3 * mad]))
        if np.isnan(grid).all():
            return {"note": "", "before": before, "after": before}
        # Tiles covered by the subject take the average of their sky neighbours (repeated until all are filled).
        for _ in range(rows + cols):
            gaps = np.isnan(grid)
            if not gaps.any():
                break
            padded = np.pad(grid, 1, constant_values=np.nan)
            near = np.stack([padded[1 + dy:1 + dy + rows, 1 + dx:1 + dx + cols]
                             for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx])
            with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
                __import__("warnings").simplefilter("ignore", RuntimeWarning)
                fill = np.nanmean(near, axis=0)
            grid = np.where(gaps & ~np.isnan(fill), fill, grid)
        grid = np.nan_to_num(grid, nan=float(np.nanmedian(grid)))
        model = cv2.resize(grid, (w, h), interpolation=cv2.INTER_CUBIC)
        model = cv2.GaussianBlur(model, (0, 0), max(th, tw) / 2)
        out[..., ch] = img[..., ch] - (model - float(model.mean()))
    out = np.clip(out, 0, None)
    lum2 = out @ np.array(IMX678_LUMA, dtype="float32")
    after = _tile_spread(lum2, mask, _sky(lum2), noise, cols)
    if after >= before:  # never make it worse
        write_fits_rgb(dst, img, header)
        return {"note": f"background already even ({before:.1f} noise levels across the sky)", "before": before, "after": before}
    write_fits_rgb(dst, out, header)
    return {"note": f"background flattened from {before:.1f} to {after:.1f} noise levels across the sky (stars and "
                    "subject left out of the model)", "before": before, "after": after}


def balance_oiii(src: Path, dst: Path) -> str:
    """Duo-Band, natural colour: OIII (500.7 nm) lands in both the green and the blue pixels, so it should read
    teal (green ≈ blue). After narrowband SPCC the C 33 nebula's green came out about 20% under its blue, so OIII
    looked blue and the Veil purple. Scale green about the sky so the nebula's green signal matches its blue."""
    np = _np()
    img, header = read_fits_rgb(src)
    lum = img @ np.array(IMX678_LUMA, dtype="float32")
    sky = [_sky(img[..., c]) for c in range(3)]
    import cv2
    soft = cv2.GaussianBlur(lum, (0, 0), 3)
    low = soft[soft <= np.percentile(soft, 50)]
    soft_noise = max(1e-9, 1.4826 * float(np.median(np.abs(low - np.median(low)))))
    peak = max(1e-9, float(np.percentile(lum, 99.9)))
    preview = np.arcsinh(np.clip(lum / peak, 0, None) * 200) / np.arcsinh(200)
    stars = star_mask(np.repeat(preview[..., None], 3, axis=2)) > 0.05
    # The nebula proper (well above the sky), without stars; its faint fringe would mostly measure noise.
    neb = (_drop_rotation_wedges(lum, (soft > _sky(lum) + 6 * soft_noise).astype("uint8")) > 0) & ~stars
    if neb.sum() < 2000:
        write_fits_rgb(dst, img, header)
        return ""
    g = float(np.sum(img[..., 1][neb] - sky[1]))
    b = float(np.sum(img[..., 2][neb] - sky[2]))
    if g <= 0 or b <= 0:
        write_fits_rgb(dst, img, header)
        return ""
    scale = min(1.4, max(0.8, b / g))
    if abs(scale - 1) < 0.03:
        write_fits_rgb(dst, img, header)
        return ""
    out = img.copy()
    out[..., 1] = sky[1] + (img[..., 1] - sky[1]) * scale
    write_fits_rgb(dst, np.clip(out, 0, None), header)
    return f"OIII balanced to teal for natural colour (green ×{scale:.2f} on the signal, the sky unchanged)"


def coverage_denoise(src: Path, dst: Path, tile: int = 96) -> str:
    """Field rotation leaves the frame's edges and corners with fewer frames, so they're noisier (C 33's bottom-
    left corner: about 1.5× the centre). Map the noise in tiles across the sky and smooth the brightness more
    where it's higher than the frame's typical noise, never on stars or the subject."""
    import cv2
    np = _np()
    img, header = read_fits_rgb(src)
    lum = img @ np.array(IMX678_LUMA, dtype="float32")
    mask = _subject_mask(lum, _sky(lum), dilate_px=10)
    hp = lum - cv2.GaussianBlur(lum, (0, 0), 2)
    h, w = lum.shape
    rows, cols = max(1, h // tile), max(1, w // tile)
    grid = np.full((rows, cols), np.nan, dtype="float32")
    for r in range(rows):
        for c in range(cols):
            t = hp[r * tile:(r + 1) * tile, c * tile:(c + 1) * tile]
            m = mask[r * tile:(r + 1) * tile, c * tile:(c + 1) * tile]
            if (~m).mean() > 0.5:
                v = t[~m]
                grid[r, c] = 1.4826 * float(np.median(np.abs(v - np.median(v))))
    if np.isnan(grid).all():
        write_fits_rgb(dst, img, header)
        return ""
    typical = float(np.nanmedian(grid))
    ratio = np.nan_to_num(grid / max(typical, 1e-12), nan=1.0)
    worst = float(ratio.max())
    if worst < 1.2:
        write_fits_rgb(dst, img, header)
        return ""
    # Just enough smoothing to bring each tile back to the typical noise (a 1.6 px blur keeps about a fifth of
    # it); more would leave plasticky patches.
    weight = np.where(ratio > 1.1, np.clip((1 - 1 / np.maximum(ratio, 1e-6)) / 0.8, 0, 1), 0).astype("float32")
    weight = cv2.GaussianBlur(cv2.resize(weight, (w, h), interpolation=cv2.INTER_LINEAR), (0, 0), tile / 2)
    weight = weight * (~mask)
    soft = cv2.GaussianBlur(lum, (0, 0), 1.6)
    out = img + (weight * (soft - lum))[..., None]
    write_fits_rgb(dst, np.clip(out, 0, None), header)
    share = float((weight > 0.2).mean() * 100)
    return f"the {share:.0f}% of the frame that fewer frames cover (edges after field rotation, up to {worst:.1f}× noisier) smoothed more"

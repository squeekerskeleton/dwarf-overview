"""Quick quality scores for raw frames, so the DWARF's rejected (failed_*) frames can be held to a much
stricter bar than the frames it passed.

For each frame: the Bayer 2x2 blocks are averaged into a half-size luminance image, stars are found with a
white top-hat, and the frame gets
- its star count (clouds, dew and haze lower it),
- the median FWHM of its brighter unsaturated stars (focus, seeing and trailing raise it),
- the median roundness (trailing and wind lower it),
- wFWHM = FWHM x (most stars in any frame / its stars), Siril's own combined measure: lower is better.

A rejected frame is let back in only if it beats the passed frames' median wFWHM and star count and is about as
round as a typical passed frame; it still has to get
through Siril's frame selection afterwards, and at most a tenth of the stack can be rejected frames.
"""
from __future__ import annotations

from pathlib import Path


def score(path: Path, max_stars: int = 150) -> dict | None:
    import cv2
    import numpy as np
    from astropy.io import fits
    try:
        raw = np.asarray(fits.getdata(path), dtype="float32")
    except (OSError, ValueError, TypeError):
        return None
    if raw.ndim == 3:
        lum = raw.mean(axis=0)
    else:
        h, w = (raw.shape[0] // 2) * 2, (raw.shape[1] // 2) * 2
        r = raw[:h, :w]
        lum = (r[0::2, 0::2] + r[0::2, 1::2] + r[1::2, 0::2] + r[1::2, 1::2]) / 4
    sat = float(lum.max()) * 0.95
    tophat = cv2.morphologyEx(lum, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    noise = 1.4826 * float(np.median(np.abs(tophat - np.median(tophat)))) or 1.0
    peaks = (tophat == cv2.dilate(tophat, np.ones((7, 7), np.uint8))) & (tophat > 6 * noise)
    ys, xs = np.nonzero(peaks)
    n = len(ys)
    if n < 5:
        return {"stars": n, "fwhm": 99.0, "round": 0.0}
    order = np.argsort(-tophat[ys, xs])
    fwhm, rnd = [], []
    yy, xx = np.mgrid[-4:5, -4:5]
    for k in order[: max_stars * 3]:
        y, x = ys[k], xs[k]
        if y < 4 or x < 4 or y >= lum.shape[0] - 4 or x >= lum.shape[1] - 4 or lum[y, x] >= sat:
            continue
        cut = tophat[y - 4:y + 5, x - 4:x + 5].clip(0)
        tot = cut.sum()
        if tot <= 0:
            continue
        cy, cx = (cut * yy).sum() / tot, (cut * xx).sum() / tot
        vy, vx = (cut * (yy - cy) ** 2).sum() / tot, (cut * (xx - cx) ** 2).sum() / tot
        cov = (cut * (yy - cy) * (xx - cx)).sum() / tot
        ev = np.linalg.eigvalsh([[vy, cov], [cov, vx]])
        if ev[0] <= 0:
            continue
        fwhm.append(2.355 * np.sqrt(ev.mean()) * 2)  # back to full-size pixels
        rnd.append(float(np.sqrt(ev[0] / ev[1])))
        if len(fwhm) >= max_stars:
            break
    if not fwhm:
        return {"stars": n, "fwhm": 99.0, "round": 0.0}
    return {"stars": n, "fwhm": float(np.median(fwhm)), "round": float(np.median(rnd))}


def screen_failed(passed: list[Path], failed: list[Path], progress=None, sample: int = 120) -> tuple[list[Path], str]:
    """Which rejected frames may join the stack. Returns (accepted, note)."""
    import random
    if not failed:
        return [], ""
    pool = sorted(passed)
    if len(pool) > sample:
        pool = random.Random(7).sample(pool, sample)
    total, done = len(pool) + len(failed), 0
    scores: dict[Path, dict] = {}
    for f in pool + list(failed):
        s = score(f)
        if s:
            scores[f] = s
        done += 1
        if progress:
            progress(done / total)
    good = [scores[f] for f in pool if f in scores]
    if len(good) < 5:
        return [], f"Left out all {len(failed)} frames the DWARF rejected (too few passed frames to compare them with)."
    most = max(s["stars"] for s in scores.values()) or 1
    w = lambda s: s["fwhm"] * most / max(1, s["stars"])

    def median(v):
        v = sorted(v)
        return v[len(v) // 2]
    bar = {"wfwhm": median([w(s) for s in good]), "stars": median([s["stars"] for s in good]),
           "round": max(0.8, 0.97 * median([s["round"] for s in good]))}  # as round as a typical passed frame
    ok = [f for f in failed if f in scores and w(scores[f]) < bar["wfwhm"] and scores[f]["stars"] >= bar["stars"]
          and scores[f]["round"] >= bar["round"]]
    cap = max(0, len(passed) // 10)
    ok = sorted(ok, key=lambda f: w(scores[f]))[:cap]
    note = (f"Let {len(ok)} of the {len(failed)} frames the DWARF rejected back in: only those sharper and with more stars than the median passed frame (wFWHM under {bar['wfwhm']:.2f}, ≥{bar['stars']} stars, "
            f"roundness ≥{bar['round']:.2f}), at most a tenth of the stack.")
    return ok, note

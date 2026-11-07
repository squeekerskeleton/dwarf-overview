"""SHO COMBINE: the Hubble palette from two Auto-apply runs of one target.

One run shot through the DWARF 3's Duo-Band (red pixels Hα, green and blue OIII) and one through an SII + OIII
dual-band filter held in front of the lens with the Astro filter selected (red pixels SII, green and blue OIII; see
filters.py). Each run's result_linear.fits (stacked, background-extracted, colour-calibrated, denoised, still linear)
gives the lines:

    Hα  = Duo-Band red          SII = SV220 red          OIII = both runs' green+blue, weighted by their noise

The SII run is aligned onto the Hα run by its stars (OpenCV, as the mosaic framing does), both are cut to the area
they share, and each line is scaled so its nebula signal sits near Hα's: SII is far fainter, and left as it is the
picture comes out green. Palettes saved:

- result_sho:  SII red, Hα green, OIII blue, with the green toned down (SCNR average neutral): the Hubble look;
- result_hoo_s: Hα red with SII added, OIII teal: closer to the eye's colours, with the SII structure in the red;
- result_sho_linear.fits: the aligned, scaled lines as linear RGB (S, H, O) for finishing by hand in Siril.

Nothing in either run's folder is changed; everything goes to a new SHO_<date> folder next to the Hα run.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from . import finishing


def _np():
    import numpy as np
    return np


def _read(path: Path):
    img, header = finishing.read_fits_rgb(path)
    np = _np()
    return np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0), header


def _view8(lum):
    """A stretched 8-bit view for feature matching (rows as stored)."""
    np = _np()
    v = lum[lum != 0]
    if v.size < 1000:
        return None
    bg, top = np.median(v), np.percentile(v, 99.8)
    return (np.clip((lum - bg) / max(1e-9, top - bg), 0, 1) ** 0.5 * 255).astype(np.uint8)


def align(src_lum, ref_lum) -> dict | None:
    """The similarity transform (rotation, scale, shift) that maps src onto ref, from their stars. None when the two
    don't share enough stars (different targets, or barely overlapping fields)."""
    import cv2
    np = _np()
    from .mosaic import _stars
    a, b = _view8(src_lum), _view8(ref_lum)
    if a is None or b is None:
        return None
    sift = cv2.SIFT_create(nfeatures=8000)
    s = min(1.0, 2000 / max(ref_lum.shape))
    k1, d1 = sift.detectAndCompute(cv2.resize(a, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), None)
    k2, d2 = sift.detectAndCompute(cv2.resize(b, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), None)
    if d1 is None or d2 is None or len(k1) < 20 or len(k2) < 20:
        return None
    good = [m for m, n in (p for p in cv2.BFMatcher().knnMatch(d1, d2, k=2) if len(p) == 2)
            if m.distance < 0.75 * n.distance]
    if len(good) < 20:
        return None
    p1 = np.float32([k1[g.queryIdx].pt for g in good]) / s
    p2 = np.float32([k2[g.trainIdx].pt for g in good]) / s
    A, inl = cv2.estimateAffinePartial2D(p1, p2, method=cv2.RANSAC, ransacReprojThreshold=4.0, maxIters=5000)
    if A is None or int(inl.sum()) < 15:
        return None
    # Refine on star centres: SIFT points sit on bright structure, star peaks pin it to a fraction of a pixel.
    st, rs = _stars(a, 2000), _stars(b, 4000)
    resid = None
    if len(st) >= 20 and len(rs) >= 20:
        proj = cv2.transform(st[None], A)[0].astype(np.float32)
        mm = cv2.BFMatcher(cv2.NORM_L2).match(proj, rs)
        dist = np.array([x.distance for x in mm])
        idx = np.array([x.trainIdx for x in mm])
        ok = dist < 3
        if ok.sum() >= 20:
            B, binl = cv2.estimateAffinePartial2D(st[ok], rs[idx[ok]], method=cv2.RANSAC, ransacReprojThreshold=1.5)
            if B is not None and int(binl.sum()) >= 15:
                A, inl = B, binl
                keep = binl.ravel() == 1
                resid = float(np.median(np.linalg.norm(cv2.transform(st[ok][keep][None], B)[0] - rs[idx[ok]][keep],
                                                       axis=1)))
    scale = float(np.sqrt(abs(np.linalg.det(A[:, :2]))))
    if not 0.5 < scale < 2:
        return None
    return {"A": np.asarray(A, dtype=np.float32), "pairs": int(np.asarray(inl).sum()), "scale": scale,
            "rotation_deg": float(np.degrees(np.arctan2(A[1][0], A[0][0]))), "residual_px": resid}


def _warp(img, A, shape):
    import cv2
    np = _np()
    h, w = shape
    out = np.stack([cv2.warpAffine(np.ascontiguousarray(img[..., c]), A, (w, h), flags=cv2.INTER_LANCZOS4,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0) for c in range(img.shape[2])], -1)
    valid = cv2.warpAffine(np.ones(img.shape[:2], np.uint8), A, (w, h), flags=cv2.INTER_NEAREST,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    out[~valid] = 0
    return out.astype(np.float32), valid


def _common_box(valid):
    """The largest centred-ish rectangle inside the shared area: rows and columns trimmed until nothing is empty."""
    np = _np()
    ys, xs = np.nonzero(valid)
    if not len(ys):
        raise ValueError("The two runs don't overlap.")
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    for _ in range(4000):
        box = valid[y0:y1, x0:x1]
        if box.all() or y1 - y0 < 50 or x1 - x0 < 50:
            break
        # Trim the edge with the most empty pixels.
        gaps = [(~box[0]).mean(), (~box[-1]).mean(), (~box[:, 0]).mean(), (~box[:, -1]).mean()]
        k = int(np.argmax(gaps))
        step = max(1, int(min(y1 - y0, x1 - x0) * 0.002))
        if k == 0:
            y0 += step
        elif k == 1:
            y1 -= step
        elif k == 2:
            x0 += step
        else:
            x1 -= step
    return int(y0), int(y1), int(x0), int(x1)


def _sky_sigma(c):
    np = _np()
    sub = c[::4, ::4]
    sub = sub[sub != 0]
    med = float(np.median(sub))
    mad = float(np.median(np.abs(sub - med))) * 1.4826
    return med, max(mad, 1e-7)


def _signal(c, sky):
    """Nebula brightness above the sky with stars taken out (a 9 px morphological opening removes the stars, a 5 px
    median the noise), then the 99th percentile."""
    import cv2
    np = _np()
    x = np.clip(c - sky, 0, None).astype("float32")
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    flat = cv2.medianBlur(cv2.morphologyEx(x, cv2.MORPH_OPEN, k), 5)
    return max(1e-7, float(np.percentile(flat, 99)))


def lines(ha_run, s_run) -> tuple:
    """(Hα, SII, OIII, info) from the two linear runs (H, W, 3), already aligned and cropped. Each line has its sky
    taken off; OIII is the noise-weighted mean of both runs' green+blue."""
    ha = ha_run[..., 0]
    s = s_run[..., 0]
    o1 = ha_run[..., 1:].mean(axis=2)
    o2 = s_run[..., 1:].mean(axis=2)
    out, info = [], {}
    for name, c in (("ha", ha), ("sii", s), ("oiii_duo", o1), ("oiii_sv", o2)):
        sky, sig = _sky_sigma(c)
        out.append(c - sky)
        info[name] = {"sky": sky, "noise": sig}
    ha, s, o1, o2 = out
    w1, w2 = 1 / info["oiii_duo"]["noise"] ** 2, 1 / info["oiii_sv"]["noise"] ** 2
    # The two OIII images come from different nights and filters: put the SV220's on the Duo-Band's scale first.
    k = _signal(o1, 0) / _signal(o2, 0)
    o = (w1 * o1 + w2 * (o2 * k)) / (w1 + w2)
    info["oiii_weights"] = [round(w1 / (w1 + w2), 3), round(w2 / (w1 + w2), 3)]
    info["oiii_scale_sv"] = round(k, 4)
    return ha, s, o, info


def _stretch_rgb(lin):
    """Linear (sky at 0) -> display with the app's colour-keeping stretch."""
    np = _np()
    return finishing.base_stretch(np.clip(lin, -0.05, None) + 0.01, stretch=100.0, black_sigma=2.5, sky=0.09)


def _scnr(img, amount: float = 0.8):
    """Average-neutral green removal, as Siril's rmgreen: green no higher than the mean of red and blue."""
    np = _np()
    g = img[..., 1]
    cap = (img[..., 0] + img[..., 2]) / 2
    img = img.copy()
    img[..., 1] = g - amount * np.clip(g - cap, 0, None)
    return img


def combine(ha_folder: str, s_folder: str, out_folder: str | None = None, sii_boost: float = 1.0) -> dict:
    """Make the SHO and HOO+S versions from an Hα+OIII run folder and an SII+OIII run folder."""
    np = _np()
    ha_p, s_p = Path(ha_folder) / "result_linear.fits", Path(s_folder) / "result_linear.fits"
    for p in (ha_p, s_p):
        if not p.exists():
            raise ValueError(f"{p} isn't there: finish an Auto-apply run in that folder first.")
    ha_img, header = _read(ha_p)
    s_img, _ = _read(s_p)
    t = align(s_img.mean(axis=2), ha_img.mean(axis=2))
    if t is None:
        raise ValueError("The two runs couldn't be aligned on their stars (different targets, or fields that "
                         "barely overlap).")
    s_img, valid = _warp(s_img, t["A"], ha_img.shape[:2])
    valid &= (ha_img != 0).all(axis=2)
    y0, y1, x0, x1 = _common_box(valid)
    share = (y1 - y0) * (x1 - x0) / (ha_img.shape[0] * ha_img.shape[1])
    if share < 0.2:
        raise ValueError(f"The two runs share only {share * 100:.0f}% of the Hα run's field: too little to combine.")
    ha_img, s_img = ha_img[y0:y1, x0:x1], s_img[y0:y1, x0:x1]
    ha, s, o, info = lines(ha_img, s_img)
    sig_h, sig_s, sig_o = _signal(ha, 0), _signal(s, 0), _signal(o, 0)
    # SII to about 70% of Hα, OIII to about Hα's level: the usual starting balance for the Hubble palette. Capped,
    # so a run with almost no SII isn't amplified into pure noise.
    ks = min(6.0, 0.7 * sig_h / sig_s) * sii_boost
    ko = min(4.0, max(0.5, sig_h / sig_o))
    S, H, O = s * ks, ha, o * ko
    out = Path(out_folder) if out_folder else Path(ha_folder).parent / time.strftime("SHO_%Y-%m-%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    lin = np.stack([S, H, O], axis=-1).astype(np.float32)
    finishing.write_fits_rgb(out / "result_sho_linear.fits", lin, header)
    # FITS rows run bottom-up: flip once for the display images (as Auto-apply does).
    sho = _scnr(_stretch_rgb(np.flipud(lin)))
    sho, _ = finishing.quality_gate(sho)
    saved = finishing.save_final(sho, out / "result_sho", header)
    # HOO with SII folded into the red: Hα + 0.5·SII red, OIII green and blue.
    hoo_s = np.stack([H + 0.5 * S, O, O], axis=-1).astype(np.float32)
    hoo_s, _ = finishing.quality_gate(_stretch_rgb(np.flipud(hoo_s)))
    saved2 = finishing.save_final(hoo_s, out / "result_hoo_s", header)
    try:
        metrics = finishing.run_metrics(sho)
    except Exception:
        metrics = {}
    res = {"folder": str(out), "final": saved["tif"], "files": saved, "hoo_s": saved2["tif"],
           "linear": str(out / "result_sho_linear.fits"),
           "aligned": {"star_pairs": t["pairs"], "rotation_deg": round(t["rotation_deg"], 2),
                       "scale": round(t["scale"], 4), "residual_px": t["residual_px"]},
           "shared_field": round(share, 3), "scales": {"sii": round(ks, 3), "oiii": round(ko, 3)},
           "signal": {"ha": sig_h, "sii": sig_s, "oiii": sig_o}, "lines": info, "metrics": metrics,
           "sources": {"ha_oiii": str(ha_folder), "sii_oiii": str(s_folder)}}
    (out / "sho.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    res["note"] = (f"Aligned on {t['pairs']} stars (rotation {t['rotation_deg']:.1f}°, scale {t['scale']:.3f}); "
                   f"{share * 100:.0f}% of the field shared. SII ×{ks:.2f} and OIII ×{ko:.2f} to balance Hα "
                   f"(SII was {sig_s / sig_h * 100:.0f}% of Hα). OIII from both runs, weighted "
                   f"{info['oiii_weights'][0]:.0%} Duo-Band / {info['oiii_weights'][1]:.0%} SII+OIII.")
    return res


def candidates() -> list[dict]:
    """Finished Auto-apply runs that have a linear result, newest first, with the filter each was shot through."""
    from . import autoprocess
    roots = [autoprocess.work_root()]
    fr = autoprocess.finished_root()
    if fr:
        roots.append(fr)
    seen, out = set(), []
    for root in roots:
        if not root.is_dir():
            continue
        for lin in root.glob("*/result_linear.fits"):
            d = lin.parent
            if d.name in seen:
                continue
            seen.add(d.name)
            meta = {}
            try:
                meta = json.loads((d / autoprocess.RUN_FILE).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
            flt = meta.get("filter") or ""
            if not flt:
                from . import filters
                flt = filters.for_session(d.name) or ("duo" if any(w in d.name.lower() for w in ("duo", "dual")) else "")
            out.append({"folder": str(d), "name": d.name, "filter": flt, "modified": lin.stat().st_mtime})
    out.sort(key=lambda r: -r["modified"])
    return out

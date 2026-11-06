"""DWARF mosaics: two or more panels shot one after the other, each in its own subfolder of the session.

Worked out on Kip's 2-panel Veil (2026-11-04, 2x130 x 60 s): stacking every frame together fails, as one 60 s
frame holds only about 50 stars and the panels barely overlap. What worked, and beat the DWARF's own mosaic
(filaments 48 sigma against 7.6, sky noise 5x lower, about 7x the stars, no seam):

1. each panel stacked on its own (32-bit, a pedestal dark, CFA equalised, the whole dithered field kept);
2. each panel's background taken out by GraXpert (smoothing 0.8: 0.4 left a dark moat round the nebula), with
   the empty border filled for GraXpert, which turns zeros into NaN, and set back to 0 (no data) afterwards;
3. the panels plate solved and joined by Siril (overlap normalisation, 200 px feather);
4. the result put into the DWARF's own mosaic framing (its stacked JPG), found by SIFT and refined on stars;
5. then the usual finish, with the layered star treatment tuned for a wide broadband field.

autoprocess.py runs these as its frame and stacking steps; the rest of the run is the same as for one field.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

# The DWARF 3's telephoto as measured by plate solving (the nominal 150 mm is 1% long).
FOCAL_MM = 148.46
PIXEL_UM = 2.0
# Panel stacks: the whole dithered field (framing=max), the sharpest 95% of frames, faint stars found too
# (sigma 0.4), and 8 star pairs enough to register: the stricter defaults dropped most 60 s frames. A stricter
# 75% cut on the Veil's upper panel made it 5% noisier with no sharper stars.
PANEL_KEEP = 95
# Disk per raw frame while one panel stacks: 32-bit converted, calibrated (RGB) and registered (on a larger canvas).
SPACE_FACTOR = 15.0
BGE_SMOOTHING = 0.8
# Layered finish for a wide broadband field: the stars as they are (no extra gamma or scaling) and a little less
# coloured; the C 33 values dimmed the Veil's dense Cygnus star field too much.
LAYERED = dict(star_gamma=1.0, star_floor=0.015, star_sat=0.7, star_scale=1.0)


def _np():
    import numpy
    return numpy


def _shot(folder: Path) -> dict:
    try:
        return json.loads((folder / "shotsInfo.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def session_root(folder: Path) -> Path:
    """A photo inside one panel's subfolder still belongs to the whole mosaic one level up."""
    if not (_shot(folder).get("mosaicInfo")) and (_shot(folder.parent).get("mosaicInfo")):
        return folder.parent
    return folder


def panels(folder: Path, shot: dict | None = None) -> list[dict]:
    """The panels of a mosaic session (empty for a single field): id, grid position, subfolder name, tag."""
    info = (shot if shot is not None else _shot(folder)).get("mosaicInfo") or {}
    subs = info.get("subviewInfo") or []
    if len(subs) < 2:
        return []
    here = [d.name for d in folder.iterdir() if d.is_dir()] if folder.is_dir() else []
    out = []
    for s in subs:
        pid = int(s.get("id") or len(out) + 1)
        sub = Path(str(s.get("path") or "").replace("\\", "/").rstrip("/")).name
        if sub not in here:  # a library copy may have renamed it: match the panel number in brackets
            sub = next((h for h in here if f"({pid})" in h), sub)
        out.append({"id": pid, "coord": s.get("coord"), "sub": sub, "tag": f"panel{pid}"})
    return sorted(out, key=lambda p: p["id"])


def reference_image(folder: Path, photo: str | None = None) -> str | None:
    """The DWARF's own mosaic image (its framing is what the result keeps)."""
    cands = []
    if folder.is_dir():
        cands = [f for f in folder.iterdir() if f.is_file() and re.match(r"stacked.*\.(jpe?g|png)$", f.name, re.I)]
    if photo and re.search(r"\.(jpe?g|png)$", photo, re.I) and Path(photo).parent == folder:
        cands.append(Path(photo))
    cands = [c for c in cands if c.exists()]
    return str(max(cands, key=lambda c: c.stat().st_size)) if cands else None


def centre(shot: dict, frame: Path | None) -> dict:
    """The mosaic's centre in degrees, from shotsInfo or else a frame's header (the DWARF writes the mosaic centre)."""
    from .platesolve import hint_from_shot_info
    hint = hint_from_shot_info(shot)
    if "ra" in hint:
        return hint
    try:
        from astropy.io import fits
        h = fits.getheader(frame)
        ra, dec = float(h["RA"]), float(h["DEC"])
        if 0 <= ra < 360 and -90 <= dec <= 90:
            return {"ra": ra, "dec": dec}
    except Exception:
        pass
    return {}


def panel_script(tune: list[str], n: int, dark: bool) -> tuple[list[str], list[str]]:
    """Siril commands for one panel, run in the panel's folder (frames in lights/, the dark two levels up)."""
    seq = "pp_p" if dark else "p"
    prep = (["convert p -out=../process", "cd ../process",
             "calibrate p -dark=../../master_dark.fits -cc=dark -cfa -equalize_cfa -debayer"] if dark else
            ["convert p -debayer -out=../process", "cd ../process"])
    lines = ["# Written by DWARF Post-Processing Hub: one mosaic panel", "requires 1.4.0", *tune, "setext fits",
             "set32bits", "cd lights", *prep,
             f"setref {seq} {max(1, (n + 1) // 2)}",
             "setfindstar -sigma=0.4",
             f"register {seq} -2pass -minpairs=8",
             f"seqapplyreg {seq} -framing=max -interp=lanczos4 -filter-wfwhm={PANEL_KEEP}%",
             # Winsorized 3/3 as in the Veil test; -maximize keeps every pixel any frame covered.
             f"stack r_{seq} rej w 3 3 -norm=addscale -weight=wfwhm -maximize -32b -out=../stack", ""]
    phases = ["convert", "calibrate", "register", "seqapplyreg", "stack"] if dark else \
        ["convert", "register", "seqapplyreg", "stack"]
    return lines, phases


def assembly_script(tune: list[str], hint: dict) -> list[str]:
    coords = f"{hint['ra']:.5f},{hint['dec']:.5f} " if "ra" in hint else ""
    return ["# Written by DWARF Post-Processing Hub: join the mosaic panels", "requires 1.4.0", *tune,
            "setext fits", "set32bits", "convert m_",
            f"seqplatesolve m_ {coords}-focal={FOCAL_MM} -pixelsize={PIXEL_UM} -radius=5 -force",
            "seqapplyreg m_ -framing=max -interp=lanczos4",
            "stack r_m_ rej n -norm=addscale -overlap_norm -maximize -feather=200 -32b -out=../result_mosaic", ""]


def coverage(d):
    """Pixels a panel really covers: isolated zeros inside don't count, and 15 px of the thin dithered edge go."""
    import cv2
    np = _np()
    raw = (d != 0).all(axis=0)
    cover = cv2.blur(raw.astype(np.float32), (61, 61)) > 0.5
    cover = cv2.erode(cover.astype(np.uint8), np.ones((41, 41), np.uint8)).astype(bool)
    return raw, cover


def bge_panel(src: Path, dst: Path, work: Path, run_gx) -> str:
    """GraXpert background extraction of one linear panel; run_gx(input, stem) returns GraXpert's output or None.
    Returns a note. Without a result the panel goes on as it is (border still 0)."""
    from astropy.io import fits
    np = _np()
    d, h = fits.getdata(src, header=True)
    d = np.nan_to_num(np.asarray(d, dtype=np.float32))
    if d.ndim == 2:
        d = d[None]
    raw, cover = coverage(d)
    bg = np.median(d[:, cover], axis=1)
    filled = d.copy()
    for c in range(d.shape[0]):
        filled[c][~(cover & raw)] = bg[c]  # GraXpert returns NaN on zeros
    tmp = work / f"{src.stem}_filled.fits"
    fits.PrimaryHDU(filled, h).writeto(tmp, overwrite=True)
    out = run_gx(tmp, f"{src.stem}_bge")
    tmp.unlink(missing_ok=True)
    if not out:
        b = d.copy()
        b[:, ~cover] = 0
        fits.PrimaryHDU(b, h).writeto(dst, overwrite=True)
        return "background left in (GraXpert didn't run)"
    b = np.nan_to_num(np.asarray(fits.getdata(out), dtype=np.float32)).reshape(d.shape) + 0.002  # 0 = no data
    b[:, ~cover] = 0
    fits.PrimaryHDU(b, h).writeto(dst, overwrite=True)
    Path(out).unlink(missing_ok=True)
    return f"background taken out (smoothing {BGE_SMOOTHING})"


def _imread(path: str):
    import cv2
    np = _np()
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)


def _stars(img, n: int = 1500):
    import cv2
    np = _np()
    s = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 2)
    mx = cv2.dilate(s, np.ones((9, 9), np.uint8))
    ys, xs = np.nonzero((s == mx) & (s > np.percentile(s, 99.5)))
    order = np.argsort(-s[ys, xs])[:n]
    return np.stack([xs[order], ys[order]], 1).astype(np.float32)


def match_framing(d, ref_path: str) -> dict | None:
    """The affine that maps our mosaic (3xHxW FITS rows, bottom-up) onto the DWARF's mosaic JPG, or None when the
    two can't be matched reliably. Keeps our pixel scale: if the DWARF's image is smaller or larger, the frame is
    scaled to ours instead of resampling the data."""
    import cv2
    np = _np()
    ref = _imread(ref_path)
    if ref is None:
        return None
    L = np.nan_to_num(d.mean(axis=0))[::-1]  # top-down, as the JPG
    v = L != 0
    if v.sum() < 1000:
        return None
    bg, top = np.median(L[v]), np.percentile(L[v], 99.8)
    ours = (np.clip((L - bg) / max(1e-9, top - bg), 0, 1) ** 0.5 * 255).astype(np.uint8)
    sift = cv2.SIFT_create(nfeatures=8000)
    s = 1600 / max(ref.shape)
    k1, d1 = sift.detectAndCompute(cv2.resize(ours, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), None)
    k2, d2 = sift.detectAndCompute(cv2.resize(ref, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), None)
    if d1 is None or d2 is None or len(k1) < 20 or len(k2) < 20:
        return None
    good = [a for a, b in (m for m in cv2.BFMatcher().knnMatch(d1, d2, k=2) if len(m) == 2)
            if a.distance < 0.75 * b.distance]
    if len(good) < 20:
        return None
    p1 = np.float32([k1[g.queryIdx].pt for g in good]) / s
    p2 = np.float32([k2[g.trainIdx].pt for g in good]) / s
    A, inl = cv2.estimateAffinePartial2D(p1, p2, method=cv2.RANSAC, ransacReprojThreshold=4.0, maxIters=5000)
    if A is None or int(inl.sum()) < 15:
        return None
    st, rs = _stars(ours), _stars(ref, 4000)
    proj = cv2.transform(st[None], A)[0].astype(np.float32)
    mm = cv2.BFMatcher(cv2.NORM_L2).match(proj, rs)
    dist = np.array([x.distance for x in mm])
    idx = np.array([x.trainIdx for x in mm])
    ok = dist < 3
    A2, inl2, resid = A, inl, None
    if ok.sum() >= 20:
        B, binl = cv2.estimateAffinePartial2D(st[ok], rs[idx[ok]], method=cv2.RANSAC, ransacReprojThreshold=1.5)
        if B is not None and int(binl.sum()) >= 15:
            A2, inl2 = B, binl
            keep = binl.ravel() == 1
            resid = float(np.median(np.linalg.norm(cv2.transform(st[ok][keep][None], B)[0] - rs[idx[ok]][keep], axis=1)))
    scale = float(np.sqrt(abs(np.linalg.det(A2[:, :2]))))
    H, W = ref.shape
    if not 0.25 < scale < 4:
        return None
    if abs(scale - 1) > 0.02:  # a different size: keep our resolution and scale the DWARF's frame instead
        A2 = A2 / scale
        W, H = int(round(W / scale)), int(round(H / scale))
    cov = float(cv2.warpAffine(v.astype(np.uint8), A2, (W, H), flags=cv2.INTER_NEAREST).mean())
    return {"A": np.asarray(A2).tolist(), "W": W, "H": H, "coverage": cov, "scale": scale,
            "rotation_deg": float(np.degrees(np.arctan2(A2[1][0], A2[0][0]))), "matches": len(good),
            "star_pairs": int(np.asarray(inl2).sum()), "residual_px": resid}


def warp(d, F: dict):
    """Our mosaic in the matched frame (FITS layout kept: channels first, rows bottom-up)."""
    import cv2
    np = _np()
    A = np.array(F["A"], np.float32)
    res = [cv2.warpAffine(np.ascontiguousarray(c[::-1]), A, (F["W"], F["H"]), flags=cv2.INTER_LANCZOS4,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)[::-1] for c in d]
    r = np.stack(res).astype(np.float32)
    r[:, ~(r != 0).all(axis=0)] = 0  # Lanczos rings at the data's edge: keep "no data" exact
    return r


def crop_to_cover(d):
    """No DWARF image to match: the bounding box of the area the panels cover."""
    np = _np()
    v = (d != 0).all(axis=0)
    ys, xs = np.nonzero(v)
    if not len(ys):
        return d
    return d[:, ys.min():ys.max() + 1, xs.min():xs.max() + 1]


WCS_KEYS = ("CRPIX1", "CRPIX2", "CD1_1", "CD1_2", "CD2_1", "CD2_2", "CRVAL1", "CRVAL2", "CTYPE1", "CTYPE2",
            "PC1_1", "PC1_2", "PC2_1", "PC2_2", "CDELT1", "CDELT2", "CUNIT1", "CUNIT2", "EQUINOX", "LONPOLE",
            "LATPOLE", "RADESYS", "A_ORDER", "B_ORDER", "AP_ORDER", "BP_ORDER")


def frame_to_dwarf(src: Path, dst: Path, ref: str | None) -> str:
    """Put the joined mosaic into the DWARF's own framing (or crop it to its covered area). Returns a note."""
    from astropy.io import fits
    np = _np()
    d, h = fits.getdata(src, header=True)
    d = np.nan_to_num(np.asarray(d, dtype=np.float32))
    if d.ndim == 2:
        d = d[None]
    F = None
    if ref:
        try:
            F = match_framing(d, ref)
        except Exception:  # matching is a nicety; the covered area is a fine fallback
            F = None
    if F and F["coverage"] > 0.8:
        out = warp(d, F)
        note = (f"framed like the DWARF's own mosaic ({F['W']}x{F['H']}, matched on {F['star_pairs']} stars"
                + (f" to {F['residual_px']:.2f} px" if F.get("residual_px") is not None else "")
                + (f"; {100 - F['coverage'] * 100:.1f}% of it had no data and is filled" if F["coverage"] < 0.999 else "")
                + ")")
    else:
        out = crop_to_cover(d)
        note = ("cropped to the area the panels cover" +
                (" (it couldn't be matched to the DWARF's own mosaic image)" if ref else ""))
    for k in WCS_KEYS:
        h.remove(k, ignore_missing=True, remove_all=True)  # the old solution no longer fits; step 5 solves again
    for k in ("BZERO", "BSCALE"):
        h.remove(k, ignore_missing=True)
    fits.PrimaryHDU(out, h).writeto(dst, overwrite=True)
    return note


def destripe(img, starless):
    """Faint row and column banding (stretched RGB, top-down): per-line sky medians, nebula and stars masked by the
    starless layer, high-passed so only the banding goes. Returns (image, banding rms in green)."""
    import cv2
    np = _np()
    m = cv2.GaussianBlur(starless.mean(axis=2), (0, 0), 8)
    sky = m < np.percentile(m, 70)
    out = img.copy()
    rms = 0.0

    def lines(ch, mask):
        v = np.array([np.median(ch[i][mask[i]]) if mask[i].sum() > 200 else np.nan for i in range(ch.shape[0])])
        v = np.where(np.isnan(v), np.nanmedian(v), v).astype(np.float32)
        return v - cv2.GaussianBlur(v.reshape(-1, 1), (0, 0), 40).ravel()

    for c in range(3):
        ch = starless[..., c]
        rows, cols = lines(ch, sky), lines(ch.T, sky.T)
        out[..., c] = img[..., c] - rows[:, None] - cols[None, :]
        if c == 1:
            rms = float(np.std(rows))
    return np.clip(out, 0, 1).astype("float32"), rms

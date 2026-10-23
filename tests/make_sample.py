"""Write synthetic DWARF-like stacks for testing: linear FITS with a gradient, green cast and stars."""
import json, sys
from pathlib import Path
import numpy as np
from astropy.io import fits

def make(out_dir, elong=1.0, gradient=0.6, w=3840, h=2160, seed=1):
    rng = np.random.default_rng(seed)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    bg = 0.02 * (1 + gradient * (xx / w) * (yy / h))
    img = np.stack([bg * 0.9, bg * 1.15, bg * 0.85])
    sx = 1.6; sy = sx * elong
    for _ in range(1500):
        x, y = rng.uniform(20, w - 20), rng.uniform(20, h - 20)
        amp = rng.pareto(1.5) * 0.02 + 0.01
        x0, x1, y0, y1 = int(x) - 10, int(x) + 11, int(y) - 10, int(y) + 11
        g = amp * np.exp(-(((xx[y0:y1, x0:x1] - x) / sx) ** 2 + ((yy[y0:y1, x0:x1] - y) / sy) ** 2) / 2)
        img[:, y0:y1, x0:x1] += g
    img += rng.normal(0, 0.002, img.shape).astype(np.float32)
    img = np.clip(img, 0, 1)
    fits.PrimaryHDU((img * 65535).astype(np.uint16)).writeto(out / "stacked-16_M31_10s.fits", overwrite=True)
    (out / "shotsInfo.json").write_text(json.dumps({"target": "M31", "exp": "10", "gain": "80", "ir": "VIS", "shotsStacked": 180}))
    return out / "stacked-16_M31_10s.fits"

if __name__ == "__main__":
    print(make(sys.argv[1], elong=float(sys.argv[2]) if len(sys.argv) > 2 else 1.0))


def make_moon(out_dir, blur=1.0, n_frames=0, w=1920, h=1080, seed=2):
    """A cratered half-lit disk as a PNG stack, plus an optional SER video with varying seeing."""
    import struct
    from PIL import Image as _Image
    rng = np.random.default_rng(seed)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy, r = w / 2, h / 2, h * 0.42
    d = np.hypot(xx - cx, yy - cy)
    disk = (d < r) & (xx > cx - 0.2 * r)
    tex = np.zeros((h, w), np.float32)
    for _ in range(400):
        x, y, rad = rng.uniform(cx - r, cx + r), rng.uniform(cy - r, cy + r), rng.uniform(3, 30)
        dd = np.hypot(xx - x, yy - y)
        tex += 0.15 * np.exp(-((dd - rad) / 1.5) ** 2) - 0.08 * (dd < rad)
    base = np.where(disk, 0.55 + tex, 0.01).astype(np.float32)

    def blurred(img, sigma):
        k = int(3 * sigma) + 1
        g = np.exp(-(np.arange(-k, k + 1) ** 2) / (2 * sigma ** 2)); g /= g.sum()
        img = np.apply_along_axis(lambda m: np.convolve(m, g, "same"), 0, img)
        return np.apply_along_axis(lambda m: np.convolve(m, g, "same"), 1, img)

    stack = np.clip(blurred(base, blur) + rng.normal(0, 0.003, base.shape), 0, 1)
    _Image.fromarray((np.repeat(stack[:, :, None], 3, 2) * 255).astype(np.uint8)).save(out / "moon_stack.png")
    if n_frames:
        small = base[::2, ::2]
        with open(out / "moon.ser", "wb") as f:
            hh, ww = small.shape
            f.write(b"LUCAM-RECORDER" + struct.pack("<7i", 0, 0, 1, ww, hh, 8, n_frames) + b"\0" * 136)
            for i in range(n_frames):
                sig = 0.6 + 2.5 * abs(np.sin(i * 0.37))
                fr = np.clip(blurred(small, sig) + rng.normal(0, 0.01, small.shape), 0, 1)
                f.write((fr * 255).astype(np.uint8).tobytes())
    return out

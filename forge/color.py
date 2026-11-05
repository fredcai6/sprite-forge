"""OKLab conversion and the fixed palette a character is drawn in."""
import json
from pathlib import Path

import numpy as np


def srgb_to_oklab(rgb):
    c = np.asarray(rgb, dtype=np.float64) / 255.0
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    l = np.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b)
    m = np.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b)
    s = np.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)
    return np.stack(
        [
            0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
            1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
            0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s,
        ],
        axis=-1,
    )


def nearest(lab, palette_lab):
    """Index of the closest palette colour for each pixel. Chroma counts double, so a dull
    grey never absorbs a teal or red accent just because they are close in lightness."""
    w = np.array([1.0, 2.0, 2.0])
    d = (((lab[..., None, :] - palette_lab) * w) ** 2).sum(-1)
    return d.argmin(-1)


def kmeans(points, k, rng, iters=30):
    centres = points[rng.choice(len(points), k, replace=False)]
    for _ in range(iters):
        lab = ((points[:, None] - centres) ** 2).sum(-1).argmin(1)
        centres = np.stack(
            [points[lab == i].mean(0) if (lab == i).any() else centres[i] for i in range(k)]
        )
    return centres, lab


def build_palette(rgb_pixels, families, seed=0):
    """Pick colours family by family, so a colour covering 2% of the art keeps its own slots
    however much green surrounds it. families: {name: {hue: [lo, hi], min_chroma, n}}."""
    rng = np.random.default_rng(seed)
    if len(rgb_pixels) > 400_000:
        rgb_pixels = rgb_pixels[rng.choice(len(rgb_pixels), 400_000, replace=False)]
    lab = srgb_to_oklab(rgb_pixels)
    chroma = np.hypot(lab[:, 1], lab[:, 2])
    hue = np.degrees(np.arctan2(lab[:, 2], lab[:, 1])) % 360
    out = {}
    for name, f in families.items():
        if "hue" in f:
            lo, hi = f["hue"]
            in_hue = (hue >= lo) & (hue < hi) if lo < hi else (hue >= lo) | (hue < hi)
            sel = in_hue & (chroma >= f.get("min_chroma", 0.04))
        else:
            sel = chroma < f.get("max_chroma", 0.04)
        pts = rgb_pixels[sel].astype(np.float64)
        centres, _ = kmeans(pts, f["n"], rng)
        centres = centres[np.argsort(srgb_to_oklab(centres)[:, 0])]
        out[name] = [[int(round(v)) for v in c] for c in centres]
    return out


def load_palette(path):
    fam = json.loads(Path(path).read_text())
    rgb = np.array([c for cs in fam.values() for c in cs], dtype=np.uint8)
    return rgb, srgb_to_oklab(rgb)

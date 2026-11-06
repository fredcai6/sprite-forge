"""Shared steps for background layers: key out magenta, quantize to a small palette on the stage grid."""
import json
import numpy as np
from PIL import Image
from scipy import ndimage

from .color import kmeans, nearest, srgb_to_oklab
from .generate import DATA, rev


def load(run, model, shot, seed):
    p = DATA / "runs" / run / model / shot / f"{seed}.png"
    return Image.open(p).convert("RGB"), {"run": run, "model": model, "shot": shot, "seed": seed}


def key_alpha(rgb):
    """True where opaque. Magenta touching the border goes, enclosed near-pure magenta goes, and
    magenta-tinted fringe pixels are eaten by one erosion."""
    r, g, b = [rgb[..., i].astype(int) for i in range(3)]
    loose = (np.minimum(r, b) - g > 38) & (r + b > 150)
    lab, _ = ndimage.label(loose)
    edge = np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    bg = np.isin(lab, edge[edge > 0]) | ((np.minimum(r, b) - g > 75) & (r > 150) & (b > 85))
    fringe = ndimage.binary_dilation(bg, iterations=2) & (np.minimum(r, b) - g > 22)
    bg |= fringe
    lab3 = srgb_to_oklab(rgb)
    chroma = np.hypot(lab3[..., 1], lab3[..., 2])
    hue = np.degrees(np.arctan2(lab3[..., 2], lab3[..., 1])) % 360
    bg |= (hue > 300) & (hue < 352) & (chroma > 0.12)
    keep = ~bg
    return ndimage.binary_opening(keep, iterations=1)


def quantize(rgba, n=48, seed=0, shade=None):
    """Snap to an n-colour palette (k-means in OKLab, chroma counted double) with hard alpha."""
    rgb, a = rgba[..., :3], rgba[..., 3] > 127
    px = rgb[a]
    if len(px) < n * 4:   # nothing (or next to nothing) to quantize: return an empty layer
        out = np.zeros_like(rgba); return out
    rng = np.random.default_rng(seed)
    if len(px) > 120_000:
        px = px[rng.choice(len(px), 120_000, replace=False)]
    lab = srgb_to_oklab(px)
    w = np.array([1.0, 2.0, 2.0])
    centres, _ = kmeans(lab * w, n, rng, iters=12)
    centres = centres / w
    out = np.zeros_like(rgba)
    flat = srgb_to_oklab(rgb.reshape(-1, 3))
    idx = np.empty(len(flat), int)
    for i in range(0, len(flat), 200_000):
        d = (((flat[i:i + 200_000, None, :] - centres) * w) ** 2).sum(-1)
        idx[i:i + 200_000] = d.argmin(-1)
    # centre colours back to sRGB: use the mean source colour of each cluster
    cols = np.zeros((n, 3))
    allrgb = rgb.reshape(-1, 3).astype(float)
    for k in range(n):
        sel = (idx == k) & a.reshape(-1)   # opaque pixels only: the RGB under transparent ones is key colour
        cols[k] = allrgb[sel].mean(0) if sel.any() else 0
    out[..., :3] = cols[idx].reshape(rgb.shape).round().astype(np.uint8)
    out[..., 3] = np.where(a, 255, 0)
    return out


def tile_seam(img, width, blend=24):
    """Cut a strip of `width` from the overhanging image where its right edge best continues its left
    edge (a gap between buildings, say), then crossfade only a few columns so nothing ghosts."""
    h, w = img.shape[:2]
    f = img.astype(float)
    best = min(range(0, w - width - blend + 1),
               key=lambda s: np.abs(f[:, s + width - 3:s + width] - f[:, s:s + 3]).mean())
    out = f[:, best:best + width].copy()
    over = f[:, best + width:best + width + blend]
    a = np.linspace(0, 1, blend)[None, :, None]
    left = out[:, :blend]
    if img.shape[2] == 4:
        al, ao = left[..., 3:] / 255, over[..., 3:] / 255
        den = (1 - a) * ao + a * al
        num = (1 - a) * over[..., :3] * ao + a * left[..., :3] * al
        out[:, :blend, :3] = np.where(den > 0, num / np.maximum(den, 1e-6), 0)
        out[:, :blend, 3] = 255 * (den[..., 0] > 0.5)
    else:
        out[:, :blend] = (1 - a) * over + a * left
    tile_seam.last = best
    return out.round().astype(np.uint8)


def write_sources(path, sources):
    path.write_text(json.dumps({"rev": rev(), "layers": sources}, indent=1))


def pink_to_brick(rgba, hue_to=25.0, red_rows=None):
    """Key-colour spill that survives as hot pink (hue 295-12) becomes a brick-salmon of the same
    lightness, so no neon magenta reaches a layer."""
    rgb = rgba[..., :3]
    lab = srgb_to_oklab(rgb)
    chroma = np.hypot(lab[..., 1], lab[..., 2])
    hue = np.degrees(np.arctan2(lab[..., 2], lab[..., 1])) % 360
    sel = (hue > 295) & (chroma > 0.08) & (rgba[..., 3] > 0)
    if red_rows is not None:  # hot pink that reads as hue ~0-12: only inside this row band (cars are legitimately red)
        band = np.zeros_like(sel)
        band[red_rows[0]:red_rows[1]] = True
        sel |= (hue < 12) & (chroma > 0.08) & band & (rgba[..., 3] > 0)
    if not sel.any():
        return rgba
    c = chroma[sel] * 0.7
    L = lab[..., 0][sel]
    a, b = c * np.cos(np.radians(hue_to)), c * np.sin(np.radians(hue_to))
    # OKLab -> linear sRGB -> sRGB
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b
    l, m, s = l_ ** 3, m_ ** 3, s_ ** 3
    lin = np.stack([4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
                    -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
                    -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s], -1)
    lin = np.clip(lin, 0, 1)
    out = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055)
    res = rgba.copy()
    res[sel, :3] = np.round(out * 255).astype(np.uint8)
    return res


def soften(rgba, contrast=1.0, lift=0.0, haze=(0, 0, 0), haze_amt=0.0, median=0):
    """Push a layer back: less contrast around mid-grey, lifted blacks, a wash of atmosphere colour, and
    optionally a median filter to drop fine cracks and speckle. Alpha is untouched."""
    from scipy import ndimage as ndi
    rgb = rgba[..., :3].astype(float)
    if median:
        rgb = np.stack([ndi.median_filter(rgb[..., i], size=median) for i in range(3)], -1)
    rgb = 140 + (rgb - 140) * contrast + lift
    rgb = rgb * (1 - haze_amt) + np.array(haze, float) * haze_amt
    out = rgba.copy()
    out[..., :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    return out

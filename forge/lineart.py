"""Deterministic line art from a finished sprite frame: black lines on white, no model involved.

Lines go where the sprite has an edge: the silhouette, every step between two clearly different palette regions, and
the dark outline pixels themselves. Used by the rig-paint track, where a code rig's frame fixes the pose and a model
only paints it.
"""
import numpy as np
from PIL import Image

from .color import srgb_to_oklab


def edges(sprite, thr=0.21, min_run=12):
    """Boolean map at sprite resolution: True where a line belongs."""
    a = sprite[..., 3] > 0
    lab = srgb_to_oklab(sprite[..., :3])
    line = np.zeros(a.shape, bool)
    for dy, dx in ((0, 1), (1, 0)):
        d = np.zeros(a.shape, bool)
        diff = np.linalg.norm(lab[dy:, dx:] - lab[:a.shape[0] - dy, :a.shape[1] - dx], axis=-1) > thr
        both = a[dy:, dx:] & a[:a.shape[0] - dy, :a.shape[1] - dx]
        d[dy:, dx:] = diff & both
        line |= d
    pad = np.pad(a, 1)
    inner = pad[1:-1, 1:-1] & ~(pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:])   # silhouette pixels
    # drop pixel-art texture: interior line fragments shorter than min_run pixels
    from scipy import ndimage
    lb, n = ndimage.label(line & ~inner, structure=np.ones((3, 3)))
    if n:
        sizes = ndimage.sum(line & ~inner, lb, range(1, n + 1))
        line = np.isin(lb, 1 + np.flatnonzero(sizes >= min_run))
    return (line | inner) & a


def layout(frames, size=1024):
    """One whole-number scale k and canvas origin (x0, y0) in the image for a whole animation: the union of the figures' extents
    fills 90% of the image and is centred, so every frame stays on one grid and the model sees a big drawing."""
    ys = np.concatenate([np.nonzero(f[..., 3])[0] for f in frames])
    xs = np.concatenate([np.nonzero(f[..., 3])[1] for f in frames])
    k = max(1, int(0.9 * size // max(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1)))
    cx, cy = (xs.max() + xs.min() + 1) / 2, (ys.max() + ys.min() + 1) / 2
    return k, (int(round(size / 2 - cx * k)), int(round(size / 2 - cy * k)))


def render(sprite, size=1024, k=1, origin=(0, 0)):
    """Line art at `size` square, the sprite scaled by whole number k with its top-left corner at `origin` (may be negative)."""
    h, w = sprite.shape[:2]
    big = np.kron(edges(sprite), np.ones((k, k), bool))
    out = np.full((size, size), 255, np.uint8)
    x0, y0 = origin
    sx0, sy0 = max(0, -x0), max(0, -y0)
    ex, ey = min(w * k, size - x0), min(h * k, size - y0)
    out[y0 + sy0:y0 + ey, x0 + sx0:x0 + ex][big[sy0:ey, sx0:ex]] = 0
    return Image.fromarray(out).convert("RGB")

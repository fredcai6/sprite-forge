"""Deterministic line art from a finished sprite frame: black lines on white, no model involved.

Lines go where the sprite has an edge: the silhouette, every step between two clearly different palette regions, and
the dark outline pixels themselves. Used by the rig-paint track, where a code rig's frame fixes the pose and a model
only paints it.
"""
import numpy as np
from PIL import Image

from .color import srgb_to_oklab


def edges(sprite, thr=0.17, min_run=7):
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


def render(sprite, size=1024, k=None):
    """Line art at `size` square, the sprite scaled by whole number k (default: the largest that fits 92%) and centred.
    Returns (image, k, (x0, y0)) where (x0, y0) is the sprite's top-left corner in the image."""
    h, w = sprite.shape[:2]
    k = k or max(1, int(0.92 * size // max(w, h)))
    e = edges(sprite)
    big = np.kron(e, np.ones((k, k), bool))
    out = np.full((size, size), 255, np.uint8)
    x0, y0 = (size - w * k) // 2, (size - h * k) // 2
    out[y0:y0 + h * k, x0:x0 + w * k][big] = 0
    return Image.fromarray(out).convert("RGB"), k, (x0, y0)

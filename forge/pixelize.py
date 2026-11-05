"""Turns a large painted image into a real sprite: one palette, one grid, hard alpha."""
import numpy as np
from PIL import Image
from scipy import ndimage

from .color import nearest, srgb_to_oklab


def key_alpha(rgb, key=(255, 0, 255), tol=0.16):
    """Alpha from a flat chroma-key background. Blended edge pixels count as background too: they
    are magenta-tinted (red and blue both well above green), which no part of the art is."""
    lab = srgb_to_oklab(rgb)
    near = np.linalg.norm(lab - srgb_to_oklab(np.array(key)), axis=-1) < tol
    r, g, b = [rgb[..., i].astype(int) for i in range(3)]
    spill = np.minimum(r, b) - g > 40
    fg = ndimage.binary_opening(~(near | spill), iterations=2)
    labels, n = ndimage.label(fg)
    sizes = ndimage.sum(fg, labels, range(1, n + 1))
    return labels == 1 + int(np.argmax(sizes))


def black_alpha(rgb, thresh=14):
    """Alpha for art delivered on pure black: the flood-filled region from the border."""
    near = rgb.max(-1) < thresh
    labels, _ = ndimage.label(near)
    border = np.unique(np.r_[labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    return ~np.isin(labels, border[border > 0])


def bbox(alpha):
    ys, xs = np.where(alpha)
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def to_sprite(rgb, alpha, pal_rgb, pal_lab, cell, origin, size, accent=3.0):
    """Sample onto a grid of `cell` source pixels. Each cell takes its most-voted palette colour,
    with saturated colours' votes multiplied by `accent`: a goggle glint covering 15% of a cell
    is what a pixel artist would keep, and averaging would turn it to mud."""
    w, h = size
    idx = nearest(srgb_to_oklab(rgb), pal_lab)
    chroma = np.hypot(pal_lab[:, 1], pal_lab[:, 2])
    weights = 1 + (accent - 1) * np.clip((chroma - 0.05) / 0.1, 0, 1)
    out = np.zeros((h, w, 4), np.uint8)
    for y in range(h):
        for x in range(w):
            x0, y0 = int(origin[0] + x * cell), int(origin[1] + y * cell)
            x1, y1 = int(origin[0] + (x + 1) * cell), int(origin[1] + (y + 1) * cell)
            x0, y0 = max(x0, 0), max(y0, 0)
            if x1 <= x0 or y1 <= y0:
                continue
            a = alpha[y0:y1, x0:x1]
            if a.size == 0 or a.mean() < 0.5:
                continue
            votes = np.bincount(idx[y0:y1, x0:x1][a], minlength=len(pal_rgb)) * weights
            out[y, x, :3] = pal_rgb[votes.argmax()]
            out[y, x, 3] = 255
    return out


def despeckle(sprite):
    """Fill transparent holes of one pixel and remove stray single opaque pixels."""
    a = sprite[..., 3] > 0
    n = ndimage.convolve(a.astype(int), np.ones((3, 3), int), mode="constant") - a
    out = sprite.copy()
    out[a & (n == 0)] = 0
    return out


def upscale(sprite, z, bg=None):
    im = Image.fromarray(sprite, "RGBA").resize(
        (sprite.shape[1] * z, sprite.shape[0] * z), Image.NEAREST
    )
    if bg is None:
        return im
    base = Image.new("RGBA", im.size, (*bg, 255))
    base.alpha_composite(im)
    return base

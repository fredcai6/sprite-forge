"""Lifts baked-in decals off a finished animation, so a game can dress the same sprite at runtime.

    FORGE_PROJECT=path python -m forge.decals characters/NAME/decals.toml

Every frame is registered to one reference frame (rotation, scale and shift against the body
region), so what is marked once on the reference lands in the right place on all of them, even on
poses a model redrew. That gives, for each animation, in sprites/ANIM-plain/:

    sheet.png   the frames with the marked decals painted out
    mask.png    opaque where a decal may show: the surface, with hands and drips left out of it
    atlas.json  the original atlas, plus "seats": each seat's [x, y] for every frame

A runtime draws a plain frame, then decals at the seats clipped by the mask, and they follow the
body and pass behind whatever crosses it. See characters/slice-ny/decals.toml for every key.
"""
import json, sys, tomllib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage, signal

from .generate import ROOT

SPIN = np.arange(-30, 31, 3.0)
SCALE = (0.92, 0.96, 1.0, 1.04, 1.08)


def frames_of(path):
    atlas = json.loads((path / "atlas.json").read_text())
    sheet = np.array(Image.open(path / "sheet.png").convert("RGBA"))
    w = atlas["frame"]["w"]
    return atlas, [sheet[:, i * w:(i + 1) * w] for i in range(atlas["count"])]


def near(rgb, colours, tol):
    d = np.stack([np.abs(rgb.astype(int) - np.array(c)).sum(2) for c in colours])
    return d.min(0) <= tol


def warp(image, angle, scale, order):
    """Turns and sizes an image about its center. Returns it with the matrix mapping source to output."""
    t = np.deg2rad(angle)
    m = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]) * scale
    center = (np.array(image.shape[:2]) - 1) / 2
    inv = np.linalg.inv(m)
    out = ndimage.affine_transform(image, inv, offset=center - inv @ center, order=order, mode="constant")
    return out, m, center


def register(ref, region, frame):
    """The turn, size and shift that best put the reference's body region on this frame."""
    best = (np.inf, None)
    a = frame[..., :3].astype(float) * (frame[..., 3:] > 0)
    for angle in SPIN:
        for scale in SCALE:
            mask, m, center = warp(region.astype(float), angle, scale, 0)
            if mask.sum() < region.sum() * 0.5:
                continue
            t = np.stack([warp(ref[..., c].astype(float), angle, scale, 1)[0] for c in range(3)], 2) * mask[..., None]
            # sum of squares under a sliding mask, by correlation
            cost = 0
            for c in range(3):
                cost = cost + signal.fftconvolve(a[..., c] ** 2, mask[::-1, ::-1], "full") \
                    - 2 * signal.fftconvolve(a[..., c], t[::-1, ::-1, c], "full") + (t[..., c] ** 2).sum()
            cost = cost / mask.sum()
            at = np.unravel_index(np.argmin(cost), cost.shape)
            if cost[at] < best[0]:
                shift = np.array(at) - (np.array(mask.shape) - 1)
                best = (cost[at], (m, center, shift))
    m, center, shift = best[1]
    return lambda xy: (m @ (np.array(xy[::-1], float) - center) + center + shift)[::-1]


def lift(frame, place, inside, spec):
    """Paints the marked decals out of one frame, and anything in their colours on the surface."""
    out = frame.copy()
    rgb, solid = frame[..., :3], frame[..., 3] > 0
    h, w = rgb.shape[:2]
    yy, xx = np.mgrid[:h, :w]
    keep = near(rgb, spec["keep"], spec.get("keep_tol", 24)) if spec.get("keep") else np.zeros((h, w), bool)
    cheese = near(rgb, spec["cheese"], spec.get("cheese_tol", 30)) & solid
    gone = near(rgb, spec["colours"], spec.get("colour_tol", 40)) & inside & ~keep
    gone = ndimage.binary_opening(gone, iterations=1)
    for x, y, r in spec["decals"]:
        cx, cy = place((x, y))
        gone |= (((xx - cx) ** 2 + (yy - cy) ** 2) <= (r + spec.get("slack", 3)) ** 2) & ~cheese & ~keep & solid & inside
    # A decal's own outline is one or two pixels; the slice's outline and the face are thicker.
    dark = rgb.sum(2) < 90
    bold = ndimage.binary_dilation(ndimage.binary_opening(dark, np.ones((3, 3))), iterations=1)
    ring = ndimage.binary_dilation(gone, iterations=spec.get("ring", 2)) & solid & ~keep & ~bold & inside
    source = cheese & ~ring
    _, (iy, ix) = ndimage.distance_transform_edt(~source, return_indices=True)
    out[ring] = frame[iy[ring], ix[ring]]
    # What is left of a decal is small islands beside the paint: scraps of a pattern, a stray outline.
    painted = ndimage.binary_dilation(ring, iterations=2)
    scraps = (~near(out[..., :3], spec["cheese"], spec.get("cheese_tol", 30))) & inside & solid
    labels, _ = ndimage.label(scraps, np.ones((3, 3)))
    _, (iy, ix) = ndimage.distance_transform_edt(~(source & ~scraps), return_indices=True)
    sizes = np.bincount(labels.ravel())
    near_paint = np.unique(labels[painted & scraps])
    for i in near_paint[sizes[near_paint] <= spec.get("scrap", 36)]:
        piece = labels == i
        out[piece] = frame[iy[piece], ix[piece]]
    return out


def surface(frame, inside, spec):
    rgb, solid = frame[..., :3], frame[..., 3] > 0
    cheese = near(rgb, spec["cheese"], spec.get("cheese_tol", 30)) & solid
    return ndimage.binary_closing(cheese, iterations=2) & inside


def polygon(points, shape):
    image = Image.new("L", (shape[1], shape[0]))
    ImageDraw.Draw(image).polygon([tuple(p) for p in points], fill=255)
    return np.array(image) > 0


def shifted(mask, place, shape):
    return np.array(Image.fromarray(mask.astype(np.uint8) * 255).transform(
        (shape[1], shape[0]), Image.Transform.AFFINE, affine(place), Image.Resampling.NEAREST)) > 0


def main(path):
    path = Path(path) if Path(path).is_absolute() else ROOT / path
    spec = tomllib.loads(path.read_text())
    sprites = path.parent / "sprites"
    _, frames = frames_of(sprites / spec.get("reference", "idle"))
    ref = frames[spec.get("reference_frame", 0)]
    region = polygon(spec["body"], ref.shape)
    cheese = polygon(spec["surface"], ref.shape)
    report = {}
    for anim in spec["anims"]:
        atlas, frames = frames_of(sprites / anim)
        seen, plain, masks, seats = {}, [], [], []
        for frame in frames:
            key = frame.tobytes()
            if key not in seen:
                place = register(ref, region, frame)
                inside = shifted(cheese, place, frame.shape)
                done = lift(frame, place, inside, spec)
                seen[key] = (done, surface(done, inside, spec), [[round(float(v)) for v in place(s)] for s in spec["seats"]])
            d, m, s = seen[key]
            plain.append(d); masks.append(m); seats.append(s)
        out = sprites / f"{anim}-plain"
        out.mkdir(exist_ok=True)
        Image.fromarray(np.concatenate(plain, 1)).save(out / "sheet.png")
        Image.merge("LA", [Image.fromarray(np.concatenate(masks, 1).astype(np.uint8) * 255)] * 2).save(out / "mask.png")
        atlas["seats"] = seats
        (out / "atlas.json").write_text(json.dumps(atlas))
        report[anim] = len(seen)
    print(path.parent.name, report)


def affine(place):
    """PIL's output-to-input matrix for a placement."""
    o, x, y = np.array(place((0, 0))), np.array(place((1, 0))), np.array(place((0, 1)))
    m = np.linalg.inv(np.column_stack([x - o, y - o]))
    t = -m @ o
    return (m[0, 0], m[0, 1], t[0], m[1, 0], m[1, 1], t[1])


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])

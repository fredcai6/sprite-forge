"""Builds an animation from generated key poses: python -m forge.animate characters/NAME/anims/X.toml

Frames are layers, as 90s sprites were: the parts that stay still are drawn once and reused exactly,
so only what is meant to move can change. Moving parts are transformed on the large image and then
pixelized, which keeps them on the palette and grid where rotating finished pixels would not.
"""
import json, subprocess, sys, tomllib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .color import load_palette
from .generate import DATA, ROOT, rev
from .pixelize import bbox, despeckle, key_alpha, to_sprite, upscale


def polygon(size, points):
    m = Image.new("L", size, 0)
    ImageDraw.Draw(m).polygon([tuple(p) for p in points], fill=255)
    return np.asarray(m) > 0


def load(run_dir, name):
    a = np.asarray(Image.open(run_dir / f"{name}.png").convert("RGB"))
    return a, key_alpha(a)


def turn(rgba, angle, pivot, dy):
    """Rotates a large RGBA layer about the neck, clockwise for positive angles, then lowers it."""
    im = Image.fromarray(rgba, "RGBA").rotate(-angle, Image.NEAREST, center=tuple(pivot), translate=(0, dy))
    return np.asarray(im)


def main():
    spec_path = Path(sys.argv[1])
    spec = tomllib.loads(spec_path.read_text())
    char_dir = spec_path.parent.parent
    char = tomllib.loads((char_dir / "character.toml").read_text())
    run_dir = DATA / "runs" / char_dir.name / spec["run"] / spec["model"]
    pal_rgb, pal_lab = load_palette(char_dir / "palette.json")
    accent = char["sprite"]["accent"]

    master, m_alpha = load(run_dir, spec["master"])
    x0, y0, x1, y1 = bbox(m_alpha)
    cell = (y1 - y0) / char["sprite"]["height"]
    w, h = spec["canvas"]
    origin = (x0 - spec["margin_left"] * cell, y1 - h * cell)
    pixelate = lambda rgb, alpha: to_sprite(rgb, alpha, pal_rgb, pal_lab, cell, origin, (w, h), accent)

    head_poly = polygon(master.shape[1::-1], spec["head"])
    neck = polygon(master.shape[1::-1], spec["neck_keep"])
    head_alpha = m_alpha & head_poly
    head_rgba = np.dstack([master, head_alpha * 255]).astype(np.uint8)

    bodies = {}
    for name, b in spec["bodies"].items():
        rgb, alpha = load(run_dir, b["pose"])
        if "legs_from_master_below" in b:
            y = b["legs_from_master_below"]
            rgb, alpha = rgb.copy(), alpha.copy()
            rgb[y:], alpha[y:] = master[y:], m_alpha[y:]
        bodies[name] = pixelate(rgb, alpha & ~(head_poly & ~neck))

    head_cache = {}
    frames = []
    for f in spec["frames"]:
        key = (f.get("rot", 0), f.get("dy", 0))
        if key not in head_cache:
            layer = turn(head_rgba, key[0], spec["pivot"], key[1])
            head_cache[key] = pixelate(layer[..., :3], layer[..., 3] > 0)
        body, head = bodies[f["body"]], head_cache[key]
        frame = np.where(head[..., 3:] > 0, head, body)
        for _ in range(f.get("hold", 1)):
            frames.append(despeckle(frame))

    out = char_dir / "sprites" / spec_path.stem
    out.mkdir(parents=True, exist_ok=True)
    sheet = np.concatenate(frames, axis=1)
    Image.fromarray(sheet, "RGBA").save(out / "sheet.png", optimize=True)
    ms = spec["frame_ms"]
    (out / "atlas.json").write_text(json.dumps({
        "frame": {"w": w, "h": h}, "count": len(frames), "frame_ms": ms, "loop": True,
        "rev": rev(), "source": {"run": spec["run"], "model": spec["model"], "master": spec["master"]},
    }, indent=1))
    z = spec.get("preview_scale", 3)
    pics = [upscale(f, z, (16, 18, 20)).convert("RGB") for f in frames]
    pics[0].save(out / "preview.gif", save_all=True, append_images=pics[1:], duration=ms, loop=0)
    colours = len(np.unique(sheet[sheet[..., 3] > 0][:, :3], axis=0))
    legs = spec["static_below_row"]
    moved = sum(int((a[legs:] != frames[0][legs:]).any()) for a in frames)
    print(f"{len(frames)} frames {w}x{h}, {colours} colours, cell {cell:.2f}, frames whose legs differ: {moved}")


if __name__ == "__main__":
    main()

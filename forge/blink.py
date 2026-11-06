"""Adds a blink frame to a held idle: the eyes closed over a copy of frame 0.

    FORGE_PROJECT=path python -m forge.blink characters/NAME/sprites/ANIM-plain cx,cy,rx,ry ...

Each eye is an ellipse on frame 0. It is filled from the skin around it, with a dark lid line through
the middle. The new frame's index goes in atlas.json as "blink".
"""
import json, sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from .generate import ROOT


def main(folder, *eyes):
    folder = ROOT / folder
    atlas = json.loads((folder / "atlas.json").read_text())
    sheet = np.array(Image.open(folder / "sheet.png").convert("RGBA"))
    w, h = atlas["frame"]["w"], atlas["frame"]["h"]
    count = atlas["count"]
    frame = sheet[:, :w].copy()
    yy, xx = np.mgrid[:h, :w]
    bright = frame[..., :3].sum(2) > 300
    shut = np.zeros((h, w), bool)
    lids = []
    for eye in eyes:
        cx, cy, rx, ry = (float(v) for v in eye.split(","))
        shut |= ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1
        lids.append((cx, cy, rx))
    skin = ~ndimage.binary_dilation(shut, iterations=1) & bright & (frame[..., 2].astype(int) < 215) & (frame[..., 3] > 0)
    _, (iy, ix) = ndimage.distance_transform_edt(~skin, return_indices=True)
    frame[shut] = frame[iy[shut], ix[shut]]
    for cx, cy, rx in lids:
        for x in range(round(cx - rx * 0.7), round(cx + rx * 0.7) + 1):
            y = round(cy + (1 - ((x - cx) / rx) ** 2) * rx * 0.28)
            frame[y:y + 2, x] = (10, 4, 14, 255)
    Image.fromarray(np.concatenate([sheet[:, :count * w], frame], 1)).save(folder / "sheet.png")
    atlas["blink"] = count
    atlas["seats"] = atlas["seats"][:count] + [atlas["seats"][0]]
    (folder / "atlas.json").write_text(json.dumps(atlas))
    mask = np.array(Image.open(folder / "mask.png").convert("LA"))
    Image.fromarray(np.concatenate([mask[:, :count * w], mask[:, :w]], 1), "LA").save(folder / "mask.png")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(*sys.argv[1:])

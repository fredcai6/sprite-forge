"""Builds the New York layers into OUT (the game's assets/backgrounds/newyork):
    python backgrounds/newyork/build.py OUT
far is opaque; mid and near are keyed and sit with their lowest content on the ground line y=600
(mid) or the bottom of the stage (near). All three wrap horizontally."""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from forge.layers import key_alpha, load, pink_to_brick, quantize, soften, tile_seam, write_sources

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
# (run, seed): far and near from the pale no-text pass, mid from the no-text magenta pass
PICK = {"far": ("newyork/layers2", 4), "mid": ("newyork/mid3", 3), "near": ("newyork/layers2", 1)}
# text the generator wrote on signs and carvings in the chosen mid image (source pixel boxes, x0 y0 x1 y1).
# Each box is filled with its own per-row median colour, which leaves a blank sign of the same colour.
ERASE_MID = [(688, 395, 933, 426), (938, 393, 954, 420), (1430, 395, 1662, 426), (1300, 592, 1338, 608)]


def erase(rgb, boxes):
    rgb = rgb.copy()
    for x0, y0, x1, y1 in boxes:
        reg = rgb[y0:y1, x0:x1]
        rgb[y0:y1, x0:x1] = np.median(reg, axis=1, keepdims=True).astype(np.uint8)
    return rgb
GROUND, W, H = 600, 2560, 720
sources = {}
for name, (run, seed) in PICK.items():
    im, meta = load(run, "klein", name if name != "mid" else "mid", seed)
    rgb = np.asarray(im)
    if name == "mid":
        rgb = erase(rgb, ERASE_MID)
        meta["erased_text_boxes"] = ERASE_MID
    if name == "far":
        rgba = np.dstack([rgb, np.full(rgb.shape[:2], 255, np.uint8)])
    else:
        a = key_alpha(rgb)
        rgba = np.dstack([rgb, a * 255]).astype(np.uint8)
    if name == "mid":
        # the lowest solid content (kerb, rubble) lands on the ground line; the sidewalk strip below it
        # belongs to the game's ground and is cut off
        bottom = 636  # in the chosen image the kerb top (objects' feet) is at y=636
        rgba = np.roll(rgba, GROUND - bottom, axis=0)
        rgba[GROUND + 6:] = 0
        rgba[:max(GROUND - bottom, 0)] = 0
        meta["shifted_by"] = GROUND - bottom
    rgba = tile_seam(rgba, W)
    meta["seam_start_x"] = int(tile_seam.last)
    if name == "mid":
        rgba = pink_to_brick(rgba)
        # the pink brownstone (source x 1330-1705): every hot pink in that box becomes brick salmon
        x0, x1 = 1330 - meta["seam_start_x"], 1705 - meta["seam_start_x"]
        part = pink_to_brick(rgba[:, max(x0, 0):max(x1, 0)].copy(), red_rows=(0, H))
        rgba[:, max(x0, 0):max(x1, 0)] = part
    if name == "near":
        # silhouettes only: the generator left pink rim light on bags and bins, so keep just the lightness
        g = rgba[..., :3].astype(float).mean(2, keepdims=True)
        rgba[..., :3] = np.clip(np.array([48, 38, 62]) + g * 0.2, 0, 255).astype(np.uint8)
    # lighter and further back: lifted values, flatter contrast, an atmosphere wash, fine cracks filtered out
    LOOK = {"far": dict(contrast=0.9, lift=6, haze=(200, 180, 210), haze_amt=0.08, median=3),
            "mid": dict(contrast=0.72, lift=26, haze=(205, 175, 200), haze_amt=0.22, median=3),
            "near": dict(contrast=0.85, lift=10, median=0)}[name]
    rgba = soften(rgba, **LOOK)
    rgba = quantize(rgba, n={"far": 40, "mid": 44, "near": 14}[name])
    Image.fromarray(rgba, "RGBA").save(out / f"{name}.png", optimize=True)
    sources[name] = meta
write_sources(out / "sources.json", sources)
print("ok")

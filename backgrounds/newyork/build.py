"""Builds the New York layers into OUT (the game's assets/backgrounds/newyork):
    python backgrounds/newyork/build.py OUT
far is opaque; mid and near are keyed and sit with their lowest content on the ground line y=600
(mid) or the bottom of the stage (near). All three wrap horizontally."""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from forge.layers import key_alpha, load, pink_to_brick, quantize, tile_seam, write_sources

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
PICK = {"far": 2, "mid": 1, "near": 1}
GROUND, W, H = 600, 2560, 720
sources = {}
for name, seed in PICK.items():
    im, meta = load("newyork/layers", "klein", name, seed)
    rgb = np.asarray(im)
    if name == "far":
        rgba = np.dstack([rgb, np.full(rgb.shape[:2], 255, np.uint8)])
    else:
        a = key_alpha(rgb)
        rgba = np.dstack([rgb, a * 255]).astype(np.uint8)
    if name == "mid":
        # the lowest solid content (kerb, rubble) lands on the ground line; the sidewalk strip below it
        # belongs to the game's ground and is cut off
        # the generated strip ends in a black bar; objects' feet sit about 14 rows above its top
        lum = rgb.mean(2)
        bottom = int(np.flatnonzero(lum.mean(1) > 20).max()) + 1 - 14
        rgba = np.roll(rgba, GROUND - bottom, axis=0)
        rgba[GROUND + 6:] = 0
        rgba[:max(GROUND - bottom, 0)] = 0
        meta["shifted_by"] = GROUND - bottom
    rgba = tile_seam(rgba, W)
    if name != "far":
        rgba = pink_to_brick(rgba, red_rows=(0, 330) if name == "mid" else (0, H))
    meta["seam_start_x"] = int(tile_seam.last)
    rgba = quantize(rgba, n={"far": 56, "mid": 64, "near": 16}[name])
    Image.fromarray(rgba, "RGBA").save(out / f"{name}.png", optimize=True)
    sources[name] = meta
write_sources(out / "sources.json", sources)
print("ok")

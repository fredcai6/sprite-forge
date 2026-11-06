"""Builds one neighbourhood into OUT/<city>/<area>/ : far, mid, near (2560x720), ground (256x120 wrapping), door.
    python backgrounds/area_build.py CITY/AREA ASSETS_BACKGROUNDS
Per-area picks (seed per shot, kerb row of the picked mid, text-erase boxes, haze) live in backgrounds/areas.py."""
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forge.color import srgb_to_oklab
from forge.layers import key_alpha, load, pink_to_brick, quantize, soften, tile_seam, write_sources

GROUND, W, H = 600, 2560, 720
AREAS = {}
exec((Path(__file__).parent / "areas.py").read_text())

key = sys.argv[1]
city, area = key.split("/")
cfg = AREAS[key]
out = Path(sys.argv[2]) / city / area
out.mkdir(parents=True, exist_ok=True)
run = area
sources = {}


def erase(rgb, boxes):
    rgb = rgb.copy()
    for x0, y0, x1, y1 in boxes:
        reg = rgb[y0:y1, x0:x1]
        rgb[y0:y1, x0:x1] = np.median(reg, axis=1, keepdims=True).astype(np.uint8)
    return rgb


for name in ("far", "mid", "near"):
    im, meta = load(f"{run}/layers", "klein", name, cfg["seeds"][name])
    rgb = np.asarray(im)
    if name == "mid" and cfg.get("erase"):
        rgb = erase(rgb, cfg["erase"]); meta["erased_text_boxes"] = cfg["erase"]
    if name == "far":
        rgba = np.dstack([rgb, np.full(rgb.shape[:2], 255, np.uint8)])
    else:
        rgba = np.dstack([rgb, key_alpha(rgb) * 255]).astype(np.uint8)
    if name == "near":
        # light framing only: drop the generated horizon bar and any wide slab (a banner or beam the model merged into one block)
        from forge.neardeclutter import clean
        a = clean(rgba[..., 3] > 0)
        rgba[..., 3] = np.where(a, 255, 0)
    if name == "mid":
        g = cfg["ground"]
        rgba = np.roll(rgba, GROUND - g, axis=0)
        rgba[GROUND + 2:] = 0
        rgba[:max(GROUND - g, 0)] = 0
        meta["shifted_by"] = GROUND - g
    rgba = tile_seam(rgba, W)
    meta["seam_start_x"] = int(tile_seam.last)
    if name == "mid":
        rgba = pink_to_brick(rgba)
    if name == "near":
        gl = rgba[..., :3].astype(float).mean(2, keepdims=True)
        rgba[..., :3] = np.clip(np.array(cfg["near_tint"]) + gl * 0.2, 0, 255).astype(np.uint8)
    rgba = soften(rgba, **cfg["look"][name])
    rgba = quantize(rgba, n={"far": 40, "mid": 44, "near": 14}[name])
    Image.fromarray(rgba, "RGBA").save(out / f"{name}.png", optimize=True)
    sources[name] = meta

# ground: downscale the swatch to 256x120 and make it wrap in both directions
im, meta = load(f"{run}/props", "klein", "ground", cfg["seeds"]["ground"])
sw, sh = im.size   # the swatch has perspective at its edges: use only its flat middle
g = np.asarray(im.crop((int(sw * .22), int(sh * .15), int(sw * .78), int(sh * .72))).resize((256, 120), Image.LANCZOS)).astype(float)
ww = np.hanning(256)[None, :, None]; wh = np.hanning(120)[:, None, None]
win = ww * wh
rolled = np.roll(np.roll(g, 128, axis=1), 60, axis=0)
rw = 1 - win
tile = g * win + rolled * (1 - win)
tile = soften(np.dstack([tile.clip(0, 255).astype(np.uint8), np.full((120, 256), 255, np.uint8)]), contrast=0.9, lift=4)
Image.fromarray(quantize(tile, n=20), "RGBA").save(out / "ground.png", optimize=True)
sources["ground"] = meta

# door: keyed, fitted to its stage height with feet at the bottom
im, meta = load(f"{run}/props", "klein", "door", cfg["seeds"]["door"])
rgb = np.asarray(im)
a = key_alpha(rgb)
lab = srgb_to_oklab(rgb)
a = ndimage.binary_fill_holes(a)
lbl, n = ndimage.label(a)
if n > 1:
    sz = ndimage.sum(a, lbl, range(1, n + 1)); a = lbl == 1 + int(np.argmax(sz))
ys, xs = np.where(a)
d = Image.fromarray(np.dstack([rgb, a * 255]).astype(np.uint8), "RGBA").crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
th = cfg.get("door_h", 170)
d = d.resize((max(1, round(d.width * th / d.height)), th), Image.LANCZOS)
d = Image.fromarray(quantize(np.asarray(d), n=44), "RGBA")
d.save(out / "door.png", optimize=True)
sources["door"] = meta
write_sources(out / "sources.json", sources)
print("ok", key, d.size)

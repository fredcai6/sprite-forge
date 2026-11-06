"""Builds a city's three layers into OUT (the game's assets/backgrounds/<city>):
    python backgrounds/city_build.py CITY OUT
far is opaque; mid and near are keyed. mid's feet sit on the ground line y=600; near keeps its own bottom edge.
All three wrap horizontally. Every pick, erased text box and ground row is a judgement made by looking at the
generated seeds, recorded in CITIES below and in OUT/sources.json."""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forge.layers import key_alpha, load, pink_to_brick, quantize, soften, tile_seam, write_sources

GROUND, W, H = 600, 2560, 720
# seeds: far/mid/near picks. ground: source row of the kerb/feet in the picked mid. erase: [x0,y0,x1,y1] boxes of
# lookalike lettering to blank. haze: atmosphere colour. near_tint: flat colour of the foreground silhouettes.
CITIES = {}
exec((Path(__file__).parent / "cities.py").read_text())


def erase(rgb, boxes):
    rgb = rgb.copy()
    for x0, y0, x1, y1 in boxes:
        reg = rgb[y0:y1, x0:x1]
        rgb[y0:y1, x0:x1] = np.median(reg, axis=1, keepdims=True).astype(np.uint8)
    return rgb


city, out = sys.argv[1], Path(sys.argv[2])
cfg = CITIES[city]
out.mkdir(parents=True, exist_ok=True)
sources = {}
for name in ("far", "mid", "near"):
    im, meta = load(cfg.get("mid_run", f"{city}/layers") if name == "mid" else f"{city}/layers", "klein", name, cfg["seeds"][name])
    rgb = np.asarray(im)
    if name == "mid" and cfg.get("erase"):
        rgb = erase(rgb, cfg["erase"])
        meta["erased_text_boxes"] = cfg["erase"]
    if name == "far":
        rgba = np.dstack([rgb, np.full(rgb.shape[:2], 255, np.uint8)])
    else:
        rgba = np.dstack([rgb, key_alpha(rgb) * 255]).astype(np.uint8)
    if name == "mid":
        rgba = np.roll(rgba, GROUND - cfg["ground"], axis=0)
        rgba[GROUND + 6:] = 0
        rgba[:max(GROUND - cfg["ground"], 0)] = 0
        meta["shifted_by"] = GROUND - cfg["ground"]
    rgba = tile_seam(rgba, W)
    meta["seam_start_x"] = int(tile_seam.last)
    if name == "mid":
        rgba = pink_to_brick(rgba)
    if name == "near":
        g = rgba[..., :3].astype(float).mean(2, keepdims=True)
        rgba[..., :3] = np.clip(np.array(cfg["near_tint"]) + g * 0.2, 0, 255).astype(np.uint8)
    look = cfg["look"][name]
    rgba = soften(rgba, **look)
    rgba = quantize(rgba, n={"far": 40, "mid": 44, "near": 14}[name])
    Image.fromarray(rgba, "RGBA").save(out / f"{name}.png", optimize=True)
    sources[name] = meta
write_sources(out / "sources.json", sources)
print("ok", city)

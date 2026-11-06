"""Maps (paper textures) and HQ pies: python backgrounds/shared/build_maps_pies.py ASSETS_BACKGROUNDS"""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from forge.generate import DATA, rev
from forge.layers import key_alpha, quantize

A = Path(sys.argv[1])
(A / "maps").mkdir(parents=True, exist_ok=True)
MAPS = {"newyork": 3, "chicago": 1, "dc": 3, "naples": 1, "california": 1}
for c, sd in MAPS.items():
    Image.open(DATA / f"runs/shared/maps/klein/{c}/{sd}.png").convert("RGB").resize((1280, 720), Image.LANCZOS).save(A / "maps" / f"{c}.png", optimize=True)
PIES = {"ny": (3, 104), "deep": (3, 96), "jumbo": (3, 150), "naples": (3, 96)}
for k, (sd, w) in PIES.items():
    rgb = np.asarray(Image.open(DATA / f"runs/shared/pies/klein/{k}/{sd}.png").convert("RGB"))
    a = key_alpha(rgb)
    ys, xs = np.where(a)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    yy, xx = np.mgrid[0:rgb.shape[0], 0:rgb.shape[1]]
    cx, cy, rx, ry = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2
    a = a & ((((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) <= 1.0)   # a pie is a disc: clip off the backdrop's glow
    im = Image.fromarray(np.dstack([rgb, a * 255]).astype(np.uint8), "RGBA").crop((x0, y0, x1, y1)).resize((w, w), Image.LANCZOS)
    Image.fromarray(quantize(np.asarray(im), n=22), "RGBA").save(A / "hq" / f"pie-{k}.png", optimize=True)
(A / "maps" / "sources.json").write_text(json.dumps({"rev": rev(), "maps": MAPS, "pies": {k: v[0] for k, v in PIES.items()}}, indent=1))
print("ok")

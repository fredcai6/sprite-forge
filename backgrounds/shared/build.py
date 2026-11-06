"""Builds the shared props into ASSETS/backgrounds/props: python backgrounds/shared/build.py ASSETS_BACKGROUNDS"""
import json, sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from forge.color import srgb_to_oklab
from forge.generate import DATA, rev
from forge.layers import key_alpha, quantize

out = Path(sys.argv[1]) / "props"
out.mkdir(parents=True, exist_ok=True)
RUN = DATA / "runs/shared/props/klein"
PICK = {"box": 1, "box_pile": 1, "box_broken": 1, "drain": 1, "sign": 1, "star": 1}
src = {}


def get(shot, seed=1, fill=True):
    src[shot] = seed
    rgb = np.asarray(Image.open(RUN / shot / f"{seed}.png").convert("RGB"))
    a = key_alpha(rgb)
    lab = srgb_to_oklab(rgb)
    ch = np.hypot(lab[..., 1], lab[..., 2])
    hue = np.degrees(np.arctan2(lab[..., 2], lab[..., 1])) % 360
    a &= ~((hue > 290) & (hue < 350) & (ch > 0.10) & (lab[..., 0] < 0.55))
    if fill:
        a = ndimage.binary_fill_holes(a)
    lbl, n = ndimage.label(a)
    if n > 1:
        sz = ndimage.sum(a, lbl, range(1, n + 1))
        a = lbl == 1 + int(np.argmax(sz))   # only the prop: the generated backdrop's corner glow is a separate piece
    ys, xs = np.where(a)
    return Image.fromarray(np.dstack([rgb, a * 255]).astype(np.uint8), "RGBA").crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))


def fit(img, w, h, n=36, bottom=True):
    s = min(w / img.width, h / img.height)
    im = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.LANCZOS)
    c = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    c.alpha_composite(im, ((w - im.width) // 2, h - im.height if bottom else (h - im.height) // 2))
    return Image.fromarray(quantize(np.asarray(c), n=n), "RGBA")


def save(im, name):
    im.save(out / name, optimize=True)


crate = get("box")
save(fit(crate, 84, 84, 30), "box.png")
# pile: a stack of the crate, three rows built from the same art so it matches the single crate
cw, ch_ = 64, 56
pile = Image.new("RGBA", (180, 120), (0, 0, 0, 0))
c = crate.resize((cw, ch_), Image.LANCZOS)
rows = [(0, [4, 58, 114]), (1, [30, 86]), (2, [58])]
for r, xs in rows:
    for x in xs:
        pile.alpha_composite(c, (x, 120 - ch_ - r * 32))
pile.alpha_composite(c.rotate(14, expand=True, resample=Image.BICUBIC).resize((52, 48)), (126, 76))
save(Image.fromarray(quantize(np.asarray(pile), n=30), "RGBA"), "box-pile.png")
save(fit(get("box_broken"), 90, 24, 24), "box-broken.png")
save(fit(get("drain"), 120, 26, 18), "drain.png")
s = get("sign")
save(fit(s, 110, 200, 20), "sign.png")
save(fit(get("star"), 24, 24, 12, bottom=False), "star.png")
# ledge: black fire-escape walkway, hand-built so it tiles exactly
t = np.zeros((28, 64, 4), np.uint8); t[..., 3] = 255
ink, steel, hi = (22, 20, 30), (48, 46, 60), (96, 92, 112)
t[..., :3] = ink
t[0:2, :, :3] = hi
t[2:5, :, :3] = steel
for x in range(0, 64, 8):
    t[5:9, x:x + 3, :3] = steel      # grating teeth
for x in range(64):
    y = 11 + int(abs((x % 16) - 8) * 1.4)
    t[y:y + 2, x, :3] = steel
t[26:28, :, :3] = steel
save(Image.fromarray(t, "RGBA"), "ledge.png")
for top in ["pepperoni", "jalapeno", "garlic", "mushroom", "basil", "pineapple", "sausage"]:
    save(fit(get("pickup_" + top), 52, 52, 28, bottom=False), f"pickup-{top}.png")
    save(fit(get("shot_" + top), 28, 28, 16, bottom=False), f"shot-{top}.png")
(out / "sources.json").write_text(json.dumps({"rev": rev(), "run": "shared/props", "seeds": src}, indent=1))
print("ok")

# ---- shots (second pass): chunky 36x36 projectile discs from runs/shared/shots, heavy outline added on the silhouette
SHOTS = Path(DATA / "runs/shared/shots/klein")
if SHOTS.exists():
    from forge.compose import keyout
    PICK_SHOT = json.loads((Path(__file__).parent / "shot_picks.json").read_text()) if (Path(__file__).parent / "shot_picks.json").exists() else {}
    for top in ["pepperoni", "jalapeno", "garlic", "mushroom", "basil", "pineapple", "sausage"]:
        sd = PICK_SHOT.get(top, 1)
        rgb = np.asarray(Image.open(SHOTS / f"shot_{top}/{sd}.png").convert("RGB"))
        a = keyout(rgb)
        ys, xs = np.where(a)
        im = Image.fromarray(np.dstack([rgb, a * 255]).astype(np.uint8), "RGBA").crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
        s = 30 / max(im.size)
        im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS)
        c = Image.new("RGBA", (36, 36), (0, 0, 0, 0))
        c.alpha_composite(im, ((36 - im.width) // 2, (36 - im.height) // 2))
        arr = quantize(np.asarray(c), n=14)
        al = arr[..., 3] > 0
        ring = ndimage.binary_dilation(al, structure=np.ones((3, 3), bool), iterations=1) & ~al
        arr[ring] = (22, 12, 34, 255)
        Image.fromarray(arr, "RGBA").save(out / f"shot-{top}.png", optimize=True)

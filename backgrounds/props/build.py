"""Builds the second-pass props into the game's assets/backgrounds:
    python backgrounds/props/build.py ASSETS_BACKGROUNDS
Each prop is keyed, cropped, scaled to the stage grid and snapped to a small palette."""
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from forge.color import srgb_to_oklab
from forge.generate import DATA
from forge.layers import key_alpha, quantize, tile_seam, write_sources

out = Path(sys.argv[1])
RUNS = {"fifth": DATA / "runs/props/fifth/klein", "fourth": DATA / "runs/props/fourth/klein", "second": DATA / "runs/props/second/klein", "third": DATA / "runs/props/third/klein"}
src = {}


def get(run, shot, seed, shadow=True):
    src.setdefault(f"{run}/{shot}", seed)
    rgb = np.asarray(Image.open(RUNS[run] / shot / f"{seed}.png").convert("RGB"))
    a = key_alpha(rgb)
    if shadow:
        lab = srgb_to_oklab(rgb)
        ch = np.hypot(lab[..., 1], lab[..., 2])
        hue = np.degrees(np.arctan2(lab[..., 2], lab[..., 1])) % 360
        a &= ~((hue > 290) & (hue < 350) & (ch > 0.10) & (lab[..., 0] < 0.55))
    lab, n = ndimage.label(a)
    if n > 1:
        sizes = ndimage.sum(a, lab, range(1, n + 1))
        a = np.isin(lab, 1 + np.flatnonzero(sizes > 0.01 * sizes.max()))
    ys, xs = np.where(a)
    box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    return Image.fromarray(np.dstack([rgb, a * 255]).astype(np.uint8), "RGBA").crop(box)


def fit(img, w, h, n=40, align="bottom", post=None):
    s = min(w / img.width, h / img.height)
    im = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.LANCZOS)
    if post:
        im = Image.fromarray(post(np.asarray(im).copy()), "RGBA")
    canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    canvas.alpha_composite(im, ((w - im.width) // 2, h - im.height if align == "bottom" else 0))
    return Image.fromarray(quantize(np.asarray(canvas), n=n), "RGBA")


def save(img, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, optimize=True)


# HQ table: 1080 wide, the top slab's upper edge at y=470, drawn on a 1280x720 stage
t = get("fifth", "table", 2)
s = 1080 / t.width
t = t.resize((1080, round(t.height * s)), Image.LANCZOS)
stage = Image.new("RGBA", (1280, 720), (0, 0, 0, 0))
stage.alpha_composite(t, (100, 470))
save(Image.fromarray(quantize(np.asarray(stage), n=40), "RGBA"), out / "hq/table.png")
TABLE_H = t.height

# HQ box: 150 wide, bottom edge on the canvas bottom (the game anchors bottom-centre at y=504)
save(fit(get("fifth", "box", 4), 150, 150, n=24), out / "hq/box.png")

# platform: a clean hand-built 64x28 steel walkway tile (the generated girders came out too noisy for a light street).
def platform_tile(h=28, w=64, tones=None):
    """Walkable top row is the lightest; riveted plate, a dark gap, then a truss with diagonal struts. Tiles exactly."""
    top, plate, plate_dk, line, rust, under = tones or ((206, 196, 214), (150, 140, 164), (118, 108, 134), (46, 34, 62), (156, 96, 78), (84, 74, 104))
    t = np.zeros((h, w, 4), np.uint8)
    t[..., 3] = 255
    t[:, :, :3] = under
    t[0, :, :3] = line                      # outline above the lip
    t[1:3, :, :3] = top                     # walkable surface at the very top row (row 1-2)
    t[3:11, :, :3] = plate
    t[9:11, :, :3] = plate_dk
    t[11, :, :3] = line
    for x in range(4, w, 16):               # rivets
        t[5, x:x + 2, :3] = top; t[6, x:x + 2, :3] = plate_dk
    t[3:11, 0, :3] = plate_dk               # plate seam at the tile joint
    # truss: a zigzag of 2px struts, period 16 so it tiles
    for x in range(w):
        y = 13 + int(abs((x % 16) - 8) * 1.6)
        t[y:y + 2, x, :3] = line
        t[y + 2:y + 3, x, :3] = plate_dk
    t[h - 3:h - 1, :, :3] = plate_dk
    t[h - 1, :, :3] = line
    t[12:h - 3, :, :3] = np.where((t[12:h - 3, :, :3] == under).all(-1)[..., None], under, t[12:h - 3, :, :3])
    # a few rust flecks, deterministic
    for x, y in [(9, 7), (10, 7), (33, 8), (34, 8), (35, 8), (52, 6), (21, 20), (22, 20), (47, 22)]:
        t[y, x, :3] = rust
    return Image.fromarray(t, "RGBA")


tile = platform_tile()
src["platform"] = "hand-built tile (forge backgrounds/props/build.py: platform_tile)"
save(tile, out / "newyork/platform.png")

# barricade: bottom of the tall crate stack, 80x100
def grime(a):
    """Street dirt: darker toward the ground, soot speckles."""
    h, w = a.shape[:2]
    rng = np.random.default_rng(3)
    k = 1 - 0.38 * (np.arange(h)[:, None] / h) ** 2
    speck = np.where(rng.random((h, w)) < 0.07, 0.7, 1.0)
    a[..., :3] = (a[..., :3] * (k * speck)[..., None]).astype(np.uint8)
    return a


def glow(a):
    """The drum is lit from the fire above it: warm orange washes the upper body."""
    h, w = a.shape[:2]
    fall = np.clip(1 - (np.arange(h)[:, None] - h * 0.3) / (h * 0.55), 0, 1) ** 1.4 * (np.arange(h)[:, None] > h * 0.3)
    warm = np.array([255, 150, 50], float)
    a[..., :3] = np.clip(a[..., :3] * (1 + 0.5 * fall[..., None]) + warm * 0.3 * fall[..., None], 0, 255).astype(np.uint8)
    return a


save(fit(get("fourth", "barricade", 1), 80, 100, n=32, post=grime), out / "newyork/barricade.png")
save(fit(get("second", "barrel", 1), 60, 110, n=40, post=glow), out / "newyork/barrel.png")
save(fit(get("second", "barrel_out", 1), 60, 110, n=32), out / "newyork/barrel-out.png")
save(fit(get("third", "door", 3, shadow=False), 112, 165, n=44), out / "newyork/door.png")
write_sources(out / "props-sources.json", {"picked_seeds": src, "table_height_px": TABLE_H,
              "box_anchor": "bottom-centre of box.png sits at (x, 504)"})
print("ok", TABLE_H)

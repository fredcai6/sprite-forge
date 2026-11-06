"""Builds the three Pizza HQ layers into OUT (the game's assets/backgrounds/hq):
    python backgrounds/hq/build.py OUT
back is the room with a blank corkboard painted on it; mid is props along the wall; front is dark
foreground silhouettes. All 1280x720, one pixel = one stage pixel."""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from forge.generate import DATA
from forge.layers import key_alpha, quantize, soften, write_sources

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
W, H, FLOOR = 1280, 720, 480
BOARD = (700, 70, 520, 350)  # outer frame x, y, w, h
FRAME = 12
RUN = DATA / "runs/hq"
src = {}


def img(run, shot, seed):
    src.setdefault(run, []).append({"shot": shot, "seed": seed})
    return Image.open(RUN / run / "klein" / shot / f"{seed}.png").convert("RGB")


def cutouts(im, minh=60):
    """Separate props on a key-colour sheet: [(rgba crop, x0)] left to right."""
    rgb = np.asarray(im)
    a = key_alpha(rgb)
    lab, n = ndimage.label(ndimage.binary_dilation(a, iterations=4))
    res = []
    for i, o in enumerate(ndimage.find_objects(lab), 1):
        m = a & (lab == i)
        ys, xs = np.where(m)
        if len(ys) == 0 or ys.max() - ys.min() < minh or ys.min() < 140 and (ys.max() - ys.min()) < 160:
            continue  # lamp glow and specks
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        crop = np.dstack([rgb, m * 255]).astype(np.uint8)[y0:y1, x0:x1]
        res.append((x0, Image.fromarray(crop, "RGBA")))
    return [c for _, c in sorted(res, key=lambda t: t[0])]


def scaled(c, s):
    return c.resize((max(1, round(c.width * s)), max(1, round(c.height * s))), Image.LANCZOS)


def paste_feet(canvas, c, x, feet):
    canvas.alpha_composite(c, (round(x), round(feet - c.height)))


# ---- back: room, floor line raised to FLOOR, blank corkboard painted on the wall
room = img("back3", "back", 3).resize((W, H), Image.LANCZOS)
a = np.asarray(room)
old = 582  # the generated baseboard
rows = np.empty((H, W, 3), np.uint8)
for y in range(H):
    sy = y * old / FLOOR if y < FLOOR else old + (y - FLOOR) * (H - old) / (H - FLOOR)
    rows[y] = a[min(int(sy), H - 1)]
rgba = np.dstack([rows, np.full((H, W), 255, np.uint8)])
back = quantize(soften(rgba, contrast=0.92, lift=10, median=3), n=44)

bx, by, bw, bh = BOARD
rng = np.random.default_rng(7)
ys, xs = np.mgrid[by:by + bh, bx:bx + bw]
cork = np.array([[108, 68, 40], [128, 82, 46], [148, 98, 56], [88, 54, 34], [166, 118, 70]], np.uint8)
lightness = np.clip(1.15 - (xs - 400) / 1100.0 - 0.15 * (ys - by) / bh, 0.35, 1.0)
tone = np.clip((rng.random(xs.shape) * 0.55 + lightness * 0.55 + rng.random(xs.shape) ** 6 * 0.3) * 4.2 - 1.3, 0, 4).astype(int)
# cork grain is blocky 2x2 so it reads at stage pixel scale
tone = np.kron(tone[::2, ::2], np.ones((2, 2), int))[:bh, :bw]
board = cork[tone]
shadow = back[by + 8:by + bh + 8, bx + 8:bx + bw + 8, :3].astype(float) * 0.55
back[by + 6:by + bh + 6, bx + 6:bx + bw + 6, :3] = np.where(True, back[by + 6:by + bh + 6, bx + 6:bx + bw + 6, :3] * 0.6, 0).astype(np.uint8)
back[by:by + bh, bx:bx + bw, :3] = board
wood = np.array([[92, 56, 34], [66, 38, 26], [30, 16, 30]], np.uint8)
fr = np.full((bh, bw), 99)
for i in range(FRAME):
    ring = np.ones((bh, bw), bool)
    ring[i + 1:bh - i - 1, i + 1:bw - i - 1] = False
    ring[:i, :] = ring[bh - i:, :] = False
    ring[:, :i] = ring[:, bw - i:] = False
    fr[ring] = 2 if (i < 2 or i >= FRAME - 2) else (1 if i % 5 == 0 else 0)
view = back[by:by + bh, bx:bx + bw, :3]
view[fr != 99] = wood[fr[fr != 99]]
# thumbtack nails in the corners
for tx, ty in [(bx + 6, by + 6), (bx + bw - 7, by + 6), (bx + 6, by + bh - 7), (bx + bw - 7, by + bh - 7)]:
    back[ty - 1:ty + 1, tx - 1:tx + 1, :3] = (200, 190, 170)
Image.fromarray(back[..., :3], "RGB").save(out / "back.png", optimize=True)

# ---- mid: props standing along the wall, feet at y=540 (nearer than the baseboard)
FEET = 540
mid = Image.new("RGBA", (W, H), (0, 0, 0, 0))
boxes = cutouts(img("props2", "boxes", 3))    # stack, bucket+mop, sack
table = cutouts(img("props2", "table", 2))    # table, stool, mixer
paste_feet(mid, scaled(boxes[0], 0.36), 24, FEET)
paste_feet(mid, scaled(boxes[1], 0.38), 190, FEET + 6)
paste_feet(mid, scaled(boxes[2], 0.34), 300, FEET - 4)
paste_feet(mid, scaled(table[0], 0.30), 755, FEET + 4)
paste_feet(mid, scaled(table[2], 0.30), 985, FEET)
paste_feet(mid, scaled(table[1], 0.34), 1120, FEET + 8)
m = quantize(soften(np.asarray(mid), contrast=0.92, lift=8, median=3), n=36)
Image.fromarray(m, "RGBA").save(out / "mid.png", optimize=True)

# ---- front: near-black silhouettes at the stage edges, cropped by the frame
front = Image.new("RGBA", (W, H), (0, 0, 0, 0))
shelf = cutouts(img("props2", "shelf", 2))[0]
sh = scaled(shelf, 0.95)
front.alpha_composite(sh, (-310, H - sh.height + 90))
stack = scaled(cutouts(img("props2", "boxes", 1))[0], 0.62)
front.alpha_composite(stack, (W - stack.width + 90, H - stack.height + 60))
d = ImageDraw.Draw(front)
ink = (14, 8, 24, 255)
d.rectangle([0, 0, W, 16], fill=ink)                      # ceiling pipe
for fx in (330, 760, 1130):
    d.rectangle([fx, 0, fx + 22, 30], fill=ink)
d.line([(660, 16), (676, 110), (650, 190)], fill=ink, width=5)   # a dangling cable
d.ellipse([638, 186, 662, 214], fill=ink)
f = np.asarray(front).astype(float)
lum = f[..., :3].mean(2, keepdims=True)
dark = np.clip(np.array([14, 9, 26]) + lum * 0.16 * np.array([1.0, 0.7, 1.2]), 0, 255)
f[..., :3] = np.where(f[..., 3:] > 0, dark, 0)
fq = quantize(f.astype(np.uint8), n=8)
Image.fromarray(fq, "RGBA").save(out / "front.png", optimize=True)

write_sources(out / "sources.json", {**src, "board": {"outer_rect": BOARD, "frame": FRAME,
              "inner_rect": [bx + FRAME, by + FRAME, bw - 2 * FRAME, bh - 2 * FRAME], "floor_line_y": FLOOR}})
print("ok")

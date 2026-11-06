"""Props for one non-New-York city: python backgrounds/props/city_props.py CITY ASSETS_BACKGROUNDS
platform is a hand-built tile (clean against light backgrounds); barricade, barrels and door are keyed from the
generated seeds picked in PICKS below and fitted to the stage grid. The barrels are recoloured per city."""
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from forge.color import srgb_to_oklab
from forge.compose import oklab_to_srgb
from forge.generate import DATA
from forge.layers import key_alpha, quantize, write_sources

city, out = sys.argv[1], Path(sys.argv[2]) / sys.argv[1]
BASE = {"newyork": "props/second"}.get(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
# seed picks; barricade crop = fraction of the keyed height kept from the bottom (drops tall signs)
PICKS = {
    "newyork": dict(barricade=(1, 1.0), door=3, barrel=1, rot=None, tone=((206, 196, 214), (150, 140, 164), (118, 108, 134), (46, 34, 62), (156, 96, 78), (84, 74, 104)), style="truss"),
    "chicago": dict(barricade=(1, 0.42), door=4, barrel=2, rot=250, tone=((214, 224, 238), (150, 166, 190), (112, 126, 152), (34, 40, 66), (150, 98, 84), (84, 94, 120)), style="truss"),
    "dc": dict(barricade=(4, 0.6), door=3, barrel=2, rot=200, tone=((246, 240, 222), (214, 206, 186), (170, 164, 146), (52, 50, 56), (96, 138, 84), (190, 184, 166)), style="ledge"),
    "naples": dict(barricade=(1, 1.0), door=2, barrel=2, rot=60, tone=((200, 232, 218), (70, 150, 136), (44, 104, 98), (36, 40, 52), (196, 120, 80), (92, 62, 48)), style="plank"),
    "california": dict(barricade=(3, 0.5), door=3, barrel=2, rot=15, tone=((236, 216, 178), (200, 176, 134), (150, 124, 92), (66, 46, 50), (120, 178, 176), (116, 94, 76)), style="plank"),
}[city]
RUN = DATA / "runs" / (BASE or f"{city}/props") / "klein"
src = {"city": city}


def get(shot, seed, shadow=True):
    src[shot] = seed
    rgb = np.asarray(Image.open(RUN / shot / f"{seed}.png").convert("RGB"))
    a = key_alpha(rgb)
    if shot.startswith("barrel"):
        # the drum's lit side is lilac, close enough to the key to be eaten by key_alpha: key by distance to the
        # border colour instead, flooding in from the border only
        lab0 = srgb_to_oklab(rgb)
        border = np.concatenate([lab0[0], lab0[-1], lab0[:, 0], lab0[:, -1]])
        d = np.linalg.norm(lab0 - np.median(border, axis=0), axis=-1)
        lbl, _ = ndimage.label(d < 0.1)
        edge = np.unique(np.r_[lbl[0], lbl[-1], lbl[:, 0], lbl[:, -1]])
        a = ~np.isin(lbl, edge[edge > 0])
        a = ndimage.binary_fill_holes(a)
    if shadow:
        lab = srgb_to_oklab(rgb)
        ch = np.hypot(lab[..., 1], lab[..., 2])
        hue = np.degrees(np.arctan2(lab[..., 2], lab[..., 1])) % 360
        kill = (hue > 290) & (hue < 350) & (ch > 0.10) & (lab[..., 0] < 0.55)
        if shot.startswith("barrel"):   # the drum itself is purple: only the ground shadow below its base goes
            ys0 = np.where(a)[0]
            y0, y1 = ys0.min(), ys0.max()
            mid_rows = a[int(y0 + 0.45 * (y1 - y0)):int(y0 + 0.7 * (y1 - y0))]
            xr = np.where(mid_rows.any(0))[0].max()
            kill[:, : xr + 2] = False   # inside the drum's own width nothing is shadow
        a &= ~kill
    if shot == "barrel_out":
        l3 = srgb_to_oklab(rgb)
        h3 = np.degrees(np.arctan2(l3[..., 2], l3[..., 1])) % 360
        a &= ~(((h3 > 300) | (h3 < 30)) & (np.hypot(l3[..., 1], l3[..., 2]) > 0.06) & (l3[..., 0] > 0.62))
        a = ndimage.binary_opening(a, iterations=2)
    lab, n = ndimage.label(a)
    if n > 1:
        sizes = ndimage.sum(a, lab, range(1, n + 1))
        a = np.isin(lab, 1 + np.flatnonzero(sizes > (0.04 if shot.startswith('barrel') else 0.01) * sizes.max()))
    if shot.startswith("barrel"):
        # The drum is a vertical cylinder: below its rim nothing may stick out past its own width. This removes the
        # ground-shadow wedge whatever its colour. Above the rim (flames, smoke) pixels are desaturated to
        # flame/grey so no key-colour tint survives in smoke.
        ys0, xs0 = np.where(a)
        y0, y1 = ys0.min(), ys0.max() + 1
        wid = np.array([a[y].sum() for y in range(a.shape[0])])
        body = np.where(wid >= 0.8 * np.median(wid[int(y0 + 0.5 * (y1 - y0)):y1 - 3]))[0]
        rim = int(body.min())
        cols = np.where(a[rim + 4:int(rim + 0.5 * (y1 - rim))].any(0))[0]   # upper drum: no shadow there
        rows_ = a[rim + 4:int(rim + 0.5 * (y1 - rim))]
        xl = int(np.median([np.where(r)[0].min() for r in rows_ if r.any()]))      # median edges: a bung handle or
        xr = int(np.median([np.where(r)[0].max() for r in rows_ if r.any()]))      # chime ring must not widen the drum
        a[rim:, :xl - 1] = False
        a[rim:, xr + 2:] = False
        # the bottom rim is an ellipse: drop anything left of/right of it that is not drum-coloured shadow
        # the base is an ellipse, so the two corners below it can only be ground shadow: clip to it
        cx, rx = (xl + xr) / 2, (xr - xl) / 2 - 0.5
        ry = 0.5 * (xr - xl) * 0.4            # the drum's bottom ellipse is about a third as tall as it is wide
        ycen = y1 - ry
        yy, xx = np.mgrid[0:a.shape[0], 0:a.shape[1]]
        outside = (((xx - cx) / rx) ** 2 + ((yy - ycen) / ry) ** 2 > 1.0) & (yy > ycen - 2)
        a &= ~outside
        a[y1 - 6:, :] &= a[y1 - 7:y1 - 6, :]   # the bottom rows may not be wider than the row above them
        lab1 = srgb_to_oklab(rgb)
        hue1 = np.degrees(np.arctan2(lab1[..., 2], lab1[..., 1])) % 360
        ch1 = np.hypot(lab1[..., 1], lab1[..., 2])
        above = np.zeros(a.shape, bool); above[:rim] = True
        tint = a & above & (ch1 > 0.02) & ~((hue1 > 20) & (hue1 < 115))
        g = (lab1[..., 0] * 255 * 0.9 + 20).clip(0, 255).astype(np.uint8)
        rgb = rgb.copy(); rgb[tint] = np.stack([g, g, g], -1)[tint]
    ys, xs = np.where(a)
    return Image.fromarray(np.dstack([rgb, a * 255]).astype(np.uint8), "RGBA").crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))


def fit(img, w, h, n=40, post=None):
    s = min(w / img.width, h / img.height)
    im = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.LANCZOS)
    if post:
        im = Image.fromarray(post(np.asarray(im).copy()), "RGBA")
    canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    canvas.alpha_composite(im, ((w - im.width) // 2, h - im.height))
    return Image.fromarray(quantize(np.asarray(canvas), n=n), "RGBA")


def platform_tile(style, tones, h=28, w=64):
    top, plate, plate_dk, line, accent, under = tones
    t = np.zeros((h, w, 4), np.uint8)
    t[..., 3] = 255
    t[:, :, :3] = under
    t[0, :, :3] = line
    t[1:3, :, :3] = top                      # walkable surface at the very top
    t[3:11, :, :3] = plate
    t[9:11, :, :3] = plate_dk
    t[11, :, :3] = line
    if style == "truss":
        for x in range(4, w, 16):
            t[5, x:x + 2, :3] = top; t[6, x:x + 2, :3] = plate_dk
        for x in range(w):
            y = 13 + int(abs((x % 16) - 8) * 1.6)
            t[y:y + 2, x, :3] = line
    elif style == "ledge":                    # marble ledge: cornice, then a plain block face with joints
        t[11:20, :, :3] = plate
        t[11, :, :3] = line
        t[19:21, :, :3] = plate_dk
        t[21, :, :3] = line
        for x in (0, 32):
            t[3:21, x, :3] = plate_dk
        t[21:, :, :3] = under
        t[h - 1, :, :3] = line
    else:                                     # planks: board seams every 16 px, dark joists below
        for x in range(0, w, 16):
            t[3:11, x, :3] = line
        for x in range(0, w, 32):
            t[12:h - 2, x:x + 4, :3] = plate_dk
            t[12:h - 2, x, :3] = line
    t[h - 3:h - 1, :, :3] = plate_dk
    t[h - 1, :, :3] = line
    for x, y in [(9, 7), (10, 7), (33, 8), (34, 8), (52, 6), (21, 17), (47, 18)]:
        t[y, x, :3] = accent
    return Image.fromarray(t, "RGBA")


def save(img, name):
    img.save(out / name, optimize=True)


save(platform_tile(PICKS["style"], PICKS["tone"]), "platform.png")

seed, keep = PICKS["barricade"]
b = get("barricade", seed)
if keep < 1:
    b = b.crop((0, round(b.height * (1 - keep)), b.width, b.height))
want = round(b.height * 0.8)
if b.width > want:
    x0 = (b.width - want) // 2
    b = b.crop((x0, 0, x0 + want, b.height))


def grime(a):
    h, w = a.shape[:2]
    rng = np.random.default_rng(3)
    k = 1 - 0.3 * (np.arange(h)[:, None] / h) ** 2
    a[..., :3] = (a[..., :3] * (k * np.where(rng.random((h, w)) < 0.06, 0.75, 1.0))[..., None]).astype(np.uint8)
    return a


save(fit(b, 80, 100, 32, grime), "barricade.png")


def rotate_body(a, hue_deg):
    """Give the drum's purple/maroon body this city's hue (flames are left alone); a touch brighter to spot it."""
    lab = srgb_to_oklab(a[..., :3])
    hue = np.degrees(np.arctan2(lab[..., 2], lab[..., 1])) % 360
    ch = np.hypot(lab[..., 1], lab[..., 2])
    sel = (a[..., 3] > 0) & ((hue > 270) | (hue < 12)) & (ch < 0.25)
    new = lab.copy()
    r = np.radians(hue_deg)
    c = np.maximum(ch, 0.07)
    new[..., 1], new[..., 2] = c * np.cos(r), c * np.sin(r)
    new[..., 0] = np.clip(lab[..., 0] * 1.0 + 0.02, 0, 1)
    res = a.copy()
    res[sel, :3] = oklab_to_srgb(new)[sel]
    return res


save(fit(get("barrel", PICKS["barrel"]), 60, 110, 40, (lambda a: rotate_body(a, PICKS["rot"])) if PICKS["rot"] is not None else None), "barrel.png")
save(fit(get("barrel_out", PICKS["barrel"]), 60, 110, 32), "barrel-out.png")
save(fit(get("door", PICKS["door"], shadow=False), 112, 165, 44), "door.png")
write_sources(out / "props-sources.json", src)
print("ok", city)

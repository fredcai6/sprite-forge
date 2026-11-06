"""Frames straight from whole generated poses, for characters that change shape between frames.

    python -m forge.compose palette NAME          build characters/NAME/palette.json from [palette] sources
    python -m forge.compose anim characters/NAME/anims/X.toml

Each frame names a generated pose (SHOT/SEED in the run), keys it out, places it on the canvas by its
ground contact and horizontal centroid, and samples it onto the pixel grid in the character's palette.
`cell` is source pixels per sprite pixel, the same for every frame of a character, so a pixel is the
same size everywhere. The atlas records the run, model, shot and seed of every frame.
"""
import glob, json, sys, tomllib
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from .color import build_palette, load_palette, srgb_to_oklab
from .generate import DATA, ROOT, rev
from .pixelize import despeckle, to_sprite, upscale


def keyout(rgb):
    """Foreground mask for art on a magenta-ish key.

    Background is (1) any strong key-colour pixel anywhere, even trapped between limbs, plus (2) the
    key-tinted gradient and the generated ground shadow, but only where they are connected to the
    image border. Dark pixels of the art that merely resemble the shadow (a turtle's shaded shell)
    are therefore never removed. Enclosed regions that are not key colour are filled, so a sprite has
    no holes. The largest piece is kept."""
    r, g, b = [rgb[..., i].astype(int) for i in range(3)]
    lab3 = srgb_to_oklab(rgb)
    chroma = np.hypot(lab3[..., 1], lab3[..., 2])
    hue = np.degrees(np.arctan2(lab3[..., 2], lab3[..., 1])) % 360
    strict = (np.minimum(r, b) - g > 75) & (r > 150) & (b > 85)
    loose = (np.minimum(r, b) - g > 38) & (r + b > 150)
    stray = (hue > 300) & (hue < 352) & (chroma > 0.12)
    shadow = (hue > 290) & (hue < 350) & (chroma > 0.065) & (lab3[..., 0] < 0.6)
    cand = loose | strict | stray | shadow
    lab, _ = ndimage.label(cand)
    edge = np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    bg = np.isin(lab, edge[edge > 0]) | strict
    fg = ndimage.binary_opening(~bg, iterations=1)
    # interior holes: fill unless they are mostly key colour
    holes, n = ndimage.label(~fg)
    border = np.unique(np.r_[holes[0], holes[-1], holes[:, 0], holes[:, -1]])
    for i in range(1, n + 1):
        if i in border:
            continue
        m = holes == i
        if strict[m].mean() < 0.5:
            fg |= m
    lab, n = ndimage.label(fg)
    if n == 0:
        return fg
    sizes = ndimage.sum(fg, lab, range(1, n + 1))
    return np.isin(lab, 1 + np.flatnonzero(sizes > 0.02 * sizes.max()))


def pose_path(run_dir, pose):
    """`SHOT/SEED` in the anim's own run, or `RUN/MODEL/SHOT/SEED` to take a pose from another run."""
    parts = pose.split("/")
    if len(parts) == 4:
        return run_dir.parents[1] / parts[0] / parts[1] / parts[2] / f"{parts[3]}.png"
    return run_dir / parts[0] / f"{parts[1]}.png"


def squeeze(rgb, alpha, cx, f, legs_y, blend=50):
    """Narrow the body above the legs to `f` of its width about column cx, the way a folded slice is
    narrow. Rows ease from the full squeeze to none across `blend` rows at the legs so the feet keep
    their stance."""
    h, w = alpha.shape
    out_rgb, out_a = np.zeros_like(rgb), np.zeros_like(alpha)
    xs = np.arange(w)
    for y in range(h):
        t = 1.0 if y < legs_y else max(0.0, 1 - (y - legs_y) / blend)
        ff = 1 - (1 - f) * t
        src_x = np.clip(np.round(cx + (xs - cx) / ff).astype(int), 0, w - 1)
        out_rgb[y], out_a[y] = rgb[y, src_x], alpha[y, src_x]
        out_a[y] &= (cx + (xs - cx) / ff >= 0) & (cx + (xs - cx) / ff < w)
    return out_rgb, out_a


def build(name):
    char_dir = ROOT / "characters" / name
    cfg = tomllib.loads((char_dir / "character.toml").read_text())
    px = []
    for pat in cfg["palette"]["sources"]:
        for f in sorted(glob.glob(str(DATA / "runs" / name / pat))):
            a = np.asarray(Image.open(f).convert("RGB"))
            px.append(a[keyout(a)])
    fams = {k: v for k, v in cfg["palette"]["families"].items()}
    pal = build_palette(np.concatenate(px), fams)
    (char_dir / "palette.json").write_text(json.dumps(pal, indent=1))
    print({k: len(v) for k, v in pal.items()})


def thicken(sprite, ow, colour):
    """Grow the outer silhouette by `ow` pixels in the outline colour (round brush)."""
    a = sprite[..., 3] > 0
    r = ow
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    brush = xx * xx + yy * yy <= r * r + 1
    grown = ndimage.binary_dilation(a, structure=brush)
    out = sprite.copy()
    new = grown & ~a
    out[new, :3] = colour
    out[new, 3] = 255
    return out


def frame_for(spec, pal, run_dir, f, accent, outline=0):
    w, h = spec["canvas"]
    cell = f.get("cell", spec["cell"]) * f.get("scale", 1.0)
    cell *= 1 + 2 * outline / (h * 0.95)   # the outline is added on top, so paint the figure smaller to keep the frame
    rgb = np.asarray(Image.open(pose_path(run_dir, f["pose"])).convert("RGB"))
    alpha = keyout(rgb)
    if f.get("flip"):
        rgb, alpha = rgb[:, ::-1], alpha[:, ::-1]
    ys, xs = np.where(alpha)
    if "squeeze" in f:
        rgb, alpha = squeeze(rgb, alpha, xs.mean(), f["squeeze"], f.get("legs_y", 700))
        ys, xs = np.where(alpha)
    cx = xs.mean() + f.get("cx", 0) * cell
    bottom = ys.max() + 1
    ox = cx - (w / 2 + f.get("dx", 0)) * cell
    oy = bottom - (h - spec.get("margin_bottom", 2 + outline) - f.get("dy", 0)) * cell
    sprite = to_sprite(rgb, alpha, pal[0], pal[1], cell, (ox, oy), (w, h), accent)
    if outline:
        dark = pal[0][np.argmin(pal[1][:, 0])]
        sprite = thicken(despeckle(sprite), outline, dark)
    return sprite


def anim(path):
    path = Path(path)
    spec = tomllib.loads(path.read_text())
    char_dir = path.parent.parent
    cfg = tomllib.loads((char_dir / "character.toml").read_text())
    run_dir = DATA / "runs" / char_dir.name / spec["run"] / spec["model"]
    pal = load_palette(char_dir / "palette.json")
    frames, used, cache = [], [], {}
    for f in spec["frames"]:
        key = json.dumps(f, sort_keys=True)
        if key not in cache:
            cache[key] = despeckle(frame_for(spec, pal, run_dir, f, cfg["sprite"]["accent"], cfg["sprite"].get("outline", 0)))
        for _ in range(f.get("hold", 1)):
            frames.append(cache[key])
            used.append(f["pose"])
    out = char_dir / "sprites" / path.stem
    out.mkdir(parents=True, exist_ok=True)
    sheet = np.concatenate(frames, axis=1)
    Image.fromarray(sheet, "RGBA").save(out / "sheet.png", optimize=True)
    w, h = spec["canvas"]
    (out / "atlas.json").write_text(json.dumps({
        "frame": {"w": w, "h": h}, "count": len(frames), "frame_ms": spec["frame_ms"], "loop": spec.get("loop", True),
        "rev": rev(), "source": {"run": spec["run"], "model": spec["model"], "cell": spec["cell"], "poses": used},
    }, indent=1))
    z = spec.get("preview_scale", 3)
    pics = [upscale(fr, z, (40, 34, 52)).convert("RGB") for fr in frames]
    pics[0].save(out / "preview.gif", save_all=True, append_images=pics[1:], duration=spec["frame_ms"], loop=0)
    cols = len(np.unique(sheet[sheet[..., 3] > 0][:, :3], axis=0))
    print(f"{out.name}: {len(frames)} frames {w}x{h}, {cols} colours")


if __name__ == "__main__":
    {"palette": build, "anim": anim}[sys.argv[1]](sys.argv[2])

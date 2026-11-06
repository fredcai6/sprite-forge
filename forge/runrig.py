"""Procedural run and walk cycles: one body cut from the character's own art, limbs drawn per frame.

    python -m forge.runrig characters/NAME/rig/run.toml

Image models redraw one wide stance for every run key, so the legs shuffle without crossing. Here the
body (legs and arms erased) is cut from a finished sprite frame and each frame's limbs are placed by
two-bone IK from ONE shared hip, using the gait tables below (taken from Muybridge, Animal Locomotion
plate 60, and the Williams contact/down/pass/up poses). The far leg and arm are the near ones half a
cycle later, drawn behind the body and darker. Feet and gloves are cropped from the character's art
and rotated on a 4x grid then voted back down, so pixels stay hard and on the palette.

Coordinates: x forward (right), y down, angles clockwise degrees. Lengths in units of the leg/arm
length S. See characters/slice-ny/rig/run.toml for every key.
"""
import json, sys, tomllib
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from .generate import ROOT, rev
from .pixelize import despeckle, upscale

# ---- gait tables, indexed by the LEG's own phase p (the other leg is p+half) ------------------------
# run: stance p0 contact, p1 down, p2 pass, p3 toe-off/flight; swing p4 heel kick, p5 knee drives, p6 pass, p7 reach.
# (ankle forward of hip, sole clearance above ground, foot angle; all in leg-lengths / degrees)
RUN_LEG = [
    (0.46, 0.00, -18),   # 0 contact: foot reaches ahead, heel first
    (0.22, 0.00, 0),     # 1 down: foot flat, knee bent under the weight
    (-0.06, 0.00, 6),    # 2 pass: stance leg under the body, rear heel lifting
    (-0.50, 0.16, 48),   # 3 toe-off / flight: rear leg trails, toe pointing down
    (-0.46, 0.44, 72),   # 4 heel kick: foot snapped up behind
    (-0.12, 0.46, 62),   # 5 knee drives forward, foot still tucked
    (0.12, 0.40, 28),    # 6 pass: knee high in front, foot under it, beside the stance leg
    (0.44, 0.20, -12),   # 7 reach: leg extends forward to land
]
RUN_HIP = [0.90, 0.80, 0.90, 0.99, 0.90, 0.80, 0.90, 0.99]    # hip height above ground per frame (down lowest)
# arm hand target relative to the shoulder (x, y) in arm-lengths, by the same-side LEG phase: back when that leg is forward
RUN_ARM = [(-0.45, 0.72), (-0.30, 0.82), (0.0, 0.88), (0.35, 0.78), (0.62, 0.50), (0.40, 0.74), (0.02, 0.88), (-0.30, 0.80)]

# walk: 8 frames, always one foot down. p0 heel strike, p1 loading, p2 mid-stance, p3 push-off, p4 toe-off, p5 swing back, p6 swing pass, p7 reach.
WALK_LEG = [
    (0.34, 0.00, -16),   # 0 heel strike
    (0.20, 0.00, 0),     # 1 loading, foot flat
    (0.00, 0.00, 0),     # 2 mid-stance, leg straight under the hip
    (-0.22, 0.00, 22),   # 3 heel rises, rear leg pushes
    (-0.34, 0.05, 48),   # 4 toe-off, toe just dragging
    (-0.12, 0.20, 40),   # 5 swing leg folds and comes through
    (0.10, 0.18, 18),    # 6 pass: knee forward beside the stance leg
    (0.30, 0.08, -8),    # 7 reach
]
WALK_HIP = [0.96, 0.93, 0.98, 0.95, 0.93, 0.95, 0.98, 0.96]
WALK_ARM = [(-0.34, 0.50), (-0.24, 0.55), (-0.02, 0.60), (0.22, 0.56), (0.34, 0.50), (0.22, 0.56), (-0.02, 0.60), (-0.24, 0.55)]


def ik(root, target, l1, l2, bend):
    """Two-bone chain from root to target; the joint sits to the `bend` side (+1: forward/right of the root-target line normal)."""
    d = np.array(target, float) - root
    dist = np.hypot(*d)
    dist = min(dist, l1 + l2 - 1e-3)
    if dist < 1e-6:
        d, dist = np.array([0.0, 1.0]), 1e-6
    u = d / np.hypot(*d)
    a = (l1 * l1 - l2 * l2 + dist * dist) / (2 * dist)
    h = np.sqrt(max(l1 * l1 - a * a, 0))
    n = np.array([-u[1], u[0]])          # clockwise-perpendicular in image coords
    if bend > 0:                          # knee: forward of the line
        if n[0] < 0:
            n = -n
    else:                                 # elbow: the candidate that sits further back, lower on a tie
        c1, c2 = np.array(root, float) + u * a + n * h, np.array(root, float) + u * a - n * h
        if (c2[0] - 0.3 * c2[1]) < (c1[0] - 0.3 * c1[1]):
            n = -n
    mid = np.array(root, float) + u * a + n * h
    end = np.array(root, float) + u * dist
    return mid, end


def vote_down(rgba, k):
    """Downsample an RGBA image by k: a cell is opaque if at least half is, and takes its most common colour."""
    H, W = rgba.shape[0] // k, rgba.shape[1] // k
    out = np.zeros((H, W, 4), np.uint8)
    a = rgba[..., 3] > 0
    key = (rgba[..., 0].astype(np.int64) << 16) | (rgba[..., 1].astype(np.int64) << 8) | rgba[..., 2]
    for y in range(H):
        for x in range(W):
            ca = a[y * k:(y + 1) * k, x * k:(x + 1) * k]
            if ca.sum() * 2 < k * k:
                continue
            ck = key[y * k:(y + 1) * k, x * k:(x + 1) * k][ca]
            v, c = np.unique(ck, return_counts=True)
            m = int(v[c.argmax()])
            out[y, x] = ((m >> 16) & 255, (m >> 8) & 255, m & 255, 255)
    return out


def rotate_patch(patch, anchor, angle, k=4):
    """Rotate an RGBA patch about `anchor` (patch pixel coords) clockwise by `angle`; hard pixels, same colours.
    Returns (patch, anchor in the new patch)."""
    if abs(angle) < 0.5:
        return patch, np.array(anchor, float)
    h, w = patch.shape[:2]
    pad = int(np.hypot(h, w)) + 2
    big = np.zeros(((h + 2 * pad) * k, (w + 2 * pad) * k, 4), np.uint8)
    up = np.repeat(np.repeat(patch, k, 0), k, 1)
    big[pad * k:pad * k + h * k, pad * k:pad * k + w * k] = up
    im = Image.fromarray(big, "RGBA").rotate(-angle, Image.NEAREST, center=((anchor[0] + pad) * k, (anchor[1] + pad) * k))
    r = np.asarray(im).copy()
    r[..., 3] = np.where(r[..., 3] > 127, 255, 0)
    out = vote_down(r, k)
    ys, xs = np.nonzero(out[..., 3])
    y0, x0 = ys.min(), xs.min()
    return out[y0:ys.max() + 1, x0:xs.max() + 1], np.array([anchor[0] + pad - x0, anchor[1] + pad - y0], float)


def brush(r):
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return xx * xx + yy * yy <= r * r + 1


def grow(mask, r):
    return ndimage.binary_dilation(mask, structure=brush(r)) if r > 0 else mask.copy()


def capsule(shape, pts, width):
    """Thick polyline mask (round joints), pixel-centre distance <= width/2."""
    H, W = shape
    yy, xx = np.mgrid[0:H, 0:W]
    yy, xx = yy + 0.5, xx + 0.5
    m = np.zeros(shape, bool)
    for a, b in zip(pts[:-1], pts[1:]):
        a, b = np.array(a, float), np.array(b, float)
        ab = b - a
        t = np.clip(((xx - a[0]) * ab[0] + (yy - a[1]) * ab[1]) / max(ab @ ab, 1e-9), 0, 1)
        d = np.hypot(xx - (a[0] + t * ab[0]), yy - (a[1] + t * ab[1]))
        m |= d <= width / 2
    return m


def shade(mask, cols):
    """Tube shading on the pixel grid: light edge top-left, dark edge bottom-right."""
    light, mid, dark = [np.array(c, np.uint8) for c in cols]
    out = np.zeros(mask.shape + (4,), np.uint8)
    out[mask, :3] = mid
    hi = mask & ~np.roll(np.roll(mask, 1, 0), 1, 1)
    lo = mask & ~np.roll(np.roll(mask, -1, 0), -1, 1)
    out[hi, :3] = light
    out[lo, :3] = dark
    out[mask, 3] = 255
    return out


def darken(rgb, f, tint=(42, 24, 48)):
    """Far-side limbs: toward the violet shadow colour."""
    return (np.asarray(rgb, float) * f + np.array(tint, float) * (1 - f)).round().astype(np.uint8)


def over(dst, src, dx, dy):
    """Composite an RGBA src onto dst at integer offset (hard alpha)."""
    h, w = src.shape[:2]
    H, W = dst.shape[:2]
    x0, y0 = max(dx, 0), max(dy, 0)
    x1, y1 = min(dx + w, W), min(dy + h, H)
    if x1 <= x0 or y1 <= y0:
        return
    s = src[y0 - dy:y1 - dy, x0 - dx:x1 - dx]
    m = s[..., 3] > 0
    d = dst[y0:y1, x0:x1]
    d[m] = s[m]


def peel(sprite, dark, n, tol=60):
    """Remove the dark outer outline: dark pixels connected to the outside, up to n layers."""
    a = sprite[..., 3] > 0
    isdark = np.abs(sprite[..., :3].astype(int) - np.array(dark)).sum(-1) < tol
    for _ in range(n):
        ext = ~ndimage.binary_dilation(a, structure=ndimage.generate_binary_structure(2, 1), border_value=0) | ~a
        edge = a & ndimage.binary_dilation(~a, structure=ndimage.generate_binary_structure(2, 1))
        rm = edge & isdark
        a = a & ~rm
    out = sprite.copy()
    out[~a] = 0
    return out


def crop_part(spec, base):
    """A foot or glove: cut from a sprite frame, outline peeled, returned with its anchor."""
    sheet = Image.open(ROOT / spec["src"]).convert("RGBA")
    fw = spec["frame_w"]
    f = np.asarray(sheet)[:, spec["frame"] * fw:(spec["frame"] + 1) * fw]
    x0, y0, x1, y1 = spec["box"]
    patch = np.zeros_like(f)
    patch[y0:y1, x0:x1] = f[y0:y1, x0:x1]
    patch = peel(patch, base["dark"], base["ow_src"])
    lab, n = ndimage.label(patch[..., 3] > 0)
    if n > 1:
        sizes = ndimage.sum(patch[..., 3] > 0, lab, range(1, n + 1))
        patch[lab != 1 + int(np.argmax(sizes))] = 0
    ys, xs = np.nonzero(patch[..., 3])
    py0, px0 = ys.min(), xs.min()
    patch = patch[py0:ys.max() + 1, px0:xs.max() + 1]
    anchor = (spec["anchor"][0] - px0, spec["anchor"][1] - py0)
    return patch, anchor


class Rig:
    def __init__(self, path):
        self.path = Path(path)
        self.cfg = tomllib.loads(self.path.read_text())
        c = self.cfg
        self.W, self.Hc = c["canvas"]
        self.dark = tuple(c["dark"])
        base = {"dark": self.dark, "ow_src": c.get("src_outline", 4)}
        self.ow_out = c.get("outline", 4)
        self.ow_in = c.get("inner_outline", 2)
        b = c["body"]
        sheet = np.asarray(Image.open(ROOT / b["src"]).convert("RGBA"))
        fw = b["frame_w"]
        body = sheet[:, b["frame"] * fw:(b["frame"] + 1) * fw].copy()
        keep = np.ones(body.shape[:2], bool)
        from PIL import ImageDraw
        for poly in b.get("erase", []):
            m = Image.new("L", (body.shape[1], body.shape[0]), 0)
            ImageDraw.Draw(m).polygon([tuple(p) for p in poly], fill=255)
            keep &= ~(np.asarray(m) > 0)
        body[~keep] = 0
        # holes left where a limb covered the body are filled from the nearest body pixel
        for fill in b.get("fill", []):
            x0, y0, x1, y1 = fill
            region = np.zeros(body.shape[:2], bool)
            region[y0:y1, x0:x1] = True
            have = body[..., 3] > 0
            hole = region & ~have
            idx = ndimage.distance_transform_edt(~have, return_distances=False, return_indices=True)
            body[hole] = body[idx[0][hole], idx[1][hole]]
        body = peel(body, self.dark, base["ow_src"])
        self.body = despeckle(body)
        self.limb_cols = {k: [tuple(x) for x in v] for k, v in c["limb"].items()}
        self.foot, self.foot_anchor = crop_part(c["foot"], base)
        self.hand, self.hand_anchor = crop_part(c["hand"], base)
        self.base = base
        self._lean = {}

    def pose_frames(self):
        c = self.cfg
        g = c["gait"]
        kind = g["kind"]
        leg_t = RUN_LEG if kind == "run" else WALK_LEG
        hip_t = RUN_HIP if kind == "run" else WALK_HIP
        arm_t = RUN_ARM if kind == "run" else WALK_ARM
        n = len(leg_t)
        return kind, leg_t, hip_t, arm_t, n

    def render(self, t):
        c, g = self.cfg, self.cfg["gait"]
        kind, leg_t, hip_t, arm_t, n = self.pose_frames()
        S = g["leg"]
        ground = g["ground"]
        ankle_h = self.foot.shape[0] - self.foot_anchor[1]
        bob = g.get("bob", 1.0)
        hipx = g["hip"][0]
        hmean = float(np.mean(hip_t))
        H = hmean + (hip_t[t % n] - hmean) * bob
        hip_y = ground - ankle_h - H * S
        dy = int(round(hip_y - g["hip"][1]))
        # leg tables may be overridden per character (stride scale, clearance scale)
        stride, clear_s = g.get("stride", 1.0), g.get("clear", 1.0)
        W, Hc = self.W, self.Hc
        layers = []   # (name, rgba full-canvas, mask)

        def leg(phase, far):
            ax, cl, ang = leg_t[phase % n]
            ax *= stride
            cl *= clear_s
            hip = np.array([hipx, hip_y])
            if g.get("lean_foot", 0):
                ang += g["lean_foot"]
            foot, fa = rotate_patch(self.foot, self.foot_anchor, ang)
            ah = foot.shape[0] - fa[1]          # ankle to lowest foot pixel at this angle
            target = np.array([hipx + ax * S, ground - ah - cl * S])
            knee, ank = ik(hip, target, g["thigh"] * S, (1 - g["thigh"]) * S, +1)
            mask = capsule((Hc, W), [hip, knee, ank], g["leg_w"])
            cols = self.limb_cols["leg"]
            if far:
                cols = [tuple(darken(x, g.get("far_dark", 0.62))) for x in cols]
            tube = shade(mask, cols)
            if far:
                foot = foot.copy()
                m = foot[..., 3] > 0
                foot[m, :3] = darken(foot[m, :3], g.get("far_dark_foot", 0.72))
            lay = tube.copy()
            over(lay, foot, int(round(ank[0] - fa[0])), int(round(ank[1] - fa[1])))
            return lay

        def arm(phase, far):
            hx, hy = arm_t[phase % n]
            sh = np.array([c["arm"]["shoulder_far" if far else "shoulder_near"][0], c["arm"]["shoulder_far" if far else "shoulder_near"][1] + dy], float)
            A = c["arm"]["len"]
            target = sh + np.array([hx * A * c["arm"].get("swing", 1.0), hy * A])
            elbow, hand = ik(sh, target, A * 0.5, A * 0.5, -1)
            mask = capsule((Hc, W), [sh, elbow, hand], c["arm"]["w"])
            cols = self.limb_cols["arm"]
            if far:
                cols = [tuple(darken(x, 0.62)) for x in cols]
            tube = shade(mask, cols)
            fo = hand - elbow
            hp, ha = rotate_patch(self.hand, self.hand_anchor, -float(np.degrees(np.arctan2(fo[0], fo[1]))) + c["arm"].get("hand_angle", 0))
            if far:
                hp = hp.copy()
                m = hp[..., 3] > 0
                hp[m, :3] = darken(hp[m, :3], 0.72)
            lay = tube.copy()
            over(lay, hp, int(round(hand[0] - ha[0])), int(round(hand[1] - ha[1])))
            return lay

        half = n // 2
        near_leg = leg(t, False)
        far_leg = leg(t + half, True)
        near_arm = arm(t + half, False)
        far_arm = arm(t, True)
        body = np.zeros((Hc, W, 4), np.uint8)
        lean = g.get("lean", 0) + g.get("lean_down", 0) * (1 if kind == "run" and (t % n) in (1, 5) else 0)
        if lean:
            key = round(lean * 2) / 2
            if key not in self._lean:
                self._lean[key] = rotate_patch(self.body, g["hip"], key)
            bp, ba = self._lean[key]
            over(body, bp, int(round(hipx - ba[0])) + int(g.get("body_dx", 0)), int(round(hip_y - ba[1])))
        else:
            over(body, self.body, int(g.get("body_dx", 0)), dy)
        order = [far_arm, far_leg, body, near_leg, near_arm]
        out = np.zeros((Hc, W, 4), np.uint8)
        dark = np.array(self.dark, np.uint8)
        for lay in order:
            m = lay[..., 3] > 0
            ring = grow(m, self.ow_in) & ~m
            out[ring, :3] = dark
            out[ring, 3] = 255
            over(out, lay, 0, 0)
        occ = out[..., 3] > 0
        extra = grow(occ, max(self.ow_out - self.ow_in, 0)) & ~occ
        out[extra, :3] = dark
        out[extra, 3] = 255
        return despeckle(out)

    def build(self):
        kind, _, _, _, n = self.pose_frames()
        frames = [self.render(t) for t in range(n)]
        return frames


def main(path):
    path = Path(path).resolve()
    rig = Rig(path)
    cfg = rig.cfg
    frames = rig.build()
    out = ROOT / "characters" / path.parent.parent.name / "sprites" / cfg["anim"]
    out.mkdir(parents=True, exist_ok=True)
    sheet = np.concatenate(frames, axis=1)
    assert set(np.unique(sheet[..., 3])) <= {0, 255}
    Image.fromarray(sheet, "RGBA").save(out / "sheet.png", optimize=True)
    w, h = cfg["canvas"]
    (out / "atlas.json").write_text(json.dumps({
        "frame": {"w": w, "h": h}, "count": len(frames), "frame_ms": cfg["frame_ms"], "loop": True,
        "rev": rev(), "source": {"rig": str(path.relative_to(ROOT)), "body": cfg["body"]["src"], "gait": cfg["gait"]["kind"]},
    }, indent=1))
    pics = [upscale(f, 3, (40, 34, 52)).convert("RGB") for f in frames]
    pics[0].save(out / "preview.gif", save_all=True, append_images=pics[1:], duration=cfg["frame_ms"], loop=0)
    print(f"{out.parent.parent.name}/{cfg['anim']}: {len(frames)} frames {w}x{h}")


if __name__ == "__main__":
    main(sys.argv[1])

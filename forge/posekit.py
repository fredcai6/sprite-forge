"""Pose source for guided generation: a side-view run cycle as joint angles, drawn three ways.

    python -m forge.posekit CHARACTER MOTION OUT_DIR   e.g. slice-ny run /tmp/pose

Per frame (8 per cycle) the table gives, for the near and far leg and arm, angles in degrees measured from
straight down, positive toward the direction of travel (right): hip (thigh), knee flexion (the shin turns back
by this much), foot (toe-down positive), shoulder (upper arm), elbow flexion (the forearm turns forward by
this much). Both legs hang from ONE hip point; the far limbs are the near ones half a cycle later, so the
legs cross at the passing frames and both leave the ground in the flight frames (4 and 0 of the cycle).
Opposite arm and leg swing together.

The numbers come from `forge/runrig.py` (gait tables taken from Muybridge, Animal Locomotion plate 60, and the
Williams contact/down/pass/up poses): its ankle and hand targets are solved here with the same two-bone IK
into angles, so the guided tracks and the code rig draw one gait. `RUN_LEG` etc. below are copies of those tables.

Renders (all return PIL images, `size` square, figure centred, black on white except `openpose`):
    openpose(frame)    coloured stick figure, OpenPose body-18 colours, black background
    mannequin(frame)   thick black limbs (and feet, hands) on white; torso=True adds body and head
    lines(frame)       black-on-white outline drawing of the whole figure, parts occluding each other
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# ---- motion tables live in forge/motions/*.toml; body plans in characters/NAME/pose.toml ---------
import tomllib
ROOT = Path(__file__).resolve().parent.parent


def ik(root, target, l1, l2, bend):
    """Two-bone chain root->target. bend=+1: joint forward of the line (knee); -1: joint further back (elbow)."""
    root = np.asarray(root, float)
    d = np.array(target, float) - root
    dist = min(np.hypot(*d), l1 + l2 - 1e-3)
    u = d / max(np.hypot(*d), 1e-9)
    a = (l1 * l1 - l2 * l2 + dist * dist) / (2 * dist)
    h = np.sqrt(max(l1 * l1 - a * a, 0))
    n = np.array([-u[1], u[0]])
    if bend > 0:
        if n[0] < 0:
            n = -n
    else:
        c1, c2 = root + u * a + n * h, root + u * a - n * h
        if (c2[0] - 0.3 * c2[1]) < (c1[0] - 0.3 * c1[1]):
            n = -n
    return root + u * a + n * h


def _ang(v):
    """Angle of a vector in degrees from straight down, clockwise (toward +x) positive."""
    return float(np.degrees(np.arctan2(v[0], v[1])))


def load_motion(name_or_path):
    p = Path(name_or_path)
    if not p.exists():
        p = ROOT / "forge" / "motions" / f"{name_or_path}.toml"
    return tomllib.loads(p.read_text())


def angle_table(motion, thigh=0.5, arm_swing=1.0):
    """Joint angles per frame for the near (n) and far (f) side. kind "gait" solves the table's ankle and hand
    targets with two-bone IK; kind "angles" is an explicit list of rows ([[frames]] with hip_n .. el_f, hip_h, dx, dy)."""
    if motion["kind"] == "angles":
        return [dict({"hip_h": 0.95}, **r, frame=i) for i, r in enumerate(motion["frames"])]
    n = len(motion["leg"])
    rows = []
    for t in range(n):
        H = motion["hip"][t]
        row = {"frame": t, "hip_h": H, "lean_extra": (motion.get("lean_extra") or [0] * n)[t]}
        for side, lp, ap in (("n", t, t + n // 2), ("f", t + n // 2, t)):
            ax, cl, foot = motion["leg"][lp % n]
            target = np.array([ax, H - cl])
            knee = ik([0, 0], target, thigh, 1 - thigh, +1)
            row[f"hip_{side}"] = round(_ang(knee), 1)
            row[f"knee_{side}"] = round(_ang(knee) - _ang(target - knee), 1)
            row[f"foot_{side}"] = float(foot)
            hx, hy = motion["arm"][ap % n]
            tgt = np.array([hx * arm_swing, hy])
            elbow = ik([0, 0], tgt, 0.5, 0.5, -1)
            row[f"sh_{side}"] = round(_ang(elbow), 1)
            row[f"el_{side}"] = round(_ang(tgt - elbow) - _ang(elbow), 1)
        rows.append(row)
    return rows


# OpenPose body-18 limb colours (BGR order in the original code; these are RGB)
OP_COLORS = [(255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0), (0, 255, 0),
             (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255),
             (170, 0, 255), (255, 0, 255), (255, 0, 170), (255, 0, 85)]


def rot(v, deg):
    a = np.radians(deg)
    return np.array([v[0] * np.cos(a) - v[1] * np.sin(a), v[0] * np.sin(a) + v[1] * np.cos(a)])


def down(deg):
    """Unit vector pointing `deg` from straight down, positive toward +x (image y is down)."""
    a = np.radians(deg)
    return np.array([np.sin(a), np.cos(a)])


class Figure:
    """A body plan (characters/NAME/pose.toml, or a dict) driven by a motion table (name, path or dict)."""

    def __init__(self, rig, motion="run"):
        if isinstance(rig, (str, Path)):
            r = Path(rig)
            if not r.exists():
                r = ROOT / "characters" / str(rig) / "pose.toml"
            rig = tomllib.loads(r.read_text())
        self.p = rig
        self.motion = load_motion(motion) if isinstance(motion, (str, Path)) else motion
        self.table = angle_table(self.motion, rig["thigh"], rig.get("arm_swing", 1.0))
        self.n = len(self.table)

    def joints(self, t):
        """All joint positions in sprite pixels for frame t (a dict of name -> np.array xy), plus part data."""
        p, row = self.p, self.table[t % self.n]
        S, A, th = p["leg"], p["arm"], p["thigh"]
        hip = np.array([p["hip_x"] + row.get("dx", 0), p["ground"] - p["ankle_h"] - row["hip_h"] * S - row.get("dy", 0)])
        lean = p["lean"] + row.get("lean_extra", 0) + row.get("lean", 0)
        j = {"hip": hip}
        for s in "nf":
            kn = hip + down(row[f"hip_{s}"]) * th * S
            an = kn + down(row[f"hip_{s}"] - row[f"knee_{s}"]) * (1 - th) * S
            f_ang = row[f"foot_{s}"]
            # foot: heel behind the ankle, toe ahead; toe-down angle rotates the foot clockwise
            toe = an + rot(np.array([p["foot"][0] * 0.75, p["foot"][1] * 0.35]), f_ang)
            heel = an + rot(np.array([-p["foot"][0] * 0.25, p["foot"][1] * 0.35]), f_ang)
            j.update({f"knee_{s}": kn, f"ankle_{s}": an, f"toe_{s}": toe, f"heel_{s}": heel})
            sh = hip + rot(np.array(p["shoulder_n" if s == "n" else "shoulder_f"], float), lean)
            el = sh + down(row[f"sh_{s}"]) * A / 2
            wr = el + down(row[f"sh_{s}"] + row[f"el_{s}"]) * A / 2
            j.update({f"shoulder_{s}": sh, f"elbow_{s}": el, f"wrist_{s}": wr})
        j["neck"] = hip + rot(np.array(p["neck"], float), lean)
        hx, hy, hr = p["head"]
        j["head"] = hip + rot(np.array([hx, hy], float), lean)
        j["lean"] = lean
        return j

    # ---- renders ----------------------------------------------------------------------------------
    def _xf(self, size, scale=None):
        W, H = self.p["canvas"]
        s = scale or min(0.9 * size / W, 0.9 * size / H)
        ox, oy = (size - W * s) / 2, (size - H * s) / 2
        return lambda v: ((v[0]) * s + ox, (v[1]) * s + oy), s

    def openpose(self, t, size=1024, scale=None):
        j = self.joints(t)
        xf, s = self._xf(size, scale)
        im = Image.new("RGB", (size, size), (0, 0, 0))
        d = ImageDraw.Draw(im)
        hr = self.p["head"][2]
        nose = j["head"] + rot(np.array([hr * 0.9, 0.1 * hr]), j["lean"])
        eye = j["head"] + rot(np.array([hr * 0.45, -0.25 * hr]), j["lean"])
        ear = j["head"] + rot(np.array([-hr * 0.2, -0.1 * hr]), j["lean"])
        # indices follow body-18: 0 nose 1 neck 2 Rsho 3 Relb 4 Rwri 5 Lsho 6 Lelb 7 Lwri 8 Rhip 9 Rkne 10 Rank 11 Lhip 12 Lkne 13 Lank 14 Reye 15 Leye 16 Rear 17 Lear
        # near side is drawn as "R" (image-left in OpenPose terms does not matter in a side view)
        P = {0: nose, 1: j["neck"], 2: j["shoulder_n"], 3: j["elbow_n"], 4: j["wrist_n"], 5: j["shoulder_f"], 6: j["elbow_f"],
             7: j["wrist_f"], 8: j["hip"], 9: j["knee_n"], 10: j["ankle_n"], 11: j["hip"], 12: j["knee_f"], 13: j["ankle_f"],
             14: eye, 15: eye + np.array([0, 0]), 16: ear, 17: ear}
        limbs = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13), (1, 0),
                 (0, 14), (14, 16), (0, 15), (15, 17)]
        lw = max(int(size / 128), 4)
        for k, (a, b) in enumerate(limbs):
            col = OP_COLORS[k % 18]
            if k in (3, 4, 5, 6, 10, 11, 12) and False:
                pass
            d.line([xf(P[a]), xf(P[b])], fill=col, width=lw)
        for i, pt in P.items():
            x, y = xf(pt)
            r = lw * 0.8
            d.ellipse([x - r, y - r, x + r, y + r], fill=OP_COLORS[i % 18])
        return im

    def _parts(self, t):
        """Drawable parts back to front: (kind, points, radii). kind 'cap' = tapered capsule chain, 'poly' = polygon, 'circle'."""
        p, j = self.p, self.joints(t)
        lw, aw, hr = p["leg_w"], p["arm_w"], p["hand_r"]
        parts = []

        def arm(s):
            return [("cap", [j[f"shoulder_{s}"], j[f"elbow_{s}"], j[f"wrist_{s}"]], [aw[0] / 2, aw[1] / 2, aw[1] / 2]),
                    ("circle", [j[f"wrist_{s}"]], [hr])]

        def leg(s):
            return [("cap", [j["hip"], j[f"knee_{s}"], j[f"ankle_{s}"]], [lw[0] / 2, (lw[0] + lw[1]) / 4, lw[1] / 2]),
                    ("cap", [j[f"heel_{s}"], j[f"toe_{s}"]], [p["foot"][1] * 0.42] * 2)]

        parts += arm("f") + leg("f")
        parts.append(("body", None, None))
        parts += leg("n") + arm("n")
        return parts, j

    def _body_poly(self, j):
        p, hip, lean = self.p, j["hip"], j["lean"]
        return [hip + rot(np.array(q, float), lean) for q in p["body"]]

    def _draw_part(self, d, part, j, xf, s, fill, grow=0.0):
        kind, pts, rad = part
        if kind == "body":
            poly = self._body_poly(j)
            d.polygon([xf(q) for q in poly], fill=fill)
            if grow:
                for a, b in zip(poly, poly[1:] + poly[:1]):
                    d.line([xf(a), xf(b)], fill=fill, width=int(2 * grow * s))
                for q in poly:
                    x, y = xf(q); r = grow * s
                    d.ellipse([x - r, y - r, x + r, y + r], fill=fill)
            if self.p.get("head_shape") == "circle":
                hx, hy = j["head"]; r = self.p["head"][2] + grow
                x, y = xf((hx, hy)); r *= s
                d.ellipse([x - r, y - r, x + r, y + r], fill=fill)
            return
        if kind == "circle":
            x, y = xf(pts[0]); r = (rad[0] + grow) * s
            d.ellipse([x - r, y - r, x + r, y + r], fill=fill)
            return
        for a, b, ra, rb in zip(pts[:-1], pts[1:], rad[:-1], rad[1:]):
            ra, rb = (ra + grow) * s, (rb + grow) * s
            (ax, ay), (bx, by) = xf(a), xf(b)
            v = np.array([bx - ax, by - ay]); L = np.hypot(*v) or 1.0
            n = np.array([-v[1], v[0]]) / L
            d.polygon([(ax + n[0] * ra, ay + n[1] * ra), (bx + n[0] * rb, by + n[1] * rb),
                       (bx - n[0] * rb, by - n[1] * rb), (ax - n[0] * ra, ay - n[1] * ra)], fill=fill)
            for (x, y), r in ((xf(a), ra), (xf(b), rb)):
                d.ellipse([x - r, y - r, x + r, y + r], fill=fill)

    def mannequin(self, t, size=1024, scale=None, torso=False, ss=2):
        parts, j = self._parts(t)
        xf, s = self._xf(size * ss, scale and scale * ss)
        im = Image.new("L", (size * ss, size * ss), 255)
        d = ImageDraw.Draw(im)
        for part in parts:
            if part[0] == "body" and not torso:
                continue
            self._draw_part(d, part, j, xf, s, 0, grow=0.6)
        return im.resize((size, size), Image.LANCZOS).convert("RGB")

    def lines(self, t, size=1024, scale=None, line=1.1, ss=2):
        parts, j = self._parts(t)
        xf, s = self._xf(size * ss, scale and scale * ss)
        im = Image.new("L", (size * ss, size * ss), 255)
        d = ImageDraw.Draw(im)
        for part in parts:
            self._draw_part(d, part, j, xf, s, 0, grow=line)
            self._draw_part(d, part, j, xf, s, 255)
        return im.resize((size, size), Image.LANCZOS).convert("RGB")


def contact_row(figure, fn, size=256):
    row = Image.new("RGB", (size * figure.n, size), (255, 255, 255))
    for t in range(figure.n):
        row.paste(fn(t).resize((size, size), Image.LANCZOS), (t * size, 0))
    return row


def main():
    char, motion, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    fig = Figure(char, motion)
    for mode in ("openpose", "mannequin", "lines"):
        fn = getattr(fig, mode)
        for t in range(fig.n):
            fn(t).save(out / f"{mode}_{t}.png")
        contact_row(fig, fn).save(out / f"{mode}_row.png")
    for r in fig.table:
        print(r)


if __name__ == "__main__":
    main()

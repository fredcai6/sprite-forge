"""Review sheet for a rig: Muybridge plate 60 reference poses on top, the frames at 3x below, a GIF-less look at the cycle.

    python -m forge.runsheet characters/NAME/rig/run.toml [OUT.png]
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .generate import ROOT

PLATE = ROOT / "characters/references/muybridge-plate60-side.png"
# (label, column, row) in the plate: contact, down, pass, flight, reach
REFS = [("contact", 3, 1), ("down", 4, 1), ("pass", 3, 0), ("flight", 1, 1), ("flight/reach", 2, 1)]
COLS = [(0, 360), (370, 740), (750, 1120), (1130, 1500), (1505, 1875)]
ROWS = [(40, 460), (510, 930)]


def main(rig, out=None):
    rig = Path(rig).resolve()
    name, anim = rig.parent.parent.name, rig.stem
    sheet = Image.open(ROOT / "characters" / name / "sprites" / anim / "sheet.png").convert("RGBA")
    import json
    a = json.loads((ROOT / "characters" / name / "sprites" / anim / "atlas.json").read_text())
    w, h, n = a["frame"]["w"], a["frame"]["h"], a["count"]
    z = 3
    plate = Image.open(PLATE).convert("RGB")
    refh = 200
    refs = []
    for lab, c, r in REFS:
        im = plate.crop((COLS[c][0], ROWS[r][0], COLS[c][1], ROWS[r][1]))
        im = im.resize((int(im.width * refh / im.height), refh))
        refs.append((lab, im))
    per = min(n, 4)
    rows = -(-n // per)
    W = max(per * w * z, sum(i.width + 6 for _, i in refs))
    canvas = Image.new("RGB", (W, refh + 16 + rows * (h * z + 16)), (30, 26, 36))
    d = ImageDraw.Draw(canvas)
    x = 0
    for lab, im in refs:
        canvas.paste(im, (x, 0)); d.text((x + 3, refh + 2), "Muybridge " + lab, fill=(255, 255, 255)); x += im.width + 6
    y0 = refh + 16
    for i in range(n):
        fr = sheet.crop((i * w, 0, (i + 1) * w, h)).resize((w * z, h * z), Image.NEAREST)
        bg = Image.new("RGBA", fr.size, (110, 100, 125, 255)); bg.alpha_composite(fr)
        cx, cy = (i % per) * w * z, y0 + (i // per) * (h * z + 16)
        canvas.paste(bg.convert("RGB"), (cx, cy))
        d.text((cx + 3, cy + h * z + 2), f"{name} {anim} {i}", fill=(255, 255, 255))
    out = Path(out or f"/mnt/data/sprite-forge/review/{name}-{anim}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    print(out, canvas.size)


if __name__ == "__main__":
    main(*sys.argv[1:])

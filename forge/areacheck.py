"""python -m forge.areacheck CITY/AREA OUT.png [x0]  -- scene at game size plus 2x mid crops for the lettering check"""
import json, sys
from pathlib import Path
from PIL import Image
A = Path.home() / "projects/probable-octo-funicular/assets"
B = A / "backgrounds" / sys.argv[1]
x0 = int(sys.argv[3]) if len(sys.argv) > 3 else 400
def L(p): return Image.open(p).convert("RGBA")
def fr(n, a):
    d = A / "sprites" / n / a; at = json.loads((d / "atlas.json").read_text()); w = at["frame"]["w"]
    return L(d / "sheet.png").crop((0, 0, w, at["frame"]["h"]))
im = Image.new("RGBA", (1280, 720), (0, 0, 0, 255))
for k in ("far", "mid"): im.alpha_composite(L(B / f"{k}.png"), (0, 0), (x0 + (60 if k == "far" else 0), 0, x0 + 1280 + (60 if k == "far" else 0), 720))
g = L(B / "ground.png")
for x in range(0, 1280, 256): im.alpha_composite(g, (x, 600))
im.alpha_composite(fr("slice-ny", "run"), (150, 600 - 124))
t = fr("teen", "walk").transpose(Image.FLIP_LEFT_RIGHT); im.alpha_composite(t, (420, 600 - t.height))
d = L(B / "door.png"); im.alpha_composite(d, (1050, 600 - d.height))
im.alpha_composite(L(B / "near.png"), (0, 0), (x0, 0, x0 + 1280, 720))
im.alpha_composite(L(A / "backgrounds/props/box.png"), (700, 600 - 84))
out = Image.new("RGB", (1280, 720 + 2 * 360), (0, 0, 0)); out.paste(im.convert("RGB"), (0, 0))
m = L(B / "mid.png"); bg = Image.new("RGBA", m.size, (200, 200, 210, 255)); bg.alpha_composite(m)
for i in range(2):
    c = bg.crop((x0 + i * 640, 220, x0 + i * 640 + 640, 600)).resize((1280, 760), Image.NEAREST).crop((0, 0, 1280, 360)) if False else bg.crop((x0 + i * 640, 300, x0 + i * 640 + 640, 480)).resize((1280, 360), Image.NEAREST)
    out.paste(c.convert("RGB"), (0, 720 + i * 360))
out.save(sys.argv[2])

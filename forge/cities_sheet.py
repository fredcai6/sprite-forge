"""Every city composited with its characters at game size: python -m forge.cities_sheet ASSETS OUT.png"""
import json, sys
from pathlib import Path
from PIL import Image, ImageDraw
A = Path(sys.argv[1]); B = A / "backgrounds"
def fr(name, anim, i=0):
    d = A / "sprites" / name / anim; at = json.loads((d / "atlas.json").read_text()); w = at["frame"]["w"]
    return Image.open(d / "sheet.png").convert("RGBA").crop((i * w, 0, (i + 1) * w, at["frame"]["h"]))
def put(im, spr, x, feet, flip=False):
    if flip: spr = spr.transpose(Image.FLIP_LEFT_RIGHT)
    im.alpha_composite(spr, (int(x - spr.width / 2), int(feet - spr.height)))
G = {"newyork": "slice-ny", "chicago": "slice-deep", "dc": "slice-jumbo", "naples": "slice-naples", "california": "slice-thin"}
E_ = {"newyork": [("teen", "walk"), ("turtle", "walk")], "chicago": [("ninja", "walk"), ("teen", "walk")], "dc": [("mutant", "walk"), ("turtle", "walk")],
      "naples": [("ninja", "throw"), ("mutant", "idle")], "california": [("teen", "idle"), ("ninja", "walk")]}
sheet = Image.new("RGB", (2560, 720 * 3), (0, 0, 0))
for k, city in enumerate(G):
    im = Image.new("RGBA", (1280, 720), (0, 0, 0, 255))
    im.alpha_composite(Image.open(B / city / "far.png").convert("RGBA"), (0, 0), (300, 0, 1580, 720))
    im.alpha_composite(Image.open(B / city / "mid.png").convert("RGBA"), (0, 0), (500, 0, 1780, 720))
    d = ImageDraw.Draw(im); d.rectangle([0, 600, 1280, 720], fill=(70, 62, 70)); d.rectangle([0, 600, 1280, 606], fill=(110, 100, 104))
    put(im, fr(G[city], "run"), 200, 600)
    for j, (n, a) in enumerate(E_[city]):
        put(im, fr(n, a), 480 + j * 190, 600 if n != "mutant" else 585, True)
    for j, (n, x) in enumerate([("barricade", 780), ("barrel", 880), ("barrel-out", 950), ("door", 1100)]):
        p = Image.open(B / city / f"{n}.png").convert("RGBA"); im.alpha_composite(p, (x - p.width // 2, 600 - p.height))
    pl = Image.open(B / city / "platform.png").convert("RGBA")
    for j in range(4): im.alpha_composite(pl, (560 + j * 64, 430))
    im.alpha_composite(Image.open(B / city / "near.png").convert("RGBA"), (0, 0), (500, 0, 1780, 720))
    sheet.paste(im.convert("RGB"), ((k % 2) * 1280, (k // 2) * 720))
sheet.save(sys.argv[2])

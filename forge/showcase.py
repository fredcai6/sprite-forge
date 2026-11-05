"""Contact sheet of the proof batch: python -m forge.showcase GAME_ASSETS OUT.png
Scenes with characters placed at game scale on top, then every animation strip at 2x."""
import json, sys
from pathlib import Path
from PIL import Image, ImageDraw

A = Path(sys.argv[1])
man = json.loads((A / "manifest.json").read_text())


def strip(name, anim):
    d = A / "sprites" / name / anim
    at = json.loads((d / "atlas.json").read_text())
    sheet = Image.open(d / "sheet.png").convert("RGBA")
    w = at["frame"]["w"]
    return [sheet.crop((i * w, 0, (i + 1) * w, at["frame"]["h"])) for i in range(at["count"])]


def put(img, spr, x, feet):
    img.alpha_composite(spr, (int(x - spr.width / 2), int(feet - spr.height)))


bg = A / "backgrounds"
hq = Image.open(bg / "hq/back.png").convert("RGBA")
hq.alpha_composite(Image.open(bg / "hq/mid.png").convert("RGBA"))
put(hq, strip("old-timer", "talk")[0], 330, 520)
put(hq, strip("slice-ny", "idle")[0], 560, 540)
put(hq, strip("slice-ny", "cold")[0], 700, 560)
put(hq, strip("slice-ny", "run")[1], 810, 600)
hq.alpha_composite(Image.open(bg / "hq/front.png").convert("RGBA"))

ny = Image.new("RGBA", (1280, 720), (0, 0, 0, 255))
far, mid, near = [Image.open(bg / f"newyork/{n}.png").convert("RGBA") for n in ("far", "mid", "near")]
ny.alpha_composite(far, (0, 0), (300, 0, 1580, 720))
ny.alpha_composite(mid, (0, 0), (500, 0, 1780, 720))
d = ImageDraw.Draw(ny)
d.rectangle([0, 600, 1280, 720], fill=(58, 50, 56))
d.rectangle([0, 600, 1280, 610], fill=(96, 86, 88))
put(ny, strip("slice-ny", "run")[0], 250, 600)
put(ny, strip("teen", "walk")[0].transpose(Image.FLIP_LEFT_RIGHT), 620, 600)
put(ny, strip("turtle", "walk")[0].transpose(Image.FLIP_LEFT_RIGHT), 900, 600)
ny.alpha_composite(near, (0, 0), (500, 0, 1780, 720))

rows = [(n, a, strip(n, a)) for n, anims in man.items() for a in anims]
Z = 2
rowh = [max(f.height for f in fr) * Z + 24 for _, _, fr in rows]
sheet = Image.new("RGBA", (2560, 720 + 10 + sum(rowh)), (24, 20, 32, 255))
sheet.paste(hq, (0, 0))
sheet.paste(ny, (1280, 0))
y = 730
dr = ImageDraw.Draw(sheet)
for (n, a, fr), h in zip(rows, rowh):
    dr.text((8, y + 2), f"{n}/{a}", fill=(230, 220, 200, 255))
    x = 8
    for f in fr:
        sheet.alpha_composite(f.resize((f.width * Z, f.height * Z), Image.NEAREST), (x, y + 18))
        x += f.width * Z + 4
        if x > 2560 - 260:
            break
    y += h
sheet.convert("RGB").save(sys.argv[2])
print(sheet.size)

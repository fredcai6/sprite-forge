"""Rebuilds characters/<name>/palette.json from its references: palette_build.py NAME"""
import glob, json, sys, tomllib
from pathlib import Path

import numpy as np
from PIL import Image

from forge.color import build_palette

char = Path(__file__).parent / sys.argv[1]
cfg = tomllib.loads((char / "character.toml").read_text())
px = []
for f in sorted(glob.glob(str(char / "references/*.png"))):
    a = np.asarray(Image.open(f).convert("RGB"))
    px.append(a[a.sum(2) > 24])
pal = build_palette(np.concatenate(px), cfg["palette"]["families"])
(char / "palette.json").write_text(json.dumps(pal, indent=1))
print({k: len(v) for k, v in pal.items()})

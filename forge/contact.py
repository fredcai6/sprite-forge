"""Contact sheet for a run: python -m forge.contact RUN_DIR OUT.png [thumb]. One row per shot."""
import sys
from pathlib import Path

from PIL import Image


def main():
    run, out = Path(sys.argv[1]), sys.argv[2]
    t = int(sys.argv[3]) if len(sys.argv) > 3 else 330
    rows = [d for d in sorted(run.iterdir()) if any(d.glob("*.png"))]
    cols = max(len(list(d.glob("*.png"))) for d in rows)
    sheet = Image.new("RGB", (cols * t, len(rows) * t), (20, 20, 24))
    for y, d in enumerate(rows):
        for x, f in enumerate(sorted(d.glob("*.png"))):
            sheet.paste(Image.open(f).convert("RGB").resize((t - 4, t - 4)), (x * t, y * t))
    sheet.save(out)


if __name__ == "__main__":
    main()

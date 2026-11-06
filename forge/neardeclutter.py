"""Clean a near layer in place: python -m forge.neardeclutter FILE...
A near layer holds only street furniture standing on the ground line: drop full-width bars, wide slabs,
specks, and any piece whose base is above y 540 (wires, banners, hanging foliage)."""
import sys
import numpy as np
from PIL import Image
from scipy import ndimage

def clean(a):
    a = a.copy()
    a[a.mean(1) > 0.5] = False
    lbl, n = ndimage.label(a)
    for i, o in enumerate(ndimage.find_objects(lbl), 1):
        w, h = o[1].stop - o[1].start, o[0].stop - o[0].start
        m = lbl == i
        if w > 150 and h < 140: a[m] = False
        elif o[0].start < 330 and w >= 80: a[m] = False   # a tall wide mass is a tree/banner floating over the street, not furniture
        elif m.sum() < 60 or o[0].stop < 540: a[m] = False   # furniture stands on the ground line; nothing floats or hangs
    return ndimage.binary_opening(a, iterations=1)

if __name__ == "__main__":
    for f in sys.argv[1:]:
        im = np.asarray(Image.open(f).convert("RGBA")).copy()
        im[..., 3] = np.where(clean(im[..., 3] > 0), 255, 0)
        Image.fromarray(im, "RGBA").save(f, optimize=True); print("cleaned", f)

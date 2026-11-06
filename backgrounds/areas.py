def _look(haze, amt=0.2, c=0.74, l=18):
    return {"far": dict(contrast=0.9, lift=4, haze=haze, haze_amt=0.08, median=3),
            "mid": dict(contrast=c, lift=l, haze=haze, haze_amt=amt, median=3),
            "near": dict(contrast=0.85, lift=10, median=0)}
def _a(far, mid, near, ground, door, ground_row, tint, haze, amt=0.2, c=0.74, l=18, erase=(), door_h=170):
    return dict(seeds=dict(far=far, mid=mid, near=near, ground=ground, door=door), ground=ground_row, near_tint=tint,
                look=_look(haze, amt, c, l), erase=list(erase), door_h=door_h)
AREAS["newyork/wallstreet"] = _a(2, 5, 4, 1, 2, 650, (40, 46, 66), (190, 205, 228))
AREAS["newyork/centralpark"] = _a(3, 2, 4, 1, 2, 645, (40, 60, 44), (214, 232, 222), 0.14, 0.78, 14)
AREAS["newyork/brooklyn"] = _a(2, 3, 4, 1, 1, 650, (60, 40, 40), (240, 214, 196), 0.16, 0.76, 16)
AREAS["chicago/loop"] = _a(1, 2, 4, 1, 3, 660, (36, 42, 66), (190, 205, 232), 0.2, 0.74, 20)
AREAS["chicago/lincolnpark"] = _a(1, 2, 4, 1, 2, 640, (36, 60, 60), (214, 232, 236), 0.14, 0.78, 14)
AREAS["chicago/wickerpark"] = _a(3, 4, 2, 1, 2, 650, (60, 36, 36), (236, 214, 200), 0.16, 0.76, 16)

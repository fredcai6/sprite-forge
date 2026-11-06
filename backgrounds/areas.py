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
AREAS["dc/georgetown"] = _a(1, 2, 4, 1, 2, 660, (60, 40, 36), (240, 224, 200), 0.16, 0.76, 16)
AREAS["dc/mall"] = _a(3, 5, 4, 1, 2, 690, (36, 60, 44), (214, 232, 244), 0.12, 0.8, 12)
AREAS["dc/federal"] = _a(4, 1, 4, 2, 2, 655, (52, 50, 56), (226, 226, 220), 0.14, 0.78, 14)
AREAS["naples/harbor"] = _a(2, 2, 4, 1, 2, 650, (62, 44, 40), (246, 214, 176), 0.14, 0.78, 14)
AREAS["naples/center"] = _a(1, 4, 4, 1, 2, 655, (62, 38, 36), (240, 206, 170), 0.16, 0.76, 16)
AREAS["naples/hillside"] = _a(1, 4, 4, 1, 3, 650, (70, 46, 56), (250, 220, 220), 0.12, 0.8, 14)
AREAS["california/downtown"] = _a(1, 3, 2, 4, 2, 655, (62, 44, 48), (255, 226, 176), 0.14, 0.78, 14)
AREAS["california/beach"] = _a(1, 4, 4, 1, 1, 655, (62, 44, 40), (255, 236, 190), 0.12, 0.8, 12)
AREAS["california/bungalows"] = _a(4, 2, 4, 3, 1, 645, (44, 36, 56), (255, 200, 160), 0.16, 0.76, 14, door_h=300)

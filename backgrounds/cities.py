def _look(haze, amt, mid_c=0.74, mid_l=20):
    return {"far": dict(contrast=0.9, lift=4, haze=haze, haze_amt=0.08, median=3),
            "mid": dict(contrast=mid_c, lift=mid_l, haze=haze, haze_amt=amt, median=3),
            "near": dict(contrast=0.85, lift=10, median=0)}


CITIES["chicago"] = dict(seeds=dict(far=4, mid=5, near=1), ground=602, erase=[], near_tint=(40, 48, 72), look=_look((190, 205, 228), 0.2))
CITIES["dc"] = dict(seeds=dict(far=3, mid=2, near=1), ground=655, erase=[], near_tint=(40, 56, 44), look=_look((226, 232, 214), 0.15, 0.76, 14))
CITIES["california"] = dict(seeds=dict(far=4, mid=5, near=1), ground=588, erase=[], near_tint=(62, 44, 52), look=_look((255, 226, 172), 0.15, 0.78, 12))
CITIES["naples"] = dict(seeds=dict(far=2, mid=2, near=1), mid_run="naples/mid2", ground=666, erase=[], near_tint=(62, 38, 40), look=_look((236, 190, 160), 0.18, 0.76, 14))

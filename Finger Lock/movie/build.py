"""Render scenes.   python movie/build.py sheets [names...]   -> contact sheets in movie/out/sheets
                   python movie/build.py render [names...]   -> one MP4 per scene in movie/out
"""
import importlib
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "scenes"))

import kit  # noqa: E402

OUT = os.path.join(HERE, "out")
os.makedirs(os.path.join(OUT, "sheets"), exist_ok=True)


# Target running time (seconds) of each scene. The scene's whole timeline (animation and
# captions) is compressed to fit. Total = 296 s = 4:56.
NEW_DUR = {"s0": 8, "s1": 15, "s2": 27, "s3": 18, "s4": 35, "s5": 24, "s6": 28,
           "s7": 20, "s8": 41, "s9": 16, "s10": 22, "s11": 28, "s12": 14}


def _compress(name, dur, caps, tag, draw):
    new = NEW_DUR.get(name, dur)
    f = new / dur
    return (name, new, [(ts * f, text) for ts, text in caps], tag,
            lambda cv, t, draw=draw, f=f: draw(cv, t / f))


def all_scenes():
    scenes = []
    for mod in ("s_a", "s_b", "s_c"):
        try:
            scenes += importlib.import_module(mod).SCENES
        except ModuleNotFoundError as e:
            if e.name != mod:
                raise
    scenes = [_compress(*sc) for sc in scenes]
    scenes.sort(key=lambda sc: int(sc[0][1:]))
    return scenes


if __name__ == "__main__":
    mode = sys.argv[1]
    want = sys.argv[2:]
    for name, dur, caps, tag, draw in all_scenes():
        if want and name not in want:
            continue
        t0 = time.time()
        if mode == "sheets":
            kit.contact_sheet(draw, dur, caps, tag, os.path.join(OUT, "sheets", f"{name}.png"), n=6)
        else:
            n = kit.render(os.path.join(OUT, f"{name}.mp4"), dur, draw, caps, tag)
            print(f"{name}: {n} frames", end=" ")
        print(f"{name} done in {time.time() - t0:.1f}s")

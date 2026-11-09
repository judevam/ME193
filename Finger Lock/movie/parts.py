"""Reusable drawing pieces for the scenes."""
import math

import numpy as np

import assets as A
from kit import COL, ramp, smooth, gray_rgb

CLS = A.CLASSES


def pick_image(label, which="clear", person="jude"):
    """A real image of this label. 'clear' = the net is very sure (falls back to the most
    confident one); 'unsure' = it is not."""
    best, top = None, None
    files = A.files_for(label, person)
    k = CLS.index(label)
    for f in files[::2]:
        crop, g = A.load(f)
        p = A.run(g)[0]
        score = p[k]
        if top is None or score > top[0]:
            top = (score, (crop, g, f, p))
        if which == "clear" and score > 0.97:
            best = (crop, g, f, p)
            if len(files) and f.endswith("_020.jpg"):
                break
        if which == "unsure" and 0.1 < score < 0.45:
            return crop, g, f, p
    if which == "clear":
        return best if best is not None else top[1]
    return None


def tag(cv, c, s, size=18, fill="text", back="panel"):
    """Text centered at c with a solid box behind it (readable on top of lines)."""
    from kit import font
    w = font(size).getlength(str(s))
    cv.rect([c[0] - w / 2 - 6, c[1] - size * 0.7, c[0] + w / 2 + 6, c[1] + size * 0.7], fill=back, width=0)
    cv.center(c, s, size, fill)


def number_grid(cv, x0, y0, cell, vals, fmt="{:.2f}", size=13, fills=None, text_fill="text",
                outline="line", count=None, signed_color=False):
    """Draw a 2-D array of numbers as cells. count = how many cells (row-major) to show."""
    h, w = vals.shape
    i = 0
    for r in range(h):
        for c in range(w):
            x, y = x0 + c * cell, y0 + r * cell
            show = count is None or i < count
            fill = None
            if fills is not None and show:
                fill = tuple(int(v) for v in fills[r, c])
            cv.rect([x, y, x + cell, y + cell], outline=outline, fill=fill, width=1)
            if show:
                v = vals[r, c]
                col = text_fill
                if signed_color:
                    col = "ok" if v > 0.005 else ("bad" if v < -0.005 else "dim")
                cv.center((x + cell / 2, y + cell / 2), fmt.format(v), size, col)
            i += 1


def signed_fills(vals, scale=None):
    """green for positive, red for negative, dark when near zero (filter weights)."""
    s = scale or max(1e-6, np.abs(vals).max())
    f = np.zeros(vals.shape + (3,), np.float32)
    pos = np.clip(vals / s, 0, 1)[..., None]
    neg = np.clip(-vals / s, 0, 1)[..., None]
    f += pos * np.array([20, 120, 60]) + neg * np.array([130, 30, 30]) + 14
    return f


def plot_axes(cv, box, xr, yr, title=None, xlabel="", ylabel=""):
    x0, y0, x1, y1 = box
    cv.rect(box, outline="line", fill="panel", width=1)
    if title:
        cv.text((x0 + 8, y0 + 5), title, 14, "dim")
    # zero lines
    def px(x):
        return x0 + (x - xr[0]) / (xr[1] - xr[0]) * (x1 - x0)

    def py(y):
        return y1 - (y - yr[0]) / (yr[1] - yr[0]) * (y1 - y0)
    if xr[0] < 0 < xr[1]:
        cv.line((px(0), y0 + 22), (px(0), y1), "line", 1)
    if yr[0] < 0 < yr[1]:
        cv.line((x0, py(0)), (x1, py(0)), "line", 1)
    if xlabel:
        cv.text((x1 - 6, y1 - 4), xlabel, 13, "dim", anchor="rd")
    if ylabel:
        cv.text((x0 + 6, y0 + 24), ylabel, 13, "dim")
    return px, py


def curve(cv, px, py, xs, ys, fill="ok", width=3, ymax=None):
    ys = [min(y, ymax) for y in ys] if ymax is not None else ys
    pts = [(px(x), py(y)) for x, y in zip(xs, ys)]
    for p, q in zip(pts[:-1], pts[1:]):
        cv.line(p, q, fill, width)


def vault_door(cv, cx, cy, r, open_p=0.0, accent="bad", dial_deg=0):
    """A drawn vault door. open_p 0..1 swings it open (it narrows) showing gold."""
    # interior with gold bars
    cv.circle((cx, cy), r - 6, outline=None, fill=(40, 30, 8))
    if open_p > 0:
        for i in range(3):
            for j in range(3):
                bx, by = cx - r * 0.55 + j * r * 0.38, cy - r * 0.3 + i * r * 0.22
                cv.rect([bx, by, bx + r * 0.34, by + r * 0.17], fill=(230, 180, 50), outline=(150, 105, 20), width=1)
    wd = r * (1 - 0.85 * open_p)
    left = cx - r
    box = [left, cy - r, left + 2 * wd, cy + r]
    cv.d.ellipse(box, fill=COL["panel2"], outline=COL["steel"], width=5)
    if wd > r * 0.4:
        mx = left + wd
        cv.d.ellipse([mx - (r - 22) * wd / r, cy - (r - 22), mx + (r - 22) * wd / r, cy + (r - 22)],
                     outline=COL["line"], width=3)
        for k in range(12):
            a = k * math.pi / 6
            bx, by = mx + (r - 14) * math.sin(a) * wd / r, cy - (r - 14) * math.cos(a)
            cv.circle((bx, by), 5, outline="steel", fill="panel", width=2)
        # dial (ticks) and hub, squeezed horizontally with the door
        for k in range(36):
            a = (k * 10 + dial_deg) * math.pi / 180
            r1, r2 = r * 0.52, r * (0.62 if k % 3 == 0 else 0.57)
            cv.line((mx + r1 * math.sin(a) * wd / r, cy - r1 * math.cos(a)),
                    (mx + r2 * math.sin(a) * wd / r, cy - r2 * math.cos(a)), "text" if k % 3 == 0 else "dim", 2)
        cv.d.ellipse([mx - 18 * wd / r, cy - 18, mx + 18 * wd / r, cy + 18], fill=COL[accent])


def tumblers(cv, x, y, filled, w=70, h=84, gap=14, cur=None, symbols=None):
    for i in range(3):
        bx = x + i * (w + gap)
        done = i < filled
        cv.rect([bx, y, bx + w, y + h], outline="ok" if done else ("warn" if cur == i else "line"),
                fill="panel", width=3)
        cv.center((bx + w / 2, y + h / 2), (symbols[i] if symbols and done else "●") if done else "?", 42,
                  "ok" if done else "dim", True)

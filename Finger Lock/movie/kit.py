"""Tiny rendering toolkit for the explainer movie (Pillow + numpy + ffmpeg).

A scene is a function draw(cv, t) that paints one frame for time t (seconds).
render() calls it for every frame, adds the caption bar, and writes an MP4.
Frames run at 12 fps so motion steps like stop-action animation.
"""
import functools
import math

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1280, 720, 12
COL = dict(bg=(6, 8, 10), panel=(12, 17, 21), panel2=(17, 25, 32), line=(34, 48, 58),
           text=(207, 227, 214), dim=(111, 133, 121), ok=(53, 255, 140), warn=(255, 176, 32),
           bad=(255, 59, 59), steel=(139, 153, 165), cyan=(80, 200, 255), gold=(255, 208, 74))
FONT_R = "C:/Windows/Fonts/consola.ttf"
FONT_B = "C:/Windows/Fonts/consolab.ttf"
CAP_Y = 640  # caption bar top


@functools.lru_cache(maxsize=None)
def font(size, bold=False):
    return ImageFont.truetype(FONT_B if bold else FONT_R, size)


def _make_bg():
    y = np.linspace(0, 1, H)[:, None]
    base = np.array(COL["bg"], np.float32)[None, None, :] * np.ones((H, W, 1), np.float32)
    glow = np.exp(-((y - 0.0) ** 2) / 0.08) * 14  # soft light from the top
    base = base + glow[:, :, None] * np.array([0.4, 0.9, 0.7])
    base[::4] *= 0.80  # scanlines
    return np.clip(base, 0, 255).astype(np.uint8)


BG = _make_bg()


# ---- easing -----------------------------------------------------------------
def ramp(t, a, b):
    """0 before a, 1 after b, linear between."""
    return float(min(1.0, max(0.0, (t - a) / (b - a)))) if b > a else float(t >= a)


def smooth(t, a, b):
    p = ramp(t, a, b)
    return p * p * (3 - 2 * p)


def lerp(a, b, p):
    return a + (b - a) * p


class Cv:
    def __init__(self, tag=""):
        self.im = Image.fromarray(BG.copy())
        self.d = ImageDraw.Draw(self.im)
        self.d.text((24, 14), "VAULT 193", font=font(18, True), fill=COL["ok"])
        self.d.text((128, 14), "// HOW THE NEURAL NET READS FINGERS", font=font(18), fill=COL["dim"])
        if tag:
            self.d.text((W - 24, 14), tag, font=font(18), fill=COL["warn"], anchor="ra")
        self.d.line([(24, 42), (W - 24, 42)], fill=COL["line"], width=1)

    # ---- primitives ----
    def text(self, xy, s, size=24, fill="text", bold=False, anchor="la"):
        self.d.text(xy, str(s), font=font(size, bold), fill=COL.get(fill, fill), anchor=anchor)

    def rect(self, box, outline=None, fill=None, width=2):
        self.d.rectangle(box, outline=COL.get(outline, outline) if outline else None,
                         fill=COL.get(fill, fill) if fill else None, width=width)

    def panel(self, box, title=None, color="line"):
        self.d.rectangle(box, fill=COL["panel"], outline=COL[color], width=1)
        if title:
            self.d.text((box[0] + 10, box[1] + 6), title, font=font(14), fill=COL["dim"])

    def line(self, p, q, fill="steel", width=2):
        self.d.line([p, q], fill=COL.get(fill, fill), width=width)

    def arrow(self, p, q, fill="steel", width=3, head=10):
        self.line(p, q, fill, width)
        a = math.atan2(q[1] - p[1], q[0] - p[0])
        pts = [q, (q[0] - head * math.cos(a - 0.45), q[1] - head * math.sin(a - 0.45)),
               (q[0] - head * math.cos(a + 0.45), q[1] - head * math.sin(a + 0.45))]
        self.d.polygon(pts, fill=COL.get(fill, fill))

    def circle(self, c, r, outline="steel", fill=None, width=2):
        self.d.ellipse([c[0] - r, c[1] - r, c[0] + r, c[1] + r],
                       outline=COL.get(outline, outline) if outline else None,
                       fill=COL.get(fill, fill) if fill else None, width=width)

    def center(self, xy, s, size=24, fill="text", bold=False):
        self.text(xy, s, size, fill, bold, anchor="mm")

    def wrap(self, xy, s, size=22, fill="text", width=500, lh=1.35, bold=False):
        f = font(size, bold)
        words, line, y = str(s).split(), "", xy[1]
        for w_ in words:
            test = (line + " " + w_).strip()
            if f.getlength(test) > width and line:
                self.d.text((xy[0], y), line, font=f, fill=COL.get(fill, fill))
                y, line = y + size * lh, w_
            else:
                line = test
        if line:
            self.d.text((xy[0], y), line, font=f, fill=COL.get(fill, fill))
        return y + size * lh

    def paste(self, arr, xy, size=None, border=None, nearest=True, jitter=0, seed=0):
        """arr: uint8 (h,w) or (h,w,3). size=(w,h) to scale."""
        im = Image.fromarray(arr if arr.ndim == 3 else arr, "RGB" if arr.ndim == 3 else "L").convert("RGB")
        if size:
            im = im.resize(size, Image.NEAREST if nearest else Image.BICUBIC)
        x, y = xy
        if jitter:
            r = np.random.RandomState(seed)
            x += int(r.randint(-jitter, jitter + 1))
            y += int(r.randint(-jitter, jitter + 1))
        self.im.paste(im, (int(x), int(y)))
        if border:
            self.d.rectangle([x - 1, y - 1, x + im.width, y + im.height], outline=COL.get(border, border), width=2)
        return im.size

    def caption(self, s):
        if not s:
            return
        self.d.rectangle([0, CAP_Y, W, H], fill=(5, 7, 9))
        self.d.line([(0, CAP_Y), (W, CAP_Y)], fill=COL["line"], width=1)
        f = font(26)
        words, lines, line = s.split(), [], ""
        for w_ in words:
            test = (line + " " + w_).strip()
            if f.getlength(test) > W - 120 and line:
                lines.append(line)
                line = w_
            else:
                line = test
        lines.append(line)
        y = CAP_Y + (H - CAP_Y - 34 * len(lines)) // 2 + 2
        for ln in lines:
            self.d.text((W // 2, y), ln, font=f, fill=COL["text"], anchor="ma")
            y += 34


def neon(gray01, lo=None, hi=None):
    """float map -> uint8 RGB in the neon-green palette."""
    lo = float(gray01.min()) if lo is None else lo
    hi = float(gray01.max()) if hi is None else hi
    g = np.clip((gray01 - lo) / (hi - lo + 1e-9), 0, 1)
    return np.stack([g * 0.30, g, g * 0.55], axis=-1).__mul__(255).astype(np.uint8)


def gray_rgb(g01):
    return (np.clip(g01, 0, 1) * 255).astype(np.uint8)


def render(path, duration, draw, captions=(), tag=""):
    n = int(round(duration * FPS))
    with imageio.get_writer(path, fps=FPS, codec="libx264", quality=None, macro_block_size=1,
                            pixelformat="yuv420p", ffmpeg_params=["-crf", "20", "-preset", "fast"]) as w:
        for i in range(n):
            w.append_data(np.asarray(frame(draw, i / FPS, captions, tag)))
    return n


def frame(draw, t, captions=(), tag=""):
    cv = Cv(tag)
    draw(cv, t)
    cap = ""
    for ts, text in captions:
        if ts <= t:
            cap = text
    cv.caption(cap)
    return cv.im


def contact_sheet(draw, duration, captions, tag, path, n=6):
    """A few stills from a scene, to check the layout without rendering video."""
    ims = [frame(draw, duration * (i + 0.5) / n, captions, tag).resize((640, 360)) for i in range(n)]
    cols = 2
    rows = (n + cols - 1) // cols
    sheet = Image.new("RGB", (640 * cols, 360 * rows))
    for i, im in enumerate(ims):
        sheet.paste(im, ((i % cols) * 640, (i // cols) * 360))
    sheet.save(path)

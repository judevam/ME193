"""Scenes 0-4: hook, pixels, one neuron, width and depth, convolution."""
import math

import numpy as np

import assets as A
import parts as P
from kit import COL, FPS, lerp, ramp, smooth, gray_rgb, neon

# ---- shared real examples ---------------------------------------------------
HERO_CROP, HERO, HERO_PATH, HERO_P = P.pick_image("2")
# The code shown in the movie is a DEMO code, not the real one (which stays in secret_code.txt).
DEMO_CODE = "145"
SHOW = {lab: P.pick_image(lab)[:2] for lab in set(DEMO_CODE)}


# =============================== scene 0: hook ================================
def s0(cv, t):
    title = "HOW DOES A NEURAL NETWORK\nREAD YOUR FINGERS?"
    if t < 3.2:
        n = int(len(title) * ramp(t, 0.2, 2.6))
        shown = title[:n]
        for i, ln in enumerate(shown.split("\n")):
            cv.center((640, 250 + i * 60), ln, 44, "ok", True)
        cv.center((640, 420), "ME193 MIDTERM  //  FINGER-CODE VAULT", 20, "dim")
        return
    seq = [(d, 3.2 + i * 2.4, 5.6 + i * 2.4) for i, d in enumerate(DEMO_CODE)]
    filled = sum(t >= e - 0.4 for _, _, e in seq)
    for lab, a, b in seq:
        if a <= t < b:
            crop, g = SHOW[lab]
            cv.paste((g * 255).astype(np.uint8), (80, 150), size=(300, 300), border="warn", jitter=1, seed=int(t * FPS))
            cv.text((80, 120), "CAMERA", 16, "dim")
            cv.text((80, 465), f"reads: {lab}", 30, "warn", True)
    if t < 10.2:
        cv.text((470, 120), "YOUR CODE", 16, "dim")
        P.tumblers(cv, 470, 150, filled, symbols=None)
    P.vault_door(cv, 960, 330, 190, open_p=smooth(t, 10.2, 11.4), accent="ok" if t > 10.2 else "bad",
                 dial_deg=filled * 72)
    if t >= 10.4:
        cv.center((640, 590), "ACCESS GRANTED", 40, "ok", True)


S0 = (12.0, [(0.0, "How does a neural network read your fingers?"),
             (3.3, "Show the camera three numbers..."),
             (10.3, "...and the vault opens. Let's see exactly how that works.")], "00 HOOK")


# =============================== scene 1: pixels ==============================
def _edge_spot(g):
    gy, gx = np.gradient(g)
    m = np.hypot(gx, gy)
    m[:12], m[-12:], m[:, :12], m[:, -12:] = 0, 0, 0, 0
    y, x = np.unravel_index(np.argmax(m), m.shape)
    return int(x), int(y)


SPOT = _edge_spot(HERO)


def s1(cv, t):
    cv.paste(HERO_CROP, (40, 120), size=(330, 330), border="steel")
    cv.text((40, 92), "WHAT THE CAMERA SEES", 16, "dim")
    if t >= 4:
        a = smooth(t, 4, 6)
        cv.arrow((385, 285), (440, 285), "warn")
        cv.paste(gray_rgb(HERO), (455, 130), size=(300, 300), border="warn")
        cv.text((455, 92), "GRAYSCALE, 96 x 96", 16, "dim")
        cv.text((455, 440), "9,216 pixels", 24, "warn", True)
    if t >= 9:
        x, y = SPOT
        sc = 300 / 96
        k = 5
        bx, by = 455 + (x - k) * sc, 130 + (y - k) * sc
        cv.rect([bx, by, bx + 10 * sc, by + 10 * sc], outline="ok", width=3)
        # zoom into 10x10 numbers
        vals = HERO[y - k:y + k, x - k:x + k]
        reveal = int(100 * ramp(t, 10, 15))
        cv.arrow((bx + 10 * sc + 6, by + 5 * sc), (790, 270), "ok")
        fills = np.stack([vals * 255] * 3, axis=-1) * 0.55
        P.number_grid(cv, 800, 120, 40, (vals * 255).round(), fmt="{:.0f}", size=14, fills=fills,
                      count=reveal, text_fill=(240, 255, 240))
        cv.text((800, 92), "ZOOM: EACH PIXEL = 0..255", 16, "dim")
    if t >= 15.5:
        p = smooth(t, 15.5, 17)
        cv.text((800, 540), "network input  =  pixel / 255", 22, "ok")
        cv.text((800, 574), "0.0 = black     1.0 = white", 20, "dim")
    if t >= 19:
        cv.text((40, 520), "A picture is just a grid of numbers.", 26, "text", True)
        cv.text((40, 556), "The network never 'sees' a hand, only 9,216 numbers.", 22, "dim")


S1 = (25.0, [(0.0, "To the computer, a photo is not a hand. It starts as a grid of colored pixels."),
             (4.0, "We turn it gray and shrink it to 96 by 96, so the network gets 9,216 pixels."),
             (9.0, "Zoom in on any part of it and every pixel is just a number from 0 to 255."),
             (15.5, "We divide by 255 so every input is between 0 and 1. These numbers are all the network gets."),
             ], "01 PIXELS")


# =============================== scene 2: one neuron ==========================
XS_A, XS_B = (0.8, 0.2, 0.5), (0.1, 0.9, 0.2)
WS, BIAS = (1.5, -2.0, 0.8), 0.3


def neuron_diagram(cv, xs, prods, total, z, y, show):
    """Neuron diagram: inputs at left, weights and products tagged on the lines."""
    ys = [215, 330, 445]
    nx, ny = 640, 330
    for i in range(3):
        p0, p1 = (182, ys[i]), (nx - 68, ny + (ys[i] - ny) * 0.3)
        cv.circle((150, ys[i]), 30, "cyan")
        cv.center((150, ys[i]), f"x{i + 1}", 22, "cyan")
        cv.arrow(p0, p1, "steel", 2)
        at = lambda f: (p0[0] + (p1[0] - p0[0]) * f, p0[1] + (p1[1] - p0[1]) * f)
        P.tag(cv, at(0.22), f"w{i + 1} = {WS[i]:+.1f}", 18, "warn")
        if show["x"]:
            cv.text((112, ys[i]), f"{xs[i]:.1f}", 26, "cyan", True, anchor="rm")
        if show["prod"] > i:
            P.tag(cv, at(0.64), f"{xs[i]:.1f}x{WS[i]:+.1f}={prods[i]:+.2f}", 17, "text")
    cv.circle((nx, ny), 70, "ok", "panel", 4)
    cv.center((nx, ny - 8), "SUM", 24, "ok", True)
    cv.center((nx, ny + 20), "+ b", 22, "warn")
    if show["sum"]:
        cv.center((nx, ny + 105), f"sum = {total:+.2f}", 22, "text")
    if show["bias"]:
        cv.center((nx, ny + 135), f"+ bias {BIAS:+.1f} = {z:+.2f}", 22, "warn", True)
    cv.arrow((nx + 72, ny), (820, ny), "steel", 3)
    cv.rect([826, ny - 38, 950, ny + 38], outline="ok", fill="panel", width=3)
    cv.center((888, ny), "ReLU", 24, "ok", True)
    cv.arrow((952, ny), (1010, ny), "steel", 3)
    if show["out"]:
        cv.center((1075, ny), f"{y:.2f}", 34, "ok", True)
        cv.text((1075, ny + 36), "output", 16, "dim", anchor="ma")


def s2(cv, t):
    A_, B_ = t < 22.5, t >= 22.5
    if t < 32:
        xs = XS_A if A_ else XS_B
        prods = [xs[i] * WS[i] for i in range(3)]
        total = sum(prods)
        z = total + BIAS
        y = max(0.0, z)
        base = 0.0 if A_ else 22.5
        tt = t - base
        show = dict(x=tt > 3, prod=int(ramp(tt, 6, 12) * 3.001), sum=tt > 12.5, bias=tt > 15, out=tt > 18)
        neuron_diagram(cv, xs, prods, total, z, y, show)
        if tt > 18:
            px, py = P.plot_axes(cv, (860, 120, 1240, 260), (-2, 2), (-0.3, 2.2), "ReLU(z) = max(0, z)", "z", "")
            P.curve(cv, px, py, [-2, 0, 2], [0, 0, 2], "ok", 3)
            cv.circle((px(z), py(y)), 7, "warn", "warn")
        cv.text((40, 90), "ONE ARTIFICIAL NEURON" + ("" if A_ else "  (second example)"), 18, "dim")
        if tt > 3:
            cv.text((40, 540), "output = ReLU( x1*w1 + x2*w2 + x3*w3 + b )", 24, "text", True)
        if tt > 18 and z <= 0:
            cv.text((40, 580), "negative total: ReLU says 0, this neuron stays quiet", 22, "bad")
        if tt > 18 and z > 0:
            cv.text((40, 580), "positive total: ReLU passes it through", 22, "ok")
    else:
        # bridge to width / V data
        cv.text((40, 90), "TWO NEURONS MAKE A V   (your warm-up data)", 18, "dim")
        px, py = P.plot_axes(cv, (60, 120, 600, 520), (-2, 2), (-0.3, 2.2), "data: y = |x|", "x", "")
        xs_ = np.linspace(-2, 2, 21)
        for x_ in xs_:
            cv.circle((px(x_), py(abs(x_))), 5, "steel", "steel")
        p = smooth(t, 33, 36)
        xx = np.linspace(-2, 2, 41)
        P.curve(cv, px, py, xx, np.maximum(0, xx), "cyan", 3)
        if p > 0.5:
            P.curve(cv, px, py, xx, np.maximum(0, -xx), "warn", 3)
        if t > 37:
            P.curve(cv, px, py, xx, np.maximum(0, xx) + np.maximum(0, -xx), "ok", 4)
        cv.text((640, 140), "neuron 1:  ReLU(x)", 26, "cyan", True)
        cv.text((640, 190), "neuron 2:  ReLU(-x)", 26, "warn", True)
        if t > 37:
            cv.text((640, 260), "together: ReLU(x) + ReLU(-x) = |x|", 26, "ok", True)
        cv.text((640, 340), "Training = nudging each w and b", 24, "text")
        cv.text((640, 376), "until the output fits the data.", 24, "text")
        cv.text((640, 440), "Same idea as fitting the V in the", 22, "dim")
        cv.text((640, 472), "warm-up, just with many more numbers.", 22, "dim")


S2 = (40.0, [(0.0, "A neuron is a tiny calculator. Each input is multiplied by a weight, they are added up, and a bias is added."),
             (12.5, "Then ReLU: if the total is positive, pass it through. If it is negative, output zero."),
             (22.5, "A different input gives a negative total, so this neuron stays silent."),
             (33.0, "Two neurons together can already draw a V, the same shape as the warm-up data."),
             (37.5, "Learning just means nudging the weights and biases until the output fits.")], "02 ONE NEURON")


# =============================== scene 3: width and depth =====================
def layer_nodes(cv, x, n, y0=130, y1=520, color="steel", r=14):
    ys = [lerp(y0, y1, (i + 0.5) / n) for i in range(n)]
    for y in ys:
        cv.circle((x, y), r, color, "panel", 2)
    return ys


def connect(cv, xa, ya, xb, yb, fill=(58, 84, 96)):
    for y in ya:
        for y2 in yb:
            cv.line((xa + 14, y), (xb - 14, y2), fill, 1)


def param_counts():
    n = A.NET
    return dict(c1=sum(p.numel() for p in n.c1.parameters()),
                c2=sum(p.numel() for p in n.c2.parameters()),
                c3=sum(p.numel() for p in n.c3.parameters()),
                bn=sum(p.numel() for m in (n.b1, n.b2, n.b3) for p in m.parameters()),
                fc=sum(p.numel() for p in n.fc.parameters()))


def s3(cv, t):
    if t < 9:
        cv.text((40, 90), "WIDTH: how many neurons in a layer", 22, "ok", True)
        n = 2 if t < 3 else (4 if t < 6 else 8)
        xi = layer_nodes(cv, 280, 2, 200, 460, "cyan")
        xh = layer_nodes(cv, 640, n, 150, 510, "ok")
        xo = layer_nodes(cv, 1000, 1, 330, 330, "warn")
        connect(cv, 280, xi, 640, xh)
        connect(cv, 640, xh, 1000, xo)
        cv.center((640, 560), f"{n} neurons wide  =  {n} different things looked for at once", 22, "text")
    elif t < 18:
        cv.text((40, 90), "DEPTH: how many layers in a chain", 22, "ok", True)
        layers = 1 if t < 12 else (2 if t < 15 else 3)
        xs = [200]
        cols = [layer_nodes(cv, 200, 2, 220, 440, "cyan")]
        xpos = [200 + i * 220 for i in range(1, layers + 1)]
        for x in xpos:
            cols.append(layer_nodes(cv, x, 5, 160, 500, "ok"))
        xo = 200 + (layers + 1) * 220
        out = layer_nodes(cv, xo, 1, 330, 330, "warn")
        prevx = 200
        for xcol, ycol in zip(xpos + [xo], cols[1:] + [out]):
            connect(cv, prevx, cols[([200] + xpos + [xo]).index(prevx)], xcol, ycol)
            prevx = xcol
        cv.center((640, 570), f"{layers} hidden layer{'s' if layers > 1 else ''}: later layers combine what earlier ones found", 22, "text")
    else:
        cv.text((40, 90), "OUR NETWORK", 22, "ok", True)
        pc = param_counts()
        boxes = [("INPUT", "96x96 gray", None, "cyan"),
                 ("CONV 1", "8 filters 5x5", pc["c1"], "ok"),
                 ("CONV 2", "16 filters 3x3", pc["c2"], "ok"),
                 ("CONV 3", "32 filters 3x3", pc["c3"], "ok"),
                 ("LINEAR", "896 -> 7 scores", pc["fc"], "warn")]
        shown = 1 + int(ramp(t, 19, 26) * 4.001)
        x = 40
        for i, (nm, sub, cnt, col) in enumerate(boxes[:shown]):
            cv.rect([x, 190, x + 200, 350], outline=col, fill="panel", width=3)
            cv.center((x + 100, 225), nm, 24, col, True)
            cv.center((x + 100, 265), sub, 17, "text")
            if cnt is not None:
                cv.center((x + 100, 310), f"{cnt:,} numbers", 19, "warn")
            if i < len(boxes) - 1:
                cv.arrow((x + 204, 270), (x + 244, 270), "steel", 3)
            x += 248
        if t > 27:
            cv.center((640, 440), f"+ {pc['bn']} batch-norm numbers   =   {A.N_PARAMS:,} learnable numbers in total", 24, "text", True)
            cv.center((640, 485), "(a pooling step follows each conv layer, shown next)", 18, "dim")


S3 = (30.0, [(0.0, "Width is how many neurons sit side by side in a layer. More neurons, more patterns looked for at once."),
             (9.0, "Depth is how many layers are chained. Later layers combine what earlier layers found."),
             (18.0, "Our network is three convolution layers followed by one linear layer."),
             (27.0, f"In total, it learns about twelve thousand numbers.")], "03 WIDTH AND DEPTH")


# =============================== scene 4: convolution =========================
FIL = int(np.argmax(A.W1[:, 0].std(axis=(1, 2))))
WF, BF = A.W1[FIL, 0], float(A.B1[FIL])
CONV = A.conv_same(HERO, WF, BF)


def _best_spot():
    m = np.abs(CONV)
    m[:8], m[-8:], m[:, :8], m[:, -8:] = 0, 0, 0, 0
    y, x = np.unravel_index(np.argmax(m), m.shape)
    return int(x), int(y)


CX, CY = _best_spot()
PATCH = np.pad(HERO, 2)[CY:CY + 5, CX:CX + 5]
PRODS = PATCH * WF
IMG_BOX = (40, 110, 310, 380)  # input image drawn here


def s4(cv, t):
    sx, sy, ex, ey = IMG_BOX
    sc = (ex - sx) / 96
    cv.paste(gray_rgb(HERO), (sx, sy), size=(ex - sx, ey - sy), border="steel")
    cv.text((sx, sy - 24), "INPUT 96 x 96", 16, "dim")
    fx = 470
    if t < 30:
        if t >= 4:
            cv.text((fx + 275, 90), "FILTER (5x5 learned weights)", 16, "dim")
            P.number_grid(cv, fx + 275, 112, 46, WF, fmt="{:+.2f}", size=13, fills=P.signed_fills(WF), signed_color=False,
                          text_fill=(235, 245, 235))
        if t >= 8:
            # window on the image at the chosen spot
            wx, wy = sx + (CX - 2) * sc, sy + (CY - 2) * sc
            cv.rect([wx, wy, wx + 5 * sc, wy + 5 * sc], outline="ok", width=3)
            cv.text((fx, 90), "IMAGE PATCH (0..1)", 16, "dim")
            P.number_grid(cv, fx, 112, 46, PATCH, fmt="{:.2f}", size=13, fills=np.stack([PATCH * 255] * 3, -1) * 0.5,
                          text_fill=(240, 255, 240))
            cv.center((fx + 252, 232), "x", 30, "warn", True)
        if t >= 12:
            n = int(25 * ramp(t, 12, 22))
            cv.center((fx + 252 + 275 + 6 + 230 + 14, 232), "=", 1, "bg")
            cv.text((fx + 275 + 230 + 14, 90), "PRODUCTS", 16, "dim")
            P.number_grid(cv, fx + 275 + 230 + 12 + 6, 112, 46, PRODS, fmt="{:+.2f}", size=13,
                          fills=P.signed_fills(PRODS), count=n, text_fill=(235, 245, 235))
        if t >= 22:
            tot = PRODS.sum()
            cv.text((fx, 360), f"sum of the 25 products  = {tot:+.2f}", 24, "text", True)
        if t >= 24:
            cv.text((fx, 396), f"+ filter bias {BF:+.2f}  =  {CONV[CY, CX]:+.2f}", 24, "warn", True)
            cv.text((fx, 432), "this ONE number becomes one pixel of the feature map", 20, "ok")
        if t >= 26:
            cv.text((fx, 480), "Move the window one pixel, repeat.", 22, "text", True)
            cv.text((fx, 512), "9,216 positions = one feature map.", 22, "dim")
    else:
        n = int(9216 * ramp(t, 30, 44))
        yy, xx = n // 96, n % 96
        wx, wy = sx + (xx - 2) * sc, sy + (yy - 2) * sc
        cv.rect([wx, wy, wx + 5 * sc, wy + 5 * sc], outline="ok", width=3)
        m = np.maximum(CONV, 0)
        img = neon(m, 0, max(1e-6, m.max()))
        mask = np.zeros((96, 96), bool)
        mask.flat[:n] = True
        img = np.where(mask[..., None], img, np.uint8(8))
        cv.paste(img, (470, 110), size=(270, 270), border="ok")
        cv.text((470, 86), f"FEATURE MAP {FIL + 1} (after ReLU)", 16, "dim")
        cv.arrow((sx + 290, sy + 135), (460, sy + 135), "ok") if False else None
        cv.text((780, 130), "Where the filter matches its", 22, "text")
        cv.text((780, 162), "pattern, the map lights up.", 22, "text")
        cv.text((780, 220), "This filter likes:", 20, "dim")
        cv.text((780, 250), "edges of the hand and fingers.", 20, "ok")
        if t >= 44:
            cv.text((780, 330), "The network has 8 such filters in", 22, "warn", True)
            cv.text((780, 362), "layer 1, each one learned, not designed.", 22, "warn", True)
            tiles = [neon(np.maximum(A.conv_same(HERO, A.W1[i, 0], float(A.B1[i])), 0), 0, 3.0) for i in range(8)]
            for i, tl in enumerate(tiles):
                cv.paste(tl, (470 + (i % 4) * 128, 410 + (i // 4) * 110 - 0), size=(112, 100), border="line")


S4 = (50.0, [(0.0, "The key idea of a convolutional network: a small filter slides across the picture."),
             (4.0, "A filter is a five by five grid of weights. The network learns these numbers during training."),
             (8.0, "Take the five by five patch of pixels under the window..."),
             (12.0, "...multiply each pixel by the matching weight..."),
             (22.0, "...add up all 25 products and the bias. That single number is one pixel of a new image."),
             (26.0, "Slide the window across every position, and you get a whole feature map."),
             (30.0, "Bright pixels show where the picture matches what this filter is looking for."),
             (44.0, "Our first layer has eight filters, so it makes eight feature maps.")], "04 CONVOLUTION")

SCENES = [("s0", *S0, s0), ("s1", *S1, s1), ("s2", *S2, s2), ("s3", *S3, s3), ("s4", *S4, s4)]

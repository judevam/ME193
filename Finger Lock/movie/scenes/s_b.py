"""Scenes 5-8: ReLU + pooling, the whole network, loss, backprop + gradient descent."""
import math

import numpy as np

import assets as A
import parts as P
from kit import COL, lerp, ramp, smooth, gray_rgb, neon
from s_a import HERO, HERO_P, CONV, CX, CY, FIL

CLS = A.CLASSES
LAB = ["0 fist", "1", "2", "3", "4", "5", "empty"]

# ---------------- scene 5: ReLU and max-pooling ---------------------------------
X0, Y0 = max(0, min(88, CX - 4)), max(0, min(88, CY - 4))
RAW = CONV[Y0:Y0 + 8, X0:X0 + 8]
RELU = np.maximum(RAW, 0)
POOLED = RELU.reshape(4, 2, 4, 2).max(axis=(1, 3))
FULL_RELU = np.maximum(CONV, 0)
FULL_POOL = FULL_RELU.reshape(48, 2, 48, 2).max(axis=(1, 3))


def s5(cv, t):
    cell = 52
    gx, gy = 60, 140
    if t < 26:
        cv.text((gx, 100), "A 8 x 8 PIECE OF THE FEATURE MAP", 16, "dim")
        # animate negatives -> 0
        neg_idx = [i for i in range(64) if RAW.flat[i] < 0]
        cut = int(len(neg_idx) * ramp(t, 6, 13))
        done = set(neg_idx[:cut])
        vals = RAW.copy()
        for i in done:
            vals.flat[i] = 0.0
        fills = P.signed_fills(vals, scale=max(1.0, np.abs(RAW).max()))
        P.number_grid(cv, gx, gy, cell, vals, fmt="{:+.1f}", size=15, fills=fills, text_fill=(235, 245, 235))
        if t >= 3:
            cv.text((gx, gy + 8 * cell + 18), "red = negative,  green = positive", 17, "dim")
        if 6 <= t < 14:
            cv.text((gx, gy + 8 * cell + 48), "ReLU: every negative value becomes 0", 22, "ok", True)
        if t >= 14:
            # max pool windows
            k = int(16 * ramp(t, 14, 23))
            for i in range(min(k + 1, 16)):
                r, c = divmod(i, 4)
                col = "warn" if i == k else "gold"
                cv.rect([gx + c * 2 * cell, gy + r * 2 * cell, gx + c * 2 * cell + 2 * cell, gy + r * 2 * cell + 2 * cell],
                        outline=col, width=3 if i == k else 1)
            ox = 640
            cv.text((ox, 100), "MAX-POOL: keep the biggest of each 2 x 2", 16, "dim")
            vals4 = np.zeros((4, 4))
            P.number_grid(cv, ox, gy, 78, POOLED, fmt="{:.1f}", size=18,
                          fills=np.stack([np.clip(POOLED / max(1e-6, POOLED.max()), 0, 1) * 180] * 3, -1) * [0.2, 1, 0.5],
                          count=min(k, 16), text_fill=(235, 255, 240))
            cv.text((gx, gy + 8 * cell + 48), "max-pool: 4 numbers in, the biggest one out", 22, "warn", True)
            if k < 16:
                r, c = divmod(k, 4)
                blk = RELU[2 * r:2 * r + 2, 2 * c:2 * c + 2]
                cv.text((ox, gy + 4 * 78 + 24), "biggest of  " + "  ".join(f"{v:.1f}" for v in blk.ravel()) + f"  =  {blk.max():.1f}", 20, "text")
    else:
        cv.text((60, 100), "THE WHOLE MAP, BEFORE AND AFTER", 16, "dim")
        cv.paste(neon(FULL_RELU, 0, max(1e-6, FULL_RELU.max())), (60, 140), size=(384, 384), border="ok")
        cv.text((60, 534), "96 x 96", 22, "text", True)
        cv.arrow((460, 330), (560, 330), "warn", 4)
        cv.center((510, 305), "pool", 20, "warn")
        cv.paste(neon(FULL_POOL, 0, max(1e-6, FULL_RELU.max())), (580, 140 + 96), size=(192, 192), border="ok")
        cv.text((580, 534 - 96 + 10), "48 x 48", 22, "text", True)
        if t >= 29:
            cv.text((830, 170), "Half the size,", 24, "text", True)
            cv.text((830, 204), "strongest signals kept.", 24, "text", True)
            cv.text((830, 270), "Next layer has less to look at,", 20, "dim")
            cv.text((830, 298), "and a finger a few pixels", 20, "dim")
            cv.text((830, 326), "to the side still gives", 20, "dim")
            cv.text((830, 354), "nearly the same answer.", 20, "dim")


S5 = (35.0, [(0.0, "Right now the feature map has positive and negative numbers. Here is a small piece of it."),
             (6.0, "ReLU is the same function from before: every negative number becomes zero."),
             (14.0, "Then pooling. Look at each two by two block and keep only the biggest number."),
             (26.0, "Applied to the whole map, pooling halves its size and keeps the strongest signals."),
             (29.0, "A smaller map is cheaper, and it makes the network tolerant to small shifts of the hand.")], "05 RELU AND POOLING")


# ---------------- scene 6: the whole network ---------------------------------
P6, Z6, (A1, A2, A3), FEATS = A.run(HERO)


def tiles(cv, maps, x0, y0, cols, tile, lo, hi, n=None):
    n = len(maps) if n is None else n
    for i in range(n):
        r, c = divmod(i, cols)
        cv.paste(neon(maps[i], lo, hi), (x0 + c * (tile + 3), y0 + r * (tile + 3)), size=(tile, tile), nearest=False)


def s6(cv, t):
    cv.paste(gray_rgb(HERO), (30, 130), size=(110, 110), border="cyan")
    cv.text((30, 104), "INPUT", 16, "dim")
    # stage 1
    x1 = 190
    if t >= 1:
        n = int(8 * ramp(t, 1, 4))
        cv.text((x1, 104), "CONV 1 + POOL", 16, "dim")
        tiles(cv, A1, x1, 130, 4, 64, 0, A1.max(), n)
        cv.arrow((144, 190), (192, 190), "steel", 3)
        cv.text((x1, 270), "8 maps, 48 x 48", 15, "ok")
    x2 = 500
    if t >= 5:
        n = int(16 * ramp(t, 5, 8))
        cv.text((x2, 104), "CONV 2 + POOL", 16, "dim")
        tiles(cv, A2, x2, 130, 4, 48, 0, A2.max(), n)
        cv.arrow((x1 + 272, 190), (x2 - 8, 190), "steel", 3)
        cv.text((x2, 336), "16 maps, 24 x 24", 15, "ok")
    x3 = 745
    if t >= 9:
        n = int(32 * ramp(t, 9, 12))
        cv.text((x3, 104), "CONV 3 + POOL", 16, "dim")
        tiles(cv, A3, x3, 130, 8, 34, 0, A3.max(), n)
        cv.arrow((x2 + 206, 190), (x3 - 8, 190), "steel", 3)
        cv.text((x3, 282), "32 maps, 12 x 12", 15, "ok")
    if t >= 13:
        # concatenated features
        cv.text((30, 358), "SHRINK EACH MAP TO 4 x 4 AND LINE THEM ALL UP", 16, "dim")
        n = int(896 * ramp(t, 13, 18))
        strip = neon(FEATS[None, :], 0, FEATS.max())
        view = np.zeros((1, 896, 3), np.uint8)
        view[0, :n] = strip[0, :n]
        cv.paste(np.repeat(view, 1, 0), (30, 382), size=(1180, 24), border="ok")
        cv.text((30, 412), "8 x 16 + 16 x 16 + 32 x 16 = 896 numbers", 18, "warn", True)
    if t >= 19:
        cv.text((30, 448), "LINEAR LAYER: 896 numbers -> 7 scores", 16, "dim")
        k = int(7 * ramp(t, 19, 23))
        sc = Z6
        top = float(np.abs(sc).max())
        for i in range(7):
            y = 472 + i * 21
            if i < k:
                cv.text((30, y), LAB[i], 16, "text")
                w = int(abs(sc[i]) / top * 220)
                x = 130 if sc[i] >= 0 else 130
                cv.rect([130, y + 2, 130 + w, y + 16], fill="ok" if sc[i] >= 0 else "bad", width=0)
                cv.text((360, y), f"{sc[i]:+.2f}", 16, "text")
    if t >= 25:
        cv.text((640, 448), "SOFTMAX: scores -> probabilities that add to 100%", 16, "dim")
        k = int(7 * ramp(t, 25, 30))
        for i in range(7):
            y = 472 + i * 21
            if i < k:
                cv.text((640, y), LAB[i], 16, "text")
                top = i == int(np.argmax(P6))
                cv.rect([740, y + 2, 740 + int(P6[i] * 380), y + 16], fill="ok" if top else "steel", width=0)
                cv.text((1130, y), f"{P6[i] * 100:5.1f}%", 16, "ok" if top else "text", top)
    if t >= 32:
        cv.center((930, 420), "", 1)
        cv.text((640, 628 - 30 - 0), "", 1)


S6 = (38.0, [(0.0, "Put it together. The picture goes through convolution one, then pooling..."),
             (5.0, "...then convolution two, with sixteen filters that look at the first maps and find finger shapes..."),
             (9.0, "...then convolution three, with thirty-two filters that combine those into bigger patterns."),
             (13.0, "Now shrink every map to four by four and line all the numbers up. That is 896 numbers."),
             (19.0, "One linear layer turns those 896 numbers into seven scores, one per answer."),
             (25.0, "Softmax turns the scores into probabilities. Here the network is almost sure this is a two.")],
      "06 THE WHOLE NETWORK")


# ---------------- scene 7: loss ---------------------------------------------------
_img = P.pick_image("2", "unsure")
if _img is not None:
    U_CROP, U_G, U_PATH, U_P = _img
else:  # fallback
    U_CROP, U_G, U_PATH, U_P = P.pick_image("1", "unsure")
U_TRUE = CLS.index("2") if _img is not None else CLS.index("1")
U_PT = float(U_P[U_TRUE])
U_LOSS = -math.log(U_PT)


def s7(cv, t):
    cv.paste(gray_rgb(U_G), (50, 130), size=(220, 220), border="warn")
    cv.text((50, 104), "A TRAINING PHOTO", 16, "dim")
    cv.text((50, 362), f"correct answer: {LAB[U_TRUE].split()[0]}", 24, "ok", True)
    k = int(7 * ramp(t, 1.5, 6))
    cv.text((330, 104), "THE NETWORK'S PROBABILITIES", 16, "dim")
    for i in range(7):
        y = 134 + i * 36
        if i < k:
            cv.text((330, y), LAB[i], 20, "ok" if i == U_TRUE else "text")
            cv.rect([430, y + 3, 430 + int(U_P[i] * 330), y + 24], fill="ok" if i == U_TRUE else "steel", width=0)
            cv.text((770, y), f"{U_P[i] * 100:5.1f}%", 20, "ok" if i == U_TRUE else "text", i == U_TRUE)
    if t >= 8:
        cv.text((330, 400), f"probability given to the right answer:  p = {U_PT:.2f}", 22, "text", True)
    if t >= 11:
        cv.text((330, 440), "loss = - ln( p )", 34, "warn", True)
        cv.text((330, 486), f"     = - ln( {U_PT:.2f} ) = {U_LOSS:.2f}", 30, "warn", True)
    if t >= 15:
        px, py = P.plot_axes(cv, (930, 120, 1250, 400), (0, 1), (0, 4.2), "loss = -ln(p)", "p", "loss")
        xs = np.linspace(0.02, 1, 60)
        P.curve(cv, px, py, xs, -np.log(xs), "steel", 3)
        pts = [(0.99, "sure + right", "ok"), (U_PT, "unsure", "warn"), (0.03, "sure + wrong", "bad")]
        for i, (p, lab, col) in enumerate(pts):
            if t >= 16 + i * 2.5:
                cv.circle((px(p), py(-math.log(p))), 8, col, col)
                cv.text((px(p) + (12 if p < 0.9 else -10), py(-math.log(p)) + (4 if p < 0.1 else -26)),
                        f"{lab}: {-math.log(p):.2f}", 15, col, anchor="la" if p < 0.9 else "ra")
    if t >= 24:
        cv.text((930, 430), "One number: how wrong", 20, "text")
        cv.text((930, 458), "the network is. Training =", 20, "text")
        cv.text((930, 486), "make the average small.", 20, "text", True)


S7 = (30.0, [(0.0, "To learn, the network needs a score for how wrong it is. Here is one training photo."),
             (1.5, "These are the probabilities the network gave for each answer."),
             (8.0, "We look only at the probability it gave to the correct answer."),
             (11.0, "Cross-entropy loss is minus the natural log of that probability."),
             (15.0, "Confident and right gives a tiny loss. Unsure gives a medium loss. Confident and wrong gives a huge loss."),
             (24.0, "Training means making this number small, on average, across every photo.")], "07 LOSS")


# ---------------- scene 8: backprop + gradient descent ---------------------------
FC_W = A.NET.fc.weight.detach().numpy()
_, _, _, F_U = A.run(U_G)
J = int(np.argmax(F_U))
W_OLD = float(FC_W[U_TRUE, J])
DZ = float(U_P[U_TRUE] - 1.0)
DW = DZ * float(F_U[J])
LR = 0.1
W_NEW = W_OLD - LR * DW
TC = A.training_curve()


def s8(cv, t):
    if t < 28:
        cv.text((50, 96), "ONE WEIGHT, ONE PHOTO  (the last layer)", 16, "dim")
        pos = dict(f=(120, 230), w=(400, 230), z=(660, 230), p=(900, 230), L=(1130, 230))
        # forward pass
        cv.circle(pos["f"], 46, "cyan", "panel", 3)
        cv.center(pos["f"], f"{F_U[J]:.2f}", 22, "cyan", True)
        cv.text((pos["f"][0], 292), "a feature", 15, "dim", anchor="ma")
        cv.arrow((170, 230), (350, 230), "steel", 3)
        P.tag(cv, (262, 205), f"w = {W_OLD:+.3f}", 18, "warn")
        cv.rect([360, 195, 440, 265], outline="warn", fill="panel", width=3)
        cv.center((400, 230), "x w", 24, "warn", True)
        cv.arrow((444, 230), (610, 230), "steel", 3)
        cv.circle(pos["z"], 46, "ok", "panel", 3)
        cv.center(pos["z"], f"{Z_SC:+.1f}" if False else "score", 18, "ok", True)
        cv.text((pos["z"][0], 292), "(+ other features)", 15, "dim", anchor="ma")
        cv.arrow((706, 230), (850, 230), "steel", 3)
        cv.center(pos["p"], f"p={U_PT:.2f}", 22, "ok", True)
        cv.text((pos["p"][0], 292), "softmax", 15, "dim", anchor="ma")
        cv.arrow((950, 230), (1085, 230), "steel", 3)
        cv.center(pos["L"], f"L={U_LOSS:.2f}", 24, "warn", True)
        cv.text((pos["L"][0], 292), "loss", 15, "dim", anchor="ma")
        cv.text((50, 340), "FORWARD:  feature  ->  x weight  ->  score  ->  probability  ->  loss", 18, "ok")
        if t >= 8:
            # backward arrows
            for a, b in [((1085, 160), (950, 160)), ((850, 160), (706, 160)), ((610, 160), (444, 160)), ((350, 160), (170, 160))]:
                cv.arrow(a, b, "bad", 3)
            cv.text((50, 372), "BACKWARD: how much did each step change the loss?  (chain rule)", 18, "bad")
        if t >= 12:
            cv.text((50, 420), f"slope of loss at the score:      p - 1 = {U_PT:.2f} - 1 = {DZ:+.2f}", 22, "text", True)
        if t >= 16:
            cv.text((50, 460), f"slope at the weight:   {DZ:+.2f} x feature {F_U[J]:.2f} = {DW:+.3f}", 22, "text", True)
        if t >= 20:
            cv.text((50, 520), f"update:  w_new = w - lr x slope = {W_OLD:+.3f} - {LR} x ({DW:+.3f}) = {W_NEW:+.3f}", 22, "warn", True)
            cv.text((50, 556), "(training used Adam, a smarter version of this same step)", 16, "dim")
        if t >= 24:
            cv.text((50, 590), "Do this for all 12,407 numbers, flowing back through every layer.", 20, "ok", True)
    elif t < 40:
        cv.text((50, 96), "GRADIENT DESCENT: walk downhill on the loss", 16, "dim")
        px, py = P.plot_axes(cv, (60, 120, 760, 560), (-3, 3), (0, 9.5), "loss as one weight changes", "weight", "loss")
        xs = np.linspace(-3, 3, 80)
        P.curve(cv, px, py, xs, xs ** 2 * 0.9 + 0.3, "steel", 3)
        n = int(7 * ramp(t, 29, 38))
        w = 2.6
        path = [w]
        for _ in range(7):
            w = w - 0.55 * 0.9 * 2 * w * 0.55
            path.append(w)
        for i in range(n + 1):
            cv.circle((px(path[i]), py(path[i] ** 2 * 0.9 + 0.3)), 9, "warn", "warn" if i == n else "panel", 3)
        cv.text((800, 160), "slope tells which way is downhill", 22, "text", True)
        cv.text((800, 200), "small steps, over and over:", 22, "text")
        cv.text((800, 240), "w_new = w - lr x slope", 26, "warn", True)
        cv.text((800, 320), "steeper = bigger step", 20, "dim")
        cv.text((800, 350), "flat valley = stop", 20, "dim")
    else:
        cv.text((50, 96), f"THE REAL TRAINING RUN  (150 epochs, {A.split_info()[0]:,} photos)", 16, "dim")
        px, py = P.plot_axes(cv, (60, 120, 700, 540), (0, 150), (0, 2.1), "loss", "epoch", "")
        n = max(2, int(len(TC["epoch"]) * ramp(t, 41, 54)))
        P.curve(cv, px, py, TC["epoch"][:n], TC["tl"][:n], "warn", 3, ymax=2.05)
        P.curve(cv, px, py, TC["epoch"][:n], TC["vl"][:n], "ok", 3, ymax=2.05)
        cv.text((80, 570), "training loss", 18, "warn")
        cv.text((260, 570), "validation loss (photos it did not train on)", 18, "ok")
        px2, py2 = P.plot_axes(cv, (740, 120, 1240, 340), (0, 150), (0, 1.0), "accuracy", "epoch", "")
        P.curve(cv, px2, py2, TC["epoch"][:n], TC["ta"][:n], "warn", 3)
        P.curve(cv, px2, py2, TC["epoch"][:n], TC["va"][:n], "ok", 3)
        e = TC["epoch"][n - 1]
        cv.text((760, 370), f"epoch {int(e)}:", 22, "text", True)
        cv.text((760, 402), f"training loss {TC['tl'][n - 1]:.2f}", 20, "warn")
        cv.text((760, 430), f"validation accuracy {TC['va'][n - 1] * 100:.0f}%", 20, "ok", True)
        if t >= 55:
            cv.text((740, 480), "Loss falls, accuracy rises.", 22, "text", True)
            cv.text((740, 512), "That is learning.", 22, "ok", True)
            cv.text((740, 548), "(validation jumps around: it has few photos)", 16, "dim")


Z_SC = 0
S8 = (60.0, [(0.0, "Backpropagation answers one question: which weights caused the error? Take one feature and one weight."),
             (8.0, "Now walk backward from the loss. At each step we ask how much the loss changes."),
             (12.0, "At the score, the slope is just the probability minus one."),
             (16.0, "Multiply by the feature to get the slope for the weight. That is the chain rule."),
             (20.0, "Gradient descent nudges the weight a small step against the slope."),
             (24.0, "The same happens to every filter and weight in the whole network."),
             (29.0, "Picture the loss as a valley. The slope tells us which way is down, so we take small steps."),
             (40.0, "Here is our real training run. The loss falls and the accuracy climbs."),
             (55.0, "After 150 rounds of this, the network reads fingers.")], "08 BACKPROP + GRADIENT DESCENT")

SCENES = [("s5", *S5, s5), ("s6", *S6, s6), ("s7", *S7, s7), ("s8", *S8, s8)]

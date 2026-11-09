"""Scenes 9-12: data + augmentation, what it learned, the lock policy, wrap-up."""
import glob
import os
import sys

import cv2
import numpy as np
import torch
import torchvision.transforms.v2 as T

import assets as A
import parts as P
from kit import COL, lerp, ramp, smooth, gray_rgb, neon
from s_a import HERO, HERO_CROP, DEMO_CODE

sys.path.insert(0, os.path.join(A.ROOT, "src"))
from lock_policy import LockPolicy  # noqa: E402

CLS = A.CLASSES
LAB = ["0 fist", "1", "2", "3", "4", "5", "empty"]

# ---------------- scene 9: data + augmentation ---------------------------------
COUNTS = A.counts()
AUG = T.Compose([T.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.85, 1.15)),
                 T.RandomHorizontalFlip(), T.ColorJitter(brightness=0.3, contrast=0.3)])
torch.manual_seed(7)
VARIANTS = [AUG(torch.from_numpy(HERO)[None, None])[0, 0].clamp(0, 1).numpy() for _ in range(8)]


def s9(cv, t):
    cv.text((50, 96), "THE DATASET  (photos taken with the same webcam)", 16, "dim")
    mx = max(COUNTS.values())
    for i, c in enumerate(CLS):
        n = int(COUNTS[c] * smooth(t, 1 + i * 0.4, 3 + i * 0.4))
        x = 70 + i * 74
        h = int(n / mx * 260)
        cv.rect([x, 420 - h, x + 52, 420], fill="ok" if c != "empty" else "cyan", width=0)
        cv.center((x + 26, 438), LAB[i].split()[0], 18, "text")
        if n:
            cv.center((x + 26, 410 - h), str(n), 16, "text")
    total = sum(COUNTS.values())
    if t >= 6:
        cv.text((70, 470), f"{total:,} photos, 3 people, 7 labels", 22, "warn", True)
    if t >= 8:
        cv.text((650, 96), "AUGMENTATION: one photo becomes many", 16, "dim")
        cv.paste(gray_rgb(HERO), (650, 130), size=(170, 170), border="warn")
        cv.text((650, 308), "original", 16, "dim")
        cv.arrow((826, 215), (860, 215), "warn", 3)
        k = int(8 * ramp(t, 9, 16))
        for i in range(k):
            r, c = divmod(i, 4)
            cv.paste(gray_rgb(VARIANTS[i]), (872 + c * 100, 130 + r * 100), size=(92, 92), border="line")
        cv.text((872, 340), "random rotate, shift, zoom, flip, brightness", 15, "dim")
        cv.text((650, 380), "Each epoch the network sees new versions,", 20, "text")
        cv.text((650, 408), f"so it cannot just memorize {total:,} photos.", 20, "text")
    if t >= 18:
        cv.rect([60, 500, 1230, 610], outline="line", fill="panel", width=1)
        cv.text((80, 512), "TEST ON PHOTOS IT NEVER TRAINED ON", 16, "dim")
        c1, n1, f1 = A.test_accuracy(A.LOG)
        c2, n2, f2 = A.test_accuracy(A.HOLDOUT_LOG)
        cv.text((80, 540), f"same sessions, unseen frames:   {f1 * 100:.0f}%  ({c1} / {n1})", 24, "ok", True)
        if t >= 21:
            cv.text((80, 574), f"a new person (Sophie), trained on the other two:   {f2 * 100:.0f}%  ({c2} / {n2})", 24, "warn", True)


S9 = (25.0, [(0.0, f"We took the photos ourselves, with the same webcam the lock uses. Seven labels, {sum(COUNTS.values()):,} photos, three people."),
             (8.0, "To get more from few photos, each training round shows randomly rotated, shifted, zoomed and flipped copies."),
             (18.0, f"Then we test on photos it never trained on. {A.test_accuracy(A.LOG)[2] * 100:.0f} percent on new frames from familiar sessions."),
             (21.0, f"But only {A.test_accuracy(A.HOLDOUT_LOG)[2] * 100:.0f} percent on a new person's hand. More hands in the training data would fix that.")], "09 DATA")


# ---------------- scene 10: what did it learn (Grad-CAM) ---------------------------
def overlay(gray, cam):
    base = cv2.cvtColor(gray_rgb(gray), cv2.COLOR_GRAY2RGB)
    heat = cv2.cvtColor(cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return (base * 0.55 + heat * 0.45).astype(np.uint8)


G_ITEMS = []
for lab in ("1", "3", "5", "0"):
    crop, g, f, p = P.pick_image(lab)
    cam, k = A.gradcam(g)
    G_ITEMS.append((lab, overlay(g, cam), CLS[k], float(p[k])))


def s10(cv, t):
    cv.text((50, 96), "WHERE DOES THE NETWORK LOOK?  (Grad-CAM: red = most important)", 16, "dim")
    n = int(4 * ramp(t, 1, 11)) + (1 if t >= 1 else 0)
    for i, (lab, img, saw, conf) in enumerate(G_ITEMS[:min(n, 4)]):
        x = 40 + i * 305
        cv.paste(img, (x, 130), size=(280, 280), border="ok")
        cv.text((x, 418), f"true {lab}  ->  saw {saw}  ({conf * 100:.0f}%)", 18, "text")
    if t >= 13:
        cv.text((50, 470), "3, 4, 5:  when it is right, it looks at the FINGERTIPS.", 22, "ok", True)
    if t >= 19:
        cv.text((50, 508), "0 and 1:  it mostly looks at EMPTY desk, where more fingers would be.", 22, "warn", True)
        cv.text((50, 540), "No finger there is evidence too.", 20, "dim")
    if t >= 27:
        cv.text((50, 590), "On a new hand it gets wrong, it often looks at empty areas instead of the fingers.", 18, "dim")


S10 = (35.0, [(0.0, "Grad-CAM shows where the network looked when it made its choice. Red means more important."),
              (13.0, "For three, four and five, when it is right, it looks at the fingertips, as hoped."),
              (19.0, "For zero and one, it mostly looks at empty desk. The absence of fingers is evidence."),
              (27.0, "On a new hand it gets wrong, it often looks at empty areas instead of the fingers.")],
       "10 WHAT IT LEARNED")


# ---------------- scene 11: lock policy on REAL predictions ------------------------
def _bursts(label):
    """All bursts (lists of files) for this label, from every person."""
    out = []
    for person in ("jude", "mia"):
        stamps = {}
        for f in A.files_for(label, person):
            stamps.setdefault("_".join(os.path.basename(f).split("_")[1:3]), []).append(f)
        out += [stamps[k] for k in sorted(stamps)]
    return out


def _preds(files):
    out = []
    for f in files:
        _, g = A.load(f)
        p = A.run(g)[0]
        k = int(p.argmax())
        out.append((CLS[k], float(p[k])))
    return out


def _clean_run(label, n=14):
    """Real consecutive frames: a flicker (rejected), then 8+ good frames in a row (accepted)."""
    def good(x):
        return x[0] == label and x[1] >= 0.75
    fallback = None
    for files in _bursts(label):
        pr = _preds(files)
        for s0 in range(3, len(pr) - 11):
            if all(good(x) for x in pr[s0:s0 + 8]):
                if any(not good(x) for x in pr[s0 - 3:s0]):
                    return pr[s0 - 3:s0 - 3 + n]
                if fallback is None:
                    fallback = pr[s0:s0 + n]
    return fallback


def _empty_frames():
    return _preds(_bursts("empty")[0])


EMPTY = _empty_frames()[:10]
SEQ = list(EMPTY)
for d in DEMO_CODE:
    SEQ += _clean_run(d) + EMPTY[:6]
STEPS = []
_pol = LockPolicy(DEMO_CODE)
_pol.arm(0.0)
for i, (lab, conf) in enumerate(SEQ):
    now = 0.1 * (i + 1)
    _pol.update(lab, conf, now)
    STEPS.append(dict(lab=lab, conf=conf, hold=_pol.hold, entered=len(_pol.entered), state=_pol.state,
                      clear=_pol.need_clear, cand=_pol.candidate))
assert STEPS[-1]["state"] == "UNLOCKED", "the real-frame replay did not unlock; pick other frames"
STATES = [("LOCKED", 60, 110), ("ARMED", 330, 110), ("OPEN", 640, 70), ("DENIED", 640, 160), ("LOCKDOWN", 900, 160)]


def s11(cv, t):
    cur = "ARMED"
    i = max(0, min(len(STEPS) - 1, int((t - 5) / 0.38))) if t >= 5 else 0
    st = STEPS[i] if t >= 5 else dict(lab="-", conf=0, hold=0, entered=0, state="LOCKED", clear=False, cand=None)
    cur = "LOCKED" if t < 5 else ("OPEN" if st["state"] == "UNLOCKED" else st["state"])
    # state machine
    for name, x, y in STATES:
        on = (name == cur)
        col = {"LOCKED": "bad", "ARMED": "warn", "OPEN": "ok", "DENIED": "bad", "LOCKDOWN": "bad"}[name]
        cv.rect([x, y, x + 150, y + 54], outline=col if on else "line", fill="panel2" if on else "panel", width=4 if on else 2)
        cv.center((x + 75, y + 27), name, 22, col if on else "dim", True)
    cv.arrow((212, 137), (326, 137), "steel", 3)
    cv.text((218, 112), "button", 15, "dim")
    cv.arrow((482, 125), (636, 100), "steel", 3)
    cv.text((494, 92), "right code", 15, "ok")
    cv.arrow((482, 150), (636, 180), "steel", 3)
    cv.text((494, 182), "wrong code", 15, "bad")
    cv.arrow((794, 187), (896, 187), "steel", 3)
    cv.text((800, 214), "3 wrong in a row", 15, "bad")
    # rules
    cv.text((50, 250), "RULES FOR ACCEPTING A DIGIT", 16, "dim")
    rules = ["same digit, 8 frames in a row", "confidence at least 75%", "hand must leave (fist or empty) before the next digit",
             "3 digits within 25 seconds"]
    for j, r in enumerate(rules):
        if t >= 1 + j * 0.8:
            cv.text((50, 278 + j * 26), "> " + r, 19, "text")
    # prediction strip (real predictions)
    if t >= 5:
        cv.text((640, 250), "REAL PREDICTIONS, ONE PER CAMERA FRAME", 16, "dim")
        lo = max(0, i - 17)
        for j, k in enumerate(range(lo, i + 1)):
            lab, conf = SEQ[k]
            x = 640 + j * 33
            good = conf >= 0.75
            colr = "ok" if (good and lab in "12345" and len(lab) == 1) else ("cyan" if lab in ("0", "empty") else "warn")
            cv.rect([x, 276, x + 30, 330], outline=colr, fill="panel", width=2)
            cv.center((x + 15, 296), "e" if lab == "empty" else lab, 22, colr, True)
            cv.center((x + 15, 318), f"{int(conf * 100)}", 13, "dim")
        cv.text((640, 338), "digit = prediction,  number below = confidence %", 14, "dim")
        # hold bar
        cv.text((640, 372), f"same digit in a row: {st['hold']} / 8", 20, "text", True)
        for h in range(8):
            cv.rect([640 + h * 36, 400, 640 + h * 36 + 30, 420], fill="ok" if h < st["hold"] else "panel2", outline="line", width=1)
        msg = ("digit locked, remove your hand" if st["clear"] else
               ("ready for the next digit" if st["entered"] else "show the first digit"))
        if st["state"] == "UNLOCKED":
            msg = "ACCESS GRANTED"
        cv.text((640, 436), msg, 22, "ok" if st["state"] == "UNLOCKED" else "warn", True)
        P.tumblers(cv, 60, 410, st["entered"] if st["state"] != "UNLOCKED" else 3, w=64, h=78)
        cv.text((60, 392), "CODE ENTERED", 14, "dim")
    if t >= 33:
        cv.rect([50, 520, 1230, 620], outline="line", fill="panel", width=1)
        cv.text((70, 530), "WRONG CODE:  denied, latch stays shut.", 20, "bad", True)
        cv.text((70, 560), "3 wrong codes in a row:  silent alarm, entry disabled for 20 seconds.", 20, "bad")
        cv.text((70, 590), "The code is stored only as a salted hash, never as plain text.", 18, "dim")


S11 = (40.0, [(0.0, "The network only guesses, frame after frame. The lock policy turns guesses into a decision."),
              (1.0, "A digit counts only when the same number is seen eight frames in a row, with at least 75 percent confidence."),
              (5.0, "These are real predictions from the camera. One noisy frame never counts."),
              (15.0, "After each digit, the hand must leave before the next one can start."),
              (28.0, "Three digits entered, the code matches, and the latch opens."),
              (33.0, "A wrong code is denied. Three wrong codes in a row trigger a lockdown.")], "11 THE LOCK POLICY")


# ---------------- scene 12: limits + wrap-up ------------------------------------------
def s12(cv, t):
    if t < 14:
        cv.text((50, 96), "LIMITS", 22, "bad", True)
        items = [f"New hands: {A.test_accuracy(A.LOG)[2] * 100:.0f}% on familiar photos, only {A.test_accuracy(A.HOLDOUT_LOG)[2] * 100:.0f}% on a new person",
                 "Lighting or a bumped camera changes the pixels",
                 "Only 125 possible codes (5 x 5 x 5)",
                 "A printed photo of a hand might fool it"]
        for j, it in enumerate(items):
            if t >= 1 + j * 2.5:
                cv.text((50, 150 + j * 70), f"{j + 1}.  {it}", 26, "text")
    elif t < 20:
        cv.text((50, 96), "ON THE UNO Q", 22, "ok", True)
        cv.text((50, 160), "Same weights, no PyTorch: just NumPy.", 28, "text", True)
        cv.text((50, 220), "about 29 ms per photo  (laptop: 3 ms)", 26, "warn")
        cv.text((50, 270), "about 10 readings per second, live camera", 26, "warn")
        cv.text((50, 320), "latch, LED matrix and vault page all run on the board", 26, "warn")
        P.vault_door(cv, 1000, 420, 150, open_p=smooth(t, 16, 18), accent="ok")
    else:
        cv.center((640, 280), "VAULT 193", 64, "ok", True)
        cv.center((640, 350), "a neural network that reads your fingers", 26, "text")
        cv.center((640, 410), "ME193 midterm", 20, "dim")


S12 = (24.0, [(0.0, "Of course it has limits. New hands, lighting, a small code space, and a printed photo could all be problems."),
              (14.0, "On the UNO Q, the same weights run with just NumPy, about thirty milliseconds a photo, live."),
              (20.0, "That is how a neural network learns to read your fingers.")], "12 WRAP-UP")

SCENES = [("s9", *S9, s9), ("s10", *S10, s10), ("s11", *S11, s11), ("s12", *S12, s12)]

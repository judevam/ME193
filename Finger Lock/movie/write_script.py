"""Writes MOVIE_SCRIPT.md: the narration, scene by scene, with timing and word budgets."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "scenes"))
import assets as A  # noqa: E402
import s_a  # noqa: E402
import s_b  # noqa: E402

_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def words(n):
    n = int(n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("-" + _ONES[n % 10] if n % 10 else "")
    if n < 1000:
        return _ONES[n // 100] + " hundred" + (" " + words(n % 100) if n % 100 else "")
    return words(n // 100) + " hundred" + (" " + words(n % 100) if n % 100 else "")   # 1690 -> sixteen hundred ninety


def say(x, nd=2):
    """-1.831 -> 'minus one point eight three'"""
    sign = "minus " if x < 0 else ""
    t = f"{abs(x):.{nd}f}"
    whole, frac = t.split(".")
    return sign + words(int(whole)) + " point " + " ".join(_ONES[int(c)] for c in frac)


def pct(f):
    return words(round(f * 100))


TOTAL = sum(A.counts().values())
ACC_SAME = A.test_accuracy(A.LOG)[2]
ACC_NEW = A.test_accuracy(A.HOLDOUT_LOG)[2]
HERO_PCT = float(s_b.P6.max())

SCENES = [
    ("Hook", 8, "00 HOOK", [
        "How does a vault read your fingers? Show three numbers and the door opens. Here is how."]),
    ("A picture is just numbers", 15, "01 PIXELS", [
        "To a computer, a photo is a grid of pixels. I shrink mine to ninety-six by ninety-six, gray, so nine thousand numbers between zero and one are all the network gets."]),
    ("One neuron", 27, "02 ONE NEURON", [
        "A neuron is a tiny calculator: multiply each input by a weight, add them up, add a bias. Then ReLU: positive totals pass, negative totals become zero.",
        "Two neurons, ReLU of x and ReLU of minus x, add up to a V, like our warm-up data. Learning means adjusting weights until the shape fits."]),
    ("Width and depth", 18, "03 WIDTH AND DEPTH", [
        "Width is how many neurons sit side by side. Depth is how many layers are chained. Ours has three convolution layers of eight, sixteen and thirty-two filters, then a linear layer: about twelve thousand learned numbers."]),
    ("Convolution", 35, "04 CONVOLUTION", [
        "The key idea is convolution. A filter is a small grid of weights, here five by five, and the network learns them.",
        "Multiply each pixel under the filter by its weight and add up the twenty-five products: " + say(float(s_a.PRODS.sum())) + ", plus the bias, gives " + say(float(s_a.CONV[s_a.CY, s_a.CX])) + ". That is one pixel of a new image.",
        "Slide the filter everywhere and you get a feature map that lights up where the picture matches the pattern, here finger edges."]),
    ("ReLU and pooling", 24, "05 RELU AND POOLING", [
        "ReLU turns every negative number into zero. Then max pooling: in each two by two block, keep only the biggest number. The map shrinks from ninety-six to forty-eight, the strongest signals survive, and small shifts of the hand matter less."]),
    ("The whole network", 28, "06 THE WHOLE NETWORK", [
        "Put it together: convolution and pooling three times, with eight, sixteen, then thirty-two filters. Shrink every map to four by four and line them up: eight hundred ninety-six numbers.",
        "One linear layer turns them into seven scores, and softmax into probabilities. Here, about " + pct(HERO_PCT) + " percent sure: a two."]),
    ("Loss", 20, "07 LOSS", [
        "To learn, the network needs a score for how wrong it is. Here the right answer is " + words(s_b.CLS[s_b.U_TRUE]) + ", but it gave it only " + pct(s_b.U_PT) + " percent. Cross-entropy loss is minus the log of that: " + say(s_b.U_LOSS) + ". Training makes the average small."]),
    ("Backprop and gradient descent", 41, "08 BACKPROP + GRADIENT DESCENT", [
        "Backpropagation finds which weights caused the error. One feature, " + say(float(s_b.F_U[s_b.J])) + ", one weight, " + say(s_b.W_OLD, 3) + ". At the score, the slope is probability minus one: " + say(s_b.DZ) + ". Times the feature: " + say(s_b.DW) + ".",
        "Step against it: new weight equals weight minus learning rate times slope, here " + say(s_b.W_NEW, 3) + ". Every weight does this; I used Adam, a smarter version.",
        "The loss is a valley, and we walk downhill. In our real run, loss falls and accuracy climbs."]),
    ("Data and augmentation", 16, "09 DATA", [
        "About " + words(round(TOTAL, -1)) + " photos of three people, with random rotations, shifts, zooms and flips. " + pct(ACC_SAME).capitalize() + " percent on new frames from familiar sessions, but only " + pct(ACC_NEW) + " on a new person's hand."]),
    ("What it learned", 22, "10 WHAT IT LEARNED", [
        "Grad-CAM shows where the network looks. For three, four and five, the fingertips. For zero and one, mostly empty desk, where more fingers would be: no finger is evidence too. On a new hand it gets wrong, it often looks away from the fingers."]),
    ("The lock policy", 28, "11 THE LOCK POLICY", [
        "The network only guesses, frame by frame, so the lock policy decides. A digit counts only if the same number appears eight frames in a row at seventy-five percent confidence; a flicker never counts. Then the hand must leave.",
        "Three right digits open the latch; a wrong code is denied, and three wrong codes trigger a twenty-second lockdown."]),
    ("Limits and wrap-up", 14, "12 WRAP-UP", [
        "Limits: new hands, lighting, only one hundred twenty-five codes, a printed photo. On the UNO Q the same weights run in NumPy, about thirty milliseconds a photo. Thanks for watching."]),
]

WPS = 2.3  # comfortable speaking pace, words per second (about 140 wpm)


def mmss(s):
    return f"{int(s // 60)}:{int(s % 60):02d}"


lines = ["# Narration script: How the Finger-Lock Neural Network Works", "",
         "Read each block while its scene plays. The time next to the scene is when it starts in the video "
         "(`movie/out/finger_lock_explainer_silent.mp4`). The word budget is about 140 words a minute; if a scene's "
         "count is over budget, speak a little faster or cut a sentence.", "",
         "This is the 5-minute cut (4:56). The numbers in the narration match what is on screen (real values from the trained network). "
         "If you retrain, the on-screen values change and these lines need updating.", ""]
t = 0
for i, (title, dur, tag, paras) in enumerate(SCENES):
    words = sum(len(p.split()) for p in paras)
    budget = int(dur * WPS)
    flag = "" if words <= budget else "  **(over budget, trim or speak faster)**"
    lines += [f"## Scene {i}: {title}  ({mmss(t)} - {mmss(t + dur)}, {dur}s)", "",
              f"*{words} words, budget {budget}*{flag}", ""]
    for p in paras:
        lines += [p, ""]
    t += dur
lines += ["---", f"Total video length: {mmss(t)}.", "",
          "## After you record",
          "Save your recording as one file (wav, mp3 or m4a), then run:", "",
          "```", ".venv\\Scripts\\python movie\\add_voice.py my_voice.wav", "```", "",
          "Options: `--music 0.15` (background music volume under the voice), `--offset 1.0` (shift your voice later by 1 second). "
          "If your recording runs a little long or short, tell me and I can stretch or trim scene durations and re-render."]
open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "MOVIE_SCRIPT.md"), "w", encoding="utf-8").write("\n".join(lines))
for i, (title, dur, tag, paras) in enumerate(SCENES):
    w = sum(len(p.split()) for p in paras)
    print(f"{i:2d} {title:32s} {w:3d} words / budget {int(dur * WPS):3d}  {'OVER' if w > int(dur * WPS) else 'ok'}")

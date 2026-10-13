"""Draw the "how the game works" infographic (docs/how_it_works.png and .svg).

    python docs/make_infographic.py
"""

from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
plt.rcParams["font.family"] = "Segoe UI"

# colours: (fill, edge/text accent)
INPUT = ("#E8F1FB", "#2563A8")
PROCESS = ("#E6F6F3", "#16806F")
DECIDE = ("#FFF4DE", "#B7791F")
HIT = ("#E5F6E8", "#2F8A3E")
WHIFF = ("#FFEFE3", "#C2630F")
MISS = ("#FCE8E8", "#B83232")
OUTPUT = ("#F1ECFA", "#6B46B5")
INK, MUTED = "#1F2937", "#4B5563"

fig = plt.figure(figsize=(16, 11), dpi=100)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 69)
ax.axis("off")
fig.patch.set_facecolor("white")


def box(x, y, w, h, colors, title, body="", tag=None, title_size=13, body_size=10.5):
    fill, accent = colors
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.1",
                                fc=fill, ec=accent, lw=2))
    ax.text(x + 1.0, y + h - 1.0, title, ha="left", va="top", fontsize=title_size,
            fontweight="bold", color=accent)
    if body:
        ax.text(x + 1.0, y + h - 3.4, body, ha="left", va="top", fontsize=body_size, color=INK,
                linespacing=1.35)
    if tag:
        ax.text(x + w - 0.8, y + h - 1.0, tag, ha="right", va="top", fontsize=9, fontweight="bold",
                color="white", bbox=dict(boxstyle="round,pad=0.25", fc=accent, ec="none"))


def arrow(x0, y0, x1, y1, label=None, color=MUTED, lx=None, ly=None, ha="left"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=16,
                                 lw=1.8, color=color))
    if label:
        ax.text(lx, ly, label, fontsize=9.5, color=color, fontweight="bold", ha=ha, va="center")


def section(y, text):
    ax.text(1.2, y, text, fontsize=10, fontweight="bold", color="#9CA3AF", rotation=90,
            ha="center", va="center")


# ── Title, legend, screenshot ────────────────────────────────────────────────
ax.text(3, 66.3, "Ping Pong: how the game works", fontsize=24, fontweight="bold", color=INK, va="center")
ax.text(3, 63.4, "A LEGO motor is the paddle, a webcam watches your hand, and every swing is judged in "
        "real time, then felt, heard and posted live.", fontsize=12, color=MUTED, va="center")
for i, (chip, label) in enumerate((("RULE", "data-tuned rule"), ("ML", "machine learning"),
                                    ("CV", "classic computer vision"))):
    y = 67.0 - i * 1.7
    ax.text(71.5, y, chip, fontsize=8.5, fontweight="bold", color="white", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.25", fc=PROCESS[1], ec="none"))
    ax.text(73.6, y, label, fontsize=9.5, color=MUTED, ha="left", va="center")
shot = HERE / "table_view.png"
if shot.exists():
    inset = fig.add_axes([0.835, 0.885, 0.135, 0.105])
    inset.imshow(mpimg.imread(shot))
    inset.set_xticks([])
    inset.set_yticks([])
    for s in inset.spines.values():
        s.set_edgecolor("#CBD5E1")

# Columns: each input sits above the decision it feeds, so arrows run straight down.
L, M, R, W = 3, 36, 69, 28          # left / middle / right column x, column width
CL, CM, CR = L + W / 2, M + W / 2, R + W / 2

# ── 1. Sense ─────────────────────────────────────────────────────────────────
section(56.7, "SENSE")
box(L, 53.5, W, 6.6, INPUT, "Paddle: LEGO Double Motor", "IMU over Bluetooth, ~63 readings/s\n(gyroscope + accelerometer)")
box(M, 53.5, W, 6.6, INPUT, "Game clock", "Each ball leaves the wall, bounces on\nyour side and arrives on a schedule")
box(R, 53.5, W, 6.6, INPUT, "Webcam", "30 frames/s, read on a background\nthread so drawing never waits")

# ── 2. Understand ────────────────────────────────────────────────────────────
section(44, "UNDERSTAND")
box(L, 38.5, W, 11.2, PROCESS, "Swing detector",
    "Swing = rotation (gyro > 600)\n  AND whip (accel > 3 g) within 0.15 s.\n"
    "0.4 s cooldown ignores follow-through.\nThresholds set from my own recordings.", tag="RULE")
box(M, 38.5, W, 11.2, PROCESS, "Ball flight",
    "Random lane for every ball.\nEasy 1.3 s · Medium 1.0 s · Hard 0.75 s\n"
    "Hard: 3% faster each hit in a row\n(floor 0.60 s) and no visual helpers.")
box(R, 44.5, 13.3, 5.2, PROCESS, "AprilTag", "", tag="CV", title_size=11.5)
ax.text(R + 1.0, 46.9, "start screen: tag's\ncolumn = level", fontsize=9.5, color=INK, va="top", linespacing=1.25)
box(R + 14.7, 44.5, 13.3, 5.2, PROCESS, "Pose", "", tag="ML", title_size=11.5)
ax.text(R + 15.7, 46.9, "MediaPipe finds\nyour right wrist", fontsize=9.5, color=INK, va="top", linespacing=1.25)
box(R, 38.5, W, 4.4, PROCESS, "Wrist → lane", "", title_size=11.5)
ax.text(R + 1.0, 39.3, "left / centre / right third of the screen (±5% edge)", fontsize=9.3, color=INK)

arrow(CL, 53.5, CL, 50.1)
arrow(CM, 53.5, CM, 50.1)
arrow(R + 6.6, 53.5, R + 6.6, 50.1)
arrow(R + 21.3, 53.5, R + 21.3, 50.1)
arrow(R + 21.3, 44.5, R + 21.3, 43.3)                      # pose -> lane
arrow(R - 0.2, 47.1, M + W + 0.4, 47.1, color=PROCESS[1])  # tag -> level
ax.text(M + W + 0.7, 48.3, "level", fontsize=9.5, color=PROCESS[1], fontweight="bold")

# ── 3. Decide ────────────────────────────────────────────────────────────────
section(29, "DECIDE")
box(L, 24.5, W, 9, DECIDE, "1. Swing while the ball is\ncoming at you?", "\n\nNo → ignored (a practice swing)")
box(M, 24.5, W, 9, DECIDE, "2. On time?", "Within 0.35 s early … 0.20 s late\nof the ball's arrival")
box(R, 24.5, W, 9, DECIDE, "3. Wrist in the ball's lane?", "Checked at the moment\nof the swing")

arrow(CL, 38.5, CL, 33.9)
arrow(CM, 38.5, CM, 33.9, "arrival time", color=PROCESS[1], lx=CM + 0.8, ly=36.2)
arrow(CR, 38.5, CR, 33.9, "your lane", color=PROCESS[1], lx=CR + 0.8, ly=36.2)
arrow(L + W + 0.3, 29, M - 0.3, 29, "yes", color=HIT[1], lx=L + W + 0.9, ly=30.2)
arrow(M + W + 0.3, 29, R - 0.3, 29, "yes", color=HIT[1], lx=M + W + 0.9, ly=30.2)

# ── 4. Result ────────────────────────────────────────────────────────────────
section(16, "RESULT")
box(L, 12, W, 8.2, MISS, "MISS", "The ball got past you.\nStreak resets to 0, then a new serve.")
box(M, 12, W, 8.2, WHIFF, "WHIFF", "Too early or wrong lane: feedback\nonly. The ball keeps coming, so\nswing again before it passes.")
box(R, 12, W, 8.2, HIT, "HIT", "Streak + 1. The ball flies back,\nbounces off the wall and returns.")

arrow(M + 4, 24.5, L + W - 3, 20.6, "too late: ball passes", color=MISS[1], lx=L + W + 1.6, ly=23.4, ha="right")
arrow(CM + 3, 24.5, CM + 3, 20.6, "too early", color=WHIFF[1], lx=CM + 3.8, ly=22.6)
arrow(R + 4, 24.5, M + W - 3, 20.6, "no", color=WHIFF[1], lx=R + 3.0, ly=22.6, ha="right")
arrow(CR + 6, 24.5, CR + 6, 20.6, "yes", color=HIT[1], lx=CR + 6.8, ly=22.6)

# ── 5. Respond: every result feeds the same four outputs ─────────────────────
section(5, "RESPOND")
bus_y = 10.2
for cx in (CL, CM, CR):
    ax.plot([cx, cx], [12, bus_y], color=OUTPUT[1], lw=1.8)
ax.plot([3 + 10.25, 76.5 + 10.25], [bus_y, bus_y], color=OUTPUT[1], lw=1.8)   # spans all four outputs
ax.text(CM, bus_y + 0.55, "every result is felt, heard, seen and posted", fontsize=9.5, color=OUTPUT[1],
        fontweight="bold", ha="center", va="bottom",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"))
outputs = [
    (3, "Paddle (haptics)", "Hit: sharp buzz + green light\nMiss: long buzz + red light"),
    (27.5, "Audio", "My recorded 'pock' + effects;\nannouncer calls each miss"),
    (52, "3D table view", "Ball, shadow, paddle swing and\nhit burst, aiming for 60 fps"),
    (76.5, "Live score (MQTT)", "Streak as a float, e.g. '3.0'\n→ ME193/Rogers/Jude"),
]
for x, title, body in outputs:
    box(x, 0.8, 20.5, 7.4, OUTPUT, title, body, title_size=11.5, body_size=9.8)
    arrow(x + 10.25, bus_y, x + 10.25, 8.6, color=OUTPUT[1])

for ext in ("png", "svg"):
    fig.savefig(HERE / f"how_it_works.{ext}", dpi=200 if ext == "png" else None, facecolor="white")
print("saved docs/how_it_works.png and .svg")

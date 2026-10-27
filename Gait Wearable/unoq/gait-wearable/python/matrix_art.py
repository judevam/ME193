"""Animations for the UNO Q's 8 x 13 LED matrix, one per activity.

Frames are drawn as text: '#' = bright, '+' = dim, '.' = off. Edit them freely; every
row must be 13 characters and every frame 8 rows.

Preview on the laptop (needs matplotlib):  python matrix_art.py
"""
ROWS, COLS = 8, 13
BRIGHT, DIM = 7, 2                 # brightness, 0..7 (3-bit grayscale)
LEVELS = {"#": BRIGHT, "+": DIM, ".": 0}


def _swap(frame):
    """Swap bright and dim: shows the other leg in front."""
    return [r.replace("#", "x").replace("+", "#").replace("x", "+") for r in frame]


# Standing: front view, two legs and feet, not moving.
STANDING = [[
    "....#####....",
    "....#...#....",
    "....#...#....",
    "....#...#....",
    "....#...#....",
    "....#...#....",
    ".####...####.",
    ".####...####.",
]]

# Walking: side view facing right. The leg in front is bright, the one behind dim.
_WALK_STRIDE = [
    "......#......",
    ".....+.#.....",
    ".....+.#.....",
    "....+...#....",
    "....+...#....",
    "...+.....#...",
    "...+.....#...",
    "...++....##..",
]
_WALK_PASS = [               # standing leg straight, the other passing with its foot lifted
    "......#......",
    "......#+.....",
    "......#.+....",
    "......#.+....",
    "......#.+....",
    "......#.++...",
    "......#......",
    "......##.....",
]
WALKING = [_WALK_STRIDE, _WALK_PASS, _swap(_WALK_STRIDE), _swap(_WALK_PASS)]

# Running: leaning forward, long stride, knee driving high.
_RUN_STRIDE = [
    ".......#.....",
    "......+.##...",
    "......+...##.",
    ".....++....#.",
    "...++.....#..",
    "..+.......#..",
    "+++.......###",
    "+............",
]
_RUN_KNEE = [
    ".......#.....",
    ".......#++++.",
    "......#...++.",
    "......#....+.",
    ".....#.....++",
    ".....#.......",
    "....#........",
    "....##.......",
]
RUNNING = [_RUN_STRIDE, _RUN_KNEE, _swap(_RUN_STRIDE), _swap(_RUN_KNEE)]


def _bike_frames():
    """A bike whose wheels' spokes and pedal crank turn (4 frames, clockwise)."""
    base = [
        "..##.....##..",   # saddle, handlebar
        "...#.....#...",
        "...#######...",   # top tube
        ".+++.....+++.",   # wheel rims (dim)
        "+...+...+...+",
        "+...+...+...+",
        "+...+...+...+",
        ".+++.....+++.",
    ]
    ring = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]  # clockwise
    frames = []
    for k in range(4):
        grid = [list(r) for r in base]
        for hub_r, hub_c in ((5, 2), (5, 10)):            # two wheels
            grid[hub_r][hub_c] = "#"
            for dr, dc in (ring[k], ring[k + 4]):        # one spoke through the hub
                grid[hub_r + dr][hub_c + dc] = "#"
        cr, cc = 5, 6                                     # pedal crank
        grid[cr][cc] = "+"
        dr, dc = ring[(2 * k) % 8]
        grid[cr + dr][cc + dc] = "#"
        frames.append(["".join(r) for r in grid])
    return frames


BIKING = _bike_frames()

# Nothing to show: one dot slowly fading in and out, so you can tell it's alive.
IDLE = [[("......" + c + "......") if r == 4 else "." * COLS for r in range(ROWS)]
        for c in "..++##++"]
IDLE = [[row.replace("#", "+") if i in (2, 3, 6, 7) else row for row in f] for i, f in enumerate(IDLE)]

# (frames, seconds per frame)
ANIMATIONS = {
    "standing": (STANDING, 1.0),
    "walking": (WALKING, 0.35),     # slow steps
    "running": (RUNNING, 0.11),     # fast legs
    "biking": (BIKING, 0.12),
    None: (IDLE, 0.25),
}


def animation(gait):
    return ANIMATIONS.get(gait, ANIMATIONS[None])


def encode(frame, rec_dot=False):
    """Frame -> 104 digits '0'..'7', row by row, for the sketch's draw()."""
    out = "".join(str(LEVELS[ch]) for row in frame for ch in row)
    if rec_dot:  # top-left corner blinks while recording
        out = str(BRIGHT) + out[1:]
    return out


def check():
    for name, (frames, _) in ANIMATIONS.items():
        for i, f in enumerate(frames):
            assert len(f) == ROWS, (name, i, len(f))
            for row in f:
                assert len(row) == COLS and set(row) <= set(LEVELS), (name, i, row)


check()


if __name__ == "__main__":
    import sys

    import matplotlib.pyplot as plt
    import numpy as np

    names = ["standing", "walking", "running", "biking", None]
    width = max(len(ANIMATIONS[n][0]) for n in names)
    fig, axes = plt.subplots(len(names), width, figsize=(width * 1.6, len(names) * 1.2),
                             facecolor="black")
    for r, name in enumerate(names):
        frames, period = ANIMATIONS[name]
        for c in range(width):
            ax = axes[r][c]
            ax.axis("off")
            if c < len(frames):
                img = np.array([[LEVELS[ch] for ch in row] for row in frames[c]])
                ax.imshow(img, cmap="Greens_r" if False else "inferno", vmin=0, vmax=7)
                ax.set_title(f"{name or 'idle'} {c + 1}" if c else f"{name or 'idle'} ({period}s)",
                             color="w", fontsize=8)
    plt.tight_layout()
    out = sys.argv[1] if len(sys.argv) > 1 else "matrix_preview.png"
    plt.savefig(out, dpi=90, facecolor="black")
    print("saved", out)

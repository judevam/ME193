"""Draw what the CNN sees: the input and every feature map of the three conv layers.

Pure numpy + cv2 (runs on the UNO Q and the laptop). A copy lives in
ArduinoApps/finger-lock-cam-test/python/ (keep the two identical).
"""
import cv2
import numpy as np

TILE = 56
COLS = 8
W = TILE * COLS
LABEL_H = 18

# dark -> neon green "night vision" colour map (BGR)
_g = np.arange(256, dtype=np.float32)
_LUT = np.stack([_g * 0.30, _g, _g * 0.55], axis=1).astype(np.uint8).reshape(256, 1, 3)


def _tile(m, lo=None, hi=None):
    lo = float(m.min()) if lo is None else lo
    hi = float(m.max()) if hi is None else hi
    img = ((m - lo) / (hi - lo + 1e-6) * 255).astype(np.uint8)
    img = cv2.resize(img, (TILE - 2, TILE - 2), interpolation=cv2.INTER_CUBIC)
    img = cv2.applyColorMap(img, _LUT)
    return cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(20, 28, 24))


def _label(text):
    bar = np.zeros((LABEL_H, W, 3), np.uint8)
    cv2.putText(bar, text, (6, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 200, 150), 1, cv2.LINE_AA)
    return bar


def _grid(maps):
    """All maps of a layer share one scale, so weak filters look dark, strong ones bright."""
    lo, hi = min(float(m.min()) for m in maps), max(float(m.max()) for m in maps)
    rows = []
    for r in range(0, len(maps), COLS):
        row = [_tile(m, lo, hi) for m in maps[r:r + COLS]]
        row += [np.zeros((TILE, TILE, 3), np.uint8)] * (COLS - len(row))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def render_jpeg(gray, a1, a2, a3, quality=75):
    """gray (96,96) in [0,1]; a1 (8,48,48); a2 (16,24,24); a3 (32,12,12) -> JPEG bytes."""
    inp = _tile(gray)
    first = np.hstack([inp, np.zeros((TILE, W - TILE, 3), np.uint8)])
    cv2.putText(first, "what the network is given (grayscale)", (TILE + 10, TILE // 2 + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 200, 150), 1, cv2.LINE_AA)
    img = np.vstack([
        _label("INPUT 96x96"), first,
        _label("LAYER 1: 8 filters, 5x5  (edges and blobs)"), _grid(list(a1)),
        _label("LAYER 2: 16 filters, 3x3  (finger shapes)"), _grid(list(a2)),
        _label("LAYER 3: 32 filters, 3x3  (combinations)"), _grid(list(a3)),
    ])
    ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return jpg.tobytes() if ok else b""

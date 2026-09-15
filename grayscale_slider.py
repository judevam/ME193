"""Show an image in a window with a slider that fades it from color to grayscale.

Usage:
    python grayscale_slider.py <image_path>

Drag the "Grayscale %" trackbar. Press 's' to save the currently displayed
blend next to the source image (suffixed "_blend"), or 'q'/Esc to quit.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

WINDOW_NAME = "Color <-> Grayscale (drag slider, s to save, q to quit)"
TRACKBAR_NAME = "Grayscale %"


def blend_with_grayscale(image: np.ndarray, percent: int) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    alpha = percent / 100.0
    return cv2.addWeighted(gray_bgr, alpha, image, 1 - alpha, 0)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    input_path = Path(sys.argv[1])
    image = cv2.imread(str(input_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {input_path}")

    cv2.namedWindow(WINDOW_NAME)
    cv2.createTrackbar(TRACKBAR_NAME, WINDOW_NAME, 0, 100, lambda _: None)

    while True:
        percent = cv2.getTrackbarPos(TRACKBAR_NAME, WINDOW_NAME)
        blended = blend_with_grayscale(image, percent)
        cv2.imshow(WINDOW_NAME, blended)

        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):  # q or Esc
            break
        if key == ord("s"):
            out_path = input_path.with_name(f"{input_path.stem}_blend{input_path.suffix}")
            cv2.imwrite(str(out_path), blended)
            print(f"Saved {out_path}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

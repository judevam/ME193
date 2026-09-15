"""Interactively explore grayscale thresholding, dilation, and erosion on an image.

Usage:
    python threshold_morphology_slider.py <image_path>

Trackbars control:
- Threshold: binary threshold cutoff (0-255) applied to the grayscale image
- Morph Kernel Size: structuring element size (pixels) used for dilation/erosion
- Dilate Iter / Erode Iter: how many times dilation/erosion is applied to the
  thresholded image
- Conv Kernel: which convolution kernel to apply to the grayscale image (see
  KERNELS below) - slide through box blur, Gaussian blur, sharpen, Laplacian
  edge detection, Sobel X/Y, and emboss

A convolution kernel is a small matrix slid over the image: at each position
you multiply the kernel values by the pixel values underneath, sum the
products, and that sum becomes the new center pixel. This differs from the
morphological structuring element used for dilation/erosion, which combines
neighboring pixels with a logical AND/OR (any/all) instead of a weighted sum.

Shows a 2x2 grid (grayscale and threshold on top, dilation and erosion on
bottom), a wide outline row (dilation minus the threshold image, which
leaves just the boundary the dilation grew outward), and a wide convolution
row with the selected kernel's matrix overlaid on it. Press 's' to save the
current grid, 'q'/Esc to quit.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

WINDOW_NAME = "Threshold / Dilation / Erosion (s to save, q to quit)"

# The grid is 2 panels wide x 4 rows tall (top, bottom, outline, convolution),
# each row topped with a 28px label bar. Panel width is chosen per-image (see
# compute_panel_width) so the whole window fits within these bounds instead
# of running off the bottom of the screen for tall/wide source images.
MAX_GRID_WIDTH = 1000
MAX_GRID_HEIGHT = 820
LABEL_HEIGHT = 28
GRID_ROWS = 4


def compute_panel_width(img_h: int, img_w: int) -> int:
    aspect = img_h / img_w
    width_limit_from_grid = MAX_GRID_WIDTH // 2
    height_budget_per_row = MAX_GRID_HEIGHT // GRID_ROWS - LABEL_HEIGHT
    width_limit_from_height = int(height_budget_per_row / aspect)
    return max(60, min(width_limit_from_grid, width_limit_from_height, img_w))

# name -> (kernel matrix, divisor). The divisor normalizes the sum (e.g. the
# classic 3x3 box blur is this matrix of 1s times 1/9) while keeping the
# matrix itself in the same whole-number form you'd see written out by hand.
KERNELS = {
    "Identity": (np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32), 1),
    "Box Blur": (np.ones((3, 3), dtype=np.float32), 9),
    "Gaussian Blur": (np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32), 16),
    "Sharpen": (np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32), 1),
    "Edge (Laplacian)": (np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32), 1),
    "Sobel X": (np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32), 1),
    "Sobel Y": (np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32), 1),
    "Emboss": (np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float32), 1),
}
KERNEL_NAMES = list(KERNELS.keys())


def label(panel: np.ndarray, text: str) -> np.ndarray:
    panel = panel.copy()
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(panel, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def overlay_kernel_matrix(panel: np.ndarray, kernel: np.ndarray, divisor: float) -> np.ndarray:
    panel = panel.copy()
    lines = ["  ".join(f"{v:>3g}" for v in row) for row in kernel]
    if divisor != 1:
        lines.append(f"(sum x 1/{divisor:g})")
    box_h = 18 * len(lines) + 10
    cv2.rectangle(panel, (0, panel.shape[0] - box_h), (190, panel.shape[0]), (0, 0, 0), -1)
    for i, line in enumerate(lines):
        y = panel.shape[0] - box_h + 16 + i * 18
        cv2.putText(panel, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def apply_kernel(gray: np.ndarray, name: str) -> np.ndarray:
    kernel, divisor = KERNELS[name]
    filtered = cv2.filter2D(gray, cv2.CV_64F, kernel / divisor)
    return cv2.convertScaleAbs(filtered)


def to_bgr(gray: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def build_grid(
    gray: np.ndarray,
    thresh_val: int,
    kernel_size: int,
    dilate_iter: int,
    erode_iter: int,
    conv_kernel_name: str,
    panel_width: int,
) -> np.ndarray:
    h, w = gray.shape[:2]
    if w != panel_width:
        scale = panel_width / w
        gray = cv2.resize(gray, (panel_width, max(1, int(h * scale))))

    _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
    kernel = np.ones((max(kernel_size, 1), max(kernel_size, 1)), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=dilate_iter)
    eroded = cv2.erode(binary, kernel, iterations=erode_iter)

    top = np.hstack([
        label(to_bgr(gray), "Grayscale"),
        label(to_bgr(binary), f"Threshold: {thresh_val}"),
    ])
    bottom = np.hstack([
        label(to_bgr(dilated), f"Dilation x{dilate_iter}"),
        label(to_bgr(eroded), f"Erosion x{erode_iter}"),
    ])

    outline = cv2.subtract(dilated, binary)
    outline_row = cv2.resize(to_bgr(outline), (top.shape[1], gray.shape[0]))
    outline_row = label(outline_row, "Outline (Dilation - Threshold)")

    conv_kernel, conv_divisor = KERNELS[conv_kernel_name]
    convolved = apply_kernel(gray, conv_kernel_name)
    conv_row = cv2.resize(to_bgr(convolved), (top.shape[1], gray.shape[0]))
    conv_row = label(conv_row, f"Convolution: {conv_kernel_name}")
    conv_row = overlay_kernel_matrix(conv_row, conv_kernel, conv_divisor)

    return np.vstack([top, bottom, outline_row, conv_row])


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    input_path = Path(sys.argv[1])
    image = cv2.imread(str(input_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {input_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    panel_width = compute_panel_width(*gray.shape[:2])

    cv2.namedWindow(WINDOW_NAME)
    cv2.createTrackbar("Threshold", WINDOW_NAME, 127, 255, lambda _: None)
    cv2.createTrackbar("Morph Kernel Size", WINDOW_NAME, 3, 31, lambda _: None)
    cv2.createTrackbar("Dilate Iter", WINDOW_NAME, 1, 10, lambda _: None)
    cv2.createTrackbar("Erode Iter", WINDOW_NAME, 1, 10, lambda _: None)
    cv2.createTrackbar("Conv Kernel", WINDOW_NAME, 0, len(KERNEL_NAMES) - 1, lambda _: None)

    grid = None
    while True:
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break  # window was closed via its titlebar, not 'q'/Esc

        thresh_val = cv2.getTrackbarPos("Threshold", WINDOW_NAME)
        kernel_size = cv2.getTrackbarPos("Morph Kernel Size", WINDOW_NAME)
        dilate_iter = cv2.getTrackbarPos("Dilate Iter", WINDOW_NAME)
        erode_iter = cv2.getTrackbarPos("Erode Iter", WINDOW_NAME)
        conv_kernel_name = KERNEL_NAMES[cv2.getTrackbarPos("Conv Kernel", WINDOW_NAME)]

        grid = build_grid(gray, thresh_val, kernel_size, dilate_iter, erode_iter, conv_kernel_name, panel_width)
        cv2.imshow(WINDOW_NAME, grid)

        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):  # q or Esc
            break
        if key == ord("s"):
            out_path = input_path.with_name(f"{input_path.stem}_threshold_morph{input_path.suffix}")
            cv2.imwrite(str(out_path), grid)
            print(f"Saved {out_path}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

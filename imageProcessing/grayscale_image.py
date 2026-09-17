"""Convert a local image file to grayscale using OpenCV.

Usage:
    python grayscale_image.py <input_path> [output_path]

If output_path is omitted, the grayscale image is saved next to the input
file, named after it with a "_grayscale" suffix.
"""

import sys
from pathlib import Path

import cv2


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_grayscale{input_path.suffix}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else default_output_path(input_path)

    image = cv2.imread(str(input_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {input_path}")

    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(str(output_path), grayscale)

    print(f"Saved grayscale image to {output_path}")


if __name__ == "__main__":
    main()

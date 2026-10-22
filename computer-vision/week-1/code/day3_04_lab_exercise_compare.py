"""
DAY 3 - LAB EXERCISE: apply blur, sharpen and edge detection - compare side by side
====================================================================================
This is the lecture's lab code, put into one side-by-side figure.

Run:  python day3_04_lab_exercise_compare.py
"""

import cv2
import numpy as np
from cv_helpers import load_image, show

img = load_image("sample.jpg")
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])   # sharpening kernel

filters = {
    "original": img,
    "blur": cv2.blur(img, (5, 5)),
    "sharpen": cv2.filter2D(img, -1, kernel),
    "edges": cv2.Canny(img, 50, 150),
}

# show() handles the colour vs grey difference for us:
# 3-channel results are converted BGR->RGB, 1-channel ones are drawn in grey.
# (The lecture did this by hand with: `if name == "edges" ... cmap='gray'`)
show(list(filters.values()), [name.capitalize() for name in filters],
     filename="day3_04_lab_compare.png", cols=4)

for name, out in filters.items():
    print(f"{name:9s} -> shape {out.shape}, dtype {out.dtype}, min {out.min()}, max {out.max()}")
# Notice "edges" has NO channel dimension: Canny always returns a single-channel image.

# ---------------------------------------------------------------
# TRY THIS:
#   a) Add "gaussian": cv2.GaussianBlur(img, (5, 5), 0) to the dictionary.
#   b) Add "sobel" (remember convertScaleAbs and grayscale!).
#   c) Save all results:  cv2.imwrite(f"{name}.png", out)
# ---------------------------------------------------------------

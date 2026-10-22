"""
DAY 3 - Script 3: Canny edge detection, step by step
=====================================================
Lecture topic: "Canny Edge Detection - combines gradient + thresholding,
produces clean edge maps."

The Canny algorithm has 4 steps:
  1. Noise reduction     - Gaussian blur (5x5)
                           (OpenCV's cv2.Canny skips this one - blur yourself first for best results)
  2. Gradient            - Sobel in x and y -> edge strength + direction
  3. Non-max suppression - thin thick edges down to 1 pixel wide
  4. Hysteresis (double threshold):
        strength > threshold2 (high)       -> definitely an edge (strong)
        strength < threshold1 (low)        -> definitely NOT an edge
        in between                         -> kept ONLY if touching a strong edge
Output: a black image with thin WHITE lines (0 or 255 only).

Run:  python day3_03_canny.py
"""

import cv2
import numpy as np
from cv_helpers import load_image, show

img = load_image("sample.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ---------- Rebuild the main steps ourselves to SEE them ----------
step1 = cv2.GaussianBlur(gray, (5, 5), 1.4)
gx = cv2.Sobel(step1, cv2.CV_64F, 1, 0, ksize=3)
gy = cv2.Sobel(step1, cv2.CV_64F, 0, 1, ksize=3)
step2 = cv2.convertScaleAbs(np.hypot(gx, gy))          # gradient strength: thick, fuzzy edges
step4 = cv2.Canny(gray, 100, 200)                      # the full algorithm: thin, clean edges

show([gray, step1, step2, step4],
     ["Input (grayscale)", "1. Blur away noise", "2. Gradient (Sobel) - thick edges",
      "3+4. Thin + threshold = Canny"],
     filename="day3_03a_canny_steps.png", cols=4)

# ---------- Effect of the two thresholds ----------
settings = [(10, 30), (50, 150), (100, 200), (200, 400)]
show([cv2.Canny(gray, lo, hi) for lo, hi in settings],
     [f"Canny({lo}, {hi})" for lo, hi in settings],
     filename="day3_03b_canny_thresholds.png", cols=4)
# Low thresholds  -> lots of edges, including noise/texture
# High thresholds -> only the strongest outlines
# A common rule of thumb: high = 2x or 3x low.

# ---------- Blur first for even cleaner edges ----------
show([cv2.Canny(gray, 50, 150), cv2.Canny(cv2.GaussianBlur(gray, (7, 7), 0), 50, 150)],
     ["Canny on raw image", "Canny after extra 7x7 blur (less noise)"],
     filename="day3_03c_canny_blur_first.png")

print("Canny output values:", np.unique(step4))   # only 0 and 255

# ---------------------------------------------------------------
# WHY DID THE LECTURE'S EDGE IMAGES LOOK ALMOST ALL BLACK?
#   1. The photo was huge (5472 x 3648). Edges are 1 pixel thin, and when
#      such a big image is shrunk to fit on screen most thin lines disappear.
#   2. The photo was soft/blurry (shallow focus), so few strong edges.
#   Fix: resize the image smaller first (load_image does this), or lower the thresholds.
#
# TRY THIS:
#   a) Find thresholds that outline the shapes but ignore the text.
#   b) Canny(gray, 200, 100) - swap them. (OpenCV quietly swaps them back!)
# ---------------------------------------------------------------

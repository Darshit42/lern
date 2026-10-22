"""
DAY 1 - Script 2: Build tiny images BY HAND (no photo needed)
==============================================================
Lecture topic: "Images as Data" - grids of pixels, intensity values, resolution.

If you can build an image from numbers, you truly understand what an image is.
Run:  python day1_02_pixels_by_hand.py
"""

import numpy as np
import cv2
from cv_helpers import show

# ---------------------------------------------------------------
# 1. A 5x5 GRAYSCALE image: one number per pixel.
#    0 = black, 255 = white, anything between = grey.
# ---------------------------------------------------------------
tiny_gray = np.array([
    [  0,   0,   0,   0,   0],
    [  0, 255, 255, 255,   0],
    [  0, 255, 128, 255,   0],
    [  0, 255, 255, 255,   0],
    [  0,   0,   0,   0,   0],
], dtype=np.uint8)
print("Grayscale image shape:", tiny_gray.shape, "(height, width) - no channel number!")

# ---------------------------------------------------------------
# 2. A 2x3 COLOUR image: three numbers per pixel (B, G, R).
# ---------------------------------------------------------------
tiny_color = np.array([
    [[255, 0, 0], [0, 255, 0],     [0, 0, 255]],     # blue, green, red
    [[0, 255, 255], [255, 255, 255], [0, 0, 0]],     # yellow, white, black
], dtype=np.uint8)
print("Colour image shape:", tiny_color.shape, "(height, width, channels)")

# ---------------------------------------------------------------
# 3. A gradient - every column is a bit brighter than the last.
# ---------------------------------------------------------------
gradient = np.tile(np.arange(0, 256, dtype=np.uint8), (60, 1))   # 60 rows x 256 columns

# ---------------------------------------------------------------
# 4. RESOLUTION = how many pixels. Fewer pixels -> less detail.
#    We shrink a smooth circle to 16x16 and blow it back up so you
#    can SEE the blocky pixels.
# ---------------------------------------------------------------
circle = np.zeros((256, 256), np.uint8)
cv2.circle(circle, (128, 128), 90, 255, -1)
low_res = cv2.resize(circle, (16, 16), interpolation=cv2.INTER_AREA)
low_res_big = cv2.resize(low_res, (256, 256), interpolation=cv2.INTER_NEAREST)

show([tiny_gray, tiny_color, gradient, circle, low_res_big],
     ["5x5 grayscale", "2x3 colour (BGR)", "Gradient 0 -> 255",
      "256x256 = high resolution", "16x16 = low resolution"],
     filename="day1_02_pixels_by_hand.png", cols=3)

# ---------------------------------------------------------------
# 5. TRAP: uint8 numbers wrap around!  250 + 10 = 4 (not 260)
#    cv2.add() "saturates" instead: 250 + 10 = 255 (stays at max).
# ---------------------------------------------------------------
a = np.array([[250]], dtype=np.uint8)   # a 1x1 "image"
b = np.array([[10]], dtype=np.uint8)
print("\nnumpy   250 + 10 =", (a + b)[0, 0], "  <- wraps around (overflow)")
print("cv2.add 250 + 10 =", cv2.add(a, b)[0, 0], "<- clipped at 255 (saturation)")

# ---------------------------------------------------------------
# TRY THIS:
#   a) Change the 128 in tiny_gray to 0 or 255. What changes?
#   b) Make the 2x3 colour image show orange: BGR = (0, 165, 255)
#   c) Shrink the circle to (8, 8) instead of (16, 16). How blocky is it?
# ---------------------------------------------------------------

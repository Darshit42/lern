"""
DAY 5 - Script 1: Structuring elements, Dilation, Erosion
==========================================================
Lecture topics: Structuring Elements, Dilation, Erosion

Morphology = changing the SHAPE of white regions in a (usually binary) image.
Think of a marker and an eraser:
    Dilation = go over every white shape with a thick marker -> shapes get FATTER
    Erosion  = shave the edges of every white shape          -> shapes get THINNER

The "structuring element" (also called kernel) is the shape of the marker tip.

IMPORTANT: morphology works on WHITE (255) as the foreground. If your objects are
black on white paper, invert first (cv2.THRESH_BINARY_INV or cv2.bitwise_not).

Run:  python day5_01_morphology_basics.py
"""

import cv2
import numpy as np
from cv_helpers import load_image, show

# =================================================================
# 1. STRUCTURING ELEMENTS - small binary matrices that "probe" the image
# =================================================================
kernel = np.ones((5, 5), np.uint8)            # lecture: 5x5 square of 1s
print("Lecture kernel (np.ones):\n", kernel)

rect = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))        # square
ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # roughly round
cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))      # plus sign
print("\nMORPH_RECT:\n", rect, "\n\nMORPH_ELLIPSE:\n", ellipse, "\n\nMORPH_CROSS:\n", cross)

# =================================================================
# 2. SEE IT ON A TINY GRID (numbers you can check by hand)
#    Dilation: output pixel = MAX under the kernel -> white if ANY neighbour is white
#    Erosion:  output pixel = MIN under the kernel -> white only if ALL neighbours are white
# =================================================================
tiny = np.zeros((7, 7), np.uint8)
tiny[2:5, 2:5] = 255                         # a 3x3 white square in the middle
k3 = np.ones((3, 3), np.uint8)
print("\nInput (1 = white):\n", tiny // 255)
print("\nDilated with 3x3 -> square grows by 1 pixel on every side:\n", cv2.dilate(tiny, k3) // 255)
print("\nEroded with 3x3 -> only the centre pixel survives:\n", cv2.erode(tiny, k3) // 255)

# =================================================================
# 3. ON A BINARY IMAGE (the clearest way to see the effect)
# =================================================================
canvas = np.zeros((220, 520), np.uint8)
cv2.putText(canvas, "CV", (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 6, 255, 18)
cv2.line(canvas, (300, 40), (500, 40), 255, 1)                   # a thin line
cv2.rectangle(canvas, (300, 80), (500, 190), 255, -1)            # a solid block
cv2.line(canvas, (400, 80), (400, 190), 0, 3)                    # a crack in the block
rng = np.random.default_rng(0)
canvas[rng.integers(0, 220, 150), rng.integers(0, 520, 150)] = 255   # white noise dots

dilated = cv2.dilate(canvas, kernel, iterations=1)    # lecture code
eroded = cv2.erode(canvas, kernel, iterations=1)      # lecture code
dilated3 = cv2.dilate(canvas, kernel, iterations=3)   # iterations = repeat it N times

show([canvas, dilated, eroded, dilated3],
     ["Input: text, thin line, cracked block, noise dots",
      "Dilate 5x5: fatter, crack filled, dots grow",
      "Erode 5x5: thinner, thin line + dots GONE",
      "Dilate x3 iterations: much fatter"],
     filename="day5_01a_dilate_erode_binary.png", cols=2, figsize_per=5)

# Kernel SHAPE matters: compare corners after dilation
shapes_demo = np.zeros((120, 120), np.uint8)
shapes_demo[50:70, 50:70] = 255
big = 25
show([cv2.dilate(shapes_demo, cv2.getStructuringElement(s, (big, big)))
      for s in (cv2.MORPH_RECT, cv2.MORPH_ELLIPSE, cv2.MORPH_CROSS)],
     ["Dilate with RECT -> square corners", "Dilate with ELLIPSE -> rounded", "Dilate with CROSS -> plus-shaped"],
     filename="day5_01b_kernel_shapes.png")

# =================================================================
# 4. ON A GRAYSCALE PHOTO (what the lecture did)
#    Dilation = local MAX -> bright areas spread, image looks brighter/blockier
#    Erosion  = local MIN -> dark areas spread, image looks darker
#    That's why the lecture's photos barely changed - the effect is clearest on
#    BINARY images with small details.
# =================================================================
gray = cv2.cvtColor(load_image("sample.jpg"), cv2.COLOR_BGR2GRAY)
show([gray, cv2.dilate(gray, kernel), cv2.erode(gray, kernel)],
     ["Grayscale photo", "Dilated (local max): brighter", "Eroded (local min): darker"],
     filename="day5_01c_grayscale_morphology.png")

# ---------------------------------------------------------------
# TRY THIS:
#   a) Change the kernel to (3,3) and (11,11). Bigger kernel = stronger effect.
#   b) Erode with iterations=2 - what happens to the letters "CV"?
#   c) Predict the 7x7 result of dilating `tiny` with a CROSS kernel, then check:
#        print(cv2.dilate(tiny, cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))) // 255)
# ---------------------------------------------------------------

"""
DAY 5 - Script 2: Opening & Closing (+ gradient, top-hat, black-hat)
======================================================================
Lecture topics: Opening, Closing, Lab Exercise (compare all side by side)

    OPENING = Erosion THEN Dilation  -> removes small white SPECKS, keeps big shapes
    CLOSING = Dilation THEN Erosion  -> fills small black HOLES/GAPS inside shapes

Memory trick:
    OPENING "opens up" gaps by deleting tiny bits of white (erase specks).
    CLOSING "closes" holes and cracks (fill gaps).
    Both keep the big shapes at roughly their original size - that's the point of
    doing the second step: it undoes the size change of the first.

Run:  python day5_02_opening_closing.py
"""

import cv2
import numpy as np
from cv_helpers import load_image, show

kernel = np.ones((5, 5), np.uint8)

# A binary test image with BOTH kinds of problem
img = np.zeros((260, 600), np.uint8)
cv2.putText(img, "Morphology", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 3, 255, 16)
rng = np.random.default_rng(1)
noisy = img.copy()
noisy[rng.integers(0, 260, 500), rng.integers(0, 600, 500)] = 255      # white specks OUTSIDE the text
holes = rng.random(img.shape) < 0.02
noisy[holes & (img == 255)] = 0                                         # black pinholes INSIDE the text

# ----- the lecture's code -----
dilated = cv2.dilate(noisy, kernel, iterations=1)
eroded = cv2.erode(noisy, kernel, iterations=1)
opened = cv2.morphologyEx(noisy, cv2.MORPH_OPEN, kernel)
closed = cv2.morphologyEx(noisy, cv2.MORPH_CLOSE, kernel)

# Prove opening = erode then dilate, closing = dilate then erode:
print("open  == dilate(erode(x)) ?", np.array_equal(opened, cv2.dilate(cv2.erode(noisy, kernel), kernel)))
print("close == erode(dilate(x)) ?", np.array_equal(closed, cv2.erode(cv2.dilate(noisy, kernel), kernel)))

# Fix BOTH problems. ORDER MATTERS here: if we OPEN first, the erosion step widens the
# pinholes inside the letters and chews them up. CLOSE first (seal the holes), THEN
# OPEN (delete the specks) keeps the letters solid.
both = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

show([noisy, dilated, eroded, opened, closed, both],
     ["Noisy: specks outside + holes inside", "Dilated", "Eroded",
      "OPENING: specks gone (holes grow)", "CLOSING: holes filled (specks stay)", "Close then Open: both fixed"],
     filename="day5_02a_opening_closing.png", cols=3)

# ----- bonus operations (sometimes asked) -----
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)   # dilate - erode = OUTLINE of shapes
tophat = cv2.morphologyEx(noisy, cv2.MORPH_TOPHAT, kernel)     # input - opening = the small bright bits removed
blackhat = cv2.morphologyEx(noisy, cv2.MORPH_BLACKHAT, kernel) # closing - input = the small dark holes filled
show([gradient, tophat, blackhat],
     ["GRADIENT = dilate - erode (outline)", "TOP-HAT = input - opening (the specks)",
      "BLACK-HAT = closing - input (the holes)"],
     filename="day5_02b_gradient_tophat_blackhat.png")

# ----- lecture lab exercise: same 5 operations on the grayscale photo -----
gray = cv2.cvtColor(load_image("sample.jpg"), cv2.COLOR_BGR2GRAY)
titles = ["Original", "Dilated", "Eroded", "Opening", "Closing"]
images = [gray, cv2.dilate(gray, kernel), cv2.erode(gray, kernel),
          cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel), cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)]
show(images, titles, filename="day5_02c_lab_compare_photo.png", cols=3)

# ---------------------------------------------------------------
# TRY THIS:
#   a) Use a 3x3 kernel for opening - do some specks survive?
#   b) Use a 9x9 kernel for opening - do the letters start to break?
#   c) Swap the order: open first, then close. Why is it worse for this image?
# ---------------------------------------------------------------

"""
DAY 4 - Script 2: Thresholding - Global, Adaptive, Otsu
========================================================
Lecture topics: Global thresholding, Adaptive thresholding, Otsu's method, Lab exercise.

Thresholding turns a grayscale image into a BINARY image (only black 0 and white 255):
     if pixel > threshold  -> 255 (white)
     else                  -> 0   (black)
Think of it as asking every pixel ONE yes/no question: "are you brighter than T?"

Run:  python day4_02_thresholding.py
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from cv_helpers import load_image, make_document_image, show, OUTPUT_DIR

img = cv2.cvtColor(load_image("sample.jpg"), cv2.COLOR_BGR2GRAY)

# =================================================================
# 1. GLOBAL threshold - ONE cut-off for the WHOLE image
#    ret, out = cv2.threshold(src, thresh, maxval, type)
#      thresh = the cut-off (127 = middle of 0-255)
#      maxval = value given to pixels that pass (255 = white)
#      ret    = the threshold that was used (the lecture throws it away with "_")
# =================================================================
ret, thresh_global = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
print("Global threshold used:", ret)

# The 5 threshold "types" (common exam question):
types = {
    "THRESH_BINARY":     cv2.THRESH_BINARY,      # > T -> 255, else 0
    "THRESH_BINARY_INV": cv2.THRESH_BINARY_INV,  # > T -> 0,   else 255 (inverted)
    "THRESH_TRUNC":      cv2.THRESH_TRUNC,       # > T -> T,   else unchanged (caps brightness)
    "THRESH_TOZERO":     cv2.THRESH_TOZERO,      # > T -> unchanged, else 0
    "THRESH_TOZERO_INV": cv2.THRESH_TOZERO_INV,  # > T -> 0,   else unchanged
}
ramp = np.tile(np.arange(256, dtype=np.uint8), (40, 1))      # a left-to-right 0..255 ramp makes the rule visible
show([ramp] + [cv2.threshold(ramp, 127, 255, t)[1] for t in types.values()],
     ["Input ramp 0 -> 255"] + list(types), filename="day4_02a_threshold_types.png", cols=3)

# =================================================================
# 2. ADAPTIVE threshold - a DIFFERENT cut-off for every small neighbourhood
#    cv2.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C)
#      adaptiveMethod: ADAPTIVE_THRESH_MEAN_C     -> T = mean of the block   - C
#                      ADAPTIVE_THRESH_GAUSSIAN_C -> T = weighted mean       - C
#      blockSize: size of the neighbourhood (ODD: 3, 5, 11, ...)
#      C: a constant subtracted from the mean (fine-tuning)
#    Great when lighting is UNEVEN (shadows, one side darker).
# =================================================================
thresh_adaptive = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

# =================================================================
# 3. OTSU - lets the computer PICK the best global threshold
#    Pass 0 as the threshold and ADD the flag cv2.THRESH_OTSU.
#    It tries every possible T and keeps the one that makes the two groups
#    (dark pixels vs bright pixels) each as uniform as possible
#    = "minimises intra-class (within-group) variance".
#    Works best when the histogram has TWO humps (bimodal).
# =================================================================
otsu_T, thresh_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print("Otsu picked threshold:", otsu_T)          # the returned value is finally useful!

# Lab exercise from the lecture: compare all four
show([img, thresh_global, thresh_adaptive, thresh_otsu],
     ["Original", "Global (127)", "Adaptive (mean, 11, 2)", f"Otsu (picked {otsu_T:.0f})"],
     filename="day4_02b_lab_compare.png", cols=4)

# Show WHERE Otsu cut the histogram
plt.figure(figsize=(7, 3))
plt.hist(img.ravel(), 256, range=[0, 256], color="#6e7781")
plt.axvline(otsu_T, color="#d1242f", lw=2, label=f"Otsu threshold = {otsu_T:.0f}")
plt.axvline(127, color="#0969da", lw=2, ls="--", label="fixed 127")
plt.legend(); plt.xlabel("pixel intensity"); plt.ylabel("frequency"); plt.title("Histogram + thresholds")
plt.savefig(os.path.join(OUTPUT_DIR, "day4_02c_otsu_on_histogram.png"), dpi=100, bbox_inches="tight")
plt.close()

# =================================================================
# 4. WHY ADAPTIVE EXISTS: uneven lighting
#    A page with a shadow: the left side is dark, the right side is bright.
#    Any SINGLE threshold either blacks out the left or loses text on the right.
# =================================================================
doc = make_document_image(noise=False)
_, doc_global = cv2.threshold(doc, 127, 255, cv2.THRESH_BINARY)
otsu_doc_T, doc_otsu = cv2.threshold(doc, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
doc_mean = cv2.adaptiveThreshold(doc, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, 10)
# Blur first: in empty paper areas the "local mean" is just noise, so tiny noise decides
# black/white and you get speckles. A light blur removes that noise -> clean background.
doc_blur = cv2.GaussianBlur(doc, (5, 5), 0)
doc_gauss = cv2.adaptiveThreshold(doc_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 10)
show([doc, doc_global, doc_otsu, doc_mean, doc_gauss],
     ["Page with a shadow", "Global 127: left side lost", f"Otsu ({otsu_doc_T:.0f}): still one cut-off",
      "Adaptive MEAN: text OK, speckled paper", "Blur + adaptive GAUSSIAN: clean"],
     filename="day4_02d_uneven_lighting.png", cols=3)

# ---------------------------------------------------------------
# WHY THE LECTURE'S ADAPTIVE RESULT LOOKS LIKE A PENCIL SKETCH:
#   blockSize=11 is tiny compared with a 5472 x 3648 photo. In big flat areas,
#   every 11x11 block is almost uniform, so tiny noise decides black/white
#   -> a grainy "sketch". Bigger blockSize and bigger C give cleaner results.
#
# TRY THIS:
#   a) Change 127 to 60 and 200 in the global threshold. What changes?
#   b) Adaptive: try blockSize 51 and C = 5, 15. (blockSize must be ODD and > 1)
#   c) Blur before Otsu: cv2.GaussianBlur(img, (5,5), 0) - cleaner result?
#   d) Use THRESH_BINARY_INV with Otsu so the dark objects become white.
# ---------------------------------------------------------------

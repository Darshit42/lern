"""
DAY 4 - Script 1: Image histograms (+ histogram equalization)
==============================================================
Lecture topic: "Image Histograms - show distribution of pixel intensities;
useful for analyzing contrast and brightness."

A histogram is a bar chart that COUNTS pixels:
   x-axis = brightness value (0 ... 255)
   y-axis = how many pixels have that value
It ignores WHERE pixels are - it only tells you how many dark / mid / bright pixels exist.

Run:  python day4_01_histograms.py
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from cv_helpers import load_image, OUTPUT_DIR

# The lecture loads straight to grayscale with the flag 0:
#     img = cv2.imread('sample.jpg', 0)        # 0 = cv2.IMREAD_GRAYSCALE
img = cv2.cvtColor(load_image("sample.jpg"), cv2.COLOR_BGR2GRAY)

# ---------------------------------------------------------------
# 1. Two ways to compute a histogram
# ---------------------------------------------------------------
# (a) Lecture way - matplotlib counts for you:
#       plt.hist(img.ravel(), 256, [0, 256])
#     (newer matplotlib wants the range as a keyword: plt.hist(img.ravel(), 256, range=[0, 256]))
#     img.ravel() flattens the 2-D grid into one long list of numbers.
#     256 = number of bars ("bins"), [0, 256] = the range to cover.
#
# (b) OpenCV way - gives you the counts as numbers:
hist = cv2.calcHist([img], [0], None, [256], [0, 256])
#                    images  channel  mask  bins   range
# OpenCV 4 returns shape (256, 1), OpenCV 5 returns (256,). .ravel() makes both a flat list of 256 counts.
hist = hist.ravel()
print("hist shape:", hist.shape)                        # (256,) -> one count per brightness value
print("pixels with value 0  :", int(hist[0]))
print("pixels with value 255:", int(hist[255]))
print("total pixels         :", int(hist.sum()), "=", img.size)
print("mean brightness      :", round(float(img.mean()), 1))

# ---------------------------------------------------------------
# 2. Reading histograms: dark / bright / low-contrast / high-contrast
# ---------------------------------------------------------------
dark = cv2.convertScaleAbs(img, alpha=0.4, beta=0)          # everything x0.4  -> squashed to the LEFT
bright = cv2.convertScaleAbs(img, alpha=0.5, beta=130)      # halved + 130     -> squashed to the RIGHT
low_contrast = cv2.convertScaleAbs(img, alpha=0.3, beta=90) # all values in a narrow middle band
equalized = cv2.equalizeHist(low_contrast)                  # spreads values over the full 0-255 range
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(low_contrast)   # bonus: local equalization

versions = [("Original", img), ("Too dark (underexposed)", dark), ("Too bright (overexposed)", bright),
            ("Low contrast", low_contrast), ("Equalized (equalizeHist)", equalized), ("CLAHE (bonus)", clahe)]

fig, axes = plt.subplots(2, len(versions), figsize=(4 * len(versions), 6.5))
for i, (title, im) in enumerate(versions):
    axes[0, i].imshow(im, cmap="gray", vmin=0, vmax=255)
    axes[0, i].set_title(title, fontsize=10)
    axes[0, i].axis("off")
    h = cv2.calcHist([im], [0], None, [256], [0, 256]).ravel()
    axes[1, i].fill_between(range(256), h, color="#0969da", step="mid")
    axes[1, i].set_xlim(0, 255)
    axes[1, i].set_yticks([])
    axes[1, i].set_xlabel("pixel intensity")
    axes[1, i].spines[["top", "right", "left"]].set_visible(False)
axes[1, 0].set_ylabel("number of pixels")
plt.tight_layout()
out = os.path.join(OUTPUT_DIR, "day4_01_histograms.png")
plt.savefig(out, dpi=100, bbox_inches="tight")
print("[saved]", out)
if plt.get_backend().lower() != "agg":
    plt.show()
plt.close()

# How to READ them:
#   Hump on the left               -> image is dark
#   Hump on the right              -> image is bright
#   Narrow hump (few values used)  -> low contrast (looks flat/grey)
#   Spread over 0-255              -> good contrast
#   Two separate humps ("bimodal") -> object + background - perfect for thresholding (next script)

# ---------------------------------------------------------------
# 3. Colour histogram: one curve per channel (B, G, R)
# ---------------------------------------------------------------
color = load_image("sample.jpg")
plt.figure(figsize=(7, 3.5))
for ch, col in enumerate(["b", "g", "r"]):          # OpenCV order is B, G, R
    plt.plot(cv2.calcHist([color], [ch], None, [256], [0, 256]), color=col, label=col.upper())
plt.xlim(0, 255)
plt.legend()
plt.title("Colour histogram (one line per channel)")
plt.xlabel("intensity")
plt.ylabel("pixel count")
out = os.path.join(OUTPUT_DIR, "day4_01b_color_histogram.png")
plt.savefig(out, dpi=100, bbox_inches="tight")
print("[saved]", out)
plt.close()

# ---------------------------------------------------------------
# TRY THIS:
#   a) Change the bins from 256 to 16:  cv2.calcHist([img], [0], None, [16], [0, 256])
#   b) Use a MASK: histogram of only the top half of the image
#        mask = np.zeros(img.shape, np.uint8); mask[:img.shape[0]//2, :] = 255
#   c) equalizeHist on the ORIGINAL image - does it look better or worse?
# ---------------------------------------------------------------

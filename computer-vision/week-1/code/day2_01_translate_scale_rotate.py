"""
DAY 2 - Script 1: Translation, Scaling, Rotation
=================================================
Lecture topics: Translation (shift), Scaling (resize), Rotation (turn around a pivot)

Run:  python day2_01_translate_scale_rotate.py
"""

import cv2
import numpy as np
from cv_helpers import load_image, show

img = load_image("sample.jpg")

# The lecture uses `rows` and `cols` - they come from the image shape:
rows, cols = img.shape[:2]          # rows = height, cols = width
print(f"rows (height) = {rows}, cols (width) = {cols}")

# =================================================================
# 1. TRANSLATION = slide the picture.  tx = right, ty = down.
#
#        M = [[1, 0, tx],
#             [0, 1, ty]]
#
#   For every pixel:  new_x = 1*x + 0*y + tx  = x + tx
#                     new_y = 0*x + 1*y + ty  = y + ty
#   The empty area that appears is filled with black.
# =================================================================
tx, ty = 100, 50
M_translate = np.float32([[1, 0, tx],
                          [0, 1, ty]])
translated = cv2.warpAffine(img, M_translate, (cols, rows))   # size is (WIDTH, HEIGHT)!
print("\nTranslation matrix:\n", M_translate)

# Negative numbers shift LEFT / UP:
translated_back = cv2.warpAffine(img, np.float32([[1, 0, -80], [0, 1, -40]]), (cols, rows))

# =================================================================
# 2. SCALING = resize.  fx = width factor, fy = height factor.
#    0.5 = half size, 2 = double size.
#    dsize=None means "work out the size from fx and fy".
#
#    Interpolation = how to invent/merge pixels when resizing:
#      INTER_NEAREST - copy nearest pixel (fast, blocky)
#      INTER_LINEAR  - average of 4 neighbours (default, good all-rounder)
#      INTER_CUBIC   - 16 neighbours (smoother when ENLARGING, slower)
#      INTER_AREA    - best when SHRINKING
# =================================================================
scaled = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
print("\nOriginal shape:", img.shape, " -> scaled 50%:", scaled.shape)

# You can also give an exact size (width, height):
squashed = cv2.resize(img, (cols, rows // 2))   # same width, half height -> looks squashed

# Zoom into a small patch to compare interpolation methods
patch = img[40:80, 40:80]
nearest = cv2.resize(patch, None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST)
cubic = cv2.resize(patch, None, fx=6, fy=6, interpolation=cv2.INTER_CUBIC)

# =================================================================
# 3. ROTATION around the centre.
#    cv2.getRotationMatrix2D(center, angle, scale)
#      center = (x, y) pivot point
#      angle  = degrees, POSITIVE = ANTI-clockwise
#      scale  = 1 keeps the size (0.7 would also shrink it)
#    Corners get cut off because the output canvas stays the same size.
# =================================================================
center = (cols / 2, rows / 2)
M_rotate = cv2.getRotationMatrix2D(center, 45, 1)
rotated = cv2.warpAffine(img, M_rotate, (cols, rows))
print("\nRotation matrix (45 deg):\n", np.round(M_rotate, 3))

# Shrinking while rotating keeps the corners inside the frame:
rotated_fit = cv2.warpAffine(img, cv2.getRotationMatrix2D(center, 45, 0.6), (cols, rows))

# Simple 90-degree rotations have a shortcut (no cropping, no black areas):
rot90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
flipped = cv2.flip(img, 1)   # 1 = mirror left-right, 0 = upside down, -1 = both

show([img, translated, translated_back, scaled, squashed, rotated, rotated_fit, rot90, flipped],
     ["Original", "Translated (+100, +50)", "Translated (-80, -40)",
      f"Scaled 50% {scaled.shape[1]}x{scaled.shape[0]}", "Resized to half height",
      "Rotated 45 (corners cut)", "Rotated 45, scale 0.6", "cv2.rotate 90 clockwise",
      "Flipped left-right"],
     filename="day2_01_translate_scale_rotate.png", cols=3)

show([nearest, cubic], ["Zoom x6: INTER_NEAREST (blocky)", "Zoom x6: INTER_CUBIC (smooth)"],
     filename="day2_01b_interpolation.png")

# ---------------------------------------------------------------
# TRY THIS:
#   a) Rotate by -45. Which way does it turn now?   (clockwise)
#   b) Rotate around the top-left corner: center = (0, 0)
#   c) Make translated output bigger so nothing is lost:
#        cv2.warpAffine(img, M_translate, (cols + tx, rows + ty))
# ---------------------------------------------------------------

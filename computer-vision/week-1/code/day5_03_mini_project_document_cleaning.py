"""
DAY 5 - MINI PROJECT: Noise removal in scanned documents
=========================================================
Lecture task:  Input  = noisy scanned document
               Output = cleaned BINARY version using opening & closing

The lecture's code ran opening + closing on the grayscale Pantone photo, so the
"Cleaned Document" looks almost the same as the original. Two things were missing:
   1. The task asks for a BINARY output, so threshold first (Day 4).
   2. The TEXT should be WHITE (foreground) - so invert with THRESH_BINARY_INV.
      Then "opening removes specks, closing fills holes" means exactly what it says.

Full pipeline:
   grayscale -> threshold (INV, Otsu) -> opening (remove specks) -> closing (fill gaps)
             -> invert back to black text on white paper

Run:  python day5_03_mini_project_document_cleaning.py
"""

import cv2
import numpy as np
from cv_helpers import make_document_image, show

doc = make_document_image(shadow=False, noise=True)          # page with dark specks

# 1. Binarize: text becomes WHITE (255), paper BLACK (0) - morphology's favourite format
T, binary = cv2.threshold(doc, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# 2. The lecture's two steps, now on the right kind of image.
#    KERNEL SIZE RULE: bigger than the noise, smaller than the thinnest part you want to keep.
#    Our specks are 1 pixel and the letter strokes are ~3 pixels, so 2x2 is right.
#    (The lecture used 3x3 - on thin text that erases thin strokes; see the comparison below.)
kernel = np.ones((2, 2), np.uint8)
doc_open = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)      # removes 1 px specks
doc_clean = cv2.morphologyEx(doc_open, cv2.MORPH_CLOSE, kernel)  # seals small gaps in letters
too_big = cv2.morphologyEx(cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)),
                           cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

# 3. Back to normal "black text on white paper"
final = cv2.bitwise_not(doc_clean)

# How many specks did we remove? Count separate white blobs before/after.
n_before = cv2.connectedComponents(binary)[0] - 1
n_after = cv2.connectedComponents(doc_clean)[0] - 1
print(f"Separate white blobs: before = {n_before}, after = {n_after}")

# What the lecture did (grayscale, no threshold) for comparison:
lecture_way = cv2.morphologyEx(cv2.morphologyEx(doc, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel)

show([doc, lecture_way, binary, doc_open, doc_clean, too_big, final],
     ["Noisy scan", "Lecture way (grayscale): cleaner, but NOT binary", f"Binary INV (Otsu T={T:.0f})",
      "After OPENING 2x2: specks gone", "After CLOSING 2x2: letters solid",
      "3x3 kernel: too big, thin strokes eaten", "Final: inverted back"],
     filename="day5_03_document_cleaning.png", cols=3)

# ---------------------------------------------------------------
# WAIT - the "lecture way" removed the specks too. How?
#   On grayscale, OPENING removes small BRIGHT things and CLOSING removes small
#   DARK things. Our specks are DARK, so it was the CLOSING step that removed
#   them - opening did nothing useful. And the output is still grey, not the
#   clean binary page the task asked for. Thresholding with INV first (text =
#   white) makes the roles clear: opening -> specks, closing -> holes.
#
# TRY THIS:
#   a) Increase the kernel to (5,5). How much of the text survives?
#   b) Try noise=True AND shadow=True in make_document_image, then use adaptive
#      thresholding from Day 4 in step 1.
# ---------------------------------------------------------------

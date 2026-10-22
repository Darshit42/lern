"""
DAY 4 - MINI PROJECT: Document Binarizer
=========================================
Lecture task:  Input  = scanned document image
               Output = clean binary (black & white) version using Otsu's method

Binarization = turning a grey scan into pure black text on pure white paper.
Smaller files, easier to read, and the first step of OCR (text recognition).

The lecture ran it on the Pantone PHOTO - not a document - so the result was
blotchy. Here we use a real-looking page, and add a smarter "auto" mode.

Run:  python day4_03_mini_project_document_binarizer.py
"""

import os
import cv2
from cv_helpers import make_document_image, show, OUTPUT_DIR


def binarize(doc, method="otsu"):
    """doc must be GRAYSCALE. Returns (binary image, description)."""
    if method == "otsu":                                   # the lecture's version
        T, out = cv2.threshold(doc, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return out, f"Otsu (T={T:.0f})"
    if method == "blur+otsu":                              # blur first -> less speckle
        blurred = cv2.GaussianBlur(doc, (5, 5), 0)
        T, out = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return out, f"Blur + Otsu (T={T:.0f})"
    if method == "adaptive":                               # best for shadows / uneven light
        blurred = cv2.GaussianBlur(doc, (5, 5), 0)         # blur first, or empty paper turns speckled
        out = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 31, 15)
        return out, "Blur + adaptive Gaussian (31, 15)"
    raise ValueError(method)


def auto_binarize(doc):
    """
    Pick a method automatically: if the left and right halves of the page have very
    different brightness, the lighting is uneven -> use adaptive; otherwise Otsu is fine.
    """
    h, w = doc.shape
    left, right = doc[:, : w // 2].mean(), doc[:, w // 2:].mean()
    uneven = abs(left - right) > 25
    out, name = binarize(doc, "adaptive" if uneven else "blur+otsu")
    return out, f"AUTO -> {name}"


# Case 1: evenly lit page - Otsu alone is great
even = make_document_image(shadow=False, noise=False)
# Case 2: page with a shadow - Otsu struggles, adaptive wins
shadowed = make_document_image(shadow=True, noise=False)

results = []
for doc in (even, shadowed):
    results.append((doc, "Input"))
    for m in ("otsu", "adaptive"):
        results.append(binarize(doc, m))
    results.append(auto_binarize(doc))

show([r[0] for r in results], [r[1] for r in results],
     filename="day4_03_document_binarizer.png", cols=4)

# Save the cleaned page to disk, like a real tool would:
clean, _ = auto_binarize(shadowed)
out_path = os.path.join(OUTPUT_DIR, "binarized_document.png")
cv2.imwrite(out_path, clean)
print("Saved", out_path)

# ---------------------------------------------------------------
# TRY THIS:
#   a) Photograph a real page with your phone, save it as page.jpg, then:
#        doc = cv2.imread("page.jpg", 0); show(auto_binarize(doc)[0])
#   b) Change the 25 in auto_binarize - when does it switch methods?
# ---------------------------------------------------------------

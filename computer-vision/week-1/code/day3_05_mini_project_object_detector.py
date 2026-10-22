"""
DAY 3 - MINI PROJECT: Edge-based object detector
=================================================
Lecture task:  Input  = photo of simple shapes
               Output = contours + labels

Pipeline:  colour -> grayscale -> (blur) -> Canny edges -> findContours
           -> bounding boxes + labels

The lecture's version put green boxes on lots of tiny bits of noise/text.
This version adds the 3 standard fixes and also names each shape.

Run:  python day3_05_mini_project_object_detector.py
"""

import cv2
import numpy as np
from cv_helpers import load_image, make_practice_image, show


def lecture_version(img):
    """The code exactly as in the lecture (labels every contour, even noise)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    # findContours returns (contours, hierarchy). The lecture ignores hierarchy with "_".
    #   RETR_EXTERNAL        -> only the OUTER outline of each object (no holes inside)
    #   CHAIN_APPROX_SIMPLE  -> store only corner points of straight lines (saves memory)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    out = img.copy()
    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)        # smallest upright box around the contour
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(out, f"Obj{i + 1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return out, len(contours)


def name_shape(cnt):
    """Guess the shape from how many corners its simplified outline has."""
    perimeter = cv2.arcLength(cnt, True)
    corners = cv2.approxPolyDP(cnt, 0.03 * perimeter, True)   # simplify outline to a polygon
    n = len(corners)
    if n == 3:
        return "Triangle"
    if n == 4:
        x, y, w, h = cv2.boundingRect(corners)
        return "Square" if 0.9 <= w / h <= 1.1 else "Rectangle"
    if n == 5:
        return "Pentagon"
    if n > 7:
        x, y, w, h = cv2.boundingRect(cnt)
        return "Circle" if 0.9 <= w / h <= 1.1 else "Ellipse"
    return f"{n}-gon"


def improved_version(img, min_area=2000, low=20, high=60):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)              # FIX 1: blur away noise first
    # Lower thresholds than the lecture's (50, 150): the YELLOW rectangle is grey-level 205
    # on a 230 background - only 25 apart - so with (50, 150) Canny misses it entirely.
    # Lesson: edges are found on BRIGHTNESS; two different colours of similar brightness
    # look the same in grayscale.
    edges = cv2.Canny(blurred, low, high)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))     # FIX 2: close small gaps in outlines
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    out = img.copy()
    kept = 0
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:                  # FIX 3: ignore tiny specks (text, noise)
            continue
        kept += 1
        label = name_shape(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.drawContours(out, [cnt], -1, (0, 255, 0), 2)     # draw the actual outline
        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 0, 0), 1)
        cv2.putText(out, label, (x, max(15, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
    return out, kept, edges


shapes = make_practice_image()        # an image of simple shapes, as the lecture asks for
lect, n_lect = lecture_version(shapes)
better, n_better, edges = improved_version(shapes)
print(f"Lecture version found {n_lect} contours (includes text bits).")
print(f"Improved version kept {n_better} real objects.")

show([shapes, edges, lect, better],
     ["Input: simple shapes", "Edges (blur + Canny + dilate)",
      f"Lecture version: {n_lect} 'objects'", f"Improved: {n_better} labelled shapes"],
     filename="day3_05_object_detector.png", cols=2, figsize_per=5)

# Run on your own photo too (if sample.jpg exists):
photo = load_image("sample.jpg")
show([lecture_version(photo)[0], improved_version(photo, min_area=1500, low=50, high=150)[0]],
     ["Your image: lecture version", "Your image: improved version"],
     filename="day3_05b_object_detector_your_image.png")

# ---------------------------------------------------------------
# TRY THIS:
#   a) Set min_area=0 in improved_version - the text bits come back.
#   a2) Set low=50, high=150 - the yellow rectangle disappears. Why? (see FIX 1 comment)
#   b) Use cv2.RETR_TREE instead of RETR_EXTERNAL - inner outlines appear too.
#   c) Draw your own shapes in Paint, save as shapes.png, and load it with
#      load_image("shapes.png").
# ---------------------------------------------------------------

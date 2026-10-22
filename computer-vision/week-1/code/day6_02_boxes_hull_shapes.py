"""
DAY 6 - Script 2: Bounding boxes, Convex hull, Shape classification
====================================================================
Lecture topics:
  Bounding box  - smallest rectangle that encloses a contour (object localization)
  Convex hull   - smallest CONVEX polygon enclosing a contour (shape simplification)
                  = "stretch a rubber band around the shape"
  Shape classification - approximate the contour by a polygon and COUNT its corners

Run:  python day6_02_boxes_hull_shapes.py
"""

import cv2
import numpy as np
from cv_helpers import make_shapes_image, show

img = make_shapes_image()
_, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# =================================================================
# 1. Three kinds of "box" around a contour
# =================================================================
boxes = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
for cnt in contours:
    # (a) upright bounding box - the lecture's. Always axis-aligned.
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(boxes, (x, y), (x + w, y + h), (255, 0, 0), 2)          # blue (BGR)
    # (b) ROTATED box - the tightest rectangle at any angle
    rect = cv2.minAreaRect(cnt)                  # ((cx, cy), (w, h), angle)
    corners = cv2.boxPoints(rect).astype(np.int32)
    cv2.drawContours(boxes, [corners], 0, (0, 0, 255), 2)                  # red
    # (c) smallest enclosing circle
    (cx, cy), r = cv2.minEnclosingCircle(cnt)
    cv2.circle(boxes, (int(cx), int(cy)), int(r), (0, 170, 0), 1)          # green

# =================================================================
# 2. Convex hull + "solidity"
#    Convex = no dents (every corner bulges outwards). A star is NOT convex.
#    solidity = contour area / hull area -> 1.0 for convex shapes, smaller for dented ones.
# =================================================================
hulls = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
for cnt in contours:
    hull = cv2.convexHull(cnt)
    cv2.drawContours(hulls, [hull], 0, (0, 0, 255), 2)
    solidity = cv2.contourArea(cnt) / cv2.contourArea(hull)
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.putText(hulls, f"solidity {solidity:.2f}", (x, y + h + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 0, 0), 1)
    print(f"convex? {cv2.isContourConvex(cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True))!s:5}  "
          f"solidity = {solidity:.2f}")

# =================================================================
# 3. Shape classification by counting corners (the lecture's rule)
#    approx = cv2.approxPolyDP(cnt, epsilon, closed=True)
#      epsilon = how far the simplified outline may stray from the real one.
#      Lecture: epsilon = 0.02 * perimeter (2% of the outline length).
#    corners: 3 -> triangle, 4 -> rectangle, >6 -> circle, else polygon
# =================================================================
labels = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
    n = len(approx)
    shape = "Triangle" if n == 3 else "Rectangle" if n == 4 else "Circle" if n > 6 else "Polygon"
    cv2.drawContours(labels, [approx], 0, (0, 200, 0), 2)
    for (px, py) in approx[:, 0]:
        cv2.circle(labels, (int(px), int(py)), 4, (0, 0, 255), -1)        # the corners it found
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.putText(labels, f"{shape} ({n})", (x, max(15, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

show([boxes, hulls, labels],
     ["boundingRect (blue), minAreaRect (red), minEnclosingCircle (green)",
      "Convex hulls - note the star", "approxPolyDP corners (red) + lecture's labels"],
     filename="day6_02a_boxes_hulls_labels.png", cols=1, figsize_per=8)
# Notice the lecture's rule is fooled: the star has 10 corners -> "Circle",
# pentagon/hexagon are just "Polygon". The mini-project fixes this.

# =================================================================
# 4. How epsilon changes the approximation (on the circle)
#    small epsilon -> follows the curve closely -> MANY corners
#    big epsilon   -> rough outline             -> FEW corners
#    That's why a circle counts as "> 6 corners" at 2%, but would look like a
#    square or triangle with a large epsilon.
# =================================================================
circle = max(contours, key=lambda c: 4 * np.pi * cv2.contourArea(c) / cv2.arcLength(c, True) ** 2)  # most circular
outs, titles = [], []
for frac in (0.002, 0.02, 0.06, 0.1):
    canvas = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    approx = cv2.approxPolyDP(circle, frac * cv2.arcLength(circle, True), True)
    cv2.drawContours(canvas, [approx], 0, (0, 0, 255), 3)
    for (px, py) in approx[:, 0]:
        cv2.circle(canvas, (int(px), int(py)), 5, (0, 160, 0), -1)
    x, y, w, h = cv2.boundingRect(circle)
    outs.append(canvas[y - 15:y + h + 15, x - 15:x + w + 15])
    titles.append(f"epsilon = {frac} x perimeter: {len(approx)} corners")
show(outs, titles, filename="day6_02b_epsilon.png", cols=4)

# ---------------------------------------------------------------
# TRY THIS:
#   a) Aspect ratio w/h from boundingRect tells a square (~1.0) from a rectangle.
#   b) Change 0.02 to 0.05 in step 3. Which labels change?
#   c) Extent = contour area / bounding box area. Compute it - which shape is lowest?
# ---------------------------------------------------------------

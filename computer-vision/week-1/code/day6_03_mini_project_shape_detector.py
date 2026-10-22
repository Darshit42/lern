"""
DAY 6 - MINI PROJECT: Shape Detector
=====================================
Lecture task:  Input  = image with multiple geometric shapes
               Output = contours, bounding boxes, convex hulls, and labels

The lecture ran it on the Pantone PHOTO (its 'sample.png' is the same photo), so it
drew hundreds of tiny contours on noise and text. This version:
  1. uses a real shapes image,
  2. thresholds with BINARY_INV (+ Otsu) so shapes are the white foreground,
  3. ignores tiny contours (area filter),
  4. names more shapes: square vs rectangle, pentagon, hexagon, star, ellipse,
  5. draws contour + bounding box + convex hull + label, as the task asks.

Run:  python day6_03_mini_project_shape_detector.py
"""

import cv2
import numpy as np
from cv_helpers import make_shapes_image, load_image, show


def lecture_shape_detector(gray):
    """The lecture's version (takes an image instead of a path)."""
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        x, y = approx[0][0]
        n = len(approx)
        shape = "Triangle" if n == 3 else "Rectangle" if n == 4 else "Circle" if n > 6 else "Polygon"
        cv2.drawContours(out, [cnt], -1, (0, 255, 0), 2)
        cv2.putText(out, shape, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    return out, len(contours)


def classify(cnt):
    perimeter = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
    n = len(approx)
    area = cv2.contourArea(cnt)
    hull_area = cv2.contourArea(cv2.convexHull(cnt))
    solidity = area / hull_area                      # < ~0.8 means big dents -> star-like
    x, y, w, h = cv2.boundingRect(approx)
    circularity = 4 * np.pi * area / perimeter ** 2  # 1.0 for a perfect circle, lower otherwise

    if solidity < 0.8:
        return "Star"
    if n == 3:
        return "Triangle"
    if n == 4:
        return "Square" if 0.9 <= w / h <= 1.1 else "Rectangle"
    if n == 5:
        return "Pentagon"
    if n == 6:
        return "Hexagon"
    if circularity > 0.85:
        return "Circle"
    return "Ellipse"


def shape_detector(gray, min_area=300):
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    found = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        name = classify(cnt)
        found.append(name)
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.drawContours(out, [cnt], -1, (0, 200, 0), 3)                       # contour - green
        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 0, 0), 2)             # box - blue
        cv2.drawContours(out, [cv2.convexHull(cnt)], 0, (0, 0, 255), 1)        # hull - red
        M = cv2.moments(cnt)
        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.putText(out, name, (cx - tw // 2, y + h + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return out, found


shapes = make_shapes_image()
lect, n_lect = lecture_shape_detector(shapes)
better, found = shape_detector(shapes)
print(f"Lecture version: {n_lect} contour(s) - THRESH_BINARY made the page the object.")
print("Improved version found:", found)

photo = cv2.cvtColor(load_image("sample.jpg"), cv2.COLOR_BGR2GRAY)
lect_photo, n_photo = lecture_shape_detector(photo)
print(f"Lecture version on a PHOTO: {n_photo} contours (lots of noise - photos aren't shape worksheets)")

show([lect, better, lect_photo],
     [f"Lecture version on shapes: {n_lect} contour", f"Improved: {len(found)} shapes labelled",
      f"Lecture version on a photo: {n_photo} contours"],
     filename="day6_03_shape_detector.png", cols=1, figsize_per=8)

# ---------------------------------------------------------------
# TRY THIS:
#   a) Draw your own shapes in Paint (black on white), save as shapes.png and run:
#        g = cv2.imread("shapes.png", 0); show(shape_detector(g)[0])
#   b) Add "Octagon" for 8 corners. What epsilon do you need?
#   c) Print the circularity of each shape - which is closest to 1?
# ---------------------------------------------------------------

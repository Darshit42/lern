"""
DAY 6 - Script 1: Contour detection (+ area, perimeter, centroid)
==================================================================
Lecture topic: "Contours are curves joining continuous points along boundaries.
Useful for shape analysis and object detection."

A contour is the OUTLINE of a white blob, stored as a list of (x, y) points -
like tracing around a coin with a pencil.

Pipeline:  grayscale -> threshold to binary -> cv2.findContours -> cv2.drawContours

Run:  python day6_01_contours.py
"""

import cv2
import numpy as np
from cv_helpers import make_shapes_image, show

img = make_shapes_image()            # black shapes on WHITE paper (like a printed worksheet)

# =================================================================
# 1. THE #1 CONTOUR BUG: findContours looks for WHITE objects on BLACK.
#    The lecture used THRESH_BINARY. With dark shapes on white paper that makes
#    the PAPER the "object" -> you get one giant contour around the whole page.
#    Fix: THRESH_BINARY_INV so the shapes become white.
# =================================================================
_, thresh_wrong = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)       # lecture
_, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)          # fixed

contours_wrong, _ = cv2.findContours(thresh_wrong, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"THRESH_BINARY     -> {len(contours_wrong)} contour(s) (the page border!)")
print(f"THRESH_BINARY_INV -> {len(contours)} contours (the 8 shapes)")

# Draw: cv2.drawContours(image, contours, contourIdx, colour, thickness)
#        contourIdx = -1 means "draw ALL of them"; thickness -1 would FILL them
draw_wrong = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
cv2.drawContours(draw_wrong, contours_wrong, -1, (0, 255, 0), 4)
img_contours = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)       # colour copy so we can draw in green
cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 3)

show([thresh_wrong, draw_wrong, thresh, img_contours],
     ["THRESH_BINARY: paper is white", f"-> {len(contours_wrong)} contour = page edge",
      "THRESH_BINARY_INV: shapes are white", f"-> {len(contours)} contours = the shapes"],
     filename="day6_01a_contours_inv_fix.png", cols=2, figsize_per=5)

# =================================================================
# 2. MEASURING each contour - "shape descriptors"
# =================================================================
info = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
print(f"\n{'#':>2} {'area':>8} {'perimeter':>10} {'centroid':>12} {'points':>7}")
for i, cnt in enumerate(contours):
    area = cv2.contourArea(cnt)                    # number of pixels inside (approx.)
    perimeter = cv2.arcLength(cnt, True)           # length of the outline; True = closed shape
    M = cv2.moments(cnt)                           # "moments" = weighted sums used to find the centre
    cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])   # centroid = centre of mass
    print(f"{i:>2} {area:>8.0f} {perimeter:>10.1f} {str((cx, cy)):>12} {len(cnt):>7}")
    cv2.circle(info, (cx, cy), 5, (0, 0, 255), -1)
    cv2.putText(info, f"A={area:.0f}", (cx - 35, cy + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
# Note: M["m00"] is the area too. If it is 0 (a contour that's just a line), skip it to avoid dividing by 0.

# =================================================================
# 3. CHAIN_APPROX_NONE vs CHAIN_APPROX_SIMPLE - how many points are stored?
# =================================================================
c_none, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
print("\nPoints stored per contour:")
print("  CHAIN_APPROX_NONE  :", [len(c) for c in c_none], "(every boundary pixel)")
print("  CHAIN_APPROX_SIMPLE:", [len(c) for c in contours], "(straight runs -> just their end points)")

# =================================================================
# 4. RETRIEVAL MODES and HIERARCHY - what about shapes with HOLES?
#    RETR_EXTERNAL - only outermost outlines (holes ignored)
#    RETR_LIST     - every outline, no family information
#    RETR_TREE     - every outline + full parent/child tree
#    hierarchy[0][i] = [next, previous, first_child, parent]   (-1 = none)
# =================================================================
ring = np.zeros((220, 460), np.uint8)
cv2.circle(ring, (110, 110), 90, 255, -1)
cv2.circle(ring, (110, 110), 45, 0, -1)          # a hole -> the ring has an outer AND an inner boundary
cv2.rectangle(ring, (250, 30), (430, 190), 255, -1)
cv2.rectangle(ring, (300, 80), (380, 140), 0, -1)
cv2.rectangle(ring, (325, 95), (355, 125), 255, -1)   # an island inside the hole

outs = []
for name, mode in [("RETR_EXTERNAL", cv2.RETR_EXTERNAL), ("RETR_LIST", cv2.RETR_LIST), ("RETR_TREE", cv2.RETR_TREE)]:
    cs, hier = cv2.findContours(ring, mode, cv2.CHAIN_APPROX_SIMPLE)
    canvas = cv2.cvtColor(ring // 3, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(canvas, cs, -1, (0, 255, 0), 3)
    outs.append((canvas, f"{name}: {len(cs)} contours"))
    if name == "RETR_TREE":
        print("\nRETR_TREE hierarchy [next, prev, first_child, parent]:\n", hier[0])

show([info] + [o[0] for o in outs], ["Area + centroid (red dot)"] + [o[1] for o in outs],
     filename="day6_01b_measurements_and_modes.png", cols=2, figsize_per=5)

# ---------------------------------------------------------------
# TRY THIS:
#   a) Use THRESH_BINARY (the lecture's) on the shapes image and count contours.
#   b) Draw only contour number 2:  cv2.drawContours(canvas, contours, 2, (0,0,255), 3)
#   c) Fill all contours: thickness = -1
#   d) Sort contours by area, biggest first:
#        contours = sorted(contours, key=cv2.contourArea, reverse=True)
# ---------------------------------------------------------------

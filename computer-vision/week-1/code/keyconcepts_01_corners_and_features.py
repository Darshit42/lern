"""
KEY CONCEPTS - Script 1: Corners, SIFT / ORB feature matching, HOG
====================================================================
From "CV-Week-1-KeyConcepts.pdf": Edge detection, Corner detection (Harris, Shi-Tomasi,
FAST), SIFT, SURF, HoG.  These are "beyond the labs" topics - know WHAT they are and
WHY they're useful. This script lets you SEE each one.

Big idea - a ladder of "interesting points":
   flat area -> nothing changes when you slide a small window around  -> useless for matching
   edge      -> changes only ACROSS the edge (sliding ALONG it looks the same)
   corner    -> changes in EVERY direction -> unique, easy to find again -> great landmark

Run:  python keyconcepts_01_corners_and_features.py
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from cv_helpers import load_image, make_shapes_image, show, OUTPUT_DIR

shapes = make_shapes_image()
shapes_bgr = cv2.cvtColor(shapes, cv2.COLOR_GRAY2BGR)
# Light blur first: our shapes are drawn without anti-aliasing, so curved edges are
# tiny pixel "staircases" - and every step of a staircase looks like a corner!
smooth = cv2.GaussianBlur(shapes, (5, 5), 0)


def mark(img, points, colour):
    out = img.copy()
    for x, y in points:
        cv2.circle(out, (int(x), int(y)), 7, colour, 2)
    return out


# =================================================================
# 1. HARRIS corner detector
#    cv2.cornerHarris(gray_float32, blockSize, ksize, k)
#      blockSize = neighbourhood size, ksize = Sobel size, k = sensitivity (0.04-0.06)
#    Output: a "cornerness" score for EVERY pixel; big = corner.
#    We keep pixels above 5% of the best score, then take the centre of each blob.
# =================================================================
harris = cv2.cornerHarris(np.float32(smooth), 2, 3, 0.04)
strong = (harris > 0.05 * harris.max()).astype(np.uint8)
n, _, _, centroids = cv2.connectedComponentsWithStats(strong)
harris_img = mark(shapes_bgr, centroids[1:], (0, 0, 255))

# =================================================================
# 2. SHI-TOMASI ("good features to track") - an improved Harris that returns the
#    N best corners as a list of points. Used for tracking (see optical flow script).
# =================================================================
pts = cv2.goodFeaturesToTrack(smooth, maxCorners=60, qualityLevel=0.05, minDistance=10)
shi_img = mark(shapes_bgr, pts.reshape(-1, 2), (0, 160, 0))

# =================================================================
# 3. FAST - a very quick corner test: look at a ring of 16 pixels around a point;
#    if a long unbroken arc is all brighter (or all darker) than the centre -> corner.
# =================================================================
kps = cv2.FastFeatureDetector_create(threshold=40).detect(smooth, None)
fast_img = mark(shapes_bgr, [k.pt for k in kps], (255, 0, 0))

show([harris_img, shi_img, fast_img],
     [f"Harris: {n - 1} corners", f"Shi-Tomasi: {len(pts)} best corners", f"FAST: {len(kps)} keypoints"],
     filename="keyconcepts_01a_corners.png")
# The real polygon corners (tips of triangle, star, square...) are found; smooth curves
# (circle, ellipse) get few or none - there's no single point where the direction jumps.

# =================================================================
# 4. SIFT and ORB - detect keypoints AND describe them, then MATCH across images
#    SIFT: Scale-Invariant Feature Transform (Lowe, 2004)
#      pipeline: Gaussian scale space -> Difference of Gaussians extrema ->
#                assign orientation -> 128-number descriptor per keypoint
#      robust to SCALE and ROTATION; heavier to compute.
#    SURF: a faster SIFT-like method (2006); patented, NOT in standard OpenCV builds.
#    ORB: free, fast alternative (FAST corners + binary BRIEF descriptors).
# =================================================================
photo = cv2.cvtColor(load_image("sample.jpg"), cv2.COLOR_BGR2GRAY)
h, w = photo.shape
# Make a second "view": rotate 25 degrees and shrink to 70% - a different camera pose
M = cv2.getRotationMatrix2D((w / 2, h / 2), 25, 0.7)
view2 = cv2.warpAffine(photo, M, (w, h))

sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(photo, None)
kp2, des2 = sift.detectAndCompute(view2, None)
print(f"SIFT: {len(kp1)} and {len(kp2)} keypoints; each descriptor has {des1.shape[1]} numbers")

# Match descriptors: for each keypoint, find its 2 nearest neighbours in the other image.
# Lowe's ratio test: keep a match only if the best is clearly better than the 2nd best.
matches = cv2.BFMatcher().knnMatch(des1, des2, k=2)
good = [m for m, n in matches if m.distance < 0.75 * n.distance]
print(f"SIFT good matches after ratio test: {len(good)}")
sift_vis = cv2.drawMatches(photo, kp1, view2, kp2, sorted(good, key=lambda m: m.distance)[:40], None,
                           flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

orb = cv2.ORB_create(1000)
okp1, odes1 = orb.detectAndCompute(photo, None)
okp2, odes2 = orb.detectAndCompute(view2, None)
omatches = sorted(cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(odes1, odes2), key=lambda m: m.distance)
print(f"ORB: {len(okp1)} keypoints, descriptor = {odes1.shape[1]} bytes (binary); {len(omatches)} matches")
orb_vis = cv2.drawMatches(photo, okp1, view2, okp2, omatches[:40], None,
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

show([sift_vis, orb_vis],
     ["SIFT: 40 best matches between original and rotated+shrunk copy",
      "ORB: same idea, faster, free"],
     filename="keyconcepts_01b_sift_orb_matching.png", cols=1, figsize_per=9)
# Uses: panorama stitching, object recognition, 3D reconstruction, tracking.


# =================================================================
# 5. HOG - Histogram of Oriented Gradients (written by hand, so you see every step)
#    pipeline: compute gradients -> split into cells (8x8) -> per cell, histogram of
#              gradient DIRECTIONS (weighted by strength) -> normalise over blocks
#    Classic use: pedestrian detection (HOG + SVM, Dalal & Triggs 2005).
# =================================================================
def hog_cells(gray, cell=8, bins=9):
    gx = cv2.Sobel(np.float32(gray), cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(np.float32(gray), cv2.CV_32F, 0, 1, ksize=1)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    ang = ang % 180                                    # "unsigned" gradients: 0-180 degrees
    ch, cw = gray.shape[0] // cell, gray.shape[1] // cell
    hist = np.zeros((ch, cw, bins), np.float32)
    for i in range(ch):
        for j in range(cw):
            a = ang[i * cell:(i + 1) * cell, j * cell:(j + 1) * cell].ravel()
            m = mag[i * cell:(i + 1) * cell, j * cell:(j + 1) * cell].ravel()
            hist[i, j] = np.histogram(a, bins=bins, range=(0, 180), weights=m)[0]
    return hist


crop = cv2.resize(shapes[30:170, 350:530], (180, 140))       # the triangle
cells = hog_cells(crop)
print(f"HOG: {cells.shape[0]}x{cells.shape[1]} cells x {cells.shape[2]} direction bins "
      f"= {cells.size} numbers describe this patch")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].imshow(crop, cmap="gray"); axes[0].set_title("Patch"); axes[0].axis("off")
axes[1].imshow(np.zeros_like(crop), cmap="gray")
for i in range(cells.shape[0]):
    for j in range(cells.shape[1]):
        cy, cx = i * 8 + 4, j * 8 + 4
        total = cells[i, j].max() + 1e-6
        for b, v in enumerate(cells[i, j]):
            if v < 0.2 * total:
                continue
            theta = np.radians(b * 20 + 10 + 90)       # draw each bin as a line ALONG the edge
            dx, dy = 4 * np.cos(theta) * v / total, 4 * np.sin(theta) * v / total
            axes[1].plot([cx - dx, cx + dx], [cy - dy, cy + dy], color="white", lw=1)
axes[1].set_title("HOG: each 8x8 cell's dominant edge directions"); axes[1].axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "keyconcepts_01c_hog.png"), dpi=100, bbox_inches="tight")
plt.close()
print("[saved]", os.path.join(OUTPUT_DIR, "keyconcepts_01c_hog.png"))

# ---------------------------------------------------------------
# TRY THIS:
#   a) Harris: change 0.01 to 0.1 - fewer, stronger corners.
#   b) Change the rotation to 90 and the scale to 0.4 - does SIFT still match?
#   c) Lower the ratio test from 0.75 to 0.6 - fewer but more reliable matches.
# ---------------------------------------------------------------

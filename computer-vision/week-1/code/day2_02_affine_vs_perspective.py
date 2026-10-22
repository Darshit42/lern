"""
DAY 2 - Script 2: Affine vs Perspective transformations (+ a document scanner)
===============================================================================
Lecture topics:
  Affine      - keeps parallel lines parallel. Defined by 3 point pairs.
  Perspective - maps any 4-corner shape to another 4-corner shape.
                Parallel lines may meet (like train tracks). Defined by 4 point pairs.

Run:  python day2_02_affine_vs_perspective.py
"""

import cv2
import numpy as np
from cv_helpers import load_image, show

img = load_image("sample.jpg")
rows, cols = img.shape[:2]

# Draw a grid on the image so we can SEE what happens to straight/parallel lines
grid = img.copy()
for x in range(0, cols, 50):
    cv2.line(grid, (x, 0), (x, rows), (60, 60, 60), 1)
for y in range(0, rows, 50):
    cv2.line(grid, (0, y), (cols, y), (60, 60, 60), 1)

# =================================================================
# 1. AFFINE:  "these 3 points should move to those 3 points"
#    (lecture's exact numbers)
# =================================================================
pts1 = np.float32([[50, 50], [200, 50], [50, 200]])     # where points ARE
pts2 = np.float32([[10, 100], [200, 50], [100, 250]])   # where we WANT them
M_affine = cv2.getAffineTransform(pts1, pts2)            # -> 2x3 matrix
affine = cv2.warpAffine(grid, M_affine, (cols, rows))
print("Affine matrix (2x3):\n", np.round(M_affine, 3))

# Mark the 3 points before/after so the mapping is visible
before = grid.copy()
for (x, y) in pts1.astype(int):
    cv2.circle(before, (x, y), 7, (0, 0, 255), -1)
for (x, y) in pts2.astype(int):
    cv2.circle(affine, (x, y), 7, (0, 0, 255), -1)

# =================================================================
# 2. PERSPECTIVE: "these 4 corners should become those 4 corners"
#    Points order here: top-left, top-right, bottom-left, bottom-right
#    (must be the SAME order in pts1 and pts2)
# =================================================================
src = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])
dst = np.float32([[cols * 0.2, rows * 0.1], [cols * 0.8, rows * 0.1], [0, rows], [cols, rows]])
M_persp = cv2.getPerspectiveTransform(src, dst)          # -> 3x3 matrix
perspective = cv2.warpPerspective(grid, M_persp, (cols, rows))
print("\nPerspective matrix (3x3):\n", np.round(M_persp, 4))

show([before, affine, perspective],
     ["Original + grid (red = 3 source pts)",
      "AFFINE: grid lines stay parallel",
      "PERSPECTIVE: vertical lines now meet"],
     filename="day2_02a_affine_vs_perspective.png")


# =================================================================
# 3. REAL USE: document scanning
#    A paper photographed at an angle looks like a tilted 4-sided shape.
#    Perspective transform "flattens" it into a neat rectangle.
# =================================================================
def make_tilted_document():
    """Fake photo: a page with text, seen at an angle, on a dark table."""
    page = np.full((400, 300, 3), 255, np.uint8)
    cv2.putText(page, "EXAM NOTES", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    for i in range(8):
        cv2.line(page, (30, 110 + i * 32), (270, 110 + i * 32), (90, 90, 90), 2)
    corners_page = np.float32([[0, 0], [300, 0], [0, 400], [300, 400]])
    corners_photo = np.float32([[120, 60], [430, 100], [60, 440], [470, 400]])  # where it appears
    M = cv2.getPerspectiveTransform(corners_page, corners_photo)
    photo = cv2.warpPerspective(page, M, (540, 500), borderValue=(40, 60, 80))
    return photo, corners_photo


photo, corners = make_tilted_document()

# Pretend we clicked the 4 corners of the paper (TL, TR, BL, BR):
pts1 = corners
pts2 = np.float32([[0, 0], [300, 0], [0, 400], [300, 400]])
M = cv2.getPerspectiveTransform(pts1, pts2)
scanned = cv2.warpPerspective(photo, M, (300, 400))

marked = photo.copy()
for (x, y) in pts1.astype(int):
    cv2.circle(marked, (x, y), 8, (0, 0, 255), -1)

show([marked, scanned], ["Photo of paper (red = 4 corners)", "After warpPerspective = 'scanned'"],
     filename="day2_02b_document_scanner.png")

# ---------------------------------------------------------------
# WHY WAS THE LECTURE'S PERSPECTIVE OUTPUT JUST A GREY SQUARE?
#   The lecture picked corners around (56,65)...(389,390) on a HUGE photo
#   (5472 x 3648). That small region is the top-left corner of the photo,
#   which was only grey background - so the "zoomed & straightened" result
#   is grey. The code was fine; the chosen points weren't on an object.
#   Lesson: the 4 source points must sit on the corners of the thing you want.
#
# TRY THIS:
#   a) Change one value in pts2 of the affine part and watch the grid.
#   b) Swap two corners in pts2 of the scanner - the page comes out mirrored/flipped.
#   c) Change the output size (300, 400) to (600, 800).
# ---------------------------------------------------------------

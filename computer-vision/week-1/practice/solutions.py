"""
WEEK 1 - SOLUTIONS to exercises.py
===================================
Try exercises.py yourself first! Run this file to see every check PASS:

    python solutions.py
"""

import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))
from cv_helpers import make_practice_image, make_shapes_image, make_document_image

img = make_practice_image()          # 400 x 600 colour image (BGR)
shapes = make_shapes_image()         # 450 x 700 grayscale: 8 black shapes on white (Day 6)


def check(name, condition):
    print(f"{'PASS' if condition else 'FAIL'}  - {name}")


def attempt(name, fn):
    try:
        fn()
    except Exception as e:
        print(f"TODO  - {name}  ({type(e).__name__}: {e})")


# ------------------------------------------------------------------
# EXERCISE 1 (Day 1): get the height, width and number of channels from img.shape
# ------------------------------------------------------------------
def ex1():
    height, width, channels = img.shape
    check("Ex1 shape", (height, width, channels) == (400, 600, 3))


# ------------------------------------------------------------------
# EXERCISE 2 (Day 1): read the BGR value of the pixel at row 100, column 100
#                     (it's inside the red square)
# ------------------------------------------------------------------
def ex2():
    pixel = img[100, 100]                       # [row, col] = [y, x]
    check("Ex2 pixel is red (BGR = 0, 0, 255)", list(pixel) == [0, 0, 255])


# ------------------------------------------------------------------
# EXERCISE 3 (Day 1): convert img to RGB, and to grayscale
# ------------------------------------------------------------------
def ex3():
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    check("Ex3 RGB first pixel of red square is (255,0,0)", list(rgb[100, 100]) == [255, 0, 0])
    check("Ex3 gray has 2 dimensions", gray.ndim == 2)


# ------------------------------------------------------------------
# EXERCISE 4 (Day 1): make a mask of the GREEN circle using HSV + cv2.inRange
#                     hint: green hue is roughly 40..80 in OpenCV
# ------------------------------------------------------------------
def ex4():
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (40, 80, 50), (80, 255, 255))
    check("Ex4 mask is white inside the circle (row 110, col 320)", mask[110, 320] == 255)
    check("Ex4 mask is black on the red square (row 100, col 100)", mask[100, 100] == 0)


# ------------------------------------------------------------------
# EXERCISE 5 (Day 2): translate img 50 px RIGHT and 30 px UP (same output size)
# ------------------------------------------------------------------
def ex5():
    rows, cols = img.shape[:2]
    M = np.float32([[1, 0, 50], [0, 1, -30]])     # up = negative y
    moved = cv2.warpAffine(img, M, (cols, rows))
    check("Ex5 translated correctly", np.array_equal(moved[100 - 30, 100 + 50], img[100, 100]))


# ------------------------------------------------------------------
# EXERCISE 6 (Day 2): resize img to exactly 300 wide x 100 high
# ------------------------------------------------------------------
def ex6():
    resized = cv2.resize(img, (300, 100))       # size is (WIDTH, HEIGHT)
    check("Ex6 resized shape is (100, 300, 3)", resized.shape == (100, 300, 3))


# ------------------------------------------------------------------
# EXERCISE 7 (Day 2): rotate img 90 degrees CLOCKWISE about its centre with
#                     getRotationMatrix2D (keep scale 1, same output size)
# ------------------------------------------------------------------
def ex7():
    rows, cols = img.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -90, 1)   # negative = clockwise
    rotated = cv2.warpAffine(img, M, (cols, rows))
    check("Ex7 angle is clockwise (-90)", np.isclose(M[0, 1], -1.0))


# ------------------------------------------------------------------
# EXERCISE 8 (Day 3): write the hand-made convolution for ONE position:
#                     multiply window and kernel element-wise and sum it
# ------------------------------------------------------------------
def ex8():
    window = np.array([[10, 10, 200], [10, 10, 200], [10, 10, 200]])
    kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    value = np.sum(window * kernel)
    check("Ex8 convolution value = 760", value == 760)


# ------------------------------------------------------------------
# EXERCISE 9 (Day 3): create a 5x5 AVERAGE kernel with numpy and apply it with filter2D
# ------------------------------------------------------------------
def ex9():
    kernel = np.ones((5, 5)) / 25
    out = cv2.filter2D(img, -1, kernel)
    check("Ex9 kernel sums to 1", np.isclose(kernel.sum(), 1))
    check("Ex9 same as cv2.blur", np.abs(out.astype(int) - cv2.blur(img, (5, 5)).astype(int)).max() <= 1)


# ------------------------------------------------------------------
# EXERCISE 10 (Day 3): Sobel in x on the grayscale image, keeping negatives,
#                      then convert to a displayable uint8 image
# ------------------------------------------------------------------
def ex10():
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    display = cv2.convertScaleAbs(sobel_x)
    check("Ex10 sobel is float64", sobel_x.dtype == np.float64)
    check("Ex10 has negative values", sobel_x.min() < 0)
    check("Ex10 display is uint8", display.dtype == np.uint8)


# ------------------------------------------------------------------
# EXERCISE 11 (Day 3): count the shapes: blur -> Canny -> findContours (external),
#                      keep only contours with area > 2000. Expect 5 shapes.
#                      hint: Canny thresholds 20, 60 (the yellow box is low contrast)
# ------------------------------------------------------------------
def ex11():
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 20, 60)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    big = [c for c in contours if cv2.contourArea(c) > 2000]
    check(f"Ex11 found 5 shapes (you found {len(big)})", len(big) == 5)


# ------------------------------------------------------------------
# EXERCISE 12 (Day 4): compute the 256-bin histogram of the grayscale image with
#                      cv2.calcHist (use .ravel() to get a flat list of 256 counts)
# ------------------------------------------------------------------
def ex12():
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    check("Ex12 histogram has 256 bins", len(hist) == 256)
    check("Ex12 counts add up to the number of pixels", int(hist.sum()) == gray.size)


# ------------------------------------------------------------------
# EXERCISE 13 (Day 4): binarize the evenly-lit document with OTSU.
#                      Keep the threshold Otsu picked in T.
# ------------------------------------------------------------------
def ex13():
    doc = make_document_image(shadow=False, noise=False)
    T, binary = cv2.threshold(doc, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    check(f"Ex13 Otsu picked a sensible T (got {T})", 60 < T < 220)
    check("Ex13 result is binary (only 0 and 255)", set(np.unique(binary)) <= {0, 255})


# ------------------------------------------------------------------
# EXERCISE 14 (Day 4): the page has a SHADOW on the left. Blur it (5x5 Gaussian),
#                      then use ADAPTIVE Gaussian thresholding, blockSize 31, C 15.
# ------------------------------------------------------------------
def ex14():
    doc = make_document_image(shadow=True, noise=False)
    blurred = cv2.GaussianBlur(doc, (5, 5), 0)
    out = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)
    check("Ex14 dark (shadowed) empty paper comes out WHITE", out[370:, :120].mean() > 240)


# ------------------------------------------------------------------
# EXERCISE 15 (Day 5): remove the white specks around the square with OPENING (5x5 kernel)
# ------------------------------------------------------------------
def ex15():
    b = np.zeros((100, 100), np.uint8)
    b[30:70, 30:70] = 255
    b[5, 5] = b[90, 10] = b[10, 90] = 255                     # three 1-pixel specks
    kernel = np.ones((5, 5), np.uint8)
    opened = cv2.morphologyEx(b, cv2.MORPH_OPEN, kernel)
    if opened is None:                                        # (OpenCV crashes on None here, so check first)
        raise TypeError("opened is still ___")
    blobs = cv2.connectedComponents(opened)[0] - 1
    check(f"Ex15 only the square is left (blobs = {blobs})", blobs == 1)


# ------------------------------------------------------------------
# EXERCISE 16 (Day 5): fill the black hole inside the square with CLOSING (5x5 kernel)
# ------------------------------------------------------------------
def ex16():
    b = np.zeros((100, 100), np.uint8)
    b[30:70, 30:70] = 255
    b[49:51, 49:51] = 0                                       # a 2x2 hole
    closed = cv2.morphologyEx(b, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    check("Ex16 hole is filled", closed[49, 49] == 255 and closed[50, 50] == 255)


# ------------------------------------------------------------------
# EXERCISE 17 (Day 6): find the 8 shapes. Threshold so the black shapes become WHITE
#                      (127, inverse!), then findContours with RETR_EXTERNAL + CHAIN_APPROX_SIMPLE
# ------------------------------------------------------------------
def ex17():
    _, thresh = cv2.threshold(shapes, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    check(f"Ex17 found 8 contours (you found {len(contours)})", len(contours) == 8)


# ------------------------------------------------------------------
# EXERCISE 18 (Day 6): compute SOLIDITY = contour area / convex hull area for every
#                      shape. The star is the only one far below 1.
# ------------------------------------------------------------------
def ex18():
    _, thresh = cv2.threshold(shapes, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    solidities = [cv2.contourArea(c) / cv2.contourArea(cv2.convexHull(c)) for c in contours]
    check("Ex18 exactly one shape has solidity < 0.8 (the star)", sum(s < 0.8 for s in solidities) == 1)


# ------------------------------------------------------------------
# EXERCISE 19 (Day 6): centroid with moments. The square below is centred at (60, 40).
# ------------------------------------------------------------------
def ex19():
    b = np.zeros((100, 120), np.uint8)
    b[30:51, 50:71] = 255                                     # rows 30-50, cols 50-70
    cnt = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]
    M = cv2.moments(cnt)
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    check(f"Ex19 centroid is (60, 40) (you got ({cx}, {cy}))", round(cx) == 60 and round(cy) == 40)


if __name__ == "__main__":
    for i, fn in enumerate([ex1, ex2, ex3, ex4, ex5, ex6, ex7, ex8, ex9, ex10, ex11,
                                  ex12, ex13, ex14, ex15, ex16, ex17, ex18, ex19], start=1):
        attempt(f"Ex{i}", fn)

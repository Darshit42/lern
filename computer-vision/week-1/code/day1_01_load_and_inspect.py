"""
DAY 1 - Script 1: Loading an image and looking inside it
=========================================================
Lecture topics: "What is Computer Vision?" + "Images as Data"

Big idea: to a computer, an image is just a big table (grid) of numbers.
Run this file:   python day1_01_load_and_inspect.py
"""

import cv2
import numpy as np
from cv_helpers import load_image, show

# ---------------------------------------------------------------
# 1. LOAD the image  (exactly like the lecture: cv2.imread)
# ---------------------------------------------------------------
img = load_image("sample.jpg")

# ---------------------------------------------------------------
# 2. SHAPE = (height, width, channels)
#    height   = number of ROWS of pixels   (up-down)
#    width    = number of COLUMNS of pixels (left-right)
#    channels = 3 for a colour picture (Blue, Green, Red in OpenCV)
# ---------------------------------------------------------------
print("Image shape:", img.shape)           # e.g. (400, 600, 3)
height, width, channels = img.shape
print(f"  height = {height} rows, width = {width} columns, channels = {channels}")
print("Total pixels:", height * width)
print("Total numbers stored:", img.size)   # height * width * channels
print("Data type:", img.dtype)             # uint8 -> whole numbers from 0 to 255

# ---------------------------------------------------------------
# 3. LOOK AT ONE PIXEL:  img[row, column]  ==  img[y, x]
#    ROW comes first! (y first, then x)  <- very common exam trap
# ---------------------------------------------------------------
y, x = 50, 50
b, g, r = img[y, x]
print(f"\nPixel at (row={y}, col={x}) = {img[y, x]}   <- order is [Blue, Green, Red]")
print(f"  Blue={b}, Green={g}, Red={r}")

# ---------------------------------------------------------------
# 4. CHANGE pixels - an image is just an array, so we can edit it
# ---------------------------------------------------------------
edited = img.copy()                        # copy, so the original stays safe
edited[0:60, 0:60] = (255, 255, 255)       # paint the top-left 60x60 block white
edited[:, width // 2] = (0, 0, 0)          # draw a black vertical line down the middle

# ---------------------------------------------------------------
# 5. THE BGR TRAP
#    OpenCV stores colours as B,G,R. Matplotlib expects R,G,B.
#    If you show an OpenCV image in matplotlib WITHOUT converting,
#    red and blue swap places. (This happened in the lecture's
#    "Mode: original" picture - the orange swatches turned BLUE!)
# ---------------------------------------------------------------
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
wrong = img_rgb.copy()  # pretend this RGB array is BGR -> show() will swap it -> colours look wrong

# show() expects BGR and converts for you, so:
#   img      -> displays correctly
#   wrong    -> is RGB being treated as BGR -> red/blue swapped
show([img, edited, wrong],
     ["Original (correct colours)", "Edited pixels", "BGR/RGB mix-up (wrong!)"],
     filename="day1_01_load_and_inspect.png")

# ---------------------------------------------------------------
# TRY THIS (experiments):
#   a) Change y, x above and print different pixels. Find a red one.
#   b) Paint a different colour block, e.g. pure green = (0, 255, 0) in BGR.
#   c) Print img[y, x, 2] - which colour is channel index 2?   (answer: Red)
#   d) Crop: small = img[100:200, 50:300]  then print small.shape
# ---------------------------------------------------------------

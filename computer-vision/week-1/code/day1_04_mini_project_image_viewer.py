"""
DAY 1 - MINI PROJECT: Image Viewer
===================================
Lecture task: build a viewer that loads an image, converts to grayscale,
applies edge detection, and lets you toggle between modes.

This is the lecture's code with ONE BUG FIXED (see "BUG FIX" below).
Run:  python day1_04_mini_project_image_viewer.py
"""

import cv2
import matplotlib.pyplot as plt
from cv_helpers import load_image, show


# ---------- Lecture version (slightly cleaned) ----------
def image_viewer(img, mode="original"):
    if mode == "gray":
        out = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif mode == "edges":
        out = cv2.Canny(img, 100, 200)
    else:
        # BUG FIX: the lecture showed the BGR image straight in matplotlib,
        # so its "Mode: original" picture had blue swatches instead of orange.
        # Converting BGR -> RGB fixes it. (show() below does this for us.)
        out = img
    return out


img = load_image("sample.jpg")

modes = ["original", "gray", "edges"]
outputs = [image_viewer(img, m) for m in modes]
show(outputs, [f"Mode: {m}" for m in modes], filename="day1_04_image_viewer.png")


# ---------- Bonus: an interactive version ----------
# If you run this on your own computer (with a screen), you can toggle modes
# with the keyboard:  o = original, g = gray, e = edges, q = quit.
def interactive_viewer(img):
    mode = "original"
    while True:
        out = image_viewer(img, mode)
        cv2.imshow("Image Viewer  (o / g / e / q)", out)   # cv2.imshow expects BGR - correct as is
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break
        mode = {ord("o"): "original", ord("g"): "gray", ord("e"): "edges"}.get(key, mode)
    cv2.destroyAllWindows()

# Uncomment to try (needs a desktop, and the full 'opencv-python' package):
# interactive_viewer(img)

# ---------------------------------------------------------------
# TRY THIS:
#   a) Add a mode "hsv" that shows the hue channel.
#   b) Add a mode "blur" using cv2.GaussianBlur(img, (9, 9), 0).
#   c) Change Canny thresholds (100, 200) to (30, 80) - more or fewer edges?
# ---------------------------------------------------------------

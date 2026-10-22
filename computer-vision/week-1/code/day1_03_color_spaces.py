"""
DAY 1 - Script 3: Colour spaces (BGR/RGB, Grayscale, HSV) + colour detection
=============================================================================
Lecture topic: "Color Spaces" - RGB vs HSV, and why converting helps colour detection.

Run:  python day1_03_color_spaces.py
"""

import cv2
import numpy as np
from cv_helpers import load_image, show

img = load_image("sample.jpg")

# ---------------------------------------------------------------
# 1. Split the 3 colour channels. Each channel is a grayscale image:
#    bright = "lots of this colour here", dark = "little of this colour".
# ---------------------------------------------------------------
b, g, r = cv2.split(img)
show([img, b, g, r], ["Original", "Blue channel", "Green channel", "Red channel"],
     filename="day1_03a_channels.png", cols=4)

# ---------------------------------------------------------------
# 2. Grayscale: 3 numbers -> 1 number per pixel.
#    OpenCV uses:  Gray = 0.299*R + 0.587*G + 0.114*B
#    (green counts most because human eyes are most sensitive to green)
# ---------------------------------------------------------------
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print("Colour shape:", img.shape, " -> Gray shape:", gray.shape)

# ---------------------------------------------------------------
# 3. HSV = Hue, Saturation, Value
#    Hue        = WHICH colour (red, yellow, green...). In OpenCV: 0-179
#    Saturation = how PURE/strong the colour is (0 = grey, 255 = vivid)
#    Value      = how BRIGHT (0 = black, 255 = bright)
#
# Note: the lecture converts to HSV and immediately back to RGB to display it,
# so the picture looks identical to the original. To actually SEE HSV, look
# at its three channels separately:
# ---------------------------------------------------------------
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)
show([gray, h, s, v], ["Grayscale", "H (hue = which colour)",
                       "S (saturation = how vivid)", "V (value = brightness)"],
     filename="day1_03b_gray_and_hsv.png", cols=4)

# ---------------------------------------------------------------
# 4. WHY HSV? Colour detection becomes easy.
#    "Find red things" in BGR is hard because shadows change all 3 numbers.
#    In HSV, red is just "Hue near 0 (or near 179)", whatever the brightness.
#
#    cv2.inRange(image, lower, upper) -> white (255) where pixel is inside
#    the range, black (0) elsewhere. That black/white image is called a MASK.
# ---------------------------------------------------------------
# Approximate OpenCV hue values:  red ~0/179, orange ~10-20, yellow ~25-35,
#                                 green ~40-80, blue ~100-130, purple ~140-160
lower_green = np.array([40, 80, 50])     # [H, S, V]
upper_green = np.array([80, 255, 255])
mask = cv2.inRange(hsv, lower_green, upper_green)

# Red wraps around the hue circle, so it needs TWO ranges joined together.
mask_red = cv2.inRange(hsv, (0, 120, 70), (10, 255, 255)) | \
           cv2.inRange(hsv, (170, 120, 70), (179, 255, 255))

only_green = cv2.bitwise_and(img, img, mask=mask)   # keep pixels where mask is white
only_red = cv2.bitwise_and(img, img, mask=mask_red)
show([img, mask, only_green, only_red],
     ["Original", "Green mask (white = green)", "Only green kept", "Only red kept"],
     filename="day1_03c_color_detection.png", cols=4)

# Find the HSV value of any BGR colour - very handy:
pure_blue = np.uint8([[[255, 0, 0]]])
print("HSV of pure blue:", cv2.cvtColor(pure_blue, cv2.COLOR_BGR2HSV)[0][0])   # [120 255 255]

# ---------------------------------------------------------------
# TRY THIS:
#   a) Detect BLUE: use hue range about 100-130.
#   b) Lower the saturation minimum to 0. Why does grey background sneak in?
#   c) Print hsv[50, 50] and compare with img[50, 50].
# ---------------------------------------------------------------

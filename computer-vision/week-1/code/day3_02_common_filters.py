"""
DAY 3 - Script 2: Common filters - Blur, Sharpen, Sobel, Laplacian
===================================================================
Lecture topics: "Why Filtering Matters" + "Common Filters"
  Blur (average, Gaussian) | Sharpen | Edge detection (Sobel, Laplacian)

Run:  python day3_02_common_filters.py
"""

import cv2
import numpy as np
from cv_helpers import load_image, show

img = load_image("sample.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Add some "camera noise" so you can see what blurring removes
rng = np.random.default_rng(1)
noisy = np.clip(img.astype(np.int16) + rng.normal(0, 25, img.shape), 0, 255).astype(np.uint8)

# =================================================================
# 1. BLURS - replace each pixel by an average of its neighbours.
#    Uses: remove noise, smooth before edge detection, privacy blur.
#    Kernel size must be ODD (3, 5, 7...) so it has a centre pixel.
# =================================================================
avg_blur = cv2.blur(noisy, (5, 5))                 # every neighbour counts the same
gauss_blur = cv2.GaussianBlur(noisy, (5, 5), 0)    # nearer neighbours count MORE (bell curve)
                                                   # 0 = let OpenCV choose sigma from the size
median = cv2.medianBlur(noisy, 5)                  # bonus: middle value - great for salt & pepper
big_blur = cv2.GaussianBlur(img, (25, 25), 0)      # bigger kernel = blurrier

show([noisy, avg_blur, gauss_blur, median, big_blur],
     ["Noisy input", "Average blur 5x5", "Gaussian blur 5x5", "Median blur 5", "Gaussian 25x25 (very blurry)"],
     filename="day3_02a_blurs.png", cols=3)

# =================================================================
# 2. SHARPEN - the lecture's kernel:
#         [[ 0, -1,  0],
#          [-1,  5, -1],
#          [ 0, -1,  0]]
#    Meaning: "take 5x this pixel, minus its 4 neighbours".
#    In flat areas: 5x - 4x = x  (unchanged).
#    At an edge the difference gets exaggerated -> crisper.
#    The numbers add up to 1, so overall brightness stays the same.
# =================================================================
sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
sharp = cv2.filter2D(img, -1, sharpen_kernel)
blurry = cv2.GaussianBlur(img, (7, 7), 0)
sharp_from_blurry = cv2.filter2D(blurry, -1, sharpen_kernel)

show([blurry, sharp_from_blurry], ["Blurry", "Blurry + sharpen kernel"],
     filename="day3_02b_sharpen.png")

# =================================================================
# 3. SOBEL - finds edges by measuring how fast brightness CHANGES.
#    cv2.Sobel(src, ddepth, dx, dy, ksize)
#       dx=1, dy=0 -> change left-to-right -> finds VERTICAL edges
#       dx=0, dy=1 -> change top-to-bottom -> finds HORIZONTAL edges
#    ddepth = cv2.CV_64F because the answer can be NEGATIVE
#    (dark->bright = positive, bright->dark = negative). uint8 can't hold
#    negatives and would lose half the edges.
#
#    Why the lecture printed "Clipping input data to the valid range...":
#    it showed a float colour Sobel result directly. Fix: use a grayscale
#    image, take the absolute value, convert back to 0-255 with convertScaleAbs.
# =================================================================
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)          # combine both directions

sobel_x_show = cv2.convertScaleAbs(sobel_x)                # abs + clip to 0-255 + uint8
sobel_y_show = cv2.convertScaleAbs(sobel_y)
mag_show = cv2.convertScaleAbs(magnitude / magnitude.max() * 255)

# =================================================================
# 4. LAPLACIAN - looks at the "second change" (change of the change).
#    Finds edges in ALL directions at once with one kernel, e.g.
#          [[0,  1, 0],
#           [1, -4, 1],
#           [0,  1, 0]]
#    Very sensitive to noise -> usually blur first.
# =================================================================
lap = cv2.Laplacian(cv2.GaussianBlur(gray, (3, 3), 0), cv2.CV_64F, ksize=3)
lap_show = cv2.convertScaleAbs(lap)

show([gray, sobel_x_show, sobel_y_show, mag_show, lap_show],
     ["Grayscale", "Sobel X (vertical edges)", "Sobel Y (horizontal edges)",
      "Sobel magnitude (all edges)", "Laplacian"],
     filename="day3_02c_edges_sobel_laplacian.png", cols=3)

# ---------------------------------------------------------------
# TRY THIS:
#   a) Change (5, 5) to (15, 15) in GaussianBlur. What happens?
#   b) Try an EVEN size like (4, 4) - read the error message.
#   c) Make a stronger sharpen kernel: centre 9, all 8 neighbours -1. Sum = ?
#   d) Run Sobel with cv2.CV_8U instead of CV_64F and compare - edges go missing!
# ---------------------------------------------------------------

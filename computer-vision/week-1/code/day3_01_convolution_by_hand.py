"""
DAY 3 - Script 1: Convolution BY HAND (the most important idea of the week)
============================================================================
Lecture topic: "Convolution Basics"
  - Kernel = small matrix (e.g. 3x3) applied to an image
  - Sliding window computes a weighted sum
  - Produces a feature map highlighting patterns

We write convolution ourselves with plain loops, then check that OpenCV's
cv2.filter2D gives the same answer. Once you get this, blur/sharpen/edges/CNNs
are all "just a different kernel".

Run:  python day3_01_convolution_by_hand.py
"""

import numpy as np
import cv2
from cv_helpers import show


def convolve_by_hand(image, kernel, verbose=False):
    """
    Slide the kernel over every position. At each position:
      1. multiply each kernel number by the pixel under it
      2. add all those products up  -> one output number
    No padding here, so the output is smaller than the input:
      output size = input size - kernel size + 1
    """
    kh, kw = kernel.shape
    ih, iw = image.shape
    out_h, out_w = ih - kh + 1, iw - kw + 1
    output = np.zeros((out_h, out_w), dtype=np.float64)

    for y in range(out_h):
        for x in range(out_w):
            window = image[y:y + kh, x:x + kw]          # the patch under the kernel
            output[y, x] = np.sum(window * kernel)      # weighted sum
            if verbose and (y, x) == (0, 0):
                print("First window:\n", window)
                print("x kernel:\n", kernel)
                print("element-wise products:\n", window * kernel)
                print("sum =", output[y, x], "\n")
    return output


# ---------------------------------------------------------------
# 1. A tiny 6x6 image: dark on the left, bright on the right.
#    There is a vertical EDGE in the middle.
# ---------------------------------------------------------------
image = np.array([
    [10, 10, 10, 200, 200, 200],
    [10, 10, 10, 200, 200, 200],
    [10, 10, 10, 200, 200, 200],
    [10, 10, 10, 200, 200, 200],
    [10, 10, 10, 200, 200, 200],
    [10, 10, 10, 200, 200, 200],
], dtype=np.float64)

# ---------------------------------------------------------------
# 2. Some kernels. Look at what each one's numbers ADD UP TO:
#      sum = 1 -> overall brightness kept     (blur, sharpen)
#      sum = 0 -> flat areas become 0         (edge detectors)
# ---------------------------------------------------------------
kernels = {
    "box blur (average)": np.ones((3, 3)) / 9,
    "sharpen (lecture)": np.array([[0, -1, 0],
                                   [-1, 5, -1],
                                   [0, -1, 0]], dtype=np.float64),
    "vertical edge (Sobel x)": np.array([[-1, 0, 1],
                                         [-2, 0, 2],
                                         [-1, 0, 1]], dtype=np.float64),
    "identity (does nothing)": np.array([[0, 0, 0],
                                         [0, 1, 0],
                                         [0, 0, 0]], dtype=np.float64),
}

print("INPUT IMAGE:\n", image.astype(int), "\n")
print("Step-by-step for the very first position, using the Sobel-x kernel:")
convolve_by_hand(image, kernels["vertical edge (Sobel x)"], verbose=True)

for name, k in kernels.items():
    out = convolve_by_hand(image, k)
    print(f"--- {name}  (kernel sum = {k.sum():.2f}) ---")
    print(np.round(out).astype(int), "\n")

# What to notice:
#   * box blur  -> the sharp 10|200 jump becomes a softer 10, 73, 137, 200 ramp
#   * sharpen   -> dark side gets darker (-180), bright side brighter (390): more contrast at the edge.
#                  Real images are uint8, so these get clipped to 0 and 255.
#   * Sobel x   -> 0 in flat areas, BIG numbers exactly where the edge is
#   * identity  -> just a copy (cropped)

# ---------------------------------------------------------------
# 3. Check against OpenCV.
#    cv2.filter2D(src, ddepth, kernel)
#       ddepth = -1 means "output has the same data type as the input"
#    OpenCV pads the borders, so its output is the SAME size as the input.
#    The middle part matches our hand-made result exactly.
# ---------------------------------------------------------------
k = kernels["vertical edge (Sobel x)"]
ours = convolve_by_hand(image, k)
opencv = cv2.filter2D(image, -1, k)
print("OpenCV filter2D output (same size, borders padded):\n", np.round(opencv).astype(int))
print("Middle part matches ours?", np.allclose(ours, opencv[1:-1, 1:-1]))

# Exam note: strictly, filter2D does "correlation" (it does NOT flip the kernel).
# True convolution flips the kernel 180 degrees first. For symmetric kernels
# (blur, sharpen) there's no difference. CNNs also use correlation but call it convolution.

# ---------------------------------------------------------------
# 4. Same idea on a real-ish picture
# ---------------------------------------------------------------
pic = np.zeros((200, 200), np.uint8)
cv2.rectangle(pic, (50, 50), (150, 150), 255, -1)
pic_f = pic.astype(np.float64)
show([pic,
      np.clip(convolve_by_hand(pic_f, np.ones((9, 9)) / 81), 0, 255).astype(np.uint8),
      np.abs(convolve_by_hand(pic_f, kernels["vertical edge (Sobel x)"])).clip(0, 255).astype(np.uint8)],
     ["Input", "9x9 box blur (hand-made)", "Sobel x (hand-made): vertical edges only"],
     filename="day3_01_convolution_by_hand.png")

# ---------------------------------------------------------------
# OUTPUT SIZE FORMULA (comes up in CNN questions):
#     output = (N - K + 2P) / S + 1
#     N = input size, K = kernel size, P = padding, S = stride (step size)
#   e.g. N=6, K=3, P=0, S=1  ->  (6-3+0)/1 + 1 = 4   (matches our 4x4 output!)
#
# TRY THIS:
#   a) Use a horizontal-edge kernel: [[-1,-2,-1],[0,0,0],[1,2,1]] on `image`. Why is it all 0?
#   b) Make a kernel that brightens: identity * 2.  What is its sum?
#   c) Make the 6x6 image have a horizontal edge instead and rerun.
# ---------------------------------------------------------------

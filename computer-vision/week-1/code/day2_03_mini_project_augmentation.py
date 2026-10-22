"""
DAY 2 - MINI PROJECT: Image Augmentation Tool
==============================================
Lecture task: a tool that applies Translate / Scale / Rotate / Perspective.

What is "augmentation"?  When training an AI model you rarely have enough
photos. So you make extra versions of each photo (shifted, rotated, zoomed...)
- it's still the same cat, just seen differently. More variety -> the model
learns better and isn't fooled by small changes.

Run:  python day2_03_mini_project_augmentation.py
"""

import cv2
import numpy as np
from cv_helpers import load_image, show


def transform_image(img, mode="original"):
    """The lecture's function - but it takes an image (not a path) and works on any size."""
    rows, cols = img.shape[:2]

    if mode == "translate":
        M = np.float32([[1, 0, 100], [0, 1, 50]])
        img = cv2.warpAffine(img, M, (cols, rows))
    elif mode == "scale":
        img = cv2.resize(img, None, fx=0.5, fy=0.5)
    elif mode == "rotate":
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1)
        img = cv2.warpAffine(img, M, (cols, rows))
    elif mode == "perspective":
        # Points chosen RELATIVE to image size, so they land on the image's content
        # (the lecture's fixed points only covered grey background - see day2_02).
        pts1 = np.float32([[cols * 0.1, rows * 0.1], [cols * 0.9, rows * 0.05],
                           [cols * 0.05, rows * 0.95], [cols * 0.95, rows * 0.9]])
        pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        img = cv2.warpPerspective(img, M, (300, 300))
    return img


def random_augment(img, rng):
    """Bonus: random augmentation, like real deep-learning pipelines do."""
    rows, cols = img.shape[:2]
    angle = rng.uniform(-25, 25)               # random small rotation
    scale = rng.uniform(0.8, 1.2)              # random zoom in/out
    tx, ty = rng.integers(-40, 41, size=2)     # random shift
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, scale)
    M[0, 2] += tx                              # add the shift into the same matrix
    M[1, 2] += ty
    out = cv2.warpAffine(img, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)
    if rng.random() < 0.5:
        out = cv2.flip(out, 1)                 # 50% chance of mirroring
    return out, f"rot {angle:.0f}, zoom {scale:.2f}, shift ({tx},{ty})"


img = load_image("sample.jpg")

modes = ["original", "translate", "scale", "rotate", "perspective"]
show([transform_image(img, m) for m in modes], [f"Mode: {m}" for m in modes],
     filename="day2_03a_augmentation_modes.png", cols=3)

rng = np.random.default_rng(seed=0)   # change the seed for different random results
results = [random_augment(img, rng) for _ in range(6)]
show([r[0] for r in results], [r[1] for r in results],
     filename="day2_03b_random_augmentations.png", cols=3)

# ---------------------------------------------------------------
# TRY THIS:
#   a) Add a mode "flip" using cv2.flip(img, 1).
#   b) Let transform_image take extra arguments, e.g. angle=45, tx=100.
#   c) Save each augmented image:  cv2.imwrite(f"aug_{i}.jpg", out)
# ---------------------------------------------------------------

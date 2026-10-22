"""
cv_helpers.py  -  small helper functions used by every script in this folder.

You do NOT need to understand this file to learn Week 1.
It just does two boring jobs for you:

  1. load_image()  -> gives you an image to play with.
                      If you put your own picture called "sample.jpg" in this
                      'code' folder, it uses that. Otherwise it creates a
                      colourful practice image automatically.

  2. show()        -> displays one or more images side by side with titles,
                      AND saves the figure into the 'outputs' folder so you
                      can look at it later (handy if no window pops up).
"""

import os
import cv2
import numpy as np
import matplotlib

# If there is no screen (e.g. running on a server), draw to files instead of windows.
if not os.environ.get("DISPLAY") and os.name != "nt" and "MPLBACKEND" not in os.environ:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(HERE, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def make_practice_image(width=600, height=400):
    """Create a colourful test image (BGR, like OpenCV) with clear shapes and colours."""
    img = np.full((height, width, 3), 230, dtype=np.uint8)          # light grey background

    # NOTE: OpenCV colours are written as (Blue, Green, Red) - not RGB!
    cv2.rectangle(img, (40, 40), (200, 180), (0, 0, 255), -1)        # red square
    cv2.circle(img, (320, 110), 70, (0, 200, 0), -1)                 # green circle
    pts = np.array([[480, 40], [560, 180], [400, 180]], np.int32)
    cv2.fillPoly(img, [pts], (255, 0, 0))                           # blue triangle
    cv2.rectangle(img, (60, 240), (260, 360), (0, 220, 255), -1)     # yellow rectangle
    cv2.ellipse(img, (440, 300), (110, 55), 0, 0, 360, (200, 0, 200), -1)  # purple ellipse
    cv2.putText(img, "CV Week 1", (300, 390), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (40, 40, 40), 2)
    return img


def load_image(name="sample.jpg", max_side=800):
    """
    Load an image the same way the lectures do: cv2.imread(...).
    Falls back to the practice image if the file is missing.

    max_side: very large photos (like the 5472 x 3648 one in the lecture) are
    shrunk so they process quickly and thin edges stay visible.
    """
    path = name if os.path.isabs(name) else os.path.join(HERE, name)
    img = cv2.imread(path) if os.path.exists(path) else None   # imread returns None if it can't read
    if img is None:
        print(f"[info] '{name}' not found -> using the built-in practice image.")
        print("       (Put your own photo named sample.jpg in the code folder to use it.)")
        return make_practice_image()

    h, w = img.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        print(f"[info] Image was {w}x{h}; shrunk to {img.shape[1]}x{img.shape[0]} for speed.")
    return img


def show(images, titles=None, filename=None, cols=None, figsize_per=4):
    """
    Show several images in a row/grid.

    images : list of images. 3-channel images are assumed to be BGR (OpenCV style)
             and are converted to RGB for display. 2-D images are shown in grey.
    titles : list of titles (same length as images).
    filename : name for the saved PNG in outputs/ (optional).
    """
    if not isinstance(images, (list, tuple)):
        images = [images]
        titles = [titles] if titles else None
    n = len(images)
    cols = cols or min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(figsize_per * cols, figsize_per * rows * 0.8))
    axes = np.array(axes).reshape(-1)

    for i, ax in enumerate(axes):
        ax.axis("off")
        if i >= n:
            continue
        im = images[i]
        if im.ndim == 3:
            ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        else:
            ax.imshow(im, cmap="gray")
        if titles:
            ax.set_title(titles[i], fontsize=11)

    plt.tight_layout()
    if filename:
        out = os.path.join(OUTPUT_DIR, filename)
        plt.savefig(out, dpi=110, bbox_inches="tight")
        print(f"[saved] {out}")
    if matplotlib.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)

"""
make_diagrams.py - draws the explanation pictures used in the notes (saved to ../images/).
You don't need to study this file; run it if you want to regenerate the diagrams.
"""

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from cv_helpers import make_practice_image, make_shapes_image

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "images")
os.makedirs(OUT, exist_ok=True)
INK, MUTED, ACCENT, ACCENT2 = "#1f2328", "#6e7781", "#d1242f", "#0969da"


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("[saved]", os.path.normpath(path))


def draw_grid(ax, arr, x0=0, y0=0, cmap="gray", vmin=0, vmax=255, fs=10, highlight=None, color=ACCENT):
    """Draw a matrix as coloured cells with the number written inside."""
    h, w = arr.shape
    for r in range(h):
        for c in range(w):
            v = arr[r, c]
            shade = plt.get_cmap(cmap)((v - vmin) / (vmax - vmin)) if cmap else "white"
            ax.add_patch(Rectangle((x0 + c, y0 + r), 1, 1, facecolor=shade, edgecolor=MUTED, lw=0.6))
            txt_col = "white" if cmap and (v - vmin) / (vmax - vmin) < 0.5 else INK
            label = f"{v:g}" if isinstance(v, (float, np.floating)) else str(v)
            ax.text(x0 + c + 0.5, y0 + r + 0.5, label, ha="center", va="center", fontsize=fs, color=txt_col)
    if highlight:
        (r0, c0, hh, ww) = highlight
        ax.add_patch(Rectangle((x0 + c0, y0 + r0), ww, hh, fill=False, edgecolor=color, lw=3))


# ---------------------------------------------------------------- 1. image = grid of numbers
def diagram_pixels():
    img = make_practice_image()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    y, x = 176, 196                     # a patch on the bottom-right corner of the red square
    patch = gray[y:y + 8, x:x + 8]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.3, 1]})
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].add_patch(Rectangle((x - 6, y - 6), 20, 20, fill=False, edgecolor=ACCENT2, lw=2.5))
    axes[0].set_title("What you see: a picture", color=INK)
    axes[0].axis("off")

    ax = axes[1]
    draw_grid(ax, patch, fs=9)
    ax.set_xlim(0, 8); ax.set_ylim(8, 0); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("What the computer sees: numbers (0 = black, 255 = white)\n"
                 "8x8 zoom of the blue box (corner of the red square), in grayscale", color=INK, fontsize=10)
    save(fig, "01_image_is_numbers.png")


# ---------------------------------------------------------------- 2. coordinates
def diagram_coordinates():
    fig, ax = plt.subplots(figsize=(7, 5))
    W, H = 8, 5
    for r in range(H):
        for c in range(W):
            ax.add_patch(Rectangle((c, r), 1, 1, facecolor="#f6f8fa", edgecolor=MUTED, lw=0.6))
    ax.add_patch(Rectangle((5, 2), 1, 1, facecolor="#ffd8d3", edgecolor=ACCENT, lw=2))
    ax.text(5.5, 2.5, "img\n[2, 5]", ha="center", va="center", fontsize=8, color=ACCENT)
    ax.add_patch(FancyArrowPatch((0, -0.4), (W + 0.6, -0.4), arrowstyle="-|>", mutation_scale=15, color=ACCENT2))
    ax.add_patch(FancyArrowPatch((-0.4, 0), (-0.4, H + 0.6), arrowstyle="-|>", mutation_scale=15, color=ACCENT2))
    ax.text(W / 2, -0.8, "x  (columns, width)  ->", ha="center", color=ACCENT2)
    ax.text(-0.9, H / 2, "y  (rows, height)  ->", rotation=-90, va="center", color=ACCENT2)
    ax.text(0.05, 0.5, "(0,0)", fontsize=8, va="center", color=INK)
    ax.text(0, H + 1.4, "Origin (0,0) is the TOP-LEFT corner. y grows DOWNWARDS.\n"
            "Indexing is img[row, col] = img[y, x]   ->  this pixel is y=2, x=5.\n"
            "img.shape = (height, width, channels) = (5, 8, 3)", fontsize=10, color=INK, va="top")
    ax.set_xlim(-1.3, W + 1); ax.set_ylim(H + 2.6, -1.2); ax.set_aspect("equal"); ax.axis("off")
    save(fig, "02_coordinates.png")


# ---------------------------------------------------------------- 3. BGR channels + HSV hue bar
def diagram_color_spaces():
    fig = plt.figure(figsize=(12, 5.2))
    # RGB cube faces as simple swatches
    ax = fig.add_axes([0.02, 0.42, 0.96, 0.5]); ax.axis("off")
    swatches = [((255, 0, 0), "Red\nRGB(255,0,0)\nBGR(0,0,255)"), ((0, 255, 0), "Green\nRGB(0,255,0)\nBGR(0,255,0)"),
                ((0, 0, 255), "Blue\nRGB(0,0,255)\nBGR(255,0,0)"), ((255, 255, 0), "Yellow\nRGB(255,255,0)\n= red + green"),
                ((255, 255, 255), "White\nRGB(255,255,255)\nall full"), ((0, 0, 0), "Black\nRGB(0,0,0)\nall zero"),
                ((128, 128, 128), "Grey\nRGB(128,128,128)\nall equal")]
    for i, (rgb, label) in enumerate(swatches):
        ax.add_patch(Rectangle((i * 1.4, 0.9), 1.1, 1.1, facecolor=np.array(rgb) / 255, edgecolor=MUTED))
        ax.text(i * 1.4 + 0.55, 0.75, label, ha="center", va="top", fontsize=9, color=INK)
    ax.set_xlim(-0.2, 7 * 1.4); ax.set_ylim(-0.4, 2.3)
    ax.set_title("RGB: every colour = a mix of Red, Green, Blue (0-255 each).  OpenCV stores them backwards: B, G, R",
                 fontsize=11, color=INK)

    # Hue bar (OpenCV 0..179)
    ax2 = fig.add_axes([0.06, 0.1, 0.88, 0.2])
    hue = np.tile(np.arange(180, dtype=np.uint8), (20, 1))
    hsv = cv2.merge([hue, np.full_like(hue, 255), np.full_like(hue, 255)])
    ax2.imshow(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), aspect="auto", extent=[0, 180, 0, 1])
    ax2.set_yticks([])
    ax2.set_xticks([0, 15, 30, 60, 90, 120, 150, 179])
    ax2.set_xticklabels(["0\nred", "15\norange", "30\nyellow", "60\ngreen", "90\ncyan", "120\nblue",
                         "150\nmagenta", "179\nred again"], fontsize=9)
    ax2.set_title("HSV: H = Hue (WHICH colour, OpenCV range 0-179)  ·  S = Saturation (how vivid, 0-255)  ·  "
                  "V = Value (how bright, 0-255)", fontsize=11, color=INK)
    save(fig, "03_color_spaces.png")


# ---------------------------------------------------------------- 4. convolution step-by-step
def diagram_convolution():
    image = np.array([[10, 10, 10, 200, 200],
                      [10, 10, 10, 200, 200],
                      [10, 10, 10, 200, 200],
                      [10, 10, 10, 200, 200],
                      [10, 10, 10, 200, 200]])
    kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    out = np.zeros((3, 3), int)
    for y in range(3):
        for x in range(3):
            out[y, x] = int(np.sum(image[y:y + 3, x:x + 3] * kernel))

    fig, ax = plt.subplots(figsize=(13, 5.2))
    draw_grid(ax, image, x0=0, y0=0, highlight=(0, 1, 3, 3))
    ax.text(2.5, -0.5, "Input image (5x5)", ha="center", color=INK)
    ax.text(5.6, 2.5, "x", fontsize=20, ha="center", va="center", color=INK)
    draw_grid(ax, kernel, x0=6.2, y0=1, cmap="coolwarm", vmin=-2, vmax=2)
    ax.text(7.7, 0.5, "Kernel 3x3 (Sobel-x)", ha="center", color=INK)
    ax.text(10.1, 2.5, "=", fontsize=20, ha="center", va="center", color=INK)
    draw_grid(ax, out, x0=10.8, y0=1, cmap="gray", vmin=0, vmax=760, highlight=(0, 1, 1, 1))
    ax.text(12.3, 0.5, "Output / feature map (3x3)", ha="center", color=INK)

    win = image[0:3, 1:4]
    terms = " + ".join(f"({k}x{v})" for k, v in zip(kernel.flatten(), win.flatten()) if k != 0)
    ax.text(0, 5.9, "Red window x kernel, then ADD everything:   " + terms + f"  =  {out[0, 1]}",
            fontsize=10, color=ACCENT)
    ax.text(0, 6.6, "Then slide the window one step right and repeat... every output number = one window position.\n"
            "Flat area (all 10s or all 200s) -> 0.  Window over the 10|200 jump -> big number = EDGE FOUND.",
            fontsize=10, color=INK, va="top")
    ax.set_xlim(-0.3, 14.5); ax.set_ylim(7.8, -1); ax.set_aspect("equal"); ax.axis("off")
    save(fig, "04_convolution_step.png")


# ---------------------------------------------------------------- 5. kernels gallery
def diagram_kernels():
    img = cv2.cvtColor(make_practice_image(), cv2.COLOR_BGR2GRAY)
    kernels = [
        ("Identity\n(sum = 1)", np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], float)),
        ("Box blur\n(sum = 1)", np.ones((3, 3)) / 9),
        ("Sharpen\n(sum = 1)", np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], float)),
        ("Sobel X\n(sum = 0)", np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float)),
        ("Sobel Y\n(sum = 0)", np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], float)),
        ("Laplacian\n(sum = 0)", np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], float)),
    ]
    fig, axes = plt.subplots(2, len(kernels), figsize=(16, 5.2), gridspec_kw={"height_ratios": [1, 1]})
    for i, (name, k) in enumerate(kernels):
        ax = axes[0, i]
        draw_grid(ax, np.round(k, 2), cmap=None, fs=11)
        ax.set_xlim(-0.05, 3.05); ax.set_ylim(3.05, -0.05); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(name, fontsize=11, color=INK)
        src = cv2.blur(img, (9, 9)) if "Sharpen" in name else img   # sharpen a blurry copy to see the effect
        res = cv2.filter2D(src.astype(np.float64), -1, k)
        res = np.clip(np.abs(res) if k.sum() == 0 else res, 0, 255).astype(np.uint8)
        axes[1, i].imshow(res, cmap="gray", vmin=0, vmax=255)
        axes[1, i].axis("off")
    fig.suptitle("Same sliding-window maths, different numbers in the kernel -> different effect", color=INK)
    save(fig, "05_kernels_gallery.png")


# ---------------------------------------------------------------- 6. Canny hysteresis
def diagram_canny_thresholds():
    x = np.arange(60)
    rng = np.random.default_rng(3)
    strength = 30 + rng.normal(0, 12, 60).clip(-25, 25)
    strength[10:14] = [120, 230, 240, 130]      # strong edge
    strength[25:33] = [110, 140, 150, 160, 150, 140, 260, 150]   # weak connected to strong
    strength[45:48] = [140, 160, 130]           # weak, isolated
    low, high = 100, 200

    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(x, strength, color=INK, lw=2)
    ax.axhline(high, color=ACCENT, lw=1.5, ls="--")
    ax.axhline(low, color=ACCENT2, lw=1.5, ls="--")
    ax.text(60.5, high, "high = 200", color=ACCENT, va="center")
    ax.text(60.5, low, "low = 100", color=ACCENT2, va="center")
    ax.annotate("above high -> STRONG edge (kept)", (11.5, 240), (2, 285), color=INK, fontsize=9,
                arrowprops=dict(arrowstyle="->", color=MUTED))
    ax.annotate("between, but CONNECTED to a strong\npart -> kept (weak edge)", (28, 160), (17, 205),
                color=INK, fontsize=9, arrowprops=dict(arrowstyle="->", color=MUTED))
    ax.annotate("between, NOT connected\n-> thrown away", (46, 160), (40, 225), color=INK, fontsize=9,
                arrowprops=dict(arrowstyle="->", color=MUTED))
    ax.annotate("below low -> never an edge (noise)", (55, 35), (44, 70), color=INK, fontsize=9,
                arrowprops=dict(arrowstyle="->", color=MUTED))
    ax.set_ylim(0, 310); ax.set_xlim(0, 60)
    ax.set_xlabel("position along a line of pixels", color=MUTED)
    ax.set_ylabel("edge strength (gradient)", color=MUTED)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#eaeef2")
    ax.set_title("Canny hysteresis: cv2.Canny(img, 100, 200)", color=INK)
    save(fig, "06_canny_hysteresis.png")


# ---------------------------------------------------------------- 7. transformations overview
def diagram_transforms():
    img = make_practice_image()
    rows, cols = img.shape[:2]
    grid = img.copy()
    for xx in range(0, cols, 50):
        cv2.line(grid, (xx, 0), (xx, rows), (90, 90, 90), 1)
    for yy in range(0, rows, 50):
        cv2.line(grid, (0, yy), (cols, yy), (90, 90, 90), 1)
    canvas = (cols, rows)
    res = [
        ("Original", grid, ""),
        ("Translation", cv2.warpAffine(grid, np.float32([[1, 0, 100], [0, 1, 50]]), canvas),
         "slide: M = [[1,0,tx],[0,1,ty]]"),
        ("Scaling", cv2.copyMakeBorder(cv2.resize(grid, None, fx=0.5, fy=0.5), 0, rows // 2, 0, cols // 2,
                                       cv2.BORDER_CONSTANT), "resize: fx, fy"),
        ("Rotation", cv2.warpAffine(grid, cv2.getRotationMatrix2D((cols / 2, rows / 2), 30, 0.8), canvas),
         "turn around a pivot"),
        ("Affine", cv2.warpAffine(grid, cv2.getAffineTransform(np.float32([[0, 0], [600, 0], [0, 400]]),
                                                               np.float32([[80, 40], [560, 0], [0, 360]])), canvas),
         "3 points; parallel stays parallel"),
        ("Perspective", cv2.warpPerspective(grid, cv2.getPerspectiveTransform(
            np.float32([[0, 0], [600, 0], [0, 400], [600, 400]]),
            np.float32([[150, 40], [450, 40], [0, 400], [600, 400]])), canvas),
         "4 points; parallel lines can meet"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, (t, im, sub) in zip(axes.flat, res):
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); ax.axis("off")
        ax.set_title(f"{t}\n{sub}", fontsize=11, color=INK)
    save(fig, "07_transformations_overview.png")


# ---------------------------------------------------------------- 8. histogram from a tiny image
def diagram_histogram():
    tiny = np.array([[0, 0, 64, 64],
                     [0, 128, 128, 64],
                     [255, 255, 128, 64],
                     [255, 255, 255, 128]])
    values, counts = np.unique(tiny, return_counts=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.4]})
    draw_grid(axes[0], tiny, fs=12)
    axes[0].set_xlim(0, 4); axes[0].set_ylim(4, 0); axes[0].set_aspect("equal"); axes[0].axis("off")
    axes[0].set_title("4x4 grayscale image (16 pixels)", color=INK)
    ax = axes[1]
    ax.bar([str(v) for v in values], counts, color=ACCENT2, width=0.6)
    for i, c in enumerate(counts):
        ax.text(i, c + 0.1, str(c), ha="center", color=INK)
    ax.set_xlabel("pixel value (intensity)", color=MUTED); ax.set_ylabel("how many pixels", color=MUTED)
    ax.set_ylim(0, max(counts) + 1)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Its histogram: just COUNT each value (counts add up to 16)", color=INK)
    save(fig, "08_histogram_explained.png")


# ---------------------------------------------------------------- 9. Otsu
def diagram_otsu():
    rng = np.random.default_rng(2)
    pixels = np.concatenate([rng.normal(60, 15, 6000), rng.normal(180, 20, 4000)]).clip(0, 255).astype(np.uint8)
    hist = np.bincount(pixels, minlength=256).astype(float)
    p = hist / hist.sum()
    within = np.full(256, np.nan)
    for t in range(1, 255):
        w0, w1 = p[:t].sum(), p[t:].sum()
        if w0 == 0 or w1 == 0:
            continue
        m0 = (np.arange(t) * p[:t]).sum() / w0
        m1 = (np.arange(t, 256) * p[t:]).sum() / w1
        v0 = (((np.arange(t) - m0) ** 2) * p[:t]).sum() / w0
        v1 = (((np.arange(t, 256) - m1) ** 2) * p[t:]).sum() / w1
        within[t] = w0 * v0 + w1 * v1
    best = int(np.nanargmin(within))
    T_cv, _ = cv2.threshold(pixels.reshape(1, -1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].bar(range(256), hist, color=MUTED, width=1)
    axes[0].axvline(best, color=ACCENT, lw=2)
    axes[0].text(best + 3, hist.max() * 0.9, f"Otsu threshold = {best}\n(cv2 says {T_cv:.0f})", color=ACCENT)
    axes[0].text(60, hist.max() * 0.15, "dark group\n(background)", ha="center", color="white", fontsize=9)
    axes[0].text(180, hist.max() * 0.15, "bright group\n(object)", ha="center", color="white", fontsize=9)
    axes[0].set_ylabel("pixel count", color=MUTED)
    axes[0].set_title("A two-hump (bimodal) histogram - the case Otsu loves", color=INK)
    axes[1].plot(range(256), within, color=ACCENT2, lw=2)
    axes[1].axvline(best, color=ACCENT, lw=2)
    axes[1].set_ylabel("within-group\nvariance", color=MUTED)
    axes[1].set_xlabel("candidate threshold T", color=MUTED)
    axes[1].set_title("Otsu tries EVERY T and keeps the one where both groups are most uniform (lowest point)",
                      color=INK, fontsize=10)
    for ax in axes:
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[1].set_xlim(0, 255)
    save(fig, "09_otsu_explained.png")


# ---------------------------------------------------------------- 10. dilation / erosion on a grid
def diagram_morphology_grid():
    N = 10
    img = np.zeros((N, N), np.uint8)
    img[2:8, 2:8] = 255               # a 6x6 white block
    img[4, 5] = 0                     # a 1-px hole inside it
    img[0, 9] = 255                   # a 1-px speck outside it
    k = np.ones((3, 3), np.uint8)
    panels = [("Input (1 = white)", img), ("DILATE 3x3: white if ANY\nneighbour is white (max)", cv2.dilate(img, k)),
              ("ERODE 3x3: white only if ALL\nneighbours are white (min)", cv2.erode(img, k)),
              ("OPEN = erode, then dilate:\nspeck gone (hole grows)", cv2.morphologyEx(img, cv2.MORPH_OPEN, k)),
              ("CLOSE = dilate, then erode:\nhole filled (speck stays)", cv2.morphologyEx(img, cv2.MORPH_CLOSE, k))]
    fig, axes = plt.subplots(1, len(panels), figsize=(18, 4.6))
    for ax, (title, im) in zip(axes, panels):
        draw_grid(ax, (im // 255).astype(int), cmap="gray", vmin=0, vmax=1, fs=10)
        ax.set_xlim(-0.05, N + 0.05); ax.set_ylim(N + 0.05, -0.05); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(title, fontsize=10.5, color=INK)
    axes[0].add_patch(Rectangle((9, 0), 1, 1, fill=False, edgecolor=ACCENT, lw=2.5))
    axes[0].add_patch(Rectangle((5, 4), 1, 1, fill=False, edgecolor=ACCENT2, lw=2.5))
    axes[0].text(0, N + 0.9, "red = 1-px speck,  blue = 1-px hole", fontsize=9, color=MUTED)
    save(fig, "10_morphology_grid.png")


# ---------------------------------------------------------------- 11. contour descriptors
def diagram_contour_features():
    img = make_shapes_image()
    crop = img[230:410, 340:540].copy()
    _, th = cv2.threshold(crop, 127, 255, cv2.THRESH_BINARY_INV)
    cnt = max(cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea)
    vis = cv2.cvtColor((crop // 2 + 120).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    vis = cv2.resize(vis, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    c2 = cnt * 2
    x, y, w, h = cv2.boundingRect(c2)
    cv2.rectangle(vis, (x, y), (x + w, y + h), (218, 105, 9), 2)                 # blue box
    cv2.drawContours(vis, [cv2.convexHull(c2)], 0, (47, 36, 209), 2)             # red hull
    cv2.drawContours(vis, [c2], 0, (55, 127, 26), 3)                             # green contour
    (cx, cy), r = cv2.minEnclosingCircle(c2)
    cv2.circle(vis, (int(cx), int(cy)), int(r), (130, 119, 110), 1)
    M = cv2.moments(c2)
    mx, my = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
    cv2.circle(vis, (mx, my), 6, (0, 0, 0), -1)

    area, hull_area = cv2.contourArea(cnt), cv2.contourArea(cv2.convexHull(cnt))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)); ax.axis("off")
    notes = [("#1a7f37", "contour (green): the outline points  -> findContours / drawContours"),
             (ACCENT2, "bounding box (blue): x, y, w, h        -> boundingRect"),
             (ACCENT, "convex hull (red): rubber band around it -> convexHull"),
             (MUTED, "min enclosing circle (grey)            -> minEnclosingCircle"),
             (INK, "centroid (black dot): centre of mass    -> moments"),
             (INK, f"area {area:.0f} px, perimeter {cv2.arcLength(cnt, True):.0f} px, "
                   f"solidity = area / hull area = {area / hull_area:.2f}")]
    for i, (col, txt) in enumerate(notes):
        ax.text(vis.shape[1] + 15, 40 + i * 48, txt, color=col, fontsize=10, va="center", family="monospace")
    ax.set_xlim(0, vis.shape[1] + 560)
    save(fig, "11_contour_features.png")


if __name__ == "__main__":
    diagram_pixels()
    diagram_coordinates()
    diagram_color_spaces()
    diagram_convolution()
    diagram_kernels()
    diagram_canny_thresholds()
    diagram_transforms()
    diagram_histogram()
    diagram_otsu()
    diagram_morphology_grid()
    diagram_contour_features()

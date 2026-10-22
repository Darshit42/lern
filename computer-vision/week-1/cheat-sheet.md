# CV Week 1 · Cheat Sheet

## Images = numbers
- Image = grid of **pixels**. `uint8` → values **0 (black) – 255 (white)**.
- `img.shape = (height, width, channels)` = `(rows, cols, 3)`. Grayscale: `(rows, cols)`.
- Pixel access: **`img[y, x]`** (row first). Origin (0,0) = **top-left**, y grows **down**.
- OpenCV size arguments are **(width, height)** = `(cols, rows)`.
- `cv2.imread` returns **`None`** if the path is wrong (no error!).
- uint8 overflow: NumPy wraps (250+10 = 4); `cv2.add` saturates (= 255).

## Colour
| | |
|---|---|
| OpenCV order | **BGR**. Matplotlib wants RGB → `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` |
| Gray | `cv2.COLOR_BGR2GRAY`; Y = 0.299R + 0.587G + 0.114B |
| HSV | `cv2.COLOR_BGR2HSV`. **H** = which colour (**0–179**), **S** = vividness, **V** = brightness (0–255) |
| Why HSV | colour is separate from brightness → robust colour detection |
| Mask | `cv2.inRange(hsv, low, high)` → 255 inside the range; `cv2.bitwise_and(img, img, mask=m)` |
| Hues | red 0/179 · orange ~15 · yellow ~30 · green ~60 · blue ~120 |

## Transformations (Day 2)  — `rows, cols = img.shape[:2]`
| Task | Code | Notes |
|---|---|---|
| Translate | `M = np.float32([[1,0,tx],[0,1,ty]])`; `cv2.warpAffine(img, M, (cols, rows))` | +tx right, +ty **down** |
| Scale | `cv2.resize(img, None, fx=.5, fy=.5, interpolation=cv2.INTER_LINEAR)` | shrink: `INTER_AREA`; enlarge: `INTER_CUBIC` |
| Rotate | `M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, scale)` + `warpAffine` | **+angle = anti-clockwise**; corners get cut |
| Affine | `cv2.getAffineTransform(pts1, pts2)` + `warpAffine` | **3 points**, 2×3 matrix, **parallel lines stay parallel** |
| Perspective | `cv2.getPerspectiveTransform(pts1, pts2)` + **`warpPerspective`** | **4 points**, 3×3 matrix, parallel lines can meet → **document scanning** |
| Flip / 90° | `cv2.flip(img, 1)` · `cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)` | |

**Augmentation** = making extra training images by transforming them → more variety, less overfitting.

## Convolution (Day 3)
- **Kernel** = small matrix. **Convolution** = slide it over the image, **multiply element-wise and sum** at each spot. Output = **feature map**. It's the **foundation of CNNs** (which learn their own kernels).
- `cv2.filter2D(img, -1, kernel)` (-1 = same depth as the input). Technically correlation (no kernel flip).
- **Kernel sum = 1** → brightness kept (blur, sharpen). **Sum = 0** → edge detector (flat → 0).
- Output size: **(N − K + 2P)/S + 1**.
- Kernel sizes must be **odd**. Bigger kernel = stronger effect.

| Filter | Code / kernel |
|---|---|
| Average blur | `cv2.blur(img, (5,5))`, all weights equal |
| Gaussian blur | `cv2.GaussianBlur(img, (5,5), 0)`, closer = heavier weight |
| Median | `cv2.medianBlur(img, 5)`, salt & pepper noise |
| Sharpen | `[[0,-1,0],[-1,5,-1],[0,-1,0]]` |
| Sobel | `cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)` → dx=1: **vertical** edges; dy=1: horizontal. `[[-1,0,1],[-2,0,2],[-1,0,1]]` |
| Laplacian | `cv2.Laplacian(gray, cv2.CV_64F)`, 2nd derivative, all directions, noise-sensitive. `[[0,1,0],[1,-4,1],[0,1,0]]` |
| Display float | `cv2.convertScaleAbs(result)` → abs + uint8 |

**`CV_64F`** because gradients can be **negative**; uint8 would lose them.

## Canny — `cv2.Canny(img, low, high)`
1. Gaussian blur (noise) → 2. Sobel gradient (strength + direction) → 3. **Non-max suppression** (1-px thin) → 4. **Hysteresis**:
   **> high** = strong edge · **< low** = discard · **between** = keep **only if connected to a strong edge**.
- Output: 1-channel, values **0/255** only. Lower thresholds → more edges/noise. high ≈ 2–3 × low.

## Contours (object detector)
```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(cv2.GaussianBlur(gray, (5,5), 0), 50, 150)
contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    if cv2.contourArea(cnt) < 500: continue          # skip noise
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)
    cv2.putText(img, "Obj", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
```
`RETR_EXTERNAL` = outer outlines only · `CHAIN_APPROX_SIMPLE` = keep corner points only · `approxPolyDP` corner count → shape name.

## Histograms (Day 4)  — `img = cv2.imread('x.jpg', 0)` loads grayscale
- Histogram = **count of pixels per intensity** (x: 0–255, y: count). Ignores *where* pixels are.
- `plt.hist(img.ravel(), 256, [0, 256])` · `cv2.calcHist([img], [0], None, [256], [0, 256])` (images, channel, mask, bins, range).
- Hump left = **dark**, right = **bright**, narrow = **low contrast**, two humps = **bimodal** (good for thresholding).
- `cv2.equalizeHist(img)` spreads values out → more contrast.

## Thresholding (Day 4)
| | Code | Notes |
|---|---|---|
| Global | `T, out = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)` | > T → 255, else 0 (127 → 0!) · returns (T, image) |
| Adaptive | `cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)` | T per neighbourhood = mean (or Gaussian) − C · blockSize **odd** · returns image only · **uneven lighting** |
| Otsu | `T, out = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)` | picks T automatically · **minimises intra-class variance** · best for bimodal |
- Types: `BINARY`, `BINARY_INV` (dark objects → white), `TRUNC` (cap at T), `TOZERO`, `TOZERO_INV`.
- Blur before thresholding = less speckle. Uses: document binarization, tumour segmentation, defect detection.

## Morphology (Day 5)  — works on **WHITE** = object (invert dark-on-light first)
| Op | Code | Effect |
|---|---|---|
| Kernel | `np.ones((5,5), np.uint8)` · `cv2.getStructuringElement(cv2.MORPH_RECT/ELLIPSE/CROSS, (5,5))` | shape + size of the probe |
| Dilation | `cv2.dilate(img, k, iterations=1)` | white grows (local **max**) · joins broken parts |
| Erosion | `cv2.erode(img, k, iterations=1)` | white shrinks (local **min**) · removes small noise |
| Opening | `cv2.morphologyEx(img, cv2.MORPH_OPEN, k)` | **erode → dilate** · removes white specks |
| Closing | `cv2.morphologyEx(img, cv2.MORPH_CLOSE, k)` | **dilate → erode** · fills small holes |
| Gradient / top-hat / black-hat | `MORPH_GRADIENT` / `MORPH_TOPHAT` / `MORPH_BLACKHAT` | outline / removed specks / filled holes |
- Kernel: **bigger than the noise, smaller than the detail you keep.** Marker = dilate, eraser = erode.

## Contours & shapes (Day 6)
```python
_, th = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)        # dark shapes → WHITE
contours, hierarchy = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img_bgr, contours, -1, (0,255,0), 2)               # -1 = all; thickness -1 = fill
x, y, w, h = cv2.boundingRect(cnt)                                  # upright box (localization)
hull = cv2.convexHull(cnt); cv2.drawContours(img_bgr, [hull], 0, (0,0,255), 2)   # rubber band
approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)            # corners: 3 tri, 4 rect, >6 circle
M = cv2.moments(cnt); cx, cy = M['m10']/M['m00'], M['m01']/M['m00']             # centroid
```
- `contourArea`, `arcLength(cnt, True)` = perimeter · `minAreaRect` (rotated box) · `minEnclosingCircle`.
- Solidity = area / hull area (star ≈ 0.5) · aspect ratio w/h (square ≈ 1) · circularity 4πA/P² (circle ≈ 1).
- `RETR_EXTERNAL` outer only · `RETR_TREE` all + hierarchy `[next, prev, child, parent]` · `CHAIN_APPROX_SIMPLE` = corner points only.
- Uses: object detection/tracking, industrial inspection, cell shape analysis.

## Key concepts (advanced, know what + why)
| Topic | One-liner |
|---|---|
| Corners | intensity changes in **all** directions → unique landmarks · **Harris, Shi-Tomasi, FAST** · matching, tracking, panoramas |
| SIFT | scale space → **DoG** extrema → orientation → **128-D** descriptor · scale/rotation invariant · heavy |
| SURF | 2006, **faster SIFT** · Hessian-based detector · 64/128-D descriptor |
| HOG | gradients → **8×8 cells** → orientation histograms → block normalisation · **pedestrian detection** · scale-sensitive |
| Stereo | two views → match points → **disparity** (bigger = closer) → depth · hard in texture-less areas |
| Epipolar | match lies on the **epipolar line** · **Essential** (calibrated) / **Fundamental** (uncalibrated) matrix |
| Optical flow | pixel motion between frames · assumes **brightness constancy** · **Lucas–Kanade** (local) / **Horn–Schunck** (global smooth) |
| Kalman | **predict → update/correct** · noisy data, smooth tracks · linear + Gaussian (EKF = non-linear) |
| Tracking | detection-based (YOLO, Faster R-CNN) · by motion (optical flow) · hybrid (**SORT / DeepSORT**) · occlusion is hard |
| Convolution | (I∗K)(x,y) = Σₘ Σₙ I(x−m, y−n)·K(m,n) |

## Applications
CV: self-driving cars, medical imaging, surveillance, AR, traffic monitoring, retail (faces, products), creative AI.
Filtering: tumour boundaries, document scanning, lane detection.

## Display gotchas
- `plt.imshow(gray, cmap='gray')` for 1-channel images (otherwise you get fake colours).
- Colour from cv2 → convert BGR→RGB first. `cv2.imshow` needs no conversion.

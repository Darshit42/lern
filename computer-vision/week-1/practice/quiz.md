# Week 1 Practice Quiz

Answer each question **out loud or on paper first**, then click ▸ to check. Questions are in the styles evals usually use: concepts, "what does this code do", "predict the output", and "find the bug".

---

## Part A · Concepts (Day 1)

**A1.** What is Computer Vision? Give 3 applications.
<details><summary>Answer</summary>The field of AI that lets machines interpret and understand visual information (images and video). Applications: self-driving cars, medical imaging (tumour detection), surveillance, augmented reality, traffic monitoring, retail (face recognition, product tracking), creative AI (art generation).</details>

**A2.** What is a pixel? What values can a pixel in an 8-bit grayscale image have?
<details><summary>Answer</summary>The smallest square ("picture element") of a digital image. In an 8-bit grayscale image each pixel is one whole number from 0 (black) to 255 (white), because 8 bits give 2⁸ = 256 values.</details>

**A3.** An image has `img.shape == (720, 1280, 3)`. What are its width, height and number of channels? How many numbers does it store in total?
<details><summary>Answer</summary>Height 720, width 1280, 3 channels. 720 × 1280 × 3 = 2,764,800 numbers.</details>

**A4.** What does "resolution" mean and why does it matter?
<details><summary>Answer</summary>The number of pixels (width × height). More pixels = more detail, but also more memory and slower processing.</details>

**A5.** In which order does OpenCV store colour channels? What goes wrong if you ignore this?
<details><summary>Answer</summary>BGR (Blue, Green, Red). Matplotlib expects RGB, so if you show a cv2 image with plt.imshow without converting, red and blue swap. Fix: <code>cv2.cvtColor(img, cv2.COLOR_BGR2RGB)</code>.</details>

**A6.** What do H, S and V stand for? What is the range of H in OpenCV and why?
<details><summary>Answer</summary>Hue (which colour), Saturation (how vivid/pure), Value (how bright). OpenCV's H is 0–179: the real range is 0–360°, and halving it lets it fit in a uint8 (max 255).</details>

**A7.** Why is HSV preferred over RGB for colour detection?
<details><summary>Answer</summary>HSV keeps the colour (Hue) separate from the brightness (Value). Lighting changes like shadows mostly change V, so you can detect a colour with a Hue range that works under different lighting. In RGB, lighting changes all three numbers.</details>

**A8.** Grayscale formula used by OpenCV? Why does green get the biggest weight?
<details><summary>Answer</summary>Y = 0.299R + 0.587G + 0.114B. Human eyes are most sensitive to green light.</details>

**A9.** What is a mask?
<details><summary>Answer</summary>A single-channel black/white image of the same size as the original: 255 (white) = keep/selected pixel, 0 (black) = ignore. Made with e.g. <code>cv2.inRange</code> and used with <code>cv2.bitwise_and</code>.</details>

---

## Part B · Concepts (Day 2)

**B1.** Write the translation matrix that moves an image 40 px right and 25 px up.
<details><summary>Answer</summary><code>np.float32([[1, 0, 40], [0, 1, -25]])</code>. Up is a negative y because y grows downwards.</details>

**B2.** In `cv2.getRotationMatrix2D(center, angle, scale)`, what does each argument do? Which direction is a positive angle?
<details><summary>Answer</summary>center = pivot point (x, y); angle in degrees; scale = resize factor (1 = same size). Positive angle = anti-clockwise.</details>

**B3.** Why do the corners of an image get cut off after rotating by 45°?
<details><summary>Answer</summary>The output size given to warpAffine is the same as the original (cols, rows), but a rotated rectangle needs a bigger box, so the corners fall outside the canvas. Empty areas become black.</details>

**B4.** What is the defining property of an affine transformation? How many points define it?
<details><summary>Answer</summary>Parallel lines stay parallel (and straight lines stay straight). It's defined by 3 point pairs, giving a 2×3 matrix.</details>

**B5.** How is a perspective transformation different from an affine one? Give one real-world use.
<details><summary>Answer</summary>Perspective maps any quadrilateral to any quadrilateral using 4 point pairs and a 3×3 matrix; parallel lines can converge. Used for document scanning (flattening a page photographed at an angle), bird's-eye view of a road, and so on.</details>

**B6.** Which interpolation would you choose to shrink an image? To enlarge it smoothly?
<details><summary>Answer</summary>Shrink: <code>cv2.INTER_AREA</code>. Enlarge smoothly: <code>cv2.INTER_CUBIC</code> (or INTER_LINEAR, the default, which is faster).</details>

**B7.** What is data augmentation and why is it used?
<details><summary>Answer</summary>Creating modified copies of training images (shifted, rotated, scaled, flipped, warped...) so a model sees more variety. This reduces overfitting and makes the model robust to position, size and angle changes.</details>

**B8.** Translation, scaling and rotation: are they affine?
<details><summary>Answer</summary>Yes. All three (and shear) are special cases of affine transformations.</details>

---

## Part C · Concepts (Day 3)

**C1.** Define kernel, convolution and feature map.
<details><summary>Answer</summary>Kernel: a small matrix of weights (e.g. 3×3). Convolution: sliding the kernel over the image and at each position computing the weighted sum (multiply element-wise, then add). Feature map: the output image of those sums, which highlights where the pattern occurs.</details>

**C2.** Compute the output of this 3×3 patch with the sharpen kernel (centre value only):
```
patch = [[10, 10, 10],        kernel = [[ 0,-1, 0],
         [10, 50, 10],                  [-1, 5,-1],
         [10, 10, 10]]                  [ 0,-1, 0]]
```
<details><summary>Answer</summary>5×50 − 10 − 10 − 10 − 10 = 250 − 40 = <b>210</b>. The bright centre gets even brighter compared with its neighbours: sharpening.</details>

**C3.** Same patch, with the 3×3 box (average) blur kernel?
<details><summary>Answer</summary>(8×10 + 50)/9 = 130/9 ≈ <b>14.4</b>. The bright spot gets smoothed away.</details>

**C4.** Kernel sums: what do they tell you? Give the sum for blur, sharpen and Sobel.
<details><summary>Answer</summary>Sum 1 keeps overall brightness (blur = 1, sharpen = 1). Sum 0 sends flat regions to 0, keeping only changes, i.e. edges (Sobel = 0, Laplacian = 0).</details>

**C5.** Difference between average blur and Gaussian blur?
<details><summary>Answer</summary>Average blur weights all neighbours equally. Gaussian blur weights them with a bell curve, so closer pixels count more. Gaussian looks more natural and preserves edges a bit better.</details>

**C6.** Why must kernel sizes like (5, 5) be odd?
<details><summary>Answer</summary>So the kernel has a single centre pixel to place on the pixel being computed.</details>

**C7.** What do `dx=1, dy=0` in `cv2.Sobel` detect? Why is `cv2.CV_64F` used?
<details><summary>Answer</summary>Changes in the x direction, which means <b>vertical</b> edges. CV_64F (float) is used because gradients can be negative (bright→dark). uint8 would clip negatives to 0 and lose those edges.</details>

**C8.** List the 4 stages of the Canny edge detector.
<details><summary>Answer</summary>1) Gaussian noise reduction, 2) gradient computation (Sobel: magnitude + direction), 3) non-maximum suppression (thin edges to 1 px), 4) hysteresis thresholding with low/high thresholds.</details>

**C9.** In `cv2.Canny(img, 100, 200)`, classify pixels with gradient strength 250, 150 and 50.
<details><summary>Answer</summary>250 → strong edge (kept). 150 → weak, kept only if connected to a strong edge. 50 → discarded.</details>

**C10.** What happens to the Canny result if you lower both thresholds?
<details><summary>Answer</summary>More edges are detected, including noise and texture. Raising them keeps only the strongest outlines.</details>

**C11.** Sobel vs Laplacian?
<details><summary>Answer</summary>Sobel is a first derivative, computed separately in x or y (it has a direction). Laplacian is a second derivative that detects edges in all directions with one kernel, but it is more sensitive to noise.</details>

**C12.** Why is convolution called "the foundation of CNNs"?
<details><summary>Answer</summary>CNN layers are made of convolution kernels. Instead of hand-designing kernels (like Sobel), the network learns the kernel values from data. Early layers end up learning edge-like filters.</details>

**C13.** Output size of a 28×28 image convolved with a 3×3 kernel, stride 1, no padding? With padding 1?
<details><summary>Answer</summary>(28 − 3 + 0)/1 + 1 = 26 → 26×26. With P = 1: (28 − 3 + 2)/1 + 1 = 28 → 28×28 (same size).</details>

**C14.** What is a contour? What do `RETR_EXTERNAL` and `CHAIN_APPROX_SIMPLE` mean?
<details><summary>Answer</summary>A contour is a curve of points along the boundary of a shape. RETR_EXTERNAL returns only the outermost contours. CHAIN_APPROX_SIMPLE compresses straight segments down to their end points to save memory.</details>

**C15.** The edge-based object detector draws boxes around lots of tiny noise. Give 2 fixes.
<details><summary>Answer</summary>Blur before Canny; skip contours with <code>cv2.contourArea(cnt) &lt; threshold</code>; dilate the edges to close gaps; tune the Canny thresholds.</details>

**C16.** Name one application of filtering in each of medical imaging, document scanning and autonomous driving.
<details><summary>Answer</summary>Tumour boundary detection; detecting page edges / sharpening text; lane detection.</details>

---

## Part D · Predict the output

**D1.**
```python
img = cv2.imread('photo.jpg')     # a 400 × 600 colour photo (height 400)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(gray.shape)
```
<details><summary>Answer</summary><code>(400, 600)</code>. Grayscale has no channel dimension.</details>

**D2.**
```python
small = cv2.resize(img, None, fx=0.5, fy=0.25)
print(small.shape)               # img is (400, 600, 3)
```
<details><summary>Answer</summary><code>(100, 300, 3)</code>. Width 600×0.5 = 300, height 400×0.25 = 100.</details>

**D3.**
```python
a = np.array([200], dtype=np.uint8)
print(a + a)
```
<details><summary>Answer</summary><code>[144]</code>. 400 wraps around: 400 − 256 = 144. (<code>cv2.add</code> would give 255.)</details>

**D4.**
```python
edges = cv2.Canny(img, 100, 200)
print(edges.shape, np.unique(edges))      # img is (400, 600, 3)
```
<details><summary>Answer</summary><code>(400, 600) [  0 255]</code>, a single-channel image containing only 0 and 255 (assuming some edges were found).</details>

**D5.**
```python
img = np.zeros((100, 200, 3), np.uint8)
img[:, :100] = (0, 0, 255)
print(img[50, 150], img[50, 50])
```
<details><summary>Answer</summary><code>[0 0 0] [  0   0 255]</code>. The left half (columns 0–99) is red in BGR. Pixel (row 50, col 150) is on the right, so it's black.</details>

**D6.**
```python
hsv = cv2.cvtColor(np.uint8([[[0, 255, 0]]]), cv2.COLOR_BGR2HSV)
print(hsv)
```
<details><summary>Answer</summary><code>[[[ 60 255 255]]]</code>. Pure green has hue 120°, which OpenCV halves to 60.</details>

---

## Part E · Find the bug

**E1.**
```python
img = cv2.imread('sample.jpg')
plt.imshow(img)          # colours look wrong
```
<details><summary>Answer</summary>BGR shown as RGB. Use <code>plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))</code>.</details>

**E2.**
```python
M = np.float32([[1, 0, 100], [0, 1, 50]])
translated = cv2.warpAffine(img, M, (rows, cols))
```
<details><summary>Answer</summary>The size must be (width, height) = <code>(cols, rows)</code>. Also <code>rows, cols</code> must be defined first: <code>rows, cols = img.shape[:2]</code>.</details>

**E3.**
```python
blur = cv2.GaussianBlur(img, (4, 4), 0)
```
<details><summary>Answer</summary>The kernel size must be odd, e.g. (3, 3) or (5, 5). An even size raises an error.</details>

**E4.**
```python
sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0)
```
<details><summary>Answer</summary>CV_8U can't hold negative gradients, so bright→dark edges are lost. Use <code>cv2.CV_64F</code> and then <code>cv2.convertScaleAbs</code>.</details>

**E5.**
```python
img = cv2.imread('sampel.jpg')
print(img.shape)          # AttributeError: 'NoneType' object has no attribute 'shape'
```
<details><summary>Answer</summary>The file name is misspelled or the path is wrong. <code>cv2.imread</code> doesn't raise an error; it returns <code>None</code>. Always check <code>if img is None</code>.</details>

**E6.**
```python
M = cv2.getPerspectiveTransform(pts1, pts2)
out = cv2.warpAffine(img, M, (300, 300))
```
<details><summary>Answer</summary>A perspective matrix is 3×3 and must be applied with <code>cv2.warpPerspective</code>.</details>

**E7.**
```python
edges = cv2.Canny(img, 100, 200)
plt.imshow(cv2.cvtColor(edges, cv2.COLOR_BGR2RGB))
```
<details><summary>Answer</summary>Canny returns a single-channel image, so a BGR→RGB conversion makes no sense: there are no colour channels to swap. In OpenCV 4.x it raises an "Invalid number of channels" error; newer versions quietly copy the grey into 3 channels, which is pointless. Use <code>plt.imshow(edges, cmap='gray')</code>.</details>

---

## Part F · Day 4: Histograms & Thresholding

**F1.** What does an image histogram show, and what is on each axis?
<details><summary>Answer</summary>The distribution of pixel intensities: how many pixels have each brightness. x-axis = intensity (0–255), y-axis = number of pixels (frequency).</details>

**F2.** A histogram has almost all its pixels between 200 and 255. Describe the image. What if they're all between 100 and 140?
<details><summary>Answer</summary>200–255: very bright / overexposed. 100–140: low contrast (flat, greyish). Histogram equalization would stretch it out.</details>

**F3.** Explain each argument of `cv2.calcHist([img], [0], None, [256], [0, 256])`.
<details><summary>Answer</summary>List of images; channel index (0, the only channel of a grayscale image); mask (None = the whole image); number of bins (256); value range [0, 256) (the end is excluded).</details>

**F4.** Global vs adaptive thresholding: what's the difference, and when should you use adaptive?
<details><summary>Answer</summary>Global uses one threshold for the whole image. Adaptive computes a separate threshold for each small neighbourhood (its mean or Gaussian-weighted mean minus C). Use adaptive when the illumination varies, e.g. shadows across a page.</details>

**F5.** What does Otsu's method do, and what does it minimise? When does it work best?
<details><summary>Answer</summary>It automatically picks the optimal global threshold by minimising intra-class (within-group) variance, i.e. making the dark and bright groups as uniform as possible. It works best when the histogram is bimodal (two humps: object and background).</details>

**F6.** In `cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)`, why is the threshold 0? What is the first return value?
<details><summary>Answer</summary>With THRESH_OTSU the given threshold is ignored, because Otsu computes it. The first return value is the threshold Otsu chose.</details>

**F7.** What are `blockSize` and `C` in `cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)`? Mean vs Gaussian method?
<details><summary>Answer</summary>blockSize = 11 is the neighbourhood size (it must be odd and greater than 1). C = 2 is subtracted from the local mean to give the threshold. MEAN_C uses the plain average of the block; GAUSSIAN_C uses a Gaussian-weighted average (nearer pixels count more).</details>

**F8.** Give 3 applications of thresholding.
<details><summary>Answer</summary>Document scanning / binarization, medical imaging (tumour segmentation), industrial inspection (defect detection), object detection (separating foreground from background).</details>

**F9.** Predict: `a = np.array([[50, 127, 128, 200]], np.uint8)`. What is `cv2.threshold(a, 127, 255, cv2.THRESH_BINARY)[1]`? And with `THRESH_TRUNC`?
<details><summary>Answer</summary>BINARY: <code>[[0, 0, 255, 255]]</code>. 127 is <i>not</i> greater than 127, so it becomes 0. TRUNC: <code>[[50, 127, 127, 127]]</code>: values above 127 are capped at 127.</details>

**F10.** Predict: an image that is 100 everywhere goes through `adaptiveThreshold(..., MEAN_C, THRESH_BINARY, 11, 2)`. What comes out?
<details><summary>Answer</summary>All white (255). Each pixel (100) is greater than its local mean minus C (100 − 2 = 98). This is why flat, noisy regions come out speckled: tiny noise decides the result.</details>

**F11.** Predict the histogram (4 bins over 0–256) of `[[0,0,64,64],[0,128,128,64],[255,255,128,64],[255,255,255,128]]`.
<details><summary>Answer</summary>Bins [0–63], [64–127], [128–191], [192–255] → <code>[3, 4, 4, 5]</code> (total 16 pixels).</details>

---

## Part G · Day 5: Morphological operations

**G1.** Define dilation and erosion. What does each do to white regions?
<details><summary>Answer</summary>Dilation: a pixel becomes white if ANY pixel under the structuring element is white (local max), so white regions expand. Erosion: a pixel stays white only if ALL pixels under the element are white (local min), so white regions shrink.</details>

**G2.** Define opening and closing (the order of operations), and what each is used for.
<details><summary>Answer</summary>Opening = erosion followed by dilation: it removes small white noise while keeping the shape of large objects. Closing = dilation followed by erosion: it fills small holes and gaps inside objects.</details>

**G3.** What is a structuring element? Name 3 shapes and say what its size controls.
<details><summary>Answer</summary>A small binary matrix used to probe the image. Shapes: rectangle, ellipse, cross (<code>cv2.getStructuringElement</code>). The size controls how strong the effect is: a bigger element removes bigger specks but also damages more detail.</details>

**G4.** A binary scan has white text with small black holes inside the letters. Which operation fixes it?
<details><summary>Answer</summary>Closing.</details>

**G5.** A binary image has white objects plus scattered white specks of noise. Which operation fixes it?
<details><summary>Answer</summary>Opening.</details>

**G6.** Predict: a 5×5 black image with one white pixel in the centre. How many white pixels after a 3×3 dilation? After a 3×3 erosion?
<details><summary>Answer</summary>Dilation: 9 (a 3×3 white square). Erosion: 0 (the single pixel doesn't have all of its neighbours white).</details>

**G7.** You have black text on white paper and dilate it. Do the letters get thicker or thinner? Why?
<details><summary>Answer</summary>Thinner! Dilation grows WHITE regions, and here the white is the paper. Invert first (THRESH_BINARY_INV or bitwise_not) so the text is white.</details>

**G8.** Why did the lecture's dilated and eroded photos barely change?
<details><summary>Answer</summary>They were grayscale photos. On grayscale, dilation and erosion are a local max and min, which just slightly brighten or darken and blur the image. The effect is clearest on binary images with small details.</details>

**G9.** What do `iterations=3` and `MORPH_GRADIENT` do?
<details><summary>Answer</summary>iterations=3 repeats the operation 3 times (a stronger effect). MORPH_GRADIENT = dilation − erosion, which gives the outline of the shapes.</details>

**G10.** Name 3 applications of morphology.
<details><summary>Answer</summary>Document cleaning (remove noise, fill gaps), medical imaging (highlight cells and structures), industrial inspection (defects), and cleaning shapes before contour analysis.</details>

---

## Part H · Day 6: Contours & shape analysis

**H1.** What is a contour? Name 4 contour descriptors.
<details><summary>Answer</summary>A curve joining continuous points along the boundary of a shape with the same intensity. Descriptors: area, perimeter, centroid, bounding box, convex hull (also aspect ratio, solidity, circularity).</details>

**H2.** Bounding box vs convex hull: definition and use of each.
<details><summary>Answer</summary>Bounding box: the smallest (upright) rectangle enclosing the contour, used for object localization. Convex hull: the smallest convex polygon enclosing the contour (a rubber band around it), used for shape simplification and finding dents (convexity defects).</details>

**H3.** How does the lecture classify shapes? What is the 0.02 in `approxPolyDP`?
<details><summary>Answer</summary>Approximate the contour to a polygon with <code>cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)</code> and count the vertices: 3 = triangle, 4 = rectangle, more than 6 = circle, otherwise polygon. 0.02 sets epsilon to 2% of the perimeter, the maximum distance the simplified outline may stray from the real one. A bigger epsilon gives fewer corners.</details>

**H4.** Dark shapes on white paper: `threshold(img, 127, 255, THRESH_BINARY)` then `findContours(..., RETR_EXTERNAL, ...)` returns 1 contour. Why? How do you fix it?
<details><summary>Answer</summary>findContours looks for WHITE objects. THRESH_BINARY makes the paper white and the shapes black, so the only external contour is the page. Use THRESH_BINARY_INV (optionally with Otsu).</details>

**H5.** `RETR_EXTERNAL` vs `RETR_TREE`? `CHAIN_APPROX_SIMPLE` vs `CHAIN_APPROX_NONE`?
<details><summary>Answer</summary>EXTERNAL: outermost contours only. TREE: all contours plus the full parent/child hierarchy. SIMPLE: stores only the end points of straight segments. NONE: stores every boundary point.</details>

**H6.** How do you compute a contour's centroid?
<details><summary>Answer</summary><code>M = cv2.moments(cnt)</code>; <code>cx = M['m10']/M['m00']</code>, <code>cy = M['m01']/M['m00']</code> (m00 is the area; check it isn't 0).</details>

**H7.** The lecture's rule labels a 5-pointed star as "Circle". Why? How could you detect a star instead?
<details><summary>Answer</summary>The star approximates to 10 vertices, and more than 6 counts as "Circle". Use solidity = area / convex hull area: a star is about 0.5, while convex shapes are about 1.</details>

**H8.** Predict: a white filled rectangle drawn with `cv2.rectangle(img, (10, 20), (50, 40), 255, -1)` on black. What does `cv2.boundingRect` return for its contour? How many points does `CHAIN_APPROX_SIMPLE` store?
<details><summary>Answer</summary><code>(10, 20, 41, 21)</code>. The drawn rectangle includes both corner pixels, so the width is 50 − 10 + 1 = 41 and the height 21. 4 points (one per corner).</details>

**H9.** Predict: a white ring (a filled circle with a smaller black circle cut out). How many contours with `RETR_EXTERNAL`? With `RETR_TREE`?
<details><summary>Answer</summary>EXTERNAL: 1 (just the outer edge). TREE: 2 (outer edge + inner hole, with the hole as a child of the outer contour).</details>

**H10.** Why does `drawContours(img, [hull], 0, ...)` put `hull` in square brackets?
<details><summary>Answer</summary>drawContours expects a LIST of contours. <code>[hull]</code> is a list with one contour, and index 0 draws that one.</details>

---

## Part I · Key concepts (advanced topics)

**I1.** Why are corners better features to track than edges?
<details><summary>Answer</summary>At a corner the intensity changes in every direction, so it's uniquely localizable and repeatable. Along an edge, sliding the window looks the same, so the position is ambiguous.</details>

**I2.** Name 3 corner detectors and 3 uses of corners.
<details><summary>Answer</summary>Harris, Shi-Tomasi, FAST. Uses: image matching, tracking, panorama stitching.</details>

**I3.** List the SIFT pipeline. What is it robust to? What is its main limitation?
<details><summary>Answer</summary>Gaussian scale space → Difference-of-Gaussians extrema → orientation assignment → 128-dimensional descriptor. Robust to scale and rotation; computationally heavy.</details>

**I4.** What is SURF and how does it differ from SIFT?
<details><summary>Answer</summary>Speeded-Up Robust Features (2006): a faster alternative to SIFT, also scale- and rotation-invariant. It detects interest points with a Hessian-matrix-based measure and describes them with 64- or 128-dimensional vectors.</details>

**I5.** Describe the HOG pipeline and its classic application.
<details><summary>Answer</summary>Compute gradients → divide into cells (e.g. 8×8) → build an orientation histogram per cell → normalize over blocks. Classic use: pedestrian detection. Lightweight and interpretable, but sensitive to scale.</details>

**I6.** What is stereo vision? What is its main challenge?
<details><summary>Answer</summary>Using two images from slightly different viewpoints to estimate depth by matching corresponding points (closer objects have bigger disparity). The challenge is reliable matching in texture-less regions.</details>

**I7.** What does epipolar geometry say? Essential vs Fundamental matrix?
<details><summary>Answer</summary>A point's match in the other image must lie on the epipolar line, which reduces the search to 1-D. Essential matrix: calibrated cameras. Fundamental matrix: uncalibrated cameras.</details>

**I8.** What is optical flow? What does it assume? Compare Lucas–Kanade and Horn–Schunck.
<details><summary>Answer</summary>It estimates the apparent motion of pixels between consecutive frames, assuming brightness constancy. Lucas–Kanade works in a local neighbourhood; Horn–Schunck uses a global smoothness constraint. It's sensitive to lighting changes and occlusions.</details>

**I9.** Describe the two steps of a Kalman filter. When does it work best?
<details><summary>Answer</summary>Predict (estimate the next state from a motion model) and update (correct it with the measurement). It works best for linear motion with Gaussian noise; the Extended Kalman Filter handles non-linear motion.</details>

**I10.** Name 3 object-tracking approaches with an example of each, and 2 challenges.
<details><summary>Answer</summary>Detection-based (YOLO / Faster R-CNN per frame + linking), tracking-by-motion (optical flow, correlation filters), hybrid (SORT, DeepSORT). Challenges: occlusion, scale change, multiple objects, cluttered backgrounds.</details>

**I11.** Write the 2-D convolution formula and name its parts.
<details><summary>Answer</summary>(I∗K)(x, y) = Σₘ Σₙ I(x − m, y − n) · K(m, n). I = image, K = kernel, (x, y) = output pixel, and m, n run over the kernel.</details>

**I12.** Which kernel highlights vertical edges: Sobel X or Sobel Y? Write it.
<details><summary>Answer</summary>Sobel X: <code>[[-1,0,1],[-2,0,2],[-1,0,1]]</code>. It responds to left-to-right changes, which form vertical edges.</details>

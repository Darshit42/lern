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

# Week 1 Question Bank: Model Answers

Answers to every question in the course's **CV-Week1-Questions.pdf** (original in `course-files/question-bank/`). Each answer is written the way you'd say it in an eval: short, correct and in plain words. Where it helps, there's a pointer to the note or code that shows it working.

> How to use: cover the answer, say yours out loud, then compare. The "Analytical & Design" ones are the longest. Practise them as 5–8 line answers.

---

## Part 1 · Concept Recall

**1. Grayscale vs colour image as a matrix?**  
A grayscale image is a **2-D matrix** `(rows, cols)` with one number per pixel. A colour image is a **3-D matrix** `(rows, cols, 3)` with three numbers per pixel, one per channel (B, G, R in OpenCV). → `notes/day1_images_and_colors.md`

**2. What do 0 and 255 mean in a grayscale image?**  
0 = **black** (no light), 255 = **white** (full intensity). Values in between are shades of grey. 255 is the maximum because pixels are stored in 8 bits (2⁸ = 256 values).

**3. How many channels does an RGB image have?**  
**Three**: Red, Green and Blue.

**4. Where is the coordinate origin in CV images?**  
At the **top-left corner** `(0, 0)`. x increases to the right and **y increases downward**.

**5. How do you access the pixel at (x = 10, y = 5) in OpenCV?**  
`img[5, 10]`: **row (y) first, then column (x)**. For a colour image it returns `[B, G, R]`.

**6. Default colour order in OpenCV?**  
**BGR** (Blue, Green, Red), not RGB.

**7. What does HSV stand for?**  
**Hue, Saturation, Value.**

**8. What does Hue represent?**  
The **type of colour** (red, green, blue…), measured as an **angle on the colour wheel**: 0° = red, 120° = green, 240° = blue. OpenCV stores it halved, 0–179.

**9. What does Saturation indicate?**  
How **pure or intense** the colour is: 0 = grey (washed out), maximum = fully vivid.

**10. What does Value indicate?**  
The **brightness**: 0 = black, maximum = the brightest version of that colour.

**11. Define a kernel.**  
A **small matrix of numbers** (e.g. 3×3 or 5×5), also called a filter, mask or convolution matrix, that slides over an image. At each position it defines how a pixel and its neighbours are **weighted and combined** into a new output pixel.

**12. Purpose of convolution in image filtering?**  
To **extract or change features**: blur (smooth), sharpen, or detect edges and textures. You slide a kernel over the image and at each spot **multiply and sum**. The output is a feature map.

**13. What does a histogram represent?**  
The **distribution of pixel intensities**: how many pixels have each value (x-axis 0–255, y-axis = count). It shows brightness and contrast.

**14. What is thresholding used for?**  
To convert a grayscale image to **binary** (black and white): pixels above a cut-off become 255 and the rest become 0. This **separates foreground from background**, e.g. text from paper or an object from the background.

**15. Name two basic morphological operations.**  
**Dilation** and **erosion**. Opening and closing are combinations of the two.

---

## Part 2 · Application & Explanation

**1. Why is HSV preferred over RGB for colour detection?**  
HSV **separates colour (Hue) from brightness (Value)**. When lighting changes (shadow or sunlight), mostly V changes while H stays about the same. So you can detect "red" with a simple Hue range. In RGB all three numbers change with the light, so a fixed colour range breaks easily. → `code/day1_03_color_spaces.py`

**2. An RGB pixel value for pure red?**  
`(255, 0, 0)` in RGB. *(In OpenCV's BGR order the same red is stored as `[0, 0, 255]`.)*

**3. What happens when you apply a blur kernel?**  
Each pixel is replaced by an **average of its neighbours** (equal weights for a box blur, bell-curve weights for a Gaussian blur). Sharp changes get smoothed out: **noise and fine detail are reduced** and edges become softer. A bigger kernel means more blur. The kernel's numbers sum to 1, so overall brightness stays the same.

**4. How does an edge-detection kernel work?**  
Its numbers **sum to 0** and have opposite signs on opposite sides (e.g. Sobel-X `[[-1,0,1],[-2,0,2],[-1,0,1]]`). Over a **flat area** the positives and negatives cancel, giving **0**. Where brightness **changes suddenly** (an edge) they don't cancel, giving a **large value**. So the output lights up exactly on edges. Sobel-X finds vertical edges and Sobel-Y finds horizontal ones. → `code/day3_01_convolution_by_hand.py`

**5. Compare global and adaptive thresholding.**  
- **Global** uses **one threshold for the whole image**. It's simple and fast, but fails when lighting is uneven.
- **Adaptive** computes a **separate threshold for each small neighbourhood** (local mean or Gaussian-weighted mean minus C). It handles shadows and uneven lighting, but has more parameters (blockSize, C) and can turn flat areas speckled.
- **Otsu** is a global method that picks the threshold automatically.
→ `notes/day4_histograms_and_thresholding.md`

**6. What does a histogram tell you about brightness and contrast?**  
- **Where** the pixels sit tells you **brightness**: bunched on the left means dark (underexposed), on the right means bright (overexposed).
- **How spread out** they are tells you **contrast**: a narrow hump means low contrast (flat, grey), spread across 0–255 means high contrast.
- Two separate humps (**bimodal**) mean a clear object and background, which is ideal for thresholding.

**7. Dilation vs erosion?**  
- **Dilation** (local maximum) **expands white regions**: it thickens objects, connects broken parts and fills small gaps.
- **Erosion** (local minimum) **shrinks white regions**: it thins objects, removes small specks and separates touching objects.
- Both use a structuring element. Its size controls how strong the effect is.

**8. Opening vs closing?**  
- **Opening = erosion then dilation**: removes **small white noise** (specks) while keeping large shapes about the same size.
- **Closing = dilation then erosion**: fills **small black holes and gaps** inside objects.
- Analogy: opening erases specks; closing fills gaps.

**9. Why are structuring elements important?**  
They **define the neighbourhood** the operation looks at, so they decide the result.
- **Size** sets the strength: a bigger element removes bigger noise but also destroys more detail.
- **Shape** (rectangle, ellipse, cross) decides what the edges and corners look like afterwards: square corners, rounded corners, or plus-shaped growth.
- Choose it **bigger than the noise but smaller than the detail you want to keep**.

**10. What is a convex hull?**  
The **smallest convex polygon that encloses a contour**, like a rubber band snapped around the shape. It simplifies the shape. Comparing area to hull area (**solidity**) reveals dents, e.g. a star ≈ 0.5 while a square ≈ 1. → `notes/day6_contours_and_shapes.md`

**11. How are bounding boxes used for object localization?**  
`cv2.boundingRect(contour)` returns `(x, y, w, h)`, the **smallest upright rectangle around the object**. It says **where** the object is in compact form: you can draw it, crop it out, track it between frames, count objects, or measure their size and aspect ratio. Detection models (YOLO, SSD) output exactly this kind of box.

**12. Why is the origin (0, 0) at the top-left?**  
Because of how images are **stored and displayed**. Screens and old CRT TVs draw the picture **row by row from the top-left**, and image files and memory store pixels in that same order. So row 0 is the top row. That's also why images are matrices indexed `[row, col]`, and why y grows downward.

**13. How does Matplotlib's RGB order differ from OpenCV's BGR?**  
They store the same three channels in **opposite order**: OpenCV `[B, G, R]`, Matplotlib `[R, G, B]`. Show a `cv2.imread` image with `plt.imshow` without converting and **red and blue swap** (orange looks blue). Fix: `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`.

**14. Role of kernels in CNNs vs classical image processing?**  
- **Classical:** a **human designs** the kernel numbers for a known job, e.g. Sobel for edges or a box filter for blur.
- **CNN:** the kernel numbers are **learned automatically** from data during training. Early layers end up learning edge and colour detectors (much like Sobel), and deeper layers combine them into textures, parts and whole objects.
- The **operation is the same** (slide, multiply, sum). Only who chooses the numbers differs.

**15. How can thresholding be applied in medical imaging?**  
Tissues show up at different **intensity ranges** in scans such as X-ray, CT and MRI. Thresholding can **segment** the region of interest, e.g. bone (very bright on CT), a tumour or a lesion. You then clean the mask with morphology and measure its **area or volume** with contours. Otsu or adaptive thresholding cope with varying scan brightness.

---

## Part 3 · Analytical & Design-Oriented

**1. Derive the mathematical expression for 2-D convolution.**  
The output at (x, y) is a weighted sum of the input pixels around (x, y), with weights from the kernel K:

  **(I ∗ K)(x, y) = Σₘ Σₙ I(x − m, y − n) · K(m, n)**

m and n run over the kernel's positions, e.g. −1..1 for 3×3. The minus signs mean the kernel is **flipped**. That's what makes it true convolution. `cv2.filter2D` and CNNs actually compute **correlation**, Σ I(x + m, y + n)·K(m, n), with no flip. For symmetric kernels the two are identical.
Output size without padding: **(N − K + 2P)/S + 1**.

**2. Limitations of global thresholding under uneven illumination, and solutions.**  
- **Problem:** one cut-off can't fit both the dark and bright parts. With a shadow, the dark side turns entirely black (text lost) or the bright side loses faint text. Otsu doesn't help, because it's still one global value.
- **Solutions:**
  - **Adaptive thresholding**: a local threshold per neighbourhood, Gaussian or mean.
  - Blur first to avoid speckle.
  - **Background normalisation**: estimate the background with a large blur or morphological closing, then divide or subtract it before thresholding.
  - **CLAHE** to even out local contrast.
  - Split the image into tiles and use Otsu per tile.

→ `code/day4_02_thresholding.py` (uneven-lighting demo)

**3. How does histogram equalization improve contrast?**  
A low-contrast image uses only a **narrow range** of intensities (a squashed histogram). Equalization remaps pixel values using the **cumulative histogram (CDF)**, new value = 255 × CDF(old value), so the output intensities are **spread across the full 0–255 range**. Similar dark and bright values get pulled further apart, so details become visible. `cv2.equalizeHist(gray)`. **CLAHE** does it per tile with a clip limit, so it doesn't over-boost noise. → `code/day4_01_histograms.py`

**4. Design a pipeline to detect ripe fruit using HSV.**  
1. Capture image → **blur** slightly (reduce noise).
2. **Convert BGR → HSV.**
3. **Threshold on Hue** for the ripe colour (e.g. red: H 0–10 *and* 170–179; ripe banana yellow: about 20–35), with minimum **S** (to ignore grey) and **V** (to ignore dark shadow). Use `cv2.inRange` to get a mask.
4. **Morphology**: opening to remove specks, closing to fill holes.
5. **Find contours**, drop the small ones (area filter), optionally check circularity.
6. Output **bounding boxes / count** of ripe fruit. Compare the ripe-hue mask with an unripe-hue mask (green, H about 40–80) to report ripeness.

Why HSV: ripeness is a **colour (Hue)** property, and HSV keeps it stable under changing light.

**5. How is contour approximation (Douglas–Peucker) used for shape classification?**  
`cv2.approxPolyDP(cnt, ε, True)` **simplifies a contour into a polygon** with fewer vertices. Douglas–Peucker keeps a point only if it lies **farther than ε** from the line joining its neighbours. Choose ε relative to the size, e.g. **ε = 0.02 × perimeter**, then **count the vertices**: 3 = triangle, 4 = rectangle/square (check aspect ratio w/h ≈ 1 for a square), 5 = pentagon, many = circle. A large ε over-simplifies (a circle can become a square). Add solidity or circularity to handle stars and ellipses. → `code/day6_02_boxes_hull_shapes.py`

**6. How can morphology clean noisy scanned documents?**  
1. Grayscale → **threshold (Otsu or adaptive) with INV**, so the text is white.
2. **Opening** with a small kernel (bigger than the specks, smaller than the strokes) **removes salt noise**.
3. **Closing** **fills small gaps and holes in the letters**.
4. Invert back to black text on white.

The order and kernel size matter. If letters have pinholes, close first. A kernel that's too big erases thin strokes. → `code/day5_03_mini_project_document_cleaning.py`

**7. Separate overlapping objects using thresholding, morphology and contours.**  
1. **Threshold** (Otsu) to get a foreground mask.
2. **Opening** to remove noise. **Erosion** shrinks the objects until the thin "bridges" where they touch disappear, splitting them.
3. A better option: compute the **distance transform** (`cv2.distanceTransform`). Each object's centre becomes a peak, so threshold the peaks to get one **marker** per object.
4. Use the markers in **watershed** (`cv2.watershed`), or dilate them back, to recover each object's full extent.
5. **findContours** on the separated regions → count, box and measure each one.

**8. RGB vs HSV for detecting traffic lights under varying lighting.**  
- **RGB:** a red light at night, at noon, or in glare gives very different R, G and B values. Fixed RGB ranges either miss lights or fire on other bright objects, so performance drops sharply as lighting changes.
- **HSV:** the light's **Hue** stays roughly constant (red about 0/179, amber about 15–25, green about 50–90). Brightness changes mostly affect V. So use a Hue range plus **high S and high V** (lights are vivid and bright). This is much more robust.
- **Caveats:** glare can wash out saturation, white balance can shift hue, and red needs two Hue ranges because it wraps around. Add **shape (circularity)** and **position** checks via contours.

**9. How does kernel size affect convolution?**  
- A **larger kernel** looks at a **bigger neighbourhood**: stronger blur and smoothing, captures **coarser and larger structures**, removes more noise but loses fine detail and blurs edges, **costs more computation** (K² multiplications per pixel) and **shrinks the output more** without padding.
- A **smaller kernel** gives finer detail but is more noise-sensitive.
- In CNNs (VGG), stacking small 3×3 kernels reaches a large receptive field with fewer parameters.

**10. How is convolution related to feature extraction in CNNs?**  
A CNN layer is **many convolutions in parallel**, each with a **learned kernel**. Each kernel produces a **feature map** showing where its pattern appears. The layers form a hierarchy:
- **Layer 1** learns edges and colour blobs (like Sobel).
- **Middle layers** combine those into textures and parts.
- **Deep layers** respond to whole objects.

Pooling shrinks the maps, and the final layers classify. So convolution is the **feature extractor**, and training finds the kernel weights automatically. → `week-2-3/notebooks/01_…`

**11. How can histograms detect underexposed or overexposed images?**  
- **Underexposed:** most pixel counts are at the **low end** (hump near 0) and the mean is low. A spike at exactly 0 means **clipped shadows**.
- **Overexposed:** most counts are at the **high end**, and a spike at 255 means **clipped highlights**.

For an automatic check, compute the mean and the percentage of pixels below about 20 or above about 235 and compare with thresholds. Fixes include **gamma correction** or histogram equalization.

**12. Trade-offs between small and large structuring elements.**  
- **Small:** removes only tiny noise, **preserves detail** and thin structures, but may leave larger noise and can't bridge large gaps.
- **Large:** removes bigger noise and fills bigger gaps, but **destroys small objects and thin lines**, rounds or merges shapes, and costs more computation.

Rule: **bigger than the noise, smaller than the smallest feature you need**. Shape matters too: an ellipse gives rounder results, a rectangle keeps corners square. You can also use a small kernel with **iterations**.

**13. How can contour features (area, perimeter, centroid) be used for classification?**  
They form a numeric **feature vector** per object:
- **Area** separates small from large objects (and filters noise).
- **Perimeter** and **circularity = 4π·A / P²** separate round from elongated or jagged shapes.
- **Aspect ratio** (w/h), **extent** (area / box area) and **solidity** (area / hull area) separate rectangles, circles and stars.
- **Centroid** gives the position, used for tracking or for "which lane or zone" decisions.

Classify with simple **rules** (thresholds) or feed the features to an ML classifier (k-NN, SVM). Example: coins sorted by area, or nuts vs bolts by circularity.

**14. Detect cracks in industrial materials using convolution and morphology.**  
1. Grayscale → **blur** (Gaussian) to suppress surface texture noise.
2. Cracks are **thin dark lines**. Use **edge or line convolution**: Sobel, Laplacian, Canny, or a **black-hat** transform (closing − image), which highlights thin dark structures.
3. **Threshold** (Otsu or adaptive) to get a binary crack mask.
4. **Morphology:** closing joins broken crack segments, and opening or an area filter removes small specks.
5. **Contours:** keep long, thin shapes (large length-to-width ratio, low solidity). Measure length and width and flag the part if they exceed tolerances.

**15. How does combining HSV segmentation with contour analysis improve detection accuracy?**  
- **HSV segmentation** finds pixels of the **right colour**, robustly under lighting changes. But colour alone gives false positives (anything with the same colour) and a noisy mask.
- **Contour analysis** adds **shape and size checks**: area range, circularity, aspect ratio and convexity. It also groups pixels into **distinct objects** with positions.

Together they reject wrong-shaped blobs, remove noise and give per-object boxes and counts. Example: traffic lights must be red **and** round **and** a certain size. That gives far fewer false positives than either check alone.

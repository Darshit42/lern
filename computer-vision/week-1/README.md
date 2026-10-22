# Computer Vision – Week 1 (start here)

Beginner-friendly study kit for the whole of **Week 1** (Polaris School of Technology): the 6 lecture days plus the Key Concepts summary sheet.

| Day | Lecture | What it's really about |
|-----|---------|------------------------|
| 1 | Introduction to Computer Vision | A picture is a grid of numbers. Colours are numbers too. |
| 2 | Image Transformations | Moving, resizing, turning and bending a picture with a little matrix. |
| 3 | Filtering & Convolutions | Sliding a small grid of numbers over a picture to blur it, sharpen it or find edges. |
| 4 | Histograms & Thresholding | Counting how many dark and bright pixels there are, then splitting the image into pure black and white. |
| 5 | Morphological Operations | A "marker and eraser" for black-and-white images: thicken, thin, remove specks, fill holes. |
| 6 | Contours & Shape Analysis | Tracing the outline of each object, then measuring it and naming the shape. |
| ★ | Key Concepts sheet | A recap of all 6 days, plus advanced topics: corners, SIFT, SURF, HOG, stereo, optical flow, Kalman filters, tracking. |

Everything here assumes **zero background**. Each idea is explained in plain words, shown in a picture, and comes with code you can run and change.

---

## Folder map

```
week-1/
├── README.md                ← you are here
├── Week1_Study_Guide.pdf    ← ALL notes + diagrams + quiz (answers shown) + cheat sheet in one printable PDF
├── cheat-sheet.md           ← revision sheet for the night before the eval
├── requirements.txt         ← the 3 Python libraries you need
├── notes/                   ← READ these, in order
│   ├── 00_python_numpy_basics.md          (only the Python you need)
│   ├── day1_images_and_colors.md
│   ├── day2_transformations.md
│   ├── day3_filtering_and_convolution.md
│   ├── day4_histograms_and_thresholding.md
│   ├── day5_morphological_operations.md
│   ├── day6_contours_and_shapes.md
│   └── key_concepts.md                     (recap + advanced topics)
├── images/                  ← explanation diagrams + example results used in the notes
├── code/                    ← RUN and EDIT these
│   ├── cv_helpers.py                       (helper: loads/creates images, shows results)
│   ├── sample.jpg                          (the test photo)
│   ├── day1_01 … day1_04                   images, pixels, colour spaces, image viewer
│   ├── day2_01 … day2_03                   translate/scale/rotate, affine vs perspective, augmentation
│   ├── day3_01 … day3_05                   convolution by hand ★, filters, Canny, object detector
│   ├── day4_01 … day4_03                   histograms, thresholding, document binarizer
│   ├── day5_01 … day5_03                   dilate/erode, opening/closing, document cleaner
│   ├── day6_01 … day6_03                   contours, boxes/hull/shapes, shape detector
│   ├── keyconcepts_01_corners_and_features.py   Harris, Shi-Tomasi, FAST, SIFT, ORB, HOG
│   ├── keyconcepts_02_motion_and_depth.py       optical flow, Kalman filter, stereo depth
│   ├── make_diagrams.py                    (redraws the diagrams in images/)
│   └── outputs/                            (results of every script; regenerated when you run them)
├── practice/
│   ├── quiz.md              ← 89 eval-style questions (all days + key concepts) with hidden answers
│   ├── exercises.py         ← 19 fill-in-the-blank coding tasks that print PASS/FAIL
│   └── solutions.py         ← answers to exercises.py
└── course-files/            ← the ORIGINAL course PDFs and Jupyter notebooks (see its README)
```

---

## Setup (5 minutes, once)

1. Install Python 3 (from python.org; on Windows tick **"Add Python to PATH"**).
2. Open a terminal in this `week-1` folder and run:
   ```
   pip install -r requirements.txt
   ```
3. Run your first script:
   ```
   cd code
   python day1_01_load_and_inspect.py
   ```
   A window shows the result, and a copy is saved in `code/outputs/`.

**Test images:**
- `code/sample.jpg` (a 1920×1080 sunset scene of a knight on a sand dune) is what most scripts load, just like the lectures do. Swap in any picture with the same name to experiment.
- Some lessons need a particular kind of image, so `cv_helpers.py` **generates** it: coloured shapes (Days 1–3), a page of text with a shadow and specks (Days 4–5), and black geometric shapes (Day 6).
- The pictures inside the notes are stored in `images/example_*.png`, so running the scripts never changes them.

**Prefer Jupyter / Google Colab?** Copy any script's contents into a notebook cell and upload `cv_helpers.py` next to it. The course's own notebooks are in `course-files/notebooks/`.

---

## Study plan

| Step | Do this | Time |
|------|---------|------|
| 1 | `notes/00_python_numpy_basics.md` (skip if you know NumPy arrays) | 20 min |
| 2 | Day 1 notes → run the 4 `day1_*` scripts → do their "TRY THIS" bits | 1 h |
| 3 | Day 2 notes → `day2_*` scripts | 1 h |
| 4 | Day 3 notes → run `day3_01_convolution_by_hand.py` **slowly**, then the rest | 1.5 h |
| 5 | Day 4 notes → `day4_*` scripts | 1 h |
| 6 | Day 5 notes → `day5_*` scripts | 45 min |
| 7 | Day 6 notes → `day6_*` scripts | 1 h |
| 8 | `notes/key_concepts.md` → `keyconcepts_*` scripts (focus on *what* and *why*) | 1 h |
| 9 | `practice/exercises.py`: try first, then check `solutions.py` | 1.5 h |
| 10 | `practice/quiz.md`: answer out loud before opening each answer | 1 h |
| 11 | Night before the eval: `cheat-sheet.md` | 20 min |

> Learning tip: after every script, change **one number** and predict what will happen *before* you run it. That habit teaches you more than reading.

**How the days connect:** Days 4 → 5 → 6 form a single pipeline used everywhere in classic CV:
`grayscale → threshold (Day 4) → clean with morphology (Day 5) → find contours and measure shapes (Day 6)`.

---

## Mistakes in the lecture material (worth knowing for the eval)

The lecture code mostly works, but several outputs look odd. Understanding *why* is a good test of whether you've really got it.

| # | What you see in the lecture | Why | Notes |
|---|---|---|---|
| 1 | "Mode: original" shows blue swatches instead of orange | BGR image shown in matplotlib (which expects RGB) | Day 1 |
| 2 | The HSV picture looks identical to the original | converted to HSV and straight back to RGB before showing | Day 1 |
| 3 | The perspective output is a plain grey square | its 4 corner points sit in the empty corner of a 5472×3648 photo | Day 2 |
| 4 | Day 2 code uses `rows, cols` without defining them | needs `rows, cols = img.shape[:2]` | Day 2 |
| 5 | "Clipping input data" warning on Sobel | showing a float, colour, possibly negative result directly | Day 3 |
| 6 | Edge images look almost completely black | 1-px edges on a huge photo vanish when it's shrunk | Day 3 |
| 7 | The object detector boxes dozens of noise specks | no blur, no minimum-area filter | Day 3 |
| 8 | The adaptive threshold looks like a pencil sketch | `blockSize=11` is tiny for a huge photo; noise decides flat areas | Day 4 |
| 9 | The "Document Binarizer" was run on a photo, not a document | — | Day 4 |
| 10 | Dilated / eroded / opened photos look almost unchanged | morphology on a grayscale photo; it's designed for binary images | Day 5 |
| 11 | The "Cleaned Document" isn't binary | no thresholding before opening/closing | Day 5 |
| 12 | Day 6 finds hundreds of contours on noise | its `sample.png` is the same *photo*, not a shapes image | Day 6 |
| 13 | On a real shapes image, the Day 6 code finds **1** contour | `THRESH_BINARY` makes the white paper the object; use `THRESH_BINARY_INV` | Day 6 |
| 14 | The convolution formula reads "I(x, −my − n)" | a typesetting slip for I(x − m, y − n) | Key concepts |

# Computer Vision – Week 1 (start here)

Beginner-friendly study kit for **CV-Week-1-Lectures.pdf** (Polaris School of Technology):

| Day | Lecture | What it's really about |
|-----|---------|------------------------|
| 1 | Introduction to Computer Vision | A picture is a grid of numbers. Colours are numbers too. |
| 2 | Image Transformations | Moving, resizing, turning and bending a picture with a little matrix. |
| 3 | Filtering & Convolutions | Sliding a small grid of numbers over a picture to blur it, sharpen it or find edges. |

Everything here assumes **zero background**. Each idea is explained in plain words, shown in a picture, and comes with code you can run and change.

---

## Folder map

```
week-1/
├── README.md                ← you are here
├── Week1_Study_Guide.pdf    ← ALL notes + diagrams + quiz (answers shown) + cheat sheet in one printable PDF
├── cheat-sheet.md           ← 1-page revision sheet (read the night before the eval)
├── requirements.txt         ← the 3 Python libraries you need
├── notes/                   ← READ these, in order
│   ├── 00_python_numpy_basics.md     (only the Python you need)
│   ├── day1_images_and_colors.md
│   ├── day2_transformations.md
│   └── day3_filtering_and_convolution.md
├── images/                  ← explanation diagrams used in the notes
├── code/                    ← RUN and EDIT these
│   ├── cv_helpers.py                  (helper: loads an image, shows results)
│   ├── day1_01_load_and_inspect.py
│   ├── day1_02_pixels_by_hand.py
│   ├── day1_03_color_spaces.py
│   ├── day1_04_mini_project_image_viewer.py
│   ├── day2_01_translate_scale_rotate.py
│   ├── day2_02_affine_vs_perspective.py
│   ├── day2_03_mini_project_augmentation.py
│   ├── day3_01_convolution_by_hand.py      ← the most important one
│   ├── day3_02_common_filters.py
│   ├── day3_03_canny.py
│   ├── day3_04_lab_exercise_compare.py
│   ├── day3_05_mini_project_object_detector.py
│   ├── make_diagrams.py               (redraws the pictures in images/)
│   └── outputs/                       (results of every script, already generated)
└── practice/
    ├── quiz.md              ← 40+ eval-style questions with hidden answers
    ├── exercises.py         ← fill-in-the-blank coding tasks
    └── solutions.py         ← answers to exercises.py
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

**The test photo:** `code/sample.jpg` (a 1920×1080 sunset scene of a knight on a sand dune) is what the scripts load, just like the lectures do. Swap in any picture with the same name to experiment. If it's missing, the scripts use a built-in practice image of coloured shapes, and the object detector always uses those shapes. The pictures inside the notes come from the shapes image and are stored in `images/example_*.png`, so running the scripts never changes them.

**Prefer Jupyter / Google Colab?** Copy any script's contents into a notebook cell. Upload `cv_helpers.py` next to it, or replace `show(...)` with `plt.imshow(...)`.

---

## Study plan

| Step | Do this | Time |
|------|---------|------|
| 1 | Read `notes/00_python_numpy_basics.md` (skip if you know NumPy arrays) | 20 min |
| 2 | Read Day 1 notes → run the 4 `day1_*` scripts → do their "TRY THIS" bits | 1 h |
| 3 | Read Day 2 notes → run the 3 `day2_*` scripts | 1 h |
| 4 | Read Day 3 notes → run `day3_01_convolution_by_hand.py` **slowly**, then the rest | 1.5 h |
| 5 | `practice/exercises.py`: try first, then check `solutions.py` | 1 h |
| 6 | `practice/quiz.md`: answer out loud before opening each answer | 45 min |
| 7 | Night before the eval: `cheat-sheet.md` | 15 min |

> Learning tip: after every script, change **one number** and predict what will happen *before* you run it. That habit teaches you more than reading.

---

## Mistakes in the lecture PDF (worth knowing for the eval)

The lecture code mostly works, but a few of its outputs look odd. Understanding why is a good test of whether you've really got it:

1. **"Mode: original" shows blue swatches instead of orange.** The BGR image went straight into matplotlib (which expects RGB), so red and blue swapped. → Day 1 notes.
2. **The HSV picture looks identical to the original.** The code converts to HSV and then straight back to RGB before showing it. → Day 1 notes.
3. **The perspective output is a plain grey square.** Its 4 corner points sit in the empty grey corner of a 5472×3648 photo. → Day 2 notes.
4. **The Day 2 code uses `rows, cols` without defining them.** You need `rows, cols = img.shape[:2]`. → Day 2 notes.
5. **The "Clipping input data" warning on Sobel** comes from showing a float, colour, possibly negative result directly. → Day 3 notes.
6. **The edge images look almost completely black.** 1-pixel-thin edges on a huge photo vanish when it's shrunk to fit the screen. → Day 3 notes.
7. **The object detector boxes dozens of noise specks.** It has no blur and no minimum-area filter. → Day 3 notes and `day3_05`.

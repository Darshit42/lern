# Original course files (Week 1)

The course's own materials, kept next to the study kit so everything for Week 1 is in one place.

| File | What it is | Study-kit notes |
|---|---|---|
| `lecture-pdfs/Days_1-3_Lectures.pdf` | Day 1 Intro, Day 2 Transformations, Day 3 Filtering | `notes/day1…`, `day2…`, `day3…` |
| `lecture-pdfs/Day_4_Histograms_and_Thresholding.pdf` | Day 4 | `notes/day4_histograms_and_thresholding.md` |
| `lecture-pdfs/Day_5_Morphological_Operations.pdf` | Day 5 | `notes/day5_morphological_operations.md` |
| `lecture-pdfs/Day_6_Contours_and_Shape_Analysis.pdf` | Day 6 | `notes/day6_contours_and_shapes.md` |
| `lecture-pdfs/Key_Concepts.pdf` | Summary of the week + advanced topics | `notes/key_concepts.md` |
| `notebooks/CV-Week1-Day1…6.ipynb` | The lab notebooks the PDFs were printed from | `code/day1_*` … `day6_*` |
| `notebooks/imagerepresentation.ipynb` | A 5×5 matrix shown as an image with its numbers | `code/day1_02_pixels_by_hand.py` |
| `notebooks/colorimgrepresent.ipynb` | A 3×3 RGB image with its numbers | `code/day1_02_pixels_by_hand.py` |
| `notebooks/Convolution.ipynb` | Convolution by hand + blur, sharpen and edge kernels | `code/day3_01_convolution_by_hand.py` |

## Running the notebooks

Open them in Jupyter or Google Colab with `sample.jpg` and `sample.png` in the same folder.

- **`sample.jpg`** is the course's Pantone photo (5472 × 3648), unchanged.
- **`sample.png` has been replaced.** In the original zip it was a 12 MB copy of the same Pantone photo. That's why the Day 6 notebook drew hundreds of tiny contours on noise instead of finding shapes. Here it's a small image of 8 black geometric shapes on white, which is what the Day 6 shape detector is meant to process.

Even with the shapes image, the Day 6 notebook uses `THRESH_BINARY`, which makes the white **paper** the object, so it finds only 1 contour (the page). Change it to `cv2.THRESH_BINARY_INV` to detect the shapes. See `notes/day6_contours_and_shapes.md`.

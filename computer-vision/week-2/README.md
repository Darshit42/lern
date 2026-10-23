# Computer Vision – Week 2 (start here)

Beginner-friendly study kit for **CV-Week-2-KeyConcepts.pdf** and the **Week 2 lab assignments** (Days 8–15): classical feature-based computer vision, classical machine learning, and the first CNN.

- **`notes/`**: plain-language explanations with diagrams, one chapter per Key Concepts topic. **Read these first.**
- **`labs/`**: every course lab notebook **worked through**. Each one:
  - keeps the course's code recognisable but **fixes its bugs**;
  - explains it line by line;
  - **runs it with outputs** (already executed: every plot and number is visible before you run anything);
  - does the exercise and the mini project;
  - ends with **viva questions and answers**.

| Day | Topic | Notes chapter | Lab notebook (course file) |
|---|---|---|---|
| 8 | Feature detection: Harris, SIFT, ORB · panorama stitching | `01_feature_detection_harris_sift_orb` | `day08_feature_detection_panorama` (Lecture1) |
| 9 | Image matching: descriptors, FLANN, ratio test · logo detector | `02_image_matching_flann_logo_detector` | `day09_image_matching_logo_detector` (Lecture2) |
| 10 | Object tracking: optical flow, KLT · dense-flow motion tracker | `03_object_tracking_optical_flow_klt` | `day10_object_tracking_optical_flow` (Lecture3) |
| 11 | ML in CV: kNN, SVM · digit recogniser | `04_ml_in_cv_knn_svm` | `day11_ml_knn_svm_digits` (Lecture4) |
| 12 | Bag of Visual Words · texture classifier | `05_bag_of_visual_words` | `day12_bag_of_visual_words_textures` (Lecture5) |
| 13 | Face detection: Haar cascades, HOG · face detector app | `06_face_detection_haar_hog` | `day13_face_detection_haar_hog` (Lecture6) |
| 14 | Pre-CNN module: maths, supervised vs unsupervised, the neuron, ANN | `07_pre_cnn_foundations` | `day14_pre_cnn_foundations` (no course file; built from the Key Concepts activities) |
| 15 | CNN basics: conv, pooling, stride, padding, flatten · MNIST CNN | `08_cnn_basics` | `day15_cnn_basics_mnist` (Lecture7) |

## What the labs found in the course notebooks (worth knowing before the viva)
- **None of the images or videos the course notebooks use were provided** (`Ganesh3.jpg`, `left.jpg`, `chaplin.mp4`, `basketball.jpg`, …). The labs download OpenCV's sample images and videos instead, or generate test data.
- **Day 8 panorama:** the course code warps the *left* image to negative x, so it is cut off and then covered by the right image. The "panorama" is just the right photo. Fixed and explained.
- **Day 9:** `for m, n in matches` crashes when FLANN returns fewer than 2 neighbours. Also, "> 10 good matches" falsely reported the logo in 10 of 10 logo-free scenes. Fixed with an inlier-based check.
- **Day 10:** the KLT tracker never re-detects lost points. `cv2.imshow` loops don't run inside notebooks.
- **Day 11:** `SVC(kernel='linear', gamma='scale')` (gamma does nothing for a linear kernel). The recogniser is tested on a *training* image.
- **Day 12:** the BoVW lab uses 3 images, one per class, so 0% accuracy is guaranteed. The lab rebuilds it on a real 7-class texture set: 79% with the course's settings, 97.5% tuned.
- **Day 13:** the HOG section has no code. The lab adds dlib's HOG face detector and OpenCV's HOG people detector. Note: in **OpenCV 5.0**, `CascadeClassifier` moved to `opencv-contrib-python` (OpenCV 4.x works as written).
- **Day 15:** the CNN code is correct. 1 epoch gave 98.56% here (the course got 98.79% with 2 epochs).
- Several textbook slogans were **measured**. Some held (SIFT is the most robust to scale and rotation). Some only partly held: "SIFT finds fewer keypoints" is only true if ORB's 500 limit is raised, and "HOG is slower than Haar" depends on the settings. The notes give the course's answer plus the measurement.

## Folder map
```
week-2/
├── README.md                 ← you are here
├── Week2_Study_Guide.pdf     ← notes + all 8 lab notebooks (with outputs) + cheat sheet, printable
├── cheat-sheet.md            ← one-page revision for the viva
├── notes/                    ← 01 … 08 plain-language chapters with diagrams
├── images/                   ← diagrams used by the notes
├── labs/                     ← day08 … day15 worked lab notebooks (data/ and output/ are created when run)
└── course-files/             ← the original Key Concepts PDF and the 7 course lab notebooks
```

## Setup (once)
```
pip install opencv-python numpy matplotlib scikit-learn scikit-image torch torchvision jupyter
jupyter notebook          # then open labs/day08_...
```
- Optional extra for Day 13 (HOG face detector): `pip install dlib-bin` (or `dlib`).
- Each notebook downloads the small images, videos and datasets it needs into `labs/data/` the first time it runs.
- **Webcam cells** (Day 13) are switched off (`RUN_WEBCAM = False`). Turn them on when running on your own computer, because webcams and `cv2.imshow` windows don't work in Colab.

## Study plan
| Step | Do this | Time |
|---|---|---|
| 1 | Notes 01–03, then labs Day 8–10 (classical features, matching, tracking) | 3 h |
| 2 | Notes 04–05, then labs Day 11–12 (kNN, SVM, BoVW) | 2 h |
| 3 | Note 06, then lab Day 13 (faces) | 1 h |
| 4 | Notes 07–08, then labs Day 14–15 (foundations → first CNN) | 2 h |
| 5 | Viva questions at the end of every lab notebook: answer out loud first | 1.5 h |
| 6 | `cheat-sheet.md` the night before | 20 min |

Week 3 (`../week-2-3/`) continues straight on from Day 15: CNN architectures, detection, segmentation, video and transformers.

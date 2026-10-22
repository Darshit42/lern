# Computer Vision Mid-Term Project (5th Semester)

Starter kits for the three project options in **CV_Project_Assignment.pdf** (copy in `course-files/`). Each option is one **complete, working Jupyter notebook**. It already ran end to end in *demo mode* on public sample data, so you can read every output before running anything.

> **Deadline in the brief: 20 October.** If that date has passed, check with your faculty. The brief asks students who expect difficulty to contact them early.

| # | Project | Notebook | What the demo shows |
|---|---|---|---|
| 1 | **Smart Society Security System** (face recognition) | `project1_smart_security_face_recognition.ipynb` | residents registered in SQLite; test photos → green **ENTRY ALLOWED** / red **ENTRY REJECTED**; accuracy, FAR/FRR, threshold trade-off plot |
| 2 | **Licence plates with OCR** (traffic violations) | `project2_license_plate_ocr_violations.ipynb` | YOLO vehicle detection; red-light rule with HSV signal reading; plate finding + EasyOCR + format correction; annotated demo video + violations CSV; accuracy report |
| 3 | **AI-Powered Attendance System** | `project3_attendance_system.ipynb` | students registered; classroom snapshots scanned (several faces per frame); attendance CSV + SQLite generated automatically; precision/recall vs ground truth |

## Which one should I pick?

| | Project 1 (security) | Project 2 (plates) | Project 3 (attendance) |
|---|---|---|---|
| Difficulty | ★★☆ | ★★★ | ★★☆ |
| Data you must collect | 20+ people × 3–5 photos + test photos | junction video(s) with readable plates | 10+ students × 3–5 photos + classroom photos/video |
| Hardest part | collecting 20 people's photos | getting footage where plates are legible; OCR errors | small faces at the back of the class |
| Course topics shown | embeddings, CNN detectors, databases | YOLO, HSV, thresholding, morphology, contours, tracking, OCR | embeddings, multi-face matching, Pandas/SQLite |

If you can easily photograph classmates, **Project 3** (only 10 people) or **Project 1** are the quickest. **Project 2** shows the most course techniques, but depends on getting good footage.

## Setup (once)

```
pip install opencv-python numpy pandas matplotlib jupyter
pip install ultralytics easyocr          # only for Project 2
jupyter notebook                         # then open the notebook
```
- Face models (about 39 MB) and YOLO weights (about 6 MB) download automatically on first run.
- Webcam cells run on your **own laptop** (set `RUN_WEBCAM = True`). They don't work in Google Colab.
- All models are **pretrained**, so no GPU is needed. Everything runs on a normal laptop CPU.

## From demo to your submission

1. Collect your data into `data/` as described at the top of each notebook.
2. Set `DEMO = False` and run all cells (*Kernel → Restart & Run All*).
3. Record a short **screen capture** of the webcam or video part working. It's the best proof of a "working prototype".
4. Write the report using the outline at the end of each notebook. The notebooks save the tables and plots you need in `report/` or `output/`.

## Report template (2–5 pages)

1. **Title, name, roll number, chosen project.**
2. **Objective:** one paragraph restating the problem.
3. **System design:** the pipeline diagram from the notebook, tools used (OpenCV YuNet + SFace, YOLOv8, EasyOCR, SQLite/Pandas) and why.
4. **Dataset:** how many people, photos or videos; how test data differs from registration data; consent.
5. **Implementation:** the main steps with 2–3 short code snippets and screenshots (the green/red screen, the attendance CSV, the annotated video frame).
6. **Results:** the accuracy tables printed by the evaluation section (accuracy, false positives, false negatives / precision, recall / OCR accuracy), plus 2–3 failure examples with an explanation.
7. **Limitations and future work:** from the notebook's final section, in your own words.
8. **References:** OpenCV Zoo (YuNet, SFace), Ultralytics YOLOv8, EasyOCR, and any dataset or footage source.

## Privacy and ethics (mention it in the report)

Faces and number plates are personal data. Collect photos **with consent**, store **embeddings rather than photos** where possible, don't publish other people's faces or plates without permission, and delete the data after evaluation. The `.gitignore` in this folder keeps `data/`, the databases and generated outputs **out of git** for this reason.

## About the demo data

To show working outputs, the notebooks download a few **public sample images that ship with open-source projects**: the `face_recognition` project's examples (US government portraits), OpenCV's and matplotlib's sample photos, and Ultralytics' test images. Each image's source URL is printed in the notebook. These downloads are not stored in the repository. Project 2's junction video is **synthetic**: the notebook draws it itself, so the ground truth is exactly known.

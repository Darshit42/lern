# Computer Vision – Weeks 2 & 3 (start here)

Beginner-friendly study kit for **CV-Week3-KeyConcepts.pdf** and the **Week 2 & 3 question bank**: the deep-learning half of the course.

- **`notes/`**: plain-language explanations with diagrams, one chapter per topic. **Read these first.**
- **`notebooks/`**: all the code, as **Jupyter notebooks** that have already been run, so every output, plot and number is visible before you run anything.

| Notes chapter | Pairs with notebook |
|---|---|
| `00_deep_learning_basics`: neurons, training loop, PyTorch in 5 minutes | 01 |
| `01_building_blocks`: FC vs conv, ReLU, dropout, pooling, vanishing gradient | 01 |
| `02_classic_cnns_lenet_alexnet_vgg` | 02 |
| `03_modern_cnns_resnet_efficientnet` | 03 |
| `04_object_detection_rcnn_yolo_ssd` | 04 |
| `05_segmentation_fcn_unet_mask_rcnn` | 05 |
| `06_video_rnn_lstm` | 06 |
| `07_attention_and_vision_transformers` | 07 |

| Notebook | Topic | In one line |
|---|---|---|
| `01_neural_network_building_blocks` | FC vs conv layers, ReLU, dropout, pooling, training | a CNN is Week-1 convolution with **learned** kernels |
| `02_classic_cnns_lenet_alexnet_vgg` | LeNet (1998), AlexNet (2012), VGG (2014) | proof of concept → breakthrough → depth & 3×3 filters |
| `03_modern_cnns_resnet_efficientnet` | ResNet (2015), EfficientNet (2019), MobileNet | skip connections make depth trainable; balanced scaling makes it efficient |
| `04_object_detection_rcnn_yolo_ssd` | R-CNN, YOLO, SSD, IoU, NMS | *what + where*: accuracy (R-CNN) vs speed (YOLO) vs balance (SSD) |
| `05_segmentation_fcn_unet_mask_rcnn` | FCN, U-Net, Mask R-CNN, RoI Align | a label for **every pixel**; encoder–decoder + skips; instance masks |
| `06_video_sequence_models_rnn_lstm` | RNN, LSTM, CNN + LSTM | CNN = *what* in each frame, LSTM = *how* it changes over time |
| `07_vision_transformers_attention` | patches, Q/K/V attention, ViT | an image as a sequence of patches; attention connects everything |

**Every notebook builds and runs something real on a laptop CPU** (each takes from under a minute to about 4 minutes):
- a LeNet that learns to read digits,
- a 52-layer plain network that fails to train next to a residual network that succeeds,
- tiny R-CNN and YOLO detectors raced against each other (YOLO over 100× faster on a laptop CPU),
- a U-Net with much sharper boundaries than an FCN,
- an LSTM that remembers what an RNN forgets,
- a CNN + LSTM that tells *left* from *right* motion when frame-only models can't,
- a tiny Vision Transformer.

The results are discussed **honestly**. Where a classic method wins on our small data (an SVM vs LeNet, a CNN vs a ViT), the notebook says so and explains why.

---

## Folder map
```
week-2-3/
├── README.md                     ← you are here
├── Week2-3_Study_Guide.pdf       ← notes + all 7 notebooks (text + outputs + plots) + answers, one printable PDF
├── cheat-sheet.md                ← revision sheet for the night before the eval
├── notes/                        ← 00 … 07 plain-language notes with diagrams (start here)
├── images/                       ← diagrams and result plots used by the notes
├── notebooks/                    ← 01 … 07 (read the text, run the cells, change the numbers)
├── practice/
│   ├── question_bank_answers.md  ← model answers to all 60 questions in the course's question bank
│   ├── practice_exercises.ipynb  ← 13 fill-in-the-blank coding exercises with PASS/FAIL checks
│   └── practice_solutions.ipynb  ← answers (all checks pass)
└── course-files/                 ← the original Key Concepts PDF, question bank and ViT notebooks
```

## Setup (once)
```
pip install torch torchvision scikit-learn matplotlib opencv-python jupyter
jupyter notebook          # then open notebooks/01_...
```
Or upload a notebook to **Google Colab** (everything is pre-installed). No GPU is needed. The datasets are generated in the notebooks or ship with scikit-learn, so nothing has to be downloaded. A few *optional* cells (pretrained Faster R-CNN / Mask R-CNN on your own photo) download weights only if you switch them on.

## Study plan
| Step | Do this | Time |
|---|---|---|
| 1 | Notes 00 + 01, then notebook 01: the building blocks everything else uses | 1.5 h |
| 2 | Notes 02 + 03, then notebooks 02 → 03: the CNN timeline (LeNet → AlexNet → VGG → ResNet → EfficientNet) | 2 h |
| 3 | Note 04, then notebook 04: detection | 1.25 h |
| 4 | Note 05, then notebook 05: segmentation + Mask R-CNN | 1.25 h |
| 5 | Note 06, then notebook 06: video, RNN and LSTM | 1 h |
| 6 | Note 07, then notebook 07: attention and Vision Transformers | 1 h |
| 7 | `practice/practice_exercises.ipynb` | 45 min |
| 8 | `practice/question_bank_answers.md`: answer out loud first, then check | 1.5 h |
| 9 | `cheat-sheet.md` the night before | 20 min |

> **Tip:** in every notebook, change one number (a learning rate, a depth, a threshold, a patch size) and predict the result before re-running the cell.

## The big picture in five lines
1. **CNNs** learn the kernels you hand-designed in Week 1 (notebook 01) and grew deeper and better: LeNet → AlexNet → VGG → **ResNet** (skip connections) → **EfficientNet** (balanced scaling).
2. **Detection** adds *where*: R-CNN (accurate, slow) → YOLO (one look, fast) → SSD (multi-scale).
3. **Segmentation** labels every pixel: FCN → **U-Net** (encoder–decoder + skips) → **Mask R-CNN** (a mask per object).
4. **Video** adds time: CNN per frame + **LSTM** across frames.
5. **Transformers** replace convolution with **attention over patches**: global context from layer 1.

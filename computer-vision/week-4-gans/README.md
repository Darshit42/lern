# Computer Vision – Week 4: GANs (start here)

Beginner-friendly study kit for the **GAN classes** (Line 3 · Applied Generative Vision):
- **Class 1:** GAN basics;
- **Classes 2 & 3:** training dynamics;
- **Class 4:** DCGAN and conditional GAN.

- **`notes/`**: plain-language chapters with diagrams, worked numbers, and exit-ticket questions with answers. **Read these first.**
- **`notebooks/`**: the course's three GAN notebooks **rebuilt from the course's own "notebook explained" guides** (the `.ipynb` files themselves weren't provided) and **run with outputs**. Every in-class task and exercise is done, and our numbers are compared with the guides'.

| Class | Notes chapter | Notebook (course file it rebuilds) |
|---|---|---|
| 1 · The GAN framework | `01_gan_basics` | `01_gan_basics_mnist` (`CV-Week4-Lecture1.ipynb`: MLP GAN on MNIST + Tasks 1–4) |
| 2 & 3 · Training dynamics | `02_gan_training_dynamics` | `02_gan_training_dynamics` (`GAN-Class-2-3-Training-Dynamics.ipynb`: x·y game, 8-cluster ring, five fixes, MNIST diagnostics) |
| 4 · DCGAN and conditional GAN | `03_dcgan_and_conditional_gan` | `03_dcgan_and_conditional_gan` (`GAN-Class-4-DCGAN-cGAN.ipynb`: DCGAN, cGAN, conditional accuracy, assignment option C) |

## What the rebuilt notebooks found
- **Class 1:**
  - The course's exact settings (Adam β1 = 0.9) **collapsed** in our run: every sample was one blob, read as "8".
  - Changing to **β1 = 0.5** (the course's experiment D) fixed it, giving all ten digits.
  - The losses looked normal either way, so always look at the images.
- **Classes 2 & 3:** these reproduced the guide closely.
  - **x·y game:** distances 3.83 / 1.36 (exact).
  - **Ring GAN** (modes covered out of 8): baseline 1/8; label smoothing 0/8; instance noise 1/8; TTUR 2/8; **minibatch std 8/8; z_dim 2 8/8**.
  - **MNIST:** Run A collapsed ("1,000 eights"), Run B made all ten digits.
- **Class 4:**
  - The DCGAN gave clean black backgrounds and evenness **0.922** (guide 0.929), against 0.816 for the MLP GAN.
  - The conditional GAN reached **84.4%** conditional accuracy (guide 84.4%).
  - Assignment C (an `nn.Embedding` instead of one-hot) reached 78.7%.

## Folder map
```
week-4-gans/
├── README.md                   ← you are here
├── Week4_GANs_Study_Guide.pdf  ← notes + all 3 notebooks (with outputs) + cheat sheet, printable
├── cheat-sheet.md              ← one-page revision
├── notes/                      ← 01 … 03 plain-language chapters
├── images/                     ← diagrams used by the notes
├── notebooks/                  ← 01 … 03 executed notebooks (MNIST downloads to notebooks/data/ when run)
└── course-files/               ← the original slides (.pptx) and "notebook explained" guides (.docx)
```

## Setup (once)
```
pip install torch torchvision matplotlib numpy jupyter
jupyter notebook        # then open notebooks/01_...
```
- MNIST (about 10 MB) downloads automatically the first time.
- On a laptop CPU, notebook 01 takes about 8 min, notebook 02 about 6 min, and notebook 03 about 15 min (three convolutional GANs). On a **Colab GPU** each takes a minute or two.
- GAN results depend on the random seed and the hardware, so compare **patterns** (collapse vs variety, which fix helped), not exact numbers.

## Study plan
| Step | Do this | Time |
|---|---|---|
| 1 | Note 01, then notebook 01 (do Tasks 1–4 yourself: predict before running) | 2 h |
| 2 | Note 02, then notebook 02 (fill in the Part 4 table before reading ours) | 2 h |
| 3 | Note 03, then notebook 03 | 2 h |
| 4 | Exit tickets at the end of each note: answer out loud | 45 min |
| 5 | `cheat-sheet.md` the night before | 20 min |

# Deep Learning & Generative AI (start here)

Beginner-friendly study kit for the **DL and GenAI** lecture handouts. It is kept separate from `../computer-vision/`.

- **`notes/`**: plain-language explanations with diagrams, worked examples, and check-yourself questions with answers. **Read these first.**
- **`notebooks/`**: code to **experiment** with, already run, so every plot and number is visible before you run anything. Each one tests the lecture's claims, and where a measurement disagrees with a slogan, the notebook says so and explains why.

| Lecture (date) | Notes chapter | Notebook |
|---|---|---|
| History of Deep Learning (04-08-2026) | `01_history_of_deep_learning` | `02_neurons_1943_to_1986` (1943 neuron, 1958 perceptron, 1969 XOR failure, 1986 backprop, all runnable) |
| AI vs ML vs DL vs GenAI & the Biological Neuron (05-08-2026) | `02_ai_ml_dl_genai_and_the_neuron` | `01_ai_ml_dl_genai_in_code` (one problem at all four levels) + `02_neurons_1943_to_1986` |
| Perceptron (06-08-2026) | `03_perceptron` | `03_perceptron_from_scratch` |
| Activation Functions (07-08-2026) + (Cont.) (11-08-2026) | `04_activation_functions` | `04_activation_functions` |
| Normalization and Regularization (04-09-2026) | `05_normalization_and_regularization` | `05_normalization_and_regularization` |

## What the experiments found (good for viva answers)
- **Both perceptron worked examples check out** in code (z = 1.4 → Pass; w → (0.4, −0.1), b → −0.9). One update brings z closer to 0 but **doesn't yet fix** the mistake.
- **XOR:** 200,000 random straight lines never beat 75%. A 2-4-1 network trained with backpropagation solves it.
- **Learning rate:** the classic perceptron started from zero weights needs **the same number of epochs for every η**, because the step function only sees signs. The "too small / too big" trade-off shows up clearly in gradient descent on a smooth loss.
- **Linear layers collapse:** a 4-layer linear network equals one layer exactly, and can't separate two moons.
- **Vanishing gradients:**
  - sigmoid loses the signal (10⁻²² at 30 layers) whatever the initialisation;
  - ReLU and tanh keep it only with **He / Xavier initialisation**;
  - with He init, ReLU was the best activation (96.9%), while sigmoid failed (27%).
- **Dying ReLU:**
  - a dead ReLU neuron gets exactly 0 gradient forever;
  - Leaky ReLU revives a single neuron but not a badly broken network;
  - **PReLU learned its way out** (10% → 97.8%).
- **Scaling:** unscaled features made gradient descent diverge or stall; standardised features converged in ≤ 200 steps.
- **BatchNorm:** a 10-layer sigmoid network went from **10% (stuck) to 97.5%**.
- **Regularisation on 200 images:**
  - L2 and dropout helped by about half a point;
  - L1 made 88% of the weights ≈ 0;
  - early stopping matched the accuracy at about 1/5 of the cost;
  - badly chosen augmentation **hurt**.
- **AI → ML → DL → GenAI on digits:**
  - rules 97% (0 vs 1 only);
  - hand-made features 74%;
  - raw-pixel logistic regression 97%;
  - deep network 98%;
  - a tiny generative model **draws new digits**.

## Folder map
```
deep-learning-genai/
├── README.md                      ← you are here
├── DL_GenAI_Study_Guide.pdf       ← notes + notebooks (with outputs) + cheat sheet, printable
├── cheat-sheet.md                 ← one-page revision
├── notes/                         ← 01 … 05 plain-language chapters
├── notebooks/                     ← 01 … 05 runnable experiments
├── images/                        ← diagrams and result plots used by the notes
└── course-files/                  ← the original lecture handouts (PDF)
```

## Setup (once)
```
pip install numpy matplotlib scikit-learn torch jupyter
jupyter notebook           # then open notebooks/01_...
```
Everything uses small built-in or generated data, so nothing is downloaded. Every notebook runs in **under about 2 minutes on a laptop CPU**. Google Colab works too.

## Study plan
| Step | Do this | Time |
|---|---|---|
| 1 | Note 01 (history) and note 02 (AI/ML/DL/GenAI + neuron) | 1 h |
| 2 | Notebooks 01 and 02 | 45 min |
| 3 | Note 03, then notebook 03 (perceptron): redo both worked examples by hand first | 1 h |
| 4 | Note 04, then notebook 04 (activations) | 1 h |
| 5 | Note 05, then notebook 05 (normalisation and regularisation) | 1.25 h |
| 6 | Check-yourself questions at the end of every note: answer out loud first | 1 h |
| 7 | `cheat-sheet.md` the night before | 20 min |

> **Tip:** in every notebook, change one number (a learning rate, an α, a dropout p, a λ) and **predict** the result before re-running.

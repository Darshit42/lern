# 03 · DCGAN and conditional GAN (GAN Class 4)

**Covers:** GAN Class 4 slides ("DCGAN and Conditional GAN") and the guide to the course notebook `GAN-Class-4-DCGAN-cGAN.ipynb`. It also covers the "later" cells of the Class 1 notebook (the DCGAN skeleton and the conditional-GAN assignment).

**Notebook:** `notebooks/03_dcgan_and_conditional_gan.ipynb` (the Class 4 notebook rebuilt and run)

> **The plan of Class 4:** the training game does **not** change (same losses, same D step then G step, Adam lr 0.0002, **β1 = 0.5**). We make two upgrades to the Class 1 GAN and measure each with the Class 2 & 3 diagnostics:
> 1. **DCGAN:** change **how G and D see images** (convolutions instead of Linear layers).
> 2. **Conditional GAN:** change **what G is told** (the digit label), so you can ask for a 7 and get a 7.

---

## 0. Revision of Classes 2 & 3 (the quiz the class starts with)

| Question | Answer |
|---|---|
| A GAN's Nash equilibrium? | p_g = p_data. D outputs ½ for everything (loss_d = 2 ln 2 ≈ 1.386, loss_g = ln 2 ≈ 0.693) |
| Why do gradient steps circle the x·y equilibrium? | each player's step depends on the other's position, so the combined motion rotates around (0, 0) |
| D(x) ≈ 1, D(G(z)) ≈ 0 for a long time? | D wins easily and is flat around the fakes, so G's gradient vanishes and G stops learning |
| Every sample looks real but all are the same digit? | **mode collapse**. Targeted fix: **minibatch std** (lets D see the lack of variety) |
| The three diagnostics? Which one alone can fool you? | fixed-noise grid, D(x) vs D(G(z)), digit histogram. **The histogram** alone fools you, because it labels junk as some digit |

**Diagnose-it checkpoint:**
- (a) Losses calm, D(x) ≈ 0.6, D(G(z)) ≈ 0.4, but 64 near-identical 1s → **collapsed**.
- (b) D(x) ≈ 0.99, D(G(z)) ≈ 0.01 for 2,000 steps, and the grid is unchanged → **D too strong**.
- (c) Both D scores swing between 0.2 and 0.9, and quality goes up and down → **unstable**: lower β1 or the learning rate.
- (d) D(x) ≈ 0.85, D(G(z)) ≈ 0.2, varied digits getting sharper, all ten in the histogram → **healthy**: train longer.

A new number used in Class 4: **evenness** (0 to 1) summarises the digit histogram. 1.0 means all ten digits appear equally often.

---

## 1. Why convolutions? (the baseline's problems)

The Class 1 **fully connected** generator (trained 2 epochs, β1 = 0.5), in the course's run:
- **Background:** grey speckle, never clean black. Each of the 784 outputs is learned **on its own**, so no pixel knows its neighbours are black.
- **Strokes:** blurry and broken. A stroke is a pattern of **neighbouring** pixels, and a Linear layer has no idea which pixels are neighbours.
- **Variety:** **no 6s at all** in 1,000 samples, with 8s and 3s dominating. **Evenness 0.79**.

**Two assumptions a convolution makes, which fit images** (you met them in Week 2–3 CNN classifiers):

| Assumption | What it means | Effect in a GAN |
|---|---|---|
| **Locality** | each output pixel is computed from a small neighbourhood (a 4×4 window), not all 784 pixels | neighbours decide together → **smooth backgrounds** |
| **Weight sharing** | the same filter slides over the whole image | a "curve" detector learned in one corner works everywhere → a stroke is learned **once** |

**Size is not the point** (parameter counts printed by the notebook):

| | Fully connected | DCGAN |
|---|---|---|
| Generator | 227,344 | 759,680 (3.3× larger) |
| Discriminator | 201,217 | **138,624 (31% smaller)** |

The convolutional D is **smaller yet a much better critic**, because each filter is reused across the image. A better critic is a better teacher: G only learns through D's gradient.

---

## 2. DCGAN: Deep Convolutional GAN (Radford, Metz & Chintala, 2015)

### Shrink in D, grow in G: the two size formulas

| Layer | Output size | With k = 4, s = 2, p = 1 |
|---|---|---|
| **Conv2d** (in D) | **(n + 2p − k) // s + 1** | halves: 28 → 14 → 7 |
| **ConvTranspose2d** (in G) | **(n − 1)·s − 2p + k** | doubles: 7 → 14 → 28 |

The first G layer is `ConvTranspose2d(100, 128, 7, 1, 0)` on a 1×1 input: (1 − 1)·1 − 0 + 7 = **7×7**. That's how a noise vector becomes a 7×7 feature map.

### What a transposed convolution does: "stamping"
Each input pixel **stamps a copy of the kernel** onto a bigger output, 2 pixels apart (stride 2). Where stamps overlap they **add**. The notebook's demo uses an all-ones 4×4 kernel (stride 2, padding 1) on the 2×2 input [[1, 0], [0, 2]]:

```
[[1, 1, 1, 0],
 [1, 3, 3, 2],
 [1, 3, 3, 2],
 [0, 2, 2, 2]]      ← the centre is 1 + 2 = 3, because the two stamps overlap there
```

In a real G the kernels are **learned**, so each feature stamps a learned patch of strokes.
**Checkerboard artifacts:** if the kernel size isn't divisible by the stride, some pixels get more overlaps than others and a faint grid appears (Odena et al., 2016). **Kernel 4 with stride 2 overlaps evenly.**

### The course's DCGAN for 28×28 MNIST

| Network | Layer | Output shape |
|---|---|---|
| G | input z reshaped with `z.view(B, -1, 1, 1)` | [B, 100, 1, 1] |
| G | `ConvTranspose2d(100, 128, 7, 1, 0)` + BatchNorm + ReLU | [B, 128, 7, 7] |
| G | `ConvTranspose2d(128, 64, 4, 2, 1)` + BatchNorm + ReLU | [B, 64, 14, 14] |
| G | `ConvTranspose2d(64, 1, 4, 2, 1)` + **Tanh** | [B, 1, 28, 28] |
| D | `Conv2d(1, 64, 4, 2, 1)` + LeakyReLU(0.2) (no BN on D's input) | [B, 64, 14, 14] |
| D | `Conv2d(64, 128, 4, 2, 1)` + BatchNorm + LeakyReLU(0.2) | [B, 128, 7, 7] |
| D | `Conv2d(128, 1, 7, 1, 0)`, then `.view(-1, 1)` | [B, 1]: one raw score |

- In G the image grows while the channel count shrinks, and D is the mirror image.
- `bias=False` before BatchNorm, because BN has its own shift.
- D ends with a **raw score** (no Sigmoid), because training uses `BCEWithLogitsLoss` (Sigmoid + BCE in one numerically stable step).
- `init_weights`: conv weights from N(0, 0.02), BatchNorm scales from N(1, 0.02), as in the paper.

### The DCGAN recipe (seven rules, and why)

| Rule | Why it helps |
|---|---|
| strided convolutions, **no pooling** | the network learns how to shrink and grow |
| **BatchNorm** in G and D | stable activations, helps stop G collapsing to one output |
| **no** BN on G's output or D's input | BN on every layer made samples oscillate |
| ReLU in G, **Tanh** output | output in [−1, 1], the range of the normalised images |
| **LeakyReLU(0.2)** in D | gradients still flow for negative inputs |
| Adam lr 0.0002, **β1 = 0.5** | less momentum in a moving game (Class 2 & 3) |
| weights from N(0, 0.02) | a small, standard start |
| (also) no fully connected hidden layers | |

### Results in the course's run (2 epochs, about 2.5 min on CPU)

| | Fully connected | DCGAN |
|---|---|---|
| Look | grey speckle, blurry | **black backgrounds, solid connected strokes** (some shapes still malformed: 2 epochs is short, real runs train 25+) |
| Evenness | 0.79 | **0.93** |
| Rarest digit | 6: none at all | 1: 12 of 1,000 |
| D(x) / D(G(z)) | ≈ 0.7 / 0.4 | ≈ 0.9 / 0.15: D is clearly ahead, but D(G(z)) isn't stuck at 0 and the samples improve → **healthy** (case d) |

**Lesson:** the architecture changed **what kind of mistakes** G makes. It is undertrained, not broken. Our rebuilt notebook reruns all of this, and your numbers will differ slightly.

### The Class 1 skeleton (CelebA faces, 32×32 colour)
`ConvTranspose2d(100, 512, 4, 1, 0)` → 4×4 → 8×8 → 16×16 → **[B, 3, 32, 32]**. The calculation: (1 − 1)·1 − 0 + 4 = 4, then (4 − 1)·2 − 2 + 4 = 8, and so on. One more `ConvTranspose2d(…, 4, 2, 1)` gives **64×64**, the standard DCGAN face size.

---

## 3. Conditional GAN: choose the digit (Mirza & Osindero, 2014)

### The problem
```python
for real, _ in train_loader:     # Classes 1, 2 and 3: the _ is the digit label, thrown away!
```
An unconditional G(z) gives **some digit chosen by chance**. To get a 7 you'd have to sample, classify and throw away nine out of ten, and if G has partly collapsed, a rare digit may **never** come out.
**Goal:** G(z, y) = "draw digit **y**, in the style **z**".

### The idea: give the label y to BOTH players

**min_G max_D E[log D(x, y)] + E[log(1 − D(G(z, y), y))]**

- **G(z, y):** "draw a digit y, in style z".
- **D(x, y):** "is x a **real image of digit y**?" A fake can now fail in **two ways**: it looks fake, **or** it is the wrong digit.
- **Why D must see the label too:** if only G had it, D would accept a perfect 3 labelled "7". G would get no signal to obey y and would learn to **ignore** it.

### How a network "reads" a label: one-hot, concatenated
- **One-hot:** 7 → [0,0,0,0,0,0,0,1,0,0].
- **Into G:** glue it to z, so G's first layer gets **110 input channels** (100 noise + 10 label): `torch.cat([z, one_hot(y, 10)], dim=1)`.
- **Into D:** spread the one-hot over **10 image-sized maps** (28×28 each, the true label's map all ones, the rest all zeros) and stack them on the image, giving **11 input channels**: `torch.cat([x, y_maps], dim=1)`.
- Everything else is the DCGAN unchanged (`CondDCGenerator`, `CondDCDiscriminator`).
- In training, G draws the **real batch's labels** and D scores each fake **with the same label G was given**. Scoring with a different random label would let D learn the shortcut "mismatch = fake", and G's gradient would push it towards the wrong label.

### Results in the course's run (2 epochs, about 3.3 min on CPU)
- A 10×10 grid where **row r asks for digit r** shows the requested digit in almost every cell, with **varied handwriting within each row** (that variety comes from z). Shapes are cleaner than the unconditional DCGAN's, because knowing the digit makes G's job easier.
- **Conditional accuracy:** generate 100 samples per digit and let the notebook's CNN digit counter (98.1% test accuracy) read them. **84.4%** were the digit asked for.

| Digit | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| Accuracy (%) | 97 | 92 | 84 | 82 | 73 | 88 | 82 | 93 | 84 | 69 |

- The largest confusions are between digits that share strokes: asked for 9 and saw a 7 (13×), 2 → 3 (12×), 3 → 5 (11×), 4 → 9 (10×). Part of the 15.6% error is the classifier's own mistakes on unusual handwriting.
- **Mismatch test:** D scored 200 generated 7s at **0.315 with label 7** but **0.141 with label 1**. D has learned to check the image–label **match**, which is the signal that teaches G to obey y.

### Content vs style
- **The label picks the digit (content). The noise picks the handwriting (style).**
- **Style vs content grid:** one fixed z per row and labels 0–9 across the columns. Thickness, slant and stroke style stay similar across the row while the digit changes.
- **Interpolation:** fix the label and walk z from z_a to z_b in small steps. The handwriting changes **smoothly**, so G has learned a continuous **style space** (a theme of the next topic, VAEs).

### True or false (slides)
- "Only the generator needs the label." **False:** D needs it to punish the wrong digit.
- "A cGAN can't suffer mode collapse." **False:** it can collapse **within** a class (every 7 identical). A grid row shows it, while conditional accuracy does not.
- "100% conditional accuracy proves the cGAN is perfect." **False:** it measures **control**, not variety or quality.

### The same idea with bigger conditions

| Condition y | Example | Model |
|---|---|---|
| class label | "draw a 7" | cGAN, AC-GAN |
| an image | sketch → photo, map → satellite | pix2pix (paired images) |
| another domain | horse ↔ zebra, no pairs | CycleGAN |
| text | "a small yellow bird" | text-to-image GANs, later diffusion (Stable Diffusion's prompt is a condition too) |

The pattern is always the same: **encode y as numbers and give it to both G and D**, so the critic can check the match.

---

## 4. Assignment (choose one, from the course)

| Option | Task |
|---|---|
| A · Fashion-MNIST | train the DCGAN on `datasets.FashionMNIST` (same 28×28) and submit a fixed-noise grid per epoch with a note |
| B · DCGAN faces | extend the Class 1 skeleton to 64×64, write a mirror D, train on CelebA (GPU recommended) |
| C · Embedding | replace the one-hot in `CondDCGenerator` with `nn.Embedding(10, 50)` and compare conditional accuracy |

**Next topic:** autoencoders and VAEs (encoder, bottleneck, decoder, reconstruction loss).

---

## Exit ticket / check yourself

1. Why does a fully connected generator give speckled backgrounds while a DCGAN gives clean ones?
2. ConvTranspose2d(k = 4, s = 2, p = 1) on 16×16? Conv2d(k = 4, s = 2, p = 1) applied three times to 64×64?
3. What does `ConvTranspose2d(100, 128, 7, 1, 0)` do to a 1×1 input?
4. Why no BatchNorm on G's last layer? Which DCGAN rule did we already use in Class 2 & 3?
5. The DCGAN's D has fewer parameters than the MLP's D. Why is it a better critic?
6. Why must D see the label in a cGAN? How is the label given to D in the course's notebook?
7. What does conditional accuracy measure, and what does it miss?
8. In a cGAN, what does the label control and what does the noise control?

<details><summary>Answers</summary>

1. A fully connected G learns each of the 784 outputs separately, so neighbouring pixels never agree. Convolutions compute each output from a small neighbourhood with shared filters, so backgrounds come out smooth and strokes connected.
2. (16 − 1)·2 − 2 + 4 = 32×32. 64 → 32 → 16 → 8.
3. It makes it 7×7 with 128 channels: (1 − 1)·1 − 0 + 7 = 7.
4. G's output must match the real pixel values, and BN there caused instability. β1 = 0.5 (less momentum in a moving game).
5. Each small filter is reused across the whole image (weight sharing and locality), so it gets more useful features from fewer weights, and G learns from a better teacher.
6. Otherwise D accepts realistic images of the wrong digit and G learns to ignore the label. The one-hot is spread over 10 image-sized maps stacked on the image (11 input channels).
7. Whether G draws the requested digit (control). It misses variety and quality, e.g. collapse within a class.
8. The label controls the content (which digit). The noise controls the style (thickness, slant, loops).
</details>

# 03 · DCGAN and conditional GAN (the Class 1 mini project and assignment)

**Covers:** the "later" cells of `CV-Week4-Lecture1.ipynb`: the DCGAN mini project (face generator on CelebA, generator skeleton only) and the conditional-GAN assignment. The Class 2 & 3 slides say these are the topic of the next class.

**Notebook:** `notebooks/03_dcgan_and_conditional_gan.ipynb` (a small DCGAN and a conditional GAN, both trained on MNIST)

---

## 1. DCGAN: Deep Convolutional GAN (Radford, Metz & Chintala, 2015)

The Class 1 GAN uses **fully connected (Linear) layers**. As Week 2–3 showed, dense layers flatten images, lose their spatial structure and need many weights. **DCGAN** replaces them with **convolutions**, which suit images much better.

| | MLP GAN (Class 1) | DCGAN |
|---|---|---|
| Generator | Linear 100 → 256 → 784 | **transposed convolutions** grow a 1×1 "image" to full size |
| Discriminator | Linear 784 → 256 → 1 | **strided convolutions** shrink the image to one score (a CNN classifier) |
| Normalisation | none | **BatchNorm** in both (not on G's output or D's input) |
| Activations | ReLU / LeakyReLU | ReLU in G (Tanh at the output), LeakyReLU(0.2) in D |
| Pooling | — | none: **stride-2** convolutions do the resizing |
| Optimiser | Adam lr 2e-4 | Adam lr 2e-4, **β1 = 0.5** (see chapter 02) |
| Result | blotchy digits | much sharper images |

### The course's DCGAN generator skeleton
```python
class DCGANGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False), nn.BatchNorm2d(512), nn.ReLU(True),  # 1×1   → 4×4
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(True),  # 4×4   → 8×8
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True),  # 8×8   → 16×16
            nn.ConvTranspose2d(128, 3, 4, 2, 1, bias=False),   nn.Tanh())                           # 16×16 → 32×32, 3 colours
    def forward(self, x):            # x must be shaped (batch, 100, 1, 1), not (batch, 100)
        return self.net(x)
```

**Transposed-convolution output size:** **out = (in − 1)·stride − 2·padding + kernel**
- (1 − 1)·1 − 0 + 4 = **4**
- (4 − 1)·2 − 2 + 4 = **8**
- (8 − 1)·2 − 2 + 4 = **16**
- (16 − 1)·2 − 2 + 4 = **32**

It is the mirror of Week 2's conv formula: transposed convolutions **grow** images, as in U-Net's decoder (Week 3).

**To finish the course's mini project you'd still need:**
- a convolutional discriminator (the mirror image: 32 → 16 → 8 → 4 → 1 score);
- a CelebA data loader (celebrity faces, usually resized to **64×64**, which needs **one more** upsampling layer);
- the same two-step training loop as Class 1.

CelebA isn't downloadable in our build environment, so notebook 03 trains a DCGAN on MNIST. The code is the same, and you can swap in CelebA on your own machine or on Colab.

---

## 2. Conditional GAN (cGAN): telling G which digit to draw

The Class 1 GAN is **unconditional**: you can't ask it for a "7". You get whatever digit the noise happens to produce.

> A **conditional GAN** (Mirza & Osindero, 2014) gives **both** networks the **label** as an extra input.
> - **G(z, y):** "draw a digit of class y" from noise z.
> - **D(x, y):** "is this a **real image of class y**?"

**How the label goes in** (the common simple way):
- **Embed** the label (e.g. `nn.Embedding(10, 10)`, a learned vector for each digit, or a one-hot vector).
- For **G:** concatenate it with z, so 100 + 10 numbers go in.
- For **D:** concatenate it with the flattened image, so 784 + 10 numbers go in.

```python
class CondGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(10, 10)
        self.fc = nn.Sequential(nn.Linear(100 + 10, 256), nn.ReLU(), nn.Linear(256, 784), nn.Tanh())
    def forward(self, z, y):
        return self.fc(torch.cat([z, self.label_emb(y)], dim=1)).view(-1, 1, 28, 28)
```

**What changes in training:**
- Real images come **with their true labels** (MNIST's labels, which the unconditional GAN threw away with `for real, _ in train_loader`).
- Fakes are generated for **random requested labels**, and D sees each fake **together with the label it was asked for**.
- So D can now reject a fake for being **unrealistic** *or* for being **the wrong digit** ("you asked for a 7 and this looks like a 1"). G is pushed to draw the digit it was asked for.

**Checking that it works:** generate a 10 × 10 grid where **row = requested digit**, then let a separately trained classifier read the fakes. "Asked for a 7, got a 7" accuracy measures how well G obeys. Notebook 03 does exactly this.

**Bonus:** conditioning also helps against **mode collapse** (chapter 02), because you explicitly ask for every class.

**Where conditioning leads:**
- **pix2pix:** condition on a whole image (sketch → photo);
- **text-to-image:** condition on a sentence;
- **StyleGAN:** control over style;
- modern diffusion models are also conditional (on text prompts).

---

## Check yourself

1. What does DCGAN replace in the MLP GAN, and why does that help?
2. Walk the DCGAN generator's sizes from 1×1 to 32×32 with the transposed-conv formula.
3. What shape must the noise have for `DCGANGenerator`, and why?
4. Which Adam setting does DCGAN recommend?
5. What extra input do G and D get in a conditional GAN? What new reason can D now have for rejecting a fake?
6. How would you measure whether a cGAN really draws the digit you asked for?

<details><summary>Answers</summary>

1. Linear layers are replaced with (transposed) convolutions plus BatchNorm. Convolutions keep spatial structure and share weights, so the images are sharper with fewer parameters.
2. out = (in − 1)·s − 2p + k: 1 → 4 (k4, s1, p0) → 8 → 16 → 32 (k4, s2, p1).
3. (batch, 100, 1, 1): convolution layers expect channels × height × width, so the noise is a 100-channel 1×1 "image".
4. lr = 0.0002 and β1 = 0.5.
5. The class label (embedded or one-hot). D can reject a fake for not matching the requested class, even if it looks realistic.
6. Generate many samples per requested class and check them with an independent classifier (the % where the predicted class equals the requested class), and look at the grid.
</details>

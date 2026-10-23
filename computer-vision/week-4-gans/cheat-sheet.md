# CV Week 4 · GANs Cheat Sheet

## Class 1 · The framework
- **GAN = Generative Adversarial Network(s)** (Goodfellow et al., 2014). **Generative** = creates data · **Adversarial** = two opponents with opposite goals · **Networks** = both are neural nets.
- **Generator G** (forger/counterfeiter): noise **z** (100 numbers, `torch.randn`) → image. **Never sees real data**; learns only through D's gradient. **Tanh** output (matches Normalize(0.5, 0.5) → −1…1). Kept after training.
- **Discriminator D** (inspector/police): image → **P(real)** via **Sigmoid**. LeakyReLU(0.2). Labels for free: real = 1, fake = 0. Usually discarded.
- Notebook nets: G 100 → 256 → 784 (Tanh), D 784 → 256 → 1 (Sigmoid), `BCELoss`, Adam lr 2e-4, batch 64, 2 epochs.
- **Step 1 (D):** real → 1, `fake.detach()` → 0, `loss_d = loss_real + loss_fake`, only `optimizer_d.step()`.
- **Step 2 (G):** `D(fake)` (no detach) vs **labels_real** = the **non-saturating** loss −log D(G(z)), strong gradients early. Only `optimizer_g.step()`.
- **BCE(p, y) = −[y log p + (1 − y) log(1 − p)]** · examples: (0.9, 1) → 0.105 · (0.1, 1) → 2.303 · (0.2, 0) → 0.223 · G on the same fake → 1.609.
- **Minimax:** min_G max_D E[log D(x)] + E[log(1 − D(G(z)))] · zero-sum.
- **Optimal D*** = p_data/(p_data + p_g) · ideal: **p_g = p_data, D = ½**, loss_d = 2 ln 2 ≈ **1.386**, loss_g = ln 2 ≈ **0.693**.
- **Losses don't measure quality: look at the images.** Fixed-noise grids show learning; interpolating z shows a smooth latent space.

## Classes 2 & 3 · Training dynamics
- A GAN chases a **moving target**: a **saddle point / equilibrium**, not a minimum.
- **Nash equilibrium:** no player gains by changing alone. GAN's: **p_g = p_data, D = ½**. It exists, but training needn't reach it.
- **x·y game:** x ← x − lr·y, y ← y + lr·x → simultaneous **spirals out** (1.41 → 3.83), alternating **orbits**. A smaller lr only slows the circling.
- **Instability:** oscillating losses · quality going backwards · one player wins (D too strong → G's gradient vanishes) · fragile settings. Momentum overshoots → **Adam β1 = 0.5** (DCGAN).
- **Who is winning:** D(x) ≈ 1 and D(G(z)) ≈ 0 = D too strong · 0.6–0.8 / 0.2–0.4 = healthy · ≈ 0.5 / 0.5 = can't tell (check the samples) · swinging = oscillation.
- **Mode collapse:** few modes (e.g. one digit); **hopping** = the collapsed output jumps around. Cause: D judges one sample at a time. Detect with 64+ sample grids, fixed-noise grids, class histograms (read with the grid), coverage.
- **Toolbox:** non-saturating loss (early gradients) · β1 0.5 (oscillation) · label smoothing 0.9 and instance noise (over-confident D) · TTUR (D lagging) · **minibatch std/discrimination (collapse)** · spectral norm · WGAN-GP (distance-based critic).
- **Ring results:** only **minibatch std** and **z_dim = 2** fixed the collapse (8/8). **Match the fix to the cause.**

## Class 4 · DCGAN & conditional GAN
- **Why convolutions:** locality + weight sharing → smooth backgrounds, connected strokes. Conv D smaller (139k vs 201k) but a better critic.
- **Sizes:** Conv2d **(n + 2p − k)//s + 1** (k4 s2 p1 halves) · ConvTranspose2d **(n − 1)s − 2p + k** (k4 s2 p1 doubles) · ConvT(100, 128, 7, 1, 0): 1×1 → **7×7**.
- **Stamp demo:** each input pixel stamps the kernel, overlaps add ([[1, 0], [0, 2]] → centre 3) · uneven overlap → **checkerboard artifacts** (k4 s2 is even).
- **DCGAN 28×28:** G 100×1×1 → 128×7×7 → 64×14×14 → 1×28×28 (Tanh) · D the mirror → 1 raw score (`BCEWithLogitsLoss`).
- **Rules:** strided convs, no pooling · BatchNorm in G and D, but **not** on G's output or D's input · ReLU in G / Tanh output · LeakyReLU(0.2) in D · Adam 2e-4, β1 0.5 · weights from N(0, 0.02).
- **cGAN** (Mirza & Osindero 2014): min_G max_D E[log D(x, y)] + E[log(1 − D(G(z, y), y))]. **The label goes to BOTH players**, otherwise G ignores it. G: z + one-hot (110 channels). D: image + 10 label maps (11 channels).
- **Conditional accuracy** = share of fakes classified as the requested digit (84.4%). It measures **control**, not variety (a cGAN can still collapse within a class).
- **Label = content (which digit), noise = style (handwriting).** Conditions can be labels, images (pix2pix), domains (CycleGAN) or text (text-to-image).

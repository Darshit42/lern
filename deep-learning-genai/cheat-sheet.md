# DL & GenAI · Cheat Sheet (lectures 04-08 → 04-09 + tokens handout)

## History (04-08)
| Year | What | Who |
|---|---|---|
| 1943 | first artificial neuron (logic with binary inputs/outputs) | McCulloch & Pitts |
| 1958 | perceptron: first **trainable** network (linearly separable only) | Rosenblatt |
| 1969 | *Perceptrons*: single-layer limits (XOR) → **AI winter** | Minsky & Papert |
| 1986 | **backpropagation** popularised → multi-layer training | Rumelhart, Hinton, Williams |
| 1998 | **LeNet-5** CNN: digits, cheques, postcodes | LeCun |
| 2006 | **Deep Belief Networks**, layer-wise pre-training → modern DL | Hinton |
| 2012 | **AlexNet** wins ImageNet: **GPUs, ReLU, Dropout** | Krizhevsky, Sutskever, Hinton |
| 2014 | **GANs** (generator vs discriminator) | Goodfellow |
| 2017 | **Transformer**, "Attention Is All You Need" | Google |
| 2020+ | foundation models & GenAI: LLMs, multimodal, diffusion (GPT, Llama, Gemini, Claude, Stable Diffusion) | many |
- **Success factors:** big datasets · GPUs/TPUs · better algorithms (backprop, ReLU, BatchNorm, Dropout, attention) · open-source frameworks (TensorFlow, PyTorch, Keras) · cloud.
- **Model evolution:** neuron → perceptron → MLP → CNN → RNN → LSTM → Transformer → foundation models.
- **"Deep"** = number of **hidden layers** (not deep thinking). Layers learn edges → corners/textures → shapes → parts → object, **automatically**.

## AI ⊃ ML ⊃ DL ⊃ GenAI (05-08)
| | AI | ML | DL | GenAI |
|---|---|---|---|---|
| idea | mimic intelligent behaviour | learn from data | multi-layer neural nets | generate new content |
| features | n/a | **manual** | **automatic** | automatic, massive scale |
| data | varies | moderate (1000s) | large (100k+) | massive (web-scale) |
| architecture | rules, search… | trees, SVM, linear | MLP, CNN, RNN/LSTM | **Transformers**, diffusion |
| example | rule-based chess engine | spam filter | CNN on X-rays | LLM writing an essay |
- Three distinguishing factors: **data, feature engineering, architecture**.
- Misconceptions: AI ≠ ML · DL isn't "just more layers" (automatic features) · GenAI *is* scaled DL · bigger isn't always better.

## Biological → artificial neuron
| dendrites | synapse strength | soma | firing threshold | axon | plasticity |
|---|---|---|---|---|---|
| inputs xᵢ | weights wᵢ | Σwᵢxᵢ + b | activation f(z) | output | weight updates (backprop) |
- Brain: ~86 billion neurons, ~100 trillion synapses, ~20 W. Artificial: numbers, synchronous steps, exact learning rule, a **drastic simplification** by design.

## Perceptron (06-08)
- **z = Σwᵢxᵢ + b**, **ŷ = step(z)** (1 if z ≥ 0). Worked example: x = (2, 3), w = (0.5, 0.8), b = −2 → z = **1.4** → Pass.
- **Update:** wᵢ ← wᵢ + η(y − ŷ)xᵢ, b ← b + η(y − ŷ), error ∈ {−1, 0, +1}. Worked example: w (0.2, −0.4), b −1, y 1, ŷ 0, η 0.1 → **w (0.4, −0.1), b −0.9**.
- η too small = slow, too big = overshoot/diverge (gradient descent). Plain perceptron from zero weights: η doesn't change the epoch count.
- **Linearly separable only:** AND ✓, OR ✓, **XOR ✗** (best single line 75%) → MLP with a hidden layer (+ backprop) solves it.

## Activation functions (07-08, 11-08)
| f | formula | range | slope | use |
|---|---|---|---|---|
| Step | 1 if z ≥ 0 else 0 | {0, 1} | 0 (no gradient) | classic perceptron |
| Linear | z | (−∞, ∞) | 1 | **regression output** |
| Sigmoid | 1/(1 + e⁻ᶻ) | (0, 1) | σ(1 − σ) ≤ **0.25** | **binary output**, gates |
| tanh | (eᶻ − e⁻ᶻ)/(eᶻ + e⁻ᶻ) | (−1, 1), **zero-centred** | 1 − tanh² ≤ 1 | RNN/LSTM states |
| ReLU | max(0, z) | [0, ∞) | 1 / 0 | **hidden layers** (+ He init) |
| Leaky ReLU | z or αz (α fixed, e.g. 0.01) | (−∞, ∞) | 1 / α | dying-ReLU concern |
| PReLU | z or az (**a learned**) | (−∞, ∞) | 1 / a | CNNs, adaptive slope |
- No activation → stacked linear layers = **one linear layer**.
- **Vanishing gradient:** small slopes (sigmoid/tanh saturation) multiplied over layers → early layers stop learning.
- **Dying ReLU:** slope **exactly 0** for negative input → the neuron never updates. Leaky/PReLU keep a non-zero slope.

## More activations (13-08)
| f | formula | note |
|---|---|---|
| ELU | z, or α(eᶻ − 1) for z < 0 | smooth, bounded below by −α |
| GELU | z·Φ(z) | soft gate; **Transformers** (BERT/GPT); min ≈ −0.17 |
| Swish/SiLU | z·σ(z) | smooth, non-monotonic; min ≈ −0.28 |
| Softmax | e^{zᵢ}/Σe^{zⱼ} on a vector | sums to 1; **single-label multi-class output**; subtract max(z) for stability |
- (2.5, 1.2, 0.3) → **0.723 / 0.197 / 0.080** (handout says 0.07: rounding slip). Multi-label → **independent sigmoids**, not softmax.
- **Output:** binary → sigmoid · multi-class → softmax · regression → linear · positive → softplus · bounded → scaled sigmoid/tanh.
- **Hidden:** ReLU default · dying → Leaky/PReLU · Transformer → GELU · bounded zero-centred → tanh · **validate, don't pick "newest"**.

## Loss functions (18-08)
| problem | loss | output | labels |
|---|---|---|---|
| regression | **MSE** (1/n)Σ(y − ŷ)² | linear | numbers |
| regression + outliers | **MAE** (1/n)Σ\|y − ŷ\| | linear | numbers |
| binary | **BCE** −[y log p + (1 − y) log(1 − p)] | sigmoid | 0/1 |
| multi-class | **CCE** −Σ yₖ log pₖ = −log p_correct | softmax | one-hot |
| multi-class | **sparse CCE** (same loss) | softmax | integer |
- Example (100, 200, 300) vs (90, 220, 270): **MSE 466.67, MAE 20**. Errors 2 vs 10: ×5 → ×25 under MSE.
- BCE for y = 1: p 0.99 → 0.010, 0.1 → 2.303, 0.01 → 4.605 (confident and wrong = huge). CCE p = 0.82 → 0.198.
- Why not MSE for classification: saturated sigmoid → tiny gradient (−0.013 vs BCE −0.99 at logit −5). BCE + sigmoid gradient = **p − y**.
- **"Sparse" = the label format**, not the predictions. PyTorch `CrossEntropyLoss` takes **logits** (don't add Softmax).

## Working of an MLP (20-08)
- input → **Z = XW + b** → **A = f(Z)** → … → ŷ → **loss** → **backprop** (chain rule) → **W ← W − η·∂L/∂W** → repeat.
- Softmax + CE backprop: **δ² = Ŷ − Y**, ∂W² = A¹ᵀδ², δ¹ = (δ²W²ᵀ) ⊙ f′(Z¹), ∂W¹ = Xᵀδ¹ (verified: numerical 10⁻¹¹, autograd 10⁻¹⁷).
- Examples: z = 2(0.5) + 3(0.2) + 1(−0.4) + 0.1 = **1.3**; Y [1, 0], Ŷ [0.7, 0.3] → L = **0.357**; w 2 − 0.1·0.5 = **1.95**.
- **Batch** = group of examples · **step** = one batch → one update · **epoch** = full pass. 50,000 / 128 → 391 steps/epoch; 50,000 / 100 × 10 epochs = **5,000 updates**.
- Train (fit weights) · validation (tune, early stop) · test (once). Underfit = poor everywhere; overfit = train ≫ validation.
- **Backprop computes gradients; the optimizer (SGD, Adam) uses them.**

## Weight initialization (21-08)
| | Xavier/Glorot | He/Kaiming |
|---|---|---|
| for | sigmoid, tanh | ReLU, Leaky, PReLU |
| normal Var | 2/(n_in + n_out) | 2/n_in |
| uniform ±a | √(6/(n_in + n_out)) | √(6/n_in) |
| n_in 100, n_out 50 | Var 0.0133, σ 0.115 | Var 0.02, σ 0.141 |
- **All-equal/zero weights → symmetry**: neurons stay identical (measured: 8 → 1 distinct). Biases = 0 is fine.
- Too small → **vanishing**; too large → **exploding**/saturation. Uniform Var = a²/3 (hence the 6).
- PyTorch default for Linear: U(±1/√n_in), Var 1/(3n_in) (6× smaller than He). `nn.init.kaiming_normal_(w, nonlinearity="relu")`.

## Normalization & regularization (04-09)
- **Normalization = optimisation** (scale). **Regularization = generalisation** (overfitting). Complementary.
- **Standardise** (x − μ)/σ · **min-max** (x − min)/(max − min) (outlier-sensitive) · pixels/255 · fit statistics on **train only** (no leakage).
- **BatchNorm:** batch μ, σ² → x̂ = (x − μ)/√(σ² + ε) → γx̂ + β · running statistics at inference · CNNs.
- **LayerNorm:** per-sample statistics over features · batch-independent · Transformers. Instance norm → style transfer; Group norm → small batches.
- **Overfitting:** training loss ↓ while validation loss ↑ (underfit = both poor).
- **L1** λΣ|w| → **sparsity** · **L2** (λ/2)Σw² → shrinkage (gradient + λw) · **Elastic Net** = both.
- **Dropout** p: zero activations randomly in training, scale survivors by 1/(1 − p), off at inference.
- **Early stopping:** best validation epoch + patience · **augmentation:** label-preserving variations · **weight decay:** (1 − ηλ)w − η∇L; with Adam use **AdamW** (decoupled).
- Guide: different scales → standardise · CNN → BatchNorm · Transformer → LayerNorm · overfit → L2 / dropout / early stopping · sparse → L1.

## Text → tokens → vectors (handout)
- Pipeline: text → **tokens** → IDs → **embedding rows** (E is V × d) → **+ position vectors** → Transformer.
- Words: huge vocabulary, unknowns · characters/bytes: long sequences · **subwords**: balance. Attention cost ∝ **n²d** (2× tokens → 4× work).
- **BPE:** merge the most frequent adjacent pair, repeat (low/lower/newest/widest → es, est, est</w>, lo, low, …).
- **WordPiece:** score count(ab)/(count(a)·count(b)) (0.06 beats 0.005); encode by **greedy longest match**, `##` = continuation.
- Embedding: lookup = one-hot × E (lookup is cheaper). Params **V × d**: 30,000 × 768 = **23.04M** (92.16 MB fp32); 50,000 × 512 = 25.6M (102.4 MB). Only the used rows get gradients ([4, 9, 4] → row 4 gets two).
- Attention is **permutation equivariant** → needs positions. Learned (L_max × d) or sinusoidal: PE(pos, 2i) = sin(pos/10000^{2i/d}), cos for 2i+1. d = 4, pos 1 → [0.8415, 0.5403, 0.0100, 0.99995]. Modern: RoPE.
- BERT: Thiruvananthapuram → 7 pieces; the same sentence costs 6 tokens in English, 13 in Spanish (**fairness**).

## Loss landscapes & optimizers (27-08)
- **W ← W − η·∂L/∂W.** η too large → overshoot, oscillate, diverge, NaN; too small → very slow. On L = w²: converges iff 0 < η < 1.
- **Schedules:** step decay η₀γ^⌊t/k⌋ (0.01 → 0.001 → 0.0001) · exponential η₀γᵗ · **ReduceLROnPlateau** (cut when validation loss stalls; patience) · **cosine** η_min + ½(η_max − η_min)(1 + cos πt/T) · **warm-up** (start small, then hand over). Measured: no gain on a small easy task, so validate.
- **Saddle** f = x² − y²: gradient 0 at (0, 0), min along x, max along y → looks "stuck". Escape steps: GD 73 · momentum 27 · Adam 20. Local min (gradient 0, not lowest) vs global min.
- **Momentum** v ← βv − η∇L (remembers the past) · **Adam** = momentum (1st moment) + adaptive scaling (2nd moment).
- **Exploding gradient:** η 0.001 × grad 10,000 = update 10. **Clip by value** [−10, −4, −1, 0.5, 3, 8] → [−2, −2, −1, 0.5, 2, 2] (direction changes) · **clip by norm** [3, 4] (‖g‖ = 5), c = 2 → [1.2, 1.6] (direction kept, the first choice). Keras `clipnorm=1.0` / `clipvalue=1.0`; PyTorch `clip_grad_norm_` between `backward()` and `step()`.
- Clipping does **not** fix vanishing gradients, bad initialization, a bad η, over/underfitting. **Clipping edits g; a schedule edits η.**

## Optimizers: GD, SGD, mini-batch, momentum, Adagrad (01-09, 02-09)
- Same rule **W ← W − η·∂L/∂W**; only the data per gradient differs. 10,000 records: **GD 1** iteration/epoch (smooth, slow, memory-hungry) · **SGD 10,000** (noisy, jitters at the minimum) · **mini-batch (1,000) 10** = N / batch size (the default).
- **Momentum:** v_t = βv_{t−1} + η∇J, θ ← θ − v_t, v₀ = 0; unrolled v = ηΣβ^(t−i)g_i; amplification **1/(1−β)** (0.9 → 10×, 0.99 → 100×); damps zig-zag, but can overshoot. Speed-up grows with ravine steepness (1.2× → 23× measured).
- **Nesterov:** gradient at the look-ahead θ − βv_{t−1}; less overshoot.
- **Adagrad:** G += g⊙g, θ ← θ − η/√(G+ε)⊙g; per-parameter rate η/√G (rare features keep big steps: η 0.1, grads 3 and 4 → G 25 → 0.02); G only grows → rate → 0 → **stalls** (fixed by RMSProp/Adam's decaying average).
- Best η differs by optimizer (SGD 0.3 · momentum 0.1 · Adagrad 0.03 · Adam 0.001 on digits); never reuse η across optimizers.

## GenAI: from text to MiniGPT
- Pipeline: text → IDs [B, T] → embeddings [B, T, d] + positions → L × (attention + FFN, residual, LayerNorm) → logits [B, T, V] → softmax → next token.
- **Embeddings from statistics:** co-occurrence → PMI = log P(w,c)/(P(w)P(c)) (cat-pet: log 5) → low-dimensional factors; cosine = direction. Meaning emerges from prediction, with no labels.
- **Attention** = softmax(QKᵀ/√d_k)V. Q "what I need", K "what I offer", V "what I give". "The cat sleeps": scores [1, 2, 1] → [0.212, 0.576, 0.212] (unscaled). ÷√d_k stops softmax becoming one-hot at large d.
- **Causal mask** = lower-triangular, −∞ before softmax. **No mask → loss 0.003 but garbage text (cheating).**
- d_head = d_model / H (256/8 = 32). Scores [B, H, T, T]. Attention = communication, **FFN** = computation (4× expand, GELU), **residual** x + F(x), **pre-norm**.
- MiniGPT: char-level, context 64, d 64, 4 heads, 2 blocks → 107K params, loss 3.25 → 0.1 in 600 AdamW steps. Sampling: temperature (sharpen/flatten), top-k, top-p; **KV cache** for fast decoding.

## GenAI: autoregressive & encoder-decoder
- **Autoregressive** = p(y₁…y_T) = ∏ p(yₜ | y<t): 0.6 × 0.5 × 0.8 = 0.24; train on −Σ log p (no underflow).
- **Encoder-only** (BERT: classify) · **decoder-only** (GPT: generate) · **encoder-decoder** (T5/BART: translate, summarize).
- Encoder → **memory M [B, S, d]** (contextual vectors, not logits). Decoder block: masked self-attention → **cross-attention (Q from decoder, K/V from M; scores T × S)** → FFN. Head on the **decoder**.
- **Teacher forcing:** input BOS + target, predict target + EOS; pad tokens ignored in the loss. **Exposure bias:** 90.5% (teacher-forced) vs 81.9% (free-running) token accuracy.
- Inference: encode once → BOS → choose → append → repeat to EOS. Cache encoder memory and K/V (T = 1,000: 500,500 → 1,000 positions). Greedy / beam / sampling.
- Choose by **what must be visible at prediction time**. Misconception: "autoregressive = decoder-only".

## Case studies (tabular ANNs)
- Workflow: inspect → explore → encode (one-hot multi-valued) → **split (stratify) → impute/scale with train stats** → baselines → ANN → test-set metrics → interpret.
- Multi-class: softmax + sparse CCE, labels 0…K−1. Binary: 1 sigmoid + BCE; the threshold is a business choice.
- Forest cover: ANN 84.9% (class run 82.3%) vs baseline 48.8% vs **random forest 90.7%**; macro F1 exposes rare classes (Aspen recall 0.46); class weights raise rare recall but cost overall (balanced → 67%).
- Airline: logistic regression 87.6% → ANN 96.1% (non-linear, U-shaped wifi effect) ≈ random forest 96.2%. Correlation ≠ importance.
- Val acc > train acc with dropout is normal. Params of Dense(n_in → n_out) = n_in·n_out + n_out.

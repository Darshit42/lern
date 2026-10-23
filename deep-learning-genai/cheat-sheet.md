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

# DL & GenAI · Cheat Sheet (lectures 04-08 → 04-09)

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

# CV Weeks 2–3 · Cheat Sheet

## Building blocks
| Term | One-liner |
|---|---|
| **Conv layer** | slides **learned** kernels over the image → feature maps · few parameters (shared weights) · keeps spatial layout · translation invariant |
| **FC (dense) layer** | every neuron sees every input · flattens (loses *where*) · huge parameter count · used at the end for the decision |
| **ReLU** | max(0, x): negatives → 0 · non-linear, cheap, gradient = 1 for x > 0 → less vanishing gradient |
| **Sigmoid / tanh** | slopes ≤ 0.25 / ≤ 1 and ~0 when saturated → gradients vanish in deep nets |
| **Dropout** | randomly zero a fraction *p* of neurons **during training only** → robust features, less overfitting · 10 neurons, p = 0.3 → ~3 dropped · survivors × 1/(1−p) |
| **Max pooling 2×2** | keep the max of each 2×2 block → halves H and W, no parameters |
| **Training loop** | forward → loss → `loss.backward()` (gradients) → `optimizer.step()` |
| **Overfitting** | great on training data, worse on new data (a train–test gap) |
| **Vanishing gradient** | the gradient is a product of many small factors → early layers stop learning |

## CNN timeline
| Model | Year | Key facts |
|---|---|---|
| **LeNet-5** | 1998 | **Yann LeCun** · digits / **MNIST** · cheques & post · conv → pool → conv → pool → FC → FC · ~60 K params · proof of concept |
| **AlexNet** | 2012 | won **ImageNet** by a huge margin (top-5 error ~26% → **15.3%**) · **5 conv + 3 FC** · ~61 M params · **ReLU, dropout, GPU** · sparked the deep-learning revolution |
| **VGG** | 2014 | Oxford **Visual Geometry Group** · **16–19 layers** · only **3×3** filters · 2×3×3 = 5×5 view, 3×3×3 = 7×7 view with fewer params + more ReLUs · baseline · ~138 M params |
| **ResNet** | 2015 | solves **vanishing gradient / degradation** · **skip connections**: out = F(x) + x · learns the **residual** · gradient F′(x) + **1** · 50 / 101 / **152** layers · used in medical imaging |
| **EfficientNet** | 2019 | **compound scaling**: depth α^φ, width β^φ, resolution γ^φ, α·β²·γ² ≈ 2 (1.2, 1.1, 1.15) · B0 (5.3 M) beats ResNet-50 (25.6 M) · mobile apps |
| **MobileNet** | 2017 | depthwise-separable conv (per-channel K×K + 1×1 mix) → ~8–9× fewer weights · phones |

## Object detection = **what + where** (class + box for every object)
| | **R-CNN** (2014) | **YOLO** (2016) | **SSD** (2016) |
|---|---|---|---|
| Stands for | Region-based CNN | You Only Look Once | Single Shot MultiBox Detector |
| How | ~2,000 region proposals → CNN on **each** → classify + refine | **one pass**: S×S grid predicts boxes + classes (a regression problem) | one pass, predictions from **multi-scale feature maps** + default boxes |
| Speed / accuracy | slow / **most accurate** | **real-time** / weaker on small, crowded objects | fast / good across sizes |
| Use for | **medical** (accuracy > speed) | **driving**, video (speed > accuracy) | mobile, balanced |
- **IoU** = overlap / union (≥ 0.5 = correct box) · **NMS** = keep the best box, drop overlapping duplicates.
- Fast R-CNN: CNN once, crop the feature map · Faster R-CNN: **RPN** proposes regions.
- Small objects in YOLO: higher resolution, finer grid, multi-scale heads (FPN), small anchors, tiling, augmentation.

## Segmentation = a label for **every pixel**
- **Semantic:** class per pixel (touching objects merge) · **Instance:** separate mask per object.
- Challenge: needs **context** (what) **and precision** (where).
- **FCN:** FC → conv layers · dense pixel map · any input size · **coarse** (predicts at low resolution, then upsamples).
- **U-Net (2015):** **encoder** (downsample → context) + **decoder** (upsample, transposed conv → precision) + **skip connections** (copy + concatenate encoder maps → fine detail) · standard in **medical imaging** (precise boundaries, small structures, small datasets) · also satellite imagery.
- **Mask R-CNN (2017):** Faster R-CNN + **mask branch** per RoI · **RPN** proposes regions · **RoI Align** (bilinear, no rounding) replaces RoI Pooling · outputs **box + class + mask** · handles overlapping objects (one mask per RoI) · masks give the true size and volume in medicine.
- Skip connections: ResNet **adds** (gradient flow, depth) · U-Net **concatenates** (spatial detail).

## Sequences & video
- Video = **spatial** (what's in each frame → **CNN**) + **temporal** (how it changes → **RNN / LSTM**).
- **RNN:** hidden state hₜ = tanh(W xₜ + U hₜ₋₁) · **short-term** memory · vanishing gradients over long sequences.
- **LSTM:** cell state + gates · **forget** (discard), **input** (store), **output** (pass on) · cₜ = f⊙cₜ₋₁ + i⊙c̃ · **long-term** memory · variable length.
- **CNN + LSTM:** CNN features per frame → LSTM over time → class (gesture, action, captioning, driving).
- RNN is still enough for short sequences (next character, last few sensor readings).

| | RNN | LSTM |
|---|---|---|
| Memory | short-term | long-term |
| Problem | vanishing gradients | solves them |
| Best for | simple, short | complex, long (video, paragraphs) |

## Attention & Vision Transformers
- **Attention(Q, K, V) = softmax(QKᵀ / √d) · V** · **Query** = what I look for · **Key** = what I offer · **Value** = what I pass on → output = weighted sum.
- **ViT (2020):** split into **patches** (e.g. 16×16 → 224² image = 196 tokens) → flatten + **linear embedding** + **position embedding** + **[CLS]** token → transformer encoder (attention + MLP, **residual** connections) → classify from [CLS].
- Patches → image becomes a sequence; any patch can attend to any other from layer 1 (**global context**); shorter sequence = affordable; local detail stays inside each patch.
- CNN = local, built-in locality assumption, data-efficient · ViT = global, needs **large datasets**, scales well, hybrid models.
- Uses: classification, medical imaging, video (temporal attention), vision + language.

# Week 2 & 3 Question Bank: Model Answers

Answers to every question in **CV-Week2-3-Questions.pdf** (original in `course-files/`). Each answer is eval-length and points to the notebook (`notebooks/0X_…`) where you can **see it happen**.

> Cover the answer, say yours out loud, then check. For "Hard" questions, aim for 5–8 lines with an example.

---

## Easy

**1. Who invented LeNet and what was its primary application?**  
**Yann LeCun** (Bell Labs, LeNet-5 in 1998). Primary application: **handwritten digit recognition**, used for reading **bank cheques and postal (zip) codes**. → nb 02

**2. What dataset was LeNet trained on?**  
**MNIST**: 60,000 training + 10,000 test images of handwritten digits, 28×28 grayscale (padded to 32×32 for LeNet).

**3. What was AlexNet's biggest achievement in 2012?**  
It **won the ImageNet competition (ILSVRC 2012) by a huge margin**: about 15.3% top-5 error vs about 26% for the runner-up. That **sparked the deep-learning revolution** in computer vision.

**4. Name two innovations introduced in AlexNet.**  
Any two of: **ReLU activation** (faster training), **dropout** (less overfitting), **GPU training** (massive speed-up). Also heavy data augmentation and overlapping max-pooling.

**5. What activation function replaced sigmoid/tanh in AlexNet?**  
**ReLU** (Rectified Linear Unit), f(x) = max(0, x). → nb 01, 02

**6. What does ReLU do to negative input values?**  
Sets them to **zero**. Positive values pass through unchanged.

**7. What is the purpose of dropout?**  
To **prevent overfitting** (a regularization technique). During training it randomly switches off (zeros) a fraction of neurons each pass, so the network can't rely on a few specific neurons and must learn robust, distributed features. It's switched off at test time. → nb 01

**8. How many layers does VGG typically have?**  
**16 to 19** weight layers (VGG-16: 13 conv + 3 FC; VGG-19: 16 conv + 3 FC).

**9. What filter size did VGG standardize?**  
Small **3×3** convolution filters, stacked one after another.

**10. What problem does ResNet solve?**  
Training **very deep networks**: the **vanishing gradient / degradation** problem. Plain networks got *worse* (even on training data) as layers were added, because gradients faded before reaching the early layers. → nb 03

**11. What is the key innovation in ResNet?**  
**Residual (skip) connections**: each block outputs **F(x) + x**, so it learns only the *residual* F(x) = H(x) − x, and gradients flow through the identity path.

**12. What does EfficientNet's compound scaling balance?**  
The network's **depth** (layers), **width** (channels) and input **resolution**, scaled **together** with one coefficient φ: depth α^φ, width β^φ, resolution γ^φ.

**13. Which CNN architecture enabled training of networks with 152 layers?**  
**ResNet** (ResNet-152).

**14. Which architecture is lightweight yet accurate for mobile vision apps?**  
**EfficientNet** (e.g. EfficientNet-B0: about 5.3 M parameters, more accurate than ResNet-50). **MobileNet** is the other classic answer; EfficientNet is built from its depthwise-separable blocks.

**15. What does object detection aim to identify in an image?**  
**What** each object is (class label) and **where** it is (bounding box), for **every** object in the image.

**16. What does R-CNN stand for?**  
**Region-based Convolutional Neural Network.**

**17. Why is YOLO considered fast?**  
It looks at the image **once**: a single forward pass of one network predicts all bounding boxes and class probabilities together (detection as one regression problem). There's no separate stage of thousands of region proposals. → nb 04 (our toy YOLO was over 100× faster than our toy R-CNN)

**18. What does SSD stand for?**  
**Single Shot MultiBox Detector.**

**19. What is semantic segmentation?**  
**Assigning a class label to every pixel** in an image (e.g. road / car / person / sky), giving a dense label map. It doesn't separate individual objects of the same class. → nb 05

**20. What is the role of the encoder in U-Net?**  
The **downsampling path**: convolution + pooling progressively **reduce resolution** while building **context and high-level features** ("what is in the image"). Like compressing a photo into a summary.

---

## Medium

**1. How did LeNet prove CNNs were better than traditional ML methods?**  
On MNIST, LeNet-5, trained **end to end on raw pixels**, matched or beat the best traditional methods of the time (k-NN, SVMs on hand-crafted features) at about 0.95% error. It **learned its own features** (strokes, edges) instead of relying on human-designed ones, and was far faster to run. This led to real deployments reading cheques and post. *(Nuance: the best SVMs were close on MNIST. The decisive gap appeared with large, complex images, e.g. AlexNet on ImageNet. See nb 02.)*

**2. Why was GPU training critical for AlexNet's success?**  
Training a CNN is **millions of independent multiply-adds** (every kernel at every position, for 1.2 M images, over many epochs). GPUs have **thousands of parallel cores**, so they ran this **many times faster** than CPUs. AlexNet (about 61 M parameters) trained in about a week on two GTX 580 GPUs. On CPUs it would have taken months, making the model impractical. → nb 02 (timing demo)

**3. Compare accuracy before and after AlexNet on ImageNet.**  
Before (2011, hand-crafted features): about **74% top-5 accuracy** (about 26% error). **AlexNet (2012): about 85%** (15.3% error), a jump of more than 10 points in one year. The course notes summarise it as "~70% → ~85%+".

**4. Why does stacking small filters (like 3×3 in VGG) improve performance?**  
1. **Same receptive field, fewer parameters:** two 3×3 layers see 5×5, and three see 7×7, using 27C² weights instead of 49C² (about 45% fewer).
2. **More non-linearity:** each layer adds a ReLU, so three layers give three non-linearities instead of one, and the network can learn more complex features.
3. The **simple, uniform** design makes deeper networks easier to build.
→ nb 02 (receptive-field demo)

**5. How do residual connections help gradient flow in ResNet?**  
Output = F(x) + x, so the gradient is **∂out/∂x = F′(x) + 1**. The "+1" from the identity path passes the gradient straight back through every block, even when F′(x) is tiny. In a plain net the gradient is a **product** of many small factors and vanishes. → nb 03 (plain 52-layer net: gradient at layer 1 = 0; residual: about 0.02)

**6. What is the difference between direct mappings and residuals?**  
- A **direct mapping**: the layers must learn the full desired output **H(x)** themselves.
- A **residual**: the layers learn only the **difference** F(x) = H(x) − x, and the input is added back.

Learning "change nothing" is easy for a residual block (F = 0) but hard for a plain one (it must learn the identity). So adding residual layers never makes the network worse.

**7. Why is EfficientNet more efficient than simply making networks deeper or wider?**  
Depth, width and resolution **depend on each other**. Higher resolution needs more depth (a larger receptive field) and more width (more fine-grained features). Scaling just one hits **diminishing returns**: lots of extra computation for little accuracy. **Compound scaling** grows all three in a fixed, balanced ratio (α = 1.2, β = 1.1, γ = 1.15; each φ step about doubles FLOPs), so extra compute goes where it helps most. EfficientNet-B7 matched the best accuracy of its time with about 8× fewer parameters.

**8. Give one real-world application of ResNet.**  
**Medical imaging**, e.g. classifying chest X-rays (pneumonia), skin-lesion images or retinal scans, usually an ImageNet-pretrained ResNet fine-tuned on medical data. ResNet is also the **backbone** of Faster/Mask R-CNN and of face-recognition models.

**9. Give one real-world application of EfficientNet.**  
**On-device mobile vision**, e.g. a phone app that identifies plant diseases, food or products from the camera in real time, without a server. Also drones and embedded cameras.

**10. Why is R-CNN slow compared to YOLO?**  
R-CNN generates about **2,000 region proposals** per image and runs the CNN **separately on each** (plus a separate classification/regression stage), so it's thousands of forward passes per image. YOLO does **one** forward pass for the whole image. → nb 04

**11. How does YOLO treat detection differently from R-CNN?**  
R-CNN is **two-stage**: propose regions, then classify each. YOLO is **one-stage**: it treats detection as a **single regression problem**. The image is divided into an S×S grid, and each cell directly predicts boxes, confidence and class probabilities in one pass, using **global context** from the whole image.

**12. Why is SSD better at detecting objects of different sizes?**  
It makes predictions from **multiple feature maps at different scales**. Early, high-resolution maps (small cells) detect **small** objects, and later, low-resolution maps (big cells) detect **large** ones, each with default boxes of several shapes. So a small cat and a large cat in one image are both found in a single pass. → nb 04

**13. Which detection architecture for medical imaging, and why?**  
The **R-CNN family** (Faster R-CNN, or **Mask R-CNN** for exact outlines). Medical tasks need **high accuracy and precise localization**, missing a tumour (a false negative) is unacceptable, and analysis is **offline**, so speed matters less. Example: finding small tumours in MRI scans.

**14. Which detection architecture for autonomous driving, and why?**  
**YOLO (or SSD).** Cars need **real-time** detection (30+ fps) of pedestrians, vehicles and signs. YOLO's single pass is fast enough, and a slightly less precise box *now* beats a perfect one too late. SSD balances speed with multi-scale accuracy (distant small cars).

**15. What challenge does semantic segmentation face compared to classification?**  
It needs both **context** (*what* is it?, which requires a large receptive field) **and precision** (*exactly where* is each boundary?, which requires fine spatial detail). Downsampling helps the first and destroys the second. Classification needs only one label per image, so it can discard spatial detail freely. Segmentation also needs pixel-level labels, which are expensive to annotate.

**16. How does FCN differ from traditional CNNs in its output?**  
A traditional CNN ends with **fully connected layers** that output **one label for the whole image** (spatial layout lost). An FCN replaces them with **convolutional layers** and outputs a **dense pixel-wise prediction map**, a class score for every pixel, for inputs of any size. → nb 05

**17. Why are FCN predictions often coarse?**  
The network predicts at a **low resolution** after repeated pooling (e.g. 1/32 of the input), then simply **upsamples** that small map. Fine detail lost in pooling can't be recovered, so boundaries are blurry and small objects are missed. → nb 05 (FCN 77% vs U-Net 99% accuracy on boundary pixels)

**18. How do skip connections in U-Net improve segmentation accuracy?**  
They **copy high-resolution feature maps from each encoder level** and **concatenate** them with the decoder at the same resolution. The decoder then gets back the **fine spatial details** (edges, exact positions) that pooling lost, combined with the **high-level context** from deep layers, which gives sharp, accurate boundaries and preserves small structures. They also help gradients flow.

**19. What is the role of the decoder in U-Net?**  
The **upsampling path**: it gradually **increases the resolution** back to the input size (transposed convolutions or upsampling), combines the encoder's context with fine details from the skip connections, and produces the **pixel-level prediction**. The encoder learns *what*; the decoder learns *where*.

**20. Why is U-Net widely used in medical imaging?**  
1. **Precise boundaries**: tumour, organ and cell outlines are needed for measurement and treatment planning.
2. Skip connections keep **small structures** (cells, lesions, vessels).
3. It **works with small datasets** (medical labels are scarce and expensive), especially with augmentation.
4. It's simple and fast to train, and handles 2-D and 3-D scans.

---

## Hard

**1. Explain how dropout improves generalization, with an example.**  
During training, dropout randomly zeros a fraction *p* of neurons each pass, so the network **can't depend on any single neuron or fixed co-adapted group**. It must spread knowledge redundantly and learn features that work even with parts missing, which are **robust features** that transfer to unseen data. It's like training many "thinned" networks and averaging them at test time, when all neurons are used and outputs are scaled accordingly.
*Example:* a layer of 10 neurons with p = 0.3: about 3 are dropped each pass. If one neuron had learned "background is grass → it's a cow", it's sometimes absent, so other neurons must learn real cow features (shape, head), and the model then works on a cow on a beach. Like a team where random members sit out practice, so everyone learns to contribute. → nb 01 (dropout narrowed the train/test gap)

**2. Why did AlexNet's use of ReLU help reduce the vanishing gradient problem?**  
Backpropagation multiplies the gradient by each layer's activation derivative. **Sigmoid's derivative is at most 0.25** (and near 0 when saturated), and tanh's also approaches 0 for large inputs, so through many layers the product **shrinks exponentially** (0.25¹⁰ ≈ 10⁻⁶). **ReLU's derivative is exactly 1 for positive inputs**, so the gradient passes through active units **without shrinking**, and it never saturates for positive values. ReLU is also cheap (no exponentials). AlexNet's ReLU network reached its target error about **6× faster** than with tanh. (Caveat: "dead" ReLUs have zero gradient for negative inputs.) → nb 01, 02

**3. How does compound scaling in EfficientNet systematically balance depth, width and resolution?**  
With one user-chosen coefficient **φ** (how much more compute you can afford):
- depth d = **α^φ**, width w = **β^φ**, resolution r = **γ^φ**,
- subject to **α · β² · γ² ≈ 2** (with α ≥ 1, β ≥ 1, γ ≥ 1).

FLOPs scale with d · w² · r² (width and resolution affect cost quadratically), so each +1 in φ about **doubles the computation**. α, β, γ are found **once** by a small grid search on the baseline B0 (α = 1.2, β = 1.1, γ = 1.15) and then reused to produce B1–B7. Every dimension grows in proportion: higher resolution comes with the extra depth (receptive field) and width (fine features) it needs. → nb 03 (φ table)

**4. Why do residual connections allow training very deep networks without degradation?**  
- **Gradient flow:** out = F(x) + x gives ∂out/∂x = F′(x) + 1. The identity term gives an uninterrupted gradient highway, so gradients don't vanish across 100+ layers.
- **Easy identity:** if extra layers aren't needed, a block just learns F(x) ≈ 0 and passes x through. A deeper ResNet can therefore always do *at least as well* as a shallower one, which removes the **degradation problem** (plain deep nets having higher *training* error than shallower ones).
- In practice, combined with BatchNorm, this enabled 50/101/152 layers.

→ nb 03: plain 22- and 52-layer nets stuck at 10% (chance); residual nets of the same depth trained to 66–76%.

**5. Compare the trade-off between accuracy and speed in R-CNN, YOLO and SSD.**  
| | Accuracy | Speed | Why |
|---|---|---|---|
| **R-CNN** (and Fast/Faster) | **highest**, esp. small objects and cluttered scenes | **slowest** (original R-CNN: tens of seconds per image) | two stages; CNN per region proposal; careful per-region classification and box refinement |
| **YOLO** | lower, struggles with **small, crowded** objects and precise localization | **fastest, real-time** (45+ fps in the original) | one pass, coarse grid, few boxes per cell |
| **SSD** | competitive; good across object sizes | fast (near real-time) | one pass + **multi-scale** feature maps and default boxes |

Choose by application: accuracy-critical and offline (medical) → R-CNN family; real-time (driving, video) → YOLO; balanced or mobile → SSD.

**6. How would you adapt YOLO for detecting very small objects?**  
- **Higher input resolution** and/or a **finer grid** (more, smaller cells).
- **Multi-scale prediction heads with a Feature Pyramid Network** (as in YOLOv3+): detect small objects on high-resolution feature maps.
- **More / smaller anchor boxes**, with anchor sizes tuned (k-means) to the small objects in the dataset.
- **Tiling / slicing** large images into overlapping crops (drones, satellites), then merging detections.
- **Augmentation** (mosaic, copy-paste of small objects, zoom) and **loss re-weighting** for small boxes; avoid excessive early downsampling.

→ nb 04 shows a one-object-per-cell grid losing two small shapes that share a cell.

**7. Why are skip connections critical in both ResNet and U-Net, though used differently?**  
Both send information **around** layers that would otherwise lose it:
- **ResNet:** short skips within a block, **added** (F(x) + x). Purpose: **gradient flow and an easy identity**, so very deep networks train without degradation.
- **U-Net:** long skips from each **encoder** level to the matching **decoder** level, **concatenated**. Purpose: restore the **fine spatial detail** lost by downsampling, giving precise boundaries (they also help gradients).

Shared principle: let later layers access earlier information directly. ResNet uses it to make *depth* trainable; U-Net uses it to make *pixel precision* possible.

**8. How does the encoder–decoder structure in U-Net ensure both context and precision?**  
- The **encoder** downsamples step by step: each level sees a wider area, building **context** (what the region is: tumour vs tissue).
- The **decoder** upsamples back to full resolution to produce a per-pixel map.
- At every level, **skip connections** hand the decoder the **high-resolution encoder features** (edges, exact positions), so it combines *what* (deep context) with *where* (shallow detail).

"Encoder compresses for context, decoder reconstructs for precision, skips bridge the two." Example: the encoder recognises that a region is a tumour, and the skips restore its exact boundary. → nb 05

**9. Designing a segmentation model for satellite imagery: which architecture and why?**  
A **U-Net** (typically with an ImageNet-pretrained **ResNet/EfficientNet encoder**), because:
- satellite targets (roads, rivers, field and building boundaries) are **thin structures** that need **pixel-level precision**, which U-Net's skip connections preserve,
- the encoder's context separates look-alike classes (roof vs road, water vs shadow),
- it trains well with limited labelled data and augmentation (rotations and flips are natural for overhead images).

Practicalities: handle huge images by **tiling** with overlap, use multispectral bands as extra input channels, and use Dice/IoU loss for class imbalance. If you must count or separate **individual** buildings or vehicles, use **Mask R-CNN** (instance segmentation) instead or as well.

**10. How did the evolution LeNet → AlexNet → VGG → ResNet → EfficientNet shape the foundation for Vision Transformers?**  
- **LeNet:** learning features **end to end** from pixels.
- **AlexNet:** **scale** (big data + GPUs) plus ReLU and dropout make large models trainable. ViTs depend on huge data and compute.
- **VGG:** **depth** with simple, **repeated identical blocks**. A ViT is a stack of identical transformer blocks.
- **ResNet:** **residual connections** (used in *every* transformer block) and normalization made very deep training reliable.
- **EfficientNet:** systematic **scaling** of model size, data and resolution, which also guides how ViT sizes (Base, Large, Huge) are scaled.

ViT keeps this deep, residual, scaled-up recipe but replaces local convolutions with **global self-attention over patches**. The CNN lineage also motivated *why*: CNNs struggle with long-range dependencies, and ViT addresses that. Hybrid CNN + Transformer models continue the line. → nb 07

---

## Mask R-CNN

**1. What key limitation of traditional object detection does Mask R-CNN address?**  
Detectors output only **bounding boxes + labels**, which are rough rectangles. Many tasks need the **exact pixel-level shape of each individual object** (**instance segmentation**). Mask R-CNN adds a mask for every detected object.

**2. How does Mask R-CNN extend Faster R-CNN?**  
It keeps Faster R-CNN (backbone + RPN + classification and box regression) and adds a **parallel mask branch**: a small FCN that predicts a **binary mask for each Region of Interest**. It also replaces **RoI Pooling with RoI Align** for pixel-accurate features.

**3. What role does the Region Proposal Network (RPN) play?**  
It slides over the shared feature map and **proposes candidate object regions** (boxes + "objectness" scores) using anchor boxes. These proposals are what the later heads classify, refine and mask. It replaces slow external proposal methods (selective search) and shares computation with the backbone.

**4. Why was RoI Align introduced, and how does it improve on RoI Pooling?**  
RoI Pooling **rounds (quantizes)** the proposal's fractional coordinates, and its sub-bins, to whole feature-map cells. This misaligns the extracted features with the actual object, which is harmless for boxes but bad for **pixel-accurate masks** (one feature cell can be 16+ image pixels). **RoI Align removes all rounding**: it samples the feature map at the exact fractional positions with **bilinear interpolation**, so features line up with the object and mask accuracy improves noticeably. → nb 05 (numeric demo)

**5. What are the three outputs produced for each detected object?**  
1. **Bounding box** (where). 2. **Class label** (what). 3. **Pixel-level mask** (its exact shape).

**6. Why are pixel-level masks more useful than bounding boxes in medical imaging?**  
1. **Precise localization:** tumours and organs are irregular, and a box includes healthy tissue.
2. **Quantitative analysis:** masks give the true **area and volume** (e.g. tumour volume for chemotherapy dosage), while boxes overestimate.
3. **Clinical decisions:** surgery and radiotherapy planning and disease monitoring need exact borders and spread.
4. **Overlaps:** masks separate structures within structures (a tumour inside an organ).

**7. Give two real-world applications of Mask R-CNN outside medical imaging.**  
Any two of:
- **autonomous driving** (pedestrian, vehicle and lane shapes)
- **crowd analysis** (separating overlapping people)
- **robotics grasping** (exact object outline)
- **photo/video editing** (cut out a person, background replacement)
- **agriculture** (individual fruit or leaf counting)
- **retail** (products on shelves)
- **satellite imagery** (individual buildings)

**8. How does Mask R-CNN handle overlapping objects in crowded scenes?**  
Each object is a separate **RoI proposal**, and each RoI gets **its own class and its own binary mask**, predicted inside its own box. Overlapping people become separate instances with separate masks, even where they touch. Masks are predicted **per class and independently** (no competition between classes), and NMS removes duplicate proposals of the same object. Semantic segmentation would merge them into one "person" blob. → nb 05 (touching-coins demo)

---

## Sequence modeling for video (CNN + LSTM)

**9. Why can't CNNs alone capture temporal dependencies in video?**  
A CNN processes **each frame independently** and has no memory of previous frames and no notion of order. It can recognise *what* is in a frame (a hand) but not *how* it changes (a wave). Shuffle the frames and a per-frame CNN gives the same answers. → nb 06 (order-blind models fail to tell left from right)

**10. What specific problem do LSTMs solve compared to vanilla RNNs?**  
The **vanishing gradient** problem over long sequences, which gives RNNs only **short-term memory**. LSTMs add a **cell state** and **gates**, so information (and gradient) can be carried across many time steps: **long-term dependencies**. → nb 06 (at 50 steps: RNN 25% = chance, LSTM 99.8%)

**11. Explain the input, forget and output gates.**  
Each gate is a sigmoid layer giving values from 0 to 1:
- **Forget gate**: decides what **past information to discard** from the cell state (1 = keep, 0 = erase).
- **Input gate**: decides what **new information to store** in the cell state.
- **Output gate**: decides what part of the memory to **pass forward** as the hidden state / output now.

c_t = f ⊙ c_{t−1} + i ⊙ c̃_t ; h_t = o ⊙ tanh(c_t).

**12. In a CNN + LSTM pipeline, what does each component contribute?**  
- **CNN:** extracts **spatial features** from each frame (objects, shapes, edges, textures): the *what*.
- **LSTM:** processes the **sequence** of per-frame feature vectors and models **temporal dependencies** (motion, order, speed, rhythm): the *how / when*.
- **Output layer:** predicts the action or gesture class.

"Photographer (CNN) + storyteller (LSTM)."

**13. Give an example of an application where CNN + LSTM is essential.**  
**Gesture or sign-language recognition**: the CNN detects the hand shape in each frame, and the LSTM recognises the movement sequence ("wave", "swipe", a sign). Others: action recognition in sports or surveillance ("throwing a ball"), video captioning, predicting pedestrian motion for self-driving cars.

**14. How does sequence modeling enable gesture recognition compared to static image classification?**  
A gesture is defined by **movement over time**, while a single frame only shows a **pose**: "raising a hand" and "waving" can share the same frame, and "swipe left" vs "swipe right" contain the same frames in **reverse order**. A static classifier sees one frame and can't tell them apart. A sequence model (LSTM) remembers previous frames and recognises the **trajectory, order and speed** of the motion. → nb 06

## RNNs vs LSTMs

**15. What is the vanishing gradient problem, and why does it affect RNNs more than LSTMs?**  
During backpropagation (through time), the gradient is **multiplied by a factor at each step or layer**. If those factors are < 1 (tanh derivatives, small recurrent weights) it **shrinks exponentially**, so early inputs get almost no learning signal and the model "forgets" them. A vanilla RNN passes its memory through **tanh and the same weight matrix at every step**, so after 20–50 steps the gradient vanishes. An LSTM updates its cell state **additively** (f ⊙ c + i ⊙ c̃). When the forget gate is about 1, memory and gradient flow through almost unchanged (the "constant error carousel"). → nb 06 (gradient reaching step 1: RNN 4e-14 vs LSTM 4e-5)

**16. Compare RNNs and LSTMs in terms of memory and best use cases.**  
| | RNN | LSTM |
|---|---|---|
| Memory | **short-term** (recent steps) | **long-term** (long dependencies) |
| Gradient | vanishes over long sequences | gates largely solve it |
| Cost | small, fast | about 4× parameters, slower |
| Best for | short, simple sequences | long, complex sequences: video, paragraphs, speech |
| Example | next word in a short sentence | context across a long paragraph or video |

**17. Why are LSTMs better suited for variable-length sequences like video?**  
Both process one step at a time with shared weights, so any length is *possible*. But only the LSTM keeps relevant information over **long and varying spans**: the gates learn what to keep and what to forget regardless of clip length, so a 10-second gesture is handled as well as a 2-second one. An RNN's memory fades after a few steps, so longer clips lose their beginning.

**18. Give an example of a task where RNNs might still be sufficient instead of LSTMs.**  
Short sequences where only recent context matters: **predicting the next character in short words**, **smoothing or forecasting a sensor reading from the last few values**, **predicting the next note from the last few notes**, or a simple short-gesture detector over a handful of frames. RNNs are smaller and faster there. (GRUs are a lighter middle ground.)

## Vision Transformers & Attention

**19. What is the purpose of Query, Key and Value in attention?**  
Each token produces three learned vectors:
- **Query** = *what I'm looking for*,
- **Key** = *what I contain / can be matched on*,
- **Value** = *the information I pass on*.

Attention compares each **Query with all Keys** (dot products) to get similarity scores, normalizes them with softmax into weights, and outputs the **weighted sum of the Values**: each token gathers information from the tokens most relevant to it. Attention(Q, K, V) = softmax(QKᵀ/√d)·V. Example: "bank" attends to "river" or "money" to resolve its meaning. → nb 07

**20. Why does splitting an image into patches help ViTs capture global context?**  
1. It turns the image into a **sequence of tokens** (like words), so the transformer's **self-attention** can relate **any patch to any other**, even far-apart ones, **from the first layer**. A CNN neuron only sees a small local window and needs many layers to connect distant regions.
2. It **reduces complexity**: attention cost grows with the square of the sequence length, and 196 patches instead of about 50,000 pixels makes it feasible.
3. It **balances local and global**: each patch embedding holds local detail, and attention across patches builds the global picture (e.g. stripes across a zebra's body).

# CV Week 2 · Cheat Sheet

## Features (Day 8)
| Term | One-liner |
|---|---|
| **Feature** | distinctive, informative pattern that can be detected, described, compared · good = **unique, repeatable, compact** |
| **Keypoint vs descriptor** | keypoint = **where** (x, y, size, angle, response) · descriptor = **what it looks like** (vector) |
| **Harris** | corner = window changes in **every** direction · gradients → corner response · `cornerHarris(gray, 2, 3, 0.04)` · simple, good for buildings · **not scale/rotation invariant**, no descriptor |
| **SIFT** | scale space → **DoG** extrema (scale inv.) → orientation (rotation inv.) → **128-D** descriptor · very robust · slow · patent expired, free in OpenCV ≥ 4.4 |
| **ORB** | **FAST** detector + **BRIEF** binary descriptor + orientation · 256 bits · Hamming distance · free, fast, rotation inv. · weaker on big scale changes |
| **Homography** | 3×3 matrix mapping one image plane to another (rotating camera or flat scene) · needs ≥ 4 pairs |
| **RANSAC** | fit to random minimal sets, keep the model with most **inliers** → ignores wrong matches |
| **Panorama** | ORB → match → homography (RANSAC) → warpPerspective + paste |

## Matching (Day 9)
- **Image matching** = finding correspondences between images → logos, object recognition, image retrieval, AR.
- **BFMatcher**: compare all pairs · `NORM_HAMMING` for ORB, `NORM_L2` for SIFT · `crossCheck=True` = mutual best.
- **FLANN**: approximate nearest neighbours via an index (KD-tree for SIFT, **LSH** for ORB) · fast, scales · approximate, needs tuning.
- **Lowe's ratio test**: keep if best < **0.7** × second best → removes ambiguous matches.
- **Logo detector**: detect → FLANN → ratio test → homography (RANSAC, > 10 good) → `perspectiveTransform` the logo corners → `polylines`.
- **SURF**: faster than SIFT, patented.

## Tracking (Day 10)
- **Tracking** = following a target across frames · challenges: appearance change, **occlusion**, fast motion.
- **Optical flow** = pixel motion between frames (brightness constancy) · **sparse** (selected keypoints, fast; Lucas–Kanade) vs **dense** (every pixel, detailed; **Farneback**).
- **KLT** = `goodFeaturesToTrack` (Shi–Tomasi corners) + `calcOpticalFlowPyrLK` · efficient · fails on occlusion and large jumps → re-initialise · `winSize` = search window, `maxLevel` = pyramid levels (big motion), `st == 1` = found.
- **Dense-flow visualisation**: `cartToPolar` → hue = **direction**, value = **speed** → HSV→BGR.

## ML in CV (Day 11)
| | **kNN** | **SVM** |
|---|---|---|
| Idea | majority vote of k nearest | max-margin hyperplane |
| Training | none (store data) | optimisation (slow on big data) |
| Prediction | slow (all distances) | fast (support vectors) |
| Pros | simple, small data | high-dimensional data, robust to overfitting |
| Cons | slow on large data, sensitive to irrelevant features and the metric | expensive, kernel/C/gamma tuning |
- **Support vectors** = closest points, they define the boundary · **RBF kernel** = curved boundary (kernel trick).
- **precision** = of predicted X, % really X · **recall** = of real X, % found · **f1** = balance · **support** = count.
- `load_digits` = 8×8, values 0–16 (not real MNIST 28×28).

## Bag of Visual Words (Day 12)
- descriptors → **k-means** → k **visual words** → per-image **histogram** (fixed length k) → SVM/kNN.
- text analogy: word ↔ visual word · dictionary ↔ cluster centres · document ↔ image · word counts ↔ histogram.
- fixed-length vector from a variable number of features · ignores spatial layout · k too small = vague, too big = sparse.
- uses: textures, materials, scenes · build the vocabulary from **training** images only.

## Face detection (Day 13)
- **detection** (where) before **recognition** (who).
- **Haar** = white − black rectangle sums · **integral image** · **AdaBoost** picks features · **cascade** rejects early → fast, lightweight · frontal, controlled · sensitive to lighting, pose, occlusion.
- `detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)` · lower scaleFactor = more sizes/slower · higher minNeighbors = fewer false positives.
- **HOG** = gradients → per-cell 9-bin direction histograms → block normalisation → SVM · robust to illumination, shape-aware · heavier · pedestrians and faces.
- Haar = **fast, lightweight** · HOG = **robust, heavier**.

## Pre-CNN foundations (Day 14)
- dot product = similarity · **convolution = sliding dot product** · gradient = steepest direction ("how steep the hill") · normalise inputs.
- **Supervised** = labelled, predict (classification / regression) · **Unsupervised** = unlabelled, discover (clustering, PCA/t-SNE, association).
- **Neuron**: z = Σwᵢxᵢ + b → f(z) · inputs = evidence, weights = importance, bias = prior tendency, activation = decision.
- **ANN**: input → hidden → output · forward pass · **backpropagation** updates the weights.
- FC on images: too many parameters, loses spatial structure → **CNN** (local filters, shared weights).

## CNN basics (Day 15)
- **Output size = (W − K + 2P)/S + 1** · stride = step · padding = zero border (keep size, cover edges).
- **Feature map** = one filter's response grid · **pooling** (max/avg) = downsample, translation invariance, less computation and overfitting · **flatten** = maps → vector for FC.
- **ReLU** fast default · **sigmoid** 0–1 probabilities · **tanh** −1…1.
- Lab CNN: 1×28×28 → conv 32×26×26 → conv 64×24×24 → pool 64×12×12 → flatten **9216** → 128 → 10 → log_softmax.
- `log_softmax` + `nll_loss` = **cross-entropy** · Adam lr 1e-3 · `zero_grad → forward → loss → backward → step`.
- `model.train()` dropout on · `model.eval()` + `torch.no_grad()` for testing · Normalize(0.1307, 0.3081) = MNIST mean/std.
- Flow: **Input → Conv → Pool → Flatten → FC → Output**.

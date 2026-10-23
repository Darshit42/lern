# 14 · RMSprop, Adam and AdamW (lecture of 03-09-2026)

**Notebook:** `notebooks/14_rmsprop_adam_adamw.ipynb` (all three written by hand and matched against PyTorch, what bias correction really fixes, Adam's scale-invariance, the RMSprop-vs-Adagrad stall test, Adam + L2 vs AdamW measured, a race at η = 0.001)

**Where this fits:** chapter 13 ended with momentum (average gradients over **time**) and Adagrad (scale each parameter by its gradient **history**, which never forgets). This lecture combines them:

| | Builds on | Adds |
|---|---|---|
| **RMSprop** | Adagrad | **forgetting**: a moving average of g² instead of a growing sum |
| **Adam** | RMSprop + momentum | a first moment (direction) + **bias correction** |
| **AdamW** | Adam | weight decay applied **separately** (decoupled) |

---

## 1. RMSprop (Root Mean Square Propagation)

Controls each parameter's step using a **moving average of squared gradients**, so directions with consistently large gradients get smaller steps.

**gradient g_t → square g_t² → moving average s_t → normalise g_t / (√s_t + ε)**

**s_t = ρ·s_{t−1} + (1 − ρ)·g_t²**
**θ_t = θ_{t−1} − η·g_t / (√s_t + ε)**

- g_t = the gradient at step t; s_t = the exponential moving average of g²;
- ρ (rho) = decay rate (the handout says ≈ 0.95; PyTorch's default `alpha` is 0.99, Keras's `rho` 0.9);
- η = learning rate; ε ≈ 10⁻⁸ avoids division by zero.

**How it works:** compute the gradient (backprop) → square each element → update the moving average → divide the gradient by its root → update the parameters.

**Why it fixes Adagrad (measured):** walking θ from 0 to 100 with η = 1, **Adagrad stalls at θ = 31** after 300 steps (its sum never forgets the huge early gradients), while **RMSprop arrives** (within 0.5 by step 125).

- **+** per-parameter learning rates; often converges faster than SGD; handles gradients of very different sizes; stable in noisy or changing settings.
- **−** needs η and ρ; no first moment (no momentum); still sensitive to η.

---

## 2. Adam (Adaptive Moment Estimation)

Momentum + RMSprop: **two** moving averages.

**m_t = β₁·m_{t−1} + (1 − β₁)·g_t**  (first moment: the mean of gradients, momentum-like)
**v_t = β₂·v_{t−1} + (1 − β₂)·g_t²**  (second (raw) moment: the mean of squared gradients, RMSprop-like)
**m̂_t = m_t / (1 − β₁ᵗ),  v̂_t = v_t / (1 − β₂ᵗ)**  (bias correction)
**θ_t = θ_{t−1} − η·m̂_t / (√v̂_t + ε)**

β₁ = **0.9**, β₂ = **0.999** (PyTorch/Keras defaults).

> **Handout inconsistency:** the text says "β₁ is commonly 0.9", but its hyperparameter table lists β₁ ~ 0.95 for Adam and AdamW. The standard default is **0.9**.

**Step by step:** compute g_t → update m_t → update v_t → bias-correct to m̂_t, v̂_t → compute the adaptive update → update the parameters.

### Why bias correction? (measured)
m₀ = v₀ = 0, so the early averages are pulled towards 0. With a constant gradient g = 2:
- **with correction:** m̂ = 2, v̂ = 4 from step 1, so the step is **exactly η**.
- **without:** m₁ = 0.2 and v₁ = 0.004, giving m/√v = 0.2/0.063 = **3.16 × η**. v is far more biased than m (β₂ = 0.999 forgets very slowly), so the uncorrected step is **too big**, peaking at **6.6 × η** around step 12 and still 1.6 × η at step 500.

So "biased toward zero" leads in practice to **oversized, unstable first steps** while the weights are still random. Bias correction makes Adam safe to start.

### Why "less manual tuning"? Adam's step ignores the gradient's scale (measured)
Scale the loss by 0.001, 1 or 1000 (as if the data had different units):
- **SGD (η = 0.1):** barely moves at ×0.001 (loss 5.11), converges at ×1, **explodes to NaN** at ×1000.
- **Adam (η = 0.05):** the **same final loss (1.44 × 10⁻⁸) at all three scales**: m̂ and √v̂ scale together, so their ratio doesn't change.

In Adam, η means roughly "how far a weight may move per step", not "multiply the gradient by this".

**Checked in code:** hand-written RMSprop, Adam and AdamW give **the same numbers as** `torch.optim.RMSprop` (ρ is called `alpha`), `Adam` and `AdamW`, to about 10⁻¹⁶.

- **+** fast convergence on many tasks; momentum-like direction + adaptive step sizes; less learning-rate tuning; good for sparse and noisy gradients.
- **−** can generalise differently from SGD on some tasks; stores two extra vectors (m and v); η still matters.

---

## 3. AdamW: Adam with decoupled weight decay

**θ_t = θ_{t−1} − η·m̂_t / (√v̂_t + ε) − η·λ·θ_{t−1}**

The first term is Adam's adaptive update; the second **directly shrinks the weights**, with strength λ.

| | Adam + L2 | AdamW |
|---|---|---|
| where λθ goes | **added to the gradient**, then divided by √v̂ with everything else (**coupled**) | **subtracted from the weights** directly (**decoupled**) |
| effect | decay is uneven: weights with **small** gradients (small √v̂) get **hugely amplified** decay | the same proportional shrink for every weight, independent of gradient history |
| λ behaves as | something tangled with the adaptive scaling | a direct "parameter-shrinkage" control |

**Measured on the digits (λ = 0.01 for both, 40 epochs, 3 seeds):**

| | final weight norm (start 13.2) | test accuracy |
|---|---|---|
| Adam, no decay | 22.1 | 98.1% |
| **Adam + L2, λ = 0.01** | **8.7** | **95.6%** |
| AdamW, λ = 0.01 | 21.9 | 98.0% |
| AdamW, λ = 0.1 | 20.7 | 98.0% |
| AdamW, λ = 1.0 | 15.3 | 97.5% |

- **The same λ means different things:** in Adam + L2 it collapsed the weights and cost 2.5 points; in AdamW it barely did anything (its per-step shrink is η·λ = 0.00001).
- AdamW needs **larger λ values** (commonly 0.01–0.1 in long Transformer training) and behaves **predictably**. **Never copy a `weight_decay` value between Adam and AdamW.**
- PyTorch's `AdamW` multiplies the weights by (1 − ηλ) before the Adam step: the same as the lecture's formula (verified).

**Why AdamW is popular:** strong optimisation **and** regularisation together; the standard for **Transformers** and modern models; weight decay can be tuned independently of the adaptive update.

---

## 4. Comparison and choosing

| Optimizer | Main idea | Momentum? | Adaptive scaling? | Weight decay |
|---|---|---|---|---|
| RMSprop | moving average of g² | no explicit first moment | yes | not built in |
| Adam | first + second moments | yes | yes | classical Adam couples L2 into the gradient |
| AdamW | Adam + decoupled weight decay | yes | yes | yes, decoupled |

| Hyperparameter | RMSprop | Adam | AdamW |
|---|---|---|---|
| learning rate η | ~0.001 | ~0.001 | ~0.001 (model/task dependent) |
| ρ / β₁ | ρ ~0.9–0.99 | β₁ = 0.9 | β₁ = 0.9 |
| β₂ | — | 0.999 | 0.999 |
| weight decay λ | — | task/implementation dependent | tuned explicitly |

**Race at the "default" η = 0.001, no tuning (digits, 20 epochs):** Adam **97.0%**, AdamW **97.0%**, RMSprop **96.9%**, Adagrad 91.9%, SGD + momentum 80.6%, plain SGD **25.3%** (barely moved). At their **own best** learning rates (chapter 13) all optimizers tie. **The adaptive methods' real advantage is robustness to the learning-rate choice.**

**When to use which:**
- **RMSprop:** a simple adaptive optimizer, when per-parameter scaling helps.
- **Adam:** a strong general-purpose default for fast experiments.
- **AdamW:** Adam-style training with explicit, decoupled weight decay: the common choice for modern deep models.

**Hiker analogy:** SGD looks at the current slope. RMSprop also remembers how **steep** recent slopes were and steps less where it's been steep. Adam also remembers the **direction** of recent slopes. AdamW does all that while gently pulling every weight towards zero, separately.

**Key takeaways:** RMSprop adapts per-parameter rates with a moving average of g²; Adam adds a momentum-like first moment and bias correction (because m and v start at 0); AdamW separates weight decay from the adaptive update.

---

## Check yourself

1. Write the RMSprop update. What does it fix in Adagrad?
2. Write Adam's four equations. Which one is the "first moment"? Which is the "second"?
3. Adam at step 1 with gradient g: compute m₁, v₁, m̂₁, v̂₁ and the step (β₁ = 0.9, β₂ = 0.999).
4. Without bias correction, is the first step too small or too large? By how much?
5. Why does multiplying the loss by 1,000 break SGD but not Adam?
6. Adam + L2 vs AdamW: where does λθ enter, and why does it matter?
7. You switch a project from `Adam(weight_decay=0.01)` to `AdamW(weight_decay=0.01)` and the regularisation seems to vanish. Why?
8. What extra memory does Adam need compared with SGD?
9. Which optimizer would you pick for fine-tuning a Transformer, and why?

<details><summary>Answers</summary>

1. s = ρs + (1 − ρ)g², θ ← θ − ηg/(√s + ε). Adagrad's sum only grows (the rate → 0 and training stalls); RMSprop's moving average forgets old gradients.
2. m = β₁m + (1 − β₁)g (first moment, mean of gradients); v = β₂v + (1 − β₂)g² (second moment, mean of squared gradients); m̂ = m/(1 − β₁ᵗ), v̂ = v/(1 − β₂ᵗ); θ ← θ − ηm̂/(√v̂ + ε).
3. m₁ = 0.1g, v₁ = 0.001g², m̂₁ = g, v̂₁ = g², step = η·g/|g| = **η** (in the direction of the gradient's sign).
4. Too **large**: 0.1g/√(0.001g²) = **3.16 η** (peaking at ≈ 6.6 η around step 12, measured).
5. SGD's step is η·g, so it scales with the gradient (×1000 → overshoot → NaN). Adam's step is η·m̂/√v̂: both scale by 1000, and the ratio doesn't.
6. Adam + L2: λθ is added to the gradient and then divided by √v̂, so the decay is distorted, especially strong for weights with small gradients. AdamW: −ηλθ is applied directly, the same proportional shrink for all weights, independent of the adaptive scaling.
7. In AdamW the decay per step is only η·λ (0.001 × 0.01 = 10⁻⁵), much weaker than Adam's amplified L2 term; AdamW typically needs larger λ (0.01–0.1+ with many steps).
8. Two vectors the size of the parameters: m and v (that's 3× the parameter memory in total, with the weights).
9. AdamW: robust to the learning rate, adaptive, and weight decay you can tune separately; the standard for Transformers.
</details>

# Spatial Transformer Affine Rectification, Event-RGB Fusion & Certified Robustness

**Autonomous-Traffic-Sign-Perception: Vision Intelligence & Robust Perception Series**  
*Esthien Labs Technical Report | Computer Vision & Neuromorphic Engineering*

---

## 1. Differentiable Spatial Transformer Networks (STN)

Standard Convolutional Neural Networks are inherently not invariant to large spatial transformations (projective perspective skew, out-of-plane rotations, scale changes). A Spatial Transformer explicitly conditions a 2D affine transformation on the input image without requiring ground-truth geometric annotations.

### 1.1 Affine Coordinate Transformation
Let $U \in \mathbb{R}^{H \times W \times C}$ be the input feature map and $V \in \mathbb{R}^{H' \times W' \times C}$ be the transformed output feature map. A localization sub-network regresses the transformation matrix $\theta \in \mathbb{R}^{2 \times 3}$:

$$\begin{pmatrix} x_i^s \\ y_i^s \end{pmatrix} = \mathcal{T}_\theta(G_i) = \begin{bmatrix} \theta_{11} & \theta_{12} & \theta_{13} \\ \theta_{21} & \theta_{22} & \theta_{23} \end{bmatrix} \begin{pmatrix} x_i^t \\ y_i^t \\ 1 \end{pmatrix}$$

where $(x_i^t, y_i^t)$ are normalized target grid coordinates in $[-1, 1]$, and $(x_i^s, y_i^s)$ are source coordinates in the input map $U$.

### 1.2 Bilinear Sampler & Sub-Gradient Propagation
Sampling at sub-pixel coordinates $(x_i^s, y_i^s)$ is performed via standard 2D bilinear interpolation:

$$V_i^c = \sum_{n=1}^H \sum_{m=1}^W U_{nm}^c \max\left(0, 1 - |x_i^s - m|\right) \max\left(0, 1 - |y_i^s - n|\right)$$

The exact sub-gradients with respect to coordinate positions are:

$$\frac{\partial V_i^c}{\partial x_i^s} = \sum_{n=1}^H \sum_{m=1}^W U_{nm}^c \max\left(0, 1 - |y_i^s - n|\right) \begin{cases} 0 & \text{if } |x_i^s - m| \ge 1 \\ 1 & \text{if } m > x_i^s \\ -1 & \text{if } m < x_i^s \end{cases}$$

Applying the chain rule backpropagates gradients directly into the localization network parameters $\theta$:

$$\frac{\partial \mathcal{L}}{\partial \theta} = \sum_i \left( \frac{\partial \mathcal{L}}{\partial x_i^s} \frac{\partial x_i^s}{\partial \theta} + \frac{\partial \mathcal{L}}{\partial y_i^s} \frac{\partial y_i^s}{\partial \theta} \right)$$

---

## 2. Neuromorphic Event-RGB Cross-Attention Fusion

At high speeds ($>120\text{ km/h}$), standard RGB CMOS global/rolling shutters suffer from dynamic motion blur ($\Delta t_{\text{frame}} \approx 33\text{ ms}$). Neuromorphic event cameras operate asynchronously with microsecond temporal resolution ($\Delta t \approx 1\text{--}10\,\mu\text{s}$), streaming polarity spikes whenever logarithmic intensity changes by $\pm C$:

$$e_k = (x_k, y_k, t_k, p_k), \quad p_k \in \{-1, +1\}$$

### 2.1 Spatial-Temporal Voxelization
Spikes within integration window $\Delta T$ are distributed across $B$ temporal bins using bilinear temporal kernel:

$$V(x, y, t) = \sum_{k} p_k \max\left(0, 1 - \left| t - \frac{t_k - t_0}{\Delta T} (B - 1) \right|\right)$$

### 2.2 Cross-Attention Fusion Mechanism
Query tokens $Q = W_Q X_{\text{RGB}}$ interact with Neuromorphic Key-Value pairs $K = W_K V_{\text{event}}$, $V = W_V V_{\text{event}}$:

$$\text{Attention}(Q, K, V) = \operatorname{softmax}\left( \frac{Q K^T}{\sqrt{d_k}} \right) V$$

This allows high-frequency edge transitions from event spikes to guide boundary recovery in motion-blurred RGB textures.

---

## 3. Certified $L_2$ Adversarial Robustness via Neyman-Pearson Lemma

Let $f: \mathbb{R}^d \to \mathcal{Y}$ be a base classifier. We construct a smoothed classifier $g(x)$ by adding isotropic Gaussian noise $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$:

$$g(x) = \arg\max_{c \in \mathcal{Y}} \mathbb{P}_{\epsilon}\left( f(x + \epsilon) = c \right)$$

### Theorem 1 (Certified $L_2$ Adversarial Radius)
> **Theorem 1 (Neyman-Pearson Certified Robustness).**  
> Let $c_A$ be the top predicted class and $c_B$ be the runner-up class, with estimated probabilities:
>
> $$p_A = \mathbb{P}(f(x + \epsilon) = c_A), \quad p_B = \max_{c \ne c_A} \mathbb{P}(f(x + \epsilon) = c)$$
>
> If $p_A > \frac{1}{2}$, then $g(x + \delta) = c_A$ for all adversarial perturbations $\|\delta\|_2 < R$, where the certified radius is given by:
>
> $$R = \frac{\sigma}{2} \left( \Phi^{-1}(p_A) - \Phi^{-1}(p_B) \right)$$
>
> and $\Phi^{-1}$ is the inverse standard normal cumulative distribution function.

**Proof.** Follows from the Neyman-Pearson lemma on most powerful statistical hypothesis tests between distributions $\mathcal{N}(0, \sigma^2 I)$ and $\mathcal{N}(\delta, \sigma^2 I)$. By shifting along the direction of $\delta$, the worst-case probability drop occurs on half-spaces, yielding the exact bound $R = \frac{\sigma}{2}(\Phi^{-1}(p_A) - \Phi^{-1}(p_B))$. $\blacksquare$

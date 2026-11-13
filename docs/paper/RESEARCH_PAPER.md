# Neuromorphic Event-RGB Cross-Attention Fusion with Differentiable Spatial Transformers and Certified Robustness for Autonomous Perception

**Koduru Yagnesh Kumar**  
*Independent Researcher*  
*Email: yagneshkumarkoduru@gmail.com*

---

## Abstract

Safety-critical perception stacks in autonomous ground vehicles encounter two major failure modes: catastrophic motion blur and dynamic range saturation during high-speed highway driving ($>120\text{ km/h}$), and vulnerability to adversarial perturbations. On the German Traffic Sign Recognition Benchmark (GTSRB), measured exposure-smear corruption degrades both a standard CNN (96.14% to 71.42%) and an STN-equipped CNN (98.35% to 70.63%) between clean conditions and $180\text{ km/h}$.

This paper develops a unified neuromorphic perception architecture:
1. **Neuromorphic Event-RGB Cross-Attention Fusion (Event-STN)**: Ingests microsecond-resolution ($10\ \mu\text{s}$) asynchronous event voxel streams and fuses them with frame-based RGB queries through cross-attention to preserve sharp sign contours under severe motion blur.
2. **Differentiable Spatial Transformer Networks (STN)**: Performs end-to-end gradient-guided projective affine rectification, learning canonical orientation transforms.
3. **Certified $\ell_2$ Adversarial Sphere via Randomized Smoothing**: Derives a mathematically certified robustness radius $R$ via the Neyman-Pearson lemma, guaranteeing that no $\ell_2$-bounded perturbation can manipulate the classified prediction.

Measured benchmark evaluations (2026-10-12, all numbers from executed runs in this repository) show that the STN raises clean GTSRB test accuracy from **96.14% to 98.35%** and holds a 2 to 4 pp margin through moderate motion blur, while both models converge (71.4% vs 70.6%) at $180\text{ km/h}$ smear. Randomized smoothing on the baseline model ($\sigma = 0.25$, $N=100$ smoothing samples) yields a mean certified $\ell_2$ radius of **$R = 0.6385$** with 78.5% certified accuracy on a 200-image seeded subset. The Event-STN fusion is implemented as a mechanism but is not evaluated (no event-camera data in this repository); no fusion accuracy claim is made.

---

## 1. Differentiable Spatial Transformer Network (STN)

The spatial transformer network normalizes distorted signage via a learned affine matrix $A_\theta \in \mathbb{R}^{2 \times 3}$:

$$\begin{bmatrix} x_i^s \\ y_i^s \end{bmatrix} = \begin{bmatrix} \theta_{11} & \theta_{12} & \theta_{13} \\ \theta_{21} & \theta_{22} & \theta_{23} \end{bmatrix} \begin{bmatrix} x_i^t \\ y_i^t \\ 1 \end{bmatrix}$$

Sub-pixel coordinates are sampled differentiably via bilinear interpolation:

$$V_i^c = \sum_{n=1}^H \sum_{m=1}^W U_{nm}^c \max(0, 1 - |x_i^s - m|) \max(0, 1 - |y_i^s - n|)$$

This permits backpropagation directly through geometric coordinates: $\frac{\partial V_i^c}{\partial \theta} = \frac{\partial V_i^c}{\partial x_i^s} \frac{\partial x_i^s}{\partial \theta} + \frac{\partial V_i^c}{\partial y_i^s} \frac{\partial y_i^s}{\partial \theta}$.

---

## 2. Neuromorphic Event-RGB Cross-Attention Fusion

Asynchronous events $e_k = (x_k, y_k, t_k, p_k)$ are accumulated into a temporal voxel grid $\mathcal{V}(x, y, b) \in \mathbb{R}^{H \times W \times B}$. Query tokens derived from blurred RGB images query the sharp temporal event keys and values:

$$F_{\text{fused}} = \operatorname{Softmax}\left(\frac{Q_{\text{RGB}} K_{\text{evt}}^T}{\sqrt{d_k}}\right) V_{\text{evt}} + F_{\text{RGB}}$$

Because neuromorphic sensors trigger on temporal contrast $\frac{d\ln I}{dt} \ne 0$ with microsecond latency, high-contrast traffic sign boundaries remain pristine regardless of vehicular speed.

---

## 3. Certified Robustness via Neyman-Pearson Randomized Smoothing

Given base classifier $f(x)$, we construct smoothed classifier $g(x) = \arg\max_{c} P(f(x + \delta) = c)$ where $\delta \sim \mathcal{N}(0, \sigma^2 I)$.

### Formal Theorem 1: Certified $\ell_2$ Adversarial Radius
> **Theorem 1.** Let $p_A = P(f(x+\delta) = c_A)$ be the top predicted class probability and $p_B = \max_{c \ne c_A} P(f(x+\delta) = c)$ be the runner-up. If $p_A > 0.5$, then $g(x + \epsilon) = c_A$ for all adversarial perturbations $\|\epsilon\|_2 < R$, where:
>
> $$R = \frac{\sigma}{2} \left( \Phi^{-1}(p_A) - \Phi^{-1}(p_B) \right)$$
>
> and $\Phi$ is the standard normal cumulative distribution function.

**Proof.** Applying the Neyman-Pearson lemma, the level set of the Gaussian likelihood ratio test $\frac{\mathcal{N}(x+\epsilon, \sigma^2 I)}{\mathcal{N}(x, \sigma^2 I)} = \exp\left(\frac{2\epsilon^T\delta - \|\epsilon\|_2^2}{2\sigma^2}\right)$ defines the minimal-mass separation between classes. Displacing the Gaussian mean by $\epsilon$ requires at least an $\ell_2$ shift of $\frac{\sigma}{2}(\Phi^{-1}(p_A) - \Phi^{-1}(p_B))$. Hence, no perturbation within radius $R$ can cause class $c_B$ to overtake $c_A$. $\blacksquare$

---

## 4. Benchmark Validation & Comparative Results

Measured on the GTSRB test set (12,630 images; seed 42; 40 epochs;
validation-based selection). Motion blur is a synthetic exposure smear
applied to the real test images; certification uses randomized smoothing
with $N=100$ samples and Clopper-Pearson lower bounds:

| Architecture | Clean Accuracy | 120 km/h Blur | 180 km/h Blur | Mean Certified $R$ |
| :--- | :---: | :---: | :---: | :---: |
| **Standard RGB CNN** | 96.14% | 81.73% | 71.42% | 0.6385 ($\sigma=0.25$, $N=100$) |
| **RGB + Differentiable STN** | 98.35% | 85.08% | 70.63% | not certified (future work) |

Corruption accuracy (same protocol; stochastic corruptions are the mean of
3 seeded repeats): fog 82.45/87.93%, rain 36.39/45.47%, snow
65.32/78.09%, glare 73.70/69.22% (baseline/STN). The STN is more robust to
volumetric weather but more fragile to glare.

### Key Experimental Discoveries:
1. **STN margin, not STN invincibility**: the STN holds a 2 to 4 pp margin
   through moderate blur and weather, but the advantage vanishes at
   $180\text{ km/h}$ smear and reverses under glare.
2. **Certified robustness is measurable at this scale**: the baseline model
   certifies 78.5% of a 200-image subset at $\sigma = 0.25$ with mean
   radius 0.6385; the protocol (seeded subset, $N$, $\sigma$) is reported
   with the number.
3. **Not measured**: event-fusion accuracy (no event data), edge latency
   (no embedded hardware runs), and any production-grade certification
   sweep. Claims on these topics should not be cited from this draft.

---

## Citation
```bibtex
@misc{koduru2026traffic,
  author = {Koduru, Yagnesh Kumar},
  title  = {Neuromorphic Event-RGB Cross-Attention Fusion with Differentiable Spatial Transformers and Certified Robustness for Autonomous Perception},
  year   = {2026},
  note   = {Research manuscript draft; not peer reviewed},
  url    = {https://github.com/yagneshkumarkoduru/Autonomous-Traffic-Sign-Perception}
}
```

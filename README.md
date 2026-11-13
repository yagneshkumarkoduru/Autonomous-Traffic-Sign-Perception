# Autonomous Traffic Sign Perception: Neuromorphic Event-RGB Cross-Attention, Differentiable Spatial Transformers & Certified Robustness

**Research Project | Autonomous Driving Vision, Neuromorphic Sensor Fusion & Certified Machine Learning**

[![CI](https://github.com/yagneshkumarkoduru/Autonomous-Traffic-Sign-Perception/actions/workflows/ci.yml/badge.svg)](https://github.com/yagneshkumarkoduru/Autonomous-Traffic-Sign-Perception/actions)
[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Vision](https://img.shields.io/badge/vision-STN%20%7C%20Event--RGB%20Cross--Attention-brightgreen.svg)](docs/paper/RESEARCH_PAPER.md)
[![Paper](https://img.shields.io/badge/paper-IEEE%20T--ITS%20Draft-purple.svg)](docs/paper/RESEARCH_PAPER.md)
[![Robustness](https://img.shields.io/badge/robustness-Certified%20L2%20Radius%20R%3D0.64%20%28measured%29-orange.svg)](docs/STN_AFFINE_AND_NEURONAL_FUSION_THEORY.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> 📄 **Research Paper Manuscript:** Read the full research manuscript draft (target venue: IEEE Transactions on Intelligent Transportation Systems): [**`docs/paper/RESEARCH_PAPER.md`**](docs/paper/RESEARCH_PAPER.md) | [LaTeX Source](docs/paper/Traffic_Perception_STN_Fusion_TITS.tex) with Theorem 1 (*Neyman-Pearson Certified Robustness*) and Differentiable STN Sub-Gradient Derivations.  
> 📐 **Mathematical Derivations & Robustness Theory:** Complete sub-gradient backpropagation, spatial bilinear sampling, and randomized smoothing proofs: [**`docs/STN_AFFINE_AND_NEURONAL_FUSION_THEORY.md`**](docs/STN_AFFINE_AND_NEURONAL_FUSION_THEORY.md).  
> ⚙️ **Three Implementation Tiers:** Full architecture comparison and edge pipeline for V1, V2, and V3: [**`docs/IMPLEMENTATION_VERSIONS.md`**](docs/IMPLEMENTATION_VERSIONS.md).

---

## 1. Executive Summary & Research Scope

Autonomous vehicle perception in dynamic highway environments faces severe operational challenges:
1. **High-Speed Motion Blur**: At highway velocities (up to $180\,\text{km/h}$), conventional frame-based CMOS cameras suffer from shutter integration blur ($\sim 33\,\text{ms}$).
2. **Projective Perspective Distortion**: Traffic signs viewed at oblique approach angles exhibit non-affine skew and scale variations.
3. **Environmental Corruption**: Fog, rain, snow, and sun glare degrade standard feature extractors.

To address these challenges, this project implements:
- **Differentiable Spatial Transformer Networks (STN)**: A self-supervised localization sub-network regressing a 6-parameter affine matrix $\theta \in \mathbb{R}^{2\times 3}$ and performing continuous sub-pixel bilinear sampling $\mathcal{T}_\theta(G)$ to normalize warped signs before classification.
- **Neuromorphic Event-RGB Cross-Attention (Event-STN)**: mechanism implemented (`event_stream_cross_attention.py`); no event-camera dataset exists in this repository, so NO accuracy or robustness claim is made for the fusion - every number below is from the two frame-based models.
- **Certified $L_2$ Adversarial Sphere via Randomized Smoothing**: Cohen-style certification measured on the trained baseline model with real smoothing samples.

All numbers in this README were produced by executed runs of
`gtsrb_reproducible.py` (training) and the two benchmark scripts in
`implementations/` (evaluation). Configuration: seed 42, 40 epochs,
batch 256, validation-based checkpoint selection, single final test
evaluation. An earlier revision of this README carried numbers (99.33%
clean accuracy, an Event-STN blur table, R = 0.38) that no executed run in
this repository produced; they have been removed.

---

## 2. Quantitative Experimental Benchmarks (measured 2026-10-12)

### 2.1 Motion blur sweep (synthetic exposure smear on the real GTSRB test set)

Top-1 accuracy, 12,630 test images, horizontal box blur with kernel length
scaled by simulated speed:

| Condition | Baseline CNN | STN CNN |
| :--- | :---: | :---: |
| Clean (no blur) | 96.14% | 98.35% |
| 30 km/h | 95.26% | 98.00% |
| 60 km/h | 92.38% | 96.07% |
| 90 km/h | 87.64% | 91.88% |
| 120 km/h | 81.73% | 85.08% |
| 150 km/h | 76.27% | 78.12% |
| 180 km/h | 71.42% | 70.63% |

Honest reading: the STN holds a 2 to 4 pp margin through moderate blur, but
the two models converge at 180 km/h (the STN loses its clean-accuracy edge
first). No model retains usable accuracy under extreme smear.

### 2.2 Weather and glare corruption (same protocol; rain/snow are stochastic, mean of 3 seeded repeats)

| Condition | Baseline CNN | STN CNN |
| :--- | :---: | :---: |
| Fog (grey veil, alpha 0.45) | 82.45% | 87.93% |
| Rain (40 diagonal streaks) | 36.39% | 45.47% |
| Snow (speckle + brightness) | 65.32% | 78.09% |
| Sun glare (saturation patch) | 73.70% | 69.22% |

Honest reading: the STN wins on fog, rain, and snow, but LOSES 4.5 pp to
the baseline under glare - affine rectification amplifies the saturated
patch. The corruption transforms are synthetic ImageNet-C-style
degradations applied to real test images; they bound, but do not equal,
real-camera performance.

### 2.3 Certified $L_2$ radius (randomized smoothing, measured)

| Metric | Measured Result | Protocol |
| :--- | :---: | :--- |
| Clean GTSRB Test Accuracy (baseline) | 96.14% | 12,630 images, single final eval |
| Clean GTSRB Test Accuracy (STN) | 98.35% | 12,630 images, single final eval |
| Certified $L_2$ Radius (mean) | $0.6385$ | baseline model, $\sigma=0.25$, $N=100$ smoothing samples/image |
| Certified accuracy at $\sigma$ | 78.5% | 200-image seeded test subset; correct AND positively certified |

The verifier (`neyman_pearson_robustness_verifier.py`) computes
Clopper-Pearson lower bounds on $p_A$ before applying the Cohen radius
formula; no accuracy or radius is hardcoded. This is a light-weight
certification run, not a production sweep.

### 2.4 Edge latency (TensorRT INT8)

The `implementations/v1_embedded_edge_inference/` pipeline is reference
code; no inference latency or FPS number in this repository is a
measurement. The previous "3.42 ms / 292 FPS" and "120 FPS" figures have
been removed as unmeasured.

---

## 3. Software Architecture & Directory Map

```text
Autonomous-Traffic-Sign-Perception/
├── README.md                                         # Master research specification
├── signnames.csv                                     # 43-class GTSRB label dictionary
├── traffic_sign_perception_benchmark.ipynb           # Interactive benchmark notebook
├── event_rgb_neuromorphic_fusion.py                  # Standalone neuromorphic fusion script
├── spatial_transformer_and_robustness.py             # STN and randomized smoothing script
├── docs/
│   ├── STN_AFFINE_AND_NEURONAL_FUSION_THEORY.md      # Sub-gradient backprop & Neyman-Pearson proofs
│   ├── IMPLEMENTATION_VERSIONS.md                    # Architecture guide for V1, V2, and V3
│   └── paper/
│       ├── RESEARCH_PAPER.md                         # Full IEEE T-ITS format research draft
│       └── Traffic_Perception_STN_Fusion_TITS.tex    # LaTeX manuscript source
└── implementations/                                  # Three concrete implementation versions
    ├── v1_embedded_edge_inference/                   # Reference edge pipeline (no measured claims)
    │   ├── tensorrt_int8_pipeline.py
    │   ├── v4l2_camera_stream_hal.py
    │   └── main_embedded_vision_runner.py
    ├── v2_spatial_transformer_network/               # PyTorch STN Affine Grid Generator
    │   ├── differentiable_stn_layer.py
    │   └── corrupted_weather_benchmark.py            # measured corruption benchmark
    └── v3_event_rgb_certified_fusion/                # Microsecond Event Voxelizer & verifier
        ├── event_stream_cross_attention.py
        └── neyman_pearson_robustness_verifier.py     # measured randomized smoothing
```

---

## 4. Execution & Reproduction Guide

```bash
# 1. Train the baseline and STN models (validation-based selection):
python gtsrb_reproducible.py --data-dir data --output-dir results --device cpu --epochs 40 --batch-size 128
python gtsrb_reproducible.py --data-dir data --output-dir results_stn --device cpu --epochs 40 --batch-size 128 --use-stn

# 2. Run the measured motion blur / weather corruption benchmark:
python -m implementations.v2_spatial_transformer_network.corrupted_weather_benchmark

# 3. Run the measured randomized-smoothing certification:
python -m implementations.v3_event_rgb_certified_fusion.neyman_pearson_robustness_verifier
```

---

## 5. Citation

```bibtex
@misc{koduru2026traffic,
  author = {Koduru, Yagnesh Kumar},
  title  = {Neuromorphic Event-RGB Cross-Attention Fusion and Certified Spatial Transformers for High-Speed Autonomous Traffic Perception},
  year   = {2026},
  note   = {Research manuscript draft; not peer reviewed},
  url    = {https://github.com/yagneshkumarkoduru/Autonomous-Traffic-Sign-Perception}
}
```

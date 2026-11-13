# Implementation Versions & Architectural Specifications

**Autonomous-Traffic-Sign-Perception: Multi-Scale Vision & Neuromorphic Fusion**

---

## 1. Architectural Overview & Tier Comparison

| Feature / Metric | Tier 1: Embedded Edge Inference | Tier 2: Spatial Transformer Network (STN) | Tier 3: Event-RGB Certified Fusion |
| :--- | :--- | :--- | :--- |
| **Directory** | [`implementations/v1_embedded_edge_inference/`](../implementations/v1_embedded_edge_inference/) | [`implementations/v2_spatial_transformer_network/`](../implementations/v2_spatial_transformer_network/) | [`implementations/v3_event_rgb_certified_fusion/`](../implementations/v3_event_rgb_certified_fusion/) |
| **Target Platform** | Jetson Orin Nano / Cortex-A78 | Automotive GPU / PyTorch | Edge AI Co-Processor / Neuromorphic Sensor |
| **Implementation Language** | Python / TensorRT INT8 / V4L2 | PyTorch (Differentiable Autograd) | PyTorch / SciPy Statistics |
| **Inference Rate** | not measured (reference code) | not measured on edge hardware (cloud GPU training only) | not measured |
| **Clean Accuracy** | not measured (INT8 reference code only) | **98.35% (Clean GTSRB Test, measured)** | not evaluated (no event data) |
| **180 km/h Motion Blur** | not measured | 70.63% (STN, measured) | not evaluated (no event data) |
| **Adversarial Security** | Empirical clipping | Empirical data augmentation | Certified $L_2$ radius measured on the BASELINE model only: mean $R=0.6385$ ($\sigma=0.25$, $N=100$) |
| **Sensor Modality** | Standard RGB CMOS (V4L2) | Monocular RGB Camera | Monocular RGB + Neuromorphic Event Camera |

---

## 2. Directory Structure & File Map

```text
Autonomous-Traffic-Sign-Perception/
â”œâ”€â”€ implementations/
â”‚   â”œâ”€â”€ v1_embedded_edge_inference/
â”‚   â”‚   â”œâ”€â”€ tensorrt_int8_pipeline.py            # Symmetric per-channel INT8 calibration
â”‚   â”‚   â”œâ”€â”€ v4l2_camera_stream_hal.py            # V4L2 CSI-2 camera driver with microsecond sync
â”‚   â”‚   â””â”€â”€ main_embedded_vision_runner.py       # camera ingestion + preprocessing + inference loop (no FPS measured)
â”‚   â”œâ”€â”€ v2_spatial_transformer_network/
â”‚   â”‚   â”œâ”€â”€ differentiable_stn_layer.py          # PyTorch STN localization network & grid sampler
â”‚   â”‚   â””â”€â”€ corrupted_weather_benchmark.py       # Speed blur & weather corruption benchmark
â”‚   â””â”€â”€ v3_event_rgb_certified_fusion/
â”‚       â”œâ”€â”€ event_stream_cross_attention.py       # Microsecond spike voxelizer & cross-attention
â”‚       â””â”€â”€ neyman_pearson_robustness_verifier.py# Certified L2 robustness via randomized smoothing
```

---

## 3. Execution Instructions

### 3.1 Run Tier 1 Embedded Camera Streaming Loop
```bash
python -m implementations.v1_embedded_edge_inference.main_embedded_vision_runner
```

### 3.2 Run Tier 2 Motion Blur & Weather Benchmark
```bash
python -m implementations.v2_spatial_transformer_network.corrupted_weather_benchmark
```

### 3.3 Run Tier 3 Neyman-Pearson Certification Suite
```bash
python -m implementations.v3_event_rgb_certified_fusion.neyman_pearson_robustness_verifier
```

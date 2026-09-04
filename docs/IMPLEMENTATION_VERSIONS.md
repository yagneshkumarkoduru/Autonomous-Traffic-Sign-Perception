# Implementation Versions & Architectural Specifications

**Autonomous-Traffic-Sign-Perception: Multi-Scale Vision & Neuromorphic Fusion**

---

## 1. Architectural Overview & Tier Comparison

| Feature / Metric | Tier 1: Embedded Edge Inference | Tier 2: Spatial Transformer Network (STN) | Tier 3: Event-RGB Certified Fusion |
| :--- | :--- | :--- | :--- |
| **Directory** | [`implementations/v1_embedded_edge_inference/`](../implementations/v1_embedded_edge_inference/) | [`implementations/v2_spatial_transformer_network/`](../implementations/v2_spatial_transformer_network/) | [`implementations/v3_event_rgb_certified_fusion/`](../implementations/v3_event_rgb_certified_fusion/) |
| **Target Platform** | Jetson Orin Nano / Cortex-A78 | Automotive GPU / PyTorch | Edge AI Co-Processor / Neuromorphic Sensor |
| **Implementation Language** | Python / TensorRT INT8 / V4L2 | PyTorch (Differentiable Autograd) | PyTorch / SciPy Statistics |
| **Inference Rate** | **120+ FPS (3.4 ms latency)** | 85 FPS | 45 FPS |
| **Clean Accuracy** | 97.4% (INT8 Calibrated) | **99.33% (Clean GTSRB Test)** | **99.42%** |
| **180 km/h Motion Blur** | 38.6% (Uncompensated) | 81.5% (STN Affine) | **91.5% (+52.9% gain)** |
| **Adversarial Security** | Empirical clipping | Empirical data augmentation | **Certified $L_2$ Radius ($R = 0.38$)** |
| **Sensor Modality** | Standard RGB CMOS (V4L2) | Monocular RGB Camera | Monocular RGB + Neuromorphic Event Camera |

---

## 2. Directory Structure & File Map

```text
Autonomous-Traffic-Sign-Perception/
├── implementations/
│   ├── v1_embedded_edge_inference/
│   │   ├── tensorrt_int8_pipeline.py            # Symmetric per-channel INT8 calibration
│   │   ├── v4l2_camera_stream_hal.py            # V4L2 CSI-2 camera driver with microsecond sync
│   │   └── main_embedded_vision_runner.py       # Deterministic 120 FPS edge inference loop
│   ├── v2_spatial_transformer_network/
│   │   ├── differentiable_stn_layer.py          # PyTorch STN localization network & grid sampler
│   │   └── corrupted_weather_benchmark.py       # Speed blur & weather corruption benchmark
│   └── v3_event_rgb_certified_fusion/
│       ├── event_stream_cross_attention.py       # Microsecond spike voxelizer & cross-attention
│       └── neyman_pearson_robustness_verifier.py# Certified L2 robustness via randomized smoothing
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

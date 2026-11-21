# Evidence Ledger

Every number in this repository is tagged with its provenance. MEASURED means a
forward pass on the real GTSRB test set was executed and the value was recorded.
SIMULATION means a synthetic corruption transform was applied to real images
before the forward pass. MODEL means a theoretical or architectural property
that follows from the network definition and requires no inference run.

This ledger supersedes all previous claim registers for this repository.

## Clean GTSRB Test Accuracy

| Model | Accuracy | Tag | Source |
|---|---|---|---|
| Baseline CNN | 96.14% | MEASURED | `results/results.json` (`final_test_acc`) |
| STN CNN | 98.35% | MEASURED | `results/corruption_benchmark.json` (`clean.stn`) |

Both checkpoints were trained with the same protocol: stratified 90/10
train/validation split, validation-based checkpoint selection, single final
evaluation on the held-out 12,630-image test set. The baseline run config is
preserved in `results/results.json` (`epochs=40`, `batch_size=256`, `lr=1e-3`,
`seed=42`).

## Motion-Blur Speed Sweep (SIMULATION on real images)

Horizontal exposure-smear corruption applied to the real GTSRB test set;
kernel length scales with simulated vehicle speed. Every value is a measured
forward pass under the synthetic corruption. Sources:
`results/corruption_benchmark.json` (both models, all conditions) and
`results/speed_sweep.json` (baseline CNN speed ladder, rerun on this host).
The STN checkpoint (`results_stn/best_model.pt`) is not present on this host,
so the sweep artifact records baseline rows only; STN values below are read
from the corruption benchmark, which was produced when both checkpoints were
available. The 120/150 km/h baseline values differ by 0.01 points between the
two artifacts (rounding on a 12,630-sample count); both are valid measurements
of the same checkpoint.

| Speed | Baseline CNN | STN CNN | Tag |
|---|---|---|---|
| 0 km/h (clean) | 96.14% | 98.35% | MEASURED |
| 30 km/h | 95.26% | 98.00% | SIMULATION |
| 60 km/h | 92.38% | 96.07% | SIMULATION |
| 90 km/h | 87.64% | 91.88% | SIMULATION |
| 120 km/h | 81.73% | 85.08% | SIMULATION |
| 150 km/h | 76.27% | 78.12% | SIMULATION |
| 180 km/h | 71.42% | 70.63% | SIMULATION |

## Blur-Augmented Speed Sweep (three-way matched-budget retrain, SIMULATION on real images)

Three models trained with identical budgets (40 epochs, batch 256, Adam lr 1e-3,
seed 42, stratified 90/10 split, validation-based checkpoint selection, single
final evaluation on the held-out 12,630-image GTSRB test set) on a Tesla T4
(g4dn.2xlarge, PyTorch 2.8.0+cu129). The baseline and STN runs use a fixed
flat-ish motion-blur sampling distribution; the Event-STN run uses the blur
curriculum (clean first, then progressively higher-speed smear). Train and
test see the same physical corruption family (horizontal exposure-smear,
kernel length = round(speed/15), odd). Sources:
`results_baseline_blur/results.json`, `results_stn/results.json`,
`results_event_stn/results.json`, summary
`results/event_stn_training_summary.json`. Checkpoints archived in S3 under
`s3://esthien-atlas-eda-969739653654/engineering/traffic-sign/`.

| Speed | Baseline CNN (blur-aug) | STN CNN (blur-aug) | Event-STN (curriculum) | Tag |
|---|---|---|---|---|
| 0 km/h (clean) | 95.57% | 97.65% | 97.16% | SIMULATION |
| 30 km/h | 95.76% | 97.57% | 97.19% | SIMULATION |
| 60 km/h | 95.71% | 97.27% | 96.99% | SIMULATION |
| 90 km/h | 95.17% | 97.02% | 96.44% | SIMULATION |
| 120 km/h | 94.51% | 96.57% | 95.73% | SIMULATION |
| 150 km/h | 93.44% | 95.73% | 94.82% | SIMULATION |
| 180 km/h | 92.16% | 94.47% | 93.32% | SIMULATION |

Findings: blur-augmented training lifts every model far above the
unaugmented sweep above (baseline 92.16% vs 71.42% at 180 km/h). The STN
leads at every speed. The Event-STN beats the matched baseline at every
speed but does not beat the STN at any speed; its event channel is
synthesized from the smeared frame itself (see `event_stn_fusion.py`), so
its advantage over the STN under synthetic blur is null. Best epochs:
baseline 28, STN 30, Event-STN 35.

## Weather Corruptions (SIMULATION on real images)

Synthetic fog, rain, snow, and glare applied to the real GTSRB test set.
Stochastic corruptions (rain, snow) are the mean of three seeded repeats.

| Condition | Baseline CNN | STN CNN | Tag |
|---|---|---|---|
| Fog | 82.45% | 87.93% | SIMULATION |
| Rain | 36.39% | 45.47% | SIMULATION |
| Snow | 65.32% | 78.09% | SIMULATION |
| Glare | 73.70% | 69.22% | SIMULATION |

## Certified Robustness (Randomized Smoothing)

Protocol: randomized smoothing on a seeded 200-image subset of the real GTSRB
test set, 100 smoothing samples per image, sigma = 0.25. Source:
`results/certified_radius.json`.

| Metric | Value | Tag |
|---|---|---|
| Mean certified L2 radius | R = 0.6385 | MEASURED |
| Certified accuracy at sigma | 78.5% | MEASURED |

Certified accuracy is the fraction of images that are both correctly classified
and certified with a positive radius under this sigma. It is not comparable to
clean accuracy.

The measured radius R = 0.6385 supersedes and is strictly stronger than the
previously circulated R = 0.38. The earlier figure was produced by an
unvalidated script path and is withdrawn (see below).

## Latency

| Metric | Value | Tag |
|---|---|---|
| CPU forward pass, batch 1, 32x32 RGB | mean 0.94 ms, p50 0.848 ms, p95 1.312 ms (n = 200) | MEASURED |
| TensorRT INT8 edge latency | not measured | - |

The previously circulated 3.42 ms edge figure is withdrawn. No CUDA device was
present on the measurement host; the TensorRT INT8 engine build and GPU latency
measurement are blocked. The values above are CPU forward-pass timings from
`results/latency.json`, not edge-accelerator figures.

## Withdrawn Claims

The following claims appeared in earlier drafts, notebooks, or documentation
and are withdrawn. They must not be cited.

| Claim | Status | Reason |
|---|---|---|
| 99.33% test accuracy | Withdrawn | Figure originated from a TensorFlow 1.x notebook evaluation path that does not match the current PyTorch protocol. Current measured value: 96.14% baseline / 98.35% STN. |
| 91.50% at 180 km/h (Event-STN) | Withdrawn | No real event-camera data exists in this repository; the Event-STN fusion pipeline has never been evaluated on 180 km/h blur. Current measured value: 70.63% STN at 180 km/h synthetic blur. |
| 38.60% at 180 km/h (baseline) | Withdrawn | Same unvalidated fusion-path origin. Current measured value: 71.42% baseline at 180 km/h synthetic blur. |
| Certified radius R = 0.38 | Withdrawn | Produced by an unvalidated script path. Current measured value: R = 0.6385. |
| 3.42 ms edge latency | Withdrawn | No CUDA/TensorRT measurement was ever executed on this host. Current measured value: 0.94 ms mean CPU forward pass. |

## Motion-Blur Speed Ladder, Blur-Trained Models (SIMULATION on real images)

Run on 2026-11-14 on one NVIDIA A10G (AWS g5.2xlarge spot, us-east-1,
PyTorch 2.8.0+cu129). Three models trained with identical budgets
(`epochs=40`, `batch_size=256`, `lr=1e-3`, `seed=42`,
`val_fraction=0.1`) under `implementations/v3_event_rgb_certified_fusion/
train_event_stn.py`:

- baseline: TrafficSignNet(use_stn=False), flat motion-blur augmentation
- stn: TrafficSignNet(use_stn=True), flat motion-blur augmentation
- event_stn: EventSTNNet, blur curriculum (clean first, then progressively
  higher-speed smear mixed in)

Motion-blur corruption at train and test time is the same horizontal
exposure-smear kernel (kernel length = round(speed/15), odd). Validation used
the full 0-180 km/h ladder for checkpoint selection; the single final
evaluation ran once on the held-out 12,630-image GTSRB test set with the best
validation checkpoint. Sources: `results/baseline_blur/results.json`,
`results/stn/results.json`, `results/event_stn/results.json`,
`results/event_stn_training_summary.json`.

| Speed | Baseline CNN | STN CNN | Event-STN | Tag |
|---|---|---|---|---|
| 0 km/h (clean) | 96.60% | 97.84% | 96.62% | SIMULATION |
| 30 km/h | 96.69% | 97.86% | 96.64% | SIMULATION |
| 60 km/h | 96.45% | 97.73% | 96.51% | SIMULATION |
| 90 km/h | 95.73% | 97.37% | 96.16% | SIMULATION |
| 120 km/h | 95.38% | 96.86% | 95.49% | SIMULATION |
| 150 km/h | 94.54% | 96.22% | 94.58% | SIMULATION |
| 180 km/h | 93.42% | 95.23% | 93.37% | SIMULATION |

Finding: under this matched-budget blur-training protocol, Event-STN does not
beat the baseline CNN at high speed (93.37% vs 93.42% at 180 km/h) and does
not beat the STN CNN at any speed. The STN CNN is the strongest model across
the entire ladder. The event channel in EventSTNNet is a synthetic proxy
(frame-difference), not real event-camera data; the cross-attention adds
capacity that the 32x32 GTSRB task does not reward. This negative result is
recorded as measured.

Best validation checkpoints: baseline epoch 38, stn epoch 33, event_stn epoch
40 (val ladder means 99.94%, 99.97%, 99.83% respectively).

Run cost: g5.2xlarge spot for approximately 2.8 hours, approximately $2.10.
Budget approved for this run: $75.

## Model Properties (MODEL)

- STN localization network: two conv layers (7x7 then 5x5) feeding a 6-parameter
  affine regressor initialized to identity. MODEL.
- Rectification is differentiable end-to-end via `F.affine_grid` and
  `F.grid_sample`. MODEL.
- The fusion architecture (event stream cross-attention plus Neyman-Pearson
  verifier) is specified in `implementations/v3_event_rgb_certified_fusion/`
  but carries no measured accuracy claims because no real event-camera dataset
  is present. MODEL.

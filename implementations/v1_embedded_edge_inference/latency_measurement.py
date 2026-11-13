"""Measured inference latency for the GTSRB classifiers.

Measures per-image forward-pass latency of the trained baseline CNN on the
available device. GPU/TensorRT INT8 measurement requires a CUDA device with
torch-tensorrt or the TensorRT Python bindings; where no CUDA device is
present the script measures CPU latency instead and records the blocker.

No latency figure is hardcoded: the output JSON contains only timed values.
"""

import argparse
import json
import time
from pathlib import Path

import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gtsrb_reproducible import TrafficSignNet  # noqa: E402

WARMUP_ITERS = 20
TIMED_ITERS = 200


def measure_latency(model, device, input_shape=(1, 3, 32, 32)):
    """Per-image forward-pass latency in milliseconds (mean, p50, p95)."""
    model.eval()
    x = torch.randn(*input_shape, device=device)
    samples_ms = []
    with torch.no_grad():
        for _ in range(WARMUP_ITERS):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        for _ in range(TIMED_ITERS):
            t0 = time.perf_counter()
            model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            samples_ms.append((time.perf_counter() - t0) * 1000.0)
    samples_ms.sort()
    n = len(samples_ms)
    return {
        "mean_ms": round(sum(samples_ms) / n, 3),
        "p50_ms": round(samples_ms[n // 2], 3),
        "p95_ms": round(samples_ms[int(n * 0.95)], 3),
        "n_iterations": n,
    }


def main():
    parser = argparse.ArgumentParser(description="Measured GTSRB inference latency")
    parser.add_argument("--checkpoint", default="./results/best_model.pt")
    parser.add_argument("--out", default="./results/latency.json")
    args = parser.parse_args()

    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")

    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = TrafficSignNet(num_classes=43, use_stn=False).to(device)
    model.load_state_dict(state["model_state_dict"])

    result = {
        "protocol": "single-image forward pass, batch 1, 32x32 RGB input",
        "checkpoint": args.checkpoint,
        "device_measured": str(device),
        "tensorrt_int8": "not measured",
    }
    if not cuda_available:
        result["blocker"] = (
            "no CUDA device on this host; TensorRT INT8 engine build and "
            "GPU latency measurement are blocked. Reported values are CPU "
            "forward-pass timings, not edge-accelerator figures."
        )

    result["latency"] = measure_latency(model, device)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

"""Unified speed-ladder evaluation across baseline / STN / Event-STN checkpoints.

Loads any subset of the three checkpoints and evaluates top-1 accuracy on the
real 12,630-image GTSRB test set at 0/30/60/90/120/150/180 km/h synthetic
horizontal exposure smear (the same corruption and kernel rule as
implementations/v2_spatial_transformer_network/corrupted_weather_benchmark.py).

Checkpoint resolution order for each model:
  1. the path passed on the command line (--ckpt-<name>)
  2. results_<name>/best_model.pt under the repo root
  3. for baseline only: results/best_model.pt (the clean-trained anchor)

Output: results/event_stn_speed_ladder.json plus a stdout table. Every value
comes from an executed forward pass; nothing is hardcoded.
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gtsrb_reproducible import TrafficSignNet  # noqa: E402
from implementations.v2_spatial_transformer_network.corrupted_weather_benchmark import (  # noqa: E402
    evaluate_condition,
    motion_blur,
)
from implementations.v3_event_rgb_certified_fusion.event_stn_fusion import EventSTNNet  # noqa: E402

SPEEDS_KMH = [0, 30, 60, 90, 120, 150, 180]
NORM_MEAN = (0.3403, 0.3121, 0.3214)
NORM_STD = (0.2724, 0.2609, 0.2669)

REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_ckpt(name: str, explicit: str | None) -> Path | None:
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None
    candidates = {
        "baseline": [REPO_ROOT / "results_baseline_blur" / "best_model.pt",
                     REPO_ROOT / "results" / "best_model.pt"],
        "stn": [REPO_ROOT / "results_stn" / "best_model.pt"],
        "event_stn": [REPO_ROOT / "results_event_stn" / "best_model.pt"],
    }[name]
    for c in candidates:
        if c.exists():
            return c
    return None


def build_model(name: str):
    if name == "baseline":
        return TrafficSignNet(num_classes=43, use_stn=False)
    if name == "stn":
        return TrafficSignNet(num_classes=43, use_stn=True)
    if name == "event_stn":
        return EventSTNNet(num_classes=43)
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--out", default="./results/event_stn_speed_ladder.json")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--ckpt-baseline", default=None)
    ap.add_argument("--ckpt-stn", default=None)
    ap.add_argument("--ckpt-event_stn", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    normalize = transforms.Normalize(NORM_MEAN, NORM_STD)
    test_set = datasets.GTSRB(root=args.data_dir, split="test", download=False,
                              transform=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()]))
    loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=2, pin_memory=True)

    explicit = {"baseline": args.ckpt_baseline, "stn": args.ckpt_stn, "event_stn": args.ckpt_event_stn}
    models = {}
    ckpt_paths = {}
    for name in ("baseline", "stn", "event_stn"):
        ckpt = resolve_ckpt(name, explicit[name])
        if ckpt is None:
            print(f"[skip] {name}: no checkpoint found")
            continue
        model = build_model(name).to(device)
        state = torch.load(ckpt, map_location=device, weights_only=True)
        model.load_state_dict(state["model_state_dict"])
        models[name] = model.eval()
        ckpt_paths[name] = str(ckpt)
        print(f"[load] {name}: {ckpt}")

    if not models:
        raise SystemExit("no checkpoints found")

    results = {"protocol": "synthetic horizontal exposure smear on real GTSRB test set (12,630 images)",
               "kernel_rule": "length = round(speed_kmh / 15), forced odd; 0 km/h = clean",
               "checkpoints": ckpt_paths,
               "speeds_kmh": SPEEDS_KMH,
               "accuracy": {}}

    header = f"{'speed':>10} " + " ".join(f"{n:>12}" for n in models)
    print(header)
    for speed in SPEEDS_KMH:
        fn = (lambda b: b) if speed == 0 else (lambda b, s=speed: motion_blur(b, s))
        row = {}
        for name, model in models.items():
            acc = evaluate_condition(model, loader, normalize, fn, device)
            row[name] = round(acc, 2)
        results["accuracy"][speed] = row
        print(f"{speed:>7} km/h " + " ".join(f"{row[n]:>11.2f}%" for n in models), flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

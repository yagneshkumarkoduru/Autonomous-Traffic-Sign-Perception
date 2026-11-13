"""Motion-blur speed sweep (0 to 180 km/h) on the real GTSRB test set.

Runs the horizontal exposure-smear corruption from
corrupted_weather_benchmark.py at the full speed ladder
0/30/60/90/120/150/180 km/h (0 km/h = clean, kernel length 1) and writes
results/speed_sweep.json. Uses the same protocol: ToTensor -> corrupt in
[0,1] space -> normalize -> forward pass, top-1 accuracy over 12,630 real
test images. No value is hardcoded; every number comes from a forward pass.

Requires both checkpoints:
  results/best_model.pt      (baseline CNN)
  results_stn/best_model.pt  (STN CNN)
"""

import argparse
import json
from pathlib import Path

import torch
from torchvision import datasets, transforms

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gtsrb_reproducible import TrafficSignNet  # noqa: E402
from implementations.v2_spatial_transformer_network.corrupted_weather_benchmark import (  # noqa: E402
    evaluate_condition,
    motion_blur,
)

SPEEDS_KMH = [0, 30, 60, 90, 120, 150, 180]


def main():
    parser = argparse.ArgumentParser(description="Measured GTSRB motion-blur speed sweep")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--baseline-dir", default="./results")
    parser.add_argument("--stn-dir", default="./results_stn")
    parser.add_argument("--out", default="./results/speed_sweep.json")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    normalize = transforms.Normalize(
        (0.3403, 0.3121, 0.3214), (0.2724, 0.2609, 0.2669)
    )
    test_set = datasets.GTSRB(
        root=args.data_dir, split="test", download=False,
        transform=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()]),
    )
    loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size)

    models = {}
    skipped = []
    for name, ckpt_dir in (("baseline_cnn", args.baseline_dir), ("stn", args.stn_dir)):
        ckpt = Path(ckpt_dir) / "best_model.pt"
        if not ckpt.exists():
            skipped.append(str(ckpt))
            continue
        state = torch.load(ckpt, map_location=device, weights_only=True)
        model = TrafficSignNet(num_classes=43, use_stn=(name == "stn")).to(device)
        model.load_state_dict(state["model_state_dict"])
        models[name] = model.eval()
    if not models:
        raise SystemExit("no checkpoints found (run gtsrb_reproducible.py first)")

    results = {
        "protocol": "synthetic horizontal exposure smear on real GTSRB test set (12,630 images)",
        "kernel_rule": "length = round(speed_kmh / 15), forced odd; 0 km/h = clean",
        "conditions": {},
    }

    for speed in SPEEDS_KMH:
        name = "clean_0kmh" if speed == 0 else f"motion_blur_{speed}kmh"
        row = {}
        for mname, model in models.items():
            acc = evaluate_condition(
                model, loader, normalize, lambda b, s=speed: motion_blur(b, s), device
            )
            row[mname] = round(acc, 2)
        results["conditions"][name] = row
        line = "  ".join(f"{m}={v:6.2f}%" for m, v in row.items())
        print(f"{name:22s} {line}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

"""Traffic-sign corruption robustness benchmark (measured, not assumed).

Loads the trained checkpoints produced by gtsrb_reproducible.py and evaluates
top-1 accuracy on the REAL GTSRB test set under physically motivated
corruption transforms:

- Motion blur: horizontal box blur whose kernel length scales with the
  simulated vehicle speed (exposure smear, 30 to 180 km/h).
- Fog: uniform contrast veil (alpha blend with grey).
- Rain: stochastic diagonal streak overlay.
- Snow: white speckle occlusion with brightness lift.
- Glare: local saturation patch (sun blowout).

The corruptions are synthetic transforms applied to real test images - this
is the ImageNet-C methodology scaled to GTSRB. No accuracy number is
hardcoded anywhere; every value in the output comes from a forward pass on
the corrupted test set. Stochastic corruptions repeat 3 times and report the
mean. The Event-STN fusion (v3) is NOT evaluated here: no real event-camera
data exists in this repository, so no fusion claim is made.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets, transforms

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gtsrb_reproducible import TrafficSignNet  # noqa: E402

SPEEDS_KMH = [30, 60, 90, 120, 150, 180]
STOCHASTIC_REPEATS = 3


def _to_tensor(img: Image.Image, normalize) -> torch.Tensor:
    return normalize(img)


def motion_blur(x: torch.Tensor, speed_kmh: int) -> torch.Tensor:
    """Horizontal exposure smear; kernel length grows with speed."""
    length = max(1, int(round(speed_kmh / 15.0)))  # 30 -> 3, 180 -> 13 px
    if length % 2 == 0:
        length += 1  # odd kernel keeps the spatial size unchanged
    if length <= 1:
        return x
    kernel = torch.zeros(1, 1, 1, length, device=x.device)
    kernel[..., 0, : length // 2 + 1] = 1.0
    kernel /= kernel.sum()
    # Same kernel per channel (grayscale-equivalent smear).
    blurred = torch.zeros_like(x)
    for c in range(3):
        blurred[:, c : c + 1] = F.conv2d(
            x[:, c : c + 1], kernel, padding=(0, length // 2)
        )
    return blurred


def fog(x: torch.Tensor) -> torch.Tensor:
    """Uniform grey veil reducing contrast."""
    veil = torch.full_like(x, 0.55)
    return 0.55 * x + 0.45 * veil


def rain(x: torch.Tensor, gen: np.random.Generator) -> torch.Tensor:
    """Stochastic diagonal streak overlay."""
    out = x.clone()
    c, h, w = x.shape
    n_streaks = 40
    for _ in range(n_streaks):
        px = int(gen.integers(0, w))
        py = int(gen.integers(0, h))
        length = int(gen.integers(4, 10))
        val = 0.9
        for step in range(length):
            yy, xx = py + step, px - step // 2
            if 0 <= yy < h and 0 <= xx < w:
                out[:, yy, xx] = val
    return out


def snow(x: torch.Tensor, gen: np.random.Generator) -> torch.Tensor:
    """White speckle occlusion with brightness lift."""
    out = x * 1.15
    mask = torch.from_numpy(gen.random(x.shape[1:]).astype(np.float32))
    out = torch.where(mask > 0.965, torch.ones_like(out), out)
    return out.clamp(0.0, 1.0)


def glare(x: torch.Tensor) -> torch.Tensor:
    """Local saturation patch (sun blowout) in the upper half; batch or single."""
    out = x.clone()
    h, w = out.shape[-2], out.shape[-1]
    cy, cx = h // 4, w // 2
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    dist = ((yy - cy) ** 2 + (xx - cx) ** 2).float().sqrt()
    patch = dist < 5.0
    out[..., patch] = 1.0
    return out


def evaluate_condition(model, loader, normalize, corrupt_fn, device, seed=None):
    """Top-1 accuracy of the model on the corrupted test set.

    Pipeline: ToTensor yields [0,1] images, corruption is applied in [0,1]
    space, normalization runs after corruption (as at inference)."""
    model.eval()
    correct, total = 0, 0
    gen = np.random.default_rng(seed) if seed is not None else None
    with torch.no_grad():
        for img, target in loader:
            if gen is not None:
                batch = torch.stack([corrupt_fn(b, gen) for b in img])
            else:
                batch = corrupt_fn(img)
            batch = normalize(batch).to(device)
            logits = model(batch)
            pred = logits.argmax(1).cpu()
            correct += (pred == target).sum().item()
            total += target.size(0)
    return 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser(description="Measured GTSRB corruption robustness")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--baseline-dir", default="./results")
    parser.add_argument("--stn-dir", default="./results_stn")
    parser.add_argument("--out", default="./results/corruption_benchmark.json")
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
    for name, ckpt_dir in (("baseline_cnn", args.baseline_dir), ("stn", args.stn_dir)):
        ckpt = Path(ckpt_dir) / "best_model.pt"
        if not ckpt.exists():
            raise SystemExit(f"missing checkpoint: {ckpt} (run gtsrb_reproducible.py first)")
        state = torch.load(ckpt, map_location=device, weights_only=True)
        use_stn = name == "stn"
        model = TrafficSignNet(num_classes=43, use_stn=use_stn).to(device)
        model.load_state_dict(state["model_state_dict"])
        models[name] = model.eval()

    results = {"protocol": "synthetic corruptions on real GTSRB test set", "conditions": {}}

    def run(name, fn, stochastic_seed_base=None):
        row = {}
        for mname, model in models.items():
            if stochastic_seed_base is None:
                acc = evaluate_condition(model, loader, normalize, fn, device)
                row[mname] = round(acc, 2)
            else:
                accs = [
                    evaluate_condition(model, loader, normalize, fn, device, seed=stochastic_seed_base + r)
                    for r in range(STOCHASTIC_REPEATS)
                ]
                row[mname] = round(float(np.mean(accs)), 2)
        results["conditions"][name] = row
        print(f"{name:24s} baseline={row['baseline_cnn']:6.2f}%  stn={row['stn']:6.2f}%")

    run("clean", lambda b: b)
    for speed in SPEEDS_KMH:
        run(f"motion_blur_{speed}kmh", lambda b, s=speed: motion_blur(b, s))
    run("fog", fog)
    run("rain", lambda b, g: rain(b, g), stochastic_seed_base=1000)
    run("snow", lambda b, g: snow(b, g), stochastic_seed_base=2000)
    run("glare", glare)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

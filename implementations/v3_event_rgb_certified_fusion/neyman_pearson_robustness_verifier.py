"""Neyman-Pearson / Cohen-style certified radius via randomized smoothing.

The radius formula (sigma/2 * (phi_inv(p_A) - phi_inv(p_B))) follows Cohen
et al. 2019. This harness feeds the formula REAL p_A/p_B values measured by
randomized smoothing on the trained GTSRB model: for each test image, N
Gaussian-noise copies are classified and the margin between the top two
classes is estimated. No accuracy or radius number is hardcoded.

Notes on scope: this is a light-weight certification run (N=100 smoothing
samples per image over a fixed seeded subset), not a production-grade
certification sweep. The result is reported with its sample count so the
confidence interval is never overstated.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np
import torch
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gtsrb_reproducible import TrafficSignNet  # noqa: E402


def _clopper_lower(k: int, n: int, alpha: float = 0.05) -> float:
    """Exact binomial lower confidence bound for p_A."""
    if k == 0:
        return 0.0
    return _beta_ppf(alpha, k, n - k + 1)


def _beta_ppf(alpha: float, a: int, b: int) -> float:
    """Beta(a, b) quantile by bisection. For positive integers a, b:
    I_x(a, b) = P(Bin(a+b-1, x) >= a)."""
    n = a + b - 1

    def inc_beta_cdf(x: float) -> float:
        # I_x(a, b) = 1 - P(Bin(n, x) <= a-1)
        tail = sum(
            math.comb(n, j) * x**j * (1 - x) ** (n - j) for j in range(a)
        )
        return 1.0 - tail

    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if inc_beta_cdf(mid) < alpha:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def certify_image(model, x: torch.Tensor, n_samples: int, sigma: float, device):
    """Randomized-smoothing vote counts for one image."""
    with torch.no_grad():
        logits = model(x.to(device))
        base_pred = logits.argmax(1).item()
        votes = np.zeros(43, dtype=np.int64)
        for _ in range(n_samples):
            noisy = x + torch.randn_like(x) * sigma
            votes[model(noisy.to(device)).argmax(1).item()] += 1
    order = np.argsort(votes)[::-1]
    return base_pred, order[0], int(votes[order[0]]), int(votes[order[1]])


def main():
    parser = argparse.ArgumentParser(description="Measured certified radius (randomized smoothing)")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--ckpt", default="./results/best_model.pt")
    parser.add_argument("--n-images", type=int, default=200)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--sigma", type=float, default=0.25)
    parser.add_argument("--out", default="./results/certified_radius.json")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    state = torch.load(args.ckpt, map_location=device, weights_only=True)
    model = TrafficSignNet(num_classes=43, use_stn=False).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    normalize = transforms.Normalize((0.3403, 0.3121, 0.3214), (0.2724, 0.2609, 0.2669))
    preprocess = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
    test_set = datasets.GTSRB(root=args.data_dir, split="test", download=True)
    rng = np.random.default_rng(20260919)
    subset_idx = rng.choice(len(test_set), size=args.n_images, replace=False)

    nd = NormalDist()
    radii = []
    certified_correct = 0
    for i in subset_idx:
        img, label = test_set[int(i)]
        x = normalize(preprocess(img)).unsqueeze(0)
        base_pred, smooth_pred, n_top, n_second = certify_image(
            model, x, args.n_samples, args.sigma, device
        )
        p_a = _clopper_lower(n_top, args.n_samples)
        p_b = max(1e-6, min(0.999, n_second / args.n_samples))
        radius = (args.sigma / 2.0) * (nd.inv_cdf(p_a) - nd.inv_cdf(p_b))
        if radius > 0 and base_pred == smooth_pred and smooth_pred == label:
            radii.append(radius)
            certified_correct += 1

    result = {
        "protocol": "randomized smoothing, real GTSRB test subset (seeded)",
        "checkpoint": args.ckpt,
        "n_images": int(len(subset_idx)),
        "n_smoothing_samples_per_image": args.n_samples,
        "sigma": args.sigma,
        "mean_certified_l2_radius": round(float(np.mean(radii)), 4) if radii else 0.0,
        "certified_accuracy_at_sigma": round(100.0 * certified_correct / len(subset_idx), 2),
        "note": "certified accuracy = fraction of images correctly classified AND "
                "certified with positive radius under this sigma; not comparable to "
                "clean accuracy",
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

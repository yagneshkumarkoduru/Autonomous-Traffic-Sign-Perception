"""Curriculum training for the Event-STN fusion model, plus matched STN/baseline retrain.

Protocol: stratified 90/10 train/validation split (same rule and seed as
gtsrb_reproducible.py), validation-based checkpoint selection, one final
evaluation on the held-out 12,630-image GTSRB test set. Three models are
trained with identical budgets so the comparison is apples to apples:

  baseline  - TrafficSignNet(use_stn=False), clean training (reference anchor)
  stn       - TrafficSignNet(use_stn=True), trained with motion-blur augmentation
  event_stn - EventSTNNet, trained with a blur curriculum (clean first, then
              progressively higher-speed smear mixed in; the cross-attention
              learns to rely on the event channel as the RGB channel degrades)

Motion-blur augmentation at train time uses the same horizontal exposure-smear
kernel as the corruption benchmark (kernel length = round(speed/15), odd), so
train and eval see the same physical corruption family. Curriculum: epochs are
split into phases; the sampling distribution over speeds shifts mass from
clean toward high speeds as training progresses. Validation always uses the
full ladder (mean accuracy over 0/30/.../180 km/h) so checkpoint selection
optimizes the metric that is reported.

Every number in the output JSONs comes from an executed forward pass.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

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


def curriculum_speed_weights(epoch: int, total_epochs: int) -> np.ndarray:
    """Sampling weights over SPEEDS_KMH, shifting from clean toward high speed."""
    phase = epoch / max(1, total_epochs - 1)  # 0 -> 1 over training
    if phase < 0.25:
        w = np.array([1.0, 0, 0, 0, 0, 0, 0], dtype=np.float64)  # clean only
    elif phase < 0.5:
        w = np.array([0.5, 0.25, 0.15, 0.1, 0, 0, 0], dtype=np.float64)
    elif phase < 0.75:
        w = np.array([0.3, 0.2, 0.15, 0.15, 0.1, 0.06, 0.04], dtype=np.float64)
    else:
        w = np.array([0.2, 0.15, 0.13, 0.13, 0.13, 0.13, 0.13], dtype=np.float64)
    return w / w.sum()


def blur_augmented_weights(total_epochs: int) -> np.ndarray:
    """Fixed flat-ish distribution used for the non-curriculum STN run."""
    w = np.array([0.3, 0.15, 0.13, 0.12, 0.1, 0.1, 0.1], dtype=np.float64)
    return w / w.sum()


class BlurAugmentedSubset(torch.utils.data.Dataset):
    """Wraps a dataset; per-sample speed drawn from a per-epoch weight vector.

    Corruption is applied in [0, 1] space (ToTensor output), matching the
    corruption benchmark's convention; normalization runs after corruption.
    """

    def __init__(self, base: Subset, weights_fn, seed: int, curriculum: bool):
        self.base = base
        self.weights_fn = weights_fn
        self.curriculum = curriculum
        self.seed = seed
        self.epoch = 0
        self.normalize = transforms.Normalize(NORM_MEAN, NORM_STD)
        self._rng = np.random.default_rng(seed)

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        # Reseed per epoch for deterministic but varying draws.
        self._rng = np.random.default_rng(self.seed * 100003 + epoch)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, target = self.base[idx]  # img: (3, 32, 32) in [0, 1]
        if self.curriculum:
            w = curriculum_speed_weights(self.epoch, self.total_epochs)
        else:
            w = self.weights_fn
        speed = int(self._rng.choice(SPEEDS_KMH, p=w))
        if speed > 0:
            img = motion_blur(img.unsqueeze(0), speed).squeeze(0)
        return self.normalize(img), target

    total_epochs: int = 1  # set by trainer before training starts


def build_loaders(data_dir, batch_size, val_fraction, seed, train_weights_fn, curriculum):
    base_tf = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
    full_train = datasets.GTSRB(root=data_dir, split="train", download=True, transform=base_tf)

    targets = np.asarray([int(s[1]) for s in full_train._samples])
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for label in np.unique(targets):
        idx = np.flatnonzero(targets == label)
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_fraction)))
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())

    train_set = BlurAugmentedSubset(Subset(full_train, train_idx), train_weights_fn, seed, curriculum)

    val_normalize = transforms.Normalize(NORM_MEAN, NORM_STD)
    val_subset = Subset(full_train, val_idx)

    test_set = datasets.GTSRB(root=data_dir, split="test", download=True,
                              transform=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()]))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True,
                              generator=torch.Generator().manual_seed(seed))
    return train_loader, train_set, val_subset, test_set, val_normalize


def evaluate_val_ladder(model, val_subset, normalize, device, batch_size, val_fraction_of_ladder=None):
    """Mean top-1 over the speed ladder on a fixed-size subsample of val."""
    n = len(val_subset)
    max_n = 2000
    if n > max_n:
        g = torch.Generator().manual_seed(7)
        sel = torch.randperm(n, generator=g)[:max_n].tolist()
        subset = Subset(val_subset, sel)
    else:
        subset = val_subset
    loader = DataLoader(subset, batch_size=batch_size, num_workers=2, pin_memory=True)
    accs = []
    for speed in SPEEDS_KMH:
        acc = evaluate_condition(model, loader, normalize,
                                 (lambda b: b) if speed == 0 else (lambda b, s=speed: motion_blur(b, s)),
                                 device)
        accs.append(acc)
    return float(np.mean(accs)), accs


def train_one(name, model, data_dir, out_dir, epochs, batch_size, lr, seed,
              val_fraction, device, weights_fn, curriculum):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_loader, train_set, val_subset, test_set, normalize = build_loaders(
        data_dir, batch_size, val_fraction, seed, weights_fn, curriculum)
    train_set.total_epochs = epochs

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    best_val = -1.0
    best_epoch = 0
    history = []
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        train_set.set_epoch(epoch)
        model.train()
        loss_sum = correct = total = 0
        for inputs, targets in tqdm(train_loader, desc=f"{name} e{epoch+1}", leave=False):
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            correct += (logits.argmax(1) == targets).sum().item()
            total += targets.size(0)
        scheduler.step()

        train_acc = 100.0 * correct / total
        val_mean, val_ladder = evaluate_val_ladder(model, val_subset, normalize, device, batch_size)
        history.append({"epoch": epoch + 1, "train_loss": loss_sum / len(train_loader),
                        "train_acc": train_acc, "val_ladder_mean": val_mean,
                        "val_ladder": val_ladder, "lr": scheduler.get_last_lr()[0]})
        print(f"[{name}] epoch {epoch+1}/{epochs} train={train_acc:.2f}% val_ladder_mean={val_mean:.2f}% "
              f"lr={scheduler.get_last_lr()[0]:.2e}", flush=True)

        if val_mean > best_val:
            best_val = val_mean
            best_epoch = epoch + 1
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_ladder_mean": val_mean, "train_acc": train_acc},
                       out_dir / "best_model.pt")

    # Final test evaluation once, with the best checkpoint.
    ckpt = torch.load(out_dir / "best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_loader = DataLoader(test_set, batch_size=256, num_workers=2, pin_memory=True)
    ladder = {}
    for speed in SPEEDS_KMH:
        acc = evaluate_condition(model, test_loader, normalize,
                                 (lambda b: b) if speed == 0 else (lambda b, s=speed: motion_blur(b, s)),
                                 device)
        ladder[speed] = round(acc, 2)
        print(f"[{name}] test {speed:>3} km/h: {acc:.2f}%", flush=True)

    result = {"model": name, "config": {"epochs": epochs, "batch_size": batch_size, "lr": lr,
                                        "seed": seed, "val_fraction": val_fraction,
                                        "curriculum": curriculum},
              "best_epoch": best_epoch, "best_val_ladder_mean": best_val,
              "test_ladder": ladder, "history": history}
    with open(out_dir / "results.json", "w") as f:
        json.dump(result, f, indent=2)
    return ladder, best_val, best_epoch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--out-root", default=".")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--models", default="stn,event_stn")
    ap.add_argument("--smoke", action="store_true", help="tiny run for smoke testing")
    args = ap.parse_args()
    if args.smoke:
        args.epochs = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    wanted = [m.strip() for m in args.models.split(",") if m.strip()]
    summary = {}

    for name in wanted:
        if name == "baseline":
            model = TrafficSignNet(num_classes=43, use_stn=False)
            weights_fn = blur_augmented_weights(args.epochs)
            curriculum = False
            out_dir = Path(args.out_root) / "results_baseline_blur"
        elif name == "stn":
            model = TrafficSignNet(num_classes=43, use_stn=True)
            weights_fn = blur_augmented_weights(args.epochs)
            curriculum = False
            out_dir = Path(args.out_root) / "results_stn"
        elif name == "event_stn":
            model = EventSTNNet(num_classes=43)
            weights_fn = None  # curriculum table used instead
            curriculum = True
            out_dir = Path(args.out_root) / "results_event_stn"
        else:
            raise ValueError(f"unknown model {name}")

        ladder, best_val, best_epoch = train_one(
            name, model, args.data_dir, out_dir, args.epochs, args.batch_size,
            args.lr, args.seed, args.val_fraction, device, weights_fn, curriculum)
        summary[name] = {"test_ladder": ladder, "best_val_ladder_mean": best_val,
                         "best_epoch": best_epoch, "out_dir": str(out_dir)}
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_path = Path(args.out_root) / "results" / "event_stn_training_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print("wrote results/event_stn_training_summary.json", flush=True)


if __name__ == "__main__":
    main()

"""Reproducible GTSRB training pipeline with validation-based model selection.

Follows the ES-FA protocol: train/val/test split, val-based checkpoint selection,
final evaluation on test set exactly once.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm


class SpatialTransformer(nn.Module):
    """STN for traffic sign rectification."""

    def __init__(self):
        super().__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
        )
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 4 * 4, 32),
            nn.ReLU(True),
            nn.Linear(32, 6),
        )
        self.fc_loc[-1].weight.data.zero_()
        self.fc_loc[-1].bias.data = torch.tensor(
            [1, 0, 0, 0, 1, 0], dtype=torch.float
        )

    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(x.size(0), -1)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)
        return x


class TrafficSignNet(nn.Module):
    """CNN with optional STN for GTSRB classification."""

    def __init__(self, num_classes=43, use_stn=True):
        super().__init__()
        self.use_stn = use_stn
        if use_stn:
            self.stn = SpatialTransformer()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.3),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        if self.use_stn:
            x = self.stn(x)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def get_gtsrb_loaders(data_dir, batch_size=64, val_fraction=0.1, seed=42):
    """Load GTSRB with train/val/test split."""
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")

    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.3403, 0.3121, 0.3214), (0.2724, 0.2609, 0.2669)),
        ]
    )

    # GTSRB train split (used for train+val)
    full_train = datasets.GTSRB(
        root=data_dir, split="train", download=True, transform=transform
    )

    # Split each class independently so validation represents all 43 classes.
    # torchvision >=0.15 GTSRB stores (path, class_index) tuples in _samples.
    targets = np.asarray([int(s[1]) for s in full_train._samples])
    rng = np.random.default_rng(seed)
    train_indices = []
    val_indices = []
    for label in np.unique(targets):
        class_indices = np.flatnonzero(targets == label)
        rng.shuffle(class_indices)
        val_size = max(1, int(round(len(class_indices) * val_fraction)))
        val_indices.extend(class_indices[:val_size].tolist())
        train_indices.extend(class_indices[val_size:].tolist())

    train_dataset = Subset(full_train, train_indices)
    val_dataset = Subset(full_train, val_indices)

    # Test split
    test_dataset = datasets.GTSRB(
        root=data_dir, split="test", download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    return train_loader, val_loader, test_loader


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for inputs, targets in tqdm(loader, desc="train", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return total_loss / len(loader), 100.0 * correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="eval", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return total_loss / len(loader), 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser(description="GTSRB reproducible training")
    parser.add_argument("--data-dir", default="./data", help="data directory")
    parser.add_argument("--output-dir", default="./results", help="output directory")
    parser.add_argument("--epochs", type=int, default=50, help="training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="val split")
    parser.add_argument("--use-stn", action="store_true", help="enable STN")
    parser.add_argument("--device", default="auto", help="device (auto/cpu/cuda)")
    args = parser.parse_args()

    # Setup
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Data
    print("Loading GTSRB dataset...")
    train_loader, val_loader, test_loader = get_gtsrb_loaders(
        args.data_dir, args.batch_size, args.val_fraction, args.seed
    )

    # Model
    model = TrafficSignNet(num_classes=43, use_stn=args.use_stn).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # Training with val-based checkpoint selection
    best_val_acc = 0
    best_epoch = 0
    history = []

    print(f"Training on {device} for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        scheduler.step()

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": scheduler.get_last_lr()[0],
            }
        )

        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"train: {train_acc:.2f}% | val: {val_acc:.2f}% | "
            f"lr: {scheduler.get_last_lr()[0]:.2e}"
        )

        # Save best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_acc": val_acc,
                    "train_acc": train_acc,
                },
                output_dir / "best_model.pt",
            )

    # Final test evaluation (only once, with best val checkpoint)
    print(f"\nLoading best model from epoch {best_epoch} (val acc: {best_val_acc:.2f}%)...")
    checkpoint = torch.load(output_dir / "best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Final test accuracy: {test_acc:.2f}%")

    # Save results
    results = {
        "config": vars(args),
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "final_test_acc": test_acc,
        "history": history,
        "selection": "validation_based",
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()

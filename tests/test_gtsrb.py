"""Tests for GTSRB reproducible training pipeline."""

import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from gtsrb_reproducible import SpatialTransformer, TrafficSignNet


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stratified_split(labels: np.ndarray, val_fraction: float, seed: int):
    """Split indices by class - mirrors the logic in get_gtsrb_loaders."""
    rng = np.random.default_rng(seed)
    train_indices = []
    val_indices = []
    for label in np.unique(labels):
        class_indices = np.flatnonzero(labels == label)
        rng.shuffle(class_indices)
        val_size = max(1, int(round(len(class_indices) * val_fraction)))
        val_indices.extend(class_indices[:val_size].tolist())
        train_indices.extend(class_indices[val_size:].tolist())
    return train_indices, val_indices


class TestSpatialTransformer:
    """Tests for the STN module."""

    def test_output_shape_matches_input(self):
        """STN should preserve spatial dimensions."""
        stn = SpatialTransformer()
        x = torch.randn(2, 3, 32, 32)
        y = stn(x)
        assert y.shape == x.shape

    def test_identity_initialization(self):
        """STN should start with identity transform."""
        stn = SpatialTransformer()
        x = torch.randn(1, 3, 32, 32)
        y = stn(x)
        # With identity init, output should be close to input
        assert torch.allclose(x, y, atol=0.1)


class TestTrafficSignNet:
    """Tests for the main classifier."""

    def test_output_shape(self):
        """Output should have correct class count."""
        model = TrafficSignNet(num_classes=43, use_stn=True)
        x = torch.randn(4, 3, 32, 32)
        y = model(x)
        assert y.shape == (4, 43)

    def test_no_stn_mode(self):
        """Model should work without STN."""
        model = TrafficSignNet(num_classes=43, use_stn=False)
        x = torch.randn(4, 3, 32, 32)
        y = model(x)
        assert y.shape == (4, 43)

    def test_deterministic_forward(self):
        """Forward pass should be deterministic in eval mode."""
        set_seed(42)
        model = TrafficSignNet(num_classes=43, use_stn=True)
        model.eval()
        x = torch.randn(2, 3, 32, 32)
        y1 = model(x)
        y2 = model(x)
        assert torch.allclose(y1, y2)


class TestStratifiedSplit:
    """Tests for the stratified splitting function."""

    def test_split_preserves_total_count(self):
        """Split should not lose any samples."""
        labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
        train_idx, val_idx = stratified_split(labels, val_fraction=0.3, seed=42)
        assert len(train_idx) + len(val_idx) == len(labels)

    def test_split_is_deterministic(self):
        """Same seed should produce same split."""
        labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
        train1, val1 = stratified_split(labels, val_fraction=0.3, seed=42)
        train2, val2 = stratified_split(labels, val_fraction=0.3, seed=42)
        assert train1 == train2
        assert val1 == val2

    def test_split_has_no_overlap(self):
        """Train and val sets should be disjoint."""
        labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
        train_idx, val_idx = stratified_split(labels, val_fraction=0.3, seed=42)
        assert len(set(train_idx) & set(val_idx)) == 0

    def test_split_respects_fraction(self):
        """Val set should be approximately the requested fraction."""
        labels = np.array([0] * 100 + [1] * 100 + [2] * 100)
        train_idx, val_idx = stratified_split(labels, val_fraction=0.2, seed=42)
        actual_fraction = len(val_idx) / len(labels)
        assert 0.15 < actual_fraction < 0.25


class TestSetSeed:
    """Tests for seed setting."""

    def test_reproducible_random(self):
        """Random should be reproducible after set_seed."""
        set_seed(123)
        a = [np.random.rand() for _ in range(5)]
        set_seed(123)
        b = [np.random.rand() for _ in range(5)]
        assert a == b

    def test_reproducible_torch(self):
        """Torch random should be reproducible after set_seed."""
        set_seed(123)
        a = torch.rand(5).tolist()
        set_seed(123)
        b = torch.rand(5).tolist()
        assert a == b

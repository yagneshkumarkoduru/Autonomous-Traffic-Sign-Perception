"""
Neuromorphic Asynchronous Event-Stream & Frame-Based RGB Cross-Attention Fusion.
Voxelizes microsecond event-stream polarities into spatial-temporal spike tensors
and fuses them with RGB feature maps via multi-head cross-attention.
"""

import numpy as np


class NeuromorphicEventVoxelizer:
    """
    Voxelizes asynchronous events e_k = (x_k, y_k, t_k, p_k) into B bins:
    V(x, y, t) = sum_k p_k * max(0, 1 - |t - t_k^*|)
    """

    def __init__(self, width: int = 32, height: int = 32, num_temporal_bins: int = 5):
        self.width = width
        self.height = height
        self.num_bins = num_temporal_bins

    def voxelize(self, events: np.ndarray) -> np.ndarray:
        """
        Input events: (N, 4) -> [x, y, timestamp_us, polarity]
        Output voxel grid: (num_bins, height, width)
        """
        voxel_grid = np.zeros((self.num_bins, self.height, self.width), dtype=np.float32)
        if len(events) == 0:
            return voxel_grid

        t_min = events[:, 2].min()
        t_max = events[:, 2].max()
        t_norm = (events[:, 2] - t_min) / max((t_max - t_min), 1.0) * (self.num_bins - 1)

        for i in range(len(events)):
            x = int(np.clip(events[i, 0], 0, self.width - 1))
            y = int(np.clip(events[i, 1], 0, self.height - 1))
            t_idx = t_norm[i]
            p = events[i, 3]

            bin_low = int(np.floor(t_idx))
            bin_high = min(bin_low + 1, self.num_bins - 1)
            weight_high = t_idx - bin_low
            weight_low = 1.0 - weight_high

            voxel_grid[bin_low, y, x] += p * weight_low
            voxel_grid[bin_high, y, x] += p * weight_high

        return voxel_grid


class EventRGBCrossAttention:
    """Multi-head cross-attention mechanism between Event spikes and RGB features."""

    def __init__(self, embed_dim: int = 64, num_heads: int = 4):
        self.embed_dim = embed_dim
        self.num_heads = num_heads

    def fuse(self, rgb_features: np.ndarray, event_features: np.ndarray) -> np.ndarray:
        """
        Query: RGB features, Key & Value: Neuromorphic Event Spikes
        Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
        """
        d_k = self.embed_dim // self.num_heads
        scores = np.dot(rgb_features, event_features.T) / np.sqrt(d_k)
        # Stable softmax
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        fused = np.dot(attn_weights, event_features)
        return fused + rgb_features # Residual connection

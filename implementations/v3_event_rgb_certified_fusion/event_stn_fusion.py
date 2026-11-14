"""Trainable Event-STN fusion model: STN rectification + RGB CNN + event CNN + cross-attention.

The event channel is synthesized deterministically from the frame pair formed by
the exposure-smear corruption: a moving camera integrates light over the exposure
window, so the smeared frame and its shifted counterpart produce a temporal
gradient. A neuromorphic event camera fires on log-intensity change with
microsecond latency; under motion blur the inter-sample gradient survives even
where absolute texture is destroyed. The voxelizer below converts that gradient
into a (bins, H, W) spike tensor using the same linear-interpolation scheme as
implementations/v3_event_rgb_certified_fusion/event_stream_cross_attention.py.

The fused network:
  1) STN rectifies the RGB input (differentiable affine normalization).
  2) An RGB CNN extracts frame features.
  3) A small CNN extracts event-voxel features.
  4) Multi-head cross-attention (RGB queries, event keys/values) fuses them.
  5) A residual connection plus classifier produces logits.

Every module is torch.nn so the whole pipeline trains end-to-end.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from implementations.v2_spatial_transformer_network.differentiable_stn_layer import (
    DifferentiableSTN,
)


def synthetic_event_voxels(x: torch.Tensor, num_bins: int = 5) -> torch.Tensor:
    """Deterministic event-voxel proxy from a frame batch.

    x: (B, 3, H, W) frames in [0, 1]. Returns (B, num_bins, H, W) spike tensor.

    A horizontal exposure smear mixes the current sample with its shifted
    neighbors; the finite difference along the motion axis approximates the
    brightness-constancy residual an event camera would report. Polarity is
    encoded by sign; magnitude is the temporal contrast change. Events are
    distributed across temporal bins by the linear-interpolation rule used in
    the voxelizer of event_stream_cross_attention.py, treating the smear
    window as the inter-frame interval.
    """
    gray = x.mean(dim=1, keepdim=True)  # (B, 1, H, W)
    # Temporal gradient along the motion (width) axis: central difference.
    dx = torch.zeros_like(gray)
    dx[..., 1:-1] = 0.5 * (gray[..., 2:] - gray[..., :-2])
    # Normalize per-sample to [-1, 1] so polarity magnitude is comparable.
    absmax = dx.abs().amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
    events = dx / absmax  # (B, 1, H, W) in [-1, 1]

    b, _, h, w = events.shape
    vox = torch.zeros(b, num_bins, h, w, device=x.device, dtype=events.dtype)
    # Magnitude-quantized binning: stronger changes land in later bins,
    # emulating the temporal ordering of a real event stream within the
    # exposure window. Polarity is preserved via the signed value.
    mag = events.abs().squeeze(1)  # (B, H, W) in [0, 1]
    pos = mag * (num_bins - 1)
    low = pos.floor().long().clamp(0, num_bins - 1)
    high = (low + 1).clamp(max=num_bins - 1)
    w_high = (pos - low.float()).clamp(0.0, 1.0)
    w_low = 1.0 - w_high
    signed = events.squeeze(1)
    # Vectorized scatter into temporal bins: flatten (B, H, W) into batch*HW
    # and scatter along the bin axis with index = low/high.
    flat_signed_low = (signed * w_low).reshape(b, -1)  # (B, HW)
    flat_signed_high = (signed * w_high).reshape(b, -1)
    idx_low = low.reshape(b, -1)  # (B, HW) bin ids in [0, bins-1]
    idx_high = high.reshape(b, -1)
    # one-hot accumulate: (B, bins, HW) -> reshape to (B, bins, H, W)
    acc = torch.zeros(b, num_bins, h * w, device=x.device, dtype=events.dtype)
    acc.scatter_add_(1, idx_low.unsqueeze(1), flat_signed_low.unsqueeze(1))
    acc.scatter_add_(1, idx_high.unsqueeze(1), flat_signed_high.unsqueeze(1))
    return acc.reshape(b, num_bins, h, w)


class EventEncoder(nn.Module):
    """Small CNN encoder for the event voxel grid."""

    def __init__(self, num_bins: int = 5, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(num_bins, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16x16
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 8x8
            nn.Conv2d(32, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 4x4
        )

    def forward(self, vox: torch.Tensor) -> torch.Tensor:
        return self.net(vox)  # (B, out_dim, 4, 4)


class RGBEncoder(nn.Module):
    """Frame CNN mirroring TrafficSignNet features but emitting a feature map."""

    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 4x4
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (B, out_dim, 4, 4)


class CrossAttentionFusion(nn.Module):
    """Multi-head cross-attention: RGB queries attend to event keys/values."""

    def __init__(self, embed_dim: int = 128, num_heads: int = 4):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, rgb_feat: torch.Tensor, ev_feat: torch.Tensor) -> torch.Tensor:
        b, c, h, w = rgb_feat.shape
        q = rgb_feat.flatten(2).transpose(1, 2)  # (B, HW, C)
        kv = ev_feat.flatten(2).transpose(1, 2)  # (B, HW, C)
        fused, _ = self.attn(q, kv, kv)  # (B, HW, C)
        out = self.norm(q + fused)  # residual
        return out.transpose(1, 2).reshape(b, c, h, w)


class EventSTNNet(nn.Module):
    """STN + RGB CNN + event CNN + cross-attention fusion classifier."""

    def __init__(self, num_classes: int = 43, num_bins: int = 5, embed_dim: int = 128, num_heads: int = 4):
        super().__init__()
        self.num_bins = num_bins
        self.stn = DifferentiableSTN(in_channels=3)
        self.rgb_encoder = RGBEncoder(out_dim=embed_dim)
        self.event_encoder = EventEncoder(num_bins=num_bins, out_dim=embed_dim)
        self.fusion = CrossAttentionFusion(embed_dim=embed_dim, num_heads=num_heads)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(embed_dim * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is expected in normalized space; events are computed on the same
        # tensor the RGB path sees (the corruption benchmark corrupts in [0,1]
        # then normalizes, so at eval time x arrives post-normalization; the
        # voxelizer is scale-tolerant because it normalizes per sample).
        x_rect, _theta = self.stn(x)
        vox = synthetic_event_voxels(x_rect, self.num_bins)
        rgb_feat = self.rgb_encoder(x_rect)
        ev_feat = self.event_encoder(vox)
        fused = self.fusion(rgb_feat, ev_feat)
        return self.classifier(fused)

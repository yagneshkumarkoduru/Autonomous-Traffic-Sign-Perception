"""
Differentiable Spatial Transformer Network (STN) for Projective Affine Rectification.
Implements localization network, 2D affine grid generator, and bilinear sampler:
V_c(x, y) = sum_{n} sum_{m} U_c(m, n) * max(0, 1 - |x_s - m|) * max(0, 1 - |y_s - n|)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentiableSTN(nn.Module):
    """Spatial Transformer Network module for projective affine normalization."""

    def __init__(self, in_channels: int = 3):
        super().__init__()
        # Localization network
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=5),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=5),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
        )

        # Regressor for the 2x3 affine matrix theta
        self.fc_loc = nn.Sequential(
            nn.Linear(32 * 5 * 5, 64),
            nn.ReLU(True),
            nn.Linear(64, 3 * 2)
        )

        # Initialize weights for identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xs = self.localization(x)
        xs = xs.view(-1, 32 * 5 * 5)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        # Differentiable grid sampling
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x_rectified = F.grid_sample(x, grid, align_corners=False)
        return x_rectified, theta

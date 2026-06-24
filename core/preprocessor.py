"""The learnable preprocessor: packed RAW -> pseudo-RGB for frozen detectors.

Deliberately small. The variable under study in this project is the
*supervision regime* (which detectors train it), NOT the processor's
capacity. Keep this fixed across all regimes so any mAP difference is
attributable to supervision, not architecture.
"""
import torch
import torch.nn as nn


class RawPreprocessor(nn.Module):
    """
    Input:  [B, 4, H, W]  packed RGGB, values ~[0, 1]
    Output: [B, 3, H, W]  pseudo-RGB in [0, 1]

    Two parts:
      1. A per-input-channel global gamma (SimROD's insight, ~free):
         RAW pixel values sit compressed in the low range; a learnable
         gamma lifts them so the frozen detector's first conv sees
         natural-image-like statistics.
      2. A 3-conv local body that does content-dependent correction and
         projects 4 -> 3 channels.
    """

    def __init__(self, in_ch: int = 4, width: int = 16):
        super().__init__()
        # exp(log_gamma) keeps gamma strictly positive without clamping
        self.log_gamma = nn.Parameter(torch.zeros(in_ch))
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(width, 3, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(1e-4, 1.0)
        gamma = torch.exp(self.log_gamma).view(1, -1, 1, 1)
        x = x.pow(gamma)            # global tone lift
        y = self.body(x)            # local correction + channel projection
        return torch.sigmoid(y)     # squash back to [0, 1]

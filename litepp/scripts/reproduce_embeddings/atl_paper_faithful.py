#!/usr/bin/env python3
"""
A minimal Adaptive Threshold Learning (ATL) module that implements the
main paper's Eq. 5 literally:

    tau_raw = sigmoid(MLP(GAP(F^(14)))) * (tau_max - tau_min) + tau_min

This is a deliberate alternative to litepp.models.adaptive_threshold's
`SceneEncoder`, whose `forward()` always concatenates an additional
"spatial_encoder" branch (AdaptiveAvgPool2d(4) + a large Linear(C*16, ...)
layer) alongside the global-pooled branch -- a real architectural
component with on the order of 10^6 parameters that the paper's Eq. 5 does
not describe. Both are legitimate design choices (the richer encoder may
well perform better in practice), but only one of them matches what is
written in the paper, so we provide this version to make the paper's
specific 11.7K-parameter claim and the simple Eq. 5 formula independently
checkable against real code and real numbers. See
litepp/scripts/reproduce_embeddings/train_atl.py for how it is trained on
real, grid-search-derived oracle thresholds rather than any simulated
target, and supplementary Sec. I.5 of the paper for the reconciliation
between this and the richer `SceneEncoder` variant.
"""
import torch
import torch.nn as nn


class ATLPaperFaithful(nn.Module):
    """tau_raw = sigmoid(MLP(GAP(F^(14)))) * (tau_max - tau_min) + tau_min"""

    def __init__(self, input_channels: int = 576, hidden_dim: int = 64,
                 tau_min: float = 0.01, tau_max: float = 0.50):
        super().__init__()
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(input_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        """feature_map: (B, C, H, W) raw Layer-14 backbone feature map."""
        pooled = self.global_pool(feature_map).flatten(1)
        logit = self.mlp(pooled).squeeze(-1)
        tau_raw = torch.sigmoid(logit) * (self.tau_max - self.tau_min) + self.tau_min
        return tau_raw

    def forward_from_gap(self, gap_vec: torch.Tensor) -> torch.Tensor:
        """Same as forward(), but starting from an already-GAP'd (B, C) vector
        (used when F^(14) is cached per-frame rather than re-pooled each call)."""
        logit = self.mlp(gap_vec).squeeze(-1)
        return torch.sigmoid(logit) * (self.tau_max - self.tau_min) + self.tau_min


if __name__ == "__main__":
    m = ATLPaperFaithful(input_channels=576, hidden_dim=64)
    n_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"ATLPaperFaithful (C=576, hidden=64): {n_params:,} trainable parameters")
    for hidden in (32, 48, 64, 96, 128):
        m = ATLPaperFaithful(input_channels=576, hidden_dim=hidden)
        n = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"  hidden_dim={hidden}: {n:,} params")

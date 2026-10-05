"""
Adaptive Threshold Learning Module for LITE++

This module learns scene-aware confidence thresholds that automatically adapt
to different environments, eliminating the need for manual threshold tuning
across datasets.

Key Insight: Optimal detection threshold varies significantly by scene:
- MOT17 (medium density): ~0.25
- MOT20 (high density): ~0.05
- Traffic scenes: ~0.30
- Retail scenes: ~0.15

The Adaptive Threshold Learning (ATL) module predicts per-scene thresholds
based on global scene features extracted from the detection backbone.

Note on SceneEncoder vs. the paper's Eq. 5: `SceneEncoder.forward()` below
concatenates a global-pooled branch with an additional `spatial_encoder`
branch (~1.18M parameters at the real YOLOv8m channel widths) that is not
described in the paper, which states the simpler
tau_raw = sigmoid(MLP(GAP(F^(14)))) * (tau_max - tau_min) + tau_min.
`SceneEncoder` is kept as a richer, optional architecture (it may perform
better in practice), but the 11.7K/37K ATL-encoder parameter counts and all
real experiments reported in the paper and its supplementary use the
literal, Eq.-5-matching implementation in
litepp/scripts/reproduce_embeddings/atl_paper_faithful.py
(class ATLPaperFaithful) instead of this class.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, List, Literal


class SceneEncoder(nn.Module):
    """
    Encodes scene-level features for threshold prediction.

    Extracts global context from backbone features to characterize:
    - Scene density (crowded vs sparse)
    - Lighting conditions
    - Domain characteristics

    Args:
        input_channels: Number of channels from backbone feature map
        hidden_dim: Hidden layer dimension
        output_dim: Output encoding dimension
    """

    def __init__(
        self,
        input_channels: int = 192,
        hidden_dim: int = 128,
        output_dim: int = 64,
    ):
        super().__init__()

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Global feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(inplace=True),
        )

        # Spatial statistics encoder (captures layout information)
        self.spatial_encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(input_channels * 16, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Encode scene from backbone features.

        Args:
            features: (B, C, H, W) backbone feature map

        Returns:
            scene_encoding: (B, 2*output_dim) scene representation
        """
        global_feat = self.global_pool(features).flatten(1)
        global_encoding = self.encoder(global_feat)

        spatial_encoding = self.spatial_encoder(features)

        return torch.cat([global_encoding, spatial_encoding], dim=1)


class AdaptiveThresholdModule(nn.Module):
    """
    Learns to predict optimal confidence thresholds based on scene characteristics.

    Features:
    - Scene-aware threshold prediction
    - Bounded output in valid range [min_threshold, max_threshold]
    - Differentiable for end-to-end training
    - Supports multiple threshold types

    Args:
        input_channels: Number of channels in backbone feature map
        hidden_dim: Hidden dimension for scene encoder
        min_threshold: Minimum allowed threshold
        max_threshold: Maximum allowed threshold
        default_threshold: Default threshold for initialization
        num_thresholds: Number of thresholds to predict
        threshold_names: Names for each threshold
    """

    def __init__(
        self,
        input_channels: int = 192,
        hidden_dim: int = 128,
        min_threshold: float = 0.01,
        max_threshold: float = 0.50,
        default_threshold: float = 0.25,
        num_thresholds: int = 1,
        threshold_names: Optional[List[str]] = None,
    ):
        super().__init__()

        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.default_threshold = default_threshold
        self.num_thresholds = num_thresholds
        self.threshold_names = threshold_names or [
            f"threshold_{i}" for i in range(num_thresholds)
        ]

        # Scene encoder
        self.scene_encoder = SceneEncoder(
            input_channels=input_channels,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim // 2,
        )

        # Threshold predictor
        self.threshold_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_thresholds),
        )

        self._init_to_default()

    def _init_to_default(self):
        """Initialize to predict default threshold."""
        target_sigmoid = (self.default_threshold - self.min_threshold) / (
            self.max_threshold - self.min_threshold
        )
        target_logit = np.log(target_sigmoid / (1 - target_sigmoid + 1e-8))

        with torch.no_grad():
            self.threshold_predictor[-1].bias.fill_(target_logit)
            self.threshold_predictor[-1].weight.fill_(0.0)

    def forward(
        self,
        features: torch.Tensor,
        return_encoding: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Predict scene-adaptive thresholds.

        Args:
            features: (B, C, H, W) backbone feature map
            return_encoding: Whether to return scene encoding

        Returns:
            thresholds: (B, num_thresholds) predicted thresholds
            scene_encoding: Optional scene encoding
        """
        scene_encoding = self.scene_encoder(features)
        logits = self.threshold_predictor(scene_encoding)

        # Scale to valid range
        thresholds = torch.sigmoid(logits)
        thresholds = (
            self.min_threshold + (self.max_threshold - self.min_threshold) * thresholds
        )

        if return_encoding:
            return thresholds, scene_encoding
        return thresholds, None

    def get_thresholds_dict(self, features: torch.Tensor) -> Dict[str, float]:
        """Get thresholds as a named dictionary."""
        thresholds, _ = self.forward(features)
        thresholds = thresholds.squeeze(0).cpu().numpy()

        return {
            name: float(thresh)
            for name, thresh in zip(self.threshold_names, thresholds)
        }
    
    @staticmethod
    def compute_two_stage_thresholds(
        tau_raw: torch.Tensor,
        min_low: float = 0.01,
        base_low: float = 0.05,
        anchor: float = 0.50,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Derives ByteTrack two-stage thresholds guaranteeing tau_h > tau_l >= min_low.

        tau_h = tau_raw  (directly the predicted scene confidence)
        tau_l = max(min_low, base_low + (tau_raw - anchor) / 2)

        With tau_raw in [0.05, 0.60], tau_h - tau_l >= 0.5*tau_raw + 0.20 > 0.

        Paper eq. (7): tau_h = tau_raw, tau_l = max(0.01, 0.05 + (tau_raw - 0.50) / 2).
        """
        tau_h = tau_raw
        tau_l = torch.clamp(base_low + (tau_raw - anchor) / 2.0, min=min_low)
        return tau_h, tau_l


class MultiThresholdPredictor(nn.Module):
    """
    Predicts multiple tracking-related thresholds:
    - Detection confidence threshold
    - Association (matching) threshold
    - Track max age (frames before deletion)

    Each threshold adapts to scene characteristics.

    Args:
        input_channels: Number of channels from backbone
        hidden_dim: Hidden layer dimension
    """

    def __init__(
        self,
        input_channels: int = 192,
        hidden_dim: int = 128,
    ):
        super().__init__()

        # Shared scene encoder
        self.scene_encoder = SceneEncoder(
            input_channels=input_channels,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim // 2,
        )

        # Detection threshold (0.01 - 0.50)
        self.det_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Association threshold (0.1 - 0.5)
        self.assoc_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Max age (15 - 90 frames)
        self.age_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Density estimator (auxiliary)
        self.density_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(input_channels, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Softplus(),
        )

        self._initialize_defaults()

    def _initialize_defaults(self):
        """Initialize heads to predict reasonable defaults."""
        # Detection: default 0.25 in range [0.01, 0.50]
        det_logit = np.log(0.49)  # sigmoid^-1((0.25-0.01)/(0.50-0.01)) ≈ 0
        with torch.no_grad():
            self.det_head[-1].bias.fill_(det_logit)
            self.det_head[-1].weight.fill_(0.0)

        # Association: default 0.3 in range [0.1, 0.5]
        assoc_logit = np.log(1.0)  # sigmoid^-1((0.3-0.1)/(0.5-0.1))
        with torch.no_grad():
            self.assoc_head[-1].bias.fill_(assoc_logit)
            self.assoc_head[-1].weight.fill_(0.0)

        # Age: default 30 in range [15, 90]
        age_logit = np.log(0.25)  # sigmoid^-1((30-15)/(90-15))
        with torch.no_grad():
            self.age_head[-1].bias.fill_(age_logit)
            self.age_head[-1].weight.fill_(0.0)

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict all tracking thresholds.

        Args:
            features: (B, C, H, W) backbone features

        Returns:
            Dictionary with all threshold predictions
        """
        scene_encoding = self.scene_encoder(features)

        # Detection threshold [0.01, 0.50]
        det_logit = self.det_head(scene_encoding)
        det_thresh = 0.01 + 0.49 * torch.sigmoid(det_logit)

        # Association threshold [0.1, 0.5]
        assoc_logit = self.assoc_head(scene_encoding)
        assoc_thresh = 0.1 + 0.4 * torch.sigmoid(assoc_logit)

        # Max age [15, 90]
        age_logit = self.age_head(scene_encoding)
        max_age = 15 + 75 * torch.sigmoid(age_logit)

        # Density estimation
        density = self.density_head(features)

        return {
            "detection_threshold": det_thresh.squeeze(-1),
            "association_threshold": assoc_thresh.squeeze(-1),
            "max_age": max_age.squeeze(-1),
            "density": density.squeeze(-1),
        }
    @staticmethod
    def compute_two_stage_thresholds(
        tau_raw: torch.Tensor,
        min_low: float = 0.01,
        base_low: float = 0.05,
        anchor: float = 0.50,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """See AdaptiveThresholdModule.compute_two_stage_thresholds."""
        tau_h = tau_raw
        tau_l = torch.clamp(base_low + (tau_raw - anchor) / 2.0, min=min_low)
        return tau_h, tau_l


class TemporalThresholdSmoother:
    """EMA smoother for frame-to-frame threshold stability.

    Ablation (paper Table 6): beta=0.9 → HOTA 63.0; per-frame (no EMA) → 62.4 (−0.6);
    beta=0.95 → 62.8 (−0.2). beta=0.9 chosen as best balance of responsiveness and stability.
    Cosine-distance reset (threshold=0.35) re-initialises EMA on abrupt scene transitions.
    """

    def __init__(self, beta: float = 0.9, reset_threshold: float = 0.35):
        self.beta = beta
        self.reset_threshold = reset_threshold
        self.prev_tau: Optional[float] = None
        self.prev_global_feat: Optional[torch.Tensor] = None

    def update(self, raw_tau: float, global_feat: torch.Tensor) -> float:
        feat_norm = F.normalize(global_feat.flatten(), p=2, dim=0)

        if self.prev_tau is None or self.prev_global_feat is None:
            smooth_tau = raw_tau
        else:
            # Compute cosine distance between consecutive scene representations
            cos_dist = 1.0 - torch.dot(feat_norm, self.prev_global_feat).item()
            if cos_dist > self.reset_threshold:
                # Abrupt scene transition detected: reset EMA memory
                smooth_tau = raw_tau
            else:
                smooth_tau = self.beta * self.prev_tau + (1.0 - self.beta) * raw_tau

        self.prev_tau = smooth_tau
        self.prev_global_feat = feat_norm
        return smooth_tau

    def reset(self):
        self.prev_tau = None
        self.prev_global_feat = None

class AdaptiveThresholdLoss(nn.Module):
    """
    Loss functions for training adaptive threshold modules.

    Supports:
    - supervised: MSE loss with ground truth optimal thresholds
    - reinforce: Policy gradient from tracking performance
    """

    def __init__(
        self,
        loss_type: Literal["supervised", "reinforce"] = "supervised",
    ):
        super().__init__()
        self.loss_type = loss_type

    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        baseline: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute loss."""
        if self.loss_type == "supervised":
            return F.mse_loss(predicted, target)

        elif self.loss_type == "reinforce":
            # target is reward (e.g., HOTA score)
            if baseline is None:
                baseline = target.mean()
            advantage = target - baseline
            log_prob = -0.5 * (predicted**2).sum(dim=-1)
            return -(advantage.detach() * log_prob).mean()

        raise ValueError(f"Unknown loss type: {self.loss_type}")


def create_adaptive_threshold_module(
    input_channels: int = 192,
    variant: Literal["single", "multi"] = "single",
    pretrained_path: Optional[str] = None,
    device: str = "cuda:0",
) -> nn.Module:
    """
    Factory function to create adaptive threshold module.

    Args:
        input_channels: Channels from backbone feature layer
        variant: 'single' for detection only, 'multi' for all thresholds
        pretrained_path: Path to pretrained weights
        device: Target device

    Returns:
        Configured threshold module
    """
    if variant == "single":
        # Range [0.05, 0.60] matches paper eq. (5): tau_raw in [tau_min, tau_max] = [0.05, 0.60].
        module = AdaptiveThresholdModule(
            input_channels=input_channels,
            hidden_dim=128,
            min_threshold=0.05,
            max_threshold=0.60,
            default_threshold=0.25,
            num_thresholds=1,
            threshold_names=["detection"],
        )
    elif variant == "multi":
        module = MultiThresholdPredictor(
            input_channels=input_channels,
            hidden_dim=128,
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")

    if pretrained_path is not None:
        state_dict = torch.load(pretrained_path, map_location=device)
        module.load_state_dict(state_dict)

    return module.to(device)


def compute_two_stage_thresholds(
    tau_raw: torch.Tensor,
    min_low: float = 0.01,
    base_low: float = 0.05,
    anchor: float = 0.50,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Module-level convenience wrapper around
    ``AdaptiveThresholdModule.compute_two_stage_thresholds`` (paper Eq. 7),
    so callers can do ``from litepp.models import compute_two_stage_thresholds``
    without needing an instantiated module. See that method for the derivation.
    """
    return AdaptiveThresholdModule.compute_two_stage_thresholds(
        tau_raw, min_low=min_low, base_low=base_low, anchor=anchor
    )

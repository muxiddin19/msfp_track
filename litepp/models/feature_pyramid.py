"""
Multi-Scale Feature Pyramid Module for LITE++

This module implements the Multi-Scale Feature Pyramid (MSFP) component that
extracts appearance features from multiple backbone layers and fuses them
into discriminative representations for multi-object tracking.

Key Contributions:
1. Multi-scale feature extraction from early/mid/late backbone layers
2. Three fusion strategies: concatenation, attention-weighted, channel-adaptive
3. Efficient spatial pooling with minimal computational overhead
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Literal, Dict, Optional, Tuple

class InstanceAdaptiveAttentionFusion(nn.Module):
  
    def __init__(
        self,
        layer_channels: List[int],
        output_dim: int = 128,
        temperature: float = 0.5,
        dropout_p: float = 0.1
    ):
        super().__init__()
        self.output_dim = output_dim
        self.temperature = temperature
        self.num_layers = len(layer_channels)

        # Linear projection layers Proj^(l) to common dimension d=128
        self.layer_projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(ch, output_dim),
                nn.LayerNorm(output_dim)
            ) for ch in layer_channels
        ])

        # Attention projection W_alpha mapping (num_layers * output_dim) -> num_layers
        concatenated_dim = self.num_layers * output_dim  # e.g., 3 * 128 = 384
        self.w_alpha = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(concatenated_dim, self.num_layers)
        )
        self.final_norm = nn.LayerNorm(output_dim)

    def forward(
        self, layer_features: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            layer_features: List of (N, C_l) tensors from RoIAlign across scales.
        Returns:
            f_i: (N, output_dim) fused instance embeddings.
            alpha_i: (N, num_layers) per-detection layer weights.
        """
        # 1. Project each layer to common dimension: List of (N, output_dim)
        projected = [proj(feat) for proj, feat in zip(self.layer_projectors, layer_features)]

        # 2. Concatenate cross-level features: (N, num_layers * output_dim)
        concat_repr = torch.cat(projected, dim=-1)

        # 3. Compute per-instance attention logits and normalize by temperature T: (N, num_layers)
        logits = self.w_alpha(concat_repr) / self.temperature
        alpha_i = F.softmax(logits, dim=-1)

        # 4. Weighted combination: f_i = sum_l alpha_i^(l) * Proj^(l)(f_i^(l))
        stacked = torch.stack(projected, dim=1)  # (N, num_layers, output_dim)
        weights = alpha_i.unsqueeze(-1)          # (N, num_layers, 1)
        fused = torch.sum(stacked * weights, dim=1)  # (N, output_dim)

        return self.final_norm(fused), alpha_i

class FeatureFusionModule(nn.Module):
    """
    Fuses features from multiple backbone layers into a unified representation.

    Supports four fusion strategies:
    - concat: Concatenate features and project via MLP
    - attention: Instance-adaptive attention weights per detection (default, Eq. 3-4)
    - global: Static learned attention weights per layer (Table 5 ablation)
    - adaptive: Channel-wise attention using Squeeze-and-Excitation

    Args:
        layer_channels: List of channel counts for each input layer
        output_dim: Dimension of the fused output features
        fusion_type: Fusion strategy to use
    """

    def __init__(
        self,
        layer_channels: List[int],
        output_dim: int = 128,
        fusion_type: Literal["concat", "attention", "global", "adaptive"] = "attention",
        dropout_p: float = 0.1,
    ):
        super().__init__()
        self.layer_channels = layer_channels
        self.output_dim = output_dim
        self.fusion_type = fusion_type
        self.total_channels = sum(layer_channels)
        self.last_attention_weights: Optional[torch.Tensor] = None

        if fusion_type == "concat":
            self.projector = nn.Sequential(
                nn.Linear(self.total_channels, output_dim * 2),
                nn.ReLU(inplace=True),
                nn.Linear(output_dim * 2, output_dim),
                nn.LayerNorm(output_dim),
            )

        elif fusion_type == "attention":
            # Instance-adaptive attention (per-detection W_alpha in R^{3 x 384})
            self.instance_attention = InstanceAdaptiveAttentionFusion(
                layer_channels=layer_channels,
                output_dim=output_dim,
                temperature=0.5,
                dropout_p=dropout_p,
            )

        elif fusion_type == "global":
            # Static global layer weights (for Table 5 ablation)
            self.layer_weights = nn.Parameter(
                torch.ones(len(layer_channels)) / len(layer_channels)
            )
            self.layer_projectors = nn.ModuleList(
                [nn.Linear(ch, output_dim) for ch in layer_channels]
            )
            self.final_norm = nn.LayerNorm(output_dim)

        elif fusion_type == "adaptive":
            # Squeeze-and-Excitation style channel attention
            self.channel_attention = nn.Sequential(
                nn.Linear(self.total_channels, self.total_channels // 4),
                nn.ReLU(inplace=True),
                nn.Linear(self.total_channels // 4, self.total_channels),
                nn.Sigmoid(),
            )
            self.projector = nn.Sequential(
                nn.Linear(self.total_channels, output_dim),
                nn.LayerNorm(output_dim),
            )

    def forward(self, layer_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Fuse features from multiple layers.

        Args:
            layer_features: List of tensors, each (N, C_i) where C_i is layer channels

        Returns:
            Fused features of shape (N, output_dim)
        """
        if self.fusion_type == "concat":
            concatenated = torch.cat(layer_features, dim=-1)
            return self.projector(concatenated)

        elif self.fusion_type == "attention":
            fused, alpha = self.instance_attention(layer_features)
            self.last_attention_weights = alpha
            return fused

        elif self.fusion_type == "global":
            weights = F.softmax(self.layer_weights, dim=0)
            self.last_attention_weights = weights
            projected = [
                proj(feat) for proj, feat in zip(self.layer_projectors, layer_features)
            ]
            weighted_sum = sum(w * feat for w, feat in zip(weights, projected))
            return self.final_norm(weighted_sum)

        elif self.fusion_type == "adaptive":
            concatenated = torch.cat(layer_features, dim=-1)
            attention = self.channel_attention(concatenated)
            attended = concatenated * attention
            return self.projector(attended)

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Return the learned attention weights (for attention/global fusion)."""
        if self.last_attention_weights is not None:
            return self.last_attention_weights.detach()
        return None


class MultiScaleFeaturePyramid(nn.Module):
    """
    Multi-Scale Feature Pyramid for appearance feature extraction.

    Extracts features from multiple backbone layers of a YOLO detector
    and fuses them for improved object re-identification.

    This approach captures both fine-grained details (early layers)
    and semantic information (late layers).

    Args:
        layer_configs: Dict mapping layer names to channel counts
        fusion_type: Feature fusion strategy
        output_dim: Final embedding dimension
        spatial_pool: Spatial pooling method ("mean", "max", "adaptive")
    """

    # Exact YOLOv8 layer specs from paper Table 1 (ACCV 2026).
    # Layers: Shallow=P3 C2f (model.4), Medium=P4 C2f (model.6 for n/s/m, model.9 for l/x),
    # Deep=P5/SPPF (model.9 for n/s/m, model.14 for l/x). Strides: 8/16/32.
    YOLO_LAYER_SPECS = {
        "yolov8n": {"layer4": {"stride": 8,  "channels": 64},  "layer6": {"stride": 16, "channels": 128}, "layer9":  {"stride": 32, "channels": 256}},
        "yolov8s": {"layer4": {"stride": 8,  "channels": 128}, "layer6": {"stride": 16, "channels": 256}, "layer9":  {"stride": 32, "channels": 512}},
        "yolov8m": {"layer4": {"stride": 8,  "channels": 192}, "layer6": {"stride": 16, "channels": 384}, "layer9":  {"stride": 32, "channels": 576}},
        "yolov8l": {"layer4": {"stride": 8,  "channels": 256}, "layer9": {"stride": 16, "channels": 512}, "layer14": {"stride": 32, "channels": 512}},
        "yolov8x": {"layer4": {"stride": 8,  "channels": 320}, "layer9": {"stride": 16, "channels": 640}, "layer14": {"stride": 32, "channels": 640}},
    }
    # Channel counts keyed by layer name, derived from YOLO_LAYER_SPECS.
    YOLO_LAYER_CHANNELS = {
        variant: {name: spec["channels"] for name, spec in layers.items()}
        for variant, layers in YOLO_LAYER_SPECS.items()
    }
    # Default layer indices per variant (Shallow/Medium/Deep at strides 8/16/32).
    YOLO_LAYER_INDICES = {
        "yolov8n": ["layer4", "layer6", "layer9"],
        "yolov8s": ["layer4", "layer6", "layer9"],
        "yolov8m": ["layer4", "layer6", "layer9"],
        "yolov8l": ["layer4", "layer9", "layer14"],
        "yolov8x": ["layer4", "layer9", "layer14"],
    }
    def __init__(
        self,
        layer_configs: Dict[str, int] = None,
        fusion_type: Literal["concat", "attention", "adaptive"] = "attention",
        output_dim: int = 128,
        spatial_pool: Literal["mean", "max", "adaptive"] = "mean",
        yolo_variant: str = "yolov8m",
    ):
        super().__init__()

        # Use provided config or default based on YOLO variant
        if layer_configs is None:
            default_channels = self.YOLO_LAYER_CHANNELS.get(
                yolo_variant, self.YOLO_LAYER_CHANNELS["yolov8m"]
            )
            default_indices = self.YOLO_LAYER_INDICES.get(yolo_variant, ["layer4", "layer6", "layer9"])
            layer_configs = {k: default_channels[k] for k in default_indices if k in default_channels}

        self.layer_configs = layer_configs
        self.layer_names = list(layer_configs.keys())
        self.layer_channels = list(layer_configs.values())
        self.spatial_pool = spatial_pool
        self.output_dim = output_dim

        # Feature fusion module
        self.fusion = FeatureFusionModule(
            layer_channels=self.layer_channels,
            output_dim=output_dim,
            fusion_type=fusion_type,
        )

        # Feature maps captured during forward pass
        self._feature_maps: Dict[str, torch.Tensor] = {}
        self._hooks = []

    def _spatial_pool_fn(self, feature_map: torch.Tensor) -> torch.Tensor:
        """Apply spatial pooling to reduce feature map to vector."""
        if self.spatial_pool == "mean":
            return torch.mean(feature_map, dim=(2, 3))
        elif self.spatial_pool == "max":
            return torch.amax(feature_map, dim=(2, 3))
        elif self.spatial_pool == "adaptive":
            mean_pool = torch.mean(feature_map, dim=(2, 3))
            max_pool = torch.amax(feature_map, dim=(2, 3))
            return (mean_pool + max_pool) / 2
        return torch.mean(feature_map, dim=(2, 3))

    def register_hooks(self, model) -> None:
        """
        Register forward hooks on the YOLO model to capture feature maps.

        Args:
            model: YOLO model instance (from ultralytics)
        """
        self._remove_hooks()
        self._feature_maps.clear()

        for layer_name in self.layer_names:
            layer_idx = int(layer_name.replace("layer", ""))

            def make_hook(name):
                def hook_fn(module, input, output):
                    self._feature_maps[name] = output
                return hook_fn

            if hasattr(model, "model") and hasattr(model.model, "model"):
                if layer_idx < len(model.model.model):
                    hook = model.model.model[layer_idx].register_forward_hook(
                        make_hook(layer_name)
                    )
                    self._hooks.append(hook)

    def _remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def extract_roi_features(
        self,
        boxes: np.ndarray,
        image_size: tuple,
    ) -> torch.Tensor:
        """
        Extract ROI features from cached feature maps.

        Args:
            boxes: Detection boxes (N, 4+) [x1, y1, x2, y2, ...]
            image_size: Original image size (H, W)

        Returns:
            Fused features (N, output_dim)
        """
        if len(boxes) == 0:
            return torch.empty(0, self.output_dim)

        h, w = image_size
        device = next(iter(self._feature_maps.values())).device

        all_box_features = []

        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                all_box_features.append(torch.zeros(self.output_dim, device=device))
                continue

            layer_features = []

            for layer_name in self.layer_names:
                feat_map = self._feature_maps[layer_name]
                _, c, fh, fw = feat_map.shape

                # Map box to feature map coordinates
                fx1 = int(x1 * fw / w)
                fy1 = int(y1 * fh / h)
                fx2 = int(x2 * fw / w) + 1
                fy2 = int(y2 * fh / h) + 1

                fx1, fy1 = max(0, fx1), max(0, fy1)
                fx2, fy2 = min(fw, fx2), min(fh, fy2)

                if fx2 <= fx1 or fy2 <= fy1:
                    roi_feat = feat_map.mean(dim=(2, 3))
                else:
                    roi = feat_map[:, :, fy1:fy2, fx1:fx2]
                    roi_feat = self._spatial_pool_fn(roi)

                layer_features.append(roi_feat.squeeze(0))

            # Fuse multi-layer features
            stacked = [f.unsqueeze(0) for f in layer_features]
            fused = self.fusion(stacked).squeeze(0)
            all_box_features.append(fused)

        features = torch.stack(all_box_features)
        features = F.normalize(features, p=2, dim=1)

        return features

    def forward(
        self, feature_maps: Dict[str, torch.Tensor], boxes: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with provided feature maps.

        Args:
            feature_maps: Dict of layer name to feature tensor
            boxes: Detection boxes (N, 4)

        Returns:
            Fused features (N, output_dim)
        """
        self._feature_maps = feature_maps
        # Assume standard image size, should be provided in practice
        return self.extract_roi_features(boxes.cpu().numpy(), (640, 640))


def create_feature_pyramid(
    yolo_variant: str = "yolov8m",
    fusion_type: str = "attention",
    output_dim: int = 128,
    layers: List[str] = None,
) -> MultiScaleFeaturePyramid:
    """
    Factory function to create a MultiScaleFeaturePyramid.

    Args:
        yolo_variant: YOLO model variant (yolov8n, yolov8s, yolov8m, etc.)
        fusion_type: Feature fusion strategy
        output_dim: Output embedding dimension
        layers: Optional custom layer names

    Returns:
        Configured MultiScaleFeaturePyramid instance
    """
    if layers is None:
        layers = MultiScaleFeaturePyramid.YOLO_LAYER_INDICES.get(yolo_variant, ["layer4", "layer6", "layer9"])

    default_channels = MultiScaleFeaturePyramid.YOLO_LAYER_CHANNELS.get(
        yolo_variant, MultiScaleFeaturePyramid.YOLO_LAYER_CHANNELS["yolov8m"]
    )

    layer_configs = {layer: default_channels.get(layer, 128) for layer in layers}

    return MultiScaleFeaturePyramid(
        layer_configs=layer_configs,
        fusion_type=fusion_type,
        output_dim=output_dim,
        yolo_variant=yolo_variant,
    )

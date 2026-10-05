"""
LITE++ Unified Module

Combines all LITE++ innovations in a single interface:
1. Multi-Scale Feature Pyramid (MSFP) - extracts from multiple backbone layers
2. Adaptive Feature Fusion - attention-weighted combination of scales
3. Scene-Adaptive Thresholds (ATL) - learned confidence threshold prediction

This is the primary module for the ACCV 2026 submission.

Usage:
    from litepp import create_litepp

    # Create LITE++ module
    litepp = create_litepp(
        yolo_model,
        fusion_type='attention',
        enable_adaptive_threshold=True,
    )

    # Extract features with adaptive threshold
    features, threshold = litepp.extract_with_adaptive_threshold(image, boxes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Literal, Union

from .feature_pyramid import FeatureFusionModule, MultiScaleFeaturePyramid
from .adaptive_threshold import (
    AdaptiveThresholdModule,
    MultiThresholdPredictor,
    create_adaptive_threshold_module,
)


class LITEPlusPlus(nn.Module):
    """
    LITE++: Multi-Scale Features with Adaptive Thresholds for Real-Time MOT.

    Key innovations:
    1. Multi-Scale Feature Pyramid (MSFP): Features from layers 4, 9, 14
    2. Adaptive Feature Fusion: Learned attention weights across scales
    3. Scene-Adaptive Thresholds: Per-scene confidence prediction

    Args:
        model: YOLO detection model (from ultralytics)
        layer_indices: Backbone layers to extract from
        fusion_type: Feature fusion strategy
        output_dim: Output embedding dimension
        enable_adaptive_threshold: Whether to use ATL
        threshold_module_path: Pretrained threshold weights
        device: Computation device
    """

    # Exact channel counts from paper Table 1 (ACCV 2026). Strides 8/16/32.
    # n/s/m use layers 4/6/9; l/x use layers 4/9/14 (variant-specific P3/P4/SPPF).
    LAYER_CHANNELS = {
        "yolov8n": {"layer4": 64,  "layer6": 128, "layer9":  256},
        "yolov8s": {"layer4": 128, "layer6": 256, "layer9":  512},
        "yolov8m": {"layer4": 192, "layer6": 384, "layer9":  576},
        "yolov8l": {"layer4": 256, "layer9": 512, "layer14": 512},
        "yolov8x": {"layer4": 320, "layer9": 640, "layer14": 640},
    }
    # Default layer indices per variant (P3/P4/SPPF at strides 8/16/32).
    LAYER_INDICES = {
        "yolov8n": ["layer4", "layer6", "layer9"],
        "yolov8s": ["layer4", "layer6", "layer9"],
        "yolov8m": ["layer4", "layer6", "layer9"],
        "yolov8l": ["layer4", "layer9", "layer14"],
        "yolov8x": ["layer4", "layer9", "layer14"],
    }

    def __init__(
        self,
        model,
        layer_indices: List[str] = None,
        fusion_type: Literal["concat", "attention", "adaptive"] = "attention",
        output_dim: int = 128,
        enable_adaptive_threshold: bool = True,
        threshold_module_path: Optional[str] = None,
        device: str = "cuda:0",
        yolo_variant: str = "auto",
    ):
        super().__init__()

        self.model = model
        self.device = device
        self.enable_adaptive_threshold = enable_adaptive_threshold
        self.output_dim = output_dim

        # Default layers: variant-specific P3/P4/SPPF indices (strides 8/16/32).
        if layer_indices is None:
            if yolo_variant in self.LAYER_INDICES:
                layer_indices = self.LAYER_INDICES[yolo_variant]
            else:
                layer_indices = ["layer4", "layer6", "layer9"]
        self.layer_indices = layer_indices

        # Auto-detect or use provided layer channels
        if yolo_variant == "auto":
            self.layer_channels = self._detect_channels(model, layer_indices, device)
        else:
            variant_config = self.LAYER_CHANNELS.get(
                yolo_variant, self.LAYER_CHANNELS["yolov8m"]
            )
            self.layer_channels = [variant_config.get(l, 128) for l in layer_indices]

        # Feature fusion module
        self.fusion_module = FeatureFusionModule(
            layer_channels=self.layer_channels,
            output_dim=output_dim,
            fusion_type=fusion_type,
        ).to(device)

        # Adaptive threshold module
        if enable_adaptive_threshold:
            deepest_channels = self.layer_channels[-1]
            self.threshold_module = create_adaptive_threshold_module(
                input_channels=deepest_channels,
                variant="single",
                pretrained_path=threshold_module_path,
                device=device,
            )
        else:
            self.threshold_module = None

        # Hook storage
        self._feature_maps: Dict[str, torch.Tensor] = {}
        self._hooks = []
        self._register_hooks()

    def _detect_channels(
        self, model, layer_indices: List[str], device: str
    ) -> List[int]:
        """Auto-detect channel counts via forward pass."""
        channels = []
        feature_shapes = {}

        def make_hook(name):
            def hook_fn(module, input, output):
                if isinstance(output, torch.Tensor):
                    feature_shapes[name] = output.shape[1]
            return hook_fn

        hooks = []
        for layer_name in layer_indices:
            layer_idx = int(layer_name.replace("layer", ""))
            if hasattr(model, "model") and hasattr(model.model, "model"):
                if layer_idx < len(model.model.model):
                    hook = model.model.model[layer_idx].register_forward_hook(
                        make_hook(layer_name)
                    )
                    hooks.append(hook)

        try:
            dummy = torch.randn(1, 3, 640, 640).to(device)
            with torch.no_grad():
                model.model.to(device)
                _ = model.model(dummy)
        except Exception:
            pass

        for hook in hooks:
            hook.remove()

        for layer_name in layer_indices:
            channels.append(feature_shapes.get(layer_name, 128))

        return channels

    def _register_hooks(self):
        """Register forward hooks to capture feature maps."""
        self._remove_hooks()

        for layer_name in self.layer_indices:
            layer_idx = int(layer_name.replace("layer", ""))

            def make_hook(name):
                def hook_fn(module, input, output):
                    self._feature_maps[name] = output
                return hook_fn

            if hasattr(self.model, "model") and hasattr(self.model.model, "model"):
                if layer_idx < len(self.model.model.model):
                    handle = self.model.model.model[layer_idx].register_forward_hook(
                        make_hook(layer_name)
                    )
                    self._hooks.append(handle)

    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def _clear_features(self):
        """Clear cached feature maps."""
        self._feature_maps.clear()

    def extract_appearance_features(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
    ) -> np.ndarray:
        """
        Extract multi-scale fused appearance features.

        Args:
            image: Input image (BGR, HWC)
            boxes: Detection boxes (N, 4+) [x1, y1, x2, y2, ...]

        Returns:
            features: (N, output_dim) normalized appearance vectors
        """
        if len(boxes) == 0:
            return np.array([])

        self._clear_features()

        # Run detection to populate feature maps
        with torch.no_grad():
            _ = self.model.predict(image, verbose=False)

        if len(self._feature_maps) != len(self.layer_indices):
            return np.zeros((len(boxes), self.output_dim))

        h, w = image.shape[:2]
        all_box_features = []

        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                all_box_features.append(
                    torch.zeros(self.output_dim, device=self.device)
                )
                continue

            layer_features = []

            for layer_name in self.layer_indices:
                feat_map = self._feature_maps[layer_name].to(self.device)
                _, c, fh, fw = feat_map.shape

                # Map to feature map coordinates
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
                    roi_feat = roi.mean(dim=(2, 3))

                layer_features.append(roi_feat)

            # Fuse multi-layer features
            fused = self.fusion_module(layer_features)
            all_box_features.append(fused.squeeze(0))

        features = torch.stack(all_box_features)
        features = F.normalize(features, p=2, dim=1)

        return features.detach().cpu().numpy()

    def extract_with_adaptive_threshold(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """
        Extract features and predict optimal threshold.

        Args:
            image: Input image
            boxes: Detection boxes

        Returns:
            features: Appearance vectors
            threshold: Predicted confidence threshold
        """
        features = self.extract_appearance_features(image, boxes)

        threshold = 0.25
        if self.enable_adaptive_threshold and self.threshold_module is not None:
            deepest = self.layer_indices[-1]
            if deepest in self._feature_maps:
                with torch.no_grad():
                    feat = self._feature_maps[deepest]
                    pred, _ = self.threshold_module(feat)
                    threshold = pred.item()

        return features, threshold

    def get_adaptive_threshold(self, image: np.ndarray) -> float:
        """Get adaptive threshold without extracting box features."""
        self._clear_features()

        with torch.no_grad():
            _ = self.model.predict(image, verbose=False)

        if not self.enable_adaptive_threshold or self.threshold_module is None:
            return 0.25

        deepest = self.layer_indices[-1]
        if deepest in self._feature_maps:
            pred, _ = self.threshold_module(self._feature_maps[deepest])
            return pred.item()

        return 0.25

    def get_scene_analysis(self, image: np.ndarray) -> Dict[str, float]:
        """Get comprehensive scene analysis."""
        self._clear_features()

        with torch.no_grad():
            _ = self.model.predict(image, verbose=False)

        analysis = {"adaptive_threshold": 0.25, "estimated_density": 0.0}

        if self.enable_adaptive_threshold and self.threshold_module is not None:
            deepest = self.layer_indices[-1]
            if deepest in self._feature_maps:
                feat = self._feature_maps[deepest]

                if isinstance(self.threshold_module, MultiThresholdPredictor):
                    results = self.threshold_module(feat)
                    analysis["adaptive_threshold"] = results[
                        "detection_threshold"
                    ].item()
                    analysis["association_threshold"] = results[
                        "association_threshold"
                    ].item()
                    analysis["max_age"] = results["max_age"].item()
                    analysis["estimated_density"] = results["density"].item()
                else:
                    pred, _ = self.threshold_module(feat)
                    analysis["adaptive_threshold"] = pred.item()

        return analysis

    def get_fusion_weights(self) -> Optional[torch.Tensor]:
        """Get attention weights from fusion module (if using attention fusion)."""
        return self.fusion_module.get_attention_weights()


def create_litepp(
    model,
    fusion_type: Literal["concat", "attention", "adaptive"] = "attention",
    layers: Optional[List[str]] = None,
    output_dim: int = 128,
    enable_adaptive_threshold: bool = True,
    threshold_module_path: Optional[str] = None,
    device: str = "cuda:0",
    yolo_variant: str = "auto",
) -> LITEPlusPlus:
    """
    Factory function to create LITE++ module.

    Args:
        model: YOLO detection model (from ultralytics)
        fusion_type: Feature fusion strategy
        layers: Backbone layers to use (default: layer4, layer9, layer14)
        output_dim: Output feature dimension
        enable_adaptive_threshold: Use scene-adaptive thresholds
        threshold_module_path: Path to pretrained threshold weights
        device: Computing device
        yolo_variant: YOLO variant for channel config (or 'auto')

    Returns:
        Configured LITE++ module
    """
    if layers is None:
        layers = ["layer4", "layer9", "layer14"]

    return LITEPlusPlus(
        model=model,
        layer_indices=layers,
        fusion_type=fusion_type,
        output_dim=output_dim,
        enable_adaptive_threshold=enable_adaptive_threshold,
        threshold_module_path=threshold_module_path,
        device=device,
        yolo_variant=yolo_variant,
    )

"""
LITE++ Model Components

This module contains the core neural network components:
- Feature pyramid extraction from YOLO backbone
- Feature fusion strategies (attention, adaptive, concat)
- Scene-adaptive threshold prediction
"""

from .feature_pyramid import (
    FeatureFusionModule,
    MultiScaleFeaturePyramid,
    InstanceAdaptiveAttentionFusion,
)
from .adaptive_threshold import (
    AdaptiveThresholdModule,
    SceneEncoder,
    MultiThresholdPredictor,
    TemporalThresholdSmoother,
    compute_two_stage_thresholds,
)
from .litepp import LITEPlusPlus, create_litepp

__all__ = [
    "FeatureFusionModule",
    "MultiScaleFeaturePyramid",
    "InstanceAdaptiveAttentionFusion",
    "AdaptiveThresholdModule",
    "SceneEncoder",
    "MultiThresholdPredictor",
    "TemporalThresholdSmoother",
    "compute_two_stage_thresholds",
    "LITEPlusPlus",
    "create_litepp",
]

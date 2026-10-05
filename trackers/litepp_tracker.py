"""
LITE++ Tracker Integration

Provides a drop-in replacement for standard ReID modules in tracking pipelines.
Integrates LITE++ multi-scale features and adaptive thresholds with popular
trackers like DeepSORT, ByteTrack, and OC-SORT.

Usage:
    from litepp.trackers import LITEPlusPlusTracker

    tracker = LITEPlusPlusTracker(
        yolo_model,
        tracker_type='deepsort',
        enable_adaptive_threshold=True,
    )

    # Standard ReID interface
    features = tracker.extract_appearance_features(image, boxes)

    # Get current adaptive threshold
    threshold = tracker.get_current_threshold()
"""

import numpy as np
from typing import Literal, Optional, Dict, List

from ..models.litepp import LITEPlusPlus, create_litepp


class LITEPlusPlusTracker:
    """
    Complete LITE++ tracking wrapper.

    Provides a unified interface for integrating LITE++ features with
    various tracking algorithms. Automatically manages:
    - Multi-scale feature extraction
    - Adaptive threshold prediction
    - Feature caching for efficiency

    Args:
        model: YOLO detection model
        tracker_type: Target tracker for compatibility hints
        fusion_type: Feature fusion strategy
        enable_adaptive_threshold: Use scene-adaptive thresholds
        threshold_module_path: Pretrained threshold weights
        device: Computation device
        layers: Backbone layers to use
        output_dim: Feature embedding dimension
    """

    def __init__(
        self,
        model,
        tracker_type: Literal["deepsort", "bytetrack", "ocsort", "botsort"] = "deepsort",
        fusion_type: Literal["concat", "attention", "adaptive"] = "attention",
        enable_adaptive_threshold: bool = True,
        threshold_module_path: Optional[str] = None,
        device: str = "cuda:0",
        layers: Optional[List[str]] = None,
        output_dim: int = 128,
    ):
        self.model = model
        self.tracker_type = tracker_type
        self.device = device

        # Initialize LITE++ module
        self.reid_module = create_litepp(
            model=model,
            fusion_type=fusion_type,
            layers=layers,
            output_dim=output_dim,
            enable_adaptive_threshold=enable_adaptive_threshold,
            threshold_module_path=threshold_module_path,
            device=device,
        )

        # Tracking state
        self.current_threshold = 0.25
        self.frame_count = 0
        self.threshold_history: List[float] = []

    def extract_appearance_features(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
    ) -> np.ndarray:
        """
        Extract appearance features for detected objects.

        Standard ReID interface compatible with DeepSORT and other trackers.

        Args:
            image: Input frame (BGR, HWC format)
            boxes: Detection boxes (N, 4+) [x1, y1, x2, y2, ...]

        Returns:
            features: (N, output_dim) normalized appearance vectors
        """
        features, threshold = self.reid_module.extract_with_adaptive_threshold(
            image, boxes
        )

        # Update tracking state
        self.current_threshold = threshold
        self.threshold_history.append(threshold)
        self.frame_count += 1

        return features

    def get_current_threshold(self) -> float:
        """Get the most recently predicted confidence threshold."""
        return self.current_threshold

    def update_threshold(self, image: np.ndarray) -> float:
        """
        Update threshold based on current scene without feature extraction.

        Useful for pre-filtering detections before feature extraction.

        Args:
            image: Input frame

        Returns:
            Predicted confidence threshold
        """
        self.current_threshold = self.reid_module.get_adaptive_threshold(image)
        return self.current_threshold

    def get_scene_analysis(self, image: np.ndarray) -> Dict[str, float]:
        """
        Get comprehensive scene analysis.

        Returns:
            Dictionary containing adaptive_threshold, density estimates, etc.
        """
        return self.reid_module.get_scene_analysis(image)

    def get_threshold_statistics(self) -> Dict[str, float]:
        """
        Get statistics about predicted thresholds over time.

        Useful for analyzing threshold behavior across a video.

        Returns:
            Dictionary with mean, std, min, max thresholds
        """
        if not self.threshold_history:
            return {
                "mean": 0.25,
                "std": 0.0,
                "min": 0.25,
                "max": 0.25,
                "frames": 0,
            }

        history = np.array(self.threshold_history)
        return {
            "mean": float(history.mean()),
            "std": float(history.std()),
            "min": float(history.min()),
            "max": float(history.max()),
            "frames": len(history),
        }

    def reset(self):
        """Reset tracker state for new video."""
        self.current_threshold = 0.25
        self.frame_count = 0
        self.threshold_history.clear()

    @property
    def output_dim(self) -> int:
        """Feature embedding dimension."""
        return self.reid_module.output_dim


class ReIDInterface:
    """
    Abstract base class defining the standard ReID interface.

    Implement this interface to create custom ReID modules compatible
    with LITE++ tracking pipelines.
    """

    def extract_appearance_features(
        self, image: np.ndarray, boxes: np.ndarray
    ) -> np.ndarray:
        """Extract appearance features for detected boxes."""
        raise NotImplementedError

    @property
    def output_dim(self) -> int:
        """Return the feature embedding dimension."""
        raise NotImplementedError


def create_tracker(
    model,
    tracker_type: str = "deepsort",
    fusion_type: str = "attention",
    enable_adaptive_threshold: bool = True,
    device: str = "cuda:0",
) -> LITEPlusPlusTracker:
    """
    Factory function to create LITE++ tracker.

    Args:
        model: YOLO detection model
        tracker_type: Target tracker type
        fusion_type: Feature fusion strategy
        enable_adaptive_threshold: Enable ATL
        device: Computing device

    Returns:
        Configured LITE++ tracker
    """
    return LITEPlusPlusTracker(
        model=model,
        tracker_type=tracker_type,
        fusion_type=fusion_type,
        enable_adaptive_threshold=enable_adaptive_threshold,
        device=device,
    )

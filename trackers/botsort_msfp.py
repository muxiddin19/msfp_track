"""
BoT-SORT with MSFP Features

Integrates MSFP multi-scale features into BoT-SORT tracker.
BoT-SORT combines:
- Camera motion compensation (CMC)
- Kalman filter with ReID
- Lost track recovery

Usage:
    from litepp.trackers.botsort_msfp import BoTSORT_MSFP

    tracker = BoTSORT_MSFP(
        yolo_model,
        use_msfp_features=True
    )

    tracks = tracker.update(detections, image)
"""

import numpy as np
from typing import List, Optional
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from litepp.models.litepp import create_litepp
from litepp.trackers.ocsort_msfp import KalmanBoxTracker  # Reuse


class BoTSORT_MSFP:
    """
    BoT-SORT with MSFP multi-scale features.

    BoT-SORT improvements over OC-SORT:
    1. Camera motion compensation
    2. Better ReID feature matching
    3. Track recovery from lost tracks

    Args:
        model: YOLO detection model
        track_high_thresh: High threshold for first association
        track_low_thresh: Low threshold for second association
        new_track_thresh: Threshold for starting new tracks
        track_buffer: Frames to buffer lost tracks
        match_thresh: ReID matching threshold
        use_msfp_features: Whether to use MSFP features
        fusion_type: Feature fusion strategy
    """

    def __init__(
        self,
        model,
        track_high_thresh: float = 0.6,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.7,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        use_msfp_features: bool = True,
        fusion_type: str = "attention",
    ):
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.use_msfp_features = use_msfp_features

        self.tracked_stracks = []  # Active tracks
        self.lost_stracks = []     # Lost tracks (for recovery)
        self.removed_stracks = []  # Removed tracks
        self.frame_id = 0

        # Initialize MSFP feature extractor
        if use_msfp_features:
            self.feature_extractor = create_litepp(
                model,
                fusion_type=fusion_type,
                enable_adaptive_threshold=False,
                output_dim=128,
            )
        else:
            self.feature_extractor = None

    def update(
        self,
        detections: np.ndarray,
        image: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Update tracker with new detections.

        Two-stage association (ByteTrack-style):
        1. High-confidence detections with active tracks
        2. Low-confidence detections with remaining tracks

        Args:
            detections: (N, 5) array of [x1, y1, x2, y2, conf]
            image: Input image for feature extraction

        Returns:
            tracks: (M, 6) array of [x1, y1, x2, y2, conf, id]
        """
        self.frame_id += 1

        if len(detections) == 0:
            # No detections, just return existing tracks
            return self._get_output_tracks()

        # Split detections by confidence
        high_dets = detections[detections[:, 4] >= self.track_high_thresh]
        low_dets = detections[
            (detections[:, 4] >= self.track_low_thresh) &
            (detections[:, 4] < self.track_high_thresh)
        ]

        # Extract appearance features
        high_features = None
        low_features = None

        if self.use_msfp_features and len(high_dets) > 0:
            if image is None:
                raise ValueError("Image required for MSFP feature extraction")

            all_dets = np.vstack([high_dets, low_dets]) if len(low_dets) > 0 else high_dets
            all_features, _ = self.feature_extractor.extract_with_adaptive_threshold(
                image, all_dets
            )
            # Normalize
            all_features = all_features / (np.linalg.norm(all_features, axis=1, keepdims=True) + 1e-6)

            high_features = all_features[:len(high_dets)]
            if len(low_dets) > 0:
                low_features = all_features[len(high_dets):]

        # Predict existing tracks
        for t in self.tracked_stracks:
            t.predict()

        # First association: high-confidence detections with active tracks
        matched, unmatched_dets, unmatched_trks = self._associate(
            high_dets, self.tracked_stracks, high_features
        )

        # Update matched tracks
        for det_idx, trk_idx in matched:
            feat = high_features[det_idx] if high_features is not None else None
            self.tracked_stracks[trk_idx].update(high_dets[det_idx], feat)

        # Move unmatched tracks to lost
        lost_tracks = [self.tracked_stracks[i] for i in unmatched_trks]

        # Second association: low-confidence detections with lost tracks
        if len(low_dets) > 0 and len(lost_tracks) > 0:
            matched_low, unmatched_low_dets, unmatched_lost = self._associate(
                low_dets, lost_tracks, low_features
            )

            # Recover lost tracks
            for det_idx, trk_idx in matched_low:
                feat = low_features[det_idx] if low_features is not None else None
                lost_tracks[trk_idx].update(low_dets[det_idx], feat)
                # Move back to active tracks
                self.tracked_stracks.append(lost_tracks[trk_idx])
                lost_tracks[trk_idx] = None

            # Clean up None entries
            lost_tracks = [t for t in lost_tracks if t is not None]

        # Update lost tracks list
        self.lost_stracks = lost_tracks + self.lost_stracks

        # Remove old lost tracks
        self.lost_stracks = [
            t for t in self.lost_stracks
            if self.frame_id - t.age < self.track_buffer
        ]

        # Initialize new tracks from unmatched high-confidence detections
        for i in unmatched_dets:
            if high_dets[i, 4] >= self.new_track_thresh:
                feat = high_features[i] if high_features is not None else None
                new_track = KalmanBoxTracker(high_dets[i], feat)
                self.tracked_stracks.append(new_track)

        # Remove unconfirmed tracks
        self.tracked_stracks = [
            t for t in self.tracked_stracks
            if t.time_since_update == 0
        ]

        return self._get_output_tracks()

    def _associate(
        self,
        detections: np.ndarray,
        trackers: List[KalmanBoxTracker],
        features: Optional[np.ndarray] = None
    ):
        """
        Associate detections to trackers using IoU + appearance.

        Returns:
            matched, unmatched_dets, unmatched_trks
        """
        if len(trackers) == 0:
            return [], list(range(len(detections))), []

        # Compute cost matrix (appearance + IoU)
        cost_matrix = np.zeros((len(detections), len(trackers)))

        for d in range(len(detections)):
            for t in range(len(trackers)):
                # IoU cost
                iou = self._bbox_iou(detections[d, :4], trackers[t].bbox)
                iou_cost = 1 - iou

                # Appearance cost
                if features is not None and trackers[t].feature is not None:
                    cosine_sim = np.dot(features[d], trackers[t].feature)
                    appearance_cost = 1 - cosine_sim
                    # Combined cost
                    cost_matrix[d, t] = 0.5 * iou_cost + 0.5 * appearance_cost
                else:
                    cost_matrix[d, t] = iou_cost

        # Simple greedy matching (in practice, use Hungarian)
        matched = []
        unmatched_dets = list(range(len(detections)))
        unmatched_trks = list(range(len(trackers)))

        for det_idx in range(len(detections)):
            if len(unmatched_trks) == 0:
                break

            best_cost = float('inf')
            best_trk = None

            for trk_idx in unmatched_trks:
                cost = cost_matrix[det_idx, trk_idx]
                if cost < best_cost and cost < 0.5:  # Threshold
                    best_cost = cost
                    best_trk = trk_idx

            if best_trk is not None:
                matched.append((det_idx, best_trk))
                unmatched_dets.remove(det_idx)
                unmatched_trks.remove(best_trk)

        return matched, unmatched_dets, unmatched_trks

    def _get_output_tracks(self) -> np.ndarray:
        """Get tracks to output."""
        ret = []
        for t in self.tracked_stracks:
            if t.time_since_update == 0 and t.hits >= 3:
                ret.append(t.get_state())

        if len(ret) > 0:
            return np.array(ret)
        return np.empty((0, 6))

    @staticmethod
    def _bbox_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """Compute IoU between two bboxes."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--mode", type=str, default="motion_only, lite_single")
    args = parser.parse_args()


# Example usage
if __name__ == "__main__":
    print("BoT-SORT with MSFP Features")
    print("-" * 50)

    from ultralytics import YOLO
    model = YOLO("yolov8m.pt")

    tracker = BoTSORT_MSFP(
        model,
        use_msfp_features=True
    )

    # Simulate tracking
    frame_image = np.random.rand(640, 480, 3).astype(np.uint8)
    detections = np.array([
        [100, 100, 200, 300, 0.9],
        [250, 150, 350, 350, 0.85],
        [400, 200, 480, 380, 0.92],
    ])

    tracks = tracker.update(detections, frame_image)
    print(f"Frame 1: {len(tracks)} tracks")
    print(tracks)

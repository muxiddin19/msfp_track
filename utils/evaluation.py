"""
Evaluation Utilities for LITE++

Functions for computing ReID and tracking metrics.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# TrackEval (1.0.dev1) predates numpy's removal of deprecated scalar aliases.
np.float = float
np.int = int
np.bool = bool


def compute_cosine_similarity(
    features1: np.ndarray,
    features2: np.ndarray,
) -> np.ndarray:
    """
    Compute pairwise cosine similarity between two feature sets.

    Args:
        features1: (N, D) feature vectors
        features2: (M, D) feature vectors

    Returns:
        similarity: (N, M) similarity matrix
    """
    # Normalize features
    features1 = features1 / (np.linalg.norm(features1, axis=1, keepdims=True) + 1e-8)
    features2 = features2 / (np.linalg.norm(features2, axis=1, keepdims=True) + 1e-8)

    return np.dot(features1, features2.T)


def compute_reid_metrics(
    query_features: np.ndarray,
    query_ids: np.ndarray,
    gallery_features: np.ndarray,
    gallery_ids: np.ndarray,
    top_k: List[int] = [1, 5, 10],
) -> Dict[str, float]:
    """
    Compute ReID evaluation metrics (CMC, mAP).

    Args:
        query_features: (N_q, D) query feature vectors
        query_ids: (N_q,) query identity labels
        gallery_features: (N_g, D) gallery feature vectors
        gallery_ids: (N_g,) gallery identity labels
        top_k: Ranks for CMC computation

    Returns:
        Dictionary with CMC@k and mAP scores
    """
    similarity = compute_cosine_similarity(query_features, gallery_features)

    # Sort gallery by similarity for each query
    indices = np.argsort(-similarity, axis=1)

    metrics = {}

    # CMC (Cumulative Matching Characteristics)
    for k in top_k:
        correct = 0
        for i, q_id in enumerate(query_ids):
            top_k_ids = gallery_ids[indices[i, :k]]
            if q_id in top_k_ids:
                correct += 1
        metrics[f"cmc@{k}"] = correct / len(query_ids)

    # mAP (Mean Average Precision)
    aps = []
    for i, q_id in enumerate(query_ids):
        sorted_ids = gallery_ids[indices[i]]
        matches = sorted_ids == q_id

        if matches.sum() == 0:
            continue

        # Compute AP
        cum_matches = np.cumsum(matches)
        precision_at_k = cum_matches / np.arange(1, len(matches) + 1)
        ap = (precision_at_k * matches).sum() / matches.sum()
        aps.append(ap)

    metrics["mAP"] = np.mean(aps) if aps else 0.0

    return metrics


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (N,4) xyxy, b: (M,4) xyxy -> (N,M) IoU."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    inter_x1 = np.maximum(ax1, bx1); inter_y1 = np.maximum(ay1, by1)
    inter_x2 = np.minimum(ax2, bx2); inter_y2 = np.minimum(ay2, by2)
    iw = np.clip(inter_x2 - inter_x1, 0, None)
    ih = np.clip(inter_y2 - inter_y1, 0, None)
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


def _load_mot_format(path: str, is_gt: bool) -> np.ndarray:
    """Load a MOT-format txt file: frame,id,x,y,w,h,[conf,class,vis] (gt)
    or frame,id,x,y,w,h,conf,-1,-1,-1 (prediction/tracker output)."""
    rows = np.loadtxt(path, delimiter=",")
    if rows.ndim == 1:
        rows = rows[None, :]
    if is_gt and rows.shape[1] >= 8:
        # standard MOTChallenge gt.txt: keep only "considered" pedestrian boxes
        mask = (rows[:, 6] == 1) & (rows[:, 7] == 1)
        rows = rows[mask]
    return rows


def _build_hota_data(gt_rows: np.ndarray, trk_rows: np.ndarray) -> Dict:
    """Build the raw per-timestep data dict TrackEval's HOTA/CLEAR/Identity
    metrics expect (see trackeval/metrics/hota.py eval_sequence), directly
    from two MOT-format arrays -- avoids TrackEval's finicky benchmark/seqmap
    folder conventions, which is what made the previous version of this
    function impossible to finish."""
    frames = np.unique(np.concatenate([gt_rows[:, 0], trk_rows[:, 0]])).astype(int) \
        if len(trk_rows) else np.unique(gt_rows[:, 0]).astype(int)

    gt_id_map: Dict[int, int] = {}
    trk_id_map: Dict[int, int] = {}
    gt_ids_list, trk_ids_list, sim_list = [], [], []
    num_gt_dets = num_trk_dets = 0

    for fi in frames:
        f_gt = gt_rows[gt_rows[:, 0] == fi]
        gt_boxes = f_gt[:, 2:6].copy()
        gt_boxes[:, 2] += gt_boxes[:, 0]; gt_boxes[:, 3] += gt_boxes[:, 1]
        gt_ids = np.array([gt_id_map.setdefault(int(x), len(gt_id_map)) for x in f_gt[:, 1]], dtype=int)

        f_trk = trk_rows[trk_rows[:, 0] == fi] if len(trk_rows) else np.zeros((0, 6))
        trk_boxes = f_trk[:, 2:6].copy()
        if len(trk_boxes):
            trk_boxes[:, 2] += trk_boxes[:, 0]; trk_boxes[:, 3] += trk_boxes[:, 1]
        trk_ids = np.array([trk_id_map.setdefault(int(x), len(trk_id_map)) for x in f_trk[:, 1]], dtype=int)

        sim = _iou_matrix(gt_boxes, trk_boxes)
        gt_ids_list.append(gt_ids); trk_ids_list.append(trk_ids); sim_list.append(sim)
        num_gt_dets += len(gt_ids); num_trk_dets += len(trk_ids)

    return {
        "num_gt_ids": len(gt_id_map), "num_tracker_ids": len(trk_id_map),
        "num_gt_dets": num_gt_dets, "num_tracker_dets": num_trk_dets,
        "gt_ids": gt_ids_list, "tracker_ids": trk_ids_list,
        "similarity_scores": sim_list, "num_timesteps": len(frames),
    }


def evaluate_tracking(
    predictions_path: str,
    groundtruth_path: str,
    metrics: List[str] = ["HOTA", "MOTA", "IDF1"],
) -> Dict[str, float]:
    """
    Evaluate tracking results using TrackEval.

    Both `predictions_path` and `groundtruth_path` point to a single
    sequence's MOT-format .txt file (frame,id,x,y,w,h,...). For multiple
    sequences, call this once per sequence and combine with TrackEval's own
    `combine_sequences` on the returned per-metric result objects if
    sequence-level aggregation is needed, the same way real_repro/
    track_and_eval.py does it.

    NOTE: an earlier version of this function set up a
    trackeval.datasets.MotChallenge2DBox + Evaluator().evaluate() call and
    then returned an empty dict with a "parse raw_results" TODO -- it never
    actually extracted or returned anything. This version builds the raw
    per-timestep data dict directly (bypassing the benchmark/seqmap folder
    convention that was never finished) and returns real, computed numbers.

    Args:
        predictions_path: Path to a prediction file (MOT format)
        groundtruth_path: Path to a ground-truth file (MOT format)
        metrics: Which of "HOTA", "MOTA", "IDF1", "IDSW" to include

    Returns:
        Dictionary with the requested evaluation results (percentage scale
        for HOTA/MOTA/IDF1, matching how the paper's tables report them)
    """
    import trackeval

    gt_rows = _load_mot_format(groundtruth_path, is_gt=True)
    trk_rows = _load_mot_format(predictions_path, is_gt=False)
    data = _build_hota_data(gt_rows, trk_rows)

    results: Dict[str, float] = {}
    if "HOTA" in metrics:
        hres = trackeval.metrics.HOTA().eval_sequence(data)
        results["HOTA"] = float(np.mean(hres["HOTA"])) * 100
        results["DetA"] = float(np.mean(hres["DetA"])) * 100
        results["AssA"] = float(np.mean(hres["AssA"])) * 100
    if "MOTA" in metrics or "IDSW" in metrics:
        cres = trackeval.metrics.CLEAR().eval_sequence(data)
        if "MOTA" in metrics:
            results["MOTA"] = float(cres["MOTA"]) * 100
        if "IDSW" in metrics:
            results["IDSW"] = int(cres["IDSW"])
    if "IDF1" in metrics:
        ires = trackeval.metrics.Identity().eval_sequence(data)
        results["IDF1"] = float(ires["IDF1"]) * 100

    return results


def compute_feature_distance_stats(
    features: np.ndarray,
    ids: np.ndarray,
) -> Dict[str, float]:
    """
    Compute intra-class and inter-class distance statistics.

    Useful for analyzing feature quality.

    Args:
        features: (N, D) feature vectors
        ids: (N,) identity labels

    Returns:
        Statistics about feature distances
    """
    unique_ids = np.unique(ids)

    intra_distances = []
    inter_distances = []

    for uid in unique_ids:
        mask = ids == uid
        class_features = features[mask]

        if len(class_features) > 1:
            # Intra-class distances
            sim = compute_cosine_similarity(class_features, class_features)
            # Get upper triangle (excluding diagonal)
            triu_idx = np.triu_indices(len(class_features), k=1)
            intra_distances.extend(1 - sim[triu_idx])

        # Inter-class distances
        other_features = features[~mask]
        if len(other_features) > 0 and len(class_features) > 0:
            sim = compute_cosine_similarity(class_features, other_features)
            inter_distances.extend(1 - sim.flatten())

    return {
        "intra_mean": np.mean(intra_distances) if intra_distances else 0.0,
        "intra_std": np.std(intra_distances) if intra_distances else 0.0,
        "inter_mean": np.mean(inter_distances) if inter_distances else 0.0,
        "inter_std": np.std(inter_distances) if inter_distances else 0.0,
        "separation": (
            (np.mean(inter_distances) - np.mean(intra_distances))
            if intra_distances and inter_distances
            else 0.0
        ),
    }

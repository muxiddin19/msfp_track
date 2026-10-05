#!/usr/bin/env python3
"""
Real oracle-threshold grid search (paper Sec. 3.3): for each MOT17-train
sequence, run the real Kalman+MSFP tracker (real_tracker.py) at each
candidate confidence threshold tau in {0.01, 0.05, 0.10, ..., 0.50},
score it with the real, official TrackEval HOTA implementation, and select
tau* = argmax_tau HOTA(tau) per sequence. These are the oracle targets that
the ATL scene encoder is trained to regress (train_atl.py).

This is the real computation that the placeholder in
litepp/scripts/measure_training_time.py (`# TODO: Replace with your actual
training code`) and the fake mock-data path previously in
run_threshold_comparison.py stood in for.
"""
import argparse
import json
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
# TrackEval (as of its latest PyPI/git release) still uses several numpy
# scalar-type aliases removed in NumPy >=1.24 (np.float, np.int, np.bool,
# np.object); restore them rather than pin an older NumPy that would
# conflict with torch/ultralytics in this environment.
for _name, _builtin in [("float", float), ("int", int), ("bool", bool), ("object", object)]:
    if not hasattr(np, _name):
        setattr(np, _name, _builtin)
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from real_tracker import ByteTrackStyleTracker

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from litepp.models.feature_pyramid import FeatureFusionModule

LAYER_CHANNELS_V8M = {"layer4": 192, "layer6": 384, "layer9": 576}
LAYER_ORDER = ["layer4", "layer6", "layer9"]
GRID = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]


def load_det_cache(cache_dir: Path, seq: str):
    records = torch.load(cache_dir / f"{seq}.pt", weights_only=False)
    by_frame = defaultdict(list)
    for r in records:
        by_frame[r["frame"]].append(r)
    return by_frame


@torch.no_grad()
def fuse_features(records, fusion, device):
    """Run the trained fusion head once per detection; reused across every
    threshold candidate (fusion does not depend on tau)."""
    if not records:
        return np.zeros((0, 128), dtype=np.float32)
    layer_feats = [torch.stack([r[name] for r in records]).to(device) for name in LAYER_ORDER]
    emb = fusion(layer_feats)
    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
    return emb.cpu().numpy()


def run_tracker_single_threshold(by_frame, fused_by_frame, tau, n_frames):
    """Single-threshold filtering (paper Eq. 5/6 oracle grid: keep score>=tau,
    no low-confidence recovery stage) using the real Kalman+appearance tracker."""
    tracker = ByteTrackStyleTracker(tau_h=tau, tau_l=tau, max_age=30, min_hits=1,
                                     appearance_weight=0.5, high_stage_cost_thresh=0.7)
    lines = []
    for frame in range(1, n_frames + 1):
        recs = by_frame.get(frame, [])
        if recs:
            boxes = np.array([r["bbox"] for r in recs], dtype=np.float32)
            scores = np.array([r["det_conf"] for r in recs], dtype=np.float32)
            feats = fused_by_frame[frame]
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            scores = np.zeros((0,), dtype=np.float32)
            feats = np.zeros((0, 128), dtype=np.float32)
        results = tracker.update(boxes, scores, feats)
        for tid, x1, y1, x2, y2 in results:
            w, h = x2 - x1, y2 - y1
            lines.append(f"{frame},{tid},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,-1,-1,-1")
    return lines


def count_frames(mot_root: Path, seq: str) -> int:
    img_dir = mot_root / seq / "img1"
    return len(list(img_dir.glob("*.jpg")))


def setup_trackeval_dirs(workdir: Path, mot_root: Path, sequences, tracker_name):
    # With SKIP_SPLIT_FOL=True, TrackEval expects GT_FOLDER/{seq}/gt/gt.txt
    # and TRACKERS_FOLDER/{tracker_name}/data/{seq}.txt directly (no extra
    # benchmark/split subfolder level).
    gt_root = workdir / "gt"
    trackers_root = workdir / "trackers"
    (trackers_root / tracker_name / "data").mkdir(parents=True, exist_ok=True)
    for seq in sequences:
        seq_gt_dir = gt_root / seq / "gt"
        seq_gt_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(mot_root / seq / "gt" / "gt.txt", seq_gt_dir / "gt.txt")
        shutil.copy(mot_root / seq / "seqinfo.ini", gt_root / seq / "seqinfo.ini")
    return gt_root, trackers_root


def evaluate_hota(workdir: Path, gt_folder: Path, trackers_folder: Path,
                   sequences, tracker_name, seq_info):
    import trackeval
    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config['PRINT_RESULTS'] = False
    eval_config['PRINT_CONFIG'] = False
    eval_config['TIME_PROGRESS'] = False
    eval_config['DISPLAY_LESS_PROGRESS'] = True
    eval_config['OUTPUT_SUMMARY'] = False
    eval_config['OUTPUT_DETAILED'] = False
    eval_config['PLOT_CURVES'] = False

    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config['GT_FOLDER'] = str(gt_folder)
    dataset_config['TRACKERS_FOLDER'] = str(trackers_folder)
    dataset_config['SKIP_SPLIT_FOL'] = True
    dataset_config['SEQ_INFO'] = seq_info
    dataset_config['TRACKERS_TO_EVAL'] = [tracker_name]
    dataset_config['CLASSES_TO_EVAL'] = ['pedestrian']
    dataset_config['BENCHMARK'] = 'MOT17'
    dataset_config['PRINT_CONFIG'] = False

    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    metrics_list = [trackeval.metrics.HOTA({'PRINT_CONFIG': False})]
    results, _ = evaluator.evaluate(dataset_list, metrics_list)

    hota_by_seq = {}
    seq_results = results['MotChallenge2DBox'][tracker_name]
    for seq in sequences:
        hota_arr = seq_results[seq]['pedestrian']['HOTA']['HOTA']
        hota_by_seq[seq] = float(np.mean(hota_arr)) * 100.0
    return hota_by_seq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mot_root", default="/nas/Dataset/MOT/MOT17/train")
    ap.add_argument("--cache_dir", default="~/msfp/cache_det")
    ap.add_argument("--checkpoint", default="~/msfp/checkpoints/fusion_attention.pt")
    ap.add_argument("--sequences", nargs="+",
                     default=["MOT17-02-FRCNN", "MOT17-04-FRCNN", "MOT17-05-FRCNN",
                              "MOT17-09-FRCNN", "MOT17-10-FRCNN", "MOT17-11-FRCNN",
                              "MOT17-13-FRCNN"])
    ap.add_argument("--workdir", default="~/msfp/trackeval_work")
    ap.add_argument("--out", default="~/msfp/checkpoints/oracle_thresholds.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mot_root = Path(args.mot_root)
    cache_dir = Path(args.cache_dir).expanduser()
    workdir = Path(args.workdir).expanduser()
    workdir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(Path(args.checkpoint).expanduser(), map_location=device, weights_only=False)
    layer_channels = [LAYER_CHANNELS_V8M[n] for n in LAYER_ORDER]
    fusion = FeatureFusionModule(layer_channels=layer_channels, output_dim=128,
                                  fusion_type="attention").to(device)
    fusion.load_state_dict(ckpt["state_dict"])
    fusion.eval()

    seq_info = {}
    print("Fusing detection features (once per sequence, reused across all tau)...")
    fused_cache = {}
    by_frame_cache = {}
    t_fuse_start = time.time()
    for seq in args.sequences:
        by_frame = load_det_cache(cache_dir, seq)
        n_frames = count_frames(mot_root, seq)
        seq_info[seq] = n_frames
        fused_by_frame = {}
        for frame, recs in by_frame.items():
            fused_by_frame[frame] = fuse_features(recs, fusion, device)
        by_frame_cache[seq] = by_frame
        fused_cache[seq] = fused_by_frame
        print(f"  {seq}: {n_frames} frames, {sum(len(v) for v in by_frame.values())} detections")
    t_fuse = time.time() - t_fuse_start

    tracker_name = "MSFP-Track-oracle"
    gt_folder, trackers_folder = setup_trackeval_dirs(workdir, mot_root, args.sequences, tracker_name)

    hota_grid = {seq: {} for seq in args.sequences}
    t_track_start = time.time()
    for tau in GRID:
        for seq in args.sequences:
            lines = run_tracker_single_threshold(by_frame_cache[seq], fused_cache[seq], tau, seq_info[seq])
            out_file = trackers_folder / tracker_name / "data" / f"{seq}.txt"
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_text("\n".join(lines), encoding="utf-8")
        hota_by_seq = evaluate_hota(workdir, gt_folder, trackers_folder, args.sequences, tracker_name, seq_info)
        for seq in args.sequences:
            hota_grid[seq][tau] = hota_by_seq[seq]
        print(f"tau={tau:.2f}: " + ", ".join(f"{s}={hota_by_seq[s]:.1f}" for s in args.sequences))
    t_track = time.time() - t_track_start

    oracle = {}
    for seq in args.sequences:
        best_tau = max(hota_grid[seq], key=hota_grid[seq].get)
        oracle[seq] = {"tau_star": best_tau, "hota_at_tau_star": hota_grid[seq][best_tau],
                        "full_grid": hota_grid[seq]}
        print(f"[{seq}] oracle tau*={best_tau:.2f} (HOTA={hota_grid[seq][best_tau]:.2f})")

    result = {
        "oracle_per_sequence": oracle,
        "timing": {
            "feature_fusion_seconds": t_fuse,
            "grid_search_tracking_and_eval_seconds": t_track,
            "total_seconds": t_fuse + t_track,
            "total_hours": (t_fuse + t_track) / 3600,
            "n_grid_points": len(GRID),
            "n_sequences": len(args.sequences),
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        },
    }
    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nReal oracle-search time: {result['timing']['total_hours']:.4f} GPU-hours -> {out_path}")


if __name__ == "__main__":
    main()

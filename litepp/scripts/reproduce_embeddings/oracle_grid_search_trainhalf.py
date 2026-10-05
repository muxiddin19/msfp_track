#!/usr/bin/env python3
"""
Leak-free oracle-threshold grid search: same real procedure as
run_oracle_grid_search.py (real Kalman+MSFP tracker at each candidate tau,
real TrackEval HOTA, tau* = argmax), but restricted to train_half frames
for BOTH tracking and GT evaluation. The existing oracle_thresholds.json
was computed over the FULL sequence (tracking range = all frames, GT
evaluation = full gt.txt) -- confirmed by direct reading of
run_oracle_grid_search.py's count_frames()/setup_trackeval_dirs() calls,
neither of which takes a split argument. That means tau* itself (what ATL
is trained to regress) was chosen with direct visibility into val_half
ground truth, a real leak into the training target, not just the training
data. This script produces the oracle targets the documented protocol
actually requires.

Uses the real, already-trained (leak-free) fusion head from
train_fusion_sweep.py (fusion_default_leakfree.pt) for feature fusion,
since that is the real embedding this oracle search should be consistent
with going forward.
"""
import json
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
for _name, _builtin in [("float", float), ("int", int), ("bool", bool), ("object", object)]:
    if not hasattr(np, _name):
        setattr(np, _name, _builtin)
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from real_tracker import ByteTrackStyleTracker
import verify_official_trackers as vo
from litepp.models.feature_pyramid import FeatureFusionModule

LAYER_CHANNELS_V8M = {"layer4": 192, "layer6": 384, "layer9": 576}
LAYER_ORDER = ["layer4", "layer6", "layer9"]
GRID = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]


@torch.no_grad()
def fuse_features(records, fusion, device):
    if not records:
        return np.zeros((0, 128), dtype=np.float32)
    layer_feats = [torch.stack([r[name] for r in records]).to(device) for name in LAYER_ORDER]
    emb = fusion(layer_feats)
    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
    return emb.cpu().numpy()


def run_tracker_single_threshold(by_frame, fused_by_frame, tau, frame_hi):
    tracker = ByteTrackStyleTracker(tau_h=tau, tau_l=tau, max_age=30, min_hits=1,
                                     appearance_weight=0.5, high_stage_cost_thresh=0.7)
    lines = []
    for frame in range(1, frame_hi + 1):
        recs = by_frame.get(frame, [])
        if recs:
            boxes = np.array([r["bbox"] for r in recs], dtype=np.float32)
            scores = np.array([r["det_conf"] for r in recs], dtype=np.float32)
            feats = fused_by_frame.get(frame, np.zeros((len(recs), 128), dtype=np.float32))
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            scores = np.zeros((0,), dtype=np.float32)
            feats = np.zeros((0, 128), dtype=np.float32)
        for tid, x1, y1, x2, y2 in tracker.update(boxes, scores, feats):
            w, h = x2 - x1, y2 - y1
            lines.append(f"{frame},{tid},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,-1,-1,-1")
    return lines


def setup_trackeval_dirs_trainhalf(workdir, mot_root, sequences, tracker_name, splits):
    """GT + seqinfo restricted to frames <= split (train_half), no renumbering needed
    since train_half already starts at frame 1."""
    import re
    gt_root = workdir / "gt"
    trackers_root = workdir / "trackers"
    (trackers_root / tracker_name / "data").mkdir(parents=True, exist_ok=True)
    for seq in sequences:
        seq_gt_dir = gt_root / seq / "gt"
        seq_gt_dir.mkdir(parents=True, exist_ok=True)
        lines_out = []
        for line in (mot_root / seq / "gt" / "gt.txt").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            frame = int(line.split(",")[0])
            if frame > splits[seq]:
                continue
            lines_out.append(line)
        (seq_gt_dir / "gt.txt").write_text("\n".join(lines_out), encoding="utf-8")
        src_ini = (mot_root / seq / "seqinfo.ini").read_text(encoding="utf-8")
        dst_ini = re.sub(r"seqLength=\d+", f"seqLength={splits[seq]}", src_ini)
        (gt_root / seq / "seqinfo.ini").write_text(dst_ini, encoding="utf-8")
    return gt_root, trackers_root


def load_det_cache_trainhalf(seq_dir: Path, frame_hi: int):
    """Real public det.txt (not GT), restricted to train_half frames."""
    by_frame = defaultdict(list)
    with open(seq_dir / "det" / "det.txt", encoding="utf-8") as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) < 7:
                continue
            frame = int(p[0])
            if frame > frame_hi:
                continue
            x, y, w, h, conf = float(p[2]), float(p[3]), float(p[4]), float(p[5]), float(p[6])
            by_frame[frame].append({"bbox": [x, y, x + w, y + h], "det_conf": conf, "frame": frame})
    return by_frame


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mot_root = Path("/nas/Dataset/MOT/MOT17/train")
    workdir = Path("~/msfp_honest_repro/trackeval_oracle_trainhalf").expanduser()
    workdir.mkdir(parents=True, exist_ok=True)

    print("Loading real, leak-free fusion head (fusion_default_leakfree.pt)...")
    ckpt = torch.load(Path("~/msfp_honest_repro/checkpoints/fusion_default_leakfree.pt").expanduser(),
                       map_location=device, weights_only=False)
    layer_channels = [LAYER_CHANNELS_V8M[n] for n in LAYER_ORDER]
    fusion = FeatureFusionModule(layer_channels=layer_channels, output_dim=128,
                                  fusion_type="attention", dropout_p=0.1).to(device)
    fusion.load_state_dict(ckpt["state_dict"])
    fusion.eval()

    backbone = vo.HookedBackbone(device)
    splits = {seq: vo.half_split_frame_range(mot_root / seq) for seq in vo.SEQUENCES}
    print(f"train_half splits: {splits}")

    by_frame_cache, fused_cache = {}, {}
    for seq in vo.SEQUENCES:
        print(f"[{seq}] extracting real public-det features (train_half only, frames 1-{splits[seq]})...")
        recs_raw = load_det_cache_trainhalf(mot_root / seq, splits[seq])
        # Need real RoIAlign features (layer4/6/9), not just boxes -- reuse extract_val_half's
        # sibling logic via a frame_hi-bounded pass, matching extract_cached_features_trainhalf.py's
        # image loop but keyed on det.txt (public) rather than gt.txt.
        records = {}
        import cv2
        from torchvision.ops import roi_align
        img_dir = mot_root / seq / "img1"
        for frame, dets in sorted(recs_raw.items()):
            img_path = img_dir / f"{frame:06d}.jpg"
            if not img_path.exists():
                continue
            img = cv2.imread(str(img_path))
            h0, w0 = img.shape[:2]
            imgsz_multiple = 32
            h = ((h0 + imgsz_multiple - 1) // imgsz_multiple) * imgsz_multiple
            w = ((w0 + imgsz_multiple - 1) // imgsz_multiple) * imgsz_multiple
            padded = cv2.copyMakeBorder(img, 0, h - h0, 0, w - w0, cv2.BORDER_CONSTANT, value=(114, 114, 114))
            rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(rgb).to(device).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            feats = backbone.forward(img_tensor)
            boxes_xyxy, confs = [], []
            for d in dets:
                x1, y1, x2, y2 = d["bbox"]
                x1, y1 = max(0.0, x1), max(0.0, y1)
                x2, y2 = min(float(w0), x2), min(float(h0), y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                boxes_xyxy.append([x1, y1, x2, y2])
                confs.append(d["det_conf"])
            if not boxes_xyxy:
                continue
            boxes_t = torch.tensor(boxes_xyxy, device=device, dtype=torch.float32)
            rois = torch.cat([torch.zeros((boxes_t.shape[0], 1), device=device), boxes_t], dim=1)
            per_layer_vecs = {}
            for name, feat_map in feats.items():
                _, c, fh, fw = feat_map.shape
                pooled = roi_align(feat_map, rois, output_size=7, spatial_scale=fh / h,
                                    sampling_ratio=2, aligned=True)
                per_layer_vecs[name] = pooled.mean(dim=(2, 3)).cpu()
            frame_recs = []
            for i in range(len(boxes_xyxy)):
                rec = {"bbox": boxes_xyxy[i], "det_conf": confs[i]}
                for name in LAYER_ORDER:
                    rec[name] = per_layer_vecs[name][i]
                frame_recs.append(rec)
            records[frame] = frame_recs
        by_frame_cache[seq] = records
        fused_cache[seq] = {f: fuse_features(r, fusion, device) for f, r in records.items()}
        print(f"  {sum(len(v) for v in records.values())} real detections")

    tracker_name = "MSFP-Track-oracle-trainhalf"
    gt_folder, trackers_folder = setup_trackeval_dirs_trainhalf(workdir, mot_root, vo.SEQUENCES, tracker_name, splits)

    hota_grid = {seq: {} for seq in vo.SEQUENCES}
    seq_info = {seq: splits[seq] for seq in vo.SEQUENCES}
    t0 = time.time()
    for tau in GRID:
        for seq in vo.SEQUENCES:
            lines = run_tracker_single_threshold(by_frame_cache[seq], fused_cache[seq], tau, splits[seq])
            out_file = trackers_folder / tracker_name / "data" / f"{seq}.txt"
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_text("\n".join(lines), encoding="utf-8")
        hota_by_seq = vo.evaluate_hota(gt_folder, trackers_folder, vo.SEQUENCES, tracker_name, seq_info)
        for seq in vo.SEQUENCES:
            hota_grid[seq][tau] = hota_by_seq[seq]
        print(f"tau={tau:.2f}: " + ", ".join(f"{s}={hota_by_seq[s]:.1f}" for s in vo.SEQUENCES))
    elapsed = time.time() - t0

    oracle = {}
    for seq in vo.SEQUENCES:
        best_tau = max(hota_grid[seq], key=hota_grid[seq].get)
        oracle[seq] = {"tau_star": best_tau, "hota_at_tau_star": hota_grid[seq][best_tau], "full_grid": hota_grid[seq]}
        print(f"[{seq}] oracle tau*={best_tau:.2f} (HOTA={hota_grid[seq][best_tau]:.2f}, train_half only)")

    out_path = Path("~/msfp_honest_repro/oracle_thresholds_trainhalf.json").expanduser()
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"oracle_per_sequence": oracle, "elapsed_seconds": elapsed}, f, indent=2)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()

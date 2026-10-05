#!/usr/bin/env python3
"""
Properly cross-validated appearance_weight sweep for real_tracker.py's
ByteTrackStyleTracker, avoiding the data-leakage trap of tuning a
hyperparameter directly on the same val_half split used for the final
reported number. MOT17-train's first half (train_half) is split again:
the first 70% of train_half frames are used only as before (for the
already-trained fusion head -- untouched), and the LAST 30% of train_half
(a slice the fusion head has already seen during triplet training, but
that we have NOT used for any threshold/hyperparameter decision so far)
is held out here purely as a tuning-validation set for appearance_weight.
The winning weight is then evaluated, for the first and only time, on the
real held-out val_half (second half of each sequence) -- the same real
protocol used throughout this project. This keeps model-selection and
final-evaluation data strictly separate.
"""
import json
from pathlib import Path

import numpy as np
for _name, _builtin in [("float", float), ("int", int), ("bool", bool), ("object", object)]:
    if not hasattr(np, _name):
        setattr(np, _name, _builtin)
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from real_tracker import ByteTrackStyleTracker
import verify_official_trackers as vo
from baseline_matrix_full_metrics import evaluate_full
from litepp.models.feature_pyramid import FeatureFusionModule

LAYER_ORDER = ["layer4", "layer6", "layer9"]
LAYER_CHANNELS_V8M = {"layer4": 192, "layer6": 384, "layer9": 576}
APPEARANCE_WEIGHTS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


@torch.no_grad()
def fuse_msfp(records, fusion, device):
    if not records:
        return np.zeros((0, 128), dtype=np.float32)
    layer_feats = [torch.stack([r[name] for r in records]).to(device) for name in LAYER_ORDER]
    emb = fusion(layer_feats)
    emb = F.normalize(emb, p=2, dim=1)
    return emb.cpu().numpy()


def run_tracker(by_frame, feat_by_frame, appearance_weight, fs, fe, fo):
    tracker = ByteTrackStyleTracker(tau_h=0.25, tau_l=0.25, max_age=30, min_hits=1,
                                     appearance_weight=appearance_weight, high_stage_cost_thresh=0.7)
    lines = []
    for frame in range(fs, fe + 1):
        recs = by_frame.get(frame, [])
        if recs:
            boxes = np.array([r["bbox"] for r in recs], dtype=np.float32)
            scores = np.array([r["det_conf"] for r in recs], dtype=np.float32)
            feats = feat_by_frame.get(frame, np.zeros((len(recs), 128), dtype=np.float32))
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            scores = np.zeros((0,), dtype=np.float32)
            feats = np.zeros((0, 128), dtype=np.float32)
        for tid, x1, y1, x2, y2 in tracker.update(boxes, scores, feats):
            w, h = x2 - x1, y2 - y1
            lines.append(f"{frame - fo},{tid},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,-1,-1,-1")
    return lines


def setup_trackeval_dirs_windowed(workdir, mot_root, sequences, tracker_name, lo_splits, hi_splits):
    """Like vo.setup_trackeval_dirs, but for a bounded (lo, hi] window rather
    than (split, end-of-sequence] -- needed for the tuning slice, which must
    not run past the real train_half/val_half boundary."""
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
            parts = line.split(",")
            frame = int(parts[0])
            if not (lo_splits[seq] < frame <= hi_splits[seq]):
                continue
            parts[0] = str(frame - lo_splits[seq])
            lines_out.append(",".join(parts))
        (seq_gt_dir / "gt.txt").write_text("\n".join(lines_out), encoding="utf-8")
        win_len = hi_splits[seq] - lo_splits[seq]
        src_ini = (mot_root / seq / "seqinfo.ini").read_text(encoding="utf-8")
        dst_ini = re.sub(r"seqLength=\d+", f"seqLength={win_len}", src_ini)
        (gt_root / seq / "seqinfo.ini").write_text(dst_ini, encoding="utf-8")
    return gt_root, trackers_root


def extract_range(seq_dir, backbone, device, frame_lo, frame_hi):
    """Real public det.txt + RoIAlign features for frames in (frame_lo, frame_hi]."""
    import cv2
    from torchvision.ops import roi_align
    from collections import defaultdict
    by_frame_det = vo.load_det(seq_dir / "det" / "det.txt")
    img_dir = seq_dir / "img1"
    records = defaultdict(list)
    for frame, dets in sorted(by_frame_det.items()):
        if not (frame_lo < frame <= frame_hi):
            continue
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
        for x, y, bw, bh, conf in dets:
            x1, y1, x2, y2 = x, y, x + bw, y + bh
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(float(w0), x2), min(float(h0), y2)
            if x2 <= x1 or y2 <= y1:
                continue
            boxes_xyxy.append([x1, y1, x2, y2])
            confs.append(conf)
        if not boxes_xyxy:
            continue
        boxes_t = torch.tensor(boxes_xyxy, device=device, dtype=torch.float32)
        rois = torch.cat([torch.zeros((boxes_t.shape[0], 1), device=device), boxes_t], dim=1)
        per_layer_vecs = {}
        for name, feat_map in feats.items():
            _, c, fh, fw = feat_map.shape
            per_layer_vecs[name] = roi_align(feat_map, rois, output_size=7, spatial_scale=fh / h,
                                              sampling_ratio=2, aligned=True).mean(dim=(2, 3)).cpu()
        for i in range(len(boxes_xyxy)):
            rec = {"frame": frame, "bbox": boxes_xyxy[i], "det_conf": confs[i]}
            for name in LAYER_ORDER:
                rec[name] = per_layer_vecs[name][i]
            records[frame].append(rec)
    return records


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mot_root = Path("/nas/Dataset/MOT/MOT17/train")
    workdir = Path("~/msfp_honest_repro/trackeval_appweight_cv").expanduser()
    workdir.mkdir(parents=True, exist_ok=True)

    print("Loading real MSFP fusion head...")
    ckpt = torch.load(Path("~/msfp_honest_repro/checkpoints/fusion_attention.pt").expanduser(),
                       map_location=device, weights_only=False)
    layer_channels = [LAYER_CHANNELS_V8M[n] for n in LAYER_ORDER]
    fusion = FeatureFusionModule(layer_channels=layer_channels, output_dim=128,
                                  fusion_type="attention").to(device)
    fusion.load_state_dict(ckpt["state_dict"])
    fusion.eval()

    backbone = vo.HookedBackbone(device)
    splits = {seq: vo.half_split_frame_range(mot_root / seq) for seq in vo.SEQUENCES}
    # Tuning split: last 30% of train_half (frames (0.7*split, split]).
    tune_splits = {seq: int(splits[seq] * 0.7) for seq in vo.SEQUENCES}

    print("=== Extracting TUNING split (last 30% of train_half, held out from val_half) ===")
    tune_cache, tune_feat_cache, tune_seq_info = {}, {}, {}
    for seq in vo.SEQUENCES:
        recs = extract_range(mot_root / seq, backbone, device, tune_splits[seq], splits[seq])
        tune_cache[seq] = recs
        tune_feat_cache[seq] = {f: fuse_msfp(r, fusion, device) for f, r in recs.items()}
        tune_seq_info[seq] = splits[seq] - tune_splits[seq]
        print(f"  [{seq}] {sum(len(v) for v in recs.values())} real detections in tuning range")

    gt_tune, trackers_tune = setup_trackeval_dirs_windowed(workdir, mot_root, vo.SEQUENCES, "_tune",
                                                            tune_splits, splits)

    sweep_results = {}
    for aw in APPEARANCE_WEIGHTS:
        name = f"aw_{aw}"
        (trackers_tune / name / "data").mkdir(parents=True, exist_ok=True)
        for seq in vo.SEQUENCES:
            lines = run_tracker(tune_cache[seq], tune_feat_cache[seq], aw,
                                 tune_splits[seq] + 1, splits[seq], tune_splits[seq])
            (trackers_tune / name / "data" / f"{seq}.txt").write_text("\n".join(lines), encoding="utf-8")
        _, agg = evaluate_full(gt_tune, trackers_tune, vo.SEQUENCES, name, tune_seq_info)
        sweep_results[aw] = agg
        print(f"[tune aw={aw}] HOTA={agg['HOTA']:.2f}")

    best_aw = max(sweep_results, key=lambda k: sweep_results[k]["HOTA"])
    print(f"\nBest appearance_weight on TUNING split: {best_aw} (HOTA={sweep_results[best_aw]['HOTA']:.2f})")

    print("\n=== Extracting real held-out VAL_HALF (never used for tuning) ===")
    val_cache, val_feat_cache, val_seq_info = {}, {}, {}
    for seq in vo.SEQUENCES:
        n_total = len(list((mot_root / seq / "img1").glob("*.jpg")))
        recs, _ = vo.extract_val_half(mot_root / seq, backbone, device, splits[seq])
        val_cache[seq] = recs
        val_feat_cache[seq] = {f: fuse_msfp(r, fusion, device) for f, r in recs.items()}
        val_seq_info[seq] = n_total - splits[seq]

    gt_val, trackers_val = vo.setup_trackeval_dirs(workdir, mot_root, vo.SEQUENCES, "_val", splits)
    (trackers_val / "final" / "data").mkdir(parents=True, exist_ok=True)
    for seq in vo.SEQUENCES:
        n_total = len(list((mot_root / seq / "img1").glob("*.jpg")))
        lines = run_tracker(val_cache[seq], val_feat_cache[seq], best_aw, splits[seq] + 1, n_total, splits[seq])
        (trackers_val / "final" / "data" / f"{seq}.txt").write_text("\n".join(lines), encoding="utf-8")
    _, final_agg = evaluate_full(gt_val, trackers_val, vo.SEQUENCES, "final", val_seq_info)

    print(f"\n[FINAL, best aw={best_aw}, real held-out val_half] HOTA={final_agg['HOTA']:.2f} "
          f"AssA={final_agg['AssA']:.2f} DetA={final_agg['DetA']:.2f} IDF1={final_agg['IDF1']:.2f} "
          f"MOTA={final_agg['MOTA']:.2f} IDSW={final_agg['IDSW']}")
    print("Reference (appearance_weight=0.5, same protocol): HOTA=48.09")

    out = {"tuning_sweep": {str(k): v for k, v in sweep_results.items()}, "best_appearance_weight": best_aw,
           "final_val_half_result": final_agg, "reference_aw05": 48.09}
    out_path = Path("~/msfp_honest_repro/appweight_nested_cv_results.json").expanduser()
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()

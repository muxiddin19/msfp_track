#!/usr/bin/env python3
"""
Standalone, self-contained verification of the main paper's cross-backbone
generalization claim (Table~\\ref{tab:generalization}: YOLOv8s baseline 58.7
-> MSFP-Track 60.5, +1.8 HOTA), reusing the exact same validated real
pipeline as the rest of this release -- real GT-box RoIAlign features, real
P=8/K=4 hard-negative triplet training (train_fusion_head.py's own
hardest_triplet_loss, imported directly, not reimplemented), real public-
detection feature extraction, real Kalman+ByteTrack-style tracker, real
TrackEval HOTA -- with YOLOv8s swapped in for YOLOv8m and its real channel
widths at the same module indices auto-detected rather than hardcoded.

This is a standalone script (does not modify extract_cached_features.py,
extract_det_features.py, train_fusion_head.py, or personpath22_zeroshot_eval.py,
which remain YOLOv8m-specific) to avoid touching already-validated production
scripts this close to the submission deadline; it imports their functions
directly wherever the logic is identical.

Protocol (v2, matching the main paper's own methodology for Table 3, which
evaluates on MOT17-val, not test): all 7 MOT17-train sequences, split by the
standard ByteTrack/FairMOT train_half/val_half convention (first half of each
sequence's frames trains the fusion head, the temporally disjoint second half
is scored), with a single fixed threshold shared by both the baseline and
MSFP conditions -- a controlled ablation isolating the fusion-strategy effect
alone, matching how the main paper's own Table 6 ablation is framed, rather
than independently oracle-tuning each condition's threshold (which an
earlier version of this script did on a 3-sequence subset, and which can
inflate an apparent gap through per-condition threshold-selection noise and
train/eval frame overlap on a small sample).
"""
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
for _name, _builtin in [("float", float), ("int", int), ("bool", bool), ("object", object)]:
    if not hasattr(np, _name):
        setattr(np, _name, _builtin)
import torch
import torch.nn.functional as F
from torchvision.ops import roi_align
from ultralytics import YOLO

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from real_tracker import ByteTrackStyleTracker
from train_fusion_head import PKBatchSampler, index_by_identity, hardest_triplet_loss
from extract_cached_features import load_gt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from litepp.models.feature_pyramid import FeatureFusionModule

LAYER_MODULE_INDEX = {"layer4": 4, "layer6": 6, "layer9": 9}
LAYER_ORDER = ["layer4", "layer6", "layer9"]
SEQUENCES = ["MOT17-02-FRCNN", "MOT17-04-FRCNN", "MOT17-05-FRCNN", "MOT17-09-FRCNN",
             "MOT17-10-FRCNN", "MOT17-11-FRCNN", "MOT17-13-FRCNN"]
# Standard ByteTrack/FairMOT train_half/val_half protocol: the first half of
# each sequence's frames trains the fusion head, the second half (unseen
# frames) is scored, removing the train/eval frame overlap that otherwise
# lets the fusion head's triplet training implicitly "see" the exact frames
# it is later judged on while the untrained single-layer baseline gets no
# such advantage -- a standard source of optimistic bias in small ablations.
FIXED_TAU = 0.25  # single threshold shared by both conditions (Table 6's own
                   # "Fixed tau=0.25" reference point), isolating the fusion
                   # effect alone rather than confounding it with independent
                   # per-condition threshold tuning.


class HookedBackbone:
    def __init__(self, model_name, device):
        self.model = YOLO(model_name).model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device
        self._feats = {}
        self.channels = {}
        for name, idx in LAYER_MODULE_INDEX.items():
            layer = self.model.model[idx]

            def make_hook(n):
                def hook(_m, _i, out):
                    self._feats[n] = out
                return hook

            layer.register_forward_hook(make_hook(name))

    @torch.no_grad()
    def forward(self, img_tensor):
        self._feats.clear()
        self.model(img_tensor)
        if not self.channels:
            for name, feat in self._feats.items():
                self.channels[name] = feat.shape[1]
        return dict(self._feats)


def preprocess_image(path, device, imgsz_multiple=32):
    img = cv2.imread(str(path))
    h0, w0 = img.shape[:2]
    h = ((h0 + imgsz_multiple - 1) // imgsz_multiple) * imgsz_multiple
    w = ((w0 + imgsz_multiple - 1) // imgsz_multiple) * imgsz_multiple
    padded = cv2.copyMakeBorder(img, 0, h - h0, 0, w - w0, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).to(device).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0), (h0, w0), (h, w)


def half_split_frame_range(seq_dir: Path):
    """Standard ByteTrack/FairMOT train_half/val_half split: first half of
    frames (by index) for training, second half (temporally disjoint,
    unseen) for evaluation."""
    n_frames = len(list((seq_dir / "img1").glob("*.jpg")))
    split = n_frames // 2
    return split  # frames 1..split = train_half; split+1..n_frames = val_half


@torch.no_grad()
def extract_gt_features(seq_dir: Path, backbone: HookedBackbone, device, max_frame=None):
    """Same real GT-box extraction as extract_cached_features.py, inlined so
    this script stays backbone-agnostic without editing that file.
    max_frame restricts extraction to the train_half (frame <= max_frame)."""
    by_frame = load_gt(seq_dir / "gt" / "gt.txt")
    img_dir = seq_dir / "img1"
    records = []
    for frame, boxes in by_frame.items():
        if max_frame is not None and frame > max_frame:
            continue
        img_path = img_dir / f"{frame:06d}.jpg"
        if not img_path.exists():
            continue
        img_tensor, (h0, w0), (hp, wp) = preprocess_image(img_path, device)
        feats = backbone.forward(img_tensor)

        boxes_xyxy, meta = [], []
        for ident, x, y, w, h, vis in boxes:
            if w <= 1 or h <= 1:
                continue
            x1, y1, x2, y2 = x, y, x + w, y + h
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(float(w0), x2), min(float(h0), y2)
            if x2 <= x1 or y2 <= y1:
                continue
            boxes_xyxy.append([x1, y1, x2, y2])
            meta.append(ident)
        if not boxes_xyxy:
            continue

        boxes_t = torch.tensor(boxes_xyxy, device=device, dtype=torch.float32)
        rois = torch.cat([torch.zeros((boxes_t.shape[0], 1), device=device), boxes_t], dim=1)
        per_layer_vecs = {}
        for name, feat_map in feats.items():
            _, c, fh, fw = feat_map.shape
            spatial_scale = fh / hp
            pooled = roi_align(feat_map, rois, output_size=7, spatial_scale=spatial_scale,
                                sampling_ratio=2, aligned=True)
            per_layer_vecs[name] = pooled.mean(dim=(2, 3)).cpu()

        for i, ident in enumerate(meta):
            rec = {"seq": seq_dir.name, "frame": frame, "identity": ident, "bbox": boxes_xyxy[i]}
            for name in LAYER_MODULE_INDEX:
                rec[name] = per_layer_vecs[name][i]
            records.append(rec)
    return records


def load_det(det_path):
    by_frame = defaultdict(list)
    with open(det_path, encoding="utf-8") as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) < 7:
                continue
            frame = int(p[0])
            x, y, w, h, conf = float(p[2]), float(p[3]), float(p[4]), float(p[5]), float(p[6])
            by_frame[frame].append((x, y, w, h, conf))
    return by_frame


@torch.no_grad()
def extract_det_features(seq_dir: Path, backbone: HookedBackbone, device, min_frame=None):
    """min_frame restricts extraction to the val_half (frame > min_frame),
    temporally disjoint from the train_half used for fusion-head training."""
    by_frame_det = load_det(seq_dir / "det" / "det.txt")
    img_dir = seq_dir / "img1"
    records = defaultdict(list)
    for frame, dets in by_frame_det.items():
        if min_frame is not None and frame <= min_frame:
            continue
        img_path = img_dir / f"{frame:06d}.jpg"
        if not img_path.exists():
            continue
        img_tensor, (h0, w0), (hp, wp) = preprocess_image(img_path, device)
        feats = backbone.forward(img_tensor)

        boxes_xyxy, confs = [], []
        for x, y, w, h, conf in dets:
            x1, y1, x2, y2 = x, y, x + w, y + h
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
            spatial_scale = fh / hp
            pooled = roi_align(feat_map, rois, output_size=7, spatial_scale=spatial_scale,
                                sampling_ratio=2, aligned=True)
            per_layer_vecs[name] = pooled.mean(dim=(2, 3)).cpu()

        for i in range(len(boxes_xyxy)):
            rec = {"frame": frame, "bbox": boxes_xyxy[i], "det_conf": confs[i]}
            for name in LAYER_MODULE_INDEX:
                rec[name] = per_layer_vecs[name][i]
            records[frame].append(rec)
    return records


def train_real_fusion_head(gt_records, layer_channels, device, order=None,
                            fusion_type="attention", epochs=50, P=8, K=4):
    """Identical procedure to train_fusion_head.py's main(), just inlined to
    accept in-memory records (this script's sequences are small enough that
    writing/reading a .pt cache isn't necessary).

    order/fusion_type let this same real triplet-training procedure also
    produce a fairly-TRAINED single-layer baseline head (order=["layer9"],
    fusion_type="concat", degenerating to a plain learned projection) --
    a raw, never-trained single-layer embedding is not a fair baseline
    against a fully-trained multi-layer fusion head, since the comparison
    would then confound "no training at all" with "fewer layers," inflating
    the apparent fusion benefit on both ends (a weaker untrained baseline
    and a disproportionately strong trained MSFP side)."""
    order = order or LAYER_ORDER
    by_identity = index_by_identity(gt_records)
    sampler = PKBatchSampler(gt_records, by_identity, P=P, K=K, window=30)
    batches_per_epoch = len(sampler)

    fusion = FeatureFusionModule(layer_channels=layer_channels, output_dim=128,
                                  fusion_type=fusion_type).to(device)
    n_params = sum(p.numel() for p in fusion.parameters() if p.requires_grad)
    opt = torch.optim.Adam(fusion.parameters(), lr=1e-4)

    for epoch in range(epochs):
        for _ in range(batches_per_epoch):
            idxs, pids = sampler.sample_batch()
            layer_feats = [torch.stack([gt_records[i][name] for i in idxs]).to(device) for name in order]
            embeddings = fusion(layer_feats)
            loss = hardest_triplet_loss(embeddings, pids, margin=0.3)
            opt.zero_grad()
            loss.backward()
            opt.step()
    fusion.eval()
    return fusion, n_params


@torch.no_grad()
def fuse_features(records, fusion, device, order=None):
    order = order or LAYER_ORDER
    if not records:
        return np.zeros((0, 128), dtype=np.float32)
    layer_feats = [torch.stack([r[name] for r in records]).to(device) for name in order]
    emb = fusion(layer_feats)
    emb = F.normalize(emb, p=2, dim=1)
    return emb.cpu().numpy()


def run_tracker(by_frame, fused_by_frame, tau, frame_start, frame_end, frame_offset):
    """Iterates the val_half's original frame numbers [frame_start, frame_end]
    but writes output with frames renumbered to 1..(frame_end-frame_offset),
    matching the renumbered val_half ground truth (see write_val_half_gt)."""
    tracker = ByteTrackStyleTracker(tau_h=tau, tau_l=tau, max_age=30, min_hits=1,
                                     appearance_weight=0.5, high_stage_cost_thresh=0.7)
    lines = []
    for frame in range(frame_start, frame_end + 1):
        recs = by_frame.get(frame, [])
        if recs:
            boxes = np.array([r["bbox"] for r in recs], dtype=np.float32)
            scores = np.array([r["det_conf"] for r in recs], dtype=np.float32)
            feats = fused_by_frame.get(frame, np.zeros((len(recs), 128), dtype=np.float32))
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            scores = np.zeros((0,), dtype=np.float32)
            feats = np.zeros((0, 128), dtype=np.float32)
        results = tracker.update(boxes, scores, feats)
        out_frame = frame - frame_offset
        for tid, x1, y1, x2, y2 in results:
            w, h = x2 - x1, y2 - y1
            lines.append(f"{out_frame},{tid},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,-1,-1,-1")
    return lines


def write_val_half_gt(src_gt_path: Path, dest_gt_path: Path, split: int):
    """Filters gt.txt to frame > split (the val_half) and renumbers frames
    to 1-indexed, matching run_tracker's out_frame renumbering, so TrackEval
    scores only the held-out half with no spurious FNs from the train_half."""
    lines_out = []
    for line in src_gt_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split(",")
        frame = int(parts[0])
        if frame <= split:
            continue
        parts[0] = str(frame - split)
        lines_out.append(",".join(parts))
    dest_gt_path.write_text("\n".join(lines_out), encoding="utf-8")


def setup_trackeval_dirs(workdir, mot_root, sequences, tracker_name, splits):
    gt_root = workdir / "gt"
    trackers_root = workdir / "trackers"
    (trackers_root / tracker_name / "data").mkdir(parents=True, exist_ok=True)
    for seq in sequences:
        seq_gt_dir = gt_root / seq / "gt"
        seq_gt_dir.mkdir(parents=True, exist_ok=True)
        write_val_half_gt(mot_root / seq / "gt" / "gt.txt", seq_gt_dir / "gt.txt", splits[seq])
        n_total = len(list((mot_root / seq / "img1").glob("*.jpg")))
        val_len = n_total - splits[seq]
        src_ini = (mot_root / seq / "seqinfo.ini").read_text(encoding="utf-8")
        import re
        dst_ini = re.sub(r"seqLength=\d+", f"seqLength={val_len}", src_ini)
        (gt_root / seq / "seqinfo.ini").write_text(dst_ini, encoding="utf-8")
    return gt_root, trackers_root


def evaluate_hota(gt_folder, trackers_folder, sequences, tracker_name, seq_info):
    import trackeval
    eval_config = trackeval.Evaluator.get_default_eval_config()
    for k in ['PRINT_RESULTS', 'PRINT_CONFIG', 'TIME_PROGRESS', 'OUTPUT_SUMMARY', 'OUTPUT_DETAILED', 'PLOT_CURVES']:
        eval_config[k] = False
    eval_config['DISPLAY_LESS_PROGRESS'] = True
    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config.update({'GT_FOLDER': str(gt_folder), 'TRACKERS_FOLDER': str(trackers_folder),
                            'SKIP_SPLIT_FOL': True, 'SEQ_INFO': seq_info,
                            'TRACKERS_TO_EVAL': [tracker_name], 'CLASSES_TO_EVAL': ['pedestrian'],
                            'BENCHMARK': 'MOT17', 'PRINT_CONFIG': False})
    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    metrics_list = [trackeval.metrics.HOTA({'PRINT_CONFIG': False})]
    results, _ = evaluator.evaluate(dataset_list, metrics_list)
    seq_results = results['MotChallenge2DBox'][tracker_name]
    return {seq: float(np.mean(seq_results[seq]['pedestrian']['HOTA']['HOTA'])) * 100.0 for seq in sequences}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mot_root = Path("/nas/Dataset/MOT/MOT17/train")
    workdir = Path("~/msfp/trackeval_yolov8s_verify_v2").expanduser()
    workdir.mkdir(parents=True, exist_ok=True)

    print("Loading real YOLOv8s backbone (COCO-pretrained, frozen) -- swapped in for YOLOv8m...")
    backbone = HookedBackbone("yolov8s.pt", device)

    splits = {seq: half_split_frame_range(mot_root / seq) for seq in SEQUENCES}
    print(f"\nStandard train_half/val_half split (frame counts): {splits}")

    print("\nExtracting real GT-box features from train_half only for real triplet training "
          f"(all {len(SEQUENCES)} MOT17-train sequences)...")
    t0 = time.time()
    gt_records = []
    for seq in SEQUENCES:
        recs = extract_gt_features(mot_root / seq, backbone, device, max_frame=splits[seq])
        gt_records.extend(recs)
        print(f"  {seq}: {len(recs)} real GT boxes (train_half, frames 1-{splits[seq]})")
    layer_channels = [backbone.channels["layer4"], backbone.channels["layer6"], backbone.channels["layer9"]]
    print(f"Real auto-detected YOLOv8s channel widths (layer4/6/9): {layer_channels}")

    print("\nTraining real single-layer baseline head (same triplet loss, layer9 only, "
          "fusion_type='concat' degenerates to a plain learned projection -- a fairly "
          "TRAINED single-layer embedding, matching how LITE itself is actually a "
          "trained projection rather than raw untrained features)...")
    single_layer_fusion, n_params_single = train_real_fusion_head(
        gt_records, [layer_channels[2]], device, order=["layer9"], fusion_type="concat")
    print(f"  {n_params_single:,} trainable params (single-layer baseline)")

    print("\nTraining real MSFP fusion head (same P=8/K=4 hard-negative triplet loss "
          "as train_fusion_head.py, 3-layer instance-adaptive attention)...")
    fusion, n_params = train_real_fusion_head(gt_records, layer_channels, device)
    t_train_and_extract = time.time() - t0
    print(f"  {n_params:,} trainable params (MSFP)")

    print("\nExtracting real public-detection features from val_half only for tracking evaluation "
          "(temporally disjoint from training frames)...")
    by_frame_cache, seq_info = {}, {}
    for seq in SEQUENCES:
        recs = extract_det_features(mot_root / seq, backbone, device, min_frame=splits[seq])
        by_frame_cache[seq] = recs
        n_total = len(list((mot_root / seq / "img1").glob("*.jpg")))
        seq_info[seq] = n_total - splits[seq]
        print(f"  {seq}: val_half frames {splits[seq]+1}-{n_total}, "
              f"{sum(len(v) for v in recs.values())} real public detections")

    fused_cache = {seq: {f: fuse_features(r, fusion, device) for f, r in by_frame_cache[seq].items()}
                   for seq in SEQUENCES}
    single_layer_cache = {seq: {f: fuse_features(r, single_layer_fusion, device, order=["layer9"])
                                 for f, r in by_frame_cache[seq].items()} for seq in SEQUENCES}

    tracker_name = "yolov8s-verify"
    gt_folder, trackers_folder = setup_trackeval_dirs(workdir, mot_root, SEQUENCES, tracker_name, splits)

    def eval_fixed_tau(fused, label):
        """Single shared threshold (FIXED_TAU) across both conditions -- a
        controlled ablation isolating the fusion-strategy effect alone,
        matching the main paper's own Table 6 methodology, rather than
        independently oracle-tuning each condition (which can inflate an
        apparent gap through per-condition threshold-selection noise on a
        small sample)."""
        for seq in SEQUENCES:
            n_total = len(list((mot_root / seq / "img1").glob("*.jpg")))
            lines = run_tracker(by_frame_cache[seq], fused[seq], FIXED_TAU,
                                 splits[seq] + 1, n_total, splits[seq])
            out_file = trackers_folder / tracker_name / "data" / f"{seq}.txt"
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_text("\n".join(lines), encoding="utf-8")
        hota = evaluate_hota(gt_folder, trackers_folder, SEQUENCES, tracker_name, seq_info)
        mean_hota = float(np.mean(list(hota.values())))
        print(f"[{label}] " + ", ".join(f"{s}={hota[s]:.2f}" for s in SEQUENCES) + f"  mean={mean_hota:.2f}")
        return mean_hota, hota

    print(f"\nReal single-layer baseline (layer9 only, val_half, shared fixed tau={FIXED_TAU})...")
    mean_baseline, per_seq_baseline = eval_fixed_tau(single_layer_cache, "Single-layer baseline (real)")

    print(f"\nReal MSFP fusion (3-layer attention head, val_half, same shared fixed tau={FIXED_TAU})...")
    mean_msfp, per_seq_msfp = eval_fixed_tau(fused_cache, "MSFP fusion (real)")

    result = {
        "backbone": "yolov8s.pt",
        "real_channels_layer4_6_9": layer_channels,
        "single_layer_head_params": n_params_single,
        "fusion_head_params": n_params,
        "sequences": SEQUENCES,
        "protocol": "train_half/val_half split (ByteTrack/FairMOT convention), "
                    f"both conditions use a REAL TRAINED head (single-layer baseline "
                    "is a trained learned projection, not raw untrained features), "
                    f"single shared fixed tau={FIXED_TAU} for both conditions",
        "single_layer_baseline_hota": per_seq_baseline,
        "msfp_hota": per_seq_msfp,
        "mean_baseline_hota": mean_baseline,
        "mean_msfp_hota": mean_msfp,
        "delta_hota": mean_msfp - mean_baseline,
        "paper_claimed_delta_hota": 1.8,
        "paper_claimed_baseline": 58.7,
        "paper_claimed_msfp": 60.5,
        "timing": {"total_seconds": t_train_and_extract,
                   "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"},
    }
    out_path = Path("~/msfp/checkpoints/cross_backbone_yolov8s_verify_v3.json").expanduser()
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"\n=== REAL cross-backbone verification v3 (YOLOv8s, {len(SEQUENCES)}-seq, "
          f"train_half/val_half, both heads trained, shared fixed tau) ===")
    print(f"Baseline (single-layer): {mean_baseline:.2f} HOTA")
    print(f"MSFP fusion:             {mean_msfp:.2f} HOTA")
    print(f"Real Delta HOTA:         {result['delta_hota']:+.2f}  "
          f"(paper claims +1.8 on full 7-seq MOT17-val with ATL)")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()

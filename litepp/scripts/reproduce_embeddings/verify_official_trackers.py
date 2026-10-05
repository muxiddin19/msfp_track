#!/usr/bin/env python3
"""
Verify the main paper's Table 4 (cross-tracker generalization) and the
"ByteTrack (motion only)" baseline using REAL, established, independently
maintained tracker implementations from the `boxmot` package (the actively
maintained successor to the `yolo_tracking` library LITE's own README cites
for its ByteTrack/BoT-SORT/OC-SORT integrations) rather than our own
from-scratch real_tracker.py -- this directly addresses the concern that a
from-scratch, untuned reimplementation of ByteTrack-style association does
not reach the same absolute HOTA level as a mature, individually-tuned
reference implementation (see the supplementary reproducibility note).

Two real, controlled comparisons, both using boxmot's native (C++-backed)
BotSort/ByteTrack implementations with their published default
hyperparameters (no tuning by us):

  1. ByteTrack (motion-only, boxmot's real implementation, no appearance
     embeddings at all) -- the real "ByteTrack (motion only)" baseline.
  2. BotSort using boxmot's real motion model, real ECC camera-motion
     compensation, and real Kalman filter, with ONLY the appearance
     embeddings swapped between (a) boxmot's own default ReID model
     (a real, independent, pretrained ReID network) and (b) our
     already-trained MSFP fusion head (the exact checkpoint used
     throughout the main paper's results, fusion_attention.pt, YOLOv8m,
     3-layer instance-adaptive attention) -- injected via boxmot's
     documented `Detections(..., embeddings=...)` API, which bypasses
     internal embedding computation entirely. This isolates "which
     appearance source" while holding the REAL, established BotSort
     association/motion/camera-compensation logic completely fixed,
     directly testing the main text's "BoT-SORT + MSFP (ours, replaces
     SBS-50)" claim with code we did not write ourselves.

Protocol: same train_half/val_half split as verify_cross_backbone.py
(ByteTrack/FairMOT convention) across all 7 MOT17-train sequences, since
the main paper's Table 4 numbers are official MOTChallenge test-server
submissions (not locally reproducible by anyone without server access);
this is a real, same-order-of-magnitude sanity check on real held-out
data, not an attempt to exactly match a hidden-test-set number.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from litepp.models.feature_pyramid import FeatureFusionModule

from boxmot import BotSort, ByteTrack
from boxmot.structures import Boxes, Detections, Frame

LAYER_MODULE_INDEX = {"layer4": 4, "layer6": 6, "layer9": 9}
LAYER_ORDER = ["layer4", "layer6", "layer9"]
LAYER_CHANNELS_V8M = {"layer4": 192, "layer6": 384, "layer9": 576}
SEQUENCES = ["MOT17-02-FRCNN", "MOT17-04-FRCNN", "MOT17-05-FRCNN", "MOT17-09-FRCNN",
             "MOT17-10-FRCNN", "MOT17-11-FRCNN", "MOT17-13-FRCNN"]


def half_split_frame_range(seq_dir: Path):
    n_frames = len(list((seq_dir / "img1").glob("*.jpg")))
    return n_frames // 2


class HookedBackbone:
    def __init__(self, device):
        self.model = YOLO("yolov8m.pt").model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device
        self._feats = {}
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
        return dict(self._feats)


def preprocess_image(path, device, imgsz_multiple=32):
    img = cv2.imread(str(path))
    h0, w0 = img.shape[:2]
    h = ((h0 + imgsz_multiple - 1) // imgsz_multiple) * imgsz_multiple
    w = ((w0 + imgsz_multiple - 1) // imgsz_multiple) * imgsz_multiple
    padded = cv2.copyMakeBorder(img, 0, h - h0, 0, w - w0, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).to(device).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0), img, (h0, w0), (h, w)


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
def extract_val_half(seq_dir: Path, backbone: HookedBackbone, device, min_frame: int):
    """Real RoIAlign features (for MSFP fusion) + real raw BGR frames (for
    boxmot's own ReID/CMC) for every real public detection in the val_half."""
    by_frame_det = load_det(seq_dir / "det" / "det.txt")
    img_dir = seq_dir / "img1"
    records = defaultdict(list)
    frame_images = {}
    for frame, dets in sorted(by_frame_det.items()):
        if frame <= min_frame:
            continue
        img_path = img_dir / f"{frame:06d}.jpg"
        if not img_path.exists():
            continue
        img_tensor, raw_bgr, (h0, w0), (hp, wp) = preprocess_image(img_path, device)
        feats = backbone.forward(img_tensor)
        frame_images[frame] = raw_bgr

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
    return records, frame_images


@torch.no_grad()
def fuse_msfp(records, fusion, device):
    if not records:
        return np.zeros((0, 128), dtype=np.float32)
    layer_feats = [torch.stack([r[name] for r in records]).to(device) for name in LAYER_ORDER]
    emb = fusion(layer_feats)
    emb = F.normalize(emb, p=2, dim=1)
    return emb.cpu().numpy()


def write_val_half_gt(src_gt_path, dest_gt_path, split):
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
    import re
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


def run_real_bytetrack_motion_only(by_frame, frame_images, frame_start, frame_end, frame_offset, sample_id):
    """Real boxmot ByteTrack, no appearance at all -- motion-only baseline."""
    tracker = ByteTrack()
    lines = []
    for frame in range(frame_start, frame_end + 1):
        recs = by_frame.get(frame, [])
        img = frame_images.get(frame)
        if img is None:
            continue
        if recs:
            geometry = torch.tensor([r["bbox"] for r in recs], dtype=torch.float32)
            scores = torch.tensor([r["det_conf"] for r in recs], dtype=torch.float32)
            class_ids = torch.zeros(len(recs), dtype=torch.int64)
        else:
            geometry = torch.zeros((0, 4), dtype=torch.float32)
            scores = torch.zeros((0,), dtype=torch.float32)
            class_ids = torch.zeros((0,), dtype=torch.int64)
        dets = Detections(geometry=Boxes(geometry), scores=scores, class_ids=class_ids, sample_id=sample_id)
        frame_obj = Frame(image=torch.from_numpy(img).permute(2, 0, 1).contiguous(), sample_id=sample_id,
                           frame_index=frame)
        tracks = tracker.update(dets, frame_obj)
        out_frame = frame - frame_offset
        try:
            ids = tracks.track_ids.tolist()
            boxes = tracks.geometry.values.tolist()
        except AttributeError:
            ids, boxes = [], []
        for tid, (x1, y1, x2, y2) in zip(ids, boxes):
            w, h = x2 - x1, y2 - y1
            lines.append(f"{out_frame},{int(tid)},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,-1,-1,-1")
    return lines


def run_real_botsort(by_frame, frame_images, frame_start, frame_end, frame_offset, sample_id,
                      embeddings_by_frame=None, use_embeddings=True, appearance_thresh=0.25,
                      proximity_thresh=0.5):
    """Real boxmot BotSort. If embeddings_by_frame is given, those precomputed
    embeddings are injected via Detections(embeddings=...) and used instead
    of BotSort's own internal ReID model; otherwise BotSort computes its own
    real embeddings from image crops (its default pretrained ReID model).

    appearance_thresh/proximity_thresh default to BotSort's own published
    defaults (tuned for its own ReID embedding distribution, e.g. OSNet);
    exposed here so a different embedding source (MSFP) can be fairly
    recalibrated rather than judged only at a threshold tuned for a
    different embedding space."""
    tracker = BotSort(use_embeddings=use_embeddings, appearance_thresh=appearance_thresh,
                       proximity_thresh=proximity_thresh)
    lines = []
    for frame in range(frame_start, frame_end + 1):
        recs = by_frame.get(frame, [])
        img = frame_images.get(frame)
        if img is None:
            continue
        if recs:
            geometry = torch.tensor([r["bbox"] for r in recs], dtype=torch.float32)
            scores = torch.tensor([r["det_conf"] for r in recs], dtype=torch.float32)
            class_ids = torch.zeros(len(recs), dtype=torch.int64)
        else:
            geometry = torch.zeros((0, 4), dtype=torch.float32)
            scores = torch.zeros((0,), dtype=torch.float32)
            class_ids = torch.zeros((0,), dtype=torch.int64)
        embs = None
        if use_embeddings and embeddings_by_frame is not None:
            emb = embeddings_by_frame.get(frame)
            embs = torch.from_numpy(emb).float() if emb is not None and len(emb) else torch.zeros((len(recs), 128))
        dets = Detections(geometry=Boxes(geometry), scores=scores, class_ids=class_ids,
                           sample_id=sample_id, embeddings=embs)
        frame_obj = Frame(image=torch.from_numpy(img).permute(2, 0, 1).contiguous(), sample_id=sample_id,
                           frame_index=frame)
        tracks = tracker.update(dets, frame_obj)
        out_frame = frame - frame_offset
        try:
            ids = tracks.track_ids.tolist()
            boxes = tracks.geometry.values.tolist()
        except AttributeError:
            ids, boxes = [], []
        for tid, (x1, y1, x2, y2) in zip(ids, boxes):
            w, h = x2 - x1, y2 - y1
            lines.append(f"{out_frame},{int(tid)},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,-1,-1,-1")
    return lines


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mot_root = Path("/nas/Dataset/MOT/MOT17/train")
    workdir = Path("~/msfp/trackeval_official_verify").expanduser()
    workdir.mkdir(parents=True, exist_ok=True)

    print("Loading real, already-trained MSFP fusion head (the exact checkpoint "
          "used throughout the main paper's results: fusion_attention.pt, YOLOv8m, "
          "3-layer instance-adaptive attention)...")
    ckpt = torch.load(Path("~/msfp/checkpoints/fusion_attention.pt").expanduser(),
                       map_location=device, weights_only=False)
    layer_channels = [LAYER_CHANNELS_V8M[n] for n in LAYER_ORDER]
    fusion = FeatureFusionModule(layer_channels=layer_channels, output_dim=128,
                                  fusion_type="attention").to(device)
    fusion.load_state_dict(ckpt["state_dict"])
    fusion.eval()

    backbone = HookedBackbone(device)

    splits = {seq: half_split_frame_range(mot_root / seq) for seq in SEQUENCES}
    print(f"train_half/val_half split (frame counts): {splits}")

    by_frame_cache, frame_images_cache, fused_cache, seq_info = {}, {}, {}, {}
    t0 = time.time()
    for seq in SEQUENCES:
        print(f"[{seq}] extracting real public-detection features (val_half)...")
        recs, imgs = extract_val_half(mot_root / seq, backbone, device, splits[seq])
        by_frame_cache[seq] = recs
        frame_images_cache[seq] = imgs
        fused_cache[seq] = {f: fuse_msfp(r, fusion, device) for f, r in recs.items()}
        n_total = len(list((mot_root / seq / "img1").glob("*.jpg")))
        seq_info[seq] = n_total - splits[seq]
        print(f"  {sum(len(v) for v in recs.values())} real detections, "
              f"val_half frames {splits[seq]+1}-{n_total}")
    t_extract = time.time() - t0

    gt_folder, trackers_folder = setup_trackeval_dirs(workdir, mot_root, SEQUENCES, "verify", splits)

    def eval_condition(run_fn, label):
        for seq in SEQUENCES:
            n_total = len(list((mot_root / seq / "img1").glob("*.jpg")))
            lines = run_fn(seq, splits[seq] + 1, n_total, splits[seq])
            out_file = trackers_folder / "verify" / "data" / f"{seq}.txt"
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_text("\n".join(lines), encoding="utf-8")
        hota = evaluate_hota(gt_folder, trackers_folder, SEQUENCES, "verify", seq_info)
        mean_hota = float(np.mean(list(hota.values())))
        print(f"[{label}] " + ", ".join(f"{s}={hota[s]:.2f}" for s in SEQUENCES) + f"  mean={mean_hota:.2f}")
        return mean_hota, hota

    print("\n=== Real boxmot ByteTrack (motion-only, no appearance) ===")
    mean_bytetrack, per_seq_bytetrack = eval_condition(
        lambda seq, fs, fe, fo: run_real_bytetrack_motion_only(
            by_frame_cache[seq], frame_images_cache[seq], fs, fe, fo, seq),
        "ByteTrack (real, motion-only)")

    print("\n=== Real boxmot BotSort, own default ReID embeddings ===")
    mean_botsort_default, per_seq_botsort_default = eval_condition(
        lambda seq, fs, fe, fo: run_real_botsort(
            by_frame_cache[seq], frame_images_cache[seq], fs, fe, fo, seq,
            embeddings_by_frame=None, use_embeddings=True),
        "BotSort (real, default ReID)")

    print("\n=== Real boxmot BotSort, our trained MSFP embeddings injected ===")
    mean_botsort_msfp, per_seq_botsort_msfp = eval_condition(
        lambda seq, fs, fe, fo: run_real_botsort(
            by_frame_cache[seq], frame_images_cache[seq], fs, fe, fo, seq,
            embeddings_by_frame=fused_cache[seq], use_embeddings=True),
        "BotSort + MSFP (real BotSort, our embeddings)")

    result = {
        "protocol": "train_half/val_half (ByteTrack/FairMOT convention), all 7 MOT17-train sequences, "
                    "real boxmot ByteTrack/BotSort (native C++ backend, published defaults, no tuning by us)",
        "mean_bytetrack_motion_only": mean_bytetrack,
        "per_seq_bytetrack_motion_only": per_seq_bytetrack,
        "mean_botsort_default_reid": mean_botsort_default,
        "per_seq_botsort_default_reid": per_seq_botsort_default,
        "mean_botsort_msfp": mean_botsort_msfp,
        "per_seq_botsort_msfp": per_seq_botsort_msfp,
        "paper_claimed_bytetrack_motion_only": 54.8,
        "paper_claimed_botsort_default_reid_sbs50": 56.3,
        "paper_claimed_botsort_msfp": 57.9,
        "timing": {"extract_seconds": t_extract,
                   "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"},
    }
    out_path = Path("~/msfp/checkpoints/official_tracker_verify.json").expanduser()
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("\n=== SUMMARY (real established boxmot trackers, val_half subset) ===")
    print(f"ByteTrack (motion-only):      {mean_bytetrack:.2f}  (paper: 54.8, official MOT17-test)")
    print(f"BotSort (default ReID):       {mean_botsort_default:.2f}  (paper: 56.3, SBS-50)")
    print(f"BotSort + MSFP (ours):        {mean_botsort_msfp:.2f}  (paper: 57.9)")
    print(f"Real Delta (MSFP vs default ReID): {mean_botsort_msfp - mean_botsort_default:+.2f}  "
          f"(paper: {57.9 - 56.3:+.2f})")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()

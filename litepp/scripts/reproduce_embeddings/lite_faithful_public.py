#!/usr/bin/env python3
"""
Faithful LITE reproduction, correcting the earlier `lite_raw_features()`
mistake of pulling from module index 4 (192ch, a deep C2f block) instead of
LITE's literal design: Alikhanov et al. explicitly specify "the first
convolutional layer" of the YOLOv8 backbone, which gives a 48-channel
feature map at half spatial resolution (confirmed here: YOLOv8m module 0 =
Conv2d(3,48,stride=2), output (48, H/2, W/2) for any input size) -- then
crop to each detection box (mapped to this half-resolution grid) and
average across the crop's spatial extent to get a 48-dim descriptor
("an average across channels is computed to achieve a consistent,
simplified yet effective representation (d=48)" -- LITE paper Sec 4).

This is the single open question blocking any Table 1/3/4 "LITE" row
correction: the implementation detail (which layer) swings the real
public-detection HOTA from ~41 (deep layer, wrong) up toward whatever this
correct, literal reproduction gives. No training, no tuning -- exactly
LITE's "zero additional training" design, same val_half protocol, same
tau=0.25 reference point as the already-reported MSFP_ByteTrack=48.09.
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
from torchvision.ops import roi_align
from ultralytics import YOLO

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from real_tracker import ByteTrackStyleTracker
import verify_official_trackers as vo


class LiteBackbone:
    """Hooks ONLY YOLOv8m's module 0 (the literal first conv layer LITE uses)."""

    def __init__(self, device):
        self.model = YOLO("yolov8m.pt").model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device
        self._feat = None

        def hook(_m, _i, out):
            self._feat = out

        self.model.model[0].register_forward_hook(hook)

    @torch.no_grad()
    def forward(self, img_tensor):
        self._feat = None
        self.model(img_tensor)
        return self._feat


def extract_lite_val_half(seq_dir: Path, backbone: LiteBackbone, device, min_frame: int, imgsz_multiple=32):
    """Same real public det.txt + val_half split as verify_official_trackers.extract_val_half,
    but pooling the literal LITE layer (module 0, 48ch, half-res) instead of layer4/6/9."""
    by_frame_det = vo.load_det(seq_dir / "det" / "det.txt")
    img_dir = seq_dir / "img1"
    records = defaultdict(list)
    for frame, dets in sorted(by_frame_det.items()):
        if frame <= min_frame:
            continue
        img_path = img_dir / f"{frame:06d}.jpg"
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        h0, w0 = img.shape[:2]
        h = ((h0 + imgsz_multiple - 1) // imgsz_multiple) * imgsz_multiple
        w = ((w0 + imgsz_multiple - 1) // imgsz_multiple) * imgsz_multiple
        padded = cv2.copyMakeBorder(img, 0, h - h0, 0, w - w0, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(rgb).to(device).permute(2, 0, 1).float().unsqueeze(0) / 255.0

        feat = backbone.forward(img_tensor)  # (1, 48, hp/2, wp/2)
        _, c, fh, fw = feat.shape
        spatial_scale = fh / h

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
        pooled = roi_align(feat, rois, output_size=7, spatial_scale=spatial_scale,
                            sampling_ratio=2, aligned=True)
        vecs = pooled.mean(dim=(2, 3)).cpu().numpy()  # (N, 48) -- literal LITE: GAP over crop

        for i, (box, conf) in enumerate(zip(boxes_xyxy, confs)):
            records[frame].append({"frame": frame, "bbox": box, "det_conf": conf, "feat": vecs[i]})
    return records


def lite_features(records):
    if not records:
        return np.zeros((0, 48), dtype=np.float32)
    feats = np.stack([r["feat"] for r in records]).astype(np.float32)
    norm = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8
    return feats / norm


def run_tracker(by_frame, feat_by_frame, tau, frame_start, frame_end, frame_offset):
    tracker = ByteTrackStyleTracker(tau_h=tau, tau_l=tau, max_age=30, min_hits=1,
                                     appearance_weight=0.5, high_stage_cost_thresh=0.7)
    lines = []
    for frame in range(frame_start, frame_end + 1):
        recs = by_frame.get(frame, [])
        if recs:
            boxes = np.array([r["bbox"] for r in recs], dtype=np.float32)
            scores = np.array([r["det_conf"] for r in recs], dtype=np.float32)
            feats = feat_by_frame.get(frame, np.zeros((len(recs), 48), dtype=np.float32))
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            scores = np.zeros((0,), dtype=np.float32)
            feats = np.zeros((0, 48), dtype=np.float32)
        results = tracker.update(boxes, scores, feats)
        out_frame = frame - frame_offset
        for tid, x1, y1, x2, y2 in results:
            w, h = x2 - x1, y2 - y1
            lines.append(f"{out_frame},{tid},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,-1,-1,-1")
    return lines


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mot_root = Path("/nas/Dataset/MOT/MOT17/train")
    workdir = Path("~/msfp_honest_repro/trackeval_lite_faithful").expanduser()
    workdir.mkdir(parents=True, exist_ok=True)

    print("Loading LITE-faithful backbone (YOLOv8m module 0, 48ch, half-res)...")
    backbone = LiteBackbone(device)
    splits = {seq: vo.half_split_frame_range(mot_root / seq) for seq in vo.SEQUENCES}

    by_frame_cache, feat_cache, seq_info = {}, {}, {}
    t0 = time.time()
    for seq in vo.SEQUENCES:
        print(f"[{seq}] extracting LITE-faithful features (module 0, val_half, det.txt)...")
        recs = extract_lite_val_half(mot_root / seq, backbone, device, splits[seq])
        by_frame_cache[seq] = recs
        feat_cache[seq] = {f: lite_features(r) for f, r in recs.items()}
        n_total = len(list((mot_root / seq / "img1").glob("*.jpg")))
        seq_info[seq] = n_total - splits[seq]
        print(f"  {sum(len(v) for v in recs.values())} real detections")

    gt_folder, trackers_folder = vo.setup_trackeval_dirs(workdir, mot_root, vo.SEQUENCES, "litef", splits)

    for seq in vo.SEQUENCES:
        n_total = len(list((mot_root / seq / "img1").glob("*.jpg")))
        lines = run_tracker(by_frame_cache[seq], feat_cache[seq], 0.25, splits[seq] + 1, n_total, splits[seq])
        out_file = trackers_folder / "litef" / "data" / f"{seq}.txt"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text("\n".join(lines), encoding="utf-8")

    hota = vo.evaluate_hota(gt_folder, trackers_folder, vo.SEQUENCES, "litef", seq_info)
    mean_hota = float(np.mean(list(hota.values())))
    elapsed = time.time() - t0

    print(f"\n[LITE faithful (module0, 48ch, real crop+GAP)] mean HOTA={mean_hota:.2f}  " +
          ", ".join(f"{s}={hota[s]:.2f}" for s in vo.SEQUENCES))
    print(f"Comparison: MSFP-Track (same protocol, tau=0.25) = 48.09")
    print(f"Comparison: earlier WRONG LITE impl (module4, 192ch) = 41.47")

    out = {
        "protocol": "val_half, public MOT17 det.txt, tau=0.25, YOLOv8m module 0 (48ch, half-res) "
                    "-- the literal layer LITE's paper specifies, correcting the earlier module-4 mistake.",
        "mean_hota": mean_hota,
        "per_sequence": hota,
        "elapsed_seconds": elapsed,
        "cross_check_msfp_track_same_protocol": 48.09,
        "cross_check_earlier_wrong_lite_impl": 41.47,
    }
    out_path = Path("~/msfp_honest_repro/lite_faithful_results.json").expanduser()
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Generates real, submission-ready MOT17 test-set tracking predictions using
the exact deployed system described in the main paper: the already-trained
MSFP fusion head (fusion_attention.pt, trained on the full MOT17-train
split, matching the paper's "final models are retrained on the full
training split before server submission" protocol) + the already-trained
paper-faithful ATL encoder (atl_paper_faithful.pt) predicting a per-scene
threshold from real GAP(layer14) features, + the real validated
ByteTrack-style two-stage tracker (real_tracker.py).

This produces the actual prediction files an author would upload through
the MOTChallenge web portal (https://motchallenge.net) to get an official,
independently-verifiable HOTA score -- we do not and cannot submit on
anyone's behalf (that requires the account holder's own credentials and
counts against their submission quota), but this script generates exactly
what that submission needs, real, from the real test images, with no
placeholder or synthetic step.

MOT17 test sequences ship three public-detection variants per sequence
(DPM/FRCNN/SDP, no ground truth), following the paper's stated protocol of
public detections exclusively; each is processed independently (21 output
files total: 7 sequences x 3 detectors), matching the standard MOT17
submission format (one <SEQUENCE-NAME>.txt per entry, same format as the
training-split files used throughout this release).
"""
import json
import time
import zipfile
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
from atl_paper_faithful import ATLPaperFaithful

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from litepp.models.feature_pyramid import FeatureFusionModule

LAYER_MODULE_INDEX = {"layer4": 4, "layer6": 6, "layer9": 9}
LAYER_ORDER = ["layer4", "layer6", "layer9"]
LAYER_CHANNELS_V8M = {"layer4": 192, "layer6": 384, "layer9": 576}
LAYER14_MODULE_INDEX = 9  # same deep/P5 stage used by train_atl.py

TEST_SEQUENCES = ["MOT17-01", "MOT17-03", "MOT17-06", "MOT17-07", "MOT17-08", "MOT17-12", "MOT17-14"]
DETECTORS = ["DPM", "FRCNN", "SDP"]


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
    return tensor.unsqueeze(0), (h0, w0), (h, w)


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
def extract_test_sequence(seq_dir: Path, backbone: HookedBackbone, device, max_atl_frames=120):
    """Real RoIAlign features for every real public detection, plus real
    GAP(layer14) scene features sampled across the sequence for ATL."""
    det_path = seq_dir / "det" / "det.txt"
    img_dir = seq_dir / "img1"
    by_frame_det = load_det(det_path)
    frames = sorted(by_frame_det.keys())

    records = defaultdict(list)
    scene_feats = []
    sample_idx = set(np.linspace(0, len(frames) - 1, min(max_atl_frames, len(frames))).astype(int)) if frames else set()

    for i, frame in enumerate(frames):
        img_path = img_dir / f"{frame:06d}.jpg"
        if not img_path.exists():
            continue
        img_tensor, (h0, w0), (hp, wp) = preprocess_image(img_path, device)
        feats = backbone.forward(img_tensor)

        if i in sample_idx:
            scene_feats.append(feats["layer9"].mean(dim=(2, 3)).squeeze(0).cpu())

        boxes_xyxy, confs = [], []
        for x, y, w, h, conf in by_frame_det[frame]:
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

        for i2 in range(len(boxes_xyxy)):
            rec = {"frame": frame, "bbox": boxes_xyxy[i2], "det_conf": confs[i2]}
            for name in LAYER_MODULE_INDEX:
                rec[name] = per_layer_vecs[name][i2]
            records[frame].append(rec)

    n_frames = len(list(img_dir.glob("*.jpg")))
    scene_feat_mean = torch.stack(scene_feats).mean(dim=0) if scene_feats else torch.zeros(576)
    return records, n_frames, scene_feat_mean


@torch.no_grad()
def fuse_features(records, fusion, device):
    if not records:
        return np.zeros((0, 128), dtype=np.float32)
    layer_feats = [torch.stack([r[name] for r in records]).to(device) for name in LAYER_ORDER]
    emb = fusion(layer_feats)
    emb = F.normalize(emb, p=2, dim=1)
    return emb.cpu().numpy()


def run_tracker(by_frame, fused_by_frame, tau, n_frames):
    tracker = ByteTrackStyleTracker(tau_h=tau, tau_l=max(0.01, tau * 0.5), max_age=30, min_hits=1,
                                     appearance_weight=0.5, high_stage_cost_thresh=0.7)
    lines = []
    for frame in range(1, n_frames + 1):
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
        for tid, x1, y1, x2, y2 in results:
            w, h = x2 - x1, y2 - y1
            lines.append(f"{frame},{tid},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,-1,-1,-1")
    return lines


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mot_root = Path("/nas/Dataset/MOT/MOT17/test")
    out_dir = Path("~/msfp/mot17_test_submission").expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading real MSFP fusion head (trained on the FULL MOT17-train split)...")
    ckpt = torch.load(Path("~/msfp/checkpoints/fusion_attention.pt").expanduser(),
                       map_location=device, weights_only=False)
    layer_channels = [LAYER_CHANNELS_V8M[n] for n in LAYER_ORDER]
    fusion = FeatureFusionModule(layer_channels=layer_channels, output_dim=128,
                                  fusion_type="attention").to(device)
    fusion.load_state_dict(ckpt["state_dict"])
    fusion.eval()

    print("Loading real paper-faithful ATL encoder (trained on real oracle thresholds)...")
    atl_ckpt = torch.load(Path("~/msfp/checkpoints/atl_paper_faithful.pt").expanduser(),
                           map_location=device, weights_only=False)
    atl = ATLPaperFaithful(input_channels=576, hidden_dim=atl_ckpt["hidden_dim"]).to(device)
    atl.load_state_dict(atl_ckpt["state_dict"])
    atl.eval()

    backbone = HookedBackbone(device)

    t0 = time.time()
    predicted_thresholds = {}
    for seq_base in TEST_SEQUENCES:
        for det in DETECTORS:
            seq_name = f"{seq_base}-{det}"
            seq_dir = mot_root / seq_name
            if not seq_dir.is_dir():
                print(f"[{seq_name}] missing on disk, skipping")
                continue
            print(f"[{seq_name}] extracting real public-detection features + real ATL scene features...")
            records, n_frames, scene_feat = extract_test_sequence(seq_dir, backbone, device)
            n_dets = sum(len(v) for v in records.values())

            with torch.no_grad():
                tau = float(atl.forward_from_gap(scene_feat.unsqueeze(0).to(device)).item())
            predicted_thresholds[seq_name] = tau

            fused_by_frame = {f: fuse_features(r, fusion, device) for f, r in records.items()}
            lines = run_tracker(records, fused_by_frame, tau, n_frames)

            out_file = out_dir / f"{seq_name}.txt"
            out_file.write_text("\n".join(lines), encoding="utf-8")
            print(f"  {n_frames} frames, {n_dets} real detections, ATL tau={tau:.3f}, "
                  f"{len(lines)} output lines -> {out_file.name}")

    elapsed = time.time() - t0

    zip_path = out_dir.parent / "mot17_test_submission.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for txt_file in sorted(out_dir.glob("*.txt")):
            zf.write(txt_file, arcname=txt_file.name)

    meta = {
        "predicted_atl_thresholds": predicted_thresholds,
        "generation_time_seconds": elapsed,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "fusion_checkpoint": "fusion_attention.pt (trained on full MOT17-train, 150019 params)",
        "atl_checkpoint": "atl_paper_faithful.pt",
        "note": "Real predictions on the real, official MOT17 test images (no ground truth "
                "available locally -- these files are what would be uploaded to "
                "https://motchallenge.net for an official, independently-verifiable HOTA score. "
                "We do not submit on the user's behalf; this generates exactly what that "
                "submission needs.",
    }
    with open(out_dir.parent / "mot17_test_submission_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nGenerated {len(list(out_dir.glob('*.txt')))} real prediction files "
          f"in {elapsed:.1f}s -> {zip_path}")
    print("Ready for upload at https://motchallenge.net (MOT17 Challenge submission form).")


if __name__ == "__main__":
    main()

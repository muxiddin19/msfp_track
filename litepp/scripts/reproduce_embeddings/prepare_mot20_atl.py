#!/usr/bin/env python3
"""
Prepares the MOT20-specific deployment per the main paper's own stated
protocol (Sec.~3.4/Training Protocol, Training/Validation Splits
paragraph): "For MOT20, ATL is retrained separately on MOT20-train
(analogous to fitting a new ReID model per dataset)" -- i.e. the MSFP
fusion head is REUSED from MOT17 (not retrained), but a fresh ATL encoder
is fit to MOT20's own real oracle thresholds.

Steps (all real, no synthetic data):
  1. Real public-detection feature extraction (det.txt) + real GAP(layer14)
     scene features for MOT20-train (4 sequences: 01, 02, 03, 05), using
     the REUSED MOT17-trained MSFP fusion head.
  2. Real oracle-threshold grid search (same GRID, same real_tracker.py,
     same real TrackEval HOTA as run_oracle_grid_search.py) on MOT20-train.
  3. Real ATLPaperFaithful training against these MOT20-specific oracle
     targets -> atl_paper_faithful_mot20.pt.

Note: MOT20's public detections use a binary (0/1) confidence convention
(confirmed by inspection), unlike MOT17-FRCNN's continuous scores, so the
oracle grid search's threshold sweep behaves more like a hard on/off
filter here than a graduated one -- this is a real, disclosed property of
the MOT20 public-detection release, not an artifact of our pipeline.
"""
import json
import time
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from real_tracker import ByteTrackStyleTracker
from atl_paper_faithful import ATLPaperFaithful
import verify_official_trackers as vo
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from litepp.models.feature_pyramid import FeatureFusionModule

LAYER_MODULE_INDEX = {"layer4": 4, "layer6": 6, "layer9": 9}
LAYER_ORDER = ["layer4", "layer6", "layer9"]
LAYER_CHANNELS_V8M = {"layer4": 192, "layer6": 384, "layer9": 576}
GRID = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
MOT20_TRAIN_SEQUENCES = ["MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05"]


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
    from collections import defaultdict
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
def extract_sequence(seq_dir: Path, backbone: HookedBackbone, device, max_atl_frames=120):
    det_path = seq_dir / "det" / "det.txt"
    img_dir = seq_dir / "img1"
    by_frame_det = load_det(det_path)
    frames = sorted(by_frame_det.keys())

    from collections import defaultdict
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


def run_tracker_single_threshold(by_frame, fused_by_frame, tau, n_frames):
    tracker = ByteTrackStyleTracker(tau_h=tau, tau_l=tau, max_age=30, min_hits=1,
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
    mot_root = Path("/nas/Dataset/MOT/MOT20/train")
    workdir = Path("~/msfp/trackeval_mot20_oracle").expanduser()
    workdir.mkdir(parents=True, exist_ok=True)

    print("Loading REUSED MOT17-trained MSFP fusion head (not retrained for MOT20, per paper protocol)...")
    ckpt = torch.load(Path("~/msfp/checkpoints/fusion_attention.pt").expanduser(),
                       map_location=device, weights_only=False)
    layer_channels = [LAYER_CHANNELS_V8M[n] for n in LAYER_ORDER]
    fusion = FeatureFusionModule(layer_channels=layer_channels, output_dim=128,
                                  fusion_type="attention").to(device)
    fusion.load_state_dict(ckpt["state_dict"])
    fusion.eval()

    backbone = HookedBackbone(device)

    print("\nExtracting real MOT20-train public-detection features + real ATL scene features...")
    t0 = time.time()
    by_frame_cache, fused_cache, seq_info, scene_feats_by_seq = {}, {}, {}, {}
    for seq in MOT20_TRAIN_SEQUENCES:
        records, n_frames, scene_feat = extract_sequence(mot_root / seq, backbone, device)
        by_frame_cache[seq] = records
        fused_cache[seq] = {f: fuse_features(r, fusion, device) for f, r in records.items()}
        seq_info[seq] = n_frames
        scene_feats_by_seq[seq] = scene_feat
        print(f"  {seq}: {n_frames} frames, {sum(len(v) for v in records.values())} real detections")

    tracker_name = "MSFP-Track-mot20-oracle"
    gt_folder, trackers_folder = vo.setup_trackeval_dirs(workdir, mot_root, MOT20_TRAIN_SEQUENCES,
                                                          tracker_name, {s: 0 for s in MOT20_TRAIN_SEQUENCES})

    print("\nRunning real oracle-threshold grid search on MOT20-train...")
    hota_grid = {seq: {} for seq in MOT20_TRAIN_SEQUENCES}
    for tau in GRID:
        for seq in MOT20_TRAIN_SEQUENCES:
            lines = run_tracker_single_threshold(by_frame_cache[seq], fused_cache[seq], tau, seq_info[seq])
            out_file = trackers_folder / tracker_name / "data" / f"{seq}.txt"
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_text("\n".join(lines), encoding="utf-8")
        hota_by_seq = vo.evaluate_hota(gt_folder, trackers_folder, MOT20_TRAIN_SEQUENCES, tracker_name, seq_info)
        for seq in MOT20_TRAIN_SEQUENCES:
            hota_grid[seq][tau] = hota_by_seq[seq]
        print(f"tau={tau:.2f}: " + ", ".join(f"{s}={hota_by_seq[s]:.1f}" for s in MOT20_TRAIN_SEQUENCES))

    oracle = {}
    for seq in MOT20_TRAIN_SEQUENCES:
        best_tau = max(hota_grid[seq], key=hota_grid[seq].get)
        oracle[seq] = {"tau_star": best_tau, "hota_at_tau_star": hota_grid[seq][best_tau]}
        print(f"[{seq}] oracle tau*={best_tau:.2f} (HOTA={hota_grid[seq][best_tau]:.2f})")

    oracle_path = Path("~/msfp/checkpoints/oracle_thresholds_mot20.json").expanduser()
    with open(oracle_path, "w", encoding="utf-8") as f:
        json.dump({"oracle_per_sequence": oracle}, f, indent=2)
    print(f"Saved real MOT20 oracle thresholds -> {oracle_path}")

    print("\nTraining real paper-faithful ATL on real MOT20 oracle thresholds...")
    X = torch.stack([scene_feats_by_seq[seq] for seq in MOT20_TRAIN_SEQUENCES]).to(device)
    y = torch.tensor([oracle[seq]["tau_star"] for seq in MOT20_TRAIN_SEQUENCES], dtype=torch.float32, device=device)

    atl = ATLPaperFaithful(input_channels=576, hidden_dim=64).to(device)
    n_params = sum(p.numel() for p in atl.parameters() if p.requires_grad)
    opt = torch.optim.Adam(atl.parameters(), lr=1e-3)
    for epoch in range(200):
        pred = atl.forward_from_gap(X)
        loss = F.mse_loss(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/200 - MSE: {loss.item():.6f}")

    with torch.no_grad():
        final_pred = atl.forward_from_gap(X)
    print("\nFinal MOT20 ATL predictions vs real oracle targets:")
    for seq, p, t in zip(MOT20_TRAIN_SEQUENCES, final_pred.cpu().tolist(), y.cpu().tolist()):
        print(f"  {seq}: predicted={p:.3f}, oracle={t:.3f}")

    out_path = Path("~/msfp/checkpoints/atl_paper_faithful_mot20.pt").expanduser()
    torch.save({"state_dict": atl.state_dict(), "hidden_dim": 64}, out_path)
    elapsed = time.time() - t0
    print(f"\nSaved MOT20-specific ATL ({n_params:,} params) -> {out_path}")
    print(f"Total real time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()

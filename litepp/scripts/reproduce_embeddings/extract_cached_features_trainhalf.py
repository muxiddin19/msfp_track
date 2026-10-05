#!/usr/bin/env python3
"""
Same real RoIAlign feature caching as extract_cached_features.py, but
restricted to train_half frames only (frame <= half_split_frame_range),
matching the documented protocol ("fusion heads are trained only on each
sequence's first half of frames") that the existing deployed
fusion_attention.pt checkpoint does NOT actually follow -- its cache
(~/msfp/cache/*.pt) spans the full sequence (verified: MOT17-02-FRCNN
cache has frames 1-600, i.e. all of it, not just the first 300).
Training a fusion head on this full-sequence cache then evaluating "on
held-out val_half" is not a clean held-out test: the same identities'
appearance from val_half frames were available during triplet training.
This script produces the cache that the documented protocol actually
requires, so the fusion head can be honestly retrained leak-free.
"""
import time
from pathlib import Path

import cv2
import torch
from torchvision.ops import roi_align
from ultralytics import YOLO

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import verify_official_trackers as vo

LAYER_CHANNELS_V8M = {"layer4": 192, "layer6": 384, "layer9": 576}
LAYER_MODULE_INDEX = {"layer4": 4, "layer6": 6, "layer9": 9}


def load_gt(gt_path, frame_hi):
    by_frame = {}
    with open(gt_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 9:
                continue
            frame, ident, x, y, w, h, conf, cls, vis = parts[:9]
            frame, ident, cls = int(frame), int(ident), int(cls)
            conf, vis = float(conf), float(vis)
            if frame > frame_hi or cls != 1 or conf == 0:
                continue
            x, y, w, h = float(x), float(y), float(w), float(h)
            by_frame.setdefault(frame, []).append((ident, x, y, w, h, vis))
    return by_frame


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


@torch.no_grad()
def extract_sequence(seq_dir: Path, backbone: HookedBackbone, device, frame_hi: int, out_path: Path):
    gt_path = seq_dir / "gt" / "gt.txt"
    img_dir = seq_dir / "img1"
    by_frame = load_gt(gt_path, frame_hi)

    records = []
    frames = sorted(by_frame.keys())
    t0 = time.time()
    for frame in frames:
        img_path = img_dir / f"{frame:06d}.jpg"
        if not img_path.exists():
            continue
        img_tensor, (h0, w0), (hp, wp) = preprocess_image(img_path, device)
        feats = backbone.forward(img_tensor)

        boxes_xyxy, meta = [], []
        for ident, x, y, w, h, vis in by_frame[frame]:
            if w <= 1 or h <= 1:
                continue
            x1, y1, x2, y2 = x, y, x + w, y + h
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(float(w0), x2), min(float(h0), y2)
            if x2 <= x1 or y2 <= y1:
                continue
            boxes_xyxy.append([x1, y1, x2, y2])
            meta.append((ident, vis))
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

        for i, (ident, vis) in enumerate(meta):
            rec = {"seq": seq_dir.name, "frame": frame, "identity": ident, "bbox": boxes_xyxy[i], "vis": vis}
            for name in LAYER_MODULE_INDEX:
                rec[name] = per_layer_vecs[name][i]
            records.append(rec)

    torch.save(records, out_path)
    print(f"  {len(records)} real boxes, {len(frames)} train_half frames (<= {frame_hi}), "
          f"{time.time() - t0:.1f}s -> {out_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mot_root = Path("/nas/Dataset/MOT/MOT17/train")
    out_dir = Path("~/msfp_honest_repro/cache_trainhalf").expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    backbone = HookedBackbone(device)
    for seq in vo.SEQUENCES:
        split = vo.half_split_frame_range(mot_root / seq)
        print(f"[{seq}] train_half = frames 1-{split} only...")
        extract_sequence(mot_root / seq, backbone, device, split, out_dir / f"{seq}.pt")
    print(f"Done -> {out_dir}")


if __name__ == "__main__":
    main()

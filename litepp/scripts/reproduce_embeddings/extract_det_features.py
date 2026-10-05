#!/usr/bin/env python3
"""
Extract and cache real multi-scale RoI features for every PUBLIC DETECTION
(not GT box) in the specified MOT17 sequences, using the frozen YOLOv8m
backbone. This matches the paper's actual Detection Protocol ("We use
public detections exclusively for track initialization and association"),
which extract_cached_features.py (GT-box features, used for the ReID
embedding-quality figures) does not cover.

Output: one .pt file per sequence under --out_dir, each a list of dicts:
{seq, frame, bbox (x1,y1,x2,y2), det_conf, feat4, feat6, feat9}.
"""
import argparse
import time
from pathlib import Path

import cv2
import torch
from torchvision.ops import roi_align
from ultralytics import YOLO

LAYER_MODULE_INDEX = {"layer4": 4, "layer6": 6, "layer9": 9}


def load_det(det_path):
    by_frame = {}
    with open(det_path, encoding="utf-8") as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) < 7:
                continue
            frame = int(p[0])
            x, y, w, h, conf = float(p[2]), float(p[3]), float(p[4]), float(p[5]), float(p[6])
            by_frame.setdefault(frame, []).append((x, y, w, h, conf))
    return by_frame


class HookedBackbone:
    def __init__(self, device):
        self.model = YOLO("yolov8m.pt").model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device
        self._feats = {}
        self._hooks = []
        for name, idx in LAYER_MODULE_INDEX.items():
            layer = self.model.model[idx]

            def make_hook(n):
                def hook(_m, _i, out):
                    self._feats[n] = out
                return hook

            self._hooks.append(layer.register_forward_hook(make_hook(name)))

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
def extract_sequence(seq_dir: Path, backbone: HookedBackbone, device, out_path: Path):
    det_path = seq_dir / "det" / "det.txt"
    img_dir = seq_dir / "img1"
    by_frame = load_det(det_path)

    records = []
    frames = sorted(by_frame.keys())
    t0 = time.time()
    for frame in frames:
        img_path = img_dir / f"{frame:06d}.jpg"
        if not img_path.exists():
            continue
        img_tensor, (h0, w0), (hp, wp) = preprocess_image(img_path, device)
        feats = backbone.forward(img_tensor)

        boxes_xyxy, confs = [], []
        for x, y, w, h, conf in by_frame[frame]:
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
        batch_idx = torch.zeros((boxes_t.shape[0], 1), device=device)
        rois = torch.cat([batch_idx, boxes_t], dim=1)

        per_layer_vecs = {}
        for name, feat_map in feats.items():
            _, c, fh, fw = feat_map.shape
            spatial_scale = fh / hp
            pooled = roi_align(feat_map, rois, output_size=7, spatial_scale=spatial_scale,
                                sampling_ratio=2, aligned=True)
            per_layer_vecs[name] = pooled.mean(dim=(2, 3)).cpu()

        for i in range(len(boxes_xyxy)):
            rec = {"seq": seq_dir.name, "frame": frame, "bbox": boxes_xyxy[i], "det_conf": confs[i]}
            for name in LAYER_MODULE_INDEX:
                rec[name] = per_layer_vecs[name][i]
            records.append(rec)

    dt = time.time() - t0
    torch.save(records, out_path)
    print(f"[{seq_dir.name}] {len(frames)} frames, {len(records)} detections, "
          f"{dt:.1f}s -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mot_root", default="/nas/Dataset/MOT/MOT17/train")
    ap.add_argument("--sequences", nargs="+",
                     default=["MOT17-02-FRCNN", "MOT17-04-FRCNN", "MOT17-05-FRCNN",
                              "MOT17-09-FRCNN", "MOT17-10-FRCNN", "MOT17-11-FRCNN",
                              "MOT17-13-FRCNN"])
    ap.add_argument("--out_dir", default="~/msfp/cache_det")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    backbone = HookedBackbone(device)
    mot_root = Path(args.mot_root)

    for seq in args.sequences:
        out_path = out_dir / f"{seq}.pt"
        if out_path.exists():
            print(f"[{seq}] already cached, skipping")
            continue
        extract_sequence(mot_root / seq, backbone, device, out_path)


if __name__ == "__main__":
    main()

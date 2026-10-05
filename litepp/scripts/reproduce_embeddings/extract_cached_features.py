#!/usr/bin/env python3
"""
Stage 1: extract and cache raw multi-scale RoI features for every GT box in
the specified MOT17 sequences, using a frozen, COCO-pretrained YOLOv8m
backbone. This is a REAL computation (no synthetic/simulated data) -- it
loads real JPEG frames, runs a real forward pass, and pools real feature
maps with torchvision's real roi_align (bilinear interpolation, matching
paper Eq. 1: GAP(RoIAlign(F^(l), b_i, k)), k=7).

Caching this once means the training loop (stage 2) and the embedding
extraction for figures (stage 3) never need to re-run the backbone, so the
reported "GPU-hours to train the fusion head" cleanly excludes one-time
frozen-feature extraction, exactly as the paper claims (backbone frozen,
only the fusion head is trained).

Output: a single .pt file per sequence under --out_dir, each containing a
list of dicts: {seq, frame, identity, bbox, feat4, feat6, feat9} where
feat* are (C,) GAP-pooled RoIAlign features for that box at that layer.
"""
import argparse
import time
from pathlib import Path

import cv2
import torch
from torchvision.ops import roi_align
from ultralytics import YOLO


LAYER_CHANNELS_V8M = {"layer4": 192, "layer6": 384, "layer9": 576}
LAYER_MODULE_INDEX = {"layer4": 4, "layer6": 6, "layer9": 9}


def load_gt(gt_path):
    """Parse a MOTChallenge gt.txt into {frame: [(id, x, y, w, h), ...]}."""
    by_frame = {}
    with open(gt_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 9:
                continue
            frame, ident, x, y, w, h, conf, cls, vis = parts[:9]
            frame, ident, cls = int(frame), int(ident), int(cls)
            conf, vis = float(conf), float(vis)
            # MOT17 class 1 = pedestrian; conf/consider flags per devkit convention
            if cls != 1 or conf == 0:
                continue
            x, y, w, h = float(x), float(y), float(w), float(h)
            by_frame.setdefault(frame, []).append((ident, x, y, w, h, vis))
    return by_frame


class HookedBackbone:
    """Registers forward hooks on a frozen YOLOv8m to capture layer4/6/9."""

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
                def hook(_module, _inp, out):
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
    # Pad to a multiple of 32 (YOLOv8 stride requirement) without resizing,
    # so pixel->feature-map coordinate mapping stays exact for RoIAlign.
    h = ((h0 + imgsz_multiple - 1) // imgsz_multiple) * imgsz_multiple
    w = ((w0 + imgsz_multiple - 1) // imgsz_multiple) * imgsz_multiple
    padded = cv2.copyMakeBorder(img, 0, h - h0, 0, w - w0, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).to(device).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0), (h0, w0), (h, w)


@torch.no_grad()
def extract_sequence(seq_dir: Path, backbone: HookedBackbone, device, out_path: Path):
    gt_path = seq_dir / "gt" / "gt.txt"
    img_dir = seq_dir / "img1"
    by_frame = load_gt(gt_path)

    records = []
    frames = sorted(by_frame.keys())
    t0 = time.time()
    for frame in frames:
        img_path = img_dir / f"{frame:06d}.jpg"
        if not img_path.exists():
            continue
        img_tensor, (h0, w0), (hp, wp) = preprocess_image(img_path, device)
        feats = backbone.forward(img_tensor)

        boxes_xyxy = []
        meta = []
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
        # roi_align expects a list-of-boxes-per-image or (batch_idx, x1,y1,x2,y2)
        batch_idx = torch.zeros((boxes_t.shape[0], 1), device=device)
        rois = torch.cat([batch_idx, boxes_t], dim=1)

        per_layer_vecs = {}
        for name, feat_map in feats.items():
            _, c, fh, fw = feat_map.shape
            # spatial_scale maps INPUT-pixel coords -> this feature map's grid
            spatial_scale = fh / hp  # == fw / wp for YOLOv8's square-stride design
            pooled = roi_align(
                feat_map, rois, output_size=7, spatial_scale=spatial_scale,
                sampling_ratio=2, aligned=True,
            )  # (N, C, 7, 7)
            vec = pooled.mean(dim=(2, 3))  # GAP, Eq. 1
            per_layer_vecs[name] = vec.cpu()

        for i, (ident, vis) in enumerate(meta):
            rec = {
                "seq": seq_dir.name, "frame": frame, "identity": ident,
                "bbox": boxes_xyxy[i], "visibility": vis,
            }
            for name in LAYER_CHANNELS_V8M:
                rec[name] = per_layer_vecs[name][i]
            records.append(rec)

    dt = time.time() - t0
    torch.save(records, out_path)
    print(f"[{seq_dir.name}] {len(frames)} frames, {len(records)} boxes, "
          f"{dt:.1f}s ({dt/max(len(frames),1):.3f}s/frame) -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mot_root", default="/nas/Dataset/MOT/MOT17/train")
    ap.add_argument("--sequences", nargs="+",
                     default=["MOT17-02-FRCNN", "MOT17-04-FRCNN", "MOT17-05-FRCNN",
                              "MOT17-09-FRCNN", "MOT17-10-FRCNN", "MOT17-11-FRCNN",
                              "MOT17-13-FRCNN"])
    ap.add_argument("--out_dir", default="~/msfp/cache")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    backbone = HookedBackbone(device)
    mot_root = Path(args.mot_root)

    for seq in args.sequences:
        seq_dir = mot_root / seq
        out_path = out_dir / f"{seq}.pt"
        if out_path.exists():
            print(f"[{seq}] already cached, skipping")
            continue
        extract_sequence(seq_dir, backbone, device, out_path)


if __name__ == "__main__":
    main()

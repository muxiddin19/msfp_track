#!/usr/bin/env python3
"""
Proper "MSFP-Track (ours)" row: real MSFP attention-fusion embeddings +
the real, already-trained ATL encoder (atl_paper_faithful.pt) predicting a
per-sequence adaptive threshold from real GAP(layer9) scene features --
distinct from the "MSFP (attn, ours)" row, which the main paper's own
structure defines as attention-fusion at a FIXED threshold (no ATL). The
earlier full-metrics pass accidentally ran both at a flat tau=0.25, making
them identical; this corrects that by giving MSFP-Track its own real,
distinguishing ATL-adaptive threshold, matching exactly the mechanism used
in generate_mot17_test_submission.py for the real official-submission runs.
"""
import json
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

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from real_tracker import ByteTrackStyleTracker
import verify_official_trackers as vo
from atl_paper_faithful import ATLPaperFaithful
from litepp.models.feature_pyramid import FeatureFusionModule
from baseline_matrix_full_metrics import evaluate_full

LAYER_ORDER = ["layer4", "layer6", "layer9"]
LAYER_CHANNELS_V8M = {"layer4": 192, "layer6": 384, "layer9": 576}


def extract_val_half_with_scene(seq_dir, backbone, device, min_frame, imgsz_multiple=32):
    """Same as vo.extract_val_half, plus a whole-frame GAP(layer9) scene
    feature per frame (needed for ATL's per-sequence threshold prediction)."""
    det_path = seq_dir / "det" / "det.txt"
    img_dir = seq_dir / "img1"
    by_frame_det = vo.load_det(det_path)
    records = defaultdict(list)
    scene_feats = []
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

        feats = backbone.forward(img_tensor)
        scene_feats.append(feats["layer9"].mean(dim=(2, 3)).squeeze(0).cpu())

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
            spatial_scale = fh / h
            pooled = roi_align(feat_map, rois, output_size=7, spatial_scale=spatial_scale,
                                sampling_ratio=2, aligned=True)
            per_layer_vecs[name] = pooled.mean(dim=(2, 3)).cpu()
        for i in range(len(boxes_xyxy)):
            rec = {"frame": frame, "bbox": boxes_xyxy[i], "det_conf": confs[i]}
            for name in LAYER_ORDER:
                rec[name] = per_layer_vecs[name][i]
            records[frame].append(rec)
    scene_feat_mean = torch.stack(scene_feats).mean(dim=0) if scene_feats else torch.zeros(576)
    return records, scene_feat_mean


@torch.no_grad()
def fuse_msfp(records, fusion, device):
    if not records:
        return np.zeros((0, 128), dtype=np.float32)
    layer_feats = [torch.stack([r[name] for r in records]).to(device) for name in LAYER_ORDER]
    emb = fusion(layer_feats)
    emb = F.normalize(emb, p=2, dim=1)
    return emb.cpu().numpy()


def run_tracker(by_frame, feat_by_frame, tau_h, tau_l, fs, fe, fo):
    tracker = ByteTrackStyleTracker(tau_h=tau_h, tau_l=tau_l, max_age=30, min_hits=1,
                                     appearance_weight=0.5, high_stage_cost_thresh=0.7)
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mot_root = Path("/nas/Dataset/MOT/MOT17/train")
    workdir = Path("~/msfp_honest_repro/trackeval_atl_adaptive").expanduser()
    workdir.mkdir(parents=True, exist_ok=True)

    print("Loading real MSFP fusion head + real ATL encoder...")
    ckpt = torch.load(Path("~/msfp_honest_repro/checkpoints/fusion_attention.pt").expanduser(),
                       map_location=device, weights_only=False)
    layer_channels = [LAYER_CHANNELS_V8M[n] for n in LAYER_ORDER]
    fusion = FeatureFusionModule(layer_channels=layer_channels, output_dim=128,
                                  fusion_type="attention").to(device)
    fusion.load_state_dict(ckpt["state_dict"])
    fusion.eval()

    atl_ckpt = torch.load(Path("~/msfp_honest_repro/checkpoints/atl_paper_faithful.pt").expanduser(),
                           map_location=device, weights_only=False)
    atl = ATLPaperFaithful(input_channels=576, hidden_dim=atl_ckpt["hidden_dim"]).to(device)
    atl.load_state_dict(atl_ckpt["state_dict"])
    atl.eval()

    backbone = vo.HookedBackbone(device)
    splits = {seq: vo.half_split_frame_range(mot_root / seq) for seq in vo.SEQUENCES}

    by_frame_cache, msfp_cache, seq_info, taus = {}, {}, {}, {}
    for seq in vo.SEQUENCES:
        print(f"[{seq}] extracting real features + real scene features (val_half, det.txt)...")
        recs, scene_feat = extract_val_half_with_scene(mot_root / seq, backbone, device, splits[seq])
        by_frame_cache[seq] = recs
        msfp_cache[seq] = {f: fuse_msfp(r, fusion, device) for f, r in recs.items()}
        n_total = len(list((mot_root / seq / "img1").glob("*.jpg")))
        seq_info[seq] = n_total - splits[seq]
        with torch.no_grad():
            tau = float(atl.forward_from_gap(scene_feat.unsqueeze(0).to(device)).item())
        taus[seq] = tau
        print(f"  real ATL-predicted tau = {tau:.3f}")

    gt_folder, trackers_folder = vo.setup_trackeval_dirs(workdir, mot_root, vo.SEQUENCES, "msfp_atl", splits)
    (trackers_folder / "msfp_atl" / "data").mkdir(parents=True, exist_ok=True)
    for seq in vo.SEQUENCES:
        n_total = len(list((mot_root / seq / "img1").glob("*.jpg")))
        fs, fe, fo = splits[seq] + 1, n_total, splits[seq]
        tau = taus[seq]
        lines = run_tracker(by_frame_cache[seq], msfp_cache[seq], tau, max(0.01, tau * 0.5), fs, fe, fo)
        (trackers_folder / "msfp_atl" / "data" / f"{seq}.txt").write_text("\n".join(lines), encoding="utf-8")

    per_seq, agg = evaluate_full(gt_folder, trackers_folder, vo.SEQUENCES, "msfp_atl", seq_info)
    print(f"\n[MSFP-Track, real ATL-adaptive tau] HOTA={agg['HOTA']:.2f} AssA={agg['AssA']:.2f} "
          f"DetA={agg['DetA']:.2f} IDF1={agg['IDF1']:.2f} MOTA={agg['MOTA']:.2f} IDSW={agg['IDSW']}")
    print(f"Per-sequence ATL taus: {taus}")

    out = {"protocol": "val_half, public MOT17 det.txt, real ATL-predicted per-sequence threshold "
                        "(not flat tau=0.25 -- that's the separate 'MSFP (attn, fixed tau)' row)",
           "per_seq": per_seq, "agg": agg, "atl_taus": taus}
    out_path = Path("~/msfp_honest_repro/msfp_track_atl_adaptive_results.json").expanduser()
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Train the paper-faithful ATL module (atl_paper_faithful.py, matching Eq. 5)
to regress the REAL, grid-search-derived per-sequence oracle thresholds
produced by run_oracle_grid_search.py -- via real mean-squared-error
supervision (paper Sec. 3.3, L_tau = ||tau_raw - tau*||^2), using REAL
GAP(Layer-14) features cached from REAL MOT17 frames (extract_cached_features.py
already stores per-detection layer9/layer14-equivalent GAP vectors; here we
need one GAP(F^(14)) vector PER FRAME, i.e. a scene-level feature, not
per-box, so this script re-derives that from the same cached backbone
feature maps used by extract_det_features.py's frame loop, for the training
split's frames only).

Because only 7 oracle targets exist (one per MOT17-train sequence), this
is a tiny, fast regression (leave-one-sequence-out is used for evaluation
in the paper's own Sec. J Thm. 2-3 discussion); here we simply fit on all
7 real oracle values and report real training time and real parameter
count, replacing the simulated MSE values previously printed by
measure_training_time.py's ATL stub.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from atl_paper_faithful import ATLPaperFaithful

LAYER14_MODULE_INDEX = 9  # YOLOv8m's deep/P5 stage per Table 1 (model.9)


@torch.no_grad()
def compute_scene_features(mot_root: Path, sequences, device, max_frames_per_seq=120):
    """GAP(F^(14)) scene-level feature per sampled frame, for each sequence."""
    model = YOLO("yolov8m.pt").model.to(device).eval()
    feats = {}

    def hook(_m, _i, out):
        feats["layer14"] = out

    model.model[LAYER14_MODULE_INDEX].register_forward_hook(hook)

    scene_feats_by_seq = {}
    for seq in sequences:
        img_dir = mot_root / seq / "img1"
        frame_files = sorted(img_dir.glob("*.jpg"))
        if len(frame_files) > max_frames_per_seq:
            idx = np.linspace(0, len(frame_files) - 1, max_frames_per_seq).astype(int)
            frame_files = [frame_files[i] for i in idx]
        vecs = []
        for fp in frame_files:
            img = cv2.imread(str(fp))
            h0, w0 = img.shape[:2]
            h = ((h0 + 31) // 32) * 32
            w = ((w0 + 31) // 32) * 32
            padded = cv2.copyMakeBorder(img, 0, h - h0, 0, w - w0, cv2.BORDER_CONSTANT, value=(114, 114, 114))
            rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb).to(device).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            model(tensor)
            gap = feats["layer14"].mean(dim=(2, 3)).squeeze(0).cpu()
            vecs.append(gap)
        scene_feats_by_seq[seq] = torch.stack(vecs)  # (n_frames, 576)
        print(f"  {seq}: {len(vecs)} scene-feature frames sampled")
    return scene_feats_by_seq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mot_root", default="/nas/Dataset/MOT/MOT17/train")
    ap.add_argument("--oracle_json", default="~/msfp/checkpoints/oracle_thresholds.json")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--out", default="~/msfp/checkpoints/atl_paper_faithful.pt")
    ap.add_argument("--timing_out", default="~/msfp/checkpoints/atl_training_time.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mot_root = Path(args.mot_root)

    with open(Path(args.oracle_json).expanduser(), encoding="utf-8") as f:
        oracle_data = json.load(f)
    oracle = oracle_data["oracle_per_sequence"]
    sequences = list(oracle.keys())
    print(f"Real oracle targets ({len(sequences)} sequences):")
    for seq in sequences:
        print(f"  {seq}: tau*={oracle[seq]['tau_star']:.2f} (HOTA={oracle[seq]['hota_at_tau_star']:.2f})")

    torch.cuda.synchronize() if device.type == "cuda" else None
    t0 = time.time()

    print("\nComputing real GAP(Layer-14) scene features per sequence...")
    scene_feats_by_seq = compute_scene_features(mot_root, sequences, device)

    # One training example per sequence: mean scene feature -> oracle tau*
    X = torch.stack([scene_feats_by_seq[seq].mean(dim=0) for seq in sequences]).to(device)
    y = torch.tensor([oracle[seq]["tau_star"] for seq in sequences], dtype=torch.float32, device=device)

    atl = ATLPaperFaithful(input_channels=576, hidden_dim=args.hidden_dim).to(device)
    n_params = sum(p.numel() for p in atl.parameters() if p.requires_grad)
    print(f"\nATL (paper-faithful, Eq. 5) trainable parameters: {n_params:,}")

    opt = torch.optim.Adam(atl.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        pred = atl.forward_from_gap(X)
        loss = F.mse_loss(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{args.epochs} - MSE: {loss.item():.6f}")

    torch.cuda.synchronize() if device.type == "cuda" else None
    elapsed = time.time() - t0

    with torch.no_grad():
        final_pred = atl.forward_from_gap(X)
    print("\nFinal predictions vs. real oracle targets:")
    for seq, p, t in zip(sequences, final_pred.cpu().tolist(), y.cpu().tolist()):
        print(f"  {seq}: predicted={p:.3f}, oracle={t:.3f}")

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": atl.state_dict(), "hidden_dim": args.hidden_dim}, out_path)

    timing = {
        "module": "ATL (paper-faithful, Eq. 5: sigmoid(MLP(GAP(F^14))))",
        "trainable_params": n_params,
        "epochs": args.epochs,
        "n_training_sequences": len(sequences),
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "training_time_seconds": elapsed,
        "training_time_hours": elapsed / 3600,
        "final_mse": loss.item(),
        "note": "Includes real GAP(Layer-14) scene-feature extraction from "
                "sampled real MOT17-train frames plus real MLP regression "
                "against the 7 real oracle thresholds from "
                "run_oracle_grid_search.py; excludes the grid-search's own "
                "tracking+HOTA cost, reported separately in "
                "oracle_thresholds.json's timing block.",
    }
    timing_path = Path(args.timing_out).expanduser()
    with open(timing_path, "w", encoding="utf-8") as f:
        json.dump(timing, f, indent=2)
    print(f"\nReal ATL training time: {timing['training_time_hours']:.4f} GPU-hours -> {timing_path}")


if __name__ == "__main__":
    main()

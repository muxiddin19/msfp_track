#!/usr/bin/env python3
"""
Trains both ATL variants -- ATLPaperFaithful (simple, literal Eq. 5) and
AdaptiveThresholdModule (richer SceneEncoder, global+spatial branches) --
via real MSE regression against the genuinely leak-free oracle thresholds
(oracle_grid_search_trainhalf.py's output), using scene features computed
ONLY from train_half frames (fixing a second, input-side instance of the
same leak: the original train_atl.py's compute_scene_features() samples
from the full sequence via an unrestricted glob, not just train_half).
"""
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atl_paper_faithful import ATLPaperFaithful
from litepp.models.adaptive_threshold import AdaptiveThresholdModule
import verify_official_trackers as vo

LAYER14_MODULE_INDEX = 9


@torch.no_grad()
def compute_scene_features_trainhalf(mot_root: Path, sequences, splits, device, max_frames_per_seq=120):
    model = YOLO("yolov8m.pt").model.to(device).eval()
    feats = {}

    def hook(_m, _i, out):
        feats["layer14"] = out

    model.model[LAYER14_MODULE_INDEX].register_forward_hook(hook)

    scene_feats_by_seq = {}
    for seq in sequences:
        img_dir = mot_root / seq / "img1"
        frame_files = sorted(img_dir.glob("*.jpg"))
        frame_files = [fp for fp in frame_files if int(fp.stem) <= splits[seq]]  # train_half only
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
        scene_feats_by_seq[seq] = torch.stack(vecs)
        print(f"  {seq}: {len(vecs)} train_half-only scene-feature frames (<= frame {splits[seq]})")
    return scene_feats_by_seq


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mot_root = Path("/nas/Dataset/MOT/MOT17/train")

    with open(Path("~/msfp_honest_repro/oracle_thresholds_trainhalf.json").expanduser(), encoding="utf-8") as f:
        oracle_data = json.load(f)
    oracle = oracle_data["oracle_per_sequence"]
    sequences = list(oracle.keys())
    splits = {seq: vo.half_split_frame_range(mot_root / seq) for seq in sequences}

    print("Real oracle targets (leak-free, train_half only):")
    for seq in sequences:
        print(f"  {seq}: tau*={oracle[seq]['tau_star']:.2f}")

    print("\nComputing real GAP(Layer-14) scene features (train_half only)...")
    scene_feats_by_seq = compute_scene_features_trainhalf(mot_root, sequences, splits, device)

    X = torch.stack([scene_feats_by_seq[seq].mean(dim=0) for seq in sequences]).to(device)
    y = torch.tensor([oracle[seq]["tau_star"] for seq in sequences], dtype=torch.float32, device=device)

    # --- ATLPaperFaithful (simple, literal Eq. 5) ---
    print("\n=== Training ATLPaperFaithful (simple) on clean targets ===")
    atl_simple = ATLPaperFaithful(input_channels=576, hidden_dim=64).to(device)
    n_params_simple = sum(p.numel() for p in atl_simple.parameters() if p.requires_grad)
    opt = torch.optim.Adam(atl_simple.parameters(), lr=1e-3)
    t0 = time.time()
    for epoch in range(200):
        pred = atl_simple.forward_from_gap(X)
        loss = F.mse_loss(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (epoch + 1) % 50 == 0:
            print(f"  epoch {epoch+1}: MSE={loss.item():.6f}")
    elapsed_simple = time.time() - t0
    print(f"  {n_params_simple} params, final MSE={loss.item():.6f}, {elapsed_simple:.1f}s")
    torch.save({"state_dict": atl_simple.state_dict(), "hidden_dim": 64},
               Path("~/msfp_honest_repro/checkpoints/atl_paper_faithful_clean.pt").expanduser())

    # --- AdaptiveThresholdModule (richer SceneEncoder) ---
    print("\n=== Training AdaptiveThresholdModule (richer SceneEncoder) on clean targets ===")
    atl_rich = AdaptiveThresholdModule(input_channels=576, hidden_dim=128, min_threshold=0.01,
                                        max_threshold=0.50, default_threshold=0.25).to(device)
    n_params_rich = sum(p.numel() for p in atl_rich.parameters() if p.requires_grad)
    # X here is already GAP-pooled (576,); AdaptiveThresholdModule expects (B,C,H,W) for its
    # SceneEncoder's spatial branch -- reshape each 576-d vector to a degenerate (576,1,1) map
    # so the spatial_encoder's AdaptiveAvgPool2d(4) still runs (replicated, not a real spatial
    # layout, since only the GAP'd vector was cached -- a real spatial map would require
    # re-running the backbone per sequence with full feature maps retained, out of scope here;
    # this isolates the SceneEncoder's *extra capacity*, not genuinely new spatial information).
    X_map = X.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 4, 4).contiguous()
    opt2 = torch.optim.Adam(atl_rich.parameters(), lr=1e-3)
    t0 = time.time()
    for epoch in range(200):
        pred, _ = atl_rich.forward(X_map)
        loss2 = F.mse_loss(pred.squeeze(-1), y)
        opt2.zero_grad()
        loss2.backward()
        opt2.step()
        if (epoch + 1) % 50 == 0:
            print(f"  epoch {epoch+1}: MSE={loss2.item():.6f}")
    elapsed_rich = time.time() - t0
    print(f"  {n_params_rich} params, final MSE={loss2.item():.6f}, {elapsed_rich:.1f}s")
    torch.save({"state_dict": atl_rich.state_dict()},
               Path("~/msfp_honest_repro/checkpoints/atl_rich_clean.pt").expanduser())

    print("\nFinal predictions, both models, vs. clean oracle targets:")
    with torch.no_grad():
        pred_simple = atl_simple.forward_from_gap(X).cpu().tolist()
        pred_rich, _ = atl_rich.forward(X_map)
        pred_rich = pred_rich.squeeze(-1).cpu().tolist()
    for seq, ps, pr, t in zip(sequences, pred_simple, pred_rich, y.cpu().tolist()):
        print(f"  {seq}: simple={ps:.3f}  rich={pr:.3f}  oracle={t:.3f}")

    print(f"\nParam counts: simple={n_params_simple}, rich={n_params_rich}")


if __name__ == "__main__":
    main()

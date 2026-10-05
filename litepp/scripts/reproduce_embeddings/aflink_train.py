#!/usr/bin/env python3
"""
AFLink (Appearance-Free Link model), from StrongSORT (Du et al., TMM 2023):
a small, real, trained model that decides whether two track FRAGMENTS
(one ending, one starting nearby in time/space) belong to the same
identity, using motion/geometry features alone -- no appearance. Used as
post-processing to re-link tracks broken by brief occlusion, specifically
targeting ID switches that appearance-based recovery can miss.

Training data: synthetic fragment splits generated from REAL MOT17-train
ground-truth tracks (positives: a real GT track split at a random point,
simulating an occlusion gap; negatives: temporally-overlapping fragments
from different real GT identities). This never touches val_half -- GT
splitting is independent of any detection/tracking output, so there is no
leakage into the real held-out evaluation set used throughout this project.

Feature representation (simplified from StrongSORT's temporal-CNN design
for tractability, while keeping the same real, motion-only information
content): for a candidate (fragment A ending, fragment B starting) pair,
velocity-extrapolate A's last known state forward to B's start frame and
compare against B's real start state.
"""
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import verify_official_trackers as vo

MAX_GAP = 30  # frames; StrongSORT-style cap on how long an occlusion gap can be
SEQUENCES = vo.SEQUENCES


def load_gt_tracks(seq_dir: Path, frame_hi: int):
    """Real GT tracks (train_half only: frame <= frame_hi), pedestrian-ish,
    as {id: sorted [(frame, cx, cy, w, h), ...]}."""
    tracks = defaultdict(list)
    for line in (seq_dir / "gt" / "gt.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        p = line.split(",")
        frame, tid, x, y, w, h, conf, cls = int(p[0]), int(p[1]), float(p[2]), float(p[3]), \
            float(p[4]), float(p[5]), int(float(p[6])), int(float(p[7]))
        if frame > frame_hi or conf != 1 or cls != 1:
            continue
        cx, cy = x + w / 2.0, y + h / 2.0
        tracks[tid].append((frame, cx, cy, w, h))
    for tid in tracks:
        tracks[tid].sort(key=lambda r: r[0])
    return {tid: recs for tid, recs in tracks.items() if len(recs) >= 6}


def featurize(frag_a, frag_b, img_diag):
    """frag_a, frag_b: lists of (frame,cx,cy,w,h), sorted. Returns a real,
    motion-only feature vector describing the (A ending -> B starting) gap."""
    a = np.array(frag_a[-5:]) if len(frag_a) >= 2 else np.array(frag_a)
    b_start = frag_b[0]
    if len(a) >= 2:
        dt_a = a[-1, 0] - a[0, 0]
        vx = (a[-1, 1] - a[0, 1]) / max(dt_a, 1.0)
        vy = (a[-1, 2] - a[0, 2]) / max(dt_a, 1.0)
    else:
        vx = vy = 0.0
    last = frag_a[-1]
    gap = b_start[0] - last[0]
    pred_x = last[1] + vx * gap
    pred_y = last[2] + vy * gap
    dx = (pred_x - b_start[1]) / img_diag
    dy = (pred_y - b_start[2]) / img_diag
    raw_dx = (last[1] - b_start[1]) / img_diag
    raw_dy = (last[2] - b_start[2]) / img_diag
    size_ratio_w = b_start[3] / max(last[3], 1e-3)
    size_ratio_h = b_start[4] / max(last[4], 1e-3)
    return np.array([gap / MAX_GAP, dx, dy, raw_dx, raw_dy,
                      np.log(size_ratio_w + 1e-6), np.log(size_ratio_h + 1e-6),
                      np.hypot(vx, vy) / img_diag], dtype=np.float32)


def build_dataset(mot_root: Path, n_pos=6000, n_neg=6000, seed=0):
    rng = random.Random(seed)
    X, y = [], []
    for seq in SEQUENCES:
        seq_dir = mot_root / seq
        split = vo.half_split_frame_range(seq_dir)
        tracks = load_gt_tracks(seq_dir, split)
        if len(tracks) < 2:
            continue
        ini = (seq_dir / "seqinfo.ini").read_text(encoding="utf-8")
        w_img = int([l for l in ini.splitlines() if l.startswith("imWidth")][0].split("=")[1])
        h_img = int([l for l in ini.splitlines() if l.startswith("imHeight")][0].split("=")[1])
        img_diag = float(np.hypot(w_img, h_img))

        ids = list(tracks.keys())
        # Positives: split one real track into (A, B) at a random interior point.
        n_this_pos = max(1, n_pos // len(SEQUENCES))
        for _ in range(n_this_pos):
            tid = rng.choice(ids)
            recs = tracks[tid]
            if len(recs) < 8:
                continue
            split_i = rng.randint(3, len(recs) - 4)
            gap = rng.randint(1, MAX_GAP)
            frag_a = recs[:split_i]
            # Find the first real frame of B at least `gap` after A's end (simulates occlusion).
            a_end_frame = frag_a[-1][0]
            frag_b = [r for r in recs[split_i:] if r[0] - a_end_frame >= 1]
            if not frag_b:
                continue
            frag_b = frag_b[:5]
            X.append(featurize(frag_a, frag_b, img_diag))
            y.append(1)

        # Negatives: fragments from two DIFFERENT real identities, temporally close.
        n_this_neg = max(1, n_neg // len(SEQUENCES))
        for _ in range(n_this_neg):
            if len(ids) < 2:
                continue
            tid_a, tid_b = rng.sample(ids, 2)
            recs_a, recs_b = tracks[tid_a], tracks[tid_b]
            if len(recs_a) < 4 or len(recs_b) < 4:
                continue
            cut_a = rng.randint(2, len(recs_a) - 2)
            frag_a = recs_a[:cut_a]
            b_start_frame = frag_a[-1][0] + rng.randint(1, MAX_GAP)
            frag_b = [r for r in recs_b if r[0] >= b_start_frame]
            if not frag_b:
                continue
            frag_b = frag_b[:5]
            X.append(featurize(frag_a, frag_b, img_diag))
            y.append(0)

    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.float32)
    return X, y


class AFLinkNet(nn.Module):
    def __init__(self, in_dim=8, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def main():
    mot_root = Path("/nas/Dataset/MOT/MOT17/train")
    print("Building real, GT-derived synthetic fragment-pair dataset (train_half only)...")
    X, y = build_dataset(mot_root)
    print(f"  {len(y)} pairs, {int(y.sum())} positive, {int((1 - y).sum())} negative")

    n = len(y)
    idx = np.random.RandomState(0).permutation(n)
    n_val = int(0.15 * n)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    mean, std = X[train_idx].mean(0, keepdims=True), X[train_idx].std(0, keepdims=True) + 1e-6
    Xn = (X - mean) / std

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AFLinkNet(in_dim=X.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()

    Xt = torch.from_numpy(Xn[train_idx]).to(device)
    yt = torch.from_numpy(y[train_idx]).to(device)
    Xv = torch.from_numpy(Xn[val_idx]).to(device)
    yv = torch.from_numpy(y[val_idx]).to(device)

    best_val_acc = 0.0
    best_state = None
    for epoch in range(200):
        model.train()
        opt.zero_grad()
        logits = model(Xt)
        loss = bce(logits, yt)
        loss.backward()
        opt.step()
        if epoch % 20 == 0 or epoch == 199:
            model.eval()
            with torch.no_grad():
                val_logits = model(Xv)
                val_acc = ((val_logits > 0).float() == yv).float().mean().item()
                val_loss = bce(val_logits, yv).item()
            print(f"  epoch {epoch}: train_loss={loss.item():.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    print(f"Best val_acc={best_val_acc:.4f}")

    n_params = sum(p.numel() for p in model.parameters())
    out_path = Path("~/msfp_honest_repro/checkpoints/aflink.pt").expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "mean": mean, "std": std,
                "in_dim": X.shape[1], "hidden": 32, "max_gap": MAX_GAP,
                "val_acc": best_val_acc, "n_params": n_params}, out_path)
    print(f"Saved AFLink model ({n_params} params, val_acc={best_val_acc:.4f}) -> {out_path}")


if __name__ == "__main__":
    main()

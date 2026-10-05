#!/usr/bin/env python3
"""
Trains several fusion-head regularization variants on the genuinely
train_half-only cache (extract_cached_features_trainhalf.py), fixing the
data-leakage issue in the existing deployed fusion_attention.pt (whose
cache spans the full sequence, including val_half frames -- confirmed by
direct inspection). Same real online hard-negative triplet procedure as
train_fusion_head.py (identical hyperparameters otherwise), only
dropout_p and weight_decay vary, to see whether regularization choice
matters and -- more importantly -- what a genuinely leak-free "default"
checkpoint's real downstream tracking performance looks like compared to
the existing (possibly leak-inflated) one.
"""
import argparse
import random
import time
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from litepp.models.feature_pyramid import FeatureFusionModule

LAYER_CHANNELS_V8M = {"layer4": 192, "layer6": 384, "layer9": 576}
LAYER_ORDER = ["layer4", "layer6", "layer9"]
SEQUENCES = ["MOT17-02-FRCNN", "MOT17-04-FRCNN", "MOT17-05-FRCNN", "MOT17-09-FRCNN",
             "MOT17-10-FRCNN", "MOT17-11-FRCNN", "MOT17-13-FRCNN"]


def load_cache(cache_dir, sequences):
    records = []
    for seq in sequences:
        records.extend(torch.load(cache_dir / f"{seq}.pt", weights_only=False))
    return records


def index_by_identity(records):
    idx = defaultdict(lambda: defaultdict(list))
    for i, r in enumerate(records):
        idx[r["seq"]][r["identity"]].append(i)
    for seq in idx:
        for ident in idx[seq]:
            idx[seq][ident].sort(key=lambda i: records[i]["frame"])
    return idx


class PKBatchSampler:
    def __init__(self, records, by_identity, P=8, K=4, window=30, min_track_len=2):
        self.records = records
        self.window = window
        self.P, self.K = P, K
        self.tracks = []
        for seq, idents in by_identity.items():
            for ident, idxs in idents.items():
                if len(idxs) >= min_track_len:
                    self.tracks.append((seq, ident, idxs))

    def sample_batch(self):
        chosen_tracks = random.sample(self.tracks, k=min(self.P, len(self.tracks)))
        batch_idx, batch_pid = [], []
        for pid, (seq, ident, idxs) in enumerate(chosen_tracks):
            anchor_pos = random.randrange(len(idxs))
            anchor_frame = self.records[idxs[anchor_pos]]["frame"]
            window_pool = [i for i in idxs if abs(self.records[i]["frame"] - anchor_frame) <= self.window]
            k = min(self.K, len(window_pool))
            picks = random.sample(window_pool, k=k)
            while len(picks) < self.K:
                picks.append(random.choice(window_pool))
            batch_idx.extend(picks)
            batch_pid.extend([pid] * self.K)
        return batch_idx, batch_pid

    def __len__(self):
        return max(1, len(self.tracks) // self.P)


def hardest_triplet_loss(embeddings, pids, margin=0.3):
    emb = F.normalize(embeddings, p=2, dim=1)
    sim = emb @ emb.t()
    dist = 1.0 - sim
    pids = torch.tensor(pids, device=embeddings.device)
    same = pids.unsqueeze(0) == pids.unsqueeze(1)
    diff = ~same
    n = emb.shape[0]
    eye = torch.eye(n, dtype=torch.bool, device=embeddings.device)
    pos_mask = same & ~eye
    losses = []
    for i in range(n):
        pos_d = dist[i][pos_mask[i]]
        neg_d = dist[i][diff[i]]
        if pos_d.numel() == 0 or neg_d.numel() == 0:
            continue
        losses.append(F.relu(pos_d.max() - neg_d.min() + margin))
    if not losses:
        return embeddings.sum() * 0.0
    return torch.stack(losses).mean()


def train_one(records, by_identity, device, dropout_p, weight_decay, epochs=50, seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
    sampler = PKBatchSampler(records, by_identity, P=8, K=4, window=30)
    batches_per_epoch = len(sampler)
    layer_channels = [LAYER_CHANNELS_V8M[n] for n in LAYER_ORDER]
    fusion = FeatureFusionModule(layer_channels=layer_channels, output_dim=128,
                                  fusion_type="attention", dropout_p=dropout_p).to(device)
    opt = torch.optim.Adam(fusion.parameters(), lr=1e-4, weight_decay=weight_decay)

    for epoch in range(epochs):
        for _ in range(batches_per_epoch):
            idxs, pids = sampler.sample_batch()
            layer_feats = [torch.stack([records[i][name] for i in idxs]).to(device) for name in LAYER_ORDER]
            embeddings = fusion(layer_feats)
            loss = hardest_triplet_loss(embeddings, pids)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if (epoch + 1) % 10 == 0:
            print(f"    epoch {epoch+1}/{epochs}: loss={loss.item():.4f}")
    return fusion


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default="~/msfp_honest_repro/cache_trainhalf")
    ap.add_argument("--out_dir", default="~/msfp_honest_repro/checkpoints")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir = Path(args.cache_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading leak-free (train_half-only) cached features...")
    records = load_cache(cache_dir, SEQUENCES)
    by_identity = index_by_identity(records)
    print(f"  {len(records)} boxes, {sum(len(v) for v in by_identity.values())} tracks")

    configs = [
        ("default_leakfree", 0.1, 0.0),       # matches original hyperparams, just leak-free data
        ("dropout0.3", 0.3, 0.0),
        ("wd1e-4", 0.1, 1e-4),
        ("dropout0.3_wd1e-4", 0.3, 1e-4),
    ]

    for name, dropout_p, wd in configs:
        print(f"\n=== Training '{name}' (dropout_p={dropout_p}, weight_decay={wd}) ===")
        t0 = time.time()
        fusion = train_one(records, by_identity, device, dropout_p, wd)
        elapsed = time.time() - t0
        out_path = out_dir / f"fusion_{name}.pt"
        torch.save({"state_dict": fusion.state_dict(), "dropout_p": dropout_p, "weight_decay": wd}, out_path)
        print(f"  Saved -> {out_path} ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()

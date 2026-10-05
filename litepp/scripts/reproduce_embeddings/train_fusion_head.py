#!/usr/bin/env python3
"""
Stage 2: train the Instance-Adaptive Attention Fusion head (the paper's
actual trainable contribution, 79K params) on cached, frozen-backbone
features from stage 1, using the online hard-negative triplet loss
described in the paper (Sec. 3.4, Eq. 8):

    L_tri = max(0, d(a,p) - d(a,n) + m)

with P=8 identities x K=4 instances per batch, positives sampled within a
30-frame temporal window of the anchor, negatives the hardest in-batch
different-identity sample, m=0.3, Adam lr=1e-4.

This REPLACES litepp/scripts/measure_training_time.py's placeholder
(`# TODO: Replace with your actual training code` + `time.sleep(0.1)`
per epoch) with a real training loop, and reports the real wall-clock
GPU time actually spent, instead of a simulated number.
"""
import argparse
import json
import random
import time
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from litepp.models.feature_pyramid import FeatureFusionModule

LAYER_CHANNELS_V8M = {"layer4": 192, "layer6": 384, "layer9": 576}
LAYER_ORDER = ["layer4", "layer6", "layer9"]


def load_cache(cache_dir: Path, sequences):
    records = []
    for seq in sequences:
        p = cache_dir / f"{seq}.pt"
        if not p.exists():
            raise FileNotFoundError(f"Missing cached features for {seq}: {p}. Run extract_cached_features.py first.")
        records.extend(torch.load(p, weights_only=False))
    return records


def index_by_identity(records):
    """seq -> identity -> sorted list of record indices (by frame)."""
    idx = defaultdict(lambda: defaultdict(list))
    for i, r in enumerate(records):
        idx[r["seq"]][r["identity"]].append(i)
    for seq in idx:
        for ident in idx[seq]:
            idx[seq][ident].sort(key=lambda i: records[i]["frame"])
    return idx


class PKBatchSampler:
    """P identities x K instances per batch; positives within a 30-frame window."""

    def __init__(self, records, by_identity, P=8, K=4, window=30, min_track_len=2):
        self.records = records
        self.window = window
        self.P, self.K = P, K
        # Flatten (seq, identity) -> indices, keeping only identities with
        # enough length to actually sample K within-window instances.
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
            window_pool = [i for i in idxs
                           if abs(self.records[i]["frame"] - anchor_frame) <= self.window]
            k = min(self.K, len(window_pool))
            picks = random.sample(window_pool, k=k)
            # pad by resampling with replacement if the track is short
            while len(picks) < self.K:
                picks.append(random.choice(window_pool))
            batch_idx.extend(picks)
            batch_pid.extend([pid] * self.K)
        return batch_idx, batch_pid

    def __len__(self):
        # one epoch := enough batches to see each track ~once as anchor
        return max(1, len(self.tracks) // self.P)


def hardest_triplet_loss(embeddings, pids, margin=0.3):
    """Online hardest-negative triplet loss over cosine distance d=1-cos_sim."""
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
        hardest_pos = pos_d.max()
        hardest_neg = neg_d.min()
        losses.append(F.relu(hardest_pos - hardest_neg + margin))
    if not losses:
        return embeddings.sum() * 0.0
    return torch.stack(losses).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default="~/msfp/cache")
    ap.add_argument("--sequences", nargs="+",
                     default=["MOT17-02-FRCNN", "MOT17-04-FRCNN", "MOT17-05-FRCNN",
                              "MOT17-09-FRCNN", "MOT17-10-FRCNN", "MOT17-11-FRCNN",
                              "MOT17-13-FRCNN"])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--P", type=int, default=8)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--margin", type=float, default=0.3)
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--out", default="~/msfp/checkpoints/fusion_attention.pt")
    ap.add_argument("--timing_out", default="~/msfp/checkpoints/training_time.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir = Path(args.cache_dir).expanduser()
    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading cached frozen-backbone features...")
    records = load_cache(cache_dir, args.sequences)
    by_identity = index_by_identity(records)
    n_tracks = sum(len(v) for v in by_identity.values())
    print(f"Loaded {len(records)} boxes, {n_tracks} (seq,identity) tracks across {len(args.sequences)} sequences")

    sampler = PKBatchSampler(records, by_identity, P=args.P, K=args.K, window=args.window)
    batches_per_epoch = len(sampler)
    print(f"Batches/epoch: {batches_per_epoch} (P={args.P}, K={args.K})")

    layer_channels = [LAYER_CHANNELS_V8M[n] for n in LAYER_ORDER]
    fusion = FeatureFusionModule(layer_channels=layer_channels, output_dim=128,
                                  fusion_type="attention").to(device)
    n_params = sum(p.numel() for p in fusion.parameters() if p.requires_grad)
    print(f"Fusion head trainable parameters: {n_params:,}")

    opt = torch.optim.Adam(fusion.parameters(), lr=args.lr)

    torch.cuda.synchronize() if device.type == "cuda" else None
    t_start = time.time()

    loss_history = []
    for epoch in range(args.epochs):
        epoch_losses = []
        for _ in range(batches_per_epoch):
            idxs, pids = sampler.sample_batch()
            layer_feats = []
            for name in LAYER_ORDER:
                feat = torch.stack([records[i][name] for i in idxs]).to(device)
                layer_feats.append(feat)

            embeddings = fusion(layer_feats)
            loss = hardest_triplet_loss(embeddings, pids, margin=args.margin)

            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_losses.append(loss.item())

        mean_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        loss_history.append(mean_loss)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{args.epochs} - triplet loss: {mean_loss:.4f}")

    torch.cuda.synchronize() if device.type == "cuda" else None
    elapsed = time.time() - t_start

    torch.save({"state_dict": fusion.state_dict(),
                "layer_order": LAYER_ORDER,
                "layer_channels": layer_channels,
                "args": vars(args)}, out_path)
    print(f"\nSaved trained fusion head to {out_path}")

    timing = {
        "module": "Instance-Adaptive Attention Fusion",
        "trainable_params": n_params,
        "epochs": args.epochs,
        "batches_per_epoch": batches_per_epoch,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "training_time_seconds": elapsed,
        "training_time_hours": elapsed / 3600,
        "final_loss": loss_history[-1] if loss_history else None,
        "note": "Real measured wall-clock time for training the fusion head only; "
                "excludes one-time frozen-backbone feature caching (stage 1), "
                "consistent with the paper's framing of training cost "
                "(backbone frozen, only the fusion head is trained).",
    }
    timing_path = Path(args.timing_out).expanduser()
    timing_path.parent.mkdir(parents=True, exist_ok=True)
    with open(timing_path, "w", encoding="utf-8") as f:
        json.dump(timing, f, indent=2)
    print(f"Real training time: {timing['training_time_hours']:.3f} GPU-hours "
          f"on {timing['device']} -> {timing_path}")


if __name__ == "__main__":
    main()

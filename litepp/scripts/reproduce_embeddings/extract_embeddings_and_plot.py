#!/usr/bin/env python3
"""
Stage 3: using real cached frozen-backbone features (stage 1) and the real
trained fusion head (stage 2), compute REAL embeddings, a REAL t-SNE
projection, and REAL genuine/impostor cosine-similarity distributions --
replacing the synthetic np.random-based placeholders that previously lived
in litepp/experiments/generate_paper_visualizations.py
(generate_tsne_visualization, generate_score_distributions).

Two embedding spaces are compared, matching the paper's Table 6 conditions:
  - "LITE baseline": the raw, frozen single-layer feature (deepest layer,
    layer9 for YOLOv8m), L2-normalized -- no fusion-head training involved.
  - "MSFP-Track": the trained Instance-Adaptive Attention Fusion output.

Outputs (written to --out_dir):
  tsne_msfptrack.pdf / .png   -- real t-SNE, 15 identities, MOT17-02/04/09
  score_distributions.pdf/.png -- real genuine/impostor cosine histograms
  real_metrics.json           -- the real AUC / gap / per-sequence numbers,
                                  for updating the paper's captions honestly.
"""
import argparse
import json
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
# Embed real (TrueType) fonts in exported PDFs instead of matplotlib's
# default Type 3 bitmap fonts, which several venues (incl. ACCV camera-ready)
# explicitly reject.
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from litepp.models.feature_pyramid import FeatureFusionModule

LAYER_CHANNELS_V8M = {"layer4": 192, "layer6": 384, "layer9": 576}
LAYER_ORDER = ["layer4", "layer6", "layer9"]
LITE_BASELINE_LAYER = "layer9"  # deepest/most semantic single layer

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})
COLORS = {'lite': '#1f77b4', 'msfp_track': '#2ca02c',
          'positive': '#2ca02c', 'negative': '#d62728'}


def load_cache(cache_dir: Path, sequences):
    records = []
    for seq in sequences:
        p = cache_dir / f"{seq}.pt"
        if not p.exists():
            raise FileNotFoundError(f"Missing cached features for {seq}: {p}")
        records.extend(torch.load(p, weights_only=False))
    return records


@torch.no_grad()
def embed_lite_baseline(records, device):
    feats = torch.stack([r[LITE_BASELINE_LAYER] for r in records]).to(device)
    return F.normalize(feats, p=2, dim=1).cpu()


@torch.no_grad()
def embed_msfp_track(records, fusion, device):
    layer_feats = [torch.stack([r[name] for r in records]).to(device) for name in LAYER_ORDER]
    emb = fusion(layer_feats)
    return F.normalize(emb, p=2, dim=1).cpu()


def genuine_impostor_scores(embeddings: torch.Tensor, identities, max_pairs=200_000, seed=0):
    """Real cosine-similarity scores for genuine (same-id) vs impostor (diff-id) pairs."""
    rng = random.Random(seed)
    by_id = defaultdict(list)
    for i, ident in enumerate(identities):
        by_id[ident].append(i)

    ids = list(by_id.keys())
    genuine, impostor = [], []

    # Genuine: all same-identity pairs (capped for very long tracks)
    for ident, idxs in by_id.items():
        if len(idxs) < 2:
            continue
        pairs = [(a, b) for ai, a in enumerate(idxs) for b in idxs[ai + 1:]]
        if len(pairs) > 2000:
            pairs = rng.sample(pairs, 2000)
        for a, b in pairs:
            genuine.append(F.cosine_similarity(embeddings[a], embeddings[b], dim=0).item())

    # Impostor: random different-identity pairs, same count as genuine
    n_target = min(len(genuine) * 1, max_pairs) or 1
    attempts = 0
    while len(impostor) < n_target and attempts < n_target * 10:
        attempts += 1
        id_a, id_b = rng.sample(ids, 2)
        a = rng.choice(by_id[id_a])
        b = rng.choice(by_id[id_b])
        impostor.append(F.cosine_similarity(embeddings[a], embeddings[b], dim=0).item())

    return np.array(genuine), np.array(impostor)


def compute_real_metrics(records, lite_emb, msfp_emb):
    identities_global = [f"{r['seq']}_{r['identity']}" for r in records]
    results = {}
    for name, emb in [("LITE (single-layer)", lite_emb), ("MSFP-Track", msfp_emb)]:
        genuine, impostor = genuine_impostor_scores(emb, identities_global)
        y = np.concatenate([np.ones_like(genuine), np.zeros_like(impostor)])
        s = np.concatenate([genuine, impostor])
        auc = roc_auc_score(y, s)
        gap = float(genuine.mean() - impostor.mean())
        results[name] = {
            "auc": float(auc), "gap": gap,
            "genuine_mean": float(genuine.mean()), "genuine_std": float(genuine.std()),
            "impostor_mean": float(impostor.mean()), "impostor_std": float(impostor.std()),
            "n_genuine_pairs": int(len(genuine)), "n_impostor_pairs": int(len(impostor)),
        }
    # Per-sequence gap range (for the "X--Y across sequences" style claim)
    per_seq_gap = {}
    for seq in sorted(set(r["seq"] for r in records)):
        seq_idx = [i for i, r in enumerate(records) if r["seq"] == seq]
        seq_ids = [identities_global[i] for i in seq_idx]
        seq_emb = msfp_emb[seq_idx]
        g, imp = genuine_impostor_scores(seq_emb, seq_ids)
        if len(g) and len(imp):
            per_seq_gap[seq] = float(g.mean() - imp.mean())
    results["msfp_track_per_sequence_gap"] = per_seq_gap
    return results


def plot_score_distributions(records, lite_emb, msfp_emb, out_dir: Path):
    identities_global = [f"{r['seq']}_{r['identity']}" for r in records]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (name, emb, color_key) in zip(
        axes, [("Single Layer (LITE)", lite_emb, 'lite'), ("MSFP-Track", msfp_emb, 'msfp_track')]
    ):
        genuine, impostor = genuine_impostor_scores(emb, identities_global)
        bins = np.linspace(-1, 1, 60)
        ax.hist(impostor, bins=bins, alpha=0.6, color=COLORS['negative'],
                label='Impostor pairs (real)', density=True)
        ax.hist(genuine, bins=bins, alpha=0.6, color=COLORS['positive'],
                label='Genuine pairs (real)', density=True)
        ax.axvline(genuine.mean(), color=COLORS['positive'], linestyle='--', linewidth=2, alpha=0.8)
        ax.axvline(impostor.mean(), color=COLORS['negative'], linestyle='--', linewidth=2, alpha=0.8)
        gap = genuine.mean() - impostor.mean()
        ax.annotate(f'Gap: {gap:.3f}', xy=(0.5, 0.92), xycoords='axes fraction',
                    fontsize=10, fontweight='bold', ha='center',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.set_xlabel('Cosine Similarity')
        ax.set_ylabel('Density')
        ax.set_title(name)
        ax.legend(loc='upper left', fontsize=8)
        ax.set_xlim([-1, 1])
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / 'score_distributions.pdf')
    plt.savefig(out_dir / 'score_distributions.png', dpi=300)
    plt.close()
    print(f"Saved real score_distributions.pdf/png to {out_dir}")


def plot_tsne(records, msfp_emb, out_dir: Path, n_identities=15, seed=0):
    rng = random.Random(seed)
    by_id = defaultdict(list)
    for i, r in enumerate(records):
        by_id[f"{r['seq']}_{r['identity']}"].append(i)
    # Keep only identities with enough detections to form a visible cluster
    eligible = [k for k, v in by_id.items() if len(v) >= 8]
    chosen = rng.sample(eligible, k=min(n_identities, len(eligible)))

    idxs, labels = [], []
    for label_i, key in enumerate(chosen):
        pool = by_id[key]
        take = pool if len(pool) <= 40 else rng.sample(pool, 40)
        idxs.extend(take)
        labels.extend([label_i] * len(take))

    emb = msfp_emb[idxs].numpy()
    labels = np.array(labels)

    tsne = TSNE(n_components=2, perplexity=min(30, max(5, len(idxs) // 10)),
                random_state=seed, init='pca', metric='cosine')
    proj = tsne.fit_transform(emb)

    fig, ax = plt.subplots(figsize=(7, 6))
    colors = plt.cm.tab20(np.linspace(0, 1, len(chosen)))
    for i in range(len(chosen)):
        mask = labels == i
        ax.scatter(proj[mask, 0], proj[mask, 1], c=[colors[i]], label=f'ID {i+1}',
                   s=40, alpha=0.8, edgecolors='white', linewidths=0.5)
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.set_title('MSFP-Track Feature Embeddings (real, MOT17-02/04/09)')
    ax.legend(loc='upper right', fontsize=7, ncol=2, framealpha=0.9,
              columnspacing=0.5, handletextpad=0.3, borderpad=0.4)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / 'tsne_msfptrack.pdf', bbox_inches='tight')
    plt.savefig(out_dir / 'tsne_msfptrack.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved real tsne_msfptrack.pdf/png to {out_dir} ({len(chosen)} identities)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default="~/msfp/cache")
    ap.add_argument("--checkpoint", default="~/msfp/checkpoints/fusion_attention.pt")
    ap.add_argument("--sequences", nargs="+",
                     default=["MOT17-02-FRCNN", "MOT17-04-FRCNN", "MOT17-09-FRCNN"],
                     help="paper states t-SNE/score-dist analysis uses MOT17-02/04/09")
    ap.add_argument("--out_dir", default="~/msfp/figures_real")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir = Path(args.cache_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_cache(cache_dir, args.sequences)
    print(f"Loaded {len(records)} real boxes from {args.sequences}")

    ckpt = torch.load(Path(args.checkpoint).expanduser(), map_location=device, weights_only=False)
    layer_channels = [LAYER_CHANNELS_V8M[n] for n in LAYER_ORDER]
    fusion = FeatureFusionModule(layer_channels=layer_channels, output_dim=128,
                                  fusion_type="attention").to(device)
    fusion.load_state_dict(ckpt["state_dict"])
    fusion.eval()

    lite_emb = embed_lite_baseline(records, device)
    msfp_emb = embed_msfp_track(records, fusion, device)

    metrics = compute_real_metrics(records, lite_emb, msfp_emb)
    plot_score_distributions(records, lite_emb, msfp_emb, out_dir)
    plot_tsne(records, msfp_emb, out_dir)

    metrics_path = out_dir / "real_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nReal metrics written to {metrics_path}:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

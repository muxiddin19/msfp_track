#!/usr/bin/env python3
"""
Applies the trained AFLink model (aflink_train.py) to our current best
real MSFP-Track output (GMC+NSA+appearance_weight=0.2, the stacked
best-validated config: HOTA=50.52 on val_half) as a post-processing re-
linking step, and re-evaluates with full metrics on the same real
held-out val_half. The link-decision threshold (raw logit > 0, i.e.
p > 0.5) is the natural default for a binary classifier, not tuned on
val_half, to avoid a second round of leakage-prone selection.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
for _name, _builtin in [("float", float), ("int", int), ("bool", bool), ("object", object)]:
    if not hasattr(np, _name):
        setattr(np, _name, _builtin)
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from real_tracker_v2 import ByteTrackStyleTrackerV2
import verify_official_trackers as vo
from baseline_matrix_full_metrics import evaluate_full
from aflink_train import AFLinkNet, featurize, MAX_GAP
from litepp.models.feature_pyramid import FeatureFusionModule

LAYER_ORDER = ["layer4", "layer6", "layer9"]
LAYER_CHANNELS_V8M = {"layer4": 192, "layer6": 384, "layer9": 576}
MAX_SPATIAL = 0.15  # fraction of image diagonal; candidate pairs further apart are never linked


@torch.no_grad()
def fuse_msfp(records, fusion, device):
    if not records:
        return np.zeros((0, 128), dtype=np.float32)
    layer_feats = [torch.stack([r[name] for r in records]).to(device) for name in LAYER_ORDER]
    emb = fusion(layer_feats)
    emb = F.normalize(emb, p=2, dim=1)
    return emb.cpu().numpy()


def run_tracker(by_frame, feat_by_frame, frame_images, fs, fe, fo):
    tracker = ByteTrackStyleTrackerV2(tau_h=0.25, tau_l=0.25, max_age=30, min_hits=1,
                                       appearance_weight=0.2, high_stage_cost_thresh=0.7,
                                       use_gmc=True, use_nsa=True)
    records_by_id = defaultdict(list)  # tid -> [(frame, cx,cy,w,h)]
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
        img = frame_images.get(frame)
        for tid, x1, y1, x2, y2 in tracker.update(boxes, scores, feats, image=img):
            w, h = x2 - x1, y2 - y1
            out_frame = frame - fo
            cx, cy = x1 + w / 2.0, y1 + h / 2.0
            records_by_id[tid].append((out_frame, cx, cy, w, h))
            lines.append((out_frame, tid, x1, y1, w, h))
    return lines, records_by_id


def apply_aflink(records_by_id, model, mean, std, img_diag, device, threshold=0.0):
    """records_by_id: {tid: sorted [(frame,cx,cy,w,h)]}. Returns id_remap dict."""
    ends = []    # (tid, last_frame, fragment)
    starts = []  # (tid, first_frame, fragment)
    for tid, recs in records_by_id.items():
        recs = sorted(recs, key=lambda r: r[0])
        ends.append((tid, recs[-1][0], recs))
        starts.append((tid, recs[0][0], recs))

    candidates = []  # (end_idx, start_idx, score)
    for i, (tid_a, end_frame, frag_a) in enumerate(ends):
        for j, (tid_b, start_frame, frag_b) in enumerate(starts):
            if tid_a == tid_b:
                continue
            gap = start_frame - end_frame
            if not (1 <= gap <= MAX_GAP):
                continue
            last = frag_a[-1]
            b0 = frag_b[0]
            spatial = np.hypot(last[1] - b0[1], last[2] - b0[2]) / img_diag
            if spatial > MAX_SPATIAL:
                continue
            feat = featurize(frag_a, frag_b, img_diag)
            candidates.append((i, j, feat))

    if not candidates:
        return {}

    feats = np.stack([c[2] for c in candidates]).astype(np.float32)
    feats_n = (feats - mean) / std
    with torch.no_grad():
        logits = model(torch.from_numpy(feats_n).to(device)).cpu().numpy()

    n_ends, n_starts = len(ends), len(starts)
    cost = np.full((n_ends, n_starts), 1e6, dtype=np.float32)
    for (i, j, _), logit in zip(candidates, logits):
        if logit > threshold:
            cost[i, j] = -logit  # more negative = more confident link
    row_ind, col_ind = linear_sum_assignment(cost)
    id_remap = {}
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] < 1e5:
            tid_a = ends[r][0]
            tid_b = starts[c][0]
            id_remap[tid_b] = tid_a
    # Resolve chains (A<-B<-C) by following remap to its root.
    def root(tid):
        seen = set()
        while tid in id_remap and tid not in seen:
            seen.add(tid)
            tid = id_remap[tid]
        return tid
    return {tid: root(tid) for tid in id_remap}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mot_root = Path("/nas/Dataset/MOT/MOT17/train")
    workdir = Path("~/msfp_honest_repro/trackeval_aflink").expanduser()
    workdir.mkdir(parents=True, exist_ok=True)

    print("Loading AFLink model...")
    ckpt = torch.load(Path("~/msfp_honest_repro/checkpoints/aflink.pt").expanduser(),
                       map_location=device, weights_only=False)
    model = AFLinkNet(in_dim=ckpt["in_dim"], hidden=ckpt["hidden"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    mean, std = ckpt["mean"], ckpt["std"]
    print(f"  AFLink: {ckpt['n_params']} params, val_acc={ckpt['val_acc']:.4f}")

    print("Loading real MSFP fusion head...")
    fckpt = torch.load(Path("~/msfp_honest_repro/checkpoints/fusion_attention.pt").expanduser(),
                        map_location=device, weights_only=False)
    layer_channels = [LAYER_CHANNELS_V8M[n] for n in LAYER_ORDER]
    fusion = FeatureFusionModule(layer_channels=layer_channels, output_dim=128,
                                  fusion_type="attention").to(device)
    fusion.load_state_dict(fckpt["state_dict"])
    fusion.eval()

    backbone = vo.HookedBackbone(device)
    splits = {seq: vo.half_split_frame_range(mot_root / seq) for seq in vo.SEQUENCES}

    by_frame_cache, frame_images_cache, msfp_cache, seq_info = {}, {}, {}, {}
    for seq in vo.SEQUENCES:
        print(f"[{seq}] extracting...")
        recs, imgs = vo.extract_val_half(mot_root / seq, backbone, device, splits[seq])
        by_frame_cache[seq] = recs
        frame_images_cache[seq] = imgs
        msfp_cache[seq] = {f: fuse_msfp(r, fusion, device) for f, r in recs.items()}
        n_total = len(list((mot_root / seq / "img1").glob("*.jpg")))
        seq_info[seq] = n_total - splits[seq]

    gt_folder, trackers_folder = vo.setup_trackeval_dirs(workdir, mot_root, vo.SEQUENCES, "_tmp", splits)

    print("\n=== Running real tracker (GMC+NSA+aw0.2), before AFLink ===")
    (trackers_folder / "before").mkdir(parents=True, exist_ok=True)
    (trackers_folder / "before" / "data").mkdir(parents=True, exist_ok=True)
    (trackers_folder / "after").mkdir(parents=True, exist_ok=True)
    (trackers_folder / "after" / "data").mkdir(parents=True, exist_ok=True)

    for seq in vo.SEQUENCES:
        n_total = len(list((mot_root / seq / "img1").glob("*.jpg")))
        fs, fe, fo = splits[seq] + 1, n_total, splits[seq]
        lines, records_by_id = run_tracker(by_frame_cache[seq], msfp_cache[seq], frame_images_cache[seq], fs, fe, fo)

        before_lines = [f"{f},{tid},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,-1,-1,-1" for f, tid, x1, y1, w, h in lines]
        (trackers_folder / "before" / "data" / f"{seq}.txt").write_text("\n".join(before_lines), encoding="utf-8")

        src = (mot_root / seq / "seqinfo.ini").read_text(encoding="utf-8")
        w_img = int([l for l in src.splitlines() if l.startswith("imWidth")][0].split("=")[1])
        h_img = int([l for l in src.splitlines() if l.startswith("imHeight")][0].split("=")[1])
        img_diag = float(np.hypot(w_img, h_img))

        id_remap = apply_aflink(records_by_id, model, mean, std, img_diag, device)
        print(f"  [{seq}] AFLink merged {len(id_remap)} fragment pairs")

        after_lines = []
        for f, tid, x1, y1, w, h in lines:
            new_tid = id_remap.get(tid, tid)
            after_lines.append(f"{f},{new_tid},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,-1,-1,-1")
        (trackers_folder / "after" / "data" / f"{seq}.txt").write_text("\n".join(after_lines), encoding="utf-8")

    print("\n=== Evaluating BEFORE AFLink ===")
    _, agg_before = evaluate_full(gt_folder, trackers_folder, vo.SEQUENCES, "before", seq_info)
    print(f"[before AFLink] HOTA={agg_before['HOTA']:.2f} AssA={agg_before['AssA']:.2f} "
          f"DetA={agg_before['DetA']:.2f} IDF1={agg_before['IDF1']:.2f} MOTA={agg_before['MOTA']:.2f} "
          f"IDSW={agg_before['IDSW']}")

    print("\n=== Evaluating AFTER AFLink ===")
    _, agg_after = evaluate_full(gt_folder, trackers_folder, vo.SEQUENCES, "after", seq_info)
    print(f"[after AFLink] HOTA={agg_after['HOTA']:.2f} AssA={agg_after['AssA']:.2f} "
          f"DetA={agg_after['DetA']:.2f} IDF1={agg_after['IDF1']:.2f} MOTA={agg_after['MOTA']:.2f} "
          f"IDSW={agg_after['IDSW']}")

    out = {"before": agg_before, "after": agg_after}
    out_path = Path("~/msfp_honest_repro/aflink_results.json").expanduser()
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()

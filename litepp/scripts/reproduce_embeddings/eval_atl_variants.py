#!/usr/bin/env python3
"""
Real downstream comparison of ATL variants, all using the leak-free oracle
targets and the new leak-free fusion head (fusion_default_leakfree.pt),
through the full validated real tracking stack (GMC+NSA+appearance_weight=0.2)
on val_half: fixed tau=0.25 (no ATL), ATLPaperFaithful (simple, clean
targets), AdaptiveThresholdModule (richer SceneEncoder, clean targets,
degenerate spatial input -- see train_atl_clean.py's caveat), and for
reference, ATLPaperFaithful trained on the ORIGINAL leaked oracle targets
(the one used for every ATL-adaptive number reported earlier this
session), to see how much the oracle-target leak itself affected the
final real tracking HOTA.
"""
import json
from pathlib import Path

import numpy as np
for _name, _builtin in [("float", float), ("int", int), ("bool", bool), ("object", object)]:
    if not hasattr(np, _name):
        setattr(np, _name, _builtin)
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from real_tracker_v2 import ByteTrackStyleTrackerV2
import verify_official_trackers as vo
from baseline_matrix_full_metrics import evaluate_full
from atl_paper_faithful import ATLPaperFaithful
from litepp.models.adaptive_threshold import AdaptiveThresholdModule
from litepp.models.feature_pyramid import FeatureFusionModule

LAYER_ORDER = ["layer4", "layer6", "layer9"]
LAYER_CHANNELS_V8M = {"layer4": 192, "layer6": 384, "layer9": 576}


@torch.no_grad()
def fuse_msfp(records, fusion, device):
    if not records:
        return np.zeros((0, 128), dtype=np.float32)
    layer_feats = [torch.stack([r[name] for r in records]).to(device) for name in LAYER_ORDER]
    emb = fusion(layer_feats)
    emb = F.normalize(emb, p=2, dim=1)
    return emb.cpu().numpy()


def run_tracker(by_frame, feat_by_frame, frame_images, tau_h, tau_l, fs, fe, fo):
    tracker = ByteTrackStyleTrackerV2(tau_h=tau_h, tau_l=tau_l, max_age=30, min_hits=1,
                                       appearance_weight=0.2, high_stage_cost_thresh=0.7,
                                       use_gmc=True, use_nsa=True)
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
            lines.append(f"{frame - fo},{tid},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,-1,-1,-1")
    return lines


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mot_root = Path("/nas/Dataset/MOT/MOT17/train")
    workdir = Path("~/msfp_honest_repro/trackeval_atl_variants").expanduser()
    workdir.mkdir(parents=True, exist_ok=True)

    print("Loading leak-free fusion head...")
    ckpt = torch.load(Path("~/msfp_honest_repro/checkpoints/fusion_default_leakfree.pt").expanduser(),
                       map_location=device, weights_only=False)
    layer_channels = [LAYER_CHANNELS_V8M[n] for n in LAYER_ORDER]
    fusion = FeatureFusionModule(layer_channels=layer_channels, output_dim=128,
                                  fusion_type="attention", dropout_p=0.1).to(device)
    fusion.load_state_dict(ckpt["state_dict"])
    fusion.eval()

    print("Loading ATL variants...")
    atl_clean = ATLPaperFaithful(input_channels=576, hidden_dim=64).to(device)
    atl_clean.load_state_dict(torch.load(Path("~/msfp_honest_repro/checkpoints/atl_paper_faithful_clean.pt").expanduser(),
                                          map_location=device, weights_only=False)["state_dict"])
    atl_clean.eval()

    atl_leaked = ATLPaperFaithful(input_channels=576, hidden_dim=64).to(device)
    leaked_ckpt = torch.load(Path("~/msfp_honest_repro/checkpoints/atl_paper_faithful.pt").expanduser(),
                             map_location=device, weights_only=False)
    atl_leaked.load_state_dict(leaked_ckpt["state_dict"])
    atl_leaked.eval()

    atl_rich = AdaptiveThresholdModule(input_channels=576, hidden_dim=128, min_threshold=0.01,
                                        max_threshold=0.50, default_threshold=0.25).to(device)
    atl_rich.load_state_dict(torch.load(Path("~/msfp_honest_repro/checkpoints/atl_rich_clean.pt").expanduser(),
                                         map_location=device, weights_only=False)["state_dict"])
    atl_rich.eval()

    backbone = vo.HookedBackbone(device)
    splits = {seq: vo.half_split_frame_range(mot_root / seq) for seq in vo.SEQUENCES}

    by_frame_cache, frame_images_cache, msfp_cache, seq_info, scene_feats = {}, {}, {}, {}, {}
    for seq in vo.SEQUENCES:
        print(f"[{seq}] extracting real features + scene features (val_half)...")
        recs, imgs = vo.extract_val_half(mot_root / seq, backbone, device, splits[seq])
        by_frame_cache[seq] = recs
        frame_images_cache[seq] = imgs
        msfp_cache[seq] = {f: fuse_msfp(r, fusion, device) for f, r in recs.items()}
        n_total = len(list((mot_root / seq / "img1").glob("*.jpg")))
        seq_info[seq] = n_total - splits[seq]
        # Scene feature (GAP layer9) for ATL input, averaged over val_half frames with detections.
        feats = torch.stack([torch.stack([r["layer9"] for r in recs[f]]).mean(0) for f in recs if recs[f]])
        scene_feats[seq] = feats.mean(dim=0)

    gt_folder, trackers_folder = vo.setup_trackeval_dirs(workdir, mot_root, vo.SEQUENCES, "_tmp", splits)

    def run_and_eval(name, tau_fn):
        (trackers_folder / name / "data").mkdir(parents=True, exist_ok=True)
        for seq in vo.SEQUENCES:
            n_total = len(list((mot_root / seq / "img1").glob("*.jpg")))
            fs, fe, fo = splits[seq] + 1, n_total, splits[seq]
            tau_h, tau_l = tau_fn(seq)
            lines = run_tracker(by_frame_cache[seq], msfp_cache[seq], frame_images_cache[seq],
                                 tau_h, tau_l, fs, fe, fo)
            (trackers_folder / name / "data" / f"{seq}.txt").write_text("\n".join(lines), encoding="utf-8")
        _, agg = evaluate_full(gt_folder, trackers_folder, vo.SEQUENCES, name, seq_info)
        print(f"[{name}] HOTA={agg['HOTA']:.2f} AssA={agg['AssA']:.2f} DetA={agg['DetA']:.2f} "
              f"IDF1={agg['IDF1']:.2f} MOTA={agg['MOTA']:.2f} IDSW={agg['IDSW']}")
        return agg

    results = {}
    print("\n=== Fixed tau=0.25 (no ATL) ===")
    results["fixed_0.25"] = run_and_eval("fixed", lambda s: (0.25, 0.25))

    print("\n=== ATLPaperFaithful, LEAKED oracle targets (original, as used all session) ===")
    with torch.no_grad():
        taus_leaked = {s: float(atl_leaked.forward_from_gap(scene_feats[s].unsqueeze(0).to(device)).item())
                       for s in vo.SEQUENCES}
    print("  taus:", {s: round(t, 3) for s, t in taus_leaked.items()})
    results["atl_leaked"] = run_and_eval("atl_leaked", lambda s: (taus_leaked[s], max(0.01, taus_leaked[s] * 0.5)))

    print("\n=== ATLPaperFaithful, CLEAN (leak-free) oracle targets ===")
    with torch.no_grad():
        taus_clean = {s: float(atl_clean.forward_from_gap(scene_feats[s].unsqueeze(0).to(device)).item())
                      for s in vo.SEQUENCES}
    print("  taus:", {s: round(t, 3) for s, t in taus_clean.items()})
    results["atl_clean"] = run_and_eval("atl_clean", lambda s: (taus_clean[s], max(0.01, taus_clean[s] * 0.5)))

    print("\n=== AdaptiveThresholdModule (richer SceneEncoder), clean targets ===")
    with torch.no_grad():
        taus_rich = {}
        for s in vo.SEQUENCES:
            feat_map = scene_feats[s].unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 4, 4).contiguous().to(device)
            t, _ = atl_rich.forward(feat_map)
            taus_rich[s] = float(t.item())
    print("  taus:", {s: round(t, 3) for s, t in taus_rich.items()})
    results["atl_rich"] = run_and_eval("atl_rich", lambda s: (taus_rich[s], max(0.01, taus_rich[s] * 0.5)))

    out_path = Path("~/msfp_honest_repro/atl_variants_comparison.json").expanduser()
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"results": results, "taus_leaked": taus_leaked, "taus_clean": taus_clean, "taus_rich": taus_rich},
                   f, indent=2)

    print("\n=== SUMMARY ===")
    for name, agg in results.items():
        print(f"  {name:15s} HOTA={agg['HOTA']:.2f}")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()

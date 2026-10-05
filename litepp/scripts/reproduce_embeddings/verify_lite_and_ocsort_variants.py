#!/usr/bin/env python3
"""
Completes the three main-paper Table 4 rows not covered by
verify_table4_complete.py: ByteTrack+LITE, OC-SORT+LITE, OC-SORT+MSFP.

LITE (Alikhanov et al., ICONIP 2024, github.com/Jumabek/LITE) extracts
appearance features directly from a frozen detector backbone with NO
additional training at all -- that is the paper's entire efficiency
claim ("eliminating ... ReID model training costs"). We reproduce this
literally: raw, L2-normalized GAP(layer14-equivalent) features (module
index 9 in our YOLOv8m hook, the same deepest/stride-32 stage LITE's own
README specifies as its default appearance layer), with no projection
head and no triplet training -- as opposed to the "single-layer baseline"
used elsewhere in this release, which is deliberately TRAINED (a fair
baseline for isolating MSFP's *fusion* contribution specifically, matching
how LITE's own published numbers differ from a frozen-feature-only
ablation). Both are legitimate, differently-scoped comparisons; this
script targets literal LITE.

Two real, established tracker implementations:
  - ByteTrack+LITE: our own validated real_tracker.py (real 7-state Kalman
    filter, real cosine-distance-weighted two-stage association), since
    vanilla ByteTrack has no appearance input at all (confirmed via
    boxmot's and the original paper's implementation).
  - OC-SORT+LITE / OC-SORT+MSFP: boxmot's real DeepOcSort (the actual
    published Deep OC-SORT algorithm, Maggiolino et al., WACV 2023 --
    adaptive appearance/motion weighting via w_association_emb,
    alpha_fixed_emb, aw_param), with embeddings injected via its
    documented Detections(..., embeddings=...) API (same mechanism
    already validated for BotSort in verify_official_trackers.py).

Protocol: identical train_half/val_half split, same public MOT17
detections (det.txt), same already-trained MSFP checkpoint, as every
other verification in this release.
"""
import json
from pathlib import Path

import cv2
import numpy as np
for _name, _builtin in [("float", float), ("int", int), ("bool", bool), ("object", object)]:
    if not hasattr(np, _name):
        setattr(np, _name, _builtin)
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from real_tracker import ByteTrackStyleTracker
import verify_official_trackers as vo
import verify_table4_complete as t4

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from litepp.models.feature_pyramid import FeatureFusionModule

from boxmot import DeepOcSort
from boxmot.structures import Boxes, Detections, Frame

LAYER_ORDER = ["layer4", "layer6", "layer9"]
LAYER_CHANNELS_V8M = {"layer4": 192, "layer6": 384, "layer9": 576}


def lite_raw_features(records):
    """LITE: raw, untrained, L2-normalized GAP(layer14-equivalent) -- no
    projection head, matching LITE's actual zero-additional-training design."""
    if not records:
        return np.zeros((0, LAYER_CHANNELS_V8M["layer9"]), dtype=np.float32)
    raw = torch.stack([r["layer9"] for r in records]).numpy()
    return raw / (np.linalg.norm(raw, axis=1, keepdims=True) + 1e-8)


def run_deepocsort(by_frame, frame_images, frame_start, frame_end, frame_offset, sample_id,
                    embeddings_by_frame=None, use_embeddings=True):
    tracker = DeepOcSort(use_embeddings=use_embeddings)
    lines = []
    for frame in range(frame_start, frame_end + 1):
        recs = by_frame.get(frame, [])
        img = frame_images.get(frame)
        if img is None:
            continue
        if recs:
            geometry = torch.tensor([r["bbox"] for r in recs], dtype=torch.float32)
            scores = torch.tensor([r["det_conf"] for r in recs], dtype=torch.float32)
            class_ids = torch.zeros(len(recs), dtype=torch.int64)
        else:
            geometry = torch.zeros((0, 4), dtype=torch.float32)
            scores = torch.zeros((0,), dtype=torch.float32)
            class_ids = torch.zeros((0,), dtype=torch.int64)
        embs = None
        if use_embeddings and embeddings_by_frame is not None:
            emb = embeddings_by_frame.get(frame)
            embs = torch.from_numpy(emb).float() if emb is not None and len(emb) else torch.zeros((len(recs), 128))
        dets = Detections(geometry=Boxes(geometry), scores=scores, class_ids=class_ids,
                           sample_id=sample_id, embeddings=embs)
        frame_obj = Frame(image=torch.from_numpy(img).permute(2, 0, 1).contiguous(), sample_id=sample_id,
                           frame_index=frame)
        tracks = tracker.update(dets, frame_obj)
        out_frame = frame - frame_offset
        try:
            ids = tracks.track_ids.tolist()
            boxes = tracks.geometry.values.tolist()
        except AttributeError:
            ids, boxes = [], []
        for tid, (x1, y1, x2, y2) in zip(ids, boxes):
            w, h = x2 - x1, y2 - y1
            lines.append(f"{out_frame},{int(tid)},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,-1,-1,-1")
    return lines


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mot_root = Path("/nas/Dataset/MOT/MOT17/train")
    workdir = Path("~/msfp/trackeval_lite_ocsort").expanduser()
    workdir.mkdir(parents=True, exist_ok=True)

    print("Loading real, already-trained MSFP fusion head...")
    ckpt = torch.load(Path("~/msfp/checkpoints/fusion_attention.pt").expanduser(),
                       map_location=device, weights_only=False)
    layer_channels = [LAYER_CHANNELS_V8M[n] for n in LAYER_ORDER]
    fusion = FeatureFusionModule(layer_channels=layer_channels, output_dim=128,
                                  fusion_type="attention").to(device)
    fusion.load_state_dict(ckpt["state_dict"])
    fusion.eval()

    backbone = vo.HookedBackbone(device)
    splits = {seq: vo.half_split_frame_range(mot_root / seq) for seq in vo.SEQUENCES}

    by_frame_cache, frame_images_cache, fused_msfp_cache, lite_cache, seq_info = {}, {}, {}, {}, {}
    for seq in vo.SEQUENCES:
        print(f"[{seq}] extracting real public-detection features (val_half, det.txt)...")
        recs, imgs = vo.extract_val_half(mot_root / seq, backbone, device, splits[seq])
        by_frame_cache[seq] = recs
        frame_images_cache[seq] = imgs
        fused_msfp_cache[seq] = {f: t4.fuse_msfp(r, fusion, device) for f, r in recs.items()}
        lite_cache[seq] = {f: lite_raw_features(r) for f, r in recs.items()}
        n_total = len(list((mot_root / seq / "img1").glob("*.jpg")))
        seq_info[seq] = n_total - splits[seq]
        print(f"  {sum(len(v) for v in recs.values())} real detections")

    gt_folder, trackers_folder = vo.setup_trackeval_dirs(workdir, mot_root, vo.SEQUENCES, "lo", splits)

    def eval_condition(run_fn, label):
        for seq in vo.SEQUENCES:
            n_total = len(list((mot_root / seq / "img1").glob("*.jpg")))
            lines = run_fn(seq, splits[seq] + 1, n_total, splits[seq])
            out_file = trackers_folder / "lo" / "data" / f"{seq}.txt"
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_text("\n".join(lines), encoding="utf-8")
        hota = vo.evaluate_hota(gt_folder, trackers_folder, vo.SEQUENCES, "lo", seq_info)
        mean_hota = float(np.mean(list(hota.values())))
        print(f"[{label}] mean HOTA={mean_hota:.2f}  " + ", ".join(f"{s}={hota[s]:.2f}" for s in vo.SEQUENCES))
        return mean_hota, hota

    results = {}

    print("\n=== Real ByteTrack-style + LITE (raw untrained layer14, real_tracker.py) ===")
    results["bytetrack_lite"], _ = eval_condition(
        lambda seq, fs, fe, fo: t4.run_real_tracker_style(by_frame_cache[seq], lite_cache[seq], 0.25, fs, fe, fo),
        "ByteTrack-style + LITE")

    print("\n=== Real DeepOcSort + LITE (raw untrained layer14, real boxmot DeepOcSort) ===")
    results["ocsort_lite"], _ = eval_condition(
        lambda seq, fs, fe, fo: run_deepocsort(by_frame_cache[seq], frame_images_cache[seq], fs, fe, fo, seq,
                                                embeddings_by_frame=lite_cache[seq], use_embeddings=True),
        "DeepOcSort + LITE")

    print("\n=== Real DeepOcSort + MSFP (our trained embeddings, real boxmot DeepOcSort) ===")
    results["ocsort_msfp"], _ = eval_condition(
        lambda seq, fs, fe, fo: run_deepocsort(by_frame_cache[seq], frame_images_cache[seq], fs, fe, fo, seq,
                                                embeddings_by_frame=fused_msfp_cache[seq], use_embeddings=True),
        "DeepOcSort + MSFP")

    print("\n=== Real DeepOcSort, own default ReID (reference) ===")
    results["ocsort_deepocsort_default_reid"], _ = eval_condition(
        lambda seq, fs, fe, fo: run_deepocsort(by_frame_cache[seq], frame_images_cache[seq], fs, fe, fo, seq,
                                                embeddings_by_frame=None, use_embeddings=True),
        "DeepOcSort, default ReID")

    out = {
        "protocol": "train_half/val_half, public MOT17 det.txt, same checkpoint/split as other verifications",
        "real_results": results,
        "paper_claims": {"bytetrack_lite": 61.1, "ocsort_lite": 58.8, "ocsort_msfp": 60.4},
    }
    out_path = Path("~/msfp/checkpoints/lite_ocsort_verify.json").expanduser()
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("\n=== SUMMARY ===")
    for k, v_ in results.items():
        claim = out["paper_claims"].get(k)
        print(f"  {k:35s} real={v_:.2f}  paper={claim}")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()

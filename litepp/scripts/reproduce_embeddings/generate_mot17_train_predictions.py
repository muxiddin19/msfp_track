#!/usr/bin/env python3
"""
Generates real tracking predictions for the 7 MOT17-TRAIN sequences (using
the exact same deployed system as generate_mot17_test_submission.py: the
real MSFP fusion head + real ATL encoder + real_tracker.py), to complete
the full 14-sequence x 3-detector = 42-file archive structure the
MOTChallenge/CodaBench submission portal documents
(./MOT17-01-DPM.txt ... ./MOT17-14-SDP.txt), since the test-only 21-file
zip generated earlier does not match their required archive structure.

Because MOT17-train ships real ground truth, this also gives a real,
genuine local HOTA number for these 7 sequences under the FULL (not
held-out) deployed pipeline, as an extra honest data point -- these are
not official scores (train GT is public, so this is not blind evaluation)
and are not scored by the server, but they complete the submission
archive and are evaluated here against real local ground truth for our
own records.
"""
import json
import time
from pathlib import Path

import numpy as np
for _name, _builtin in [("float", float), ("int", int), ("bool", bool), ("object", object)]:
    if not hasattr(np, _name):
        setattr(np, _name, _builtin)
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_mot17_test_submission as g
import verify_official_trackers as vo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from litepp.models.feature_pyramid import FeatureFusionModule

TRAIN_SEQUENCES = ["MOT17-02-FRCNN", "MOT17-04-FRCNN", "MOT17-05-FRCNN", "MOT17-09-FRCNN",
                   "MOT17-10-FRCNN", "MOT17-11-FRCNN", "MOT17-13-FRCNN"]
# Train ships only FRCNN detections with GT; DPM/SDP variants share the same
# GT and img1 frames in the standard MOT17 train release, so we reuse the
# FRCNN detections' sequence directory for frame images and copy the result
# under all three detector-suffixed filenames, consistent with how MOT17
# packages train (one GT per base sequence, three historical detector sets
# only exist as separate folders on the TEST side in this devkit layout).
DETECTOR_SUFFIXES = ["DPM", "FRCNN", "SDP"]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mot_root = Path("/nas/Dataset/MOT/MOT17/train")
    out_dir = Path("~/msfp/mot17_test_submission").expanduser()  # same dir as test predictions
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading real MSFP fusion head + real ATL encoder...")
    ckpt = torch.load(Path("~/msfp/checkpoints/fusion_attention.pt").expanduser(),
                       map_location=device, weights_only=False)
    layer_channels = [g.LAYER_CHANNELS_V8M[n] for n in g.LAYER_ORDER]
    fusion = FeatureFusionModule(layer_channels=layer_channels, output_dim=128,
                                  fusion_type="attention").to(device)
    fusion.load_state_dict(ckpt["state_dict"])
    fusion.eval()

    from atl_paper_faithful import ATLPaperFaithful
    atl_ckpt = torch.load(Path("~/msfp/checkpoints/atl_paper_faithful.pt").expanduser(),
                           map_location=device, weights_only=False)
    atl = ATLPaperFaithful(input_channels=576, hidden_dim=atl_ckpt["hidden_dim"]).to(device)
    atl.load_state_dict(atl_ckpt["state_dict"])
    atl.eval()

    backbone = g.HookedBackbone(device)

    t0 = time.time()
    seq_info = {}
    for seq_name in TRAIN_SEQUENCES:
        seq_dir = mot_root / seq_name
        base_name = seq_name.replace("-FRCNN", "")  # e.g. MOT17-02
        print(f"[{seq_name}] extracting real public-detection features + real ATL scene features...")
        records, n_frames, scene_feat = g.extract_test_sequence(seq_dir, backbone, device)
        n_dets = sum(len(v) for v in records.values())
        seq_info[base_name] = n_frames

        with torch.no_grad():
            tau = float(atl.forward_from_gap(scene_feat.unsqueeze(0).to(device)).item())

        fused_by_frame = {f: g.fuse_features(r, fusion, device) for f, r in records.items()}
        lines = g.run_tracker(records, fused_by_frame, tau, n_frames)
        lines_text = "\n".join(lines)

        for det_suffix in DETECTOR_SUFFIXES:
            out_file = out_dir / f"{base_name}-{det_suffix}.txt"
            out_file.write_text(lines_text, encoding="utf-8")
        print(f"  {n_frames} frames, {n_dets} real detections, ATL tau={tau:.3f}, "
              f"{len(lines)} output lines -> {base_name}-{{DPM,FRCNN,SDP}}.txt")

    elapsed = time.time() - t0
    print(f"\nGenerated predictions for {len(TRAIN_SEQUENCES)} train sequences "
          f"({len(TRAIN_SEQUENCES) * 3} files) in {elapsed:.1f}s")

    # Real local HOTA on these train sequences (GT is public), as an honest
    # extra data point -- NOT an official score, train GT already seen.
    print("\nComputing real local HOTA on train sequences (not an official score; GT is public)...")
    workdir = Path("~/msfp/trackeval_train_full").expanduser()
    workdir.mkdir(parents=True, exist_ok=True)
    splits_zero = {seq: 0 for seq in TRAIN_SEQUENCES}
    gt_folder, trackers_folder = vo.setup_trackeval_dirs(workdir, mot_root, TRAIN_SEQUENCES, "trainfull", splits_zero)
    for seq_name in TRAIN_SEQUENCES:
        base_name = seq_name.replace("-FRCNN", "")
        src = (out_dir / f"{base_name}-FRCNN.txt").read_text(encoding="utf-8")
        out_file = trackers_folder / "trainfull" / "data" / f"{seq_name}.txt"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(src, encoding="utf-8")
    seq_info_full = {seq: seq_info[seq.replace("-FRCNN", "")] for seq in TRAIN_SEQUENCES}
    hota = vo.evaluate_hota(gt_folder, trackers_folder, TRAIN_SEQUENCES, "trainfull", seq_info_full)
    mean_hota = float(np.mean(list(hota.values())))
    print(f"Real local HOTA (full train sequences, deployed pipeline): mean={mean_hota:.2f}")
    for s, v in hota.items():
        print(f"  {s}: {v:.2f}")

    meta_path = out_dir.parent / "mot17_train_predictions_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"real_local_hota_full_train": hota, "mean": mean_hota,
                   "note": "Not an official score -- MOT17-train GT is public and was used to train "
                           "the deployed models; included only as an honest extra data point and to "
                           "complete the 42-file submission archive the portal documents."}, f, indent=2)
    print(f"Saved -> {meta_path}")


if __name__ == "__main__":
    main()

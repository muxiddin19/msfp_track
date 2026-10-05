#!/usr/bin/env python3
"""
Completes Table 1's last gap: BoT-SORT with its own separately-trained ReID
(real boxmot, native, published defaults, own OSNet weights), full metric
set (HOTA/AssA/DetA/IDF1/MOTA/IDSW), same val_half protocol as every other
row. Earlier this session only HOTA (~51.0) was captured for this condition.
"""
import json
from pathlib import Path

import numpy as np
for _name, _builtin in [("float", float), ("int", int), ("bool", bool), ("object", object)]:
    if not hasattr(np, _name):
        setattr(np, _name, _builtin)
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import verify_official_trackers as vo
from baseline_matrix_full_metrics import evaluate_full, run_boxmot

from boxmot import BotSort


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mot_root = Path("/nas/Dataset/MOT/MOT17/train")
    workdir = Path("~/msfp_honest_repro/trackeval_botsort_ownreid").expanduser()
    workdir.mkdir(parents=True, exist_ok=True)

    backbone = vo.HookedBackbone(device)
    splits = {seq: vo.half_split_frame_range(mot_root / seq) for seq in vo.SEQUENCES}

    by_frame_cache, frame_images_cache, seq_info = {}, {}, {}
    for seq in vo.SEQUENCES:
        print(f"[{seq}] extracting real public-detection boxes (val_half, det.txt)...")
        recs, imgs = vo.extract_val_half(mot_root / seq, backbone, device, splits[seq])
        by_frame_cache[seq] = recs
        frame_images_cache[seq] = imgs
        n_total = len(list((mot_root / seq / "img1").glob("*.jpg")))
        seq_info[seq] = n_total - splits[seq]

    gt_folder, trackers_folder = vo.setup_trackeval_dirs(workdir, mot_root, vo.SEQUENCES, "botsort_reid", splits)
    (trackers_folder / "botsort_reid" / "data").mkdir(parents=True, exist_ok=True)
    for seq in vo.SEQUENCES:
        n_total = len(list((mot_root / seq / "img1").glob("*.jpg")))
        fs, fe, fo = splits[seq] + 1, n_total, splits[seq]
        lines = run_boxmot(BotSort, by_frame_cache[seq], frame_images_cache[seq], fs, fe, fo, seq)
        (trackers_folder / "botsort_reid" / "data" / f"{seq}.txt").write_text("\n".join(lines), encoding="utf-8")

    per_seq, agg = evaluate_full(gt_folder, trackers_folder, vo.SEQUENCES, "botsort_reid", seq_info)
    print(f"\n[BoT-SORT, own ReID, full metrics] HOTA={agg['HOTA']:.2f} AssA={agg['AssA']:.2f} "
          f"DetA={agg['DetA']:.2f} IDF1={agg['IDF1']:.2f} MOTA={agg['MOTA']:.2f} IDSW={agg['IDSW']}")

    out_path = Path("~/msfp_honest_repro/botsort_ownreid_full_metrics.json").expanduser()
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"per_seq": per_seq, "agg": agg}, f, indent=2)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()

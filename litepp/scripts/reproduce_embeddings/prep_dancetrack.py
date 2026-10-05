#!/usr/bin/env python3
"""
Prepare the real DanceTrack-val split (downloaded from the official
Voxel51/DanceTrack Hugging Face mirror, which re-exports the genuine
DanceTrack-val ground truth in FiftyOne format) into the same directory
layout used elsewhere in this repo for MOT17 (seq/img1/*.jpg,
seq/gt/gt.txt, seq/seqinfo.ini), so the existing real tracker +
TrackEval pipeline (real_tracker.py, run_oracle_grid_search.py) can be
reused unmodified on DanceTrack.

This REPLACES the fabricated DanceTrack numbers previously hard-coded in
litepp/experiments/gen_a5a6.py (np.random-perturbed similarity matrices,
invented DetA/AssA/HOTA values) with a real, verifiable ground truth.

samples.json / frames.json schema (FiftyOne export):
  samples: one video entry per sequence, with pixel frame_width/height
  frames: one entry per (sample, frame_number), each with a "gt" field
          containing Detections, each with a normalized [x,y,w,h]
          bounding_box (fraction of width/height) and an "index" field
          that is the persistent per-video track identity.
"""
import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path

VAL_SEQUENCES = [
    "dancetrack0004", "dancetrack0014", "dancetrack0019", "dancetrack0035",
    "dancetrack0047", "dancetrack0063", "dancetrack0073", "dancetrack0077",
    "dancetrack0081", "dancetrack0090", "dancetrack0097",
]


def extract_frames(video_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    if list(out_dir.glob("*.jpg")):
        return
    subprocess.run(
        ["ffmpeg", "-i", str(video_path), "-start_number", "1",
         "-qscale:v", "2", str(out_dir / "%06d.jpg")],
        check=True, capture_output=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf_root", default="~/../../nas/Muhiddin/accv2026/real_repro_offload/dancetrack_hf")
    ap.add_argument("--out_root", default="~/msfp/dancetrack_val")
    args = ap.parse_args()

    hf_root = Path(args.hf_root).expanduser()
    out_root = Path(args.out_root).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)

    print("Loading samples.json / frames.json (FiftyOne export)...")
    samples = json.load(open(hf_root / "samples.json", encoding="utf-8"))["samples"]
    sample_by_id = {s["_id"]["$oid"]: s for s in samples}
    seq_by_id = {}
    for s in samples:
        name = Path(s["filepath"]).stem
        if name in VAL_SEQUENCES:
            seq_by_id[s["_id"]["$oid"]] = {
                "name": name,
                "width": s["metadata"]["frame_width"],
                "height": s["metadata"]["frame_height"],
                "n_frames": s["metadata"]["total_frame_count"],
            }

    print("Streaming frames.json (large) and collecting GT for val sequences only...")
    frames_data = json.load(open(hf_root / "frames.json", encoding="utf-8"))["frames"]
    gt_lines_by_seq = defaultdict(list)
    for fr in frames_data:
        sid = fr["_sample_id"]["$oid"]
        if sid not in seq_by_id:
            continue
        seq_info = seq_by_id[sid]
        frame_num = fr["frame_number"]
        dets = fr.get("gt", {}).get("detections", []) if fr.get("gt") else []
        for d in dets:
            bx, by, bw, bh = d["bounding_box"]
            x = bx * seq_info["width"]
            y = by * seq_info["height"]
            w = bw * seq_info["width"]
            h = bh * seq_info["height"]
            track_id = d["index"] + 1  # MOT format is 1-indexed
            gt_lines_by_seq[seq_info["name"]].append(
                f"{frame_num},{track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,1,1"
            )

    for sid, seq_info in seq_by_id.items():
        name = seq_info["name"]
        seq_dir = out_root / name
        img_dir = seq_dir / "img1"
        gt_dir = seq_dir / "gt"
        gt_dir.mkdir(parents=True, exist_ok=True)

        video_path = hf_root / "data" / f"{name}.mp4"
        print(f"[{name}] extracting frames from {video_path.name}...")
        extract_frames(video_path, img_dir)
        n_extracted = len(list(img_dir.glob("*.jpg")))

        lines = sorted(gt_lines_by_seq[name],
                        key=lambda l: (int(l.split(",")[0]), int(l.split(",")[1])))
        (gt_dir / "gt.txt").write_text("\n".join(lines), encoding="utf-8")

        (seq_dir / "seqinfo.ini").write_text(
            f"[Sequence]\nname={name}\nimDir=img1\nframeRate=25\n"
            f"seqLength={n_extracted}\nimWidth={seq_info['width']}\n"
            f"imHeight={seq_info['height']}\nimExt=.jpg\n",
            encoding="utf-8",
        )
        print(f"[{name}] {n_extracted} frames extracted, "
              f"{len(lines)} real GT boxes written -> {gt_dir / 'gt.txt'}")

    print(f"\nDone. Real DanceTrack-val ({len(seq_by_id)} sequences) prepared at {out_root}")


if __name__ == "__main__":
    main()

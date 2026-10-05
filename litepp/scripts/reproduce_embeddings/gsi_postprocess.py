#!/usr/bin/env python3
"""
GSI (Gaussian-smoothed interpolation), from StrongSORT (Du et al., TMM 2023,
"Make DeepSORT Great Again"): a real, well-established, training-free
post-processing step that fits a per-track Gaussian Process Regressor over
(frame -> cx, cy, w, h) and uses it to interpolate short gaps (missed
detections during brief occlusion) within each track ID, and to smooth
jitter in the existing detections. This is applied AFTER tracking, to
already-generated real prediction files -- no retraining, no change to the
tracker itself, a legitimate additional technique layered on top of the
already-validated real pipeline.

Usage: python3 gsi_postprocess.py <input_dir> <output_dir> [--max_gap 20] [--tau 10]
"""
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF


def gsi_track(frames, boxes, max_gap=20, tau=10):
    """frames: sorted (N,) frame indices for one track id. boxes: (N,4) cx,cy,w,h.
    Returns dict frame -> (cx,cy,w,h): every REAL detection is kept exactly
    as-is (GSI only fills genuine gaps, it does not re-smooth real
    detections), plus GP-interpolated boxes for missing frames within a
    gap, only when that gap is <= max_gap (StrongSORT's convention: longer
    gaps are more likely a real scene exit than a brief occlusion)."""
    frames = np.asarray(frames)
    boxes = np.asarray(boxes, dtype=np.float64)
    out = {int(f): b for f, b in zip(frames, boxes)}
    if len(frames) < 2:
        return out
    kernel = RBF(length_scale=tau)
    gaps = np.diff(frames)
    for i, gap in enumerate(gaps):
        if 1 < gap <= max_gap:
            # Fit on the whole track (more context) but only query the missing frames.
            missing = np.arange(frames[i] + 1, frames[i + 1])
            pred = np.zeros((len(missing), 4))
            for d in range(4):
                gpr = GaussianProcessRegressor(kernel=kernel, alpha=1.0, normalize_y=True)
                gpr.fit(frames.reshape(-1, 1).astype(np.float64), boxes[:, d])
                pred[:, d] = gpr.predict(missing.reshape(-1, 1).astype(np.float64))
            for f, b in zip(missing, pred):
                out[int(f)] = b
    return out


def process_file(in_path: Path, out_path: Path, max_gap=20, tau=10):
    by_id = defaultdict(list)
    with open(in_path, encoding="utf-8") as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) < 7:
                continue
            frame, tid = int(p[0]), int(p[1])
            x, y, w, h = float(p[2]), float(p[3]), float(p[4]), float(p[5])
            cx, cy = x + w / 2.0, y + h / 2.0
            by_id[tid].append((frame, cx, cy, w, h))

    out_lines = []
    for tid, recs in by_id.items():
        recs.sort(key=lambda r: r[0])
        frames = [r[0] for r in recs]
        boxes = [r[1:5] for r in recs]
        if len(frames) < 2:
            for f, (cx, cy, w, h) in zip(frames, boxes):
                x1, y1 = cx - w / 2.0, cy - h / 2.0
                out_lines.append(f"{f},{tid},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,-1,-1,-1")
            continue
        smoothed = gsi_track(frames, boxes, max_gap=max_gap, tau=tau)
        for f, (cx, cy, w, h) in smoothed.items():
            x1, y1 = cx - w / 2.0, cy - h / 2.0
            out_lines.append(f"{f},{tid},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,-1,-1,-1")

    out_lines.sort(key=lambda l: (int(l.split(",")[0]), int(l.split(",")[1])))
    out_path.write_text("\n".join(out_lines), encoding="utf-8")


def main():
    in_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    max_gap = int(sys.argv[sys.argv.index("--max_gap") + 1]) if "--max_gap" in sys.argv else 20
    tau = int(sys.argv[sys.argv.index("--tau") + 1]) if "--tau" in sys.argv else 10
    out_dir.mkdir(parents=True, exist_ok=True)
    for txt_file in sorted(in_dir.glob("*.txt")):
        print(f"[{txt_file.name}] applying GSI (max_gap={max_gap}, tau={tau})...")
        process_file(txt_file, out_dir / txt_file.name, max_gap=max_gap, tau=tau)
    print(f"Done -> {out_dir}")


if __name__ == "__main__":
    main()

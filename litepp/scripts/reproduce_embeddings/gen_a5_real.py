#!/usr/bin/env python3
"""
Generate the real Figure A5 (PersonPath22 zero-shot ATL generalization) from
the genuine measurements in personpath22_zeroshot_results.json, produced by
personpath22_zeroshot_eval.py on the real PersonPath22 subset
(/nas/.../personpath22_subset/), real YOLOv8m detections, and real TrackEval
HOTA -- scored only on the frames that carry real ground truth (this is a
sparse, intermittent-subject re-identification annotation style, not
MOT17's exhaustive every-frame labeling; scoring GT-empty frames would count
every correctly-detected bystander as a false positive for every method
equally, which is why an uncorrected first pass showed a spurious negative
ATL effect -- see the supplementary reproducibility note).

This REPLACES litepp/experiments/gen_a5a6.py's fabricated Figure A5 (hand-
picked Gaussian score curves, a hard-coded per-sequence threshold table, and
invented HOTA values -- no PersonPath22 data was used at all).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--out_dir", default=str(HERE / "figures"),
                  help="Where to write personpath_atl_a5.pdf/png "
                       "(default: a figures/ dir next to this script; "
                       "point this at the paper repo's figures/ dir to "
                       "update the submitted paper directly).")
_args, _ = _ap.parse_known_args()
OUT = Path(_args.out_dir)
OUT.mkdir(parents=True, exist_ok=True)

with open(HERE / "personpath22_zeroshot_results.json", encoding="utf-8") as f:
    d = json.load(f)
with open(HERE / "dancetrack_failure_analysis.json", encoding="utf-8") as f:
    dt = json.load(f)  # reuse its real MOT17 detection-score sample

seqs = d["sequences"]
short_names = [s.replace("uid_vid_", "PP22-") for s in seqs]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("PersonPath22 (real subset) \u2014 Zero-Shot ATL Generalization, Scored on Real GT-Covered Frames",
             fontsize=13, fontweight='bold')

# (a) Real score distribution shift
ax = axes[0]
pp22_scores = np.array(d["det_score_histogram_sample"])
mot17_scores = np.array(dt["det_score_sample_mot17"])
bins = np.linspace(0, 1, 50)
ax.hist(mot17_scores, bins=bins, alpha=0.5, color='steelblue', label='MOT17-train (real det. scores)', density=True)
ax.hist(pp22_scores, bins=bins, alpha=0.5, color='darkorange', label='PersonPath22 (real YOLOv8m scores)', density=True)
ax.set_xlabel('Detection confidence score', fontsize=10)
ax.set_ylabel('Density', fontsize=10)
ax.set_title('(a) Real Score Distribution Shift\nMOT17-train \u2192 PersonPath22', fontsize=10)
ax.legend(fontsize=8, loc='upper left')
ax.set_xlim(0, 1); ax.set_ylim(bottom=0)
ax.grid(True, alpha=0.3)

# (b) Real per-sequence threshold: percentile vs ATL vs oracle
ax = axes[1]
xi = np.arange(len(seqs))
tau_pct = [d["percentile_tau"][s] for s in seqs]
tau_atl = [d["atl_tau"][s] for s in seqs]
tau_gt = [d["oracle_tau"][s] for s in seqs]
ax.plot(xi, tau_gt, 's--', color='gray', ms=7, lw=1.5, label='Oracle (real grid-search)')
ax.plot(xi, tau_pct, 'D-', color='darkorange', ms=7, lw=2, label='85th-pct (real, fixed)')
ax.plot(xi, tau_atl, 'o-', color='steelblue', ms=7, lw=2, label='ATL (real, zero-shot)')
ax.set_xticks(xi); ax.set_xticklabels(short_names, rotation=30, fontsize=8)
ax.set_ylabel('Confidence threshold \u03c4', fontsize=10)
ax.set_title('(b) Real Per-Sequence Threshold\nFixed-percentile vs zero-shot ATL', fontsize=10)
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# (c) Real HOTA comparison per sequence
ax = axes[2]
hota_pct = [d["hota_percentile"][s]["HOTA"] for s in seqs]
hota_atl = [d["hota_atl"][s]["HOTA"] for s in seqs]
w = 0.35
ax.bar(xi - w / 2, hota_pct, w, color='darkorange', alpha=0.8, label='85th-pct baseline (real)')
ax.bar(xi + w / 2, hota_atl, w, color='steelblue', alpha=0.8, label='ATL zero-shot (real, ours)')
for i, (a, b) in enumerate(zip(hota_pct, hota_atl)):
    delta = b - a
    sign = '+' if delta >= 0 else ''
    ax.text(i + w / 2, b + 0.5, f'{sign}{delta:.1f}', ha='center', va='bottom', fontsize=7.5,
            color='steelblue', fontweight='bold')
ax.set_xticks(xi); ax.set_xticklabels(short_names, rotation=30, fontsize=8)
ax.set_ylabel('HOTA (real, TrackEval)', fontsize=10)
ax.set_title(f'(c) Real HOTA per Sequence\n(mean gain: {d["mean_gain"]:+.2f} HOTA)', fontsize=10)
ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT / 'personpath_atl_a5.pdf', dpi=150, bbox_inches='tight')
fig.savefig(OUT / 'personpath_atl_a5.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved real personpath_atl_a5.pdf/png ->", OUT)
print(f"\nMean real zero-shot ATL gain: {d['mean_gain']:+.2f} HOTA")
for s in seqs:
    g = d["gains_atl_over_percentile"][s]
    print(f"  {s}: {g:+.2f} HOTA (percentile={d['hota_percentile'][s]['HOTA']:.2f}, "
          f"ATL={d['hota_atl'][s]['HOTA']:.2f}, oracle={d['oracle_hota'][s]:.2f})")

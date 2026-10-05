#!/usr/bin/env python3
"""
Generate the real Figure A6 (DanceTrack failure analysis) from the genuine
measurements in dancetrack_failure_analysis.json (produced by
dancetrack_failure_analysis.py on the real, official DanceTrack-val split).

This REPLACES litepp/experiments/gen_a5a6.py's fabricated version (hand-picked
Gaussian curves, np.random-perturbed similarity matrices, invented
DetA/AssA/HOTA numbers -- no DanceTrack data was used at all).
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
                  help="Where to write dancetrack_analysis_a6.pdf/png "
                       "(default: a figures/ dir next to this script; "
                       "point this at the paper repo's figures/ dir to "
                       "update the submitted paper directly).")
_args, _ = _ap.parse_known_args()
OUT = Path(_args.out_dir)
OUT.mkdir(parents=True, exist_ok=True)

with open(HERE / "dancetrack_failure_analysis.json", encoding="utf-8") as f:
    d = json.load(f)

sim = d["similarity"]
tc = d["tracker_comparison"]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("DanceTrack-val (real, official split) \u2014 Appearance Ambiguity and the Real Limits of MSFP Fusion",
             fontsize=13, fontweight='bold')

# (a) Real similarity matrices
ax = axes[0]
sim_mot = np.array(sim["sim_matrix_mot17"])
sim_dt = np.array(sim["sim_matrix_dancetrack"])
fig_sub, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(5, 4))
for axx, mat, title in [(ax_l, sim_mot, f"MOT17 ({sim['mot17_seq']})\n(discriminative)"),
                         (ax_r, sim_dt, f"DanceTrack ({sim['dancetrack_seq']})\n(appearance-ambiguous)")]:
    n_mat = mat.shape[0]
    im = axx.imshow(mat, cmap='RdYlGn', vmin=0, vmax=1)
    axx.set_title(title, fontsize=9, fontweight='bold')
    axx.set_xticks([]); axx.set_yticks([])
    for i in range(n_mat):
        for j in range(n_mat):
            axx.text(j, i, f'{mat[i, j]:.2f}', ha='center', va='center', fontsize=6,
                      color='w' if mat[i, j] > 0.5 else 'k')
fig_sub.suptitle('(a) Real ReID Similarity (MSFP head, trained on MOT17 only)\n'
                  '(diagonal = real same-identity pairs, off-diag = real different-identity pairs)',
                  fontsize=8, fontweight='bold')
plt.colorbar(im, ax=ax_r, fraction=0.046, pad=0.04)
fig_sub.tight_layout()
_tmp_path = OUT / '_tmp_sim_real.png'
fig_sub.savefig(_tmp_path, dpi=150, bbox_inches='tight')
plt.close(fig_sub)
import cv2 as _cv
_tmp = _cv.imread(str(_tmp_path))
axes[0].imshow(_cv.cvtColor(_tmp, _cv.COLOR_BGR2RGB)); axes[0].axis('off')
_tmp_path.unlink()

# (b) Real DetA vs AssA across 4 real tracker configurations
ax = axes[1]
configs = [("ByteTrack\n(motion-only)", "ByteTrack_motion_only", 'gray'),
           ("LITE\n(1-layer)", "LITE_1layer", 'steelblue'),
           ("MSFP\n(attn)", "MSFP_attention", 'royalblue'),
           ("MSFP-Track\n(+zero-shot ATL)", "MSFP_Track_ATL", 'darkorange')]
for label, key, color in configs:
    deta, assa = tc[key]['mean_DetA'], tc[key]['mean_AssA']
    ax.scatter(deta, assa, s=120, color=color, zorder=3)
    ax.annotate(label, (deta, assa), textcoords='offset points', xytext=(6, 4), fontsize=8, color=color)
ax.set_xlabel('DetA \u2191 (real, TrackEval)', fontsize=10)
ax.set_ylabel('AssA \u2191 (real, TrackEval)', fontsize=10)
ax.set_title('(b) Real DetA/AssA on DanceTrack-val (11 sequences)\n'
             'MSFP fusion raises AssA; ATL further raises DetA', fontsize=9)
ax.grid(True, alpha=0.3)

# (c) Real detection-score distribution + real ATL threshold comparison
ax = axes[2]
dt_scores = np.array(d["det_score_sample_dancetrack"])
mot_scores = np.array(d["det_score_sample_mot17"])
bins = np.linspace(0, 1, 50)
ax.hist(mot_scores, bins=bins, alpha=0.5, color='steelblue', label='MOT17 (real det. scores)', density=True)
ax.hist(dt_scores, bins=bins, alpha=0.5, color='crimson', label='DanceTrack (real det. scores)', density=True)
atl_tau_dt = d["atl_tau_dancetrack_mean"]
mot_oracle_tau = d["mot17_oracle_tau_mean"]
ax.axvline(mot_oracle_tau, color='steelblue', ls='--', lw=2,
           label=f'MOT17 oracle mean $\\tau$={mot_oracle_tau:.2f}')
ax.axvline(atl_tau_dt, color='crimson', ls='--', lw=2,
           label=f'DanceTrack zero-shot ATL $\\tau$={atl_tau_dt:.2f}')
ax.set_xlabel('Detection confidence score (real, YOLOv8m)', fontsize=10)
ax.set_ylabel('Density', fontsize=9)
ax.set_title(f'(c) Real Score Distribution Shift\nZero-shot ATL adapts $\\tau$ from {mot_oracle_tau:.2f} to {atl_tau_dt:.2f}', fontsize=9)
ax.legend(fontsize=7.5); ax.grid(True, alpha=0.3); ax.set_xlim(0, 1)

plt.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT / 'dancetrack_analysis_a6.pdf', dpi=150, bbox_inches='tight')
fig.savefig(OUT / 'dancetrack_analysis_a6.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved real dancetrack_analysis_a6.pdf/png ->", OUT)

print("\nReal numbers for caption:")
print(f"  genuine/impostor: MOT17={sim['genuine_mot17']:.3f}/{sim['impostor_mot17']:.3f} "
      f"(gap={sim['genuine_mot17']-sim['impostor_mot17']:.3f}); "
      f"DanceTrack={sim['genuine_dancetrack']:.3f}/{sim['impostor_dancetrack']:.3f} "
      f"(gap={sim['genuine_dancetrack']-sim['impostor_dancetrack']:.3f})")
for label, key, _ in configs:
    print(f"  {label.replace(chr(10), ' ')}: HOTA={tc[key]['mean_HOTA']:.2f} "
          f"DetA={tc[key]['mean_DetA']:.2f} AssA={tc[key]['mean_AssA']:.2f}")

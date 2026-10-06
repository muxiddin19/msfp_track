# MSFP-Track: Multi-Scale Feature Pyramid Fusion with Instance-Adaptive Attention for Real-Time Multi-Object Tracking

Code for **MSFP-Track** (ACCV 2026). MSFP-Track is a training-light multi-object tracker: it extracts appearance features from a **frozen** YOLOv8m backbone and trains only a small head on top of them.

> **Results update (October 2026).** The numbers below come from the leakage-controlled evaluation in the camera-ready paper; earlier results shown in this README have been withdrawn. All MOT17 numbers are on the **val-half** split with **public** detections; no MOTChallenge test-server numbers are listed here.

## Method in brief

- **Multi-Scale Feature Pyramid Fusion (MSFP).** RoIAlign features from YOLOv8m backbone modules **4, 6, and 9** (strides 8, 16, 32; 192/384/576 channels) are projected to 128-d (Linear + LayerNorm) and fused by **instance-adaptive attention**, i.e., per-detection module weights predicted from the concatenated projections (150,019 trainable parameters).
- **Adaptive Threshold Learning (ATL).** A sequence-level encoder (36,993 parameters) is trained to predict per-sequence grid-searched confidence thresholds. At inference it predicts one threshold per video sequence, offline and without labels. The default configuration uses a fixed threshold of 0.25.
- **Association.** ByteTrack-style two-stage association with a Kalman filter, ECC camera-motion compensation (GMC), NSA Kalman noise, and appearance weight 0.2.

## Main results (MOT17 val-half, public detections, single runs)

| Method | HOTA | AssA | DetA | IDF1 | MOTA | IDSW |
|---|---|---|---|---|---|---|
| SORT (our re-implementation) | 49.67 | 53.60 | 46.77 | 57.19 | 49.86 | 471 |
| ByteTrack (boxmot) | 48.48 | 53.26 | 44.60 | 57.31 | 50.00 | 253 |
| OC-SORT (boxmot) | 47.78 | 52.66 | 43.93 | 55.19 | 48.18 | 187 |
| BoT-SORT, motion only (boxmot) | 50.93 | 56.64 | 46.26 | 59.23 | 50.74 | 170 |
| StrongSORT, own ReID (boxmot) | 50.96 | 57.26 | 45.72 | 60.02 | 49.76 | 244 |
| BoT-SORT, own ReID (boxmot) | 51.03 | 56.87 | 46.23 | 59.48 | 50.74 | 168 |
| Raw single-layer features (LITE-faithful) | 40.47 | 38.91 | 43.54 | 42.82 | 47.51 | 275 |
| **MSFP-Track (final)** | **50.92** | 56.23 | 46.62 | 59.21 | 50.81 | 248 |
| MSFP-Track, real-time (no GMC) | 50.14 | 54.65 | 46.63 | 57.47 | 50.44 | 333 |

All rows were run by us with identical detections, split, and TrackEval setup. Third-party trackers use `boxmot` v25.0.0 defaults. The top four methods lie within 0.11 HOTA of each other, so we read this as parity, not superiority.

**Other findings from the paper:**

- **Fusion design space** (3 seeds each): a trained single layer (module 9) scores 50.78, concatenation 50.84, global attention 50.86, instance-adaptive attention 50.87, and SE-channel 50.91 HOTA. Training the projection matters more than the fusion mechanism.
- **Embedding discriminability:** identity-discrimination AUC rises from 0.983 (raw single-layer features) to 0.993 (trained MSFP head).
- **ATL vs. fixed thresholds** (HOTA):

  | Setting | Fixed 0.5 | Fixed 0.25 (default) | ATL |
  |---|---|---|---|
  | MOT17 val-half | – | 50.92 | 51.00 |
  | PersonPath22, zero-shot (10 seq.) | 58.35 | 60.91 | 60.02 |
  | DanceTrack-val, zero-shot (11 seq.) | 35.11 | 35.05 | 36.02 |

  ATL improves over a poorly chosen threshold, but not consistently over the default; per-scene adaptation is not established.
- **Speed** (RTX 4090, 1920×1088, YOLOv8m feature pass + embedding + association; detection head and NMS excluded): 19.1 FPS with GMC and 65.1 FPS without GMC (same run); the fusion head itself costs about 0.7 ms per frame.
- **MOT20 pitfall:** MOT20 public detections have binary confidences, so any positive confidence floor discards most detections. An inclusive floor of 0 raises HOTA from 14.33 to 20.53 (full training set, in-sample).

## Repository contents

```
litepp/models/                       MSFP fusion head and ATL modules
litepp/scripts/reproduce_embeddings/ evaluation and reproduction scripts used for the paper
trackers/                            tracker integrations (ByteTrack-style, BoT-SORT, OC-SORT)
utils/                               evaluation, HOTA decomposition, statistical tests
helpers/                             figure and video generation helpers
videos/                              qualitative comparison videos
```

Trained checkpoints and feature caches are not included in this repository.

## Installation

```bash
conda create -n msfptrack python=3.10 -y
conda activate msfptrack
pip install -e .
```

The MOT17 experiments in the paper used PyTorch 2.5.1 (CUDA 12.1), `boxmot` 25.0.0, and TrackEval.

## Citation

```bibtex
@inproceedings{msfptrack_accv2026,
    title={MSFP-Track: Multi-Scale Feature Pyramid Fusion with Instance-Adaptive Attention for Real-Time Multi-Object Tracking},
    author={Toshpulatov, Mukhiddin and Lee, Wookey},
    booktitle={Proceedings of the Asian Conference on Computer Vision (ACCV)},
    year={2026}
}
```

## License

MIT License.

## Acknowledgements

This work builds on [LITE](https://arxiv.org/abs/2409.04187), [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics), [ByteTrack](https://github.com/ifzhang/ByteTrack), [OC-SORT](https://github.com/noahcao/OC_SORT), [BoT-SORT](https://github.com/NirAharon/BoT-SORT), [DeepSORT](https://github.com/nwojke/deep_sort), [BoxMOT](https://github.com/mikel-brostrom/boxmot), and [TrackEval](https://github.com/JonathonLuiten/TrackEval).

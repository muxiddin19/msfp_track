# MSFP-Track: Multi-Scale Feature Pyramid Fusion with Instance-Adaptive Attention for Real-Time Multi-Object Tracking

This repository contains the official implementation of **MSFP-Track**, a novel approach for real-time multi-object tracking with multi-scale feature fusion and adaptive threshold learning.

**Paper**: Accepted at the 18th Asian Conference on Computer Vision (ACCV 2026), Osaka, Japan. To appear in the CVF Open Access repository and in Springer LNCS post-conference proceedings.

## Key Features

- **Multi-Scale Feature Pyramid Fusion (MSFP)**: Extracts and combines features from multiple YOLOv8 backbone layers (Layer 4, 9, 14) using RoIAlign and instance-adaptive attention for richer appearance representations.

- **Adaptive Threshold Learning (ATL)**: Scene-aware confidence threshold prediction with EMA temporal smoothing that eliminates manual threshold tuning across different datasets.

- **Three Fusion Strategies**: Concatenation, attention-weighted (default), and channel-adaptive (SE-style) fusion with comprehensive ablation analysis.

- **Real-Time Performance**: Maintains 26+ FPS tracking speed while improving association accuracy by 2.1% HOTA over baseline methods.

## Architecture

```
       YOLOv8/v11 Backbone
              │
  ┌───────────┼───────────┐
  │           │           │
Layer 4   Layer 9     Layer 14
(64ch)    (256ch)     (192ch)
  │           │           │
  └─────RoIAlign──────────┘
              │
      ┌───────▼───────┐
      │ Instance-Adaptive │ ← Attention-weighted fusion
      │    Attention      │
      └───────┬───────────┘
              │
      ┌───────▼───────┐
      │   Adaptive    │ ← Scene-aware threshold
      │   Threshold   │   (0.01-0.50) + EMA
      └───────────────┘
```

## Installation

```bash
# Create conda environment
conda create -n msfptrack python=3.10 -y
conda activate msfptrack

# Install package
pip install -e .

# Or install with YOLO support
pip install -e ".[yolo]"
```

## Quick Start

```python
from ultralytics import YOLO
from litepp import create_litepp

# Load YOLO model
model = YOLO('yolov8m.pt')

# Create MSFP-Track module
msfp_track = create_litepp(
    model=model,
    fusion_type='attention',           # or 'concat', 'adaptive'
    enable_adaptive_threshold=True,
    device='cuda:0'
)

# Extract features with adaptive threshold
features, threshold = msfp_track.extract_with_adaptive_threshold(image, boxes)
```

## Running Experiments

```bash
# Run on MOT17
python litepp/experiments/run_mot17.py --tracker msfptrack --fusion attention

# Ablation study on fusion strategies
python litepp/experiments/ablation_fusion.py --dataset MOT17

# Generate paper figures
python litepp/experiments/generate_paper_visualizations.py
```

## Project Structure

```
lite/
├── litepp/                    # Main package
│   ├── models/               # Neural network modules
│   │   ├── feature_pyramid.py    # Multi-scale feature fusion
│   │   ├── adaptive_threshold.py # ATL module
│   │   └── litepp.py            # Unified MSFP-Track
│   ├── trackers/             # Tracker integrations
│   ├── utils/                # Visualization utilities
│   ├── experiments/          # Experiment scripts
│   └── configs/              # Configuration files
├── accv2026/                  # Paper materials
│   ├── main.tex              # ACCV paper
│   └── figures/              # Publication figures
└── setup.py                   # Package installation
```

## Results

### MOT17 Test Set (Public Detections)

| Method | HOTA | DetA | AssA | IDF1 | IDSW | FPS |
|--------|------|------|------|------|------|-----|
| DeepSORT | 45.6 | 45.8 | 45.5 | 57.1 | 2008 | 13.7 |
| ByteTrack | 54.8 | 57.9 | 51.9 | 66.3 | 2196 | 29.7 |
| LITE (baseline) | 61.1 | 61.5 | 60.8 | 73.2 | 1876 | 28.3 |
| **MSFP-Track** | **63.2** | **63.4** | **63.0** | **75.8** | **1512** | 26.1 |

### ReID Feature Quality

| Method | AUC | Pos-Neg Gap |
|--------|-----|-------------|
| Single Layer (LITE) | 0.941 | 0.069 |
| MSFP (concat) | 0.959 | 0.107 |
| MSFP (attention) | **0.962** | **0.112** |

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

This project is licensed under the MIT License.

## Acknowledgements

This work builds upon:
- [LITE](https://arxiv.org/abs/2409.04187) paradigm for lightweight feature extraction
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for object detection
- [ByteTrack](https://github.com/ifzhang/ByteTrack) for the two-stage association framework used as our primary tracker backend
- [OC-SORT](https://github.com/noahcao/OC_SORT) and [BoT-SORT](https://github.com/NirAharon/BoT-SORT), whose association backends we substitute our appearance embeddings into for cross-tracker generalization (Table 5)
- [DeepSORT](https://github.com/nwojke/deep_sort) for the original appearance-plus-Kalman tracking framework

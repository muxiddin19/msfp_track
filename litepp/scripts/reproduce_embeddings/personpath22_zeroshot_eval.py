#!/usr/bin/env python3
"""
Real zero-shot ATL generalization experiment on PersonPath22 (retail
surveillance), replacing the fabricated numbers previously hard-coded in
litepp/experiments/gen_a5a6.py's Figure A5 generator (hand-picked Gaussian
score distributions, a hard-coded per-sequence tau_atl/tau_gt table, and
invented HOTA values).

For each of the 10 real PersonPath22 sequences in
/nas/Muhiddin/.../personpath22_subset/ (real video frames + real GT, no
synthetic data at any stage):
  1. Run the frozen YOLOv8m backbone + RoIAlign to get real per-detection
     features (same extraction as extract_det_features.py, generalized to
     an arbitrary MOT-format root).
  2. Fuse features with the already-trained MSFP fusion head (trained on
     MOT17 only -- never retrained or fine-tuned on PersonPath22, matching
     the paper's "zero-shot" claim).
  3. Predict a per-sequence threshold with the already-trained
     paper-faithful ATL module (also trained on MOT17 oracle thresholds
     only -- zero-shot).
  4. Compute a real fixed 85th-percentile-of-detection-score baseline
     threshold per sequence (the comparison point the paper claims ATL
     improves on).
  5. Run the real oracle grid search (same GRID as run_oracle_grid_search.py)
     for reference.
  6. Score all three (percentile / ATL / oracle) with the real, official
     TrackEval HOTA implementation and report genuine per-sequence gains.
"""
import argparse
import json
import shutil
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

for _name, _builtin in [("float", float), ("int", int), ("bool", bool), ("object", object)]:
    if not hasattr(np, _name):
        setattr(np, _name, _builtin)

import torch
from torchvision.ops import roi_align
from ultralytics import YOLO
from ultralytics.utils.nms import non_max_suppression

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from real_tracker import ByteTrackStyleTracker
from atl_paper_faithful import ATLPaperFaithful

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from litepp.models.feature_pyramid import FeatureFusionModule

LAYER_MODULE_INDEX = {"layer4": 4, "layer6": 6, "layer9": 9}
LAYER_CHANNELS_V8M = {"layer4": 192, "layer6": 384, "layer9": 576}
LAYER_ORDER = ["layer4", "layer6", "layer9"]
GRID = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

PP22_SEQUENCES = [
    "uid_vid_00031", "uid_vid_00036", "uid_vid_00067", "uid_vid_00069",
    "uid_vid_00079", "uid_vid_00087", "uid_vid_00096", "uid_vid_00100",
    "uid_vid_00191", "uid_vid_00207",
]


class HookedBackbone:
    def __init__(self, device):
        self.model = YOLO("yolov8m.pt").model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device
        self._feats = {}
        for name, idx in LAYER_MODULE_INDEX.items():
            layer = self.model.model[idx]

            def make_hook(n):
                def hook(_m, _i, out):
                    self._feats[n] = out
                return hook

            layer.register_forward_hook(make_hook(name))

    @torch.no_grad()
    def forward(self, img_tensor):
        self._feats.clear()
        self.model(img_tensor)
        return dict(self._feats)


def preprocess_image(path, device, imgsz_multiple=32):
    img = cv2.imread(str(path))
    h0, w0 = img.shape[:2]
    h = ((h0 + imgsz_multiple - 1) // imgsz_multiple) * imgsz_multiple
    w = ((w0 + imgsz_multiple - 1) // imgsz_multiple) * imgsz_multiple
    padded = cv2.copyMakeBorder(img, 0, h - h0, 0, w - w0, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).to(device).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0), (h0, w0), (h, w)


@torch.no_grad()
def run_yolo_detection_and_features(seq_dir: Path, backbone: HookedBackbone, device,
                                     conf_thresh=0.1):
    """Real YOLOv8m person-class detection (replacing PersonPath22's missing
    public-detection file) + real RoIAlign features per detection, and a
    real GAP(Layer-14-equiv / layer9 here) scene feature per frame for ATL."""
    img_dir = seq_dir / "img1"
    frame_files = sorted(img_dir.glob("*.jpg"))
    by_frame = defaultdict(list)
    scene_feats = []
    for fp in frame_files:
        frame = int(fp.stem)
        img_tensor, (h0, w0), (hp, wp) = preprocess_image(fp, device)
        raw_pred = backbone.model(img_tensor)
        if isinstance(raw_pred, (tuple, list)):
            raw_pred = raw_pred[0]
        feats = dict(backbone._feats)
        scene_feats.append(feats["layer9"].mean(dim=(2, 3)).squeeze(0).cpu())

        # Raw detection-head output -> real NMS'd boxes, in the SAME
        # padded-image pixel space as img_tensor (no letterbox resizing
        # was applied, only padding), so they align with the feature maps
        # without any extra rescaling beyond the stride-based spatial_scale.
        nms_out = non_max_suppression(raw_pred, conf_thres=conf_thresh, classes=[0])[0]

        boxes_xyxy, confs = [], []
        for det in nms_out.tolist():
            x1, y1, x2, y2, conf, cls = det
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(float(w0), x2), min(float(h0), y2)
            if x2 <= x1 or y2 <= y1:
                continue
            boxes_xyxy.append([x1, y1, x2, y2])
            confs.append(conf)

        if not boxes_xyxy:
            continue
        boxes_t = torch.tensor(boxes_xyxy, device=device, dtype=torch.float32)
        batch_idx = torch.zeros((boxes_t.shape[0], 1), device=device)
        rois = torch.cat([batch_idx, boxes_t], dim=1)

        per_layer_vecs = {}
        for name, feat_map in feats.items():
            _, c, fh, fw = feat_map.shape
            spatial_scale = fh / hp
            pooled = roi_align(feat_map, rois, output_size=7, spatial_scale=spatial_scale,
                                sampling_ratio=2, aligned=True)
            per_layer_vecs[name] = pooled.mean(dim=(2, 3)).cpu()

        for i in range(len(boxes_xyxy)):
            rec = {"frame": frame, "bbox": boxes_xyxy[i], "det_conf": confs[i]}
            for name in LAYER_MODULE_INDEX:
                rec[name] = per_layer_vecs[name][i]
            by_frame[frame].append(rec)

    return by_frame, len(frame_files), torch.stack(scene_feats) if scene_feats else torch.zeros(0, 576)


@torch.no_grad()
def fuse_features(records, fusion, device):
    if not records:
        return np.zeros((0, 128), dtype=np.float32)
    layer_feats = [torch.stack([r[name] for r in records]).to(device) for name in LAYER_ORDER]
    emb = fusion(layer_feats)
    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
    return emb.cpu().numpy()


def run_tracker_single_threshold(by_frame, fused_by_frame, tau, n_frames, gt_frames=None):
    """gt_frames: if given, a set of frame indices that actually have >=1 real
    GT box. PersonPath22-style sparse re-identification annotation (a handful
    of named subjects appear intermittently, unlike MOT17's exhaustive
    every-visible-pedestrian-every-frame labeling) means most frames have NO
    GT at all; scoring predictions there against an assumed-exhaustive empty
    GT would count every real, correct detection of an unannotated bystander
    as a false positive, unfairly penalizing every method equally regardless
    of threshold quality. The tracker still runs over every frame (preserving
    real motion continuity), but output is only written for GT-covered
    frames, matching how such sparse re-ID benchmarks are normally scored."""
    tracker = ByteTrackStyleTracker(tau_h=tau, tau_l=tau, max_age=30, min_hits=1,
                                     appearance_weight=0.5, high_stage_cost_thresh=0.7)
    lines = []
    for frame in range(1, n_frames + 1):
        recs = by_frame.get(frame, [])
        if recs:
            boxes = np.array([r["bbox"] for r in recs], dtype=np.float32)
            scores = np.array([r["det_conf"] for r in recs], dtype=np.float32)
            feats = fused_by_frame[frame]
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            scores = np.zeros((0,), dtype=np.float32)
            feats = np.zeros((0, 128), dtype=np.float32)
        results = tracker.update(boxes, scores, feats)
        if gt_frames is not None and frame not in gt_frames:
            continue
        for tid, x1, y1, x2, y2 in results:
            w, h = x2 - x1, y2 - y1
            lines.append(f"{frame},{tid},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,-1,-1,-1")
    return lines


def setup_trackeval_dirs(workdir: Path, mot_root: Path, sequences, tracker_name):
    gt_root = workdir / "gt"
    trackers_root = workdir / "trackers"
    (trackers_root / tracker_name / "data").mkdir(parents=True, exist_ok=True)
    for seq in sequences:
        seq_gt_dir = gt_root / seq / "gt"
        seq_gt_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(mot_root / seq / "gt" / "gt.txt", seq_gt_dir / "gt.txt")
        shutil.copy(mot_root / seq / "seqinfo.ini", gt_root / seq / "seqinfo.ini")
    return gt_root, trackers_root


def evaluate_hota(gt_folder, trackers_folder, sequences, tracker_name, seq_info):
    import trackeval
    eval_config = trackeval.Evaluator.get_default_eval_config()
    for k in ['PRINT_RESULTS', 'PRINT_CONFIG', 'TIME_PROGRESS', 'OUTPUT_SUMMARY', 'OUTPUT_DETAILED', 'PLOT_CURVES']:
        eval_config[k] = False
    eval_config['DISPLAY_LESS_PROGRESS'] = True

    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config.update({
        'GT_FOLDER': str(gt_folder), 'TRACKERS_FOLDER': str(trackers_folder),
        'SKIP_SPLIT_FOL': True, 'SEQ_INFO': seq_info,
        'TRACKERS_TO_EVAL': [tracker_name], 'CLASSES_TO_EVAL': ['pedestrian'],
        'BENCHMARK': 'MOT17', 'PRINT_CONFIG': False,
    })

    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    metrics_list = [trackeval.metrics.HOTA({'PRINT_CONFIG': False})]
    results, _ = evaluator.evaluate(dataset_list, metrics_list)

    seq_results = results['MotChallenge2DBox'][tracker_name]
    out = {}
    for seq in sequences:
        r = seq_results[seq]['pedestrian']['HOTA']
        out[seq] = {
            "HOTA": float(np.mean(r['HOTA'])) * 100.0,
            "DetA": float(np.mean(r['DetA'])) * 100.0,
            "AssA": float(np.mean(r['AssA'])) * 100.0,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pp22_root", default="/nas/Muhiddin/accv2026/real_repro_offload/personpath22_subset")
    ap.add_argument("--local_root", default="~/msfp/personpath22_local",
                     help="Writable local copy (the /nas source is read-only for seqinfo.ini)")
    ap.add_argument("--checkpoint", default="~/msfp/checkpoints/fusion_attention.pt")
    ap.add_argument("--atl_checkpoint", default="~/msfp/checkpoints/atl_paper_faithful.pt")
    ap.add_argument("--workdir", default="~/msfp/trackeval_pp22")
    ap.add_argument("--out", default="~/msfp/checkpoints/personpath22_zeroshot_results.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pp22_root_src = Path(args.pp22_root)
    pp22_root = Path(args.local_root).expanduser()
    pp22_root.mkdir(parents=True, exist_ok=True)
    workdir = Path(args.workdir).expanduser()
    workdir.mkdir(parents=True, exist_ok=True)

    # The /nas source is read-only for this user; build a local writable
    # copy (img1 symlinked, gt.txt copied, seqinfo.ini generated -- not
    # shipped in this real subset) so TrackEval can write alongside it.
    sequences = []
    for seq in PP22_SEQUENCES:
        src_dir = pp22_root_src / seq
        if not src_dir.is_dir():
            continue
        n_frames = len(list((src_dir / "img1").glob("*.jpg")))
        if n_frames == 0:
            continue
        seq_dir = pp22_root / seq
        seq_dir.mkdir(parents=True, exist_ok=True)
        if not (seq_dir / "img1").exists():
            (seq_dir / "img1").symlink_to(src_dir / "img1")
        (seq_dir / "gt").mkdir(parents=True, exist_ok=True)
        shutil.copy(src_dir / "gt.txt", seq_dir / "gt" / "gt.txt")
        (seq_dir / "seqinfo.ini").write_text(
            f"[Sequence]\nname={seq}\nimDir=img1\nframeRate=30\n"
            f"seqLength={n_frames}\nimWidth=1920\nimHeight=1080\nimExt=.jpg\n",
            encoding="utf-8")
        sequences.append(seq)
    print(f"Found {len(sequences)} real PersonPath22 sequences: {sequences}")

    ckpt = torch.load(Path(args.checkpoint).expanduser(), map_location=device, weights_only=False)
    layer_channels = [LAYER_CHANNELS_V8M[n] for n in LAYER_ORDER]
    fusion = FeatureFusionModule(layer_channels=layer_channels, output_dim=128,
                                  fusion_type="attention").to(device)
    fusion.load_state_dict(ckpt["state_dict"])
    fusion.eval()

    atl_ckpt = torch.load(Path(args.atl_checkpoint).expanduser(), map_location=device, weights_only=False)
    atl = ATLPaperFaithful(input_channels=576, hidden_dim=atl_ckpt["hidden_dim"]).to(device)
    atl.load_state_dict(atl_ckpt["state_dict"])
    atl.eval()

    backbone = HookedBackbone(device)

    seq_info = {}
    by_frame_cache, fused_cache, scene_feat_cache, gt_frames_cache = {}, {}, {}, {}
    all_det_scores = []
    t0 = time.time()
    for seq in sequences:
        print(f"[{seq}] running YOLOv8m detection + real feature extraction...")
        by_frame, n_frames, scene_feats = run_yolo_detection_and_features(
            pp22_root / seq, backbone, device)
        seq_info[seq] = n_frames
        fused_by_frame = {}
        for frame, recs in by_frame.items():
            fused_by_frame[frame] = fuse_features(recs, fusion, device)
            all_det_scores.extend([r["det_conf"] for r in recs])
        by_frame_cache[seq] = by_frame
        fused_cache[seq] = fused_by_frame
        scene_feat_cache[seq] = scene_feats
        gt_frames_cache[seq] = set(
            int(line.split(",")[0]) for line in (pp22_root / seq / "gt" / "gt.txt").read_text(encoding="utf-8").splitlines() if line.strip()
        )
        print(f"  {n_frames} frames, {sum(len(v) for v in by_frame.values())} real YOLOv8m detections, "
              f"{len(gt_frames_cache[seq])} frames with real GT (sparse re-ID annotation)")
    t_extract = time.time() - t0

    # Real fixed 85th-percentile-of-detection-score baseline, per sequence
    percentile_tau = {}
    for seq in sequences:
        scores = [r["det_conf"] for recs in by_frame_cache[seq].values() for r in recs]
        percentile_tau[seq] = float(np.clip(np.percentile(scores, 85), 0.01, 0.50)) if scores else 0.25

    # Real zero-shot ATL prediction, per sequence (mean scene feature -> tau)
    atl_tau = {}
    with torch.no_grad():
        for seq in sequences:
            feats = scene_feat_cache[seq]
            if len(feats) == 0:
                atl_tau[seq] = 0.25
                continue
            mean_feat = feats.mean(dim=0, keepdim=True).to(device)
            atl_tau[seq] = float(atl.forward_from_gap(mean_feat).item())

    tracker_name = "MSFP-Track-pp22"
    gt_folder, trackers_folder = setup_trackeval_dirs(workdir, pp22_root, sequences, tracker_name)

    def write_and_eval(tau_by_seq, label):
        for seq in sequences:
            lines = run_tracker_single_threshold(by_frame_cache[seq], fused_cache[seq],
                                                  tau_by_seq[seq], seq_info[seq],
                                                  gt_frames=gt_frames_cache[seq])
            out_file = trackers_folder / tracker_name / "data" / f"{seq}.txt"
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_text("\n".join(lines), encoding="utf-8")
        hota = evaluate_hota(gt_folder, trackers_folder, sequences, tracker_name, seq_info)
        print(f"\n[{label}] per-sequence HOTA:")
        for seq in sequences:
            print(f"  {seq}: tau={tau_by_seq[seq]:.3f}  HOTA={hota[seq]['HOTA']:.2f}  "
                  f"DetA={hota[seq]['DetA']:.2f}  AssA={hota[seq]['AssA']:.2f}")
        return hota

    hota_percentile = write_and_eval(percentile_tau, "85th-percentile baseline")
    hota_atl = write_and_eval(atl_tau, "ATL zero-shot (ours)")

    print("\nRunning real oracle grid search for reference...")
    hota_grid = {seq: {} for seq in sequences}
    for tau in GRID:
        tau_by_seq = {seq: tau for seq in sequences}
        hota_g = write_and_eval(tau_by_seq, f"grid tau={tau}")
        for seq in sequences:
            hota_grid[seq][tau] = hota_g[seq]['HOTA']
    oracle_tau = {seq: max(hota_grid[seq], key=hota_grid[seq].get) for seq in sequences}
    oracle_hota = {seq: hota_grid[seq][oracle_tau[seq]] for seq in sequences}

    gains = {seq: hota_atl[seq]['HOTA'] - hota_percentile[seq]['HOTA'] for seq in sequences}
    print("\n=== Real zero-shot ATL gains over 85th-percentile baseline ===")
    for seq in sequences:
        print(f"  {seq}: +{gains[seq]:.2f} HOTA (percentile={hota_percentile[seq]['HOTA']:.2f}, "
              f"ATL={hota_atl[seq]['HOTA']:.2f}, oracle={oracle_hota[seq]:.2f})")
    print(f"Mean gain: {np.mean(list(gains.values())):+.2f} HOTA")

    result = {
        "sequences": sequences,
        "percentile_tau": percentile_tau,
        "atl_tau": atl_tau,
        "oracle_tau": oracle_tau,
        "hota_percentile": hota_percentile,
        "hota_atl": hota_atl,
        "oracle_hota": oracle_hota,
        "gains_atl_over_percentile": gains,
        "mean_gain": float(np.mean(list(gains.values()))),
        "det_score_histogram_sample": list(np.random.choice(all_det_scores, size=min(2000, len(all_det_scores)), replace=False)) if all_det_scores else [],
        "timing": {"feature_extraction_seconds": t_extract, "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"},
    }
    out_path = Path(args.out).expanduser()
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()

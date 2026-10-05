#!/usr/bin/env python3
"""
Same 9-condition honest baseline matrix as baseline_matrix_public_gp114.py,
now with the faithful LITE implementation (module 0, 48ch, half-res --
lite_faithful_public.py) in place of the earlier mistaken module-4 version,
and reporting the FULL Table 1 column set (HOTA, AssA, DetA, IDF1, MOTA,
IDSW) for every row via trackeval's HOTA+CLEAR+Identity metrics together,
not just HOTA. Each condition's predictions are saved to their own tracker
folder (not overwritten) so all metrics are computed from the real,
on-disk prediction files.
"""
import json
import time
from pathlib import Path

import numpy as np
for _name, _builtin in [("float", float), ("int", int), ("bool", bool), ("object", object)]:
    if not hasattr(np, _name):
        setattr(np, _name, _builtin)
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from real_tracker import ByteTrackStyleTracker, KalmanTrack, iou_matrix, greedy_or_hungarian_match
import verify_official_trackers as vo
from lite_faithful_public import LiteBackbone, extract_lite_val_half, lite_features
from litepp.models.feature_pyramid import FeatureFusionModule

from boxmot import BotSort, ByteTrack, OcSort, DeepOcSort, StrongSort
from boxmot.structures import Boxes, Detections, Frame
import trackeval

LAYER_ORDER = ["layer4", "layer6", "layer9"]
LAYER_CHANNELS_V8M = {"layer4": 192, "layer6": 384, "layer9": 576}


@torch.no_grad()
def fuse_msfp(records, fusion, device):
    if not records:
        return np.zeros((0, 128), dtype=np.float32)
    layer_feats = [torch.stack([r[name] for r in records]).to(device) for name in LAYER_ORDER]
    emb = fusion(layer_feats)
    emb = F.normalize(emb, p=2, dim=1)
    return emb.cpu().numpy()


class SortTracker:
    def __init__(self, iou_thresh=0.3, max_age=30, min_hits=1):
        self.iou_thresh, self.max_age, self.min_hits = iou_thresh, max_age, min_hits
        self.tracks, self.frame_idx = [], 0

    def update(self, boxes, scores):
        self.frame_idx += 1
        boxes = np.asarray(boxes, dtype=np.float32)
        predicted = [t.predict() for t in self.tracks]
        dummy_feat = np.zeros(4, dtype=np.float32)
        if self.tracks and len(boxes) > 0:
            iou = iou_matrix(predicted, boxes)
            matches, unmatched_tracks, unmatched_dets = greedy_or_hungarian_match(1 - iou, 1 - self.iou_thresh)
        else:
            matches, unmatched_tracks = [], list(range(len(self.tracks)))
            unmatched_dets = list(range(len(boxes)))
        for t_i, d_i in matches:
            self.tracks[t_i].update(boxes[d_i], dummy_feat, self.frame_idx)
        for d_i in unmatched_dets:
            self.tracks.append(KalmanTrack(boxes[d_i], dummy_feat, self.frame_idx))
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        results = []
        for t in self.tracks:
            if t.time_since_update == 0 and (t.hits >= self.min_hits or self.frame_idx <= self.min_hits):
                x1, y1, x2, y2 = t.get_state()
                results.append((t.id, x1, y1, x2, y2))
        return results


def run_sort(by_frame, fs, fe, fo):
    tracker = SortTracker()
    lines = []
    for frame in range(fs, fe + 1):
        recs = by_frame.get(frame, [])
        boxes = np.array([r["bbox"] for r in recs], dtype=np.float32) if recs else np.zeros((0, 4), dtype=np.float32)
        for tid, x1, y1, x2, y2 in tracker.update(boxes, None):
            w, h = x2 - x1, y2 - y1
            lines.append(f"{frame - fo},{tid},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,-1,-1,-1")
    return lines


def run_real_tracker_style(by_frame, feat_by_frame, tau_h, tau_l, fs, fe, fo, feat_dim, appearance_weight=0.5):
    tracker = ByteTrackStyleTracker(tau_h=tau_h, tau_l=tau_l, max_age=30, min_hits=1,
                                     appearance_weight=appearance_weight, high_stage_cost_thresh=0.7)
    lines = []
    for frame in range(fs, fe + 1):
        recs = by_frame.get(frame, [])
        if recs:
            boxes = np.array([r["bbox"] for r in recs], dtype=np.float32)
            scores = np.array([r["det_conf"] for r in recs], dtype=np.float32)
            feats = feat_by_frame.get(frame, np.zeros((len(recs), feat_dim), dtype=np.float32))
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            scores = np.zeros((0,), dtype=np.float32)
            feats = np.zeros((0, feat_dim), dtype=np.float32)
        for tid, x1, y1, x2, y2 in tracker.update(boxes, scores, feats):
            w, h = x2 - x1, y2 - y1
            lines.append(f"{frame - fo},{tid},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,-1,-1,-1")
    return lines


def run_boxmot(tracker_cls, by_frame, frame_images, fs, fe, fo, sample_id, **kwargs):
    tracker = tracker_cls(**kwargs)
    lines = []
    for frame in range(fs, fe + 1):
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
        dets = Detections(geometry=Boxes(geometry), scores=scores, class_ids=class_ids, sample_id=sample_id)
        frame_obj = Frame(image=torch.from_numpy(img).permute(2, 0, 1).contiguous(), sample_id=sample_id,
                           frame_index=frame)
        tracks = tracker.update(dets, frame_obj)
        try:
            ids = tracks.track_ids.tolist()
            boxes = tracks.geometry.values.tolist()
        except AttributeError:
            ids, boxes = [], []
        for tid, (x1, y1, x2, y2) in zip(ids, boxes):
            w, h = x2 - x1, y2 - y1
            lines.append(f"{frame - fo},{int(tid)},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,-1,-1,-1")
    return lines


def evaluate_full(gt_folder, trackers_folder, sequences, tracker_name, seq_info):
    eval_config = trackeval.Evaluator.get_default_eval_config()
    for k in ['PRINT_RESULTS', 'PRINT_CONFIG', 'TIME_PROGRESS', 'OUTPUT_SUMMARY', 'OUTPUT_DETAILED', 'PLOT_CURVES']:
        eval_config[k] = False
    eval_config['DISPLAY_LESS_PROGRESS'] = True
    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config.update({'GT_FOLDER': str(gt_folder), 'TRACKERS_FOLDER': str(trackers_folder),
                            'SKIP_SPLIT_FOL': True, 'SEQ_INFO': seq_info,
                            'TRACKERS_TO_EVAL': [tracker_name], 'CLASSES_TO_EVAL': ['pedestrian'],
                            'BENCHMARK': 'MOT17', 'PRINT_CONFIG': False})
    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    metrics_list = [trackeval.metrics.HOTA({'PRINT_CONFIG': False}),
                    trackeval.metrics.CLEAR({'PRINT_CONFIG': False}),
                    trackeval.metrics.Identity({'PRINT_CONFIG': False})]
    results, _ = evaluator.evaluate(dataset_list, metrics_list)
    seq_results = results['MotChallenge2DBox'][tracker_name]
    per_seq = {}
    for seq in sequences:
        r = seq_results[seq]['pedestrian']
        per_seq[seq] = {
            "HOTA": float(np.mean(r['HOTA']['HOTA'])) * 100.0,
            "AssA": float(np.mean(r['HOTA']['AssA'])) * 100.0,
            "DetA": float(np.mean(r['HOTA']['DetA'])) * 100.0,
            "IDF1": float(r['Identity']['IDF1']) * 100.0,
            "MOTA": float(r['CLEAR']['MOTA']) * 100.0,
            "IDSW": int(r['CLEAR']['IDSW']),
        }
    agg = {k: float(np.mean([per_seq[s][k] for s in sequences])) for k in ["HOTA", "AssA", "DetA", "IDF1", "MOTA"]}
    agg["IDSW"] = int(np.sum([per_seq[s]["IDSW"] for s in sequences]))
    return per_seq, agg


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mot_root = Path("/nas/Dataset/MOT/MOT17/train")
    workdir = Path("~/msfp_honest_repro/trackeval_full_metrics").expanduser()
    workdir.mkdir(parents=True, exist_ok=True)

    print("Loading real MSFP fusion head...")
    ckpt = torch.load(Path("~/msfp_honest_repro/checkpoints/fusion_attention.pt").expanduser(),
                       map_location=device, weights_only=False)
    layer_channels = [LAYER_CHANNELS_V8M[n] for n in LAYER_ORDER]
    fusion = FeatureFusionModule(layer_channels=layer_channels, output_dim=128,
                                  fusion_type="attention").to(device)
    fusion.load_state_dict(ckpt["state_dict"])
    fusion.eval()

    backbone = vo.HookedBackbone(device)
    lite_backbone = LiteBackbone(device)
    splits = {seq: vo.half_split_frame_range(mot_root / seq) for seq in vo.SEQUENCES}

    by_frame_cache, frame_images_cache, msfp_cache, seq_info = {}, {}, {}, {}
    lite_by_frame_cache, lite_feat_cache = {}, {}
    for seq in vo.SEQUENCES:
        print(f"[{seq}] extracting layer4/6/9 features (val_half, det.txt)...")
        recs, imgs = vo.extract_val_half(mot_root / seq, backbone, device, splits[seq])
        by_frame_cache[seq] = recs
        frame_images_cache[seq] = imgs
        msfp_cache[seq] = {f: fuse_msfp(r, fusion, device) for f, r in recs.items()}
        n_total = len(list((mot_root / seq / "img1").glob("*.jpg")))
        seq_info[seq] = n_total - splits[seq]

        print(f"[{seq}] extracting LITE-faithful module-0 features...")
        lrecs = extract_lite_val_half(mot_root / seq, lite_backbone, device, splits[seq])
        lite_by_frame_cache[seq] = lrecs
        lite_feat_cache[seq] = {f: lite_features(r) for f, r in lrecs.items()}

    gt_folder, trackers_folder = vo.setup_trackeval_dirs(workdir, mot_root, vo.SEQUENCES, "_tmp", splits)

    all_results = {}

    def run_and_eval(name, run_fn):
        tdir_name = name
        (trackers_folder / tdir_name / "data").mkdir(parents=True, exist_ok=True)
        for seq in vo.SEQUENCES:
            n_total = len(list((mot_root / seq / "img1").glob("*.jpg")))
            lines = run_fn(seq, splits[seq] + 1, n_total, splits[seq])
            (trackers_folder / tdir_name / "data" / f"{seq}.txt").write_text("\n".join(lines), encoding="utf-8")
        per_seq, agg = evaluate_full(gt_folder, trackers_folder, vo.SEQUENCES, tdir_name, seq_info)
        all_results[name] = {"per_seq": per_seq, "agg": agg}
        print(f"[{name}] HOTA={agg['HOTA']:.2f} AssA={agg['AssA']:.2f} DetA={agg['DetA']:.2f} "
              f"IDF1={agg['IDF1']:.2f} MOTA={agg['MOTA']:.2f} IDSW={agg['IDSW']}")

    t0 = time.time()
    run_and_eval("SORT", lambda s, fs, fe, fo: run_sort(by_frame_cache[s], fs, fe, fo))
    run_and_eval("DeepSORT_style", lambda s, fs, fe, fo: run_real_tracker_style(
        by_frame_cache[s], lite_feat_cache[s], 0.0, 0.0, fs, fe, fo, 48))
    run_and_eval("StrongSort", lambda s, fs, fe, fo: run_boxmot(
        StrongSort, by_frame_cache[s], frame_images_cache[s], fs, fe, fo, s))
    run_and_eval("ByteTrack_motion", lambda s, fs, fe, fo: run_boxmot(
        ByteTrack, by_frame_cache[s], frame_images_cache[s], fs, fe, fo, s))
    run_and_eval("OCSORT_motion", lambda s, fs, fe, fo: run_boxmot(
        OcSort, by_frame_cache[s], frame_images_cache[s], fs, fe, fo, s))
    run_and_eval("BotSort_motion", lambda s, fs, fe, fo: run_boxmot(
        BotSort, by_frame_cache[s], frame_images_cache[s], fs, fe, fo, s, use_embeddings=False))
    run_and_eval("DeepOcSort", lambda s, fs, fe, fo: run_boxmot(
        DeepOcSort, by_frame_cache[s], frame_images_cache[s], fs, fe, fo, s))
    run_and_eval("LITE_faithful", lambda s, fs, fe, fo: run_real_tracker_style(
        lite_by_frame_cache[s], lite_feat_cache[s], 0.25, 0.25, fs, fe, fo, 48))
    run_and_eval("MSFP_Track", lambda s, fs, fe, fo: run_real_tracker_style(
        by_frame_cache[s], msfp_cache[s], 0.25, 0.25, fs, fe, fo, 128))
    elapsed = time.time() - t0

    out_path = Path("~/msfp_honest_repro/baseline_matrix_full_metrics.json").expanduser()
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"protocol": "val_half, public MOT17 det.txt, current validated code, gp114",
                   "results": all_results, "elapsed_seconds": elapsed}, f, indent=2)

    print("\n=== FULL METRIC SUMMARY ===")
    print(f"{'Method':20s} {'HOTA':>7s} {'AssA':>7s} {'DetA':>7s} {'IDF1':>7s} {'MOTA':>7s} {'IDSW':>7s}")
    for name, d in all_results.items():
        a = d["agg"]
        print(f"{name:20s} {a['HOTA']:7.2f} {a['AssA']:7.2f} {a['DetA']:7.2f} {a['IDF1']:7.2f} {a['MOTA']:7.2f} {a['IDSW']:7d}")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()

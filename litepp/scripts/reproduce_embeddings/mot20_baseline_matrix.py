#!/usr/bin/env python3
"""
Honest Table 2 (MOT20) baseline matrix, mirroring the MOT17 work: real
val_half split of the 4 MOT20 train sequences (01/02/03/05, the only ones
with local ground truth), public MOT20 detections, current validated code.

Rows: DeepSORT-style, ByteTrack (motion), OC-SORT (motion), BoT-SORT (own
ReID), LITE (faithful, module-0), MSFP (attn, fixed tau=0.25), MSFP-Track
(real ATL-adaptive threshold, using the MOT20-specific ATL checkpoint
trained earlier this session on real MOT20 oracle thresholds, reusing the
MOT17-trained fusion head per the paper's own stated protocol).

Known real property carried over from earlier MOT20 work this session:
MOT20-03/05 have extreme public-detection sparsity (~1-2% of detections at
conf=1, nearly all others at conf=0), which will genuinely crush every
tracker's performance on those two sequences specifically -- this is a
property of the public detection release, not a pipeline bug, and applies
equally to every condition tested here.
"""
import json
from pathlib import Path

import cv2
import numpy as np
for _name, _builtin in [("float", float), ("int", int), ("bool", bool), ("object", object)]:
    if not hasattr(np, _name):
        setattr(np, _name, _builtin)
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import verify_official_trackers as vo
from real_tracker import ByteTrackStyleTracker
from lite_faithful_public import LiteBackbone, lite_features
from atl_paper_faithful import ATLPaperFaithful
from litepp.models.feature_pyramid import FeatureFusionModule
from baseline_matrix_full_metrics import run_real_tracker_style, run_boxmot

from boxmot import BotSort, ByteTrack, OcSort
import trackeval

LAYER_ORDER = ["layer4", "layer6", "layer9"]
LAYER_CHANNELS_V8M = {"layer4": 192, "layer6": 384, "layer9": 576}
MOT20_SEQUENCES = ["MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05"]


def extract_mot20_val_half(seq_dir, backbone, lite_backbone, device, min_frame, imgsz_multiple=32):
    by_frame_det = vo.load_det(seq_dir / "det" / "det.txt")
    img_dir = seq_dir / "img1"
    records, lite_records, frame_images = {}, {}, {}
    scene_feats = []
    for frame, dets in sorted(by_frame_det.items()):
        if frame <= min_frame:
            continue
        img_path = img_dir / f"{frame:06d}.jpg"
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        h0, w0 = img.shape[:2]
        h = ((h0 + imgsz_multiple - 1) // imgsz_multiple) * imgsz_multiple
        w = ((w0 + imgsz_multiple - 1) // imgsz_multiple) * imgsz_multiple
        padded = cv2.copyMakeBorder(img, 0, h - h0, 0, w - w0, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(rgb).to(device).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        frame_images[frame] = img

        feats = backbone.forward(img_tensor)
        scene_feats.append(feats["layer9"].mean(dim=(2, 3)).squeeze(0).cpu())
        lite_feat_map = lite_backbone.forward(img_tensor)

        boxes_xyxy, confs = [], []
        for x, y, bw, bh, conf in dets:
            x1, y1, x2, y2 = x, y, x + bw, y + bh
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(float(w0), x2), min(float(h0), y2)
            if x2 <= x1 or y2 <= y1:
                continue
            boxes_xyxy.append([x1, y1, x2, y2])
            confs.append(conf)
        if not boxes_xyxy:
            continue

        from torchvision.ops import roi_align
        boxes_t = torch.tensor(boxes_xyxy, device=device, dtype=torch.float32)
        rois = torch.cat([torch.zeros((boxes_t.shape[0], 1), device=device), boxes_t], dim=1)

        per_layer_vecs = {}
        for name, feat_map in feats.items():
            _, c, fh, fw = feat_map.shape
            per_layer_vecs[name] = roi_align(feat_map, rois, output_size=7, spatial_scale=fh / h,
                                              sampling_ratio=2, aligned=True).mean(dim=(2, 3)).cpu()
        _, _, lfh, lfw = lite_feat_map.shape
        lite_pooled = roi_align(lite_feat_map, rois, output_size=7, spatial_scale=lfh / h,
                                 sampling_ratio=2, aligned=True).mean(dim=(2, 3)).cpu().numpy()

        recs_f, lrecs_f = [], []
        for i in range(len(boxes_xyxy)):
            rec = {"frame": frame, "bbox": boxes_xyxy[i], "det_conf": confs[i]}
            for name in LAYER_ORDER:
                rec[name] = per_layer_vecs[name][i]
            recs_f.append(rec)
            lrecs_f.append({"frame": frame, "bbox": boxes_xyxy[i], "det_conf": confs[i], "feat": lite_pooled[i]})
        records[frame] = recs_f
        lite_records[frame] = lrecs_f
    scene_feat_mean = torch.stack(scene_feats).mean(dim=0) if scene_feats else torch.zeros(576)
    return records, lite_records, frame_images, scene_feat_mean


@torch.no_grad()
def fuse_msfp(records, fusion, device):
    if not records:
        return np.zeros((0, 128), dtype=np.float32)
    layer_feats = [torch.stack([r[name] for r in records]).to(device) for name in LAYER_ORDER]
    emb = fusion(layer_feats)
    emb = F.normalize(emb, p=2, dim=1)
    return emb.cpu().numpy()


def setup_trackeval_dirs_mot20(workdir, mot_root, sequences, tracker_name, splits):
    import re
    gt_root = workdir / "gt"
    trackers_root = workdir / "trackers"
    (trackers_root / tracker_name / "data").mkdir(parents=True, exist_ok=True)
    for seq in sequences:
        seq_gt_dir = gt_root / seq / "gt"
        seq_gt_dir.mkdir(parents=True, exist_ok=True)
        lines_out = []
        for line in (mot_root / seq / "gt" / "gt.txt").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            parts = line.split(",")
            frame = int(parts[0])
            if frame <= splits[seq]:
                continue
            parts[0] = str(frame - splits[seq])
            lines_out.append(",".join(parts))
        (seq_gt_dir / "gt.txt").write_text("\n".join(lines_out), encoding="utf-8")
        n_total = len(list((mot_root / seq / "img1").glob("*.jpg")))
        val_len = n_total - splits[seq]
        src_ini = (mot_root / seq / "seqinfo.ini").read_text(encoding="utf-8")
        dst_ini = re.sub(r"seqLength=\d+", f"seqLength={val_len}", src_ini)
        (gt_root / seq / "seqinfo.ini").write_text(dst_ini, encoding="utf-8")
    return gt_root, trackers_root


def evaluate_full_mot20(gt_folder, trackers_folder, sequences, tracker_name, seq_info):
    eval_config = trackeval.Evaluator.get_default_eval_config()
    for k in ['PRINT_RESULTS', 'PRINT_CONFIG', 'TIME_PROGRESS', 'OUTPUT_SUMMARY', 'OUTPUT_DETAILED', 'PLOT_CURVES']:
        eval_config[k] = False
    eval_config['DISPLAY_LESS_PROGRESS'] = True
    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config.update({'GT_FOLDER': str(gt_folder), 'TRACKERS_FOLDER': str(trackers_folder),
                            'SKIP_SPLIT_FOL': True, 'SEQ_INFO': seq_info,
                            'TRACKERS_TO_EVAL': [tracker_name], 'CLASSES_TO_EVAL': ['pedestrian'],
                            'BENCHMARK': 'MOT20', 'PRINT_CONFIG': False})
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
    mot_root = Path("/nas/Dataset/MOT/MOT20/train")
    workdir = Path("~/msfp_honest_repro/trackeval_mot20_matrix").expanduser()
    workdir.mkdir(parents=True, exist_ok=True)

    print("Loading real MSFP fusion head (MOT17-trained, reused per paper protocol) + MOT20 ATL encoder...")
    ckpt = torch.load(Path("~/msfp_honest_repro/checkpoints/fusion_attention.pt").expanduser(),
                       map_location=device, weights_only=False)
    layer_channels = [LAYER_CHANNELS_V8M[n] for n in LAYER_ORDER]
    fusion = FeatureFusionModule(layer_channels=layer_channels, output_dim=128,
                                  fusion_type="attention").to(device)
    fusion.load_state_dict(ckpt["state_dict"])
    fusion.eval()

    atl_ckpt = torch.load(Path("~/msfp_honest_repro/checkpoints/atl_paper_faithful_mot20.pt").expanduser(),
                           map_location=device, weights_only=False)
    atl = ATLPaperFaithful(input_channels=576, hidden_dim=atl_ckpt["hidden_dim"]).to(device)
    atl.load_state_dict(atl_ckpt["state_dict"])
    atl.eval()

    backbone = vo.HookedBackbone(device)
    lite_backbone = LiteBackbone(device)
    splits = {seq: vo.half_split_frame_range(mot_root / seq) for seq in MOT20_SEQUENCES}
    print(f"train_half/val_half split: {splits}")

    by_frame_cache, lite_cache_raw, frame_images_cache, msfp_cache, lite_feat_cache, seq_info, taus = \
        {}, {}, {}, {}, {}, {}, {}
    for seq in MOT20_SEQUENCES:
        print(f"[{seq}] extracting real features (val_half, public det.txt)...")
        recs, lrecs, imgs, scene_feat = extract_mot20_val_half(mot_root / seq, backbone, lite_backbone, device,
                                                                 splits[seq])
        by_frame_cache[seq] = recs
        lite_cache_raw[seq] = lrecs
        frame_images_cache[seq] = imgs
        msfp_cache[seq] = {f: fuse_msfp(r, fusion, device) for f, r in recs.items()}
        lite_feat_cache[seq] = {f: lite_features(r) for f, r in lrecs.items()}
        n_total = len(list((mot_root / seq / "img1").glob("*.jpg")))
        seq_info[seq] = n_total - splits[seq]
        n_dets = sum(len(v) for v in recs.values())
        with torch.no_grad():
            tau = float(atl.forward_from_gap(scene_feat.unsqueeze(0).to(device)).item())
        taus[seq] = tau
        print(f"  {n_dets} real detections, real ATL tau={tau:.3f}")

    gt_folder, trackers_folder = setup_trackeval_dirs_mot20(workdir, mot_root, MOT20_SEQUENCES, "_tmp", splits)

    all_results = {}

    def run_and_eval(name, run_fn):
        (trackers_folder / name / "data").mkdir(parents=True, exist_ok=True)
        for seq in MOT20_SEQUENCES:
            n_total = len(list((mot_root / seq / "img1").glob("*.jpg")))
            lines = run_fn(seq, splits[seq] + 1, n_total, splits[seq])
            (trackers_folder / name / "data" / f"{seq}.txt").write_text("\n".join(lines), encoding="utf-8")
        per_seq, agg = evaluate_full_mot20(gt_folder, trackers_folder, MOT20_SEQUENCES, name, seq_info)
        all_results[name] = {"per_seq": per_seq, "agg": agg}
        print(f"[{name}] HOTA={agg['HOTA']:.2f} AssA={agg['AssA']:.2f} DetA={agg['DetA']:.2f} "
              f"IDF1={agg['IDF1']:.2f} MOTA={agg['MOTA']:.2f} IDSW={agg['IDSW']}")

    run_and_eval("DeepSORT_style", lambda s, fs, fe, fo: run_real_tracker_style(
        by_frame_cache[s], lite_feat_cache[s], 0.0, 0.0, fs, fe, fo, 48))
    run_and_eval("ByteTrack_motion", lambda s, fs, fe, fo: run_boxmot(
        ByteTrack, by_frame_cache[s], frame_images_cache[s], fs, fe, fo, s))
    run_and_eval("OCSORT_motion", lambda s, fs, fe, fo: run_boxmot(
        OcSort, by_frame_cache[s], frame_images_cache[s], fs, fe, fo, s))
    run_and_eval("BotSort_ownReID", lambda s, fs, fe, fo: run_boxmot(
        BotSort, by_frame_cache[s], frame_images_cache[s], fs, fe, fo, s))
    run_and_eval("LITE_faithful", lambda s, fs, fe, fo: run_real_tracker_style(
        by_frame_cache[s], lite_feat_cache[s], 0.25, 0.25, fs, fe, fo, 48))
    run_and_eval("MSFP_attn_fixed", lambda s, fs, fe, fo: run_real_tracker_style(
        by_frame_cache[s], msfp_cache[s], 0.25, 0.25, fs, fe, fo, 128))
    run_and_eval("MSFP_Track_ATL", lambda s, fs, fe, fo: run_real_tracker_style(
        by_frame_cache[s], msfp_cache[s], taus[s], max(0.01, taus[s] * 0.5), fs, fe, fo, 128))

    out_path = Path("~/msfp_honest_repro/mot20_baseline_matrix_results.json").expanduser()
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"protocol": "MOT20 val_half (01/02/03/05 train sequences), public detections, gp114",
                   "atl_taus": taus, "results": all_results}, f, indent=2)

    print("\n=== MOT20 FULL METRIC SUMMARY ===")
    print(f"{'Method':20s} {'HOTA':>7s} {'AssA':>7s} {'DetA':>7s} {'IDF1':>7s} {'MOTA':>7s} {'IDSW':>7s}")
    for name, d in all_results.items():
        a = d["agg"]
        print(f"{name:20s} {a['HOTA']:7.2f} {a['AssA']:7.2f} {a['DetA']:7.2f} {a['IDF1']:7.2f} {a['MOTA']:7.2f} {a['IDSW']:7d}")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()

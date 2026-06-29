#!/usr/bin/env python3
"""
eval_norm_nod.py -- measure frozen-detector mAP on packed NOD test split to pick
the loader's normalization empirically, mirroring eval_norm_rod.py.

WHAT DIFFERS FROM ROD'S EVAL
----------------------------
1. log1p is ALREADY the chosen transform (validated on ROD: +56% mAP over
   linear, same frozen FCOS, same day/night rationale). For NOD the open
   question is the REF *value* for 14-bit data, not which transform. So this
   script SWEEPS REF candidates under log1p (--refs) rather than swapping
   transforms. linear/asinh remain available via --norms for completeness.
2. NOD has 3 classes: person 0->1, bicycle 1->2, car 2->3 (COCO ids). Simpler
   than ROD's 5.
3. TWO CAMERAS. Runs the sweep PER CAMERA (Nikon, Sony) so you see whether the
   best REF differs by sensor -- input to the cross-sensor generalization
   analysis. --cameras to restrict.

No training. FCOS COCO-pretrained head, frozen. Offline (cached weights).

Usage (on gpu node):
  python eval_norm_nod.py --nod_root /scratch/INC1526354/nod --limit 0
  python eval_norm_nod.py ... --refs 3000 6000 10000 16000   # REF sweep
  python eval_norm_nod.py ... --cameras Sony --limit 200      # quick, one camera
"""
import argparse
import json
import os
import numpy as np
import torch
from torchvision.models.detection import fcos_resnet50_fpn, FCOS_ResNet50_FPN_Weights
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# compact -> COCO id (person, bicycle, car). NOD's 3 classes.
COMPACT_TO_COCO = {0: 1, 1: 2, 2: 3}
COCO_TO_COMPACT = {v: k for k, v in COMPACT_TO_COCO.items()}
NOD_COCO_IDS = set(COMPACT_TO_COCO.values())

CAMERAS = ["Nikon", "Sony"]

# fixed scales for the non-swept transforms (ablation only)
ASINH_S = 1000.0
LINEAR_DIV = 6000.0


def make_log1p(ref):
    def _f(x):
        return np.clip(np.log1p(x) / np.log1p(ref), 0.0, 1.0)
    return _f


def make_asinh(ref):
    def _f(x):
        return np.clip(np.arcsinh(x / ASINH_S) / np.arcsinh(ref / ASINH_S), 0.0, 1.0)
    return _f


def norm_linear(x):
    return np.clip(x / LINEAR_DIV, 0.0, 1.0)


def to_rgb(arr):
    """packed [4,H,W] uint16 -> [3,H,W] float (R, mean(G1,G2), B), still unnormalized."""
    R, G1, G2, B = arr[0], arr[1], arr[2], arr[3]
    rgb = np.stack([R, (G1 + G2) / 2.0, B], axis=0).astype(np.float32)
    return rgb


def load_gt(json_path):
    """Return [{'file_name','boxes':[xyxy],'labels':[compact]}, ...] in id order."""
    j = json.load(open(json_path))
    by_id = {}
    for im in j["images"]:
        by_id[im["id"]] = {"file_name": im["file_name"], "boxes": [], "labels": []}
    for a in j["annotations"]:
        x, y, w, h = a["bbox"]
        by_id[a["image_id"]]["boxes"].append([x, y, x + w, y + h])
        by_id[a["image_id"]]["labels"].append(a["category_id"])   # already compact
    return [by_id[k] for k in sorted(by_id)]


def build_norm_arms(refs, norms):
    """Build {label: fn}. For log1p/asinh, one arm per REF; linear is single."""
    arms = {}
    for nm in norms:
        if nm == "log1p":
            for r in refs:
                arms[f"log1p@{int(r)}"] = make_log1p(r)
        elif nm == "asinh":
            for r in refs:
                arms[f"asinh@{int(r)}"] = make_asinh(r)
        elif nm == "linear":
            arms["linear"] = norm_linear
        else:
            raise ValueError(f"unknown norm {nm}")
    return arms


def eval_camera(camera, model, device, nod_root, refs, norms, limit, score_thresh, split):
    json_path = os.path.join(nod_root, "annotations_packed", f"nod_packed_{camera}_{split}.json")
    packed_dir = os.path.join(nod_root, "packed", camera)
    records = load_gt(json_path)
    if limit:
        records = records[:limit]
    print(f"\n##### CAMERA {camera} [{split}]: {len(records)} images #####")

    arms = build_norm_arms(refs, norms)
    results = {}
    for label, normfn in arms.items():
        metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
        n_done = 0
        for rec in records:
            npy_path = os.path.join(packed_dir, rec["file_name"])
            if not os.path.exists(npy_path):
                continue
            arr = np.load(npy_path)
            rgb = normfn(to_rgb(arr))
            t = torch.from_numpy(rgb).float().to(device)   # [3,H,W] in [0,1]
            with torch.no_grad():
                out = model([t])[0]
            keep = []
            for box, lbl, scr in zip(out["boxes"], out["labels"], out["scores"]):
                li = int(lbl)
                if li in NOD_COCO_IDS and float(scr) >= score_thresh:
                    keep.append((box.tolist(), COCO_TO_COMPACT[li], float(scr)))
            if keep:
                pred = {
                    "boxes": torch.tensor([k[0] for k in keep], dtype=torch.float32),
                    "labels": torch.tensor([k[1] for k in keep], dtype=torch.int64),
                    "scores": torch.tensor([k[2] for k in keep], dtype=torch.float32),
                }
            else:
                pred = {"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.int64),
                        "scores": torch.zeros(0)}
            tgt = {
                "boxes": torch.tensor(rec["boxes"], dtype=torch.float32) if rec["boxes"]
                         else torch.zeros(0, 4),
                "labels": torch.tensor(rec["labels"], dtype=torch.int64) if rec["labels"]
                          else torch.zeros(0, dtype=torch.int64),
            }
            metric.update([pred], [tgt])
            n_done += 1
            if n_done % 100 == 0:
                print(f"  [{camera}/{label}] {n_done}/{len(records)}")
        m = metric.compute()
        results[label] = {"mAP": float(m["map"]), "mAP50": float(m["map_50"])}
        print(f"=== {camera}/{label}: mAP={results[label]['mAP']:.4f}  mAP50={results[label]['mAP50']:.4f}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nod_root", required=True, help="contains packed/<cam>/ and annotations_packed/")
    ap.add_argument("--cameras", nargs="+", default=CAMERAS, choices=CAMERAS)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"],
                    help="which split to evaluate. Use val for REF SELECTION, test for final REPORTING.")
    ap.add_argument("--refs", nargs="+", type=float, default=[3000, 6000, 10000, 16000],
                    help="REF candidates to sweep for log1p/asinh")
    ap.add_argument("--norms", nargs="+", default=["log1p"], choices=["log1p", "asinh", "linear"],
                    help="which transform families to test (default log1p only)")
    ap.add_argument("--limit", type=int, default=0, help="0 = all; else first N images per camera")
    ap.add_argument("--score_thresh", type=float, default=0.05)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    weights = FCOS_ResNet50_FPN_Weights.DEFAULT
    model = fcos_resnet50_fpn(weights=weights).eval().to(device)
    print("FCOS loaded (frozen COCO head)")
    print(f"split: {args.split}   refs swept: {[int(r) for r in args.refs]}   norms: {args.norms}")

    all_results = {}
    for cam in args.cameras:
        all_results[cam] = eval_camera(cam, model, device, args.nod_root,
                                       args.refs, args.norms, args.limit, args.score_thresh, args.split)

    print("\n==== SUMMARY (FCOS, frozen COCO head) ====")
    for cam, res in all_results.items():
        print(f"\n[{cam}]")
        print(f"  {'arm':14s} {'mAP':>8s} {'mAP50':>8s}")
        best = max(res.items(), key=lambda kv: kv[1]["mAP"])
        for label, r in res.items():
            star = "  <-- best" if label == best[0] else ""
            print(f"  {label:14s} {r['mAP']:>8.4f} {r['mAP50']:>8.4f}{star}")


if __name__ == "__main__":
    main()
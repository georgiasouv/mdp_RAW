#!/usr/bin/env python3
"""
eval_norm_rod.py -- measure frozen-detector mAP on packed ROD test split under
several candidate normalizations, to pick the loader's default empirically.

No training. FCOS is loaded with its COCO-pretrained head (frozen). ROD classes
map to COCO ids: person 0->1, bicycle 1->2, car 2->3, train 3->7, truck 4->8.
For each normalization we normalize every packed .npy, run FCOS, filter preds to
those 5 COCO ids, remap back to compact {0..4}, and score mAP vs the GT boxes.

Runs offline (cached weights). Intended for gpu-04 via srun.

Usage (on cluster):
  python eval_norm_rod.py --rod_root /scratch/INC1526354/rod --limit 0
  python eval_norm_rod.py ... --limit 200      # quick subset
  python eval_norm_rod.py ... --norms log1p    # only one transform
"""
import argparse
import json
import os
import numpy as np
import torch
from torchvision.models.detection import fcos_resnet50_fpn, FCOS_ResNet50_FPN_Weights
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# compact -> COCO id (the verified mapping)
COMPACT_TO_COCO = {0: 1, 1: 2, 2: 3, 3: 7, 4: 8}
COCO_TO_COMPACT = {v: k for k, v in COMPACT_TO_COCO.items()}
ROD_COCO_IDS = set(COMPACT_TO_COCO.values())

# normalization candidates -- the swappable transforms under test
S = 10000.0
REF = 600000.0
DIV = 102000.0


def norm_linear(x):
    return np.clip(x / DIV, 0.0, 1.0)


def norm_asinh(x):
    return np.clip(np.arcsinh(x / S) / np.arcsinh(REF / S), 0.0, 1.0)


def norm_log1p(x):
    return np.clip(np.log1p(x) / np.log1p(REF), 0.0, 1.0)


NORMS = {"linear": norm_linear, "asinh": norm_asinh, "log1p": norm_log1p}


def to_rgb(arr):
    """packed [4,H,W] uint32 -> [3,H,W] float (R, mean(G1,G2), B), still unnormalized."""
    R, G1, G2, B = arr[0], arr[1], arr[2], arr[3]
    rgb = np.stack([R, (G1 + G2) / 2.0, B], axis=0).astype(np.float32)
    return rgb


def load_gt(json_path):
    """Return {file_name: {'boxes':[xyxy], 'labels':[compact]}} and image order."""
    j = json.load(open(json_path))
    by_id = {}
    for im in j["images"]:
        by_id[im["id"]] = {"file_name": im["file_name"], "boxes": [], "labels": []}
    for a in j["annotations"]:
        x, y, w, h = a["bbox"]
        by_id[a["image_id"]]["boxes"].append([x, y, x + w, y + h])
        by_id[a["image_id"]]["labels"].append(a["category_id"])  # already compact
    return [by_id[k] for k in sorted(by_id)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rod_root", required=True, help="contains packed/test/ and rod_packed_test.json")
    ap.add_argument("--limit", type=int, default=0, help="0 = all; else first N images")
    ap.add_argument("--norms", nargs="+", default=list(NORMS.keys()))
    ap.add_argument("--score_thresh", type=float, default=0.05)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    weights = FCOS_ResNet50_FPN_Weights.DEFAULT
    model = fcos_resnet50_fpn(weights=weights).eval().to(device)
    print("FCOS loaded (frozen COCO head)")

    json_path = os.path.join(args.rod_root, "rod_packed_test.json")
    packed_dir = os.path.join(args.rod_root, "packed", "test")
    records = load_gt(json_path)
    if args.limit:
        records = records[:args.limit]
    print(f"test images: {len(records)}")

    results = {}
    for norm_name in args.norms:
        normfn = NORMS[norm_name]
        metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
        n_done = 0
        for rec in records:
            npy_path = os.path.join(packed_dir, rec["file_name"])
            if not os.path.exists(npy_path):
                continue
            arr = np.load(npy_path)
            rgb = to_rgb(arr)
            rgb = normfn(rgb)                              # apply candidate transform
            t = torch.from_numpy(rgb).float().to(device)   # [3,H,W] in [0,1]
            with torch.no_grad():
                out = model([t])[0]
            # filter to ROD's COCO ids, remap to compact
            keep = []
            for box, lbl, scr in zip(out["boxes"], out["labels"], out["scores"]):
                li = int(lbl)
                if li in ROD_COCO_IDS and float(scr) >= args.score_thresh:
                    keep.append((box.tolist(), COCO_TO_COMPACT[li], float(scr)))
            if keep:
                pred = {
                    "boxes": torch.tensor([k[0] for k in keep], dtype=torch.float32),
                    "labels": torch.tensor([k[1] for k in keep], dtype=torch.int64),
                    "scores": torch.tensor([k[2] for k in keep], dtype=torch.float32),
                }
            else:
                pred = {
                    "boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.int64),
                    "scores": torch.zeros(0),
                }
            tgt = {
                "boxes": torch.tensor(rec["boxes"], dtype=torch.float32) if rec["boxes"]
                         else torch.zeros(0, 4),
                "labels": torch.tensor(rec["labels"], dtype=torch.int64) if rec["labels"]
                          else torch.zeros(0, dtype=torch.int64),
            }
            metric.update([pred], [tgt])
            n_done += 1
            if n_done % 250 == 0:
                print(f"  [{norm_name}] {n_done}/{len(records)}")
        m = metric.compute()
        results[norm_name] = {"mAP": float(m["map"]), "mAP50": float(m["map_50"])}
        print(f"=== {norm_name}: mAP={results[norm_name]['mAP']:.4f}  mAP50={results[norm_name]['mAP50']:.4f}")

    print("\n==== SUMMARY (FCOS, frozen COCO head) ====")
    print(f"{'norm':10s} {'mAP':>8s} {'mAP50':>8s}")
    for n, r in results.items():
        print(f"{n:10s} {r['mAP']:>8.4f} {r['mAP50']:>8.4f}")


if __name__ == "__main__":
    main()

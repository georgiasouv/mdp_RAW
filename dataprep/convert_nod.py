#!/usr/bin/env python3
"""
convert_nod.py -- build packed-coordinate COCO jsons for NOD (two cameras).

Unlike ROD, NOD's source jsons are ALREADY split (train/val/test) and carry
CONTIGUOUS category ids {1,2,3}, so there is NO .txt-manifest reconciliation and
NO id-gap remap. The two additions over convert_rod.py are: (a) a per-camera
loop, since Nikon and Sony have different packed dims, and (b) using the PACKED
.npy dims as the declared width/height (see note below).

Reads (the `raw_` prefixed jsons — the RAW-image annotations, NOT `rawpy_`):
  Nikon: raw_new_Nikon750_{train,val,test}.json
  Sony : raw_new_Sony_RX100m7_{train,val}.json
         raw_str_labeled_new_Sony_RX100m7_test.json   (test has a different name)

Produces (one per camera+split) in --out:
  nod_packed_{Nikon,Sony}_{split}.json
    - boxes  : [x,y,w,h] / SCALE   (xywh in PACKED px; loader converts to xyxy)
    - labels : COCO {1,2,3} -> compact {0,1,2}  (person, bicycle, car)
    - file_name : DSC_xxxx.NEF / .ARW -> DSC_xxxx.npy

NOTE ON DIMS (the one subtlety):
  The source annotation dims (Nikon 3936x2624, Sony 5472x3648) are SMALLER than
  rawpy's raw_image_visible that we packed (Nikon 3968x2640 -> packed 1984x1320,
  Sony 5496x3672 -> packed 2748x1836). The extra margin is trailing bottom-right,
  and boxes are origin-anchored, so boxes/2 land correctly on the larger packed
  array (verified by overlay). Therefore the declared width/height in the output
  json is the PACKED .npy dims (what the loader reads), not anno/2.

Usage:
  python convert_nod.py --annsrc /cifs/.../NOD/annotations_raw \
                        --out /scratch/INC1526354/nod/annotations_packed
  python convert_nod.py ... --check-only          # validate + stats, no write
  python convert_nod.py ... --cameras Nikon        # one camera
  python convert_nod.py ... --splits val           # one split
"""
import argparse
import json
import os
import sys
from collections import Counter

SCALE = 2   # rawpy visible dims are 2x packed dims (2x2 Bayer block)

# Packed .npy dims per camera (= rawpy raw_image_visible / SCALE). These are the
# dims the loader sees, declared as width/height in the output json.
PACK_DIMS = {
    "Nikon": {"w": 1984, "h": 1320},   # from visible 3968x2640
    "Sony":  {"w": 2748, "h": 1836},   # from visible 5496x3672
}

# Source-annotation dims (for the sanity assertion; the smaller, processed crop).
ANNO_DIMS = {
    "Nikon": {"w": 3936, "h": 2624},
    "Sony":  {"w": 5472, "h": 3648},
}

# COCO category id -> compact contiguous id. NOD COCO is 1=person 2=bicycle
# 3=car; compact {0,1,2} matches ROD's first three (0=person 1=bicycle 2=car).
COCO_TO_COMPACT = {1: 0, 2: 1, 3: 2}
COMPACT_NAMES = {0: "person", 1: "bicycle", 2: "car"}

# Per-camera source json filenames (note Sony test is named differently).
SRC_JSON = {
    "Nikon": {
        "train": "raw_new_Nikon750_train.json",
        "val":   "raw_new_Nikon750_val.json",
        "test":  "raw_new_Nikon750_test.json",
    },
    "Sony": {
        "train": "raw_new_Sony_RX100m7_train.json",
        "val":   "raw_new_Sony_RX100m7_val.json",
        "test":  "raw_str_labeled_new_Sony_RX100m7_test.json",
    },
}


def convert_one(camera, split, annsrc, strict=True):
    """Build one packed COCO json dict for camera+split."""
    src_path = os.path.join(annsrc, camera, SRC_JSON[camera][split])
    coco = json.load(open(src_path))

    # NOD image ids are STRINGS (the stem, e.g. 'DSC_0830'); index by that.
    by_img = {}
    for im in coco["images"]:
        by_img[im["id"]] = {
            "file_name": im["file_name"],
            "width": im["width"],
            "height": im["height"],
            "anns": [],
        }
    for a in coco["annotations"]:
        by_img[a["image_id"]]["anns"].append((a["category_id"], *a["bbox"]))

    pw, ph = PACK_DIMS[camera]["w"], PACK_DIMS[camera]["h"]
    aw, ah = ANNO_DIMS[camera]["w"], ANNO_DIMS[camera]["h"]

    images, annotations = [], []
    img_id = 0
    ann_id = 0
    label_hist = Counter()
    anno_dim_violations = 0

    for src_id in sorted(by_img):
        rec = by_img[src_id]

        # sanity: source annotation must be in the expected anno coord space
        if strict and (rec["width"] != aw or rec["height"] != ah):
            anno_dim_violations += 1

        stem = os.path.splitext(rec["file_name"])[0]
        img_id += 1
        images.append({
            "id": img_id,
            "file_name": stem + ".npy",
            "width": pw,
            "height": ph,
        })

        for (cat, x, y, w, h) in rec["anns"]:
            if cat not in COCO_TO_COMPACT:
                raise ValueError(f"unknown COCO category id {cat} in {src_path}")
            compact = COCO_TO_COMPACT[cat]
            label_hist[compact] += 1
            ann_id += 1
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": compact,
                "bbox": [x / SCALE, y / SCALE, w / SCALE, h / SCALE],
                "area": (w / SCALE) * (h / SCALE),
                "iscrowd": 0,
            })

    categories = [
        {"id": cid, "name": COMPACT_NAMES[cid], "supercategory": "object"}
        for cid in sorted(COMPACT_NAMES)
    ]
    out = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
        "info": {"description": f"NOD packed {camera} {split}"},
    }
    stats = {
        "images": len(images),
        "annotations": len(annotations),
        "label_hist": dict(sorted(label_hist.items())),
        "anno_dim_violations": anno_dim_violations,
    }
    return out, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annsrc", required=True, help="NOD annotations_raw dir (contains Nikon/ Sony/)")
    ap.add_argument("--out", required=True, help="output dir for nod_packed_*.json")
    ap.add_argument("--cameras", nargs="+", default=["Nikon", "Sony"], choices=list(SRC_JSON))
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--check-only", action="store_true", help="validate + stats, no write")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    grand = Counter()
    for cam in args.cameras:
        for sp in args.splits:
            print(f"=== {cam} / {sp} ===")
            out, stats = convert_one(cam, sp, args.annsrc)
            print(f"  images               : {stats['images']}")
            print(f"  annotations          : {stats['annotations']}")
            print(f"  label hist (compact) : {stats['label_hist']}")
            print(f"  anno-dim violations  : {stats['anno_dim_violations']}")
            for k, v in stats["label_hist"].items():
                grand[k] += v
            if not args.check_only:
                dst = os.path.join(args.out, f"nod_packed_{cam}_{sp}.json")
                json.dump(out, open(dst, "w"))
                print(f"  WROTE {dst}")
            else:
                print("  (check-only: not written)")
    print(f"\nGRAND label totals (compact): {dict(sorted(grand.items()))}")
    print(f"  expected names: {COMPACT_NAMES}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
convert_rod.py -- build packed-coordinate COCO jsons for ROD.

Reads:
  - json_raw_coco/{train,val,test}.json  : annotations, NATIVE ids {0,1,2,6,7},
                                            RAW coords (width 2880 x height 1856)
  - splits/{train,val,test}_split.txt     : authoritative split manifest (bare stems)

Produces (one per split):
  - rod_packed_{train,val,test}.json       : PACKED coords (1440 x 928),
                                             COMPACT ids {0,1,2,3,4},
                                             file_name rewritten .raw -> .npy

Key reconciliation logic:
  * The three input jsons disagree with the folder layout AND with the .txt
    manifest on which split a few night-* images belong to. The .txt manifest
    is AUTHORITATIVE. We pool ALL annotations across the three jsons (keyed by
    file_name, since image_id is per-file and collides), then assign each image
    to a split purely by .txt membership.
  * The .txt manifest has ONE leak: night-06979 is in both train and val.
    Resolved by precedence train > val > test (assigned to train, skipped in val).

Transforms applied per box:
  * rescale: every bbox [x,y,w,h] divided by SCALE (=2), raw -> packed coords
  * relabel: native id -> compact id via NATIVE_TO_COMPACT
  * filename: 'day-00000.raw' -> 'day-00000.npy'
  * image width/height: 2880x1856 -> 1440x928

Usage:
  python convert_rod.py --annsrc ~/data/rod_annsrc --out ~/data/rod_annsrc/packed_json
  python convert_rod.py ... --check-only      # validate inputs + print stats, no write
  python convert_rod.py ... --splits val      # process only one split
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

# ---- ROD-specific constants (modular: live here, not hardcoded across pipeline) ----
SCALE = 2                       # raw sensor dims are 2x packed dims (2x2 Bayer block)
RAW_W, RAW_H = 2880, 1856       # raw mosaic dims (for sanity assertions)
PACK_W, PACK_H = 1440, 928      # packed dims = raw / SCALE

# native ROD category id -> compact contiguous id {0..4}
# native: 0=person 1=bicycle 2=car 6=train 7=truck  (the id gap 3,4,5 is why we remap)
NATIVE_TO_COMPACT = {0: 0, 1: 1, 2: 2, 6: 3, 7: 4}

COMPACT_NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "train", 4: "truck"}

SPLIT_PRECEDENCE = ["train", "val", "test"]   # resolves duplicate-stem leaks


def load_splits(annsrc):
    """Load .txt manifests -> {split: set(stems)}, deduplicated by precedence.
    A stem in multiple splits is kept only in the highest-precedence one."""
    raw = {}
    for sp in SPLIT_PRECEDENCE:
        path = os.path.join(annsrc, f"{sp}_split.txt")
        with open(path) as f:
            raw[sp] = [l.strip() for l in f if l.strip()]
    # dedup by precedence: assign each stem to the first split that claims it
    stem_to_split = {}
    for sp in SPLIT_PRECEDENCE:
        for stem in raw[sp]:
            if stem not in stem_to_split:
                stem_to_split[stem] = sp
    # report leaks
    total_listed = sum(len(v) for v in raw.values())
    total_unique = len(stem_to_split)
    if total_listed != total_unique:
        print(f"  NOTE: {total_listed - total_unique} duplicate stem(s) across splits "
              f"resolved by precedence {SPLIT_PRECEDENCE}")
    return stem_to_split, raw


def pool_annotations(annsrc):
    """Pool annotations from all 3 jsons, keyed by file_name (image_id collides
    across files). Returns {file_name: {'width','height','anns':[(cat,x,y,w,h),...]}}."""
    pooled = {}
    for sp in SPLIT_PRECEDENCE:
        path = os.path.join(annsrc, f"{sp}.json")
        coco = json.load(open(path))
        id2name = {im["id"]: im["file_name"] for im in coco["images"]}
        id2wh = {im["id"]: (im["width"], im["height"]) for im in coco["images"]}
        for im in coco["images"]:
            fn = im["file_name"]
            if fn not in pooled:
                pooled[fn] = {"width": im["width"], "height": im["height"], "anns": []}
        for a in coco["annotations"]:
            fn = id2name[a["image_id"]]
            pooled[fn]["anns"].append((a["category_id"], *a["bbox"]))
    return pooled


def stem_of(file_name):
    """'day-00000.raw' -> 'day-00000'"""
    return os.path.splitext(file_name)[0]


def convert_split(split, stem_to_split, pooled, strict=True):
    """Build one packed COCO json dict for the given split."""
    images, annotations = [], []
    img_id = 0
    ann_id = 0
    label_hist = Counter()
    skipped_imgs = 0
    raw_coord_violations = 0

    # images belonging to this split, in sorted stem order for determinism
    split_stems = sorted(s for s, sp in stem_to_split.items() if sp == split)

    for stem in split_stems:
        # the manifest uses bare stems; annotations use '<stem>.raw'
        fn_raw = stem + ".raw"
        if fn_raw not in pooled:
            skipped_imgs += 1
            continue
        rec = pooled[fn_raw]

        # sanity: source must be in RAW coord space
        if strict and (rec["width"] != RAW_W or rec["height"] != RAW_H):
            raw_coord_violations += 1

        img_id += 1
        images.append({
            "id": img_id,
            "file_name": stem + ".npy",          # rewrite extension
            "width": PACK_W,
            "height": PACK_H,
        })

        for (cat, x, y, w, h) in rec["anns"]:
            if cat not in NATIVE_TO_COMPACT:
                raise ValueError(f"unknown native category id {cat} in {fn_raw}")
            compact = NATIVE_TO_COMPACT[cat]
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
        "info": {"description": f"ROD packed {split}", "split_source": "splits/*.txt"},
    }
    stats = {
        "images": len(images),
        "annotations": len(annotations),
        "label_hist": dict(sorted(label_hist.items())),
        "skipped_imgs": skipped_imgs,
        "raw_coord_violations": raw_coord_violations,
    }
    return out, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annsrc", required=True, help="dir with {train,val,test}.json + *_split.txt")
    ap.add_argument("--out", required=True, help="output dir for rod_packed_*.json")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--check-only", action="store_true", help="validate + stats, no write")
    args = ap.parse_args()

    print("=== loading splits ===")
    stem_to_split, raw = load_splits(args.annsrc)
    for sp in SPLIT_PRECEDENCE:
        n = sum(1 for v in stem_to_split.values() if v == sp)
        print(f"  {sp:5s}: {n} stems (after dedup)")

    print("=== pooling annotations ===")
    pooled = pool_annotations(args.annsrc)
    print(f"  pooled images: {len(pooled)}")
    print(f"  pooled annotations: {sum(len(v['anns']) for v in pooled.values())}")

    os.makedirs(args.out, exist_ok=True)
    for sp in args.splits:
        print(f"=== convert {sp} ===")
        out, stats = convert_split(sp, stem_to_split, pooled)
        print(f"  images               : {stats['images']}")
        print(f"  annotations          : {stats['annotations']}")
        print(f"  label hist (compact) : {stats['label_hist']}")
        print(f"  skipped (no anns rec): {stats['skipped_imgs']}")
        print(f"  raw-coord violations : {stats['raw_coord_violations']}")
        if not args.check_only:
            dst = os.path.join(args.out, f"rod_packed_{sp}.json")
            json.dump(out, open(dst, "w"))
            print(f"  WROTE {dst}")
        else:
            print("  (check-only: not written)")


if __name__ == "__main__":
    main()
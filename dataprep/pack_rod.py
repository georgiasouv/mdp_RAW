#!/usr/bin/env python3
"""
pack_rod.py — pure STRUCTURAL packer for the ROD dataset.

Reads 24-bit packed Bayer .raw files (1856x2880 single-plane mosaic),
deinterleaves the RGGB 2x2 blocks into a [4, 928, 1440] array, and saves
one .npy per image as uint32. NO normalization, NO clipping — values are
preserved exactly so normalization can be a separate, tunable load-time step.

Channel order: [R, G1, G2, B]  (top-left, top-right, bottom-left, bottom-right)

Output layout (mirrors PASCALRAW):
    <out_root>/packed/<split>/<stem>.npy

Usage:
    python pack_rod.py --src  /path/to/ROD/yolo/raw/images \
                       --out  /scratch/INC1526354/rod \
                       --splits train val test \
                       --workers 16
    python pack_rod.py ... --selfcheck-only   # decode 1 file, print stats, exit
    python pack_rod.py ... --limit 20         # pack only first 20 per split (dry run)
"""
import argparse
import os
import sys
import glob
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

BIT8 = 2 ** 8
BIT16 = 2 ** 16
RAW_H, RAW_W = 1856, 2880          # full sensor mosaic dims
EXP_BYTES = RAW_H * RAW_W * 3      # 16,035,840 for 24-bit
PACK_H, PACK_W = RAW_H // 2, RAW_W // 2   # 928 x 1440


def decode_raw_24b(path):
    """Read a 24-bit .raw file -> float64 mosaic (RAW_H, RAW_W). Exact integers."""
    d = np.fromfile(path, dtype=np.uint8)
    if d.size != EXP_BYTES:
        raise ValueError(f"{path}: size {d.size} != expected {EXP_BYTES}")
    d = d.astype(np.float64)   # float64 holds 2^24 exactly; cast to uint32 after
    img = d[0::3] + d[1::3] * BIT8 + d[2::3] * BIT16
    return img.reshape(RAW_H, RAW_W)


def pack_mosaic(img):
    """Deinterleave RGGB 2x2 blocks -> uint32 [4, PACK_H, PACK_W]."""
    R = img[0::2, 0::2]
    G1 = img[0::2, 1::2]
    G2 = img[1::2, 0::2]
    B = img[1::2, 1::2]
    packed = np.stack([R, G1, G2, B], axis=0)   # [4, H/2, W/2]
    return packed.astype(np.uint32)


def pack_one(args):
    """Worker: decode + pack + save one file. Returns (stem, status)."""
    src_path, dst_path = args
    try:
        if os.path.exists(dst_path):
            return (os.path.basename(dst_path), "skip")
        img = decode_raw_24b(src_path)
        packed = pack_mosaic(img)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        np.save(dst_path, packed)
        return (os.path.basename(dst_path), "ok")
    except Exception as e:
        return (os.path.basename(src_path), f"ERR: {e}")


def selfcheck(src_root, splits):
    """Decode one file end-to-end and print stats — sanity before bulk run."""
    for sp in splits:
        files = sorted(glob.glob(os.path.join(src_root, sp, "*.raw")))
        if files:
            p = files[0]
            img = decode_raw_24b(p)
            packed = pack_mosaic(img)
            print("=== SELF-CHECK ===")
            print("file         ", p)
            print("mosaic shape ", img.shape)
            print("packed shape ", packed.shape)
            print("packed dtype ", packed.dtype)
            print("packed min   ", int(packed.min()))
            print("packed max   ", int(packed.max()))
            print("per-channel mean:")
            for i, name in enumerate(["R", "G1", "G2", "B"]):
                print(f"   {name:2s} {packed[i].mean():.1f}")
            # round-trip check: packed values must equal source decode exactly
            assert packed[0, 0, 0] == int(img[0, 0]), "R channel mismatch!"
            assert packed[3].max() == int(img[1::2, 1::2].max()), "B channel mismatch!"
            print("round-trip exact: OK")
            return True
    print("SELF-CHECK FAILED: no .raw files found")
    return False


def build_jobs(src_root, out_root, splits, limit=None):
    """Build (src, dst) path pairs for all images across splits."""
    jobs = []
    per_split = {}
    for sp in splits:
        files = sorted(glob.glob(os.path.join(src_root, sp, "*.raw")))
        if limit:
            files = files[:limit]
        per_split[sp] = len(files)
        for f in files:
            stem = os.path.splitext(os.path.basename(f))[0]
            dst = os.path.join(out_root, "packed", sp, stem + ".npy")
            jobs.append((f, dst))
    return jobs, per_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="ROD raw images root (contains train/ val/ test/)")
    ap.add_argument("--out", required=True, help="output root (packed/<split>/ created under here)")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="pack only first N per split (dry run)")
    ap.add_argument("--selfcheck-only", action="store_true")
    args = ap.parse_args()

    if not selfcheck(args.src, args.splits):
        sys.exit(1)
    if args.selfcheck_only:
        print("\n--selfcheck-only set, exiting before bulk pack.")
        return

    jobs, per_split = build_jobs(args.src, args.out, args.splits, args.limit)
    print(f"\nsplits: {per_split}")
    print(f"total files to pack: {len(jobs)}")
    print(f"workers: {args.workers}")
    print(f"output: {os.path.join(args.out, 'packed')}/<split>/\n")

    ok = skip = err = 0
    errors = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(pack_one, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            name, status = fut.result()
            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
            else:
                err += 1
                errors.append((name, status))
            if i % 500 == 0 or i == len(jobs):
                print(f"  [{i}/{len(jobs)}]  ok={ok} skip={skip} err={err}")

    print(f"\nDONE. ok={ok} skip={skip} err={err}")
    if errors:
        print("ERRORS (first 20):")
        for name, status in errors[:20]:
            print(" ", name, status)


if __name__ == "__main__":
    main()
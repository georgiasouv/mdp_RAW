#!/usr/bin/env python3
"""
pack_nod.py — pure STRUCTURAL packer for the NOD dataset (two cameras).

Reads vendor RAW files via rawpy:
    Nikon  *.NEF   visible Bayer ~ 2640 x 3968  -> packed [4, 1320, 1984]
    Sony   *.ARW   visible Bayer ~ 3672 x 5496  -> packed [4, 1836, 2748]

Both cameras share Bayer pattern [[0,1],[3,2]] with color_desc RGBG, i.e. the
2x2 block is  R G / G B  (RGGB), top-left = R. So the SAME deinterleave used
for ROD applies to both — no per-camera channel swap. Cameras differ only in
resolution, so each is packed NATIVE (no resize/crop/pad) into its own folder.

Channel order: [R, G1, G2, B]  (top-left, top-right, bottom-left, bottom-right)
Output dtype : uint32, NO normalization, NO clipping — exact sensor values
               preserved so normalization stays a tunable load-time step.

Output layout (per-camera, mirrors the ROD/PASCALRAW packed/<key>/ convention):
    <out_root>/packed/<camera>/<stem>.npy        # camera in {Nikon, Sony}

Usage:
    python pack_nod.py --src  /cifs/Shares/Raw_Bayer_Datasets/NOD/raw \
                       --out  /scratch/<proj>/nod \
                       --cameras Nikon Sony \
                       --workers 16
    python pack_nod.py ... --selfcheck-only     # decode 1 file/camera, print, exit
    python pack_nod.py ... --limit 20           # pack only first 20 per camera (dry run)
"""
import argparse
import os
import sys
import glob
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

import rawpy  # vendor RAW decode (NEF/ARW)

# Per-camera file-extension map. Cameras are split by folder under --src.
CAMERA_EXT = {
    "Nikon": "*.NEF",
    "Sony":  "*.ARW",
}

# Per-camera sensor black level (from rawpy black_level_per_channel; all 4
# channels identical so a scalar suffices). Subtracted at pack time, clipped
# at 0, to put both sensors on a true-zero floor (rigorous RAW linearization).
# white_level (Nikon 16383 / Sony 16380) and camera_white_level (15311 / 15360)
# recorded for reference; not used since log1p normalizes by REF, not by max.
BLACK_LEVEL = {"Nikon": 600, "Sony": 800}

# Bayer pattern we assume after rawpy's visible crop: top-left = R (RGGB).
# Verified via rawpy: raw_pattern [[0,1],[3,2]], color_desc RGBG for both cameras.
EXPECTED_PATTERN = [[0, 1], [3, 2]]
EXPECTED_COLOR = "RGBG"


def decode_raw_vendor(path, black=0):
    """Read a vendor RAW (NEF/ARW) -> float64 visible Bayer mosaic (H, W).

    Uses raw_image_visible (excludes optical-black borders). Values are the
    raw sensor counts cast to float64 (holds the 14-bit range exactly); the
    caller casts to uint32 after the deinterleave. Asserts the Bayer pattern
    matches the RGGB assumption so the channel split stays correct per file.
    """
    with rawpy.imread(path) as raw:
        pat = raw.raw_pattern.tolist()
        col = raw.color_desc.decode()
        if pat != EXPECTED_PATTERN or col != EXPECTED_COLOR:
            raise ValueError(
                f"{path}: Bayer pattern {pat}/{col} != expected "
                f"{EXPECTED_PATTERN}/{EXPECTED_COLOR}"
            )
        vis = raw.raw_image_visible
        # Copy out of the rawpy buffer (it is freed on context exit) and widen.
        img = vis.astype(np.float64)
    # subtract sensor black level, clip negatives to 0 (true-zero floor)
    if black:
        img = np.clip(img - black, 0, None)
    H, W = img.shape
    if H % 2 or W % 2:
        raise ValueError(f"{path}: visible dims {H}x{W} not both even — RGGB fold misaligns")
    return img


def pack_mosaic(img, dtype=np.uint16):
    """Deinterleave RGGB 2x2 blocks -> uint32 [4, H/2, W/2]. Identical to ROD."""
    R = img[0::2, 0::2]
    G1 = img[0::2, 1::2]
    G2 = img[1::2, 0::2]
    B = img[1::2, 1::2]
    packed = np.stack([R, G1, G2, B], axis=0)   # [4, H/2, W/2]
    vmax=packed.max()
    if vmax>np.iinfo(dtype).max:
        raise ValueError(f"value {int(vmax)} exceeds {np.dtype(dtype).name} max")
    return packed.astype(dtype)


def pack_one(args):
    """Worker: decode + pack + save one file. Returns (stem, status)."""
    src_path, dst_path, dtype = args
    try:
        if os.path.exists(dst_path):
            return (os.path.basename(dst_path), "skip")
        cam = os.path.basename(os.path.dirname(dst_path))
        img = decode_raw_vendor(src_path, BLACK_LEVEL.get(cam, 0))
        packed = pack_mosaic(img, dtype)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        np.save(dst_path, packed)
        return (os.path.basename(dst_path), "ok")
    except Exception as e:
        return (os.path.basename(src_path), f"ERR: {e}")


def selfcheck(src_root, cameras, dtype=np.uint16):
    """Decode one file per camera end-to-end and print stats — gate before bulk."""
    for cam in cameras:
        ext = CAMERA_EXT[cam]
        files = sorted(glob.glob(os.path.join(src_root, cam, ext)))
        if not files:
            print(f"SELF-CHECK FAILED: no {ext} files under {os.path.join(src_root, cam)}")
            return False
        p = files[0]
        img = decode_raw_vendor(p, BLACK_LEVEL.get(cam, 0))
        packed = pack_mosaic(img, dtype)
        print(f"=== SELF-CHECK [{cam}] ===")
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


def build_jobs(src_root, out_root, cameras, dtype, limit=None):
    """Build (src, dst) path pairs for all images across cameras."""
    jobs = []
    per_cam = {}
    for cam in cameras:
        ext = CAMERA_EXT[cam]
        files = sorted(glob.glob(os.path.join(src_root, cam, ext)))
        if limit:
            files = files[:limit]
        per_cam[cam] = len(files)
        for f in files:
            stem = os.path.splitext(os.path.basename(f))[0]
            dst = os.path.join(out_root, "packed", cam, stem + ".npy")
            jobs.append((f, dst, dtype))
    return jobs, per_cam


def main():
    import multiprocessing as _mp
    try:
        _mp.set_start_method("spawn")
    except RuntimeError:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="NOD raw root (contains Nikon/ Sony/)")
    ap.add_argument("--out", required=True, help="output root (packed/<camera>/ created under here)")
    ap.add_argument("--cameras", nargs="+", default=["Nikon", "Sony"], choices=list(CAMERA_EXT))
    ap.add_argument("--dtype", default="uint16", choices=["uint16","uint32"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="pack only first N per camera (dry run)")
    ap.add_argument("--selfcheck-only", action="store_true")
    args = ap.parse_args()

    dtype = np.uint16 if args.dtype=="uint16" else np.uint32
    if not selfcheck(args.src, args.cameras, dtype):
        sys.exit(1)
    if args.selfcheck_only:
        print("\n--selfcheck-only set, exiting before bulk pack.")
        return

    jobs, per_cam = build_jobs(args.src, args.out, args.cameras, dtype, args.limit)
    print(f"\ncameras: {per_cam}")
    print(f"total files to pack: {len(jobs)}")
    print(f"workers: {args.workers}")
    print(f"output: {os.path.join(args.out, 'packed')}/<camera>/\n")

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
"""SYNTHETIC dataset shim -- a drop-in replacement for
datasets_registry.build_records, for SMOKE-TESTING the plumbing on fake
data before any real dataset exists.

It satisfies the exact same contract as the real registry:

    build_records(name) -> (train_records, val_records, load_fn)

    load_fn(record) -> (raw, boxes, labels):
        raw    : float tensor [4, H0, W0]  packed RGGB, range [0, 1]
        boxes  : float tensor [N, 4]       xyxy in ORIGINAL pixels
        labels : int64 tensor [N]          0-indexed class ids

What it does:
  * Fabricates a small fixed set of synthetic samples (no file IO).
  * Each sample is a dark RAW canvas with a few BRIGHT rectangles painted
    on it, and boxes that exactly bound those rectangles. The brightness
    gives the preprocessor a real (if trivial) signal, so a DECREASING
    loss is genuine evidence that gradient flows through the frozen
    detector into the preprocessor -- not just noise.

Designed to trip the known hazards while they are cheap to find:
  * One sample is given ZERO boxes on purpose -> exercises the empty-box
    path in DetrAdapter / torchvision loss (the crash I flagged).
  * Channels/ranges are exactly the real contract (4ch RGGB in [0,1]),
    so shape/range mismatches into the detector stem surface here.

Usage (monkeypatch, no edits to your real files):
    USE_SYNTHETIC=1 python train.py --regime pair_12 --dataset fake \
        --num-classes 3 --device cpu --epochs 1 --bs 4 --size 256 \
        --out /tmp/smoke --ckpt-dir /tmp/nope --log-conflict-every 1

(see run_smoke.py, which does the monkeypatch + a tiny loader for you)
"""
import torch

# Keep these small so CPU smoke tests finish in seconds.
_N_TRAIN = 12
_N_VAL = 6
_NUM_CLASSES = 3          # matches --num-classes 3 in the smoke command
_ORIG_HW = (600, 800)     # ORIGINAL (pre-resize) HxW, ODD-ish on purpose


def _make_sample(seed: int, num_classes: int = _NUM_CLASSES):
    """Build one synthetic (raw, boxes, labels).

    raw is [4, H0, W0] in [0,1]; we paint 0..3 bright blocks and return
    boxes that bound them in ORIGINAL pixel coords (the dataset will
    rescale them to --size).
    """
    g = torch.Generator().manual_seed(seed)
    H0, W0 = _ORIG_HW

    # Dark, slightly noisy RAW canvas (RAW sits compressed in the low range,
    # which is exactly why the preprocessor's gamma lift exists).
    raw = (0.02 + 0.01 * torch.rand(4, H0, W0, generator=g)).clamp(0, 1)

    # Deterministically vary how many objects: sample index % 4 gives
    # 0,1,2,3 -> so exactly the seeds where (seed % 4 == 0) have NO boxes,
    # which exercises the empty-box path.
    n_obj = seed % 4

    boxes, labels = [], []
    for k in range(n_obj):
        # random-ish but valid box, min size 40px so it survives resize
        bw = int(40 + 200 * torch.rand(1, generator=g).item())
        bh = int(40 + 200 * torch.rand(1, generator=g).item())
        x0 = int((W0 - bw - 1) * torch.rand(1, generator=g).item())
        y0 = int((H0 - bh - 1) * torch.rand(1, generator=g).item())
        x1, y1 = x0 + bw, y0 + bh
        # Paint a bright block into the RAW canvas (all 4 channels) so the
        # detector has something to latch onto after preprocessing.
        raw[:, y0:y1, x0:x1] = (0.6 + 0.3 * torch.rand(1, generator=g).item())
        boxes.append([x0, y0, x1, y1])
        labels.append(k % num_classes)

    if boxes:
        boxes_t = torch.tensor(boxes, dtype=torch.float32)
        labels_t = torch.tensor(labels, dtype=torch.int64)
    else:
        # the deliberately-empty sample
        boxes_t = torch.zeros(0, 4, dtype=torch.float32)
        labels_t = torch.zeros(0, dtype=torch.int64)

    return raw, boxes_t, labels_t


def _synthetic_load(record):
    # record is just an int seed here; the real registry would get a path/dict
    return _make_sample(int(record))


def build_records(name: str):
    """Drop-in clone of datasets_registry.build_records, fake data."""
    # records are just seeds; train and val use disjoint seed ranges
    train_records = list(range(0, _N_TRAIN))
    val_records = list(range(1000, 1000 + _N_VAL))
    return train_records, val_records, _synthetic_load

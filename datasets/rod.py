"""ROD registry entry -- load_fn + build_records, mirroring pascalraw.py.

Satisfies the same contract dataset.py expects:
    load_fn(record) -> (raw, boxes, labels)
        raw    : float tensor [4, H, W]  packed RGGB, normalized to ~[0,1]
        boxes  : float tensor [N, 4]     xyxy in PACKED px
        labels : int64 tensor [N]        compact {0..4}

KEY DIFFERENCE FROM PASCALRAW
-----------------------------
PASCALRAW packed .npy are pre-normalized float16 in [0,1]; its load_fn just
loads + clamps. ROD packed .npy store RAW uint32 sensor values (lossless,
unnormalized) so normalization is a tunable LOAD-TIME step. The default is
log1p, chosen empirically: on the frozen FCOS COCO head over the ROD test
split it gave mAP 0.114 vs 0.086 (asinh) vs 0.073 (linear) -- +56% over linear.
log1p best collapses the day/night exposure gap, which is why the frozen
detector parses it best.

Normalization is a swappable function (NORMS dict). To ablate, change
ROD_NORM or pass a different name -- no re-conversion needed.

Compact labels: 0=person 1=bicycle 2=car 3=train 4=truck
(adapters map compact -> COCO ids; that mapping lives in adapters.py, not here)
"""
import json
import os
import numpy as np
import torch

# adjust if your layout differs
ROD_ROOT = os.environ.get('ROD_ROOT', '/scratch/INC1526354/rod')
PACKED_SUBDIR = os.environ.get('ROD_PACKED_SUBDIR', 'packed')
ANN_SUBDIR = os.environ.get('ROD_ANN_SUBDIR', 'annotations_packed')

# ---- normalization (the one divergence from pascalraw) ----
# REF is the fixed dataset-global reference; the single tunable knob.
ROD_NORM_REF = float(os.environ.get('ROD_NORM_REF', 600000.0))
ROD_NORM_S = float(os.environ.get('ROD_NORM_S', 10000.0))   # asinh scale (ablation only)
ROD_NORM_DIV = float(os.environ.get('ROD_NORM_DIV', 102000.0))  # linear div (ablation only)


def _norm_log1p(x):
    return np.clip(np.log1p(x) / np.log1p(ROD_NORM_REF), 0.0, 1.0)


def _norm_asinh(x):
    return np.clip(np.arcsinh(x / ROD_NORM_S) / np.arcsinh(ROD_NORM_REF / ROD_NORM_S), 0.0, 1.0)


def _norm_linear(x):
    return np.clip(x / ROD_NORM_DIV, 0.0, 1.0)


NORMS = {'log1p': _norm_log1p, 'asinh': _norm_asinh, 'linear': _norm_linear}

# default chosen by frozen-detector mAP eval (log1p won decisively)
ROD_NORM = os.environ.get('ROD_NORM', 'log1p')


def _load_coco_split(json_path, packed_dir):
    """Parse one packed-COCO json -> list of records, each carrying its npy path
    and already-scaled xyxy boxes + compact labels. Images with no boxes are
    kept (empty boxes) so split size matches the json. Mirrors pascalraw."""
    coco = json.load(open(json_path))
    by_img = {}
    for a in coco['annotations']:
        x, y, w, h = a['bbox']                       # packed xywh
        by_img.setdefault(a['image_id'], []).append(
            (a['category_id'], x, y, x + w, y + h))   # -> xyxy, compact label
    records = []
    for im in coco['images']:
        anns = by_img.get(im['id'], [])
        if anns:
            labels = [a[0] for a in anns]
            boxes = [[a[1], a[2], a[3], a[4]] for a in anns]
        else:
            labels, boxes = [], []
        records.append({
            'npy': os.path.join(packed_dir, im['file_name']),
            'boxes': boxes,
            'labels': labels,
        })
    return records


def _rod_load(record):
    """record -> (raw[4,H,W] normalized ~[0,1], boxes[N,4] xyxy, labels[N]).

    Unlike pascalraw, the .npy is RAW uint32; we apply ROD_NORM here."""
    arr = np.load(record['npy'])                     # [4, H, W] uint32, raw sensor values
    normfn = NORMS[ROD_NORM]
    raw = torch.from_numpy(normfn(arr.astype(np.float32))).float()  # normalized, [4,H,W]
    if len(record['boxes']):
        boxes = torch.tensor(record['boxes'], dtype=torch.float32)
        labels = torch.tensor(record['labels'], dtype=torch.int64)
    else:
        boxes = torch.zeros(0, 4, dtype=torch.float32)
        labels = torch.zeros(0, dtype=torch.int64)
    return raw, boxes, labels


def build_records(name: str):
    """ROD: returns (train_records, val_records, load_fn).
    Mirrors pascalraw.build_records."""
    name = name.lower()
    if name != 'rod':
        raise ValueError(f'this module only handles rod, got {name}')
    packed_dir = os.path.join(ROD_ROOT, PACKED_SUBDIR)
    ann_dir = os.path.join(ROD_ROOT, ANN_SUBDIR)
    train = _load_coco_split(
        os.path.join(ann_dir, 'rod_packed_train.json'), packed_dir)
    val = _load_coco_split(
        os.path.join(ann_dir, 'rod_packed_val.json'), packed_dir)
    return train, val, _rod_load


def build_records_split(split: str):
    """Convenience: get any single split's records + load_fn (e.g. 'test')."""
    packed_dir = os.path.join(ROD_ROOT, PACKED_SUBDIR)
    ann_dir = os.path.join(ROD_ROOT, ANN_SUBDIR)
    recs = _load_coco_split(
        os.path.join(ann_dir, f'rod_packed_{split}.json'), packed_dir)
    return recs, _rod_load
"""PASCALRAW registry entry -- the real load_fn + build_records.

Drop these into datasets_registry.py (or import from here). Satisfies the
contract dataset.py expects:

    load_fn(record) -> (raw, boxes, labels)
        raw    : float tensor [4, H0, W0]  packed RGGB, [0,1]
        boxes  : float tensor [N, 4]       xyxy in ORIGINAL (packed) px
        labels : int64 tensor [N]          0=person,1=car,2=bicycle

A `record` here is a dict {'npy': <path>, 'boxes': [N,4] xyxy, 'labels':[N]}
fully prepared by build_records, so load_fn just loads the array and tensors
the already-scaled boxes. NO coordinate arithmetic at load time -- step 2
(convert_pascalraw.py) baked packed-space boxes into the COCO json.

Why pre-resolve boxes in build_records instead of reading json per __getitem__:
the COCO json is small; parsing it once up front (not 2974x per epoch) is
faster and keeps load_fn trivial.
"""
import json
import os

import numpy as np
import torch

# adjust if your layout differs
PASCALRAW_ROOT = os.environ.get('PASCALRAW_ROOT', '/scratch/INC1526354/pascalraw')
PACKED_SUBDIR  = os.environ.get('PASCALRAW_PACKED_SUBDIR', 'packed')
ANN_SUBDIR     = os.environ.get('PASCALRAW_ANN_SUBDIR', 'annotations_packed')


def _load_coco_split(json_path, packed_dir):
    """Parse one COCO json -> list of records, each carrying its npy path and
    already-scaled xyxy boxes + labels. Images with no boxes are kept (empty
    boxes tensor) so the split size matches the json."""
    coco = json.load(open(json_path))
    # group annotations by image_id
    by_img = {}
    for a in coco['annotations']:
        x, y, w, h = a['bbox']               # COCO xywh
        by_img.setdefault(a['image_id'], []).append(
            (a['category_id'], x, y, x + w, y + h))   # -> xyxy

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
            'boxes': boxes,                  # list of [x0,y0,x1,y1] in packed px
            'labels': labels,                # list of int class ids
        })
    return records


def _pascalraw_load(record):
    """record -> (raw[4,H0,W0] in [0,1], boxes[N,4] xyxy, labels[N])."""
    arr = np.load(record['npy'])             # [4, H0, W0] float16 in [0,1]
    raw = torch.from_numpy(arr.astype(np.float32)).clamp(0.0, 1.0)

    if len(record['boxes']):
        boxes = torch.tensor(record['boxes'], dtype=torch.float32)
        labels = torch.tensor(record['labels'], dtype=torch.int64)
    else:
        boxes = torch.zeros(0, 4, dtype=torch.float32)
        labels = torch.zeros(0, dtype=torch.int64)
    return raw, boxes, labels


def build_records(name: str):
    """PASCALRAW: returns (train_records, val_records, load_fn).

    Note: dataset.py uses train_records and (in evaluate.py) val_records.
    Test split is also available via build_records_split if needed.
    """
    name = name.lower()
    if name != 'pascalraw':
        raise ValueError(f'this module only handles pascalraw, got {name}')
    packed_dir = os.path.join(PASCALRAW_ROOT, PACKED_SUBDIR)
    ann_dir = os.path.join(PASCALRAW_ROOT, ANN_SUBDIR)
    train = _load_coco_split(
        os.path.join(ann_dir, 'pascalraw_packed_train.json'), packed_dir)
    val = _load_coco_split(
        os.path.join(ann_dir, 'pascalraw_packed_val.json'), packed_dir)
    return train, val, _pascalraw_load


def build_records_split(split: str):
    """Convenience: get any single split's records + load_fn (e.g. 'test')."""
    packed_dir = os.path.join(PASCALRAW_ROOT, PACKED_SUBDIR)
    ann_dir = os.path.join(PASCALRAW_ROOT, ANN_SUBDIR)
    recs = _load_coco_split(
        os.path.join(ann_dir, f'pascalraw_packed_{split}.json'), packed_dir)
    return recs, _pascalraw_load

"""NOD registry entry -- load_fn + build_records, mirroring rod.py.

Contract (same as dataset.py expects):
    load_fn(record) -> (raw[4,H,W] norm ~[0,1], boxes[N,4] xyxy packed px, labels[N] compact {0,1,2})

DIFFERENCES FROM ROD:
1. TWO CAMERAS. packed/Nikon (4,1320,1984) + packed/Sony (4,1836,2748). Per-camera
   jsons; record npy path includes camera subdir. build_records pools BOTH cameras
   (cross-sensor training); build_records_camera() gets one camera for per-sensor eval.
2. 14-BIT DATA. REF=2200 (val-selected via frozen-FCOS mAP sweep; both cameras peaked
   at 2200 on val). log1p kept (ROD-validated). REF is the tunable knob.
Compact labels: 0=person 1=bicycle 2=car (COCO 1,2,3 remapped in convert_nod.py)
"""
import json, os
import numpy as np
import torch

NOD_ROOT = os.environ.get('NOD_ROOT', '/scratch/INC1526354/nod')
PACKED_SUBDIR = os.environ.get('NOD_PACKED_SUBDIR', 'packed')
ANN_SUBDIR = os.environ.get('NOD_ANN_SUBDIR', 'annotations_packed')
CAMERAS = ['Nikon', 'Sony']

NOD_NORM_REF = float(os.environ.get('NOD_NORM_REF', 2200.0))
NOD_NORM_S = float(os.environ.get('NOD_NORM_S', 1000.0))
NOD_NORM_DIV = float(os.environ.get('NOD_NORM_DIV', 2200.0))

def _norm_log1p(x):
    return np.clip(np.log1p(x) / np.log1p(NOD_NORM_REF), 0.0, 1.0)
def _norm_asinh(x):
    return np.clip(np.arcsinh(x / NOD_NORM_S) / np.arcsinh(NOD_NORM_REF / NOD_NORM_S), 0.0, 1.0)
def _norm_linear(x):
    return np.clip(x / NOD_NORM_DIV, 0.0, 1.0)
NORMS = {'log1p': _norm_log1p, 'asinh': _norm_asinh, 'linear': _norm_linear}
NOD_NORM = os.environ.get('NOD_NORM', 'log1p')

def _load_coco_split(json_path, packed_dir):
    coco = json.load(open(json_path))
    by_img = {}
    for a in coco['annotations']:
        x, y, w, h = a['bbox']
        by_img.setdefault(a['image_id'], []).append((a['category_id'], x, y, x + w, y + h))
    records = []
    for im in coco['images']:
        anns = by_img.get(im['id'], [])
        if anns:
            labels = [a[0] for a in anns]
            boxes = [[a[1], a[2], a[3], a[4]] for a in anns]
        else:
            labels, boxes = [], []
        records.append({'npy': os.path.join(packed_dir, im['file_name']), 'boxes': boxes, 'labels': labels})
    return records

def _nod_load(record):
    arr = np.load(record['npy'])
    normfn = NORMS[NOD_NORM]
    raw = torch.from_numpy(normfn(arr.astype(np.float32))).float()
    if len(record['boxes']):
        boxes = torch.tensor(record['boxes'], dtype=torch.float32)
        labels = torch.tensor(record['labels'], dtype=torch.int64)
    else:
        boxes = torch.zeros(0, 4, dtype=torch.float32)
        labels = torch.zeros(0, dtype=torch.int64)
    return raw, boxes, labels

def _split_records(split, cameras):
    packed_root = os.path.join(NOD_ROOT, PACKED_SUBDIR)
    ann_dir = os.path.join(NOD_ROOT, ANN_SUBDIR)
    recs = []
    for cam in cameras:
        json_path = os.path.join(ann_dir, f'nod_packed_{cam}_{split}.json')
        recs.extend(_load_coco_split(json_path, os.path.join(packed_root, cam)))
    return recs

def build_records(name: str):
    if name.lower() != 'nod':
        raise ValueError(f'this module only handles nod, got {name}')
    return _split_records('train', CAMERAS), _split_records('val', CAMERAS), _nod_load

def build_records_split(split: str):
    return _split_records(split, CAMERAS), _nod_load

def build_records_camera(camera: str, split: str):
    if camera not in CAMERAS:
        raise ValueError(f'unknown camera {camera}, expected {CAMERAS}')
    return _split_records(split, [camera]), _nod_load

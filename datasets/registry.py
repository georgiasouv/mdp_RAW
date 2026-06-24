"""Dataset dispatcher.
Maps a dataset name to its (train_records, val_records, load_fn). Each
dataset lives in its OWN module under datasets/ and is imported lazily here,
so adding a dataset = adding one module + one branch. The contract every
dataset module must satisfy:
    build_records() -> (train_records, val_records, load_fn)
    load_fn(record) -> (raw, boxes, labels):
        raw    : float tensor [4, H0, W0]  packed RGGB, [0, 1]
        boxes  : float tensor [N, 4]       xyxy in ORIGINAL (packed) px
        labels : int64 tensor [N]          0-indexed class ids for THIS dataset
"""
# ---------------------------------------------------------------------------
# Per-dataset class mapping: local 0-indexed id -> detector class id.
#
# WHY: detectors don't speak each dataset's compact {0,1,2,...} label space.
# Frozen COCO detectors emit COCO ids (person=1, car=3, ...). So each dataset
# declares how ITS labels map to the id space the detectors expect.
#
#   - COCO-subset datasets (PASCALRAW): map -> COCO ids, detectors stay frozen
#     on their pretrained COCO heads (build_detector weights_path=None).
#   - Non-COCO datasets (ROD, etc.): map -> the FINE-TUNED detector's id space,
#     and build_detector loads that checkpoint (weights_path set). The map is
#     still just local->detector-id; only the target space differs.
#
# Adding a dataset = add its module + its branch in build_records + its entry
# here. Nothing in the detector/eval code changes.
# ---------------------------------------------------------------------------
CLASS_MAPS = {
    # PASCALRAW local ids -> COCO ids
    #   0 person -> COCO 1 ;  1 car -> COCO 3 ;  2 bicycle -> COCO 2
    'pascalraw': {0: 1, 1: 3, 2: 2},
    # ROD local (compact) ids -> COCO ids. All 5 classes are genuine COCO
    # classes, so ROD is a COCO-subset case: detectors stay frozen on their
    # pretrained COCO heads (weights_path=None), no fine-tuning.
    #   0 person->1, 1 bicycle->2, 2 car->3, 3 train->7, 4 truck->8
    'rod': {0: 1, 1: 2, 2: 3, 3: 7, 4: 8},
    # 'nod':  {...},   # fill when prepared
    # 'lod':  {...},
    # 'aodraw': {...},
}


def get_class_map(name: str) -> dict:
    """local-dataset-id -> detector-id map for `name`. None if unset
    (i.e. dataset already speaks the detector's id space)."""
    return CLASS_MAPS.get(name.lower())


def build_records(name: str):
    name = name.lower()
    if name == 'pascalraw':
        from datasets.pascalraw import build_records as _pr
        return _pr('pascalraw')
    elif name == 'rod':
        from datasets.rod import build_records as _rod
        return _rod('rod')
    # elif name == 'nod':
    #     from datasets.nod import build_records as _nod
    #     return _nod('nod')
    # elif name == 'lod': ...
    # elif name == 'aodraw': ...
    raise ValueError(f'no registry entry for dataset: {name}')
"""Dataset dispatcher.

Maps a dataset name to its (train_records, val_records, load_fn). Each
dataset lives in its OWN module under datasets/ and is imported lazily here,
so adding a dataset = adding one module + one branch.
"""

CLASS_MAPS = {
    'pascalraw': {0: 1, 1: 3, 2: 2},
    'rod': {0: 1, 1: 2, 2: 3, 3: 7, 4: 8},
}

CLASS_NAMES = {
    'pascalraw': {0: 'person', 1: 'car', 2: 'bicycle'},
    'rod': {0: 'person', 1: 'bicycle', 2: 'car', 3: 'train', 4: 'truck'},
}


def get_class_map(name: str) -> dict:
    return CLASS_MAPS.get(name.lower())


def get_class_names(name: str) -> dict:
    return CLASS_NAMES.get(name.lower())


def build_records(name: str):
    name = name.lower()
    if name == 'pascalraw':
        from datasets.pascalraw import build_records as _pr
        return _pr('pascalraw')
    elif name == 'rod':
        from datasets.rod import build_records as _rod
        return _rod('rod')
    raise ValueError(f'no registry entry for dataset: {name}')


def build_records_split(name: str, split: str):
    """Per-split records + load_fn for `name`, dispatching to each dataset's
    own build_records_split. Mirrors build_records' dispatch so eval loads the
    correct dataset (not a hardcoded one). Returns (records, load_fn)."""
    name = name.lower()
    if name == 'pascalraw':
        from datasets.pascalraw import build_records_split as _pr
        return _pr(split)
    elif name == 'rod':
        from datasets.rod import build_records_split as _rod
        return _rod(split)
    raise ValueError(f'no registry entry for dataset: {name}')

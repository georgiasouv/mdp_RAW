"""Generic RAW detection dataset.

Everything dataset-AGNOSTIC lives here: fixed-size resize (so batches are
rectangular and the cluster runs fast), box rescaling, batching.

Everything dataset-SPECIFIC -- the actual file reading for PASCALRAW /
NOD / ROD / LOD / AODRAW, which all sit on disk differently -- is supplied
as a `load_fn(record)` by datasets_registry.py. This is the only piece you
write; the contract is small and explicit (see below).
"""
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class RawDetectionDataset(Dataset):
    """
    load_fn(record) MUST return:
        raw    : float tensor [4, H0, W0] packed RGGB, range [0, 1]
        boxes  : float tensor [N, 4] xyxy in ORIGINAL pixels
        labels : int64 tensor [N]
    """
    def __init__(self, records, load_fn, size=(512, 512)):
        self.records = records
        self.load_fn = load_fn
        self.size = size

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        raw, boxes, labels = self.load_fn(self.records[index])
        _, H0, W0 = raw.shape
        Ht, Wt = self.size
        raw = F.interpolate(raw.unsqueeze(0), size=(Ht, Wt),
                            mode='bilinear', align_corners=False).squeeze(0)
        if boxes.numel():
            boxes = boxes.clone()
            boxes[:, [0, 2]] *= (Wt / W0)
            boxes[:, [1, 3]] *= (Ht / H0)
        return raw, {'boxes': boxes, 'labels': labels}


def collate(batch):
    raws = torch.stack([b[0] for b in batch], 0)   # [B, 4, H, W]
    targets = [b[1] for b in batch]
    return raws, targets

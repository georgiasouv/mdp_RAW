"""Detector adapters.

The whole multi-detector idea needs ONE uniform interface over detectors
that internally disagree about everything (input range, box format,
normalization, CLASS-ID SPACE). Each adapter hides those quirks behind:

    adapter.loss(images, targets) -> scalar loss

Contract:
    images  : [B, 3, H, W] float in [0, 1]  (the preprocessor's output)
    targets : list (len B) of {'boxes': [N,4] xyxy ABSOLUTE px,
                               'labels': [N] int64 LOCAL dataset ids (0-indexed)}

Class-id remapping:
    Targets carry the dataset's LOCAL 0-indexed labels. Detectors expect
    their OWN id space (frozen COCO heads -> COCO ids; fine-tuned heads ->
    that checkpoint's ids). Each adapter is given a `class_map`
    (local_id -> detector_id) at build time and applies it to labels inside
    loss()/the eval path. class_map=None means "labels already in the
    detector's space" (identity). This keeps the mapping a per-DATASET
    property (declared in datasets/registry.py), not hardcoded in detectors.

Key trick that makes the whole project work:
    We freeze detector parameters (requires_grad_(False)) and freeze BN
    stats (BN -> eval), but we still call the detector's *training* loss
    path. PyTorch autograd happily propagates gradient THROUGH the frozen
    detector back to `images` (and thus into the preprocessor), because a
    frozen leaf only stops grad for ITSELF, not for upstream tensors.
    So: detectors don't learn, the preprocessor does.
"""
import torch
import torch.nn as nn
import torchvision

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def remap_labels(labels, class_map):
    """Map LOCAL dataset ids -> detector ids via class_map (a dict).
    class_map=None -> identity. Unmapped ids raise, to fail loud rather
    than silently mislabel."""
    if class_map is None:
        return labels
    out = labels.clone()
    for local_id, det_id in class_map.items():
        out[labels == local_id] = det_id
    # sanity: every label must have been a known local id
    known = torch.zeros_like(labels, dtype=torch.bool)
    for local_id in class_map:
        known |= (labels == local_id)
    if not bool(known.all()):
        bad = labels[~known].unique().tolist()
        raise ValueError(f'labels {bad} not in class_map keys {list(class_map)}')
    return out


class DetectorAdapter(nn.Module):
    def __init__(self, model: nn.Module, class_map: dict = None):
        super().__init__()
        self.model = model
        self.class_map = class_map      # local_id -> detector_id (or None)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self._freeze_bn()

    def _freeze_bn(self):
        for m in self.model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                m.eval()

    def train(self, mode: bool = True):
        # Even when the harness calls .train(), keep BN frozen so the
        # running stats of the off-the-shelf detector are not corrupted.
        super().train(mode)
        self._freeze_bn()
        return self

    def loss(self, images, targets):
        raise NotImplementedError


class TorchvisionAdapter(DetectorAdapter):
    """Faster R-CNN / RetinaNet / FCOS.

    These accept a LIST of [0,1] image tensors plus targets with xyxy
    absolute boxes, and in training mode return a dict of losses. They
    normalize + resize internally, so we hand them [0,1] images as-is.
    """
    def loss(self, images, targets):
        # torchvision detectors return a loss DICT only in training mode; in
        # eval mode they return predictions (a list), which breaks the line
        # below. Self-heal + assert with a clear message rather than crashing
        # cryptically downstream.
        if not self.model.training:
            self.model.train()
            self._freeze_bn()   # keep BN frozen even after flipping to train
        # remap LOCAL labels -> detector (COCO) ids
        targets = [{'boxes': t['boxes'],
                    'labels': remap_labels(t['labels'], self.class_map)}
                   for t in targets]
        imgs = [img for img in images]            # list of [3,H,W]
        loss_dict = self.model(imgs, targets)     # needs train() mode
        if not isinstance(loss_dict, dict):
            raise RuntimeError(
                f'{type(self.model).__name__} returned {type(loss_dict).__name__}, '
                f'not a loss dict -- detector is not in training mode. This '
                f'usually means an eval pass (e.g. the val probe) left it in '
                f'eval mode without restoring. Ensure evaluate_map restores '
                f'detector training mode.')
        return sum(loss_dict.values())


class DetrAdapter(DetectorAdapter):
    """HF DetrForObjectDetection / DeformableDetr.

    Wants normalized pixel_values [B,3,H,W] and labels carrying
    NORMALIZED cxcywh boxes. We convert here.
    """
    def loss(self, images, targets):
        mean = images.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = images.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
        pixel_values = (images - mean) / std
        _, _, H, W = images.shape
        labels = []
        for t in targets:
            b = t['boxes']
            cx = (b[:, 0] + b[:, 2]) / 2 / W
            cy = (b[:, 1] + b[:, 3]) / 2 / H
            bw = (b[:, 2] - b[:, 0]) / W
            bh = (b[:, 3] - b[:, 1]) / H
            labels.append({'class_labels': remap_labels(t['labels'], self.class_map),
                           'boxes': torch.stack([cx, cy, bw, bh], dim=1)})
        return self.model(pixel_values=pixel_values, labels=labels).loss


def build_detector(name: str, num_classes: int,
                   weights_path: str = None, device: str = 'cuda',
                   class_map: dict = None) -> DetectorAdapter:
    """Build a detector, optionally load a fine-tuned-on-RAW checkpoint,
    wrap + freeze.

    name in {'fasterrcnn', 'retinanet', 'fcos', 'detr'}.
    class_map : local-dataset-id -> detector-id, applied to GT labels in
                loss()/eval. None = identity. Comes from registry.get_class_map.

    GOTCHA: torchvision Faster R-CNN counts background, so it needs
    num_classes + 1; RetinaNet/FCOS do not. If weights_path is None we
    use the COCO-pretrained head (91 classes) directly -- only valid when
    your dataset's classes map to COCO (the class_map handles the id
    translation). Otherwise pass a fine-tuned checkpoint.

    DETR: when weights_path is None we keep the FULL pretrained COCO head
    (91 classes). We deliberately do NOT pass num_labels/ignore_mismatched
    here -- doing so replaces the classifier with a RANDOM head, which
    destroys the pretrained detection knowledge the frozen-detector premise
    relies on. The class_map translates dataset labels into COCO ids instead.
    """
    if name == 'fasterrcnn':
        m = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=None if weights_path else 'DEFAULT',
            num_classes=(num_classes + 1) if weights_path else 91)
        adapter_cls = TorchvisionAdapter
    elif name == 'retinanet':
        m = torchvision.models.detection.retinanet_resnet50_fpn(
            weights=None if weights_path else 'DEFAULT',
            num_classes=num_classes if weights_path else 91)
        adapter_cls = TorchvisionAdapter
    elif name == 'fcos':
        m = torchvision.models.detection.fcos_resnet50_fpn(
            weights=None if weights_path else 'DEFAULT',
            num_classes=num_classes if weights_path else 91)
        adapter_cls = TorchvisionAdapter
    elif name == 'detr':
        from transformers import DetrForObjectDetection
        if weights_path:
            # fine-tuned head sized to the dataset
            m = DetrForObjectDetection.from_pretrained(
                'facebook/detr-resnet-50',
                num_labels=num_classes, ignore_mismatched_sizes=True)
        else:
            # KEEP the full pretrained COCO head (91 classes) -- do NOT
            # replace it with a random num_labels head. class_map maps
            # dataset labels -> COCO ids.
            m = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50')
        adapter_cls = DetrAdapter
    else:
        raise ValueError(f'unknown detector: {name}')

    if weights_path:
        sd = torch.load(weights_path, map_location='cpu')
        m.load_state_dict(sd.get('model', sd), strict=False)

    return adapter_cls(m.to(device), class_map=class_map).to(device)

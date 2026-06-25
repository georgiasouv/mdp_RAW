"""Shared evaluation core -- ONE implementation of "preprocessor + detector
-> mAP", used by BOTH evaluate.py (final results) and the training-time
convergence probe. Sharing matters: if the probe computed mAP differently
from final eval, it would give a misleading convergence signal.

Class-id handling (mirror of adapters.py): detectors PREDICT in their own id
space (frozen COCO heads -> COCO ids). To score against the dataset's LOCAL
0-indexed GT, evaluate_map takes the same `class_map` (local_id -> detector_id)
and applies its INVERSE to predictions: keep only boxes whose predicted id is
a mapped detector id (drop COCO's other classes), then translate that id back
to the local id. class_map=None -> identity (labels already local).
"""
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_convert


# The COCO-style metric keys we surface (the full YOLO/MMDetection table).
# torchmetrics' MeanAveragePrecision computes all of these; we just keep them.
COCO_METRIC_KEYS = (
    'map', 'map_50', 'map_75',
    'map_small', 'map_medium', 'map_large',
    'mar_1', 'mar_10', 'mar_100',
    'mar_small', 'mar_medium', 'mar_large',
)


def filter_and_invert_preds(pred, class_map):
    """Keep only predictions whose label is a mapped detector id, then map
    that detector id back to the LOCAL dataset id. class_map=None -> identity.

    pred : {'boxes':[N,4], 'scores':[N], 'labels':[N]} in DETECTOR id space.
    returns the same dict filtered + relabelled into LOCAL id space.
    """
    if class_map is None:
        return pred
    inv = {det_id: local_id for local_id, det_id in class_map.items()}
    labels = pred['labels']
    # mask: keep predictions whose detector id is one we care about
    keep = torch.zeros_like(labels, dtype=torch.bool)
    for det_id in inv:
        keep |= (labels == det_id)
    boxes = pred['boxes'][keep]
    scores = pred['scores'][keep]
    kept_labels = labels[keep]
    # translate detector ids -> local ids
    local = kept_labels.clone()
    for det_id, local_id in inv.items():
        local[kept_labels == det_id] = local_id
    return {'boxes': boxes, 'scores': scores, 'labels': local}


@torch.no_grad()
def predict_torchvision(model, images):
    model.eval()
    return model(list(images))      # list of {'boxes','labels','scores'}


@torch.no_grad()
def predict_detr(model, images):
    model.eval()
    mean = images.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = images.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    out = model(pixel_values=(images - mean) / std)
    _, _, H, W = images.shape
    probs = out.logits.softmax(-1)[..., :-1]    # drop no-object slot
    preds = []
    for b in range(images.shape[0]):
        scr, lbl = probs[b].max(-1)
        bx = box_convert(out.pred_boxes[b], 'cxcywh', 'xyxy')
        bx = bx * torch.tensor([W, H, W, H], device=bx.device)
        preds.append({'boxes': bx, 'scores': scr, 'labels': lbl})
    return preds


def predict_fn_for(detector_name):
    return predict_detr if detector_name == 'detr' else predict_torchvision


@torch.no_grad()
def evaluate_map(prep, det_model, detector_name, dl, device, class_map=None,
                 class_names=None):
    """Run prep -> detector over a dataloader, return the FULL COCO metric dict
    (map, map_50, map_75, map_small/medium/large, mar_*). See COCO_METRIC_KEYS.

    prep        : trained RawPreprocessor (set to eval by caller or here)
    det_model   : the RAW detector model (adapter.model), NOT the adapter
    detector_name : for choosing the prediction path
    dl          : val/test DataLoader yielding (raw[B,4,H,W], targets)
    class_map   : local_id -> detector_id (from registry.get_class_map). Its
                  inverse is applied to predictions so they score against LOCAL
                  GT. None = predictions already in local id space.
    class_names : optional {local_id: name} for labelling per-class AP in the
                  output. None -> per-class rows are labelled by integer id.
    Restores BOTH prep and det_model to their prior train/eval mode on exit.
    Critical: torchvision detectors return a loss dict in train mode but
    predictions in eval mode, so leaving the detector in eval would crash
    the next training step (sum(predictions.values()) -> AttributeError).

    Per-class AP: the metric is built with class_metrics=True so torchmetrics
    returns map_per_class + the matching classes tensor. We surface those as
    res['map_per_class'] (list[float]) and res['classes'] (list[int]), so a
    shared-preprocessor regime that helps one class at another's expense is
    visible rather than hidden inside the pooled mean.
    """
    prep_was_training = prep.training
    det_was_training = det_model.training
    prep.eval()
    metric = MeanAveragePrecision(box_format='xyxy', class_metrics=True)
    predict = predict_fn_for(detector_name)
    for raw, targets in dl:
        raw = raw.to(device)
        rgb = prep(raw)
        preds = predict(det_model, rgb)
        # filter to mapped detector ids + translate back to LOCAL ids
        preds = [filter_and_invert_preds({k: v.detach() for k, v in p.items()},
                                         class_map) for p in preds]
        tgt = [{'boxes': t['boxes'].to(device),
                'labels': t['labels'].to(device)} for t in targets]
        metric.update(preds, tgt)
    computed = metric.compute()
    # keep the full COCO breakdown; cast tensors to float (some are -1 when a
    # size bucket has no GT objects -- COCO's "not applicable" sentinel).
    res = {k: float(computed[k]) for k in COCO_METRIC_KEYS if k in computed}
    # per-class AP: torchmetrics returns map_per_class (one AP per class seen)
    # and classes (the matching local ids, since preds/GT are in local space).
    # With a single class present these can be 0-dim scalars, so normalise to
    # lists. classes are ints (the local dataset ids).
    if 'map_per_class' in computed and 'classes' in computed:
        per = computed['map_per_class']
        cls = computed['classes']
        res['map_per_class'] = [float(x) for x in per.reshape(-1)]
        res['classes'] = [int(x) for x in cls.reshape(-1)]
        if class_names is not None:
            res['class_names'] = [class_names.get(c, str(c)) for c in res['classes']]
    # restore prior modes -- probe must not alter training state
    if prep_was_training:
        prep.train()
    if det_was_training:
        det_model.train()
    return res


def format_coco_table(metrics, indent='    '):
    """Render a metrics dict as a compact YOLO/MMDetection-style table string.
    Values of -1 (size bucket absent in this split) are shown as 'n/a'.
    If per-class AP is present (map_per_class), a per-class line is appended,
    labelled by class_names when available else by integer class id.
    """
    def fmt(k):
        v = metrics.get(k)
        if v is None:
            return '  -  '
        if v < 0:
            return ' n/a '
        return f'{v:.4f}'

    lines = [
        f"{indent}{'IoU=0.50:0.95':>14} | {'IoU=0.50':>10} | {'IoU=0.75':>10}",
        f"{indent}{'AP  '+fmt('map'):>14} | {fmt('map_50'):>10} | {fmt('map_75'):>10}",
        f"{indent}{'by size:':>14}   small={fmt('map_small')}  "
        f"medium={fmt('map_medium')}  large={fmt('map_large')}",
        f"{indent}{'AR  ':>14}   @1={fmt('mar_1')}  @10={fmt('mar_10')}  "
        f"@100={fmt('mar_100')}",
        f"{indent}{'AR by size:':>14} small={fmt('mar_small')}  "
        f"medium={fmt('mar_medium')}  large={fmt('mar_large')}",
    ]

    # per-class AP line (only if computed). Labels come from class_names if the
    # caller passed them through evaluate_map, else fall back to integer ids.
    per = metrics.get('map_per_class')
    if per is not None:
        names = metrics.get('class_names')
        ids = metrics.get('classes', list(range(len(per))))
        labels = names if names is not None else [str(i) for i in ids]
        parts = '  '.join(
            f"{lab}={(' n/a ' if v < 0 else f'{v:.4f}')}"
            for lab, v in zip(labels, per)
        )
        lines.append(f"{indent}{'per class:':>14} {parts}")
    return '\n'.join(lines)
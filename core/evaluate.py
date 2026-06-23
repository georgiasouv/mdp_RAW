"""Evaluate a trained preprocessor in front of ONE detector -> mAP json.

This is the atom of the readout. Run it for every (trained-regime,
eval-detector) pair you care about. Crucially, the leave-one-detector-out
result is just a SUBSET of these runs -- e.g. evaluate the pair_12
preprocessor in front of `detr` (the held-out family). No extra training.
"""
import argparse
import json
import os

import torch
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_convert

from core.preprocessor import RawPreprocessor
from core.adapters import build_detector
from datasets.dataset import RawDetectionDataset, collate
from core.eval_core import predict_torchvision, predict_detr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prep', required=True, help='preprocessor.pth from a train run')
    ap.add_argument('--detector', required=True,
                    choices=['fasterrcnn', 'retinanet', 'fcos', 'detr'])
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--num-classes', type=int, required=True)
    ap.add_argument('--ckpt-dir', default='./detectors')
    ap.add_argument('--size', type=int, default=512)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', required=True)
    ap.add_argument('--wandb', action='store_true',
                    help='log mAP to Weights & Biases (off by default)')
    ap.add_argument('--wandb-project', default='mdp-raw-preprocessing')
    ap.add_argument('--wandb-name', default=None,
                    help='run name to log under; default {dataset}__{regime}__eval_{detector}')
    args = ap.parse_args()

    from datasets.registry import build_records
    _train, val_records, load_fn = build_records(args.dataset)
    ds = RawDetectionDataset(val_records, load_fn, size=(args.size, args.size))
    dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=8, collate_fn=collate)

    ck = torch.load(args.prep, map_location='cpu')
    prep = RawPreprocessor().to(args.device)
    prep.load_state_dict(ck['model'])
    prep.eval()
    regime = ck.get('regime', 'unknown')

    ckpt = os.path.join(args.ckpt_dir, f'{args.dataset}_{args.detector}.pth')
    ckpt = ckpt if os.path.exists(ckpt) else None
    det = build_detector(args.detector, args.num_classes, ckpt, args.device).model

    metric = MeanAveragePrecision(box_format='xyxy')
    predict = predict_detr if args.detector == 'detr' else predict_torchvision

    for raw, targets in dl:
        raw = raw.to(args.device)
        rgb = prep(raw)
        preds = predict(det, rgb)
        tgt = [{'boxes': t['boxes'].to(args.device),
                'labels': t['labels'].to(args.device)} for t in targets]
        metric.update([{k: v.detach() for k, v in p.items()} for p in preds], tgt)

    res = {k: float(v) for k, v in metric.compute().items()
           if k in ('map', 'map_50', 'map_75')}
    res.update({'regime': regime, 'detector': args.detector, 'dataset': args.dataset})
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(res, f, indent=2)
    print(res)

    if args.wandb:
        import wandb
        name = args.wandb_name or f'{args.dataset}__{regime}__eval_{args.detector}'
        wandb.init(project=args.wandb_project, name=name, config=vars(args))
        wandb.log({f'mAP/{args.detector}/map': res['map'],
                   f'mAP/{args.detector}/map_50': res['map_50'],
                   f'mAP/{args.detector}/map_75': res['map_75']})
        wandb.finish()


if __name__ == '__main__':
    main()
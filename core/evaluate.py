"""Evaluate a trained preprocessor in front of ONE detector -> mAP json.
Evaluates on the TEST split by default (held-out final results). class_map
handling mirrors eval_core.evaluate_map.
"""
import argparse
import json
import os
import torch
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from core.preprocessor import RawPreprocessor
from core.adapters import build_detector
from datasets.dataset import RawDetectionDataset, collate
from core.eval_core import predict_torchvision, predict_detr, filter_and_invert_preds
from datasets.registry import get_class_map
from datasets.registry import build_records_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prep', required=True)
    ap.add_argument('--detector', required=True,
                    choices=['fasterrcnn', 'retinanet', 'fcos', 'detr'])
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--num-classes', type=int, required=True)
    ap.add_argument('--split', default='test', choices=['train', 'val', 'test'],
                    help='split to evaluate on. Default test (held-out final).')
    ap.add_argument('--ckpt-dir', default='./detectors')
    ap.add_argument('--size', type=int, default=512)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', required=True)
    ap.add_argument('--wandb', action='store_true')
    ap.add_argument('--wandb-project', default='mdp-raw-preprocessing')
    ap.add_argument('--wandb-name', default=None)
    args = ap.parse_args()

    eval_records, load_fn = build_records_split(args.dataset, args.split)
    ds = RawDetectionDataset(eval_records, load_fn, size=(args.size, args.size))
    dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=8, collate_fn=collate)

    class_map = get_class_map(args.dataset)

    ck = torch.load(args.prep, map_location='cpu')
    prep = RawPreprocessor().to(args.device)
    prep.load_state_dict(ck['model'])
    prep.eval()
    regime = ck.get('regime', 'unknown')

    ckpt = os.path.join(args.ckpt_dir, f'{args.dataset}_{args.detector}.pth')
    ckpt = ckpt if os.path.exists(ckpt) else None
    det = build_detector(args.detector, args.num_classes, ckpt, args.device,
                         class_map=class_map).model

    metric = MeanAveragePrecision(box_format='xyxy')
    predict = predict_detr if args.detector == 'detr' else predict_torchvision
    for raw, targets in dl:
        raw = raw.to(args.device)
        rgb = prep(raw)
        preds = predict(det, rgb)
        preds = [filter_and_invert_preds({k: v.detach() for k, v in p.items()},
                                         class_map) for p in preds]
        tgt = [{'boxes': t['boxes'].to(args.device),
                'labels': t['labels'].to(args.device)} for t in targets]
        metric.update(preds, tgt)

    res = {k: float(v) for k, v in metric.compute().items()
           if k in ('map', 'map_50', 'map_75')}
    res.update({'regime': regime, 'detector': args.detector,
                'dataset': args.dataset, 'split': args.split})
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

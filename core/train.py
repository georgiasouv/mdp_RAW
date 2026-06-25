"""Train ONE preprocessor against ONE regime (a set of frozen detectors)
on ONE dataset. One run == one (regime x dataset) cell of the grid.

The regime is the independent variable of the whole study:
  solo_*  : single detector            -> baselines + LOO comparators
  pair_*  : two heterogeneous detectors -> the leave-one-detector-out
            generalizers (evaluated later on the held-out third family)
  triple  : all three                   -> the full-ensemble processor
  homo_*  : same family x3              -> the control that separates
            "more gradient magnitude" from "detector DIVERSITY".
            (Identical frozen copies give identical grads, so cosine==1
            and the only thing that changes vs solo is gradient scale.)

RESUME: pass --resume PATH to continue from a checkpoint. The checkpoint now
stores optimizer state too, so resuming continues AdamW's momentum/variance
rather than cold-starting (which would lurch). Resuming to a higher --epochs
extends a FIXED budget uniformly; do it for ALL regimes together to keep the
cross-regime comparison unconfounded (never extend just one regime).
"""
import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from core.preprocessor import RawPreprocessor
from core.adapters import build_detector
from datasets.dataset import RawDetectionDataset, collate
from core.conflict import pairwise_cosine
from core.combine import combine

REGIMES = {
    'solo_T':  ['detr'],
    'solo_1':  ['fcos'],
    'solo_2':  ['fasterrcnn'],
    'pair_T1': ['detr', 'fcos'],
    'pair_T2': ['detr', 'fasterrcnn'],
    'pair_12': ['fcos', 'fasterrcnn'],
    'triple':  ['detr', 'fcos', 'fasterrcnn'],
    'homo_T':  ['detr', 'detr', 'detr'],
    'homo_1':  ['fcos', 'fcos', 'fcos'],
    'homo_2':  ['fasterrcnn', 'fasterrcnn', 'fasterrcnn'],
}


def to_device(targets, device):
    return [{k: v.to(device) for k, v in t.items()} for t in targets]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--regime', required=True, choices=list(REGIMES))
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--num-classes', type=int, required=True)
    ap.add_argument('--ckpt-dir', default='./detectors',
                    help='dir holding {dataset}_{name}.pth frozen detectors')
    ap.add_argument('--out', required=True)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--size', type=int, default=512)
    ap.add_argument('--log-conflict-every', type=int, default=50)
    ap.add_argument('--resume', default=None,
                    help='path to a preprocessor.pth to resume from. Restores '
                         'model + optimizer + epoch so AdamW momentum/variance '
                         'continue (no cold-start lurch). Training runs until '
                         '--epochs total; to EXTEND a finished run, point '
                         '--resume at it and raise --epochs. Extend ALL regimes '
                         'together to keep the comparison fair.')
    ap.add_argument('--combine', default='normgrad',
                    choices=['sum', 'normgrad', 'mgda', 'pcgrad', 'cagrad'],
                    help="how to merge detector gradients. 'normgrad' "
                         "(default) equalizes per-detector influence; 'sum' "
                         "is the naive baseline for the ablation.")
    ap.add_argument('--val-every', type=int, default=1,
                    help="run an OBSERVATIONAL val-mAP probe every N epochs "
                         "(default 1 = every epoch, so the convergence curve "
                         "is always visible). METHODOLOGICAL CONTRACT: the "
                         "probe (1) only evaluates the TRAINING detector(s), "
                         "NEVER a held-out detector, and (2) is purely "
                         "diagnostic -- it does NOT select checkpoints or "
                         "trigger early stopping. Training uses a FIXED epoch "
                         "budget identical across all regimes, so cross-regime "
                         "comparisons are not confounded by regime-dependent "
                         "model selection. Set 0 to disable.")
    ap.add_argument('--val-max-batches', type=int, default=0,
                    help='cap val batches for a faster probe (0 = full val set)')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--wandb', action='store_true',
                    help='log to Weights & Biases (off by default)')
    ap.add_argument('--wandb-project', default='mdp-raw-preprocessing')
    ap.add_argument('--wandb-entity', default=None,
                    help='wandb entity; if init fails, falls back to offline')
    ap.add_argument('--wandb-name', default=None,
                    help='run name; defaults to {dataset}__{regime}')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    use_wandb = args.wandb
    if use_wandb:
        import wandb
        try:
            wandb.init(project=args.wandb_project,
                       entity=args.wandb_entity,
                       name=args.wandb_name or f'{args.dataset}__{args.regime}',
                       config=vars(args))
        except Exception as e:
            print(f'[wandb] online init failed ({e}); using OFFLINE mode '
                  f'(sync later with: wandb sync wandb/latest-run)', flush=True)
            os.environ['WANDB_MODE'] = 'offline'
            wandb.init(project=args.wandb_project,
                       name=args.wandb_name or f'{args.dataset}__{args.regime}',
                       config=vars(args))

    from datasets.registry import build_records
    train_records, val_records, load_fn = build_records(args.dataset)
    from datasets.registry import get_class_map, get_class_names
    class_map = get_class_map(args.dataset)
    class_names = get_class_names(args.dataset)
    print(f'[train] class_map for {args.dataset}: {class_map}', flush=True)
    ds = RawDetectionDataset(train_records, load_fn, size=(args.size, args.size))
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=8,
                    collate_fn=collate, drop_last=True)

    # val loader only built if the observational probe is enabled
    val_dl = None
    if args.val_every > 0:
        val_ds = RawDetectionDataset(val_records, load_fn,
                                     size=(args.size, args.size))
        val_dl = DataLoader(val_ds, batch_size=args.bs, shuffle=False,
                            num_workers=8, collate_fn=collate)

    prep = RawPreprocessor().to(args.device)

    detectors = []
    for i, name in enumerate(REGIMES[args.regime]):
        ckpt = os.path.join(args.ckpt_dir, f'{args.dataset}_{name}.pth')
        ckpt = ckpt if os.path.exists(ckpt) else None
        det = build_detector(name, args.num_classes, ckpt, args.device, class_map=class_map)
        detectors.append((f'{name}_{i}', det))

    opt = torch.optim.AdamW(prep.parameters(), lr=args.lr)

    # -------- RESUME: restore model + optimizer + epoch counter -----------
    start_epoch = 0
    if args.resume:
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(f'--resume checkpoint not found: {args.resume}')
        ck = torch.load(args.resume, map_location=args.device)
        prep.load_state_dict(ck['model'])
        if 'optimizer' in ck:
            opt.load_state_dict(ck['optimizer'])
            print(f'[resume] restored optimizer state from {args.resume}',
                  flush=True)
        else:
            print(f'[resume] WARNING: {args.resume} has no optimizer state '
                  f'(old checkpoint) -- AdamW will cold-start, expect a brief '
                  f'lurch in loss at resume.', flush=True)
        # ck['epoch'] is the last COMPLETED epoch (0-indexed); continue after it
        start_epoch = int(ck.get('epoch', -1)) + 1
        print(f'[resume] continuing from epoch {start_epoch + 1} '
              f'(checkpoint finished epoch {start_epoch}) toward {args.epochs}',
              flush=True)
        if start_epoch >= args.epochs:
            print(f'[resume] checkpoint already at/over --epochs={args.epochs}; '
                  f'nothing to do. Raise --epochs to extend.', flush=True)

    prep.train()

    from tqdm import tqdm
    import time

    conflict_log = []
    # step counter continues across resume so wandb x-axis is monotonic
    step = start_epoch * len(dl)
    print(f'[train] {args.dataset} / {args.regime}: {len(dl)} batches/epoch x '
          f'{args.epochs} epochs on {args.device} '
          f'(starting at epoch {start_epoch + 1})', flush=True)
    print(f'[train] loading first batch (reading 46MB .npy files over the '
          f'network is slow on epoch 1; subsequent epochs are cached)...',
          flush=True)
    for epoch in range(start_epoch, args.epochs):
        pbar = tqdm(dl, desc=f'epoch {epoch + 1}/{args.epochs}',
                    dynamic_ncols=True)
        t0 = time.time()
        for i, (raw, targets) in enumerate(pbar):
            if epoch == start_epoch and i == 0:
                print(f'\n[train] first batch arrived in {time.time() - t0:.1f}s '
                      f'-- training is now running', flush=True)
            raw = raw.to(args.device)
            targets = to_device(targets, args.device)

            rgb = prep(raw)                                   # [B,3,H,W] in [0,1]

            losses = {name: det.loss(rgb, targets) for name, det in detectors}

            log_now = len(detectors) > 1 and step % args.log_conflict_every == 0
            if log_now:
                # diagnostic reads per-detector grads (retain_graph internally);
                # safe because combine() below re-differentiates on the same graph
                cos, mean_off, norms = pairwise_cosine(losses, prep.parameters())
                conflict_log.append({
                    'step': step, 'epoch': epoch, 'mean_off_diag': mean_off,
                    'pairs': {f'{a}|{b}': c for (a, b), c in cos.items()},
                    'grad_norms': norms,
                    'losses': {k: float(v.detach()) for k, v in losses.items()},
                })
                if use_wandb:
                    wandb.log({'conflict/mean_cosine': mean_off,
                               **{f'conflict/cos__{a}__{b}': c
                                  for (a, b), c in cos.items()},
                               **{f'grad_norm/{k}': v for k, v in norms.items()}},
                              step=step)

            # combine detector gradients into prep.grad (writes .grad directly),
            # then step. No total.backward() -- combine() did the differentiation.
            opt.zero_grad()
            grad_norms, _ = combine(args.combine, losses, list(prep.parameters()))
            opt.step()
            step += 1

            total_val = sum(float(v.detach()) for v in losses.values())
            # live loss on the bar: total + each detector (spot scale imbalance)
            pbar.set_postfix(loss=f'{total_val:.3f}',
                             **{k: f'{float(v.detach()):.2f}' for k, v in losses.items()})
            if use_wandb:
                wandb.log({'loss/total': total_val, 'epoch': epoch,
                           **{f'loss/{k}': float(v.detach()) for k, v in losses.items()},
                           **{f'grad_norm_pre/{k}': n for k, n in grad_norms.items()}},
                          step=step)

        # checkpoint now includes optimizer state so the run is RESUMABLE
        torch.save({'model': prep.state_dict(),
                    'optimizer': opt.state_dict(),
                    'epoch': epoch,
                    'regime': args.regime, 'dataset': args.dataset},
                   os.path.join(args.out, 'preprocessor.pth'))
        print(f'[train] epoch {epoch + 1}/{args.epochs} done '
              f'(last loss {total_val:.3f}); checkpoint saved', flush=True)

        # -------- OBSERVATIONAL val-mAP probe (diagnostic only) -----------
        # CONTRACT: evaluates ONLY the training detector(s); never a held-out
        # detector. Does NOT select checkpoints or early-stop -- the final
        # checkpoint above is always kept, fixed epoch budget. This exists
        # solely to make the convergence curve visible (e.g. to calibrate the
        # epoch budget on a pilot run).
        if val_dl is not None and (epoch + 1) % args.val_every == 0:
            from core.eval_core import evaluate_map, format_coco_table
            import itertools
            probe_dl = val_dl
            if args.val_max_batches > 0:
                probe_dl = list(itertools.islice(val_dl, args.val_max_batches))
            for name, det in detectors:        # ONLY training detectors
                det_name = name.rsplit('_', 1)[0]   # 'fcos_0' -> 'fcos'
                m = evaluate_map(prep, det.model, det_name, probe_dl, args.device, class_map=class_map)
                # full COCO/YOLO-style table (AP + AP@.5/.75 + by-size + AR)
                print(f'[val-probe] epoch {epoch + 1}  detector={name}', flush=True)
                print(format_coco_table(m), flush=True)
                if use_wandb:
                    # log every COCO metric so all curves are in wandb
                    wandb.log({f'val_probe/{name}/{k}': v for k, v in m.items()
                               if v is not None and isinstance(v, (int, float)) and v >= 0}, step=step)
                    wandb.log({'epoch': epoch + 1}, step=step)

    with open(os.path.join(args.out, 'conflict_log.json'), 'w') as f:
        json.dump(conflict_log, f, indent=2)
    print(f'done: {args.dataset} / {args.regime}  ({step} steps)')
    if use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
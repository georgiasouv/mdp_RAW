#!/usr/bin/env python
"""Convergence analyzer for the multi-detector preprocessor runs.

Reads the [val-probe] lines that train.py prints each epoch from the Slurm
.out logs, reconstructs the val-mAP-per-epoch curve for each (regime, detector),
and reports whether each regime has CONVERGED or is STILL CLIMBING (=> resume).

It answers one question per regime: "if I stop here, am I leaving mAP on the
table?" For multi-detector regimes, the regime is only "converged" when EVERY
training detector's curve has plateaued -- the slowest detector governs.

Usage:
    python analyze_convergence.py --logs cluster_scripts/logs
    python analyze_convergence.py --logs cluster_scripts/logs --window 10 --slope-thresh 0.0005

Reads logs matching {regime}_{combiner}_{jobid}.out (the newest per regime).
"""
import argparse
import glob
import os
import re
from collections import defaultdict

# Matches:  [val-probe] epoch 12  detector=fcos_0
HEADER_RE = re.compile(r'\[val-probe\]\s+epoch\s+(\d+)\s+detector=(\S+)')
# Matches the AP line:    AP  0.0529 |     0.1442 |     0.0240
AP_RE = re.compile(r'AP\s+([0-9.]+)\s*\|')


def parse_log(path):
    """-> dict: detector_name -> list of (epoch, map) sorted by epoch."""
    series = defaultdict(dict)   # detector -> {epoch: map}
    with open(path, 'r', errors='ignore') as fh:
        lines = fh.readlines()
    for i, line in enumerate(lines):
        h = HEADER_RE.search(line)
        if not h:
            continue
        epoch, detector = int(h.group(1)), h.group(2)
        # the AP value is on one of the next few lines
        for j in range(i + 1, min(i + 5, len(lines))):
            m = AP_RE.search(lines[j])
            if m:
                series[detector][epoch] = float(m.group(1))
                break
    # to sorted lists
    return {det: sorted(ep_map.items()) for det, ep_map in series.items()}


def trend(points, window):
    """Given [(epoch, map), ...], analyze the last `window` epochs.
    Returns (last_epoch, last_map, improvement, slope_per_epoch).
    improvement = map at end minus map `window` epochs earlier.
    slope = least-squares slope of map vs epoch over the window.
    """
    if not points:
        return None
    last_epoch, last_map = points[-1]
    win = points[-window:] if len(points) >= window else points
    if len(win) < 2:
        return last_epoch, last_map, 0.0, 0.0
    first_map = win[0][1]
    improvement = last_map - first_map
    # least-squares slope
    xs = [e for e, _ in win]
    ys = [m for _, m in win]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs) or 1e-12
    slope = num / den
    return last_epoch, last_map, improvement, slope


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logs', default='cluster_scripts/logs',
                    help='dir of Slurm .out logs')
    ap.add_argument('--window', type=int, default=10,
                    help='analyze the last N epochs to judge convergence')
    ap.add_argument('--slope-thresh', type=float, default=0.0005,
                    help='mAP/epoch slope below this (over the window) counts '
                         'as plateaued. Default 0.0005 = <0.005 mAP gain over 10 '
                         'epochs is considered converged.')
    args = ap.parse_args()

    # find the newest .out log per regime (strip the _<jobid> suffix)
    logs = glob.glob(os.path.join(args.logs, '*.out'))
    # group by regime tag (everything before the final _<digits>.out)
    by_regime = {}
    for p in logs:
        base = os.path.basename(p)
        m = re.match(r'(.+)_(\d+)\.out$', base)
        if not m:
            continue
        tag, jobid = m.group(1), int(m.group(2))
        if tag not in by_regime or jobid > by_regime[tag][0]:
            by_regime[tag] = (jobid, p)

    if not by_regime:
        print(f'No .out logs found in {args.logs}')
        return

    print(f'Convergence analysis (window = last {args.window} epochs, '
          f'plateau if slope < {args.slope_thresh} mAP/epoch)\n')
    print(f'{"regime":24s} {"detector":12s} {"last_ep":>7s} {"map":>8s} '
          f'{"Δ_window":>9s} {"slope":>9s}  verdict')
    print('-' * 86)

    regime_verdicts = {}
    for tag in sorted(by_regime):
        _, path = by_regime[tag]
        series = parse_log(path)
        if not series:
            print(f'{tag:24s} {"(no val-probe lines found)":<40s}')
            regime_verdicts[tag] = 'NO DATA'
            continue
        regime_climbing = False
        for det in sorted(series):
            t = trend(series[det], args.window)
            if t is None:
                continue
            last_ep, last_map, improvement, slope = t
            climbing = slope >= args.slope_thresh
            regime_climbing = regime_climbing or climbing
            verdict = 'CLIMBING' if climbing else 'plateaued'
            print(f'{tag:24s} {det:12s} {last_ep:>7d} {last_map:>8.4f} '
                  f'{improvement:>+9.4f} {slope:>+9.5f}  {verdict}')
        regime_verdicts[tag] = 'RESUME' if regime_climbing else 'converged'

    # summary
    print('\n' + '=' * 50)
    print('SUMMARY (regime-level verdict):')
    print('=' * 50)
    need_resume = [r for r, v in regime_verdicts.items() if v == 'RESUME']
    converged = [r for r, v in regime_verdicts.items() if v == 'converged']
    for r in sorted(converged):
        print(f'  [converged] {r}')
    for r in sorted(need_resume):
        print(f'  [RESUME   ] {r}  <- still climbing, extend epochs')
    if need_resume:
        print(f'\n{len(need_resume)} regime(s) still climbing. To extend ALL '
              f'regimes uniformly (keeps the comparison fair), re-run each with:')
        print(f'    --resume runs/pascalraw__<tag>/preprocessor.pth --epochs <higher>')
    else:
        print('\nAll regimes converged. Safe to proceed to evaluation.')


if __name__ == '__main__':
    main()
    
    
# 
"""Turn a folder of evaluate.py jsons into the two paper tables:

  1. {out}_matrix.csv : rows = train regime, cols = eval detector.
     This is the mAP VECTOR per processor -- read off how detector-agnostic
     each processor is.

  2. {out}_loo.csv : the leave-one-detector-out comparison, the headline.
     For each pair regime, compare its mAP on the HELD-OUT detector against
     the solo processor trained on that detector. delta > 0 means detector
     diversity transferred to an unseen detector family == the finding.
"""
import argparse
import csv
import glob
import json
import os

# pair regime -> (held-out detector it never trained on, solo baseline regime)
LOO = {
    'pair_12': ('detr',       'solo_T'),   # trained on fcos+fasterrcnn, test detr
    'pair_T2': ('fcos',       'solo_1'),   # trained on detr+fasterrcnn, test fcos
    'pair_T1': ('fasterrcnn', 'solo_2'),   # trained on detr+fcos, test fasterrcnn
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', required=True, help='dir of evaluate.py *.json')
    ap.add_argument('--metric', default='map', choices=['map', 'map_50', 'map_75'])
    ap.add_argument('--out', default='readout')
    args = ap.parse_args()

    table = {}   # (regime, detector) -> metric value
    for f in glob.glob(os.path.join(args.results, '*.json')):
        r = json.load(open(f))
        table[(r['regime'], r['detector'])] = r[args.metric]

    regimes = sorted({k[0] for k in table})
    dets = sorted({k[1] for k in table})

    with open(args.out + '_matrix.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['regime'] + dets)
        for rg in regimes:
            row = [f'{table[(rg, d)]:.4f}' if (rg, d) in table else ''
                   for d in dets]
            w.writerow([rg] + row)

    with open(args.out + '_loo.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['held_out', 'pair_regime', 'mAP_pair',
                    'solo_regime', 'mAP_solo', 'delta', 'diversity_helps'])
        for pair, (held, solo) in LOO.items():
            if (pair, held) not in table or (solo, held) not in table:
                continue
            mp, ms = table[(pair, held)], table[(solo, held)]
            w.writerow([held, pair, f'{mp:.4f}', solo, f'{ms:.4f}',
                        f'{mp - ms:+.4f}', mp >= ms])

    print('wrote', args.out + '_matrix.csv', 'and', args.out + '_loo.csv')


if __name__ == '__main__':
    main()

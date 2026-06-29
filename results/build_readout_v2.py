import csv, glob, json, os
RES = 'results/eval_final'
rows = {}
for f in glob.glob(os.path.join(RES, '*.json')):
    base = os.path.basename(f).replace('.json','')
    tag, det = base.split('__')
    rows[(tag, det)] = json.load(open(f))['map']
tags = sorted({k[0] for k in rows})
dets = ['fcos','fasterrcnn','detr','retinanet']
print('=== MATRIX (rows=run, cols=detector, test mAP) ===')
print('%-20s %8s %8s %8s %8s' % ('run',*dets))
for t in tags:
    print('%-20s %8s %8s %8s %8s' % (t, *['%.4f'%rows[(t,d)] if (t,d) in rows else '-' for d in dets]))
solos = {'fcos':'solo_1_normgrad','fasterrcnn':'solo_2_normgrad','detr':'solo_T_normgrad'}
solo_rn = max(rows.get((s,'retinanet'),0) for s in solos.values())
print('\n=== RETINANET TRANSFER (held-out for all 28; best solo baseline = %.4f) ===' % solo_rn)
print('%-20s %10s %9s  %s' % ('run','rn_mAP','delta','beats_solo'))
for t in tags:
    if t.startswith('solo'): continue
    v = rows.get((t,'retinanet'))
    if v is not None:
        print('%-20s %10.4f %+9.4f  %s' % (t, v, v-solo_rn, 'YES' if v>=solo_rn else 'no'))

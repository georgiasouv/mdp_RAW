#!/usr/bin/env python3
"""Re-split packed ROD by the authoritative .txt manifest and regenerate jsons.
Pools all packed .npy, moves each into packed/<split>/ per the deduped manifest,
and writes rod_packed_<split>.json (split-qualified file_name, boxes /2,
native{0,1,2,6,7}->compact{0,1,2,3,4}) from the pooled json_raw_coco anns."""
import json, os, shutil

PACKED = '/scratch/INC1526354/rod/packed'
ANN_OUT = '/scratch/INC1526354/rod/annotations_packed'
TXT = '/cifs/Shares/Raw_Bayer_Datasets/ROD/splits/{}_split.txt'
SRC_JSON = '/cifs/Shares/Raw_Bayer_Datasets/ROD/json_raw_coco/{}.json'
NATIVE2COMPACT = {0:0, 1:1, 2:2, 6:3, 7:4}
SPLITS = ['train','val','test']

# 1. deduped manifest: stem -> split (train>val>test precedence)
raw = {s: [ln.strip() for ln in open(TXT.format(s)) if ln.strip()] for s in SPLITS}
stem2split = {}
for s in SPLITS:                       # train first => wins ties
    for stem in raw[s]:
        stem2split.setdefault(stem, s)
print('manifest stems:', len(stem2split))

# 2. locate every packed .npy (any current split dir) -> its current path
cur = {}
for s in SPLITS:
    for f in os.listdir(f'{PACKED}/{s}'):
        cur[os.path.splitext(f)[0]] = f'{PACKED}/{s}/{f}'
print('packed npy found:', len(cur))

# 3. move .npy into correct split dir per manifest
moved = same = 0
for stem, split in stem2split.items():
    src = cur[stem]
    dst = f'{PACKED}/{split}/{stem}.npy'
    if os.path.abspath(src) == os.path.abspath(dst):
        same += 1; continue
    shutil.move(src, dst); moved += 1
print(f'npy: moved={moved} already_correct={same}')

# 4. pool original annotations (raw coords, native ids) keyed by stem
pool = {}   # stem -> {'w','h','anns':[(cat,x,y,w,h)]}
for s in SPLITS:
    d = json.load(open(SRC_JSON.format(s)))
    id2name = {im['id']: os.path.splitext(im['file_name'])[0] for im in d['images']}
    id2wh   = {im['id']: (im['width'], im['height']) for im in d['images']}
    for im in d['images']:
        st = id2name[im['id']]
        pool.setdefault(st, {'w':id2wh[im['id']][0],'h':id2wh[im['id']][1],'anns':[]})
    for a in d['annotations']:
        st = id2name[a['image_id']]
        x,y,w,h = a['bbox']
        pool[st]['anns'].append((a['category_id'],x,y,w,h))
print('annotated stems pooled:', len(pool))

# 5. write one json per split from the manifest partition
CATS = [{'id':0,'name':'person'},{'id':1,'name':'bicycle'},{'id':2,'name':'car'},
        {'id':3,'name':'train'},{'id':4,'name':'truck'}]
for split in SPLITS:
    images, anns, iid, aid = [], [], 0, 0
    for stem, sp in stem2split.items():
        if sp != split: continue
        p = pool.get(stem, {'w':2880,'h':1856,'anns':[]})
        images.append({'id':iid,'file_name':f'{split}/{stem}.npy','width':1440,'height':928})
        for cat,x,y,w,h in p['anns']:
            anns.append({'id':aid,'image_id':iid,'category_id':NATIVE2COMPACT[cat],
                         'bbox':[x/2,y/2,w/2,h/2],'area':(w/2)*(h/2),'iscrowd':0})
            aid += 1
        iid += 1
    out = f'{ANN_OUT}/rod_packed_{split}.json'
    json.dump({'images':images,'annotations':anns,'categories':CATS}, open(out,'w'))
    print(f'{split}: imgs={len(images)} anns={len(anns)} -> {out}')
print('DONE')

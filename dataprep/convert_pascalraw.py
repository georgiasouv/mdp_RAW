"""STEP 2 -- convert ALL PASCALRAW annotations to packed-space COCO JSON.

Uses the EXACT scaling logic you verified visually in step 1:
    XML box (in its <size>, ~600x400)
      --x (packed_dim / xml_dim) per axis-->  packed space  (training grid)
Then clips to frame + drops degenerate boxes, logging how many.

Produces one COCO file per split:
    {out_dir}/pascalraw_packed_train.json
    {out_dir}/pascalraw_packed_val.json
    {out_dir}/pascalraw_packed_test.json

COCO schema (what load_fn + torchmetrics expect):
    images:      [{id, file_name='<stem>.npy', width, height}]
    annotations: [{id, image_id, category_id, bbox=[x,y,w,h], area, iscrowd}]
    categories:  [{id:0,name:person},{id:1,name:car},{id:2,name:bicycle}]

Robust to split-file format: each line may be a bare id (2014_000466),
an id with extension (2014_000466.npy / .xml / .nef), or a full path --
we extract the stem either way. If --split-file is omitted, falls back to
globbing every packed/*.npy.

Reads only the npy HEADER (mmap) for H,W -- never loads the 46MB array --
so converting all 4259 is fast.
"""
import argparse
import glob
import json
import os
import xml.etree.ElementTree as ET

import numpy as np

CLASS_MAP = {'person': 0, 'car': 1, 'bicycle': 2}
CLASS_NAMES = ['person', 'car', 'bicycle']


def stem_of(line):
    """Extract the bare id from a split-file line in any reasonable format."""
    s = line.strip()
    if not s:
        return None
    s = os.path.basename(s)                 # strip any path
    for ext in ('.npy', '.xml', '.nef', '.NEF', '.png', '.jpg'):
        if s.endswith(ext):
            s = s[:-len(ext)]
            break
    return s


def parse_xml(xml_path):
    root = ET.parse(xml_path).getroot()
    size = root.find('size')
    xml_w = int(size.find('width').text)
    xml_h = int(size.find('height').text)
    objs = []
    for o in root.findall('object'):
        name = o.find('name').text.strip().lower()
        if name not in CLASS_MAP:
            print(f'  [warn] unexpected class "{name}" in {os.path.basename(xml_path)}')
            continue
        b = o.find('bndbox')
        objs.append((CLASS_MAP[name],
                     float(b.find('xmin').text), float(b.find('ymin').text),
                     float(b.find('xmax').text), float(b.find('ymax').text)))
    return xml_w, xml_h, objs


def convert_split(ids, packed_dir, xml_dir, out_json, split_name):
    images, annotations = [], []
    ann_id = 1
    n_dropped = n_missing_npy = n_missing_xml = n_empty = 0

    for img_id, stem in enumerate(ids, start=1):
        npy = os.path.join(packed_dir, stem + '.npy')
        xml = os.path.join(xml_dir, stem + '.xml')
        if not os.path.exists(npy):
            n_missing_npy += 1
            continue
        if not os.path.exists(xml):
            n_missing_xml += 1
            continue

        arr = np.load(npy, mmap_mode='r')        # header only, no 46MB load
        _, ph, pw = arr.shape
        xml_w, xml_h, objs = parse_xml(xml)
        sx, sy = pw / xml_w, ph / xml_h

        images.append({'id': img_id, 'file_name': stem + '.npy',
                       'width': pw, 'height': ph})

        kept_here = 0
        for cls, x0, y0, x1, y1 in objs:
            X0, Y0, X1, Y1 = x0 * sx, y0 * sy, x1 * sx, y1 * sy
            X0, X1 = max(0.0, min(X0, pw)), max(0.0, min(X1, pw))
            Y0, Y1 = max(0.0, min(Y0, ph)), max(0.0, min(Y1, ph))
            if X1 - X0 < 1.0 or Y1 - Y0 < 1.0:
                n_dropped += 1
                continue
            annotations.append({
                'id': ann_id, 'image_id': img_id, 'category_id': cls,
                'bbox': [round(X0, 2), round(Y0, 2),
                         round(X1 - X0, 2), round(Y1 - Y0, 2)],
                'area': round((X1 - X0) * (Y1 - Y0), 2), 'iscrowd': 0})
            ann_id += 1
            kept_here += 1
        if kept_here == 0:
            n_empty += 1

    coco = {'images': images, 'annotations': annotations,
            'categories': [{'id': i, 'name': n} for i, n in enumerate(CLASS_NAMES)]}
    os.makedirs(os.path.dirname(out_json) or '.', exist_ok=True)
    json.dump(coco, open(out_json, 'w'))

    print(f'[{split_name}] images={len(images)} boxes={len(annotations)} '
          f'| dropped_degenerate={n_dropped} empty_images={n_empty} '
          f'missing_npy={n_missing_npy} missing_xml={n_missing_xml}')
    print(f'[{split_name}] -> {out_json}')
    return len(images), len(annotations)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--xml-dir', default='archive/original_annotations')
    ap.add_argument('--packed-dir', default='packed')
    ap.add_argument('--splits-dir', default='splits',
                    help='dir holding {train,val,test}_split.txt')
    ap.add_argument('--out-dir', default='annotations_packed')
    ap.add_argument('--splits', nargs='+', default=['train', 'val', 'test'])
    args = ap.parse_args()

    packed_dir = os.path.join(args.root, args.packed_dir)
    xml_dir = os.path.join(args.root, args.xml_dir)
    splits_dir = os.path.join(args.root, args.splits_dir)

    grand_imgs = grand_boxes = 0
    for split in args.splits:
        split_file = os.path.join(splits_dir, f'{split}_split.txt')
        if os.path.exists(split_file):
            ids = [stem_of(l) for l in open(split_file)]
            ids = [i for i in ids if i]
        else:
            print(f'[{split}] no split file at {split_file}; '
                  f'skipping (use --splits to control)')
            continue
        out_json = os.path.join(args.out_dir, f'pascalraw_packed_{split}.json')
        ni, nb = convert_split(ids, packed_dir, xml_dir, out_json, split)
        grand_imgs += ni
        grand_boxes += nb

    print(f'\n[done] total: {grand_imgs} images, {grand_boxes} boxes across '
          f'{len(args.splits)} splits, written to {args.out_dir}/')


if __name__ == '__main__':
    main()

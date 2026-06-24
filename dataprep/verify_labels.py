"""STEP 1 (v2) -- verify packed-space labels by drawing them on a minimal-ISP
rendering of the PACKED .npy ITSELF. No external jpg, no scale hops.

Why this is the honest check:
  The detector consumes the packed RAW (via the preprocessor). So we make the
  PACKED array human-viewable with a tiny ISP (RGGB->RGB, gray-world white
  balance, gamma) and draw the boxes on THAT. Same 3017x2006 grid as training
  => a box that's right here is right for training, by construction. Zero
  coordinate reconciliation.

Label scaling (the only coordinate transform):
  XML box in its <size> (~600x400)  --x (packed_dim / xml_dim) per axis-->
  packed space. Measured per file, not hardcoded (your "x10 then /2" made exact).

Self-diagnosing channel order:
  We assume packed planes are [R, G1, G2, B]. The script prints per-channel
  means; in daylight R!=B generally, so if colors look wrong in the output,
  the printed means tell us the assumed order is off.
"""
import argparse
import glob
import os
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw

CLASS_MAP = {'person': 0, 'car': 1, 'bicycle': 2}
CLASS_NAMES = ['person', 'car', 'bicycle']
COLORS = [(0, 200, 255), (255, 0, 0), (0, 255, 0)]   # person, car, bicycle


def minimal_isp(arr, gamma=2.2, ch_order=(0, 1, 2, 3)):
    """[4,H,W] packed RGGB linear -> [H,W,3] uint8 viewable RGB.

    Steps: build RGB from planes (R, mean(G1,G2), B), gray-world white
    balance (scale each channel so means match), then display gamma +
    percentile normalize. Visualization only -- never touches label coords.
    ch_order maps which plane is (R, G1, G2, B)."""
    a = arr.astype(np.float32)
    R = a[ch_order[0]]
    G = 0.5 * (a[ch_order[1]] + a[ch_order[2]])
    B = a[ch_order[3]]
    rgb = np.stack([R, G, B], axis=-1)

    # gray-world white balance: equalize channel means to the green mean
    means = rgb.reshape(-1, 3).mean(0) + 1e-6
    rgb = rgb * (means[1] / means)              # scale R and B toward G
    rgb = np.clip(rgb, 0, None)

    # display gamma + robust normalize so dark RAW is visible
    rgb = rgb ** (1.0 / gamma)
    p99 = np.percentile(rgb, 99) + 1e-6
    return (np.clip(rgb / p99, 0, 1) * 255).astype(np.uint8)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--xml-dir', default='archive/original_annotations')
    ap.add_argument('--packed-dir', default='packed')
    ap.add_argument('--out', default='step1_vis')
    ap.add_argument('--n', type=int, default=6)
    ap.add_argument('--gamma', type=float, default=2.2)
    args = ap.parse_args()

    packed_dir = os.path.join(args.root, args.packed_dir)
    xml_dir = os.path.join(args.root, args.xml_dir)
    os.makedirs(args.out, exist_ok=True)

    stems = sorted(os.path.splitext(os.path.basename(p))[0]
                   for p in glob.glob(os.path.join(packed_dir, '*.npy')))
    done = 0
    for stem in stems:
        if done >= args.n:
            break
        npy = os.path.join(packed_dir, stem + '.npy')
        xml = os.path.join(xml_dir, stem + '.xml')
        if not (os.path.exists(npy) and os.path.exists(xml)):
            continue

        arr = np.load(npy)
        _, ph, pw = arr.shape
        xml_w, xml_h, objs = parse_xml(xml)
        if not objs:
            continue
        sx, sy = pw / xml_w, ph / xml_h

        # scale -> packed, clip to frame, drop degenerate
        clean = []
        for cls, x0, y0, x1, y1 in objs:
            X0, Y0, X1, Y1 = x0 * sx, y0 * sy, x1 * sx, y1 * sy
            X0, X1 = max(0, min(X0, pw)), max(0, min(X1, pw))
            Y0, Y1 = max(0, min(Y0, ph)), max(0, min(Y1, ph))
            if X1 - X0 >= 1 and Y1 - Y0 >= 1:
                clean.append((cls, X0, Y0, X1, Y1))

        # channel means -> diagnose RGGB order
        chmeans = arr.astype(np.float32).reshape(4, -1).mean(1)

        # render packed RAW via minimal ISP, draw boxes (packed grid, no scale)
        im = Image.fromarray(minimal_isp(arr, gamma=args.gamma))
        dr = ImageDraw.Draw(im)
        for cls, x0, y0, x1, y1 in clean:
            dr.rectangle([x0, y0, x1, y1], outline=COLORS[cls], width=6)
            dr.text((x0 + 5, y0 + 5), CLASS_NAMES[cls], fill=(255, 255, 0))
        im.save(os.path.join(args.out, f'{stem}.png'))

        print(f'[{stem}] xml {xml_w}x{xml_h} -> packed {pw}x{ph} '
              f'(sx={sx:.4f} sy={sy:.4f}); {len(clean)} boxes; '
              f'ch_means(0,1,2,3)=[{", ".join(f"{m:.3f}" for m in chmeans)}]')
        for cls, x0, y0, x1, y1 in clean:
            print(f'    {CLASS_NAMES[cls]:8s} packed xyxy = '
                  f'({x0:.0f},{y0:.0f},{x1:.0f},{y1:.0f})')
        done += 1

    print(f'\n[step1] wrote {done} overlays to {args.out}/')
    print('[step1] open them: boxes should sit on objects; colors should look '
          'natural. If colors are off, the printed ch_means tell us the RGGB '
          'order assumption needs swapping.')


if __name__ == '__main__':
    main()

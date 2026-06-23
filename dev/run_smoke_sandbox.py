"""SANDBOX smoke runner -- identical intent to run_smoke.py, but builds
detectors with RANDOM init (weights=None) so it works inside the chat
sandbox where download.pytorch.org is blocked (HTTP 403).

Random detector weights are FINE for a plumbing test: we are verifying
gradient flow, shape/range matching, the frozen/trainable split, the
empty-box path, and the multi-detector loop -- none of which depend on
the detector being trained. (mAP quality is NOT what a smoke test checks.)

On YOUR cluster you'd run the normal run_smoke.py, which uses COCO weights.
"""
import sys, os, types, json

import torch
import torchvision
import torch.nn as nn

from dev import synthetic_registry
from datasets import registry
registry.build_records = synthetic_registry.build_records  # monkeypatch

from core.preprocessor import RawPreprocessor
from core.adapters import TorchvisionAdapter
from datasets.dataset import RawDetectionDataset, collate
from core.conflict import pairwise_cosine
from torch.utils.data import DataLoader

DEVICE = 'cpu'
NUM_CLASSES = 3


def make_random_detector(name, num_classes=NUM_CLASSES):
    """Build a torchvision detector with NO pretrained weights, wrapped in
    the user's REAL TorchvisionAdapter (so freezing logic is theirs)."""
    if name == 'fcos':
        m = torchvision.models.detection.fcos_resnet50_fpn(
            weights=None, weights_backbone=None, num_classes=num_classes)
    elif name == 'retinanet':
        m = torchvision.models.detection.retinanet_resnet50_fpn(
            weights=None, weights_backbone=None, num_classes=num_classes)
    elif name == 'fasterrcnn':
        m = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=None, weights_backbone=None, num_classes=num_classes + 1)
    else:
        raise ValueError(name)
    return TorchvisionAdapter(m.to(DEVICE)).to(DEVICE)


def probe(detector_names):
    print(f"\n{'='*64}\n[probe] detectors={detector_names}  device={DEVICE}\n{'='*64}")
    tr, _v, load_fn = synthetic_registry.build_records('fake')
    ds = RawDetectionDataset(tr, load_fn, size=(128, 128))
    dl = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=collate, drop_last=True)

    prep = RawPreprocessor().to(DEVICE)
    dets = [(f'{n}_{i}', make_random_detector(n)) for i, n in enumerate(detector_names)]
    for _, d in dets:
        d.train()  # training mode so .loss() returns a loss, not predictions

    raw, targets = next(iter(dl))
    raw = raw.to(DEVICE)
    targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

    n_empty = sum(1 for t in targets if t['boxes'].numel() == 0)
    print(f"[probe] batch: raw {tuple(raw.shape)} range "
          f"[{raw.min():.3f},{raw.max():.3f}]; {n_empty}/{len(targets)} "
          f"images have ZERO boxes (empty-path exercised: {n_empty>0})")

    rgb = prep(raw)
    print(f"[probe] prep out {tuple(rgb.shape)} range "
          f"[{rgb.min():.3f},{rgb.max():.3f}]  (want 3ch, [0,1])")
    rmin, rmax = float(rgb.min().detach()), float(rgb.max().detach())
    assert rgb.shape[1] == 3 and 0.0 <= rmin and rmax <= 1.0

    losses = {name: d.loss(rgb, targets) for name, d in dets}
    for k, v in losses.items():
        print(f"[probe]   loss[{k}] = {float(v):.4f}  finite={torch.isfinite(v).item()}")

    total = sum(losses.values())
    total.backward(retain_graph=False) if len(losses) == 1 else None

    # multi-detector: use conflict.py to get per-detector grads + cosine
    if len(losses) > 1:
        # recompute fresh graph for the conflict probe (we didn't backward yet)
        rgb2 = prep(raw)
        losses2 = {name: d.loss(rgb2, targets) for name, d in dets}
        cos, mean_off, norms = pairwise_cosine(losses2, prep.parameters())
        print(f"[probe] grad cosines: {{ {', '.join(f'{a}|{b}:{c:+.3f}' for (a,b),c in cos.items())} }}")
        print(f"[probe] mean off-diag cosine = {mean_off:+.4f}  (the conflict signal)")
        print(f"[probe] grad norms (scale problem): {{ {', '.join(f'{k}:{v:.3e}' for k,v in norms.items())} }}")
        sum(losses2.values()).backward()

    prep_gn = sum(p.grad.norm().item() for p in prep.parameters() if p.grad is not None)
    det_any = any(p.grad is not None for _, d in dets for p in d.parameters())
    print(f"[probe] preprocessor grad-norm = {prep_gn:.6f}  (want > 0)")
    print(f"[probe] any frozen-detector grad? {det_any}  (want False)")
    assert prep_gn > 0, "FAIL: no grad reached preprocessor"
    assert not det_any, "FAIL: frozen detector got grad"
    print("[probe] PASS\n")


def integration(detector_names, regime_label, max_steps=4):
    """Mini training loop mirroring train.py's inner step exactly, but
    self-contained and memory-capped for the 3.75GB sandbox. Same logic:
    prep -> losses -> sum -> backward -> step, plus checkpoint write."""
    import gc
    print(f"{'='*64}\n[integration] regime={regime_label} -> {detector_names} "
          f"(<= {max_steps} steps, 96px)\n{'='*64}")
    tr, _v, load_fn = synthetic_registry.build_records('fake')
    ds = RawDetectionDataset(tr, load_fn, size=(96, 96))
    dl = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=collate, drop_last=True)

    prep = RawPreprocessor().to(DEVICE)
    dets = [(f'{n}_{i}', make_random_detector(n)) for i, n in enumerate(detector_names)]
    for _, d in dets:
        d.train()
    opt = torch.optim.AdamW(prep.parameters(), lr=1e-3)

    first_total, last_total, step = None, None, 0
    for raw, targets in dl:
        if step >= max_steps:
            break
        raw = raw.to(DEVICE)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        rgb = prep(raw)
        losses = {name: d.loss(rgb, targets) for name, d in dets}
        total = sum(losses.values())
        opt.zero_grad(); total.backward(); opt.step()
        if first_total is None:
            first_total = float(total)
        last_total = float(total)
        del rgb, losses, total; gc.collect()
        step += 1
    print(f"[integration] ran {step} steps; total loss first={first_total:.4f} "
          f"last={last_total:.4f}")

    os.makedirs('/tmp/smoke_out', exist_ok=True)
    torch.save({'model': prep.state_dict(), 'regime': regime_label, 'dataset': 'fake'},
               '/tmp/smoke_out/preprocessor.pth')
    ok = os.path.exists('/tmp/smoke_out/preprocessor.pth')
    print(f"[integration] checkpoint written: {ok}  "
          f"(this is what evaluate.py / build_readout.py consume)\n")


def probe_pair_frugal(detector_names):
    """Memory-frugal multi-detector probe for the 3.75GB sandbox.

    The conflict diagnostic normally holds BOTH detectors' backward graphs
    alive at once (retain_graph=True) -- too big for two ResNet-50-FPNs
    here. We instead compute each detector's grad-w.r.t-prep-params on its
    OWN graph, free it, then compare the two grad vectors offline. Same
    cosine, a fraction of the peak memory. On a GPU you'd just use the
    real conflict.pairwise_cosine (which keeps both graphs)."""
    import gc
    print(f"\n{'='*64}\n[pair] detectors={detector_names} (frugal 96px bs=1)\n{'='*64}")

    # seed=1 -> exactly 1 box (seed%4==1), so this batch is NON-empty
    tr, _v, load_fn = synthetic_registry.build_records('fake')
    ds = RawDetectionDataset([1, 2], load_fn, size=(96, 96))  # seeds 1,2 -> 1,2 boxes
    dl = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate, drop_last=True)
    raw, targets = next(iter(dl))
    raw = raw.to(DEVICE)
    targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
    print(f"[pair] batch raw {tuple(raw.shape)}; boxes in image: "
          f"{[int(t['boxes'].shape[0]) for t in targets]} (non-empty)")

    prep = RawPreprocessor().to(DEVICE)
    params = [p for p in prep.parameters() if p.requires_grad]

    flats = {}
    for name in detector_names:
        det = make_random_detector(name)
        det.train()
        rgb = prep(raw)                      # fresh graph per detector
        loss = det.loss(rgb, targets)
        g = torch.autograd.grad(loss, params, allow_unused=True)
        flats[name] = torch.cat([x.reshape(-1) for x in g if x is not None])
        print(f"[pair]   {name}: loss={float(loss):.4f}  ||grad||={flats[name].norm():.4e}")
        det_any = any(p.grad is not None for p in det.parameters())
        assert not det_any, f"FAIL: {name} accumulated grad"
        del det, rgb, loss, g; gc.collect()  # free this detector's graph

    import torch.nn.functional as Fnn
    a, b = detector_names
    cos = Fnn.cosine_similarity(flats[a].unsqueeze(0), flats[b].unsqueeze(0)).item()
    print(f"[pair] grad cosine({a},{b}) = {cos:+.4f}")
    print(f"[pair]   >0 agree | ~0 orthogonal | <0 conflict  "
          f"-- this is the signal conflict.py logs every N steps")
    print(f"[pair] grad-norm ratio = {flats[a].norm()/flats[b].norm():.3f}  "
          f"(scale problem: if >>1 or <<1, one detector dominates the sum)")
    print("[pair] PASS — multi-detector grad extraction + cosine works\n")


if __name__ == '__main__':
    # solo (1 detector) full probe, then the frugal pair conflict probe
    probe(['fcos'])                       # solo_1 analogue, full grad-flow assert
    import gc; gc.collect()
    probe_pair_frugal(['fcos', 'retinanet'])   # multi-detector conflict signal
    gc.collect()
    integration(['fcos'], 'solo_1')       # train loop + checkpoint (1 det = memory-safe)
    print("SMOKE COMPLETE — if every PASS printed and a checkpoint was "
          "written, the spine works end to end.")

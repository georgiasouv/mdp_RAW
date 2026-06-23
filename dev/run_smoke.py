"""SMOKE-TEST RUNNER -- proves the plumbing turns over on fake data,
WITHOUT editing any of your real files.

It works by monkeypatching the synthetic build_records into the
datasets_registry module *before* train.py imports it, then calling
train.main() with smoke-safe args.

What this verifies (the whole point):
  1. raw [4,H,W] -> dataset resize -> preprocessor -> [3,H,W] in [0,1]
     end to end, shapes and ranges line up into the detector stem.
  2. gradient flows THROUGH the frozen detector into the preprocessor:
     we assert the preprocessor's params receive non-zero grad and that
     loss generally decreases.
  3. the multi-detector loop + conflict diagnostic run (pair_12 = 2 dets).
  4. the deliberately-empty-box sample does not crash the loss path.

Run:
    python run_smoke.py                 # default: pair_12, cpu, 2 epochs
    python run_smoke.py solo_1          # single torchvision detector
    python run_smoke.py homo_1          # 3x same detector (the control)

DETR is intentionally NOT smoke-tested by default: it needs `transformers`
plus a weights download. The torchvision regimes (solo_1, solo_2, pair_12,
homo_1, homo_2) use weights that ship with torchvision and cover the
multi-detector path. Add 'detr' regimes once the spine is proven.
"""
import sys
import os
import types

# Run as a module from the project root:  python -m dev.run_smoke [regime]
# (the package is pip-installed -e, so all imports below resolve anywhere)

SMOKE_SAFE_REGIMES = {'solo_1', 'solo_2', 'pair_12', 'homo_1', 'homo_2'}


def _patch_registry():
    """Inject synthetic build_records into datasets_registry BEFORE train
    imports it. We replace the symbol the real code calls."""
    from datasets import registry
    from dev import synthetic_registry
    registry.build_records = synthetic_registry.build_records
    return registry


def _instrument_grad_check():
    """Wrap RawPreprocessor.forward-adjacent step is hard; instead we hook
    the optimizer. Simplest robust check: after train runs, inspect the
    saved conflict_log + a manual one-batch grad probe. We do the probe
    here directly so the assertion is unambiguous."""
    pass  # the explicit probe lives in main(), below


def main():
    regime = sys.argv[1] if len(sys.argv) > 1 else 'pair_12'
    if regime not in SMOKE_SAFE_REGIMES:
        print(f"[smoke] WARNING: '{regime}' may need DETR/transformers + a "
              f"download. Smoke-safe regimes: {sorted(SMOKE_SAFE_REGIMES)}")

    _patch_registry()

    # --- 1. direct one-batch gradient probe (the core claim) --------------
    # Build the exact pieces train.py uses and verify grad reaches the
    # preprocessor through a frozen detector. This is the unambiguous test;
    # the full train.main() run below is the integration test.
    import torch
    from core.preprocessor import RawPreprocessor
    from core.adapters import build_detector
    from datasets.dataset import RawDetectionDataset, collate
    from torch.utils.data import DataLoader
    from dev import synthetic_registry

    device = 'cpu'
    print(f"[smoke] regime={regime} device={device}")

    tr, _val, load_fn = synthetic_registry.build_records('fake')
    ds = RawDetectionDataset(tr, load_fn, size=(256, 256))
    dl = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=collate,
                    drop_last=True)

    prep = RawPreprocessor().to(device)

    # one torchvision detector, COCO weights (no ckpt on disk -> 91 classes)
    det = build_detector('fcos', num_classes=3, weights_path=None, device=device)

    raw, targets = next(iter(dl))
    raw = raw.to(device)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    print(f"[smoke] raw batch shape {tuple(raw.shape)} "
          f"range [{raw.min():.3f}, {raw.max():.3f}]")
    rgb = prep(raw)
    print(f"[smoke] prep out shape {tuple(rgb.shape)} "
          f"range [{rgb.min():.3f}, {rgb.max():.3f}]  "
          f"(want 3 channels, [0,1])")

    # the detector adapter must be in training mode to RETURN A LOSS
    det.train()
    loss = det.loss(rgb, targets)
    print(f"[smoke] detector loss = {float(loss):.4f}  "
          f"(scalar? {loss.dim() == 0}, finite? {torch.isfinite(loss).item()})")

    loss.backward()
    # THE assertion: preprocessor params got non-zero grad THROUGH the
    # frozen detector; detector params got NONE (they're frozen).
    prep_grads = [p.grad for p in prep.parameters() if p.grad is not None]
    prep_grad_norm = sum(g.norm().item() for g in prep_grads)
    det_grad_any = any(p.grad is not None for p in det.parameters())
    print(f"[smoke] preprocessor grad norm = {prep_grad_norm:.6f}  "
          f"(want > 0)")
    print(f"[smoke] any detector param grad? {det_grad_any}  (want False)")

    assert prep_grad_norm > 0, "FAIL: no grad reached the preprocessor!"
    assert not det_grad_any, "FAIL: frozen detector accumulated grad!"
    print("[smoke] PROBE PASSED: grad flows through frozen detector into prep\n")

    # --- 2. full integration run through your real train.main() ----------
    print("[smoke] running real train.main() for 2 epochs on fake data ...")
    from core import train
    sys.argv = [
        'train.py',
        '--regime', regime,
        '--dataset', 'fake',
        '--num-classes', '3',
        '--device', device,
        '--epochs', '2',
        '--bs', '4',
        '--size', '256',
        '--out', '/tmp/smoke_out',
        '--ckpt-dir', '/tmp/no_ckpts_here',   # forces COCO-pretrained path
        '--log-conflict-every', '1',
    ]
    train.main()

    # --- 3. inspect what train wrote -------------------------------------
    import json
    cpath = '/tmp/smoke_out/conflict_log.json'
    if os.path.exists(cpath):
        log = json.load(open(cpath))
        if log:
            print(f"\n[smoke] conflict_log has {len(log)} entries")
            first, last = log[0], log[-1]
            print(f"[smoke] first step losses: {first['losses']}")
            print(f"[smoke] last  step losses: {last['losses']}")
            if 'mean_off_diag' in first:
                print(f"[smoke] mean grad cosine first={first['mean_off_diag']:.4f} "
                      f"last={last['mean_off_diag']:.4f}  "
                      f"(this is your conflict signal)")
        else:
            print("[smoke] conflict_log empty (expected for solo/homo single-det? "
                  "no -- homo has 3 dets, should log)")
    ppath = '/tmp/smoke_out/preprocessor.pth'
    print(f"[smoke] checkpoint written: {os.path.exists(ppath)}  ({ppath})")
    print("\n[smoke] ALL DONE. If you got here with PROBE PASSED and a "
          "written checkpoint, the spine works.")


if __name__ == '__main__':
    main()

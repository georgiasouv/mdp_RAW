"""Property-based unit test for CAGrad (_cagrad_solve).

CAGrad's output comes from a solver, so we verify the two guarantees it must
satisfy by construction rather than hand-checking the vector:
  (P1) c=0  -> update equals the plain average gradient g0
  (P2) c>0  -> update stays in the ball ||d - g0|| <= c*||g0||
Plus sanity: finite output, correct shape.
"""
import sys
import torch
import core.combine as C


def _flat(grad_list):
    return torch.cat([g.reshape(-1) for g in grad_list])


def _g0(per, names):
    G = torch.stack([_flat(per[n]) for n in names], dim=0)
    return G.mean(dim=0)


def run_case(label, per, names, params):
    ok = True
    g0 = _g0(per, names)
    g0n = g0.norm().item()

    d0 = _flat(C._cagrad_solve(per, names, params, c=0.0))
    if d0.shape != g0.shape:
        print(f"  [{label}] FAIL shape: d0 {tuple(d0.shape)} vs g0 {tuple(g0.shape)}")
        return False
    e0 = (d0 - g0).norm().item()
    p1 = (e0 < 1e-5) and bool(torch.isfinite(d0).all())
    ok &= p1
    print(f"  [{label}] P1 c=0: ||d-g0||={e0:.2e}  -> {'PASS' if p1 else 'FAIL'}")

    for c in (0.1, 0.5, 1.0):
        d = _flat(C._cagrad_solve(per, names, params, c=c))
        dist = (d - g0).norm().item()
        bound = c * g0n
        p2 = (dist <= bound + 1e-4) and bool(torch.isfinite(d).all())
        ok &= p2
        print(f"  [{label}] P2 c={c}: ||d-g0||={dist:.4f}  bound={bound:.4f}  -> {'PASS' if p2 else 'FAIL'}")
    return ok


def main():
    torch.manual_seed(0)
    all_ok = True

    # Case A: hand-crafted 2 conflicting detectors, single 2-vector param.
    # g_a=[1,0], g_b=[-1,1] -> g0=[0,0.5]; at c=0, d must equal [0,0.5].
    pA = [torch.zeros(2)]
    perA = {'a': [torch.tensor([1.0, 0.0])],
            'b': [torch.tensor([-1.0, 1.0])]}
    all_ok &= run_case('handcrafted-2det', perA, ['a', 'b'], pA)

    # Case B: random 3 detectors over MULTIPLE params (exercises unflatten).
    pB = [torch.zeros(4), torch.zeros(2, 3), torch.zeros(5)]
    namesB = ['det0', 'det1', 'det2']
    perB = {n: [torch.randn_like(p) for p in pB] for n in namesB}
    all_ok &= run_case('random-3det-multiparam', perB, namesB, pB)

    print("\nRESULT:", "ALL PASS" if all_ok else "FAILURES PRESENT")
    sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    main()

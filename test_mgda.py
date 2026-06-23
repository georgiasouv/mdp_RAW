"""Property test for MGDA (_mgda_solve): min-norm point in the gradient hull."""
import sys
import torch
import core.combine as C


def _flat(gl):
    return torch.cat([g.reshape(-1) for g in gl])


def run_case(label, per, names, params):
    ok = True
    flats = {n: _flat(per[n]) for n in names}
    d = _flat(C._mgda_solve(per, names, params))
    P = next(iter(flats.values())).shape
    if d.shape != P:
        print("  [%s] FAIL shape: %s vs %s" % (label, tuple(d.shape), tuple(P)))
        return False
    if not bool(torch.isfinite(d).all()):
        print("  [%s] FAIL: non-finite" % label)
        return False
    dn = d.norm().item()
    p1 = True
    for n in names:
        gn = flats[n].norm().item()
        if dn > gn + 1e-4:
            p1 = False
            print("  [%s] P1 FAIL: ||d||=%.4f > ||%s||=%.4f" % (label, dn, n, gn))
    if p1:
        norms = {n: round(flats[n].norm().item(), 3) for n in names}
        print("  [%s] P1 min-norm: ||d||=%.4f vs %s -> PASS" % (label, dn, norms))
    ok &= p1
    p2 = True
    dots = {}
    for n in names:
        dot = torch.dot(d, flats[n]).item()
        dots[n] = round(dot, 4)
        if dot < -1e-3:
            p2 = False
    print("  [%s] P2 common-descent: %s -> %s" % (label, dots, "PASS" if p2 else "FAIL"))
    ok &= p2
    return ok


def main():
    torch.manual_seed(0)
    ok = True
    pA = [torch.zeros(2)]
    perA = {'a': [torch.tensor([1.0, 0.0])], 'b': [torch.tensor([-1.0, 1.0])]}
    ok &= run_case('handcrafted-2det-conflict', perA, ['a', 'b'], pA)
    pB = [torch.zeros(3)]
    perB = {'a': [torch.tensor([2.0, 0.0, 0.0])], 'b': [torch.tensor([1.0, 1.0, 0.0])]}
    ok &= run_case('handcrafted-2det-aligned', perB, ['a', 'b'], pB)
    pC = [torch.zeros(4), torch.zeros(2, 3), torch.zeros(5)]
    namesC = ['det0', 'det1', 'det2']
    perC = {n: [torch.randn_like(p) for p in pC] for n in namesC}
    ok &= run_case('random-3det-multiparam', perC, namesC, pC)
    print("\nRESULT:", "ALL PASS" if ok else "FAILURES PRESENT")
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()

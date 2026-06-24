"""Gradient-conflict diagnostic (the free ride-along that decides which
branch of the outcome tree you're in).

For each detector we compute the gradient of ITS loss w.r.t. the SHARED
preprocessor parameters, then take pairwise cosine similarities:

    cos > 0  : detectors mostly agree about how to change the processor
    cos ~ 0  : orthogonal demands
    cos < 0  : genuine conflict -- one detector's improvement degrades
               another's. THIS is what motivates conflict-aware
               aggregation (PCGrad / CAGrad) and what makes the
               "diversity is a regularizer" story non-trivial.

Cost note: we reuse the (expensive) detector forward graph via
retain_graph=True, so each extra detector is one cheap backward through
the *tiny* preprocessor, not a fresh forward. autograd.grad returns grads
WITHOUT populating .grad, so it never interferes with the optimizer step
that runs on loss.backward() afterwards.
"""
import torch
import torch.nn.functional as F


def _flatten(grads):
    return torch.cat([g.reshape(-1) for g in grads if g is not None])


def pairwise_cosine(per_detector_losses: dict, params):
    """
    per_detector_losses : {name: scalar loss tensor (requires grad)}
    params              : iterable of preprocessor parameters

    Returns (cos, mean_off_diag, grad_norms):
      cos           : {(name_i, name_j): cosine} for i < j
      mean_off_diag : float -- the single headline conflict number
      grad_norms    : {name: ||grad||} -- to see if one detector dominates
                      by sheer magnitude (the scale problem, distinct from
                      the direction problem cosine measures)
    """
    params = [p for p in params if p.requires_grad]
    names = list(per_detector_losses.keys())

    flat = {}
    for k in names:
        g = torch.autograd.grad(per_detector_losses[k], params,
                                retain_graph=True, allow_unused=True)
        flat[k] = _flatten(g)

    norms = {k: flat[k].norm().item() for k in names}
    cos, vals = {}, []
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            i, j = names[a], names[b]
            c = F.cosine_similarity(flat[i].unsqueeze(0),
                                    flat[j].unsqueeze(0)).item()
            cos[(i, j)] = c
            vals.append(c)
    mean_off = sum(vals) / len(vals) if vals else float('nan')
    return cos, mean_off, norms

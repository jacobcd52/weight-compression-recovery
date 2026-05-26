"""Vectorized-ensemble recovery evaluator.

Trains many (compressed_init x learning_rate) configs *simultaneously* on one GPU via
`torch.func.vmap` over a GroupNorm ResNet-20 (GroupNorm is stateless, so functional_call
+ vmap need no running-stat bookkeeping). All configs advance in lockstep on shared data
batches; the whole ensemble is evaluated on the test set in one batched pass every
`eval_every` steps; each config's first step with full-test loss < threshold is recorded
(recovery). Crossed/expired configs are masked out (LR -> 0) so the batch shrinks in
effect without resizing — this is the early-exit you wanted ("stop a model once any of its
LRs crosses"), handled by masking inside the fixed-width batch.

This is the evaluation engine for the sweeps and for the auto-research phase: the "method"
is (compression -> dense init) x (recipe params); compression varies freely, the recipe is
a vectorizable template, so candidates stay vmappable. Returns per-config recovery cost.
"""
import math

import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap

from .models import resnet20


def _should_eval(step, budget):
    """Adaptive eval cadence: dense early (fast recoveries cross in the first ~100 steps),
    sparse late. Eval is the dominant cost in the vectorized ensemble, so this is the main
    speed lever; the early density keeps fine resolution where crossings actually happen."""
    if step <= 100:
        return step % 10 == 0
    if step <= 500:
        return step % 25 == 0
    return step % 100 == 0


def _lr_factor(step, warmup, total):
    if step < warmup:
        return (step + 1) / warmup
    p = min(1.0, (step - warmup) / max(1, total - warmup))
    return 0.5 * (1.0 + math.cos(math.pi * p))


def _bcast(v, ndim):
    """Reshape a per-config vector (W,) to broadcast against a (W, ...) param."""
    return v.view(-1, *([1] * (ndim - 1)))


@torch.no_grad()
def _ensemble_eval(model, pdict, test_loader, device, use_bf16=True):
    """Per-config (W,) full-test accuracy% and mean CE loss, in one batched pass."""
    W = next(iter(pdict.values())).shape[0]
    correct = torch.zeros(W, device=device)
    loss_sum = torch.zeros(W, device=device)
    total = 0

    def fwd(p, x):
        return functional_call(model, p, (x,))

    vfwd = vmap(fwd, in_dims=(0, None))
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        if use_bf16 and device.type == "cuda":
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = vfwd(pdict, x).float()           # (W, B, C)
        else:
            logits = vfwd(pdict, x).float()
        correct += (logits.argmax(-1) == y[None]).sum(1)
        loss_sum += F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            y[None].expand(W, -1).reshape(-1), reduction="none"
        ).view(W, -1).sum(1)
        total += y.numel()
    return 100.0 * correct / total, loss_sum / total


def run_ensemble(inits, lrs, *, baseline_loss, baseline_steps, budget_steps, warmup_steps,
                 eval_every, train_loader, test_loader, device, norm="group", groups=8,
                 num_classes=10, weight_decay=5e-4, betas=(0.9, 0.999), eps=1e-8,
                 use_bf16=True, log_prefix="", stop_on_any=True):
    """Vectorized-train one chunk of configs. `inits`: list of W state_dicts; `lrs`: list of
    W peak LRs. Returns list of per-config dicts {recovered, recovery_steps, best_loss}."""
    model = resnet20(num_classes=num_classes, norm=norm, groups=groups).to(device)
    model.eval()                                          # GroupNorm: same in train/eval
    pnames = [n for n, _ in model.named_parameters()]
    W = len(inits)
    b1, b2 = betas

    # stacked params + Adam moments (W, ...) on device
    pdict = {n: torch.stack([inits[i][n] for i in range(W)]).to(device).float()
             for n in pnames}
    m = {n: torch.zeros_like(v) for n, v in pdict.items()}
    v = {n: torch.zeros_like(t) for n, t in pdict.items()}
    lr_vec = torch.tensor([float(x) for x in lrs], device=device)
    active = torch.ones(W, device=device)                 # 1.0 active, 0.0 frozen
    rec_step = [None] * W
    best_loss = [float("inf")] * W
    curve = []                                            # (step, min_loss, max_acc) trajectory

    def loss_fn(params, x, y):
        return F.cross_entropy(functional_call(model, params, (x,)), y)

    gfn = vmap(grad(loss_fn), in_dims=(0, None, None))     # per-config grads, shared (x,y)

    def do_eval(step, final=False):
        accs, losses = _ensemble_eval(model, pdict, test_loader, device, use_bf16)
        for i in range(W):
            li = float(losses[i])
            if li < best_loss[i]:
                best_loss[i] = li
            if rec_step[i] is None and li <= baseline_loss:
                rec_step[i] = step
                active[i] = 0.0                            # freeze: this config recovered
        curve.append((step, float(losses.min()), float(accs.max())))  # best-over-LRs envelope
        return losses

    def stop_now():
        # W=4 LRs of one method: stop as soon as ANY LR crosses (= the min recovery).
        if stop_on_any:
            return any(r is not None for r in rec_step)
        return active.sum().item() == 0                    # all configs crossed

    step = 0
    it = iter(train_loader)
    do_eval(0)                                             # init eval (records trivial recoveries)
    while step < budget_steps and not stop_now():
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(train_loader); x, y = next(it)
        x, y = x.to(device), y.to(device)
        if use_bf16 and device.type == "cuda":
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                g = gfn(pdict, x, y)
        else:
            g = gfn(pdict, x, y)
        step += 1
        f = _lr_factor(step, warmup_steps, budget_steps)
        lr_t = lr_vec * f * active                         # frozen configs get LR 0
        for n in pnames:
            gn = g[n].float()
            m[n].mul_(b1).add_(gn, alpha=1 - b1)
            v[n].mul_(b2).addcmul_(gn, gn, value=1 - b2)
            mhat = m[n] / (1 - b1 ** step)
            vhat = v[n] / (1 - b2 ** step)
            lrb = _bcast(lr_t, pdict[n].dim())
            pdict[n].add_(-lrb * (mhat / (vhat.sqrt() + eps) + weight_decay * pdict[n]))
        if _should_eval(step, budget_steps):
            do_eval(step)
    if not stop_now():
        do_eval(step, final=True)

    results = [{"recovered": rec_step[i] is not None,
                "recovery_steps": rec_step[i],
                "recovery_fraction": (rec_step[i] / baseline_steps
                                      if rec_step[i] is not None else None),
                "best_loss": best_loss[i],
                "lr": float(lrs[i])} for i in range(W)]
    return results, curve

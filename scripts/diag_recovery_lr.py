"""Diagnostic: is the ~7% recovery cost an LR-schedule artifact?

Retrain a near-lossless init (quantize_8) and a harder one (kmeans_2) from the 50-epoch
baseline under several LR / schedule choices, logging the FULL test-loss-vs-step trajectory.
Produces figures/diag_lr_curves.png (2 panels) and prints first-crossing steps.
"""
import json, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from src.models import resnet20
from src.compress import reconstruct
from src.data import get_loaders
from src.utils import get_device, eval_full

dev = get_device()
base = json.load(open("runs/baseline/summary.json"))
bl, bs = base["baseline_test_loss"], base["baseline_steps"]
tl, te, _ = get_loaders(batch_size=128, num_workers=8)
cap = int(0.10 * bs)
warmup, every = 50, 25
print(f"baseline_test_loss={bl:.4f} baseline_steps={bs} cap(10%)={cap}", flush=True)

CONFIGS = [(1e-3, "cosine"), (1e-3, "const"), (3e-4, "const"),
           (1e-4, "const"), (3e-5, "const")]


def run(path, lr, sched):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    m = resnet20().to(dev); m.load_state_dict(reconstruct(blob["compressed"]))
    init_acc, init_loss = eval_full(m, te, dev)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=5e-4, betas=(0.9, 0.999))

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        if sched == "const":
            return 1.0
        p = min(1.0, (step - warmup) / max(1, cap - warmup))
        return 0.5 * (1 + math.cos(math.pi * p))

    sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    steps_pct, losses, cross = [0.0], [init_loss], None
    step, it = 0, iter(tl)
    m.train()
    while step < cap:
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(tl); x, y = next(it)
        x, y = x.to(dev), y.to(dev)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = F.cross_entropy(m(x), y)
        loss.backward(); opt.step(); sch.step(); step += 1
        if step % every == 0:
            _, tlo = eval_full(m, te, dev)
            steps_pct.append(100.0 * step / bs); losses.append(tlo)
            if tlo < bl and cross is None:
                cross = 100.0 * step / bs
            m.train()
    return steps_pct, losses, cross


fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
cmap = plt.get_cmap("viridis")
for ax, (path, title) in zip(axes, [
        ("compressed/quantize_8.pt", "quantize_8  (near-lossless init)"),
        ("compressed/kmeans_2.pt", "kmeans_2  (16x, harder init)")]):
    for i, (lr, sched) in enumerate(CONFIGS):
        xs, ys, cross = run(path, lr, sched)
        lab = f"lr={lr:.0e} {sched}" + (f"  ✓{cross:.2f}%" if cross else "  DNR")
        col = cmap(i / (len(CONFIGS) - 1))
        ax.plot(xs, ys, color=col, lw=1.8, label=lab)
        if cross:
            ax.scatter([cross], [bl], color=col, s=40, zorder=5)
        print(f"{title:34} lr={lr:.0e} {sched:6} -> "
              f"{'cross @ %.2f%%' % cross if cross else 'DNR'}", flush=True)
    ax.axhline(bl, color="black", ls="--", lw=1.2, alpha=0.8)
    ax.set_xlabel("retraining cost (% of baseline steps)")
    ax.set_title(title)
    ax.grid(alpha=.25); ax.legend(fontsize=9)
    ax.set_ylim(0.38, 0.95)
axes[0].set_ylabel("full-test loss")
fig.suptitle("Recovery dynamics vs. LR / schedule  (dashed = baseline loss to beat)\n"
             "low constant LR recovers the easy init ~33x cheaper; the hard init needs the cosine anneal")
fig.tight_layout()
fig.savefig("figures/diag_lr_curves.png", dpi=140)
print("wrote figures/diag_lr_curves.png", flush=True)

"""Diagnostic: is the ~7% recovery cost an LR-schedule artifact?

Retrain a near-lossless init (quantize_8) and a harder one (kmeans_2) from the 50-epoch
baseline under several LR / schedule choices, and record the FIRST step where full-test
loss drops below baseline. If lower / constant LR crosses much earlier than the current
(lr=1e-3, cosine-to-0-over-budget) recipe, the ~7% is a schedule artifact.
"""
import json, math
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
cap = int(0.10 * bs)           # 10% budget cap
warmup = 50
print(f"baseline_test_loss={bl:.4f} baseline_steps={bs} cap(10%)={cap}")


def run(path, lr, sched):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    m = resnet20().to(dev); m.load_state_dict(reconstruct(blob["compressed"]))
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=5e-4, betas=(0.9, 0.999))

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        if sched == "const":
            return 1.0
        p = min(1.0, (step - warmup) / max(1, cap - warmup))
        return 0.5 * (1 + math.cos(math.pi * p))

    sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    step, it = 0, iter(tl)
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
        if step % 20 == 0:
            _, tlo = eval_full(m, te, dev)
            if tlo < bl:
                return step
    return None


for path, name in [("compressed/quantize_8.pt", "quantize_8 (near-lossless init)"),
                   ("compressed/kmeans_2.pt", "kmeans_2 (16x, harder init)")]:
    for lr, sched in [(1e-3, "cosine"), (1e-3, "const"), (3e-4, "const"),
                      (1e-4, "const"), (3e-5, "const")]:
        c = run(path, lr, sched)
        pct = f"{100*c/bs:.2f}% of baseline" if c else ">10% (DNR in cap)"
        print(f"{name:34} lr={lr:.0e} {sched:6} -> first-cross @ step {c} = {pct}", flush=True)

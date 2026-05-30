"""Plot CRVQ-11M stage-2 LR sweep: valid loss vs epoch for each LR."""
import os, re, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG_DIR = "/workspace/projects/CRVQ/runs/lr_sweep_logs"
TAGS = [("1em5", "1e-5 (paper default)", "#d62728"),
        ("3em5", "3e-5",                 "#ff7f0e"),
        ("1em4", "1e-4",                 "#2ca02c"),
        ("3em4", "3e-4 (best)",          "#1f77b4")]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
final_ppls = {}
for tag, label, col in TAGS:
    path = os.path.join(LOG_DIR, f"{tag}.log")
    txt = open(path).read().replace("\r", "\n")
    vls = [float(x) for x in re.findall(r"valid loss=([0-9.e+-]+)", txt)]
    a1.plot(range(len(vls)), vls, marker="o", color=col, lw=1.8, ms=5, label=label)
    m = re.search(r"perplexity =\s*([0-9.]+)", txt[::-1])  # last occurrence
    ppl = float(re.findall(r"perplexity =\s*([0-9.]+)", txt)[-1])
    final_ppls[tag] = ppl
    a2.bar(label, ppl, color=col)
    a2.text(label, ppl + 0.1, f"{ppl:.2f}", ha="center", fontsize=10)

a1.set_xlabel("epoch (eval point)"); a1.set_ylabel("KL valid loss")
a1.set_title("Stage-2 KL distillation loss per LR  (lower = better)")
a1.legend(fontsize=9); a1.grid(True, alpha=.2)
a1.axhline(0, color="k", lw=.5)

a2.axhline(7.08, ls=":", color="k", lw=1, label="baseline (uncompressed) = 7.08")
a2.set_ylabel("Final SS-eval perplexity"); a2.set_title("Final perplexity after 10 epochs")
a2.legend(); a2.grid(True, alpha=.2, axis="y")

fig.suptitle("CRVQ on SimpleStories-11M: stage-2 LR sweep (4 LRs × 10 epochs)", fontweight="bold")
fig.tight_layout(); os.makedirs("/workspace/projects/CRVQ/figures", exist_ok=True)
fig.savefig("/workspace/projects/CRVQ/figures/lr_sweep.png", dpi=140); plt.close(fig)
print("wrote /workspace/projects/CRVQ/figures/lr_sweep.png")
print("final ppls:", final_ppls)

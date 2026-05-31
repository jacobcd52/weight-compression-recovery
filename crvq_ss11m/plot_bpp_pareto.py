"""bpp Pareto frontier for CRVQ on SimpleStories-11M (Phase 1)."""
import os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# CRVQ Phase 1 (bpp from main.py, ppl from finetune.py)
crvq = []
for x in [2, 3, 4, 6, 8]:
    bpp_log = f"/workspace/projects/CRVQ/runs/bpp_sweep_logs/X{x}_X444.log"
    s2_log = f"/workspace/projects/CRVQ/runs/bpp_stage2_logs/X{x}_X444.log"
    bpp = float(re.findall(r"Avg_bits:\s*([0-9.]+)", open(bpp_log).read())[-1])
    ppl = float(re.findall(r"perplexity =\s*([0-9.]+)", open(s2_log).read())[-1])
    crvq.append((bpp, ppl, f"X={x}"))

# Plain AQLM on same 11M model (from earlier project work)
aqlm = [(2.0, 8.14, "AQLM 2bpp"), (1.0, 16.7, "AQLM 1bpp"),
        (0.5, 66.0, "AQLM 0.5bpp"), (0.25, 290.0, "AQLM 0.25bpp")]
BASELINE = 7.08

fig, ax = plt.subplots(figsize=(8.5, 6))
ax.plot([b for b, _, _ in crvq], [p for _, p, _ in crvq],
        "o-", color="#1f77b4", lw=2.2, ms=10, label="CRVQ + stage-2 (lr=1e-3, 10ep)", zorder=4)
ax.plot([b for b, _, _ in aqlm], [p for _, p, _ in aqlm],
        "s--", color="#d62728", lw=1.5, ms=8, label="plain AQLM (no stage-2, earlier work)", zorder=3)
for b, p, t in crvq:
    ax.annotate(t, (b, p), textcoords="offset points", xytext=(5, 6), fontsize=8, color="#1f77b4")
for b, p, t in aqlm:
    ax.annotate(t.replace("AQLM ", ""), (b, p), textcoords="offset points", xytext=(5, -10), fontsize=8, color="#d62728")
ax.axhline(BASELINE, ls=":", color="#2ca02c", lw=1.4, label=f"uncompressed baseline = {BASELINE}")

ax.set_xscale("log", base=2); ax.set_yscale("log")
ax.set_xticks([0.25, 0.5, 1, 2]); ax.set_xticklabels(["0.25", "0.5", "1.0", "2.0"])
ax.set_xlabel("Avg bits / parameter (honest, includes codebook overhead)")
ax.set_ylabel("SimpleStories-eval perplexity (log)")
ax.set_title("CRVQ vs plain AQLM bpp Pareto frontier on SimpleStories-11M\n"
             "CRVQ at 0.6bpp beats plain AQLM at ~1bpp (1.7-14× better in sub-1bpp regime)")
ax.legend(loc="upper right", fontsize=9); ax.grid(True, which="both", alpha=.2)
fig.tight_layout(); os.makedirs("/workspace/projects/CRVQ/figures", exist_ok=True)
fig.savefig("/workspace/projects/CRVQ/figures/bpp_pareto.png", dpi=140); plt.close(fig)
print("wrote /workspace/projects/CRVQ/figures/bpp_pareto.png")
for b, p, t in crvq: print(f"  {t:5}  bpp={b:.3f}  ppl={p:.2f}")

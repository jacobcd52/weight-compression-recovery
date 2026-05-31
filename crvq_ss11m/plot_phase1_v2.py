"""Two figures for Phase 1:
  (a) bpp Pareto with code/codebook/scale breakdown.
  (b) Stage-2 KL valid-loss curves per X (compression-ratio context).
"""
import os, re, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

bd = json.load(open("/workspace/projects/CRVQ/runs/bpp_breakdown.json"))
XS = ["2", "3", "4", "6", "8"]
COLORS = {"2": "#4d3a9a", "3": "#7e5edb", "4": "#0f7b5e", "6": "#d28100", "8": "#a23434"}
PPLS = {}; CURVES = {}
for x in XS:
    log = f"/workspace/projects/CRVQ/runs/bpp_stage2_logs/X{x}_X444.log"
    txt = open(log).read().replace("\r", "\n")
    PPLS[x] = float(re.findall(r"perplexity =\s*([0-9.]+)", txt)[-1])
    CURVES[x] = [float(v) for v in re.findall(r"valid loss=([0-9.e+-]+)", txt)]

# ========== Figure A: Pareto with three series per point ==========
figA, ax = plt.subplots(figsize=(10, 6.5))
for x in XS:
    d = bd[x]; ppl = PPLS[x]
    total_r = d["total_bpp"] / 16
    code_r  = d["code_bpp"]  / 16   # asymptotic (scales with model size)
    cbook_r = d["codebook_bpp"] / 16  # fixed per layer, amortizes at scale
    # Three markers per X, same color
    ax.scatter([total_r], [ppl], s=140, marker="o", color=COLORS[x], edgecolor="k", lw=.5,
               label=f"X={x}  total" if x == XS[0] else None, zorder=5)
    ax.scatter([code_r], [ppl], s=110, marker="s", color=COLORS[x], edgecolor="k", lw=.5,
               label=f"X={x}  codes only (asymptotic)" if x == XS[0] else None, zorder=4)
    ax.scatter([cbook_r], [ppl], s=70, marker="^", facecolor="white", edgecolor=COLORS[x], lw=1.5,
               label=f"X={x}  codebook overhead" if x == XS[0] else None, zorder=4)
    ax.plot([cbook_r, code_r, total_r], [ppl]*3, ":", color=COLORS[x], alpha=.4, lw=1, zorder=2)
    ax.annotate(f"X={x}", (total_r, ppl), xytext=(7, 6), textcoords="offset points",
                color=COLORS[x], fontsize=9, weight="bold")

# Connect totals across X with a line
totals = [bd[x]["total_bpp"]/16 for x in XS]; codes = [bd[x]["code_bpp"]/16 for x in XS]
ax.plot(totals, [PPLS[x] for x in XS], "-", color="#444", lw=1.2, alpha=.5, zorder=3, label="Pareto (total)")
ax.plot(codes,  [PPLS[x] for x in XS], "--", color="#444", lw=1.2, alpha=.4, zorder=3, label="Pareto (asymptotic)")

ax.axhline(7.08, ls=":", color="#2ca02c", lw=1.4, label="uncompressed baseline = 7.08")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Compression ratio (vs fp16; lower = more compressed)")
ax.set_ylabel("SimpleStories-eval perplexity (log)")
ax.set_title("CRVQ-11M Phase 1 — bpp Pareto with code / codebook breakdown\n"
             "○ total honest   ■ codes only (asymptotic, large-scale)   △ codebook overhead (fixed per layer)")
# Make a clean legend
hh, ll = ax.get_legend_handles_labels()
# Keep only one of each kind
keep_labels = ["X=2  total", "X=2  codes only (asymptotic)", "X=2  codebook overhead",
               "Pareto (total)", "Pareto (asymptotic)", "uncompressed baseline = 7.08"]
hh = [h for h, l in zip(hh, ll) if l in keep_labels]
ll = [l for l in ll if l in keep_labels]
# relabel
ll = ["○ total honest", "■ codes only (asymptotic)", "△ codebook overhead",
      "Pareto curve (total)", "Pareto curve (asymptotic)", "uncompressed baseline = 7.08"]
ax.legend(hh, ll, loc="upper right", fontsize=9)
ax.grid(True, which="both", alpha=.18)
figA.tight_layout()
os.makedirs("/workspace/projects/CRVQ/figures", exist_ok=True)
figA.savefig("/workspace/projects/CRVQ/figures/phase1_pareto_v2.png", dpi=140)
plt.close(figA)
print("wrote phase1_pareto_v2.png")

# ========== Figure B: Stage-2 KL valid loss curves per X ==========
figB, ax = plt.subplots(figsize=(9, 6))
for x in XS:
    cur = CURVES[x]
    if not cur: continue
    label = f"X={x}  bpp={bd[x]['total_bpp']:.3f}  ratio={bd[x]['total_bpp']/16:.4f}  final ppl={PPLS[x]:.2f}"
    ax.plot(range(len(cur)), cur, marker="o", lw=1.8, ms=4.5, color=COLORS[x], label=label)
ax.set_xlabel("epoch (eval point)"); ax.set_ylabel("KL valid loss")
ax.set_title("Stage-2 KL valid loss per epoch, by compression point  (lr=1e-3, 10 epochs)\n"
             "Lower bpp starts much higher in KL but all curves still descending at epoch 10")
ax.legend(fontsize=8, loc="upper right"); ax.grid(True, alpha=.2)
ax.set_yscale("log")
figB.tight_layout()
figB.savefig("/workspace/projects/CRVQ/figures/phase1_loss_curves.png", dpi=140)
plt.close(figB)
print("wrote phase1_loss_curves.png")

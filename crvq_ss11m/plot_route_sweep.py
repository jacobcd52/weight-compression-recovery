"""Plot the route sweep: 4 routes to same asymptotic bpp, see how they differ in honest bpp + quality."""
import os, re, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

route_bpp = json.load(open("/workspace/projects/CRVQ/runs/route_bpp.json"))
phase1 = json.load(open("/workspace/projects/CRVQ/runs/bpp_breakdown.json"))

# Phase 1 ppls (10 epochs)
phase1_ppls = {}
for x in ["2", "3", "4", "6", "8"]:
    log = f"/workspace/projects/CRVQ/runs/bpp_stage2_logs/X{x}_X444.log"
    phase1_ppls[x] = float(re.findall(r"perplexity =\s*([0-9.]+)", open(log).read())[-1])

# Route ppls (5 epochs)
route_ppls = {}
for tag in route_bpp:
    log = f"/workspace/projects/CRVQ/runs/route_sweep_logs/{tag}.log"
    route_ppls[tag] = float(re.findall(r"perplexity =\s*([0-9.]+)", open(log).read())[-1])

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 6))
COL = {"X2_g8_mr0.025": "#4d3a9a", "X4_g16_mr0.05": "#0f7b5e",
       "X8_g32_mr0.1": "#a23434", "X1_g4_mr0.025": "#d28100"}
LABEL = {"X2_g8_mr0.025": "X=2 g=8 mr=0.025",
         "X4_g16_mr0.05": "X=4 g=16 mr=0.05",
         "X8_g32_mr0.1": "X=8 g=32 mr=0.1",
         "X1_g4_mr0.025": "X=1 g=4 mr=0.025"}

# Phase 1 frontier (honest + asymp) for context
p1_honest = [(phase1[x]["total_bpp"]/16, phase1_ppls[x], f"X={x}") for x in ["2","3","4","6","8"]]
p1_asymp = [(phase1[x]["code_bpp"]/16, phase1_ppls[x], f"X={x}") for x in ["2","3","4","6","8"]]

for ax, points, title in [(a1, p1_honest, "Honest compression ratio (incl codebook overhead)"),
                            (a2, p1_asymp, "Asymptotic compression ratio (codes only — large-scale limit)")]:
    # Phase 1 Pareto
    ax.plot([p[0] for p in points], [p[1] for p in points], "o-", color="#999",
            ms=8, lw=1.5, label="Phase 1 (X=[2,3,4,6,8], g=8, mr=0.025, 10 ep)", zorder=3)
    for x_, p_, t_ in points:
        ax.annotate(t_, (x_, p_), xytext=(4, -10), textcoords="offset points", fontsize=7.5, color="#666")

    # Route sweep points
    for tag, d in route_bpp.items():
        x_ = (d["honest_bpp"] if ax is a1 else d["code_bpp"]) / 16
        y_ = route_ppls[tag]
        ax.scatter([x_], [y_], s=180, marker="D", color=COL[tag], edgecolor="k", lw=.7, zorder=6,
                   label=LABEL[tag])
    ax.axhline(7.08, ls=":", color="#2ca02c", lw=1.4, label="baseline = 7.08")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Compression ratio (vs fp16)")
    ax.set_ylabel("SS-eval perplexity (log)")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, which="both", alpha=.2)

fig.suptitle("CRVQ-11M: route sweep at fixed asymptotic compression (0.288 bpp = 0.018 ratio)\n"
             "4 routes give same asymp ratio but very different honest ratio at 11M scale (codebook overhead)",
             fontweight="bold")
fig.tight_layout()
os.makedirs("/workspace/projects/CRVQ/figures", exist_ok=True)
fig.savefig("/workspace/projects/CRVQ/figures/route_sweep.png", dpi=140); plt.close(fig)
print("wrote route_sweep.png")

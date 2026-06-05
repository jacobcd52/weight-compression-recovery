"""Overlay the AQLM points (runs/aqlm_frontier/*.json) on the 11M seed frontier
(runs/llm_seedfrontier/frontier.json) to see whether AQLM is a Pareto improvement.
x = compression ratio (log), y = recovery cost (% of from-scratch). Plots AQLM at BOTH the honest
ratio (comparable to the seed frontier) and the amortized ratio (large-model projection).

    python -m src.llm.plot_aqlm_frontier
"""
import glob, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    fr = json.load(open("runs/llm_seedfrontier/frontier.json"))
    budget_pct = fr["budget_fraction"] * 100
    seed = [r for r in fr["results"] if r.get("valid", True)]
    aqlm = [json.load(open(p)) for p in sorted(glob.glob("runs/aqlm_frontier/*.json"))]

    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    # existing seed/auto frontier points
    rec = [r for r in seed if r.get("recovered")]; dnr = [r for r in seed if not r.get("recovered")]
    ax.scatter([r["ratio"] for r in rec], [r["recovery_fraction"]*100 for r in rec],
               s=70, c="#888", edgecolor="k", linewidth=.5, zorder=3, label="prior methods (recover)")
    if dnr:
        ax.scatter([r["ratio"] for r in dnr], [budget_pct]*len(dnr), s=60, facecolors="none",
                   edgecolors="#bbb", linewidth=1.2, zorder=2, label="prior methods (DNR)")
    for r in seed:
        y = r["recovery_fraction"]*100 if r.get("recovered") else budget_pct
        ax.annotate(r["name"].replace("seed_", "").replace("resnet_best_", "RN:"),
                    (r["ratio"], y), textcoords="offset points", xytext=(4, 3), fontsize=7, color="#aaa")

    # AQLM points: honest ratio (filled blue) + amortized ratio (open blue), connected
    for a in aqlm:
        yh = a["recovery_fraction"]*100 if a.get("recovered") else budget_pct
        ax.plot([a["ratio_amortized"], a["ratio_honest"]], [yh, yh], ":", color="#1f77b4", lw=.8, zorder=3)
    ah = [a for a in aqlm]
    ax.scatter([a["ratio_honest"] for a in ah], [a["recovery_fraction"]*100 if a.get("recovered") else budget_pct for a in ah],
               s=95, marker="D", c="#1f77b4", edgecolor="k", linewidth=.6, zorder=5, label="AQLM (honest ratio)")
    ax.scatter([a["ratio_amortized"] for a in ah], [a["recovery_fraction"]*100 if a.get("recovered") else budget_pct for a in ah],
               s=95, marker="D", facecolors="none", edgecolors="#1f77b4", linewidth=1.6, zorder=5, label="AQLM (amortized ratio)")
    for a in ah:
        y = a["recovery_fraction"]*100 if a.get("recovered") else budget_pct
        ax.annotate(f"{a['nominal_bpp']:g}bpp", (a["ratio_amortized"], y),
                    textcoords="offset points", xytext=(-2, 6), fontsize=7.5, color="#1f77b4")

    ax.axhline(budget_pct, ls="--", c="#7d3a3a", lw=1, label=f"{budget_pct:.0f}% budget cap (DNR above)")
    ax.set_xscale("log")
    ax.set_xlabel("compression ratio (log — lower = more compressed)")
    ax.set_ylabel("recovery cost (% of from-scratch training)")
    ax.set_title("11M frontier: AQLM (full paper method) vs prior methods")
    ax.legend(fontsize=8.5, loc="upper left"); ax.grid(True, which="both", alpha=.15)
    fig.tight_layout(); os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/llm_aqlm_frontier.png", dpi=130); plt.close(fig)
    print("wrote figures/llm_aqlm_frontier.png")
    print("AQLM points:")
    for a in sorted(ah, key=lambda x: x["nominal_bpp"]):
        tag = f"REC@{a['recovery_fraction']*100:.1f}%" if a.get("recovered") else "DNR"
        print(f"  {a['nominal_bpp']:g}bpp: honest={a['ratio_honest']:.4f} amort={a['ratio_amortized']:.4f} "
              f"recon={a['recon_loss']:.2f} {tag}")


if __name__ == "__main__":
    main()

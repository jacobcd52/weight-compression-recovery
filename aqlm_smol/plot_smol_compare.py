"""SmolLM2-360M honest head-to-head: AQLM (calibrated VQ, +free in-datacenter stage-2 PV-tuning)
vs naive per-row uniform quantization. x = honest block bits/param (codebooks/scales included),
y = WikiText2 perplexity. All points use AQLM's identical evaluate_perplexity (seqlen 2048).
Output: figures/smol_aqlm_vs_simple.png
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

simple = json.load(open("runs/simple_eval.json"))
aq = json.load(open("runs/aqlm_smol_summary.json"))
base = aq["baseline"]["wikitext2"]

fig, ax = plt.subplots(figsize=(9, 6))

# naive uniform quant (no finetune)
s = [r for r in simple if r["method"].startswith("uniform")]
ax.plot([r["bpp_block"] for r in s], [r["wikitext2"] for r in s], "o-", color="#d62728",
        ms=9, lw=1.6, label="naive uniform quant (no prep)", zorder=4)
for r in s:
    ax.annotate(r["method"].replace("uniform_", "").replace("bit", "b"),
                (r["bpp_block"], r["wikitext2"]), textcoords="offset points", xytext=(6, 5), fontsize=8, color="#d62728")

# AQLM stage-1 (no finetune) and stage-2 (free prep), with arrows showing the finetune gain
b1 = [p["bpp_block"] for p in aq["points"]]
s1 = [p["stage1_wt2"] for p in aq["points"]]
s2 = [p["stage2_wt2"] for p in aq["points"]]
ax.plot(b1, s1, "s--", color="#9467bd", ms=8, lw=1.2, alpha=.7, label="AQLM stage-1 only (no prep)", zorder=4)
ax.plot(b1, s2, "D-", color="#1f77b4", ms=11, lw=2.0, label="AQLM + free stage-2 PV-tuning", zorder=6)
for x, y1, y2 in zip(b1, s1, s2):
    ax.annotate("", xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.3, alpha=.55), zorder=5)
for p in aq["points"]:
    ax.annotate(f"{p['nominal_bpp']:g}bpp\n→{p['stage2_wt2']:.0f}", (p["bpp_block"], p["stage2_wt2"]),
                textcoords="offset points", xytext=(7, -4), fontsize=8.5, color="#1f77b4", weight="bold")

ax.axhline(base, ls=":", color="#2ca02c", lw=1.5, label=f"original (uncompressed) = {base:.1f}")
ax.set_yscale("log"); ax.set_xscale("log", base=2)
ax.set_xticks([0.5, 1, 2, 3, 4, 8, 16]); ax.set_xticklabels(["0.5", "1", "2", "3", "4", "8", "16"])
ax.set_xlabel("honest compression — block bits / parameter (lower = smaller)")
ax.set_ylabel("WikiText2 perplexity (log)")
ax.set_title("SmolLM2-360M: AQLM (calibrated VQ + free stage-2) vs naive quant\n"
             "compressed-model quality vs size (transformer-block weights)")
ax.legend(fontsize=9, loc="upper right"); ax.grid(True, which="both", alpha=.18)
fig.tight_layout(); os.makedirs("figures", exist_ok=True)
fig.savefig("figures/smol_aqlm_vs_simple.png", dpi=140); plt.close(fig)
print("wrote figures/smol_aqlm_vs_simple.png")
print(f"baseline wt2={base:.2f}")
print("simple:", [(r['method'], round(r['bpp_block'],2), round(r['wikitext2'],1)) for r in s])
print("AQLM  :", [(p['nominal_bpp'], round(p['bpp_block'],2), round(p['stage2_wt2'],1)) for p in aq['points']])

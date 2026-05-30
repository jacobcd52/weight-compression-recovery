"""Scale-up comparison: SmolLM2-360M vs TinyLlama-1.1B on the AQLM-vs-naive-quant frontier.
Both axes log; identical eval methodology for every point (AQLM evaluate_perplexity, seqlen 2048).
Output: figures/scaleup_aqlm_vs_simple.png
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(simple_path, aq_path):
    s = json.load(open(simple_path))
    s = [r for r in s if r["method"].startswith("uniform")]
    a = json.load(open(aq_path))
    return s, a


s_smol, a_smol = load("runs/simple_eval.json", "runs/aqlm_smol_summary.json")
s_tiny, a_tiny = load("runs/simple_eval_tiny.json", "runs/aqlm_tiny_summary.json")

fig, ax = plt.subplots(figsize=(10, 6.5))


def plot_model(s, a, *, color_simple, color_aqlm, label_prefix, marker, lstyle):
    base = a["baseline"]["wikitext2"]
    # naive
    ax.plot([r["bpp_block"] for r in s], [r["wikitext2"] for r in s],
            marker=marker, color=color_simple, ms=8, lw=1.4, linestyle=lstyle,
            label=f"{label_prefix} naive uniform quant", zorder=3)
    # AQLM stage-2 (free prep)
    ax.plot([p["bpp_block"] for p in a["points"]], [p["stage2_wt2"] for p in a["points"]],
            marker=marker, color=color_aqlm, ms=11, lw=2.2, linestyle=lstyle,
            label=f"{label_prefix} AQLM + free stage-2", zorder=5)
    # baseline
    ax.axhline(base, ls=":", color=color_aqlm, lw=1.1, alpha=.6)
    ax.text(0.42, base, f"{label_prefix} base {base:.1f}", fontsize=8, color=color_aqlm,
            va="bottom", ha="left")


plot_model(s_smol, a_smol, color_simple="#d62728", color_aqlm="#9467bd",
           label_prefix="SmolLM2-360M:", marker="o", lstyle="--")
plot_model(s_tiny, a_tiny, color_simple="#ff7f0e", color_aqlm="#1f77b4",
           label_prefix="TinyLlama-1.1B:", marker="D", lstyle="-")

ax.set_yscale("log"); ax.set_xscale("log", base=2)
ax.set_xticks([0.5, 1, 2, 3, 4, 8, 16]); ax.set_xticklabels(["0.5", "1", "2", "3", "4", "8", "16"])
ax.set_xlabel("honest compression — block bits / parameter (lower = smaller)")
ax.set_ylabel("WikiText2 perplexity (log)")
ax.set_title("Scale-up: AQLM (+free stage-2) vs naive quant on SmolLM2-360M and TinyLlama-1.1B\n"
             "compressed-model quality vs size (transformer-block weights)")
ax.legend(fontsize=9, loc="upper right"); ax.grid(True, which="both", alpha=.18)
fig.tight_layout(); os.makedirs("figures", exist_ok=True)
fig.savefig("figures/scaleup_aqlm_vs_simple.png", dpi=140); plt.close(fig)
print("wrote figures/scaleup_aqlm_vs_simple.png")
for tag, a in [("smol", a_smol), ("tiny", a_tiny)]:
    print(f"  {tag} AQLM stage-2:",
          [(p["nominal_bpp"], round(p["bpp_block"],3), round(p["stage2_wt2"],1)) for p in a["points"]])

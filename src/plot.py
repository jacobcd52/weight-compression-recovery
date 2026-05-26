"""Build the Pareto figure + results/summary.csv from all retrain summaries.

x: compression ratio (log, lower = more compressed)
y: recovery fraction (linear, 0..0.10); DNR runs sit at the budget cap with a
   right-arrow annotation. Color = technique, marker = plain vs distill.
The Pareto lower-left envelope over recovered runs is drawn as a line.
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

EXCLUDE = {"baseline"}


def load_summaries(runs_dir="runs"):
    rows = []
    for path in sorted(glob.glob(os.path.join(runs_dir, "*", "summary.json"))):
        with open(path) as f:
            s = json.load(f)
        name = os.path.basename(os.path.dirname(path))
        if name in EXCLUDE or "technique" not in s or s.get("smoke"):
            continue
        rows.append(s)
    return pd.DataFrame(rows)


def pareto_front(df):
    """Lower-left Pareto front over recovered runs (min ratio & min recovery_fraction)."""
    rec = df[df["recovered"]].copy()
    if rec.empty:
        return rec
    rec = rec.sort_values(["compression_ratio", "recovery_fraction"])
    front, best_y = [], float("inf")
    for _, r in rec.iterrows():
        if r["recovery_fraction"] <= best_y + 1e-12:
            front.append(r)
            best_y = r["recovery_fraction"]
    return pd.DataFrame(front)


def make_plot(df, out_dir="figures", budget=0.10):
    os.makedirs(out_dir, exist_ok=True)
    techniques = sorted(df["technique"].unique())
    cmap = plt.get_cmap("tab10")
    colors = {t: cmap(i % 10) for i, t in enumerate(techniques)}

    plt.rcParams.update({"font.size": 13})
    fig, ax = plt.subplots(figsize=(11, 7))

    for t in techniques:
        sub = df[df["technique"] == t]
        c = colors[t]
        for distill, marker in [(False, "o"), (True, "^")]:
            d = sub[(sub["did_distill"] == distill) & (sub["recovered"])]
            if not d.empty:
                ax.scatter(d["compression_ratio"], d["recovery_fraction"],
                           s=90, color=c, marker=marker, edgecolors="k",
                           linewidths=0.5, zorder=3)
            dnr = sub[(sub["did_distill"] == distill) & (~sub["recovered"])]
            if not dnr.empty:
                ax.scatter(dnr["compression_ratio"], [budget] * len(dnr),
                           s=110, color=c, marker=marker, facecolors="none",
                           linewidths=1.6, zorder=3)
                for _, r in dnr.iterrows():
                    ax.annotate("", xy=(r["compression_ratio"] * 1.6, budget),
                                xytext=(r["compression_ratio"], budget),
                                arrowprops=dict(arrowstyle="->", color=c, lw=1.4),
                                zorder=2)

    front = pareto_front(df)
    if len(front) >= 1:
        front = front.sort_values("compression_ratio")
        ax.plot(front["compression_ratio"], front["recovery_fraction"],
                "-", color="black", lw=2.0, alpha=0.7, zorder=4)

    ax.axhline(budget, color="grey", ls="--", lw=1.0, alpha=0.7)

    ax.set_xscale("log")
    ax.set_xlabel("Compression ratio  (compressed / fp32 bytes; lower = more compressed)")
    ax.set_ylabel("Recovery fraction  (retrain steps / baseline steps)")
    ax.set_ylim(0, budget * 1.08)
    ax.set_title("Weight-compression recovery: compression vs. retraining cost\n"
                 "ResNet-20 / CIFAR-10 (open markers at top = DNR within 10% budget)")
    ax.grid(True, which="both", alpha=0.25)

    # Two legends: one mapping every technique to its colour (so the DNR-only
    # techniques are identifiable too), one explaining marker style.
    from matplotlib.lines import Line2D
    tech_handles = [Line2D([], [], marker="o", linestyle="none",
                           markerfacecolor=colors[t], markeredgecolor="k",
                           markersize=8, label=t) for t in techniques]
    style_handles = [
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="grey",
               markeredgecolor="k", markersize=8, label="recovered (plain)"),
        Line2D([], [], marker="^", linestyle="none", markerfacecolor="grey",
               markeredgecolor="k", markersize=8, label="recovered (distill)"),
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="none",
               markeredgecolor="grey", markersize=9, label="DNR (≥ 10% budget)"),
        Line2D([], [], color="black", lw=2, alpha=0.7, label="Pareto frontier"),
    ]
    leg1 = ax.legend(handles=tech_handles, loc="center left", fontsize=9,
                     title="technique", framealpha=0.93)
    ax.add_artist(leg1)
    ax.legend(handles=style_handles, loc="lower right", fontsize=9, framealpha=0.93)
    fig.tight_layout()

    png = os.path.join(out_dir, "pareto.png")
    pdf = os.path.join(out_dir, "pareto.pdf")
    fig.savefig(png, dpi=150)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"wrote {png} and {pdf}")


def main():
    df = load_summaries()
    if df.empty:
        print("no retrain summaries found yet")
        return
    os.makedirs("results", exist_ok=True)
    cols = ["run_name", "technique", "knob", "mode", "did_distill",
            "compression_ratio", "compressed_bytes", "recovered",
            "recovery_steps", "recovery_fraction", "init_acc",
            "final_test_acc", "baseline_test_acc"]
    cols = [c for c in cols if c in df.columns]
    df_sorted = df.sort_values(["technique", "compression_ratio"])
    df_sorted.to_csv("results/summary.csv", index=False, columns=cols)
    print(f"wrote results/summary.csv ({len(df_sorted)} runs)")
    make_plot(df)


if __name__ == "__main__":
    main()

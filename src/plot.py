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
        # only the current regime: recovery measured by TEST LOSS (has final_test_loss)
        if "final_test_loss" not in s:
            continue
        s["run_name"] = name
        rows.append(s)
    # Prefer fine-cadence re-runs: if "<x>_fine" exists, drop the coarse "<x>".
    names = {r["run_name"] for r in rows}
    superseded = {n[:-5] for n in names if n.endswith("_fine")}
    rows = [r for r in rows if r["run_name"] not in superseded]
    return pd.DataFrame(rows)


def attach_amortized(df, compressed_dir="compressed"):
    """Add an 'amortized_ratio' column: per-weight bytes only (codebook/scale overhead
    removed), computed from the saved compressed files. Approximates the ratio at large
    model size. Falls back to the full ratio if the compressed file is missing."""
    import torch
    from .compress import amortized_bytes
    vals = []
    for _, r in df.iterrows():
        base = r["run_name"]
        for suf in ("_distill", "_fine", "_smoke"):
            if base.endswith(suf):
                base = base[: -len(suf)]
        p = os.path.join(compressed_dir, base + ".pt")
        if os.path.exists(p):
            blob = torch.load(p, map_location="cpu", weights_only=False)
            vals.append(amortized_bytes(blob["compressed"]) / blob["baseline_bytes"])
        else:
            vals.append(r["compression_ratio"])
    df = df.copy()
    df["amortized_ratio"] = vals
    return df


def pareto_front(df, ratio_col="compression_ratio"):
    """Lower-left Pareto front over recovered runs (min ratio & min recovery_fraction)."""
    rec = df[df["recovered"]].copy()
    if rec.empty:
        return rec
    rec = rec.sort_values([ratio_col, "recovery_fraction"])
    front, best_y = [], float("inf")
    for _, r in rec.iterrows():
        if r["recovery_fraction"] <= best_y + 1e-12:
            front.append(r)
            best_y = r["recovery_fraction"]
    return pd.DataFrame(front)


def make_plot(df, out_dir="figures", budget=0.10, ratio_col="compression_ratio",
              out_name="pareto", subtitle="honest byte accounting"):
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
                ax.scatter(d[ratio_col], d["recovery_fraction"],
                           s=90, color=c, marker=marker, edgecolors="k",
                           linewidths=0.5, zorder=3)
            dnr = sub[(sub["did_distill"] == distill) & (~sub["recovered"])]
            if not dnr.empty:
                ax.scatter(dnr[ratio_col], [budget] * len(dnr),
                           s=110, color=c, marker=marker, facecolors="none",
                           linewidths=1.6, zorder=3)
                for _, r in dnr.iterrows():
                    ax.annotate("", xy=(r[ratio_col] * 1.6, budget),
                                xytext=(r[ratio_col], budget),
                                arrowprops=dict(arrowstyle="->", color=c, lw=1.4),
                                zorder=2)

    front = pareto_front(df, ratio_col)
    if len(front) >= 1:
        front = front.sort_values(ratio_col)
        ax.plot(front[ratio_col], front["recovery_fraction"],
                "-", color="black", lw=2.0, alpha=0.7, zorder=4)

    ax.axhline(budget, color="grey", ls="--", lw=1.0, alpha=0.7)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(f"Compression ratio  ({subtitle}; lower = more compressed)")
    ax.set_ylabel("Recovery cost  (retrain steps / baseline steps, log)")
    ax.set_ylim(8e-4, budget * 1.35)
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

    png = os.path.join(out_dir, f"{out_name}.png")
    pdf = os.path.join(out_dir, f"{out_name}.pdf")
    fig.savefig(png, dpi=150)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"wrote {png} and {pdf}")


def main():
    df = load_summaries()
    if df.empty:
        print("no retrain summaries found yet")
        return
    df = attach_amortized(df)
    os.makedirs("results", exist_ok=True)
    cols = ["run_name", "technique", "knob", "mode", "did_distill",
            "compression_ratio", "amortized_ratio", "compressed_bytes", "recovered",
            "recovery_steps", "recovery_fraction", "init_acc", "init_loss",
            "final_test_acc", "final_test_loss", "baseline_test_acc", "baseline_test_loss"]
    cols = [c for c in cols if c in df.columns]
    df_sorted = df.sort_values(["technique", "compression_ratio"])
    df_sorted.to_csv("results/summary.csv", index=False, columns=cols)
    print(f"wrote results/summary.csv ({len(df_sorted)} runs)")
    make_plot(df, ratio_col="compression_ratio", out_name="pareto",
              subtitle="honest bytes incl. codebook/scale overhead")
    make_plot(df, ratio_col="amortized_ratio", out_name="pareto_amortized",
              subtitle="amortized: size-independent codebook/scale overhead removed")


if __name__ == "__main__":
    main()

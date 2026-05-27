"""Generate the auto-research report page: docs/autoresearch.html.

Reads runs/autoresearch/archive.json, draws two figures (the discovered Pareto
frontier, and the search-progress curve = best ratio vs proposal number), and
writes a self-contained, GitHub-Pages-friendly page linked from index.html:
    https://jacobcd52.github.io/weight-compression-recovery/autoresearch.html

Run:  python -m src.autoresearch.report
"""
import base64
import datetime
import html
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..report import CSS, _img_data_uri

ARCHIVE = "runs/autoresearch/archive.json"
FIG_PARETO = "figures/autoresearch_pareto.png"
FIG_PROGRESS = "figures/autoresearch_progress.png"


def _load():
    d = json.load(open(ARCHIVE))
    return d["entries"], float(d.get("budget_fraction", 0.05))


def _pareto(recovered):
    """Non-dominated set on (ratio down, recovery_fraction down)."""
    front = []
    for a in recovered:
        dominated = any(
            b is not a and b["ratio"] <= a["ratio"]
            and b["recovery_fraction"] <= a["recovery_fraction"]
            and (b["ratio"] < a["ratio"] or b["recovery_fraction"] < a["recovery_fraction"])
            for b in recovered)
        if not dominated:
            front.append(a)
    return sorted(front, key=lambda e: e["ratio"])


def _idx(name):
    m = re.search(r"(\d+)$", name)
    return int(m.group(1)) if m else 0


def make_figures(entries, budget):
    os.makedirs("figures", exist_ok=True)
    rec = [e for e in entries if e.get("recovered")]
    seeds = [e for e in rec if e.get("gen") == 0]
    evolved = [e for e in rec if e.get("gen") == 1]
    front = _pareto(rec)

    # ---- Pareto frontier ----
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    if seeds:
        ax.scatter([e["ratio"] for e in seeds], [e["recovery_fraction"] * 100 for e in seeds],
                   s=70, c="#f0a030", marker="s", edgecolor="k", linewidth=.6,
                   zorder=3, label="hand-written seeds")
    if evolved:
        ax.scatter([e["ratio"] for e in evolved], [e["recovery_fraction"] * 100 for e in evolved],
                   s=42, c="#5ab1ef", marker="o", edgecolor="k", linewidth=.4,
                   alpha=.85, zorder=3, label="LLM-evolved")
    ax.plot([e["ratio"] for e in front], [e["recovery_fraction"] * 100 for e in front],
            "-o", color="k", lw=2, ms=7, zorder=4, label="Pareto frontier")
    for e in front:
        ax.annotate(e["name"].replace("seed_", "").replace("cand_", "c"),
                    (e["ratio"], e["recovery_fraction"] * 100),
                    textcoords="offset points", xytext=(6, 6), fontsize=8, color="#cfd8e3")
    ax.axhline(budget * 100, ls="--", c="#7d3a3a", lw=1)
    ax.text(ax.get_xlim()[1], budget * 100, f" {budget*100:.0f}% budget cap",
            color="#7d3a3a", fontsize=8, va="bottom", ha="right")
    ax.set_xscale("log")
    ax.set_xlabel("compression ratio  (compressed bytes / fp32 bytes — lower = more compressed)")
    ax.set_ylabel("recovery cost  (% of 30-epoch training)")
    ax.set_title("Auto-research: discovered compression / recovery frontier")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, which="both", alpha=.15)
    fig.tight_layout(); fig.savefig(FIG_PARETO, dpi=130); plt.close(fig)

    # ---- search progress: best recovered ratio vs proposal index ----
    ev_sorted = sorted(evolved, key=lambda e: _idx(e["name"]))
    seed_best = min((e["ratio"] for e in seeds), default=1.0)
    xs, ys, best = [], [], seed_best
    for e in ev_sorted:
        best = min(best, e["ratio"])
        xs.append(_idx(e["name"])); ys.append(best)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.axhline(seed_best, ls="--", c="#f0a030", lw=1.5, label=f"best seed (quant4) = {seed_best:.3f}")
    if xs:
        ax.step(xs, ys, where="post", color="#5ab1ef", lw=2, label="best evolved (recovers, running min)")
        ax.scatter([_idx(e["name"]) for e in ev_sorted],
                   [e["ratio"] for e in ev_sorted], s=14, c="#3a566b", alpha=.5, zorder=2,
                   label="individual recovered candidates")
    ax.set_xlabel("proposal number")
    ax.set_ylabel("compression ratio (lower = better)")
    ax.set_title("Search progress — best recovering scheme found so far")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=.15)
    fig.tight_layout(); fig.savefig(FIG_PROGRESS, dpi=130); plt.close(fig)
    return front


def _winner_code(entries):
    """Source of the lowest-ratio recovering candidate, for display."""
    rec = [e for e in entries if e.get("recovered")]
    w = min(rec, key=lambda e: e["ratio"])
    return w, w["code"]


def build_html(entries, budget, front):
    now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    n = len(entries)
    nrec = sum(1 for e in entries if e.get("recovered"))
    ninvalid = sum(1 for e in entries if not e.get("valid", True))
    seed_best = min((e["ratio"] for e in entries if e.get("gen") == 0 and e.get("recovered")),
                    default=float("nan"))
    best = min((e for e in entries if e.get("recovered")), key=lambda e: e["ratio"])
    winner, wcode = _winner_code(entries)

    uri_pareto = _img_data_uri(FIG_PARETO)
    uri_prog = _img_data_uri(FIG_PROGRESS)

    frow = []
    for e in front:
        kind = "seed" if e.get("gen") == 0 else "evolved"
        frow.append(
            f"<tr class='rec'><td>{html.escape(e['name'])}</td><td>{kind}</td>"
            f"<td>{e['ratio']:.4f}</td><td>{1/e['ratio']:.1f}&times;</td>"
            f"<td>{e['recovery_fraction']*100:.2f}%</td>"
            f"<td>{html.escape(str(e.get('best_lr','')))}</td>"
            f"<td>{html.escape(', '.join(e.get('parents') or []))}</td></tr>")
    ftable = "".join(frow)

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Auto-research — evolving weight-compression schemes</title>
<style>{CSS}
pre{{background:#0b0e12;border:1px solid var(--line);border-radius:10px;padding:14px;
overflow:auto;font-size:12.5px;line-height:1.4}}
</style></head><body><div class="wrap">

<h1>Auto-research: LLM-evolved compression schemes</h1>
<p class="sub">An LLM (Claude Sonnet) repeatedly proposes a per-tensor weight-compression scheme;
each is sandbox-built, its honest compressed size measured, and its <i>recovery cost</i> scored by
the vmap ensemble. Survivors seed the next proposals (FunSearch / MAP-Elites style).<br>
ResNet-20 / CIFAR-10 (GroupNorm) &middot; {now} &middot;
<a href="index.html">&larr; back to main report</a></p>

<div class="kpi">
  <div class="b"><div class="n">{n}</div><div class="l">schemes evaluated</div></div>
  <div class="b"><div class="n">{nrec}</div><div class="l">recovered in budget</div></div>
  <div class="b"><div class="n">{best['ratio']:.3f}</div><div class="l">best ratio (≈{1/best['ratio']:.0f}× smaller)</div></div>
  <div class="b"><div class="n">{best['ratio']/seed_best:.2f}×</div><div class="l">vs best hand-written seed</div></div>
</div>

<h2>What the search found</h2>
<div class="card"><ul>
<li>Starting from hand-written seeds (best: <b>4-bit quant @ ratio {seed_best:.3f}</b>, which recovers
for free), the LLM pushed the recovering frontier down to <b>ratio {best['ratio']:.3f}</b>
(≈{1/best['ratio']:.0f}× smaller than fp32) at a cost of only
<b>{best['recovery_fraction']*100:.1f}%</b> of a 30-epoch training run — well inside the
{budget*100:.0f}% budget.</li>
<li>Every frontier point the LLM discovered is a variant of one idea it converged on:
<b>group-wise low-bit quantization with a NormalFloat codebook</b> — i.e. split each weight tensor
into small groups, give each group its own scale, and quantize to 2 bits using levels placed at the
quantiles of a Gaussian. This is exactly the recipe behind modern LLM weight quantization
(GPTQ / AWQ / QLoRA's NF4, AQLM) — <i>rediscovered from scratch</i> by the search.</li>
<li>The frontier is a clean trade-off swept via the <code>group_size</code> knob: bigger groups =
fewer scales to store = lower ratio, but slightly higher recovery cost. The winner
(<b>{html.escape(winner['name'])}</b>) instead keeps small groups for fidelity and
<i>delta-encodes + 8-bit-quantizes the per-group scales</i> so the scale overhead doesn't dominate.</li>
<li>{ninvalid} of {n} proposals were rejected by the static safety screen or failed to build/run
(they don't count against the frontier).</li>
</ul></div>

<h2>Discovered Pareto frontier</h2>
<div class="card">{('<img alt="auto-research pareto" src="'+uri_pareto+'">') if uri_pareto else
'<p class="sub">figure pending.</p>'}
<p class="sub">Orange squares = the six hand-written seeds; blue dots = LLM-evolved schemes (only
those that recovered within budget are shown); black line = the non-dominated frontier. Left and
down is better (smaller + cheaper to recover).</p></div>

<h2>Search progress</h2>
<div class="card">{('<img alt="search progress" src="'+uri_prog+'">') if uri_prog else
'<p class="sub">figure pending.</p>'}
<p class="sub">Running minimum compression ratio among schemes that still recover, vs. proposal
number. The dashed orange line is the best hand-written seed; the search first matches it, then
breaks below it once it discovers group-wise + NormalFloat quantization.</p></div>

<h2>Frontier points</h2>
<div class="card" style="overflow:auto">
<table><thead><tr><th>name</th><th>origin</th><th>ratio</th><th>× smaller</th>
<th>recovery cost</th><th>best LR</th><th>parents</th></tr></thead>
<tbody>{ftable}</tbody></table></div>

<h2>Winning program ({html.escape(winner['name'])}, ratio {winner['ratio']:.4f})</h2>
<div class="card"><pre>{html.escape(wcode)}</pre></div>

<p class="foot">Compressed size is measured as <code>len(zlib.compress(pickle(payload)))</code> on a
single blob — an objective, hard-to-game byte count. Each scheme is built in a sandboxed subprocess
(static screen for IO / eval / network / pickle-load; timeout) and only the reconstructed
initialization is returned for scoring. Repo:
<a href="https://github.com/jacobcd52/weight-compression-recovery">jacobcd52/weight-compression-recovery</a>.
Regenerated by <code>python -m src.autoresearch.report</code>.</p>
</div></body></html>"""


def main():
    entries, budget = _load()
    front = make_figures(entries, budget)
    os.makedirs("docs", exist_ok=True)
    htmltext = build_html(entries, budget, front)
    with open("docs/autoresearch.html", "w") as f:
        f.write(htmltext)
    print(f"wrote docs/autoresearch.html ({len(entries)} schemes, {len(front)} frontier points, "
          f"best ratio {min(e['ratio'] for e in entries if e.get('recovered')):.4f})")


if __name__ == "__main__":
    main()

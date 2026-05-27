"""Build the LLM auto-research report page: docs/llm_autoresearch.html — the evolved
compression/recovery frontier (vs the hand-written seeds), the search-progress curve, a
by-specialization breakdown, and the winning scheme's source. Linked from llm.html.

    python -m src.llm.autoresearch_report
"""
import datetime
import html
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..report import CSS, _img_data_uri

ARCHIVE = "runs/llm_autoresearch/archive.json"
FIG_PARETO = "figures/llm_ar_pareto.png"
FIG_PROG = "figures/llm_ar_progress.png"
FAM_COLOR = {"scalar": "#5ab1ef", "vq": "#e0529c", "entropy": "#2ea043", "hybrid": "#f0a030",
             "seed": "#888888", "evolved": "#b39ddb"}


def _pareto(rec):
    out = []
    for a in rec:
        if not any(b is not a and b["ratio"] <= a["ratio"]
                   and b["recovery_fraction"] <= a["recovery_fraction"]
                   and (b["ratio"] < a["ratio"] or b["recovery_fraction"] < a["recovery_fraction"])
                   for b in rec):
            out.append(a)
    return sorted(out, key=lambda e: e["ratio"])


def _idx(name):
    m = re.search(r"(\d+)$", name)
    return int(m.group(1)) if m else 0


def make_figs(entries, budget_pct):
    os.makedirs("figures", exist_ok=True)
    rec = [e for e in entries if e.get("recovered")]
    dnr = [e for e in entries if e.get("valid", True) and not e.get("recovered")]
    front = _pareto(rec)

    # --- Pareto: evolved frontier, coloured by specialization ---
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    for fam in sorted(set(e.get("family", "?") for e in rec)):
        pts = [e for e in rec if e.get("family") == fam]
        ax.scatter([p["ratio"] for p in pts], [p["recovery_fraction"] * 100 for p in pts],
                   s=46, c=FAM_COLOR.get(fam, "#ccc"), edgecolor="k", linewidth=.4, alpha=.9,
                   zorder=3, label=fam)
    if dnr:
        ax.scatter([e["ratio"] for e in dnr], [budget_pct] * len(dnr), s=42, facecolors="none",
                   edgecolors="#c0506a", linewidth=1.0, zorder=2, label="DNR")
    ax.plot([e["ratio"] for e in front], [e["recovery_fraction"] * 100 for e in front],
            "-o", color="k", lw=2, ms=6, zorder=4, label="Pareto frontier")
    if front:
        bestp = min(rec, key=lambda e: e["ratio"])
        ax.annotate(f"  {bestp['name']} ({bestp.get('family')})\n  ratio {bestp['ratio']:.4f}",
                    (bestp["ratio"], bestp["recovery_fraction"] * 100), fontsize=8, color="#cfd8e3")
    ax.axhline(budget_pct, ls="--", c="#7d3a3a", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("compression ratio (log — lower = more compressed)")
    ax.set_ylabel("recovery cost (% of from-scratch training)")
    ax.set_title("LLM auto-research: evolved compression / recovery frontier")
    ax.legend(fontsize=8, ncol=2); ax.grid(True, which="both", alpha=.15)
    fig.tight_layout(); fig.savefig(FIG_PARETO, dpi=130); plt.close(fig)

    # --- progress: best recovered ratio vs proposal number ---
    seeds = [e for e in rec if e.get("gen") == 0]
    ev = sorted([e for e in rec if e.get("gen") == 1], key=lambda e: _idx(e["name"]))
    seed_best = min((e["ratio"] for e in seeds), default=1.0)
    xs, ys, best = [], [], seed_best
    for e in ev:
        best = min(best, e["ratio"]); xs.append(_idx(e["name"])); ys.append(best)
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.axhline(seed_best, ls="--", c="#888", lw=1.5, label=f"best hand seed (group-NF2) {seed_best:.4f}")
    if xs:
        ax.step(xs, ys, where="post", color="#e0529c", lw=2.2, label="best evolved (recovers)")
        ax.scatter([_idx(e["name"]) for e in ev], [e["ratio"] for e in ev], s=12,
                   c=[FAM_COLOR.get(e.get("family"), "#ccc") for e in ev], alpha=.6, zorder=2)
    ax.set_xlabel("proposal number"); ax.set_ylabel("compression ratio (lower = better)")
    ax.set_title("Search progress — best recovering scheme found")
    ax.legend(fontsize=9); ax.grid(alpha=.15)
    fig.tight_layout(); fig.savefig(FIG_PROG, dpi=130); plt.close(fig)
    return front


def build_html(entries, front, budget_pct):
    now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    rec = [e for e in entries if e.get("recovered")]
    n = len(entries); ncand = len([e for e in entries if e.get("gen") == 1])
    inv = len([e for e in entries if not e.get("valid", True)])
    best = min(rec, key=lambda e: e["ratio"])
    seed_best = min((e["ratio"] for e in rec if e.get("gen") == 0), default=float("nan"))
    from collections import Counter
    fam_rec = Counter(e.get("family") for e in rec if e.get("gen") == 1)

    uri_p = _img_data_uri(FIG_PARETO); uri_g = _img_data_uri(FIG_PROG)
    frows = "".join(
        f"<tr class='rec'><td>{html.escape(e['name'])}</td><td>{html.escape(str(e.get('family')))}</td>"
        f"<td>{e['ratio']:.4f}</td><td>{1/e['ratio']:.0f}&times;</td>"
        f"<td>{e['recovery_fraction']*100:.2f}%</td>"
        f"<td>{html.escape(', '.join(e.get('parents') or []))}</td></tr>"
        for e in front)

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LLM auto-research — evolved compression schemes</title>
<style>{CSS}
pre{{background:#0b0e12;border:1px solid var(--line);border-radius:10px;padding:14px;overflow:auto;
font-size:12px;line-height:1.4}}</style></head><body><div class="wrap">

<h1>LLM auto-research: evolving compression schemes</h1>
<p class="sub">Specialized LLM proposers (scalar-quant, <b>vector-quant on Opus</b>, entropy, hybrid)
evolve per-tensor compression schemes for the 11M victim; each is sandbox-built, honest-byte-counted,
and scored by recovery cost. <a href="llm.html">&larr; LLM report</a> &middot; {now}</p>

<div class="kpi">
  <div class="b"><div class="n">{ncand}</div><div class="l">schemes proposed</div></div>
  <div class="b"><div class="n">{len(rec)}</div><div class="l">recovered</div></div>
  <div class="b"><div class="n">{best['ratio']:.4f}</div><div class="l">best ratio (&asymp;{1/best['ratio']:.0f}&times;)</div></div>
  <div class="b"><div class="n">{seed_best/best['ratio']:.2f}&times;</div><div class="l">smaller vs best hand seed</div></div>
</div>

<h2>What the search found</h2>
<div class="card"><ul>
<li>Best evolved scheme: <b>{html.escape(best['name'])}</b> ({html.escape(str(best.get('family')))})
at ratio <b>{best['ratio']:.4f}</b> (&approx;{1/best['ratio']:.0f}&times; smaller), recovering in
{best['recovery_fraction']*100:.1f}% of from-scratch training — beating the best hand-written seed
(group-NF2 @ {seed_best:.4f}).</li>
<li>The win came from the <b>vector-quant (Opus)</b> role: <b>additive multi-codebook VQ</b>
(AQLM / "Aggressive Compression Enables LLM Weight Theft" style) — encode each weight sub-vector as
a SUM of entries from several small learned codebooks — fused with group-wise quantized scales.</li>
<li>Recovered schemes by specialization: {', '.join(f'{k} {v}' for k, v in fam_rec.most_common())}.
Diversity held across all roles; the low-ratio frontier is dominated by VQ.</li>
<li>{ncand} proposed, {inv} invalid (build timeout / screen), {len(rec)} recovered within the
{budget_pct:.0f}% budget.</li>
</ul></div>

<h2>Evolved Pareto frontier</h2>
<div class="card">{('<img src="'+uri_p+'">') if uri_p else '<p class="sub">pending</p>'}
<p class="sub">Coloured by proposer specialization; open red = did-not-recover; black line = frontier.
Lower-left is better.</p></div>

<h2>Search progress</h2>
<div class="card">{('<img src="'+uri_g+'">') if uri_g else '<p class="sub">pending</p>'}
<p class="sub">Running-minimum recovering ratio vs proposal number; dashed = best hand seed.</p></div>

<h2>Frontier schemes</h2>
<div class="card" style="overflow:auto"><table><thead><tr><th>name</th><th>role</th><th>ratio</th>
<th>&times;</th><th>recovery</th><th>parents</th></tr></thead><tbody>{frows}</tbody></table></div>

<h2>Winning scheme ({html.escape(best['name'])}, ratio {best['ratio']:.4f})</h2>
<div class="card"><pre>{html.escape(best['code'])}</pre></div>

<p class="foot">Bytes = len(zlib.compress(pickle(payload))) measured in a sandboxed subprocess.
Repo: <a href="https://github.com/jacobcd52/weight-compression-recovery">jacobcd52/weight-compression-recovery</a>.
Regenerated by <code>python -m src.llm.autoresearch_report</code>.</p>
</div></body></html>"""


def main():
    d = json.load(open(ARCHIVE))
    entries = d["entries"]; budget_pct = d.get("budget_fraction", 0.30) * 100
    front = make_figs(entries, budget_pct)
    os.makedirs("docs", exist_ok=True)
    with open("docs/llm_autoresearch.html", "w") as f:
        f.write(build_html(entries, front, budget_pct))
    rec = [e for e in entries if e.get("recovered")]
    print(f"wrote docs/llm_autoresearch.html ({len(entries)} entries, {len(front)} frontier, "
          f"best ratio {min(e['ratio'] for e in rec):.4f})")


if __name__ == "__main__":
    main()

"""Build the LLM-phase report page: docs/llm.html (baseline curve, paper comparison, and the
compression/recovery frontier once it's available). Self-contained, GitHub-Pages-friendly:
    https://jacobcd52.github.io/weight-compression-recovery/llm.html

    python -m src.llm.report
"""
import base64
import datetime
import html
import json
import math
import os

from ..report import CSS, _img_data_uri

VICTIM_DIR = "runs/llm_victim"
FRONTIER = "runs/llm_seedfrontier/frontier.json"
FIG_BASE = "figures/llm_baseline_curve.png"
FIG_FRONTIER = "figures/llm_seed_frontier.png"

# measured: released SimpleStories-11M eval loss on OUR held-out test set (same tokenizer/metric)
RELEASED_LOSS = 1.7786
# from the suite's training config (35M_config.yaml) + paper Sec 3.5
PAPER_TOKENS = 60000 * 32768   # num_iterations * total_batch_size = ~1.97B
PAPER_LR = 1e-4


def _pareto(recovered):
    front = []
    for a in recovered:
        if not any(b is not a and b["ratio"] <= a["ratio"]
                   and b["recovery_fraction"] <= a["recovery_fraction"]
                   and (b["ratio"] < a["ratio"] or b["recovery_fraction"] < a["recovery_fraction"])
                   for b in recovered):
            front.append(a)
    return sorted(front, key=lambda e: e["ratio"])


def make_frontier_fig(fr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs("figures", exist_ok=True)
    res = fr["results"]
    rec = [r for r in res if r.get("recovered")]
    dnr = [r for r in res if r.get("valid", True) and not r.get("recovered")]
    front = _pareto(rec)
    budget_pct = fr["budget_fraction"] * 100
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    if rec:
        ax.scatter([r["ratio"] for r in rec], [r["recovery_fraction"] * 100 for r in rec],
                   s=70, c="#5ab1ef", edgecolor="k", linewidth=.5, zorder=3, label="recovers")
    if dnr:
        ax.scatter([r["ratio"] for r in dnr], [budget_pct] * len(dnr),
                   s=70, facecolors="none", edgecolors="#c0506a", linewidth=1.4, zorder=3,
                   label="DNR (did not recover in budget)")
    if front:
        ax.plot([r["ratio"] for r in front], [r["recovery_fraction"] * 100 for r in front],
                "-o", color="k", lw=2, ms=6, zorder=4, label="Pareto frontier")
    for r in res:
        if r.get("valid", True):
            y = r["recovery_fraction"] * 100 if r.get("recovered") else budget_pct
            ax.annotate(r["name"].replace("seed_", "").replace("resnet_best_", "RN:"),
                        (r["ratio"], y), textcoords="offset points", xytext=(5, 4),
                        fontsize=7, color="#cfd8e3")
    ax.axhline(budget_pct, ls="--", c="#7d3a3a", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("compression ratio (compressed bytes / fp32, log — lower = more compressed)")
    ax.set_ylabel("recovery cost (% of from-scratch training)")
    ax.set_title("11M SimpleStories victim — compression / recovery frontier (hand-written seeds)")
    ax.legend(loc="upper left", fontsize=9); ax.grid(True, which="both", alpha=.15)
    fig.tight_layout(); fig.savefig(FIG_FRONTIER, dpi=130); plt.close(fig)
    return front


def build_html(vic, fr, front):
    now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    our_tokens = vic["tokens"]; our_loss = vic["target_loss"]
    uri_base = _img_data_uri(FIG_BASE)
    uri_fr = _img_data_uri(FIG_FRONTIER) if os.path.exists(FIG_FRONTIER) else None

    # frontier table
    ftable = ""
    if fr:
        rows = []
        for r in sorted(fr["results"], key=lambda r: r.get("ratio", 1.0)):
            if not r.get("valid", True):
                continue
            rec = r.get("recovered")
            pill = ('<span class="pill rec">REC</span>' if rec else '<span class="pill dnr">DNR</span>')
            cost = f"{r['recovery_fraction']*100:.2f}%" if rec else "&mdash;"
            on_front = any(f["name"] == r["name"] for f in front)
            rows.append(
                f"<tr class='{'rec' if rec else 'dnr'}'><td>{html.escape(r['name'])}"
                f"{' &#9733;' if on_front else ''}</td>"
                f"<td>{r['ratio']:.4f}</td><td>{1/r['ratio']:.0f}&times;</td>"
                f"<td>{r.get('step0_loss','')}</td><td>{pill}</td><td>{cost}</td>"
                f"<td>{html.escape(str(r.get('best_lr') or ''))}</td></tr>")
        ftable = "".join(rows)

    frontier_section = f"""
<h2>Compression / recovery frontier (hand-written seeds)</h2>
<div class="card">{('<img alt="frontier" src="'+uri_fr+'">') if uri_fr else
'<p class="sub">frontier figure pending — the seed sweep is still running.</p>'}
<p class="sub">Each point is a compression scheme applied to the victim's weights, then retrained
on SimpleStories. x = compression ratio (log, left = more compressed); y = % of the victim's
from-scratch training cost needed to get back to its loss. Open red markers at the top = did not
recover within the {fr['budget_fraction']*100:.0f}% budget. &#9733; = on the Pareto frontier.</p>
</div>
<div class="card" style="overflow:auto"><table><thead><tr><th>scheme</th><th>ratio</th>
<th>&times;smaller</th><th>recon loss</th><th>result</th><th>recovery cost</th><th>best LR</th>
</tr></thead><tbody>{ftable}</tbody></table></div>""" if fr else """
<h2>Compression / recovery frontier</h2>
<div class="card"><p class="sub">The seed sweep is still running — this section will populate
when it finishes.</p></div>"""

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Weight-compression recovery — 11M SimpleStories LLM</title>
<style>{CSS}</style></head><body><div class="wrap">

<h1>Scaling up: 11M SimpleStories LLM</h1>
<p class="sub">Same threat model as the <a href="index.html">ResNet study</a>, now on a real
(small) language model. We compress a trained LLM's weights, then ask how cheaply an attacker with
the training data can retrain back to its performance.<br>
Llama-arch 11M (6 layers, d=384, GQA, vocab 4096, ctx 512) &middot; {now}</p>

<div class="kpi">
  <div class="b"><div class="n">{our_loss:.3f}</div><div class="l">victim eval loss (target)</div></div>
  <div class="b"><div class="n">{math.exp(our_loss):.2f}</div><div class="l">victim perplexity</div></div>
  <div class="b"><div class="n">{our_tokens/1e6:.0f}M</div><div class="l">training tokens (single-pass)</div></div>
  <div class="b"><div class="n">{vic['best_lr']:.0e}</div><div class="l">best LR (swept)</div></div>
</div>

<h2>The victim model</h2>
<div class="card"><ul>
<li>We train our <b>own</b> 11M model from random init, <b>single-pass</b> over {our_tokens/1e6:.0f}M
tokens of SimpleStories, sweeping the learning rate ({', '.join(f'{l:.0e}' for l in vic['lrs'])})
and keeping the best ({vic['best_lr']:.0e}). This is the "victim" whose weights get stolen.</li>
<li>We deliberately keep it <b>undertrained</b> — single-pass, loss still descending — because that
is the realistic regime for token-limited frontier models (no multi-epoch memorisation).</li>
<li><b>Target</b> for recovery = this victim's eval loss ({our_loss:.4f}); <b>denominator</b> =
its from-scratch step count ({vic['steps']} steps). Recovery cost is reported as a % of that.</li>
<li>We compress all {vic['n_compressible']} 2D weight matrices ({vic['fp32_bytes']/1e6:.0f} MB fp32:
attention + MLP + tied embedding); RMSNorm scales are tiny and kept dense.</li>
</ul></div>

<h2>Baseline training curve (LR sweep)</h2>
<div class="card">{('<img alt="baseline curve" src="'+uri_base+'">') if uri_base else
'<p class="sub">baseline curve pending.</p>'}
<p class="sub">Eval loss vs. step for each LR; the lowest (1e-3) becomes the victim. The curve is
still sloping down at the end — the model is undertrained by construction.</p></div>

<h2>How does this compare to the published model?</h2>
<div class="card">
<p>We evaluated the <b>released</b> SimpleStories-11M on our exact held-out set (same tokenizer and
metric), and read the suite's training recipe from the paper (arXiv:2504.09184, Sec 3.5) and its
training config.</p>
<table><thead><tr><th></th><th>Our victim</th><th>Released SimpleStories-11M</th></tr></thead>
<tbody>
<tr><td>training tokens</td><td>{our_tokens/1e6:.0f}M (single-pass)</td>
<td>~{PAPER_TOKENS/1e9:.1f}B (~{PAPER_TOKENS/316e6:.0f} epochs)</td></tr>
<tr><td>learning rate</td><td>{vic['best_lr']:.0e} (swept)</td><td>{PAPER_LR:.0e} (fixed)</td></tr>
<tr><td>eval loss (our test set)</td><td>{our_loss:.3f}</td><td>{RELEASED_LOSS:.3f}</td></tr>
<tr><td>perplexity</td><td>{math.exp(our_loss):.2f}</td><td>{math.exp(RELEASED_LOSS):.2f}</td></tr>
</tbody></table>
<p class="sub">Our victim sees <b>~20&times; fewer tokens</b> (single-pass vs ~6 epochs) and lands
<b>~{100*(math.exp(our_loss)/math.exp(RELEASED_LOSS)-1):.0f}% higher perplexity</b> — undertrained,
but a coherent story model. That gap is the point: a token-limited, not-fully-converged model is
the realistic target for weight theft.</p>
</div>

{frontier_section}

<p class="foot">Repo: <a href="https://github.com/jacobcd52/weight-compression-recovery">
jacobcd52/weight-compression-recovery</a>. Regenerated by <code>python -m src.llm.report</code>.
Released-model loss measured on our held-out SimpleStories test split; paper token count from the
35M training config (60000 iters &times; 32768 tokens/step).</p>
</div></body></html>"""


def main():
    vic = json.load(open(os.path.join(VICTIM_DIR, "summary.json")))
    fr = json.load(open(FRONTIER)) if os.path.exists(FRONTIER) else None
    front = make_frontier_fig(fr) if fr else []
    os.makedirs("docs", exist_ok=True)
    with open("docs/llm.html", "w") as f:
        f.write(build_html(vic, fr, front))
    print(f"wrote docs/llm.html (victim loss {vic['target_loss']}, "
          f"frontier={'yes' if fr else 'pending'})")


if __name__ == "__main__":
    main()

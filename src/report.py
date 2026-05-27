"""Generate a self-contained, clickable HTML report at docs/index.html.

Embeds the Pareto figure as a base64 data URI (single-file page), renders the
key findings computed from the run summaries, and a full results table. Designed
to be served via GitHub Pages (repo Settings -> Pages, source = /docs on main),
giving a tap-friendly URL: https://jacobcd52.github.io/weight-compression-recovery/
"""
import base64
import datetime
import glob
import html
import json
import os

from .plot import load_summaries, pareto_front


def _img_data_uri(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _findings(df):
    """Compute plain-language headline findings from the results dataframe."""
    out = []
    n = len(df)
    rec = df[df["recovered"]]
    nrec = len(rec)
    bloss = df["baseline_test_loss"].iloc[0] if "baseline_test_loss" in df else float("nan")
    out.append(f"<b>{nrec} of {n}</b> (technique, knob) runs recovered — i.e. retrained to a "
               f"full-test loss below the baseline's {bloss:.3f} within the 10% budget; the rest "
               f"are <b>DNR</b> (did not recover in budget). Recovery metric = test loss.")
    if nrec:
        # most-compressed recovery
        mc = rec.sort_values("compression_ratio").iloc[0]
        out.append(f"Most aggressive compression that still recovers: "
                   f"<b>{html.escape(mc['run_name'])}</b> at compression ratio "
                   f"<b>{mc['compression_ratio']:.3f}</b> "
                   f"(≈{1/mc['compression_ratio']:.0f}× smaller), "
                   f"recovered using {mc['recovery_fraction']*100:.1f}% of baseline training cost.")
        # cheapest recovery
        ch = rec.sort_values("recovery_fraction").iloc[0]
        out.append(f"Cheapest recovery: <b>{html.escape(ch['run_name'])}</b> bounced back in "
                   f"only <b>{ch['recovery_fraction']*100:.1f}%</b> of the original training cost "
                   f"(near-lossless reconstructed init).")
        techs = sorted(rec["technique"].unique())
        out.append("Techniques with at least one recovery: <b>" +
                   ", ".join(html.escape(t) for t in techs) + "</b>.")
    dnr_techs = sorted(set(df["technique"]) - set(rec["technique"]))
    if dnr_techs:
        out.append("Techniques with no recovery in budget: <b>" +
                   ", ".join(html.escape(t) for t in dnr_techs) + "</b>.")
    out.append("The <b>amortized</b> plot below re-counts the compression ratio without the "
               "size-independent codebook/scale overhead (a large fixed cost on a 270k-param "
               "model) — approximating the ratio at large model size, which especially helps "
               "the vector-quantization methods (k-means, additive_vq, AQLM).")
    return out


CSS = """
:root{--bg:#0f1419;--card:#1a2029;--fg:#e6e6e6;--muted:#9aa6b2;--accent:#5ab1ef;
--good:#2ea043;--goodbg:#13351d;--bad:#7d3a3a;--badbg:#2a1717;--line:#2a3340;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.55 -apple-system,
BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;}
.wrap{max-width:960px;margin:0 auto;padding:24px 18px 80px;}
h1{font-size:26px;margin:.2em 0 .1em} h2{font-size:20px;margin:1.6em 0 .5em;
border-bottom:1px solid var(--line);padding-bottom:.3em}
.sub{color:var(--muted);margin:0 0 1.2em}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;
padding:16px 18px;margin:14px 0;}
ul{margin:.4em 0;padding-left:1.2em} li{margin:.45em 0}
a{color:var(--accent)} code{background:#0b0e12;padding:1px 5px;border-radius:5px;
font-size:.92em}
img{width:100%;height:auto;border-radius:10px;background:#fff;padding:6px}
table{border-collapse:collapse;width:100%;font-size:14px;margin-top:.5em}
th,td{padding:7px 10px;text-align:right;border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left}
th{color:var(--muted);font-weight:600;position:sticky;top:0;background:var(--card)}
tr.rec td{background:var(--goodbg)} tr.dnr td{color:var(--muted)}
.pill{display:inline-block;padding:1px 8px;border-radius:99px;font-size:12px;font-weight:700}
.pill.rec{background:var(--good);color:#fff} .pill.dnr{background:var(--bad);color:#fff}
.kpi{display:flex;gap:14px;flex-wrap:wrap} .kpi .b{flex:1;min-width:130px;text-align:center;
background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px}
.kpi .n{font-size:24px;font-weight:700} .kpi .l{color:var(--muted);font-size:13px}
.foot{color:var(--muted);font-size:13px;margin-top:30px}
"""


def build_html(df, fig_uri, complete, fig_uri_amortized=None, n_expected=29,
               fig_uri_baseline=None, fig_uri_retrain=None, fig_uri_diag=None):
    now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    baseline = df["baseline_test_acc"].iloc[0]
    baseline_loss = df["baseline_test_loss"].iloc[0] if "baseline_test_loss" in df else float("nan")
    n = len(df); nrec = int(df["recovered"].sum())
    status = "Final" if complete else f"In progress ({n}/{n_expected} runs)"

    findings = "".join(f"<li>{x}</li>" for x in _findings(df))

    # results table sorted by technique then ratio
    df_sorted = df.sort_values(["technique", "compression_ratio"])
    rows = []
    for _, r in df_sorted.iterrows():
        cls = "rec" if r["recovered"] else "dnr"
        pill = ('<span class="pill rec">REC</span>' if r["recovered"]
                else '<span class="pill dnr">DNR</span>')
        recfrac = (f"{r['recovery_fraction']*100:.1f}%" if r["recovered"] else "&mdash;")
        mode = "distill" if r.get("did_distill") else "plain"
        rows.append(
            f"<tr class='{cls}'><td>{html.escape(str(r['technique']))}</td>"
            f"<td>{html.escape(str(r['knob']))}</td><td>{mode}</td>"
            f"<td>{r['compression_ratio']:.4f}</td>"
            f"<td>{1/r['compression_ratio']:.1f}&times;</td>"
            f"<td>{pill}</td><td>{recfrac}</td>"
            f"<td>{r['final_test_acc']:.2f}</td></tr>")
    table = "".join(rows)

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Weight-Compression Recovery — Pareto frontier</title>
<style>{CSS}</style></head><body><div class="wrap">

<h1>Weight-compression recovery</h1>
<p class="sub">Threat model: if an attacker steals a few compressed bytes of a trained model
and has the training data, how cheaply can they retrain back to full accuracy?<br>
ResNet-20 / CIFAR-10 (GroupNorm, best-LR baseline) &middot; {status} &middot; generated {now}</p>

<div class="kpi">
  <div class="b"><div class="n">{baseline_loss:.3f}</div><div class="l">target loss (baseline @ epoch 30)</div></div>
  <div class="b"><div class="n">{baseline:.1f}%</div><div class="l">baseline test acc</div></div>
  <div class="b"><div class="n">{nrec}/{n}</div><div class="l">runs recovered</div></div>
  <div class="b"><div class="n">5%</div><div class="l">retrain budget cap</div></div>
</div>

<div class="card" style="border-color:var(--accent)">
<b>&#129518; Auto-research.</b> An LLM evolved its own compression schemes against this same
recovery metric and beat the hand-tuned baseline — pushing the recovering frontier to
<b>≈19× compression</b> (ratio 0.053) for ~2% retraining, by rediscovering group-wise NormalFloat
quantization. <a href="autoresearch.html">See the auto-research report &rarr;</a>
</div>
<div class="card" style="border-color:var(--accent)">
<b>&#128218; Scaling up to a real LLM.</b> The same threat model on an 11M-param SimpleStories
language model (undertrained, single-pass). Victim training, paper comparison, and the
compression/recovery frontier. <a href="llm.html">See the LLM report &rarr;</a>
</div>

<h2>What this measures</h2>
<div class="card"><ul>
<li><b>Compression ratio</b> = compressed bytes / fp32 bytes of the conv+linear weights
(honest byte counts: bitmask/index overhead, codebooks, U+V, etc.). Lower = more compressed.
BatchNorm/biases are kept dense and excluded from both sides.</li>
<li><b>Recovery</b> = retraining (dense-from-init, plain CE; best of a small LR sweep) until the
full-test loss drops to the loss a <b>30-epoch baseline</b> reaches. Using the epoch-30 loss (not
the fully-converged loss) avoids the knife-edge of beating a fully-trained model.</li>
<li><b>Recovery cost</b> = retraining steps / <b>30-epoch</b> steps, capped at <b>5%</b>; a run
that doesn't reach the bar in that budget is <b>DNR</b>. 0% = the reconstruction already matches
a 30-epoch model with no retraining.</li>
<li><b>Amortized ratio</b> (2nd plot) removes the size-independent codebook/scale overhead to
approximate the ratio at large model size.</li>
</ul></div>

<h2>Headline findings</h2>
<div class="card"><ul>{findings}</ul></div>

<h2>Pareto frontier — honest bytes</h2>
<div class="card">{('<img alt="Pareto frontier" src="'+fig_uri+'">') if fig_uri else
'<p class="sub">figure pending — will appear when plotting runs.</p>'}
<p class="sub">x = compression ratio (log, left = more compressed); y = fraction of original
training cost needed to recover (log, lower = cheaper). Open markers at the top = DNR. Circles =
plain, triangles = distillation. Black line = Pareto frontier.</p></div>

<h2>Pareto frontier — amortized (large-model proxy)</h2>
<div class="card">{('<img alt="Amortized Pareto frontier" src="'+fig_uri_amortized+'">')
if fig_uri_amortized else '<p class="sub">amortized figure pending.</p>'}
<p class="sub">Same as above but the x-axis counts only per-weight bytes (codes/values/indices),
dropping the size-independent codebook and scale overhead that dominates on a tiny 270k-param
model. This is the ratio the vector-quantization methods would approach at large model size.</p></div>

<h2>Recovery dynamics vs. LR / schedule (diagnostic)</h2>
<div class="card">{('<img alt="LR diagnostic" src="'+fig_uri_diag+'">')
if fig_uri_diag else '<p class="sub">LR diagnostic figure pending.</p>'}
<p class="sub">Why the headline recovery cost (~7%) is partly an LR artifact: the near-lossless
init (quantize_8, left) recovers in <b>~0.2%</b> at a gentle constant LR (3e-5) but ~6.7% under
the high-LR cosine — a ~33x inflation. The harder init (kmeans_2, right) instead needs the cosine
anneal-to-zero to dip below baseline. So recovery cost should be measured as the minimum over a
small LR/schedule grid.</p></div>

<h2>Retraining loss curves</h2>
<div class="card">{('<img alt="Retraining curves" src="'+fig_uri_retrain+'">')
if fig_uri_retrain else '<p class="sub">retraining curves pending.</p>'}
<p class="sub">Full-test loss vs. retraining compute for a curated set of runs. The dashed line
is the baseline loss to drop below. Note the early high-LR bump and the late dip — recovery
clusters near the end of the cosine decay (~7%), which is a schedule artifact rather than the
true minimum compute to recover.</p></div>

<h2>Baseline training curve</h2>
<div class="card">{('<img alt="Baseline curve" src="'+fig_uri_baseline+'">')
if fig_uri_baseline else '<p class="sub">baseline curve pending.</p>'}
<p class="sub">The baseline is deliberately undertrained (50 epochs) so it is not overfit;
its test loss is the recovery target above.</p></div>

<h2>All runs</h2>
<div class="card" style="overflow:auto;max-height:640px">
<table><thead><tr><th>technique</th><th>knob</th><th>mode</th><th>ratio</th><th>x smaller</th>
<th>result</th><th>recovery cost</th><th>final acc %</th></tr></thead>
<tbody>{table}</tbody></table></div>

<p class="foot">Repo: <a href="https://github.com/jacobcd52/weight-compression-recovery">
jacobcd52/weight-compression-recovery</a> &middot; full spec in BRIEF.md, narrative in RESULTS.md.
This page is regenerated by <code>python -m src.report</code>.</p>
</div></body></html>"""


def main():
    df = load_summaries()
    if df.empty:
        print("no summaries yet")
        return
    os.makedirs("docs", exist_ok=True)
    fig_uri = _img_data_uri("figures/pareto.png")
    fig_uri_amortized = _img_data_uri("figures/pareto_amortized.png")
    fig_uri_baseline = _img_data_uri("figures/baseline_curve.png")
    fig_uri_retrain = _img_data_uri("figures/retrain_curves.png")
    fig_uri_diag = _img_data_uri("figures/diag_lr_curves.png")
    n_expected = 29
    complete = len(df) >= n_expected
    htmltext = build_html(df, fig_uri, complete, fig_uri_amortized, n_expected,
                          fig_uri_baseline, fig_uri_retrain, fig_uri_diag)
    with open("docs/index.html", "w") as f:
        f.write(htmltext)
    print(f"wrote docs/index.html ({len(df)} runs, complete={complete}, "
          f"figures={'2' if fig_uri and fig_uri_amortized else 'partial'})")


if __name__ == "__main__":
    main()

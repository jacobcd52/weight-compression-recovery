"""Stage-2 PV-tuning perplexity trajectories for every config we ran.
x = optimizer step (eval_every_steps=5), y = WikiText2 ppl (log).
Parses the AQLM finetune.py log lines "wikitext2 perplexity: X" between section markers.
Output: figures/stage2_curves.png
"""
import os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def evals_between(log_path, start_pat=None, end_pat=None):
    """Return ordered list of wikitext2 ppl values between markers (or whole log if None)."""
    txt = open(log_path).read().replace("\r", "\n")
    if start_pat:
        m = re.search(start_pat, txt); txt = txt[m.end():] if m else ""
    if end_pat:
        m = re.search(end_pat, txt); txt = txt[:m.start()] if m else txt
    return [float(x) for x in re.findall(r"wikitext2 perplexity:\s*([0-9.]+)", txt)]


runs = [
    ("SmolLM2-360M 2 bpp", "/workspace/autopilot/jobs/aqlm_ft3/log",  None, None, "#9467bd", "--"),
    ("SmolLM2-360M 1 bpp", "/workspace/autopilot/jobs/aqlm_pipe/log", r"## 1 bpp ##", r"## 0\.5 bpp ##", "#c44ad0", "--"),
    ("SmolLM2-360M 0.5 bpp","/workspace/autopilot/jobs/aqlm_pipe/log", r"## 0\.5 bpp ##", None,                 "#e377c2", "--"),
    ("TinyLlama-1.1B 2 bpp","/workspace/autopilot/jobs/aqlm_tiny/log", r"## 2 bpp ##", r"## 1 bpp ##",  "#1f77b4", "-"),
    ("TinyLlama-1.1B 1 bpp","/workspace/autopilot/jobs/aqlm_tiny/log", r"## 1 bpp ##", r"## 0\.5 bpp ##", "#17becf", "-"),
    ("TinyLlama-1.1B 0.5 bpp","/workspace/autopilot/jobs/aqlm_tiny/log", r"## 0\.5 bpp ##", None,         "#2ca02c", "-"),
]

fig, ax = plt.subplots(figsize=(10, 6.5))
EVAL_EVERY = 5
for label, path, start, end, color, ls in runs:
    ys = evals_between(path, start, end)
    if not ys:
        print(f"WARN no evals for {label}"); continue
    xs = [EVAL_EVERY * (i + 1) for i in range(len(ys))]
    ax.plot(xs, ys, marker="o", ms=4.5, lw=1.6, linestyle=ls, color=color,
            label=f"{label}  (start={ys[0]:.1f} → best={min(ys):.1f})")

# baselines as dotted reference lines
ax.axhline(11.47, ls=":", color="#9467bd", lw=1.0, alpha=.6); ax.text(2, 11.47, "SmolLM2 base 11.5", fontsize=8, color="#9467bd", va="bottom")
ax.axhline(7.78,  ls=":", color="#1f77b4", lw=1.0, alpha=.6); ax.text(2, 7.78, "TinyLlama base 7.78", fontsize=8, color="#1f77b4", va="bottom")

ax.set_yscale("log"); ax.set_xlabel("optimizer step (stage-2 PV-tuning)")
ax.set_ylabel("WikiText2 perplexity (log)")
ax.set_title("AQLM stage-2 trajectories — WikiText2 ppl vs step\n"
             "(SmolLM2-360M dashed, TinyLlama-1.1B solid; eval every 5 steps; 1 epoch)")
ax.legend(fontsize=8.5, loc="upper right"); ax.grid(True, which="both", alpha=.18)
fig.tight_layout(); os.makedirs("figures", exist_ok=True)
fig.savefig("figures/stage2_curves.png", dpi=140); plt.close(fig)
print("wrote figures/stage2_curves.png")

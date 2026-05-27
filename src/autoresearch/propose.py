"""Proposer: ask an LLM for a new candidate compression program.

Backend is abstracted: `claude -p` (uses the pod's persisted Claude Code login — no API key)
or the Anthropic Messages API (if ANTHROPIC_API_KEY is set). The prompt gives the interface
spec, a few parent programs with their metrics, and asks for a mutation or recombination.
"""
import os
import re
import subprocess

from .candidate import CANDIDATE_SPEC

_SYS = ("You are optimizing neural-network WEIGHT COMPRESSION schemes. Output ONLY one Python "
        "code block — no prose, no tool use, no file access.")


def build_prompt(parents, target_loss, budget_pct, best_ratio=None):
    lines = [_SYS, "",
             "Goal: compress ResNet-20 conv/linear weights to the FEWEST bytes such that a dense "
             f"model reconstructed from them retrains (<= {budget_pct:.0f}% of a 30-epoch budget, "
             f"best of an LR sweep) to test loss <= {target_loss:.3f}. Bytes are measured as the "
             "zlib-compressed serialized payload, so make it small AND low-entropy.", ""]
    if best_ratio is not None:
        lines += [f"The best scheme that STILL RECOVERS has ratio {best_ratio:.4f}. BEAT IT: push to "
                  f"a LOWER ratio (more aggressive) while still recovering in budget. Schemes milder "
                  f"than {best_ratio:.3f} are not interesting.", ""]
    lines += ["INTERFACE:", CANDIDATE_SPEC, "",
             "Some existing schemes and how they scored (ratio = bytes/fp32; cost = retrain "
             "fraction, lower is better; DNR = did not recover in budget):"]
    for p in parents:
        cost = (f"{p['recovery_fraction']*100:.2f}%" if p.get("recovered") else "DNR")
        lines.append(f"\n# {p['name']}  (ratio={p['ratio']:.4f}, recovery_cost={cost})\n"
                     f"```python\n{p['code'].strip()}\n```")
    lines.append("\nWrite ONE NEW scheme that pushes the frontier — either improve one above or "
                 "recombine two (e.g. better quantization grid, group/vector quantization, "
                 "low-rank + sparse, entropy-friendly coding). Output ONLY the code block "
                 "(KNOBS, compress_tensor, reconstruct_tensor).")
    return "\n".join(lines)


def parse_code(text):
    # consume any fence language tag up to the newline (```python / ```py / ```python3 / bare ```)
    blocks = re.findall(r"```[ \t]*[A-Za-z0-9_+\-]*[ \t]*\r?\n(.*?)```", text, re.DOTALL)
    # prefer a block that actually defines the interface, else the largest block
    good = [b for b in blocks if "def compress_tensor" in b and "def reconstruct_tensor" in b]
    pool = good or blocks
    if pool:
        return max(pool, key=len).strip()
    # unterminated fence (e.g. response truncated at max_tokens): take text after the opening fence
    m = re.search(r"```[ \t]*[A-Za-z0-9_+\-]*[ \t]*\r?\n(.*)$", text, re.DOTALL)
    if m and "def compress_tensor" in m.group(1):
        return m.group(1).strip()
    return text.strip() if "def compress_tensor" in text and not text.lstrip().startswith("`") else None


def propose_claude_p(prompt, model="claude-sonnet-4-6", timeout=180):
    env = dict(os.environ, IS_SANDBOX="1")
    try:
        r = subprocess.run(["claude", "-p", "--dangerously-skip-permissions", "--model", model],
                           input=prompt, capture_output=True, text=True, timeout=timeout, env=env)
        if r.returncode != 0:
            return None, (r.stderr or "")[-400:]
        return parse_code(r.stdout), None
    except subprocess.TimeoutExpired:
        return None, f"claude -p timeout >{timeout}s"


def propose_api(prompt, model="claude-sonnet-4-6", timeout=120, max_tokens=8192):
    import anthropic
    client = anthropic.Anthropic()
    msg = client.messages.create(model=model, max_tokens=max_tokens,
                                 messages=[{"role": "user", "content": prompt}])
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    code = parse_code(text)
    if code is None and msg.stop_reason == "max_tokens":
        return None, "truncated at max_tokens (no parseable code)"
    return code, None


def propose(prompt, backend="claude_p", model="claude-sonnet-4-6"):
    if backend == "api":
        return propose_api(prompt, model)
    return propose_claude_p(prompt, model)

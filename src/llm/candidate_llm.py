"""LLM-side candidate builder. Same candidate interface and safety screen as the ResNet
auto-research (KNOBS / compress_tensor / reconstruct_tensor, honest zlib byte count), but the
sandbox is pure-numpy: it operates on an exported .npz of the victim's 2D weight matrices and
writes a reconstructed .npz (the init to retrain from). No torch in the worker.

The victim weights are exported once by `export_victim_weights`; each candidate's reconstruction
+ measured bytes are produced in a sandboxed subprocess with a timeout.
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np

from ..autoresearch.candidate import _payload_bytes, validate_code  # reuse screen + byte count


def export_victim_weights(victim_pt, out_npz):
    """Pull the compressible 2D weights out of a victim checkpoint into a flat fp32 .npz."""
    import torch
    from .model import build_config, compressible_names
    from transformers import LlamaForCausalLM
    ck = torch.load(victim_pt, map_location="cpu", weights_only=False)
    cfg = build_config()
    m = LlamaForCausalLM(cfg)
    m.load_state_dict(ck["state_dict"])
    names = compressible_names(m)
    params = dict(m.named_parameters())
    arrays = {n: params[n].detach().to(torch.float32).numpy().copy() for n in names}
    fp32_bytes = int(sum(a.size for a in arrays.values()) * 4)
    np.savez(out_npz, **arrays)
    return arrays, fp32_bytes


def _worker(code_path, weights_npz, out_npz):
    """Pure-numpy: reconstruct every weight via the candidate, measure total zlib bytes."""
    import math
    code = open(code_path).read()
    ns = {"np": np, "math": math, "__builtins__": __builtins__}
    exec(code, ns)
    knobs = ns.get("KNOBS", {})
    ctens, rtens = ns["compress_tensor"], ns["reconstruct_tensor"]
    data = np.load(weights_npz)
    recon, total = {}, 0
    for k in data.files:
        w = data[k].astype(np.float32)
        payload = ctens(w, **knobs)
        total += _payload_bytes(payload)
        rec = np.asarray(rtens(payload, tuple(w.shape)), dtype=np.float32)
        assert rec.shape == w.shape, f"shape {rec.shape} != {w.shape} for {k}"
        recon[k] = rec
    fp32_bytes = int(sum(data[k].size for k in data.files) * 4)
    np.savez(out_npz, __compressed_bytes__=np.int64(total),
             __baseline_bytes__=np.int64(fp32_bytes), **recon)


def run_candidate(code, weights_npz, timeout=180):
    """Build the reconstructed init in a sandboxed subprocess.
    Returns dict(ok, recon_path|None, compressed_bytes, baseline_bytes, ratio, error)."""
    ok, reason = validate_code(code)
    if not ok:
        return {"ok": False, "error": f"validate: {reason}"}
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code); code_path = f.name
    out_path = code_path + ".recon.npz"
    try:
        r = subprocess.run(
            [sys.executable, "-m", "src.llm.candidate_llm", "--worker",
             "--code", code_path, "--weights", weights_npz, "--out", out_path],
            cwd=os.getcwd(), capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0 or not os.path.exists(out_path):
            return {"ok": False, "error": (r.stderr or "")[-800:]}
        d = np.load(out_path)
        cb = int(d["__compressed_bytes__"]); bb = int(d["__baseline_bytes__"])
        return {"ok": True, "recon_path": out_path, "compressed_bytes": cb,
                "baseline_bytes": bb, "ratio": cb / bb}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"timeout >{timeout}s"}
    finally:
        if os.path.exists(code_path):
            os.remove(code_path)


def load_recon(recon_path):
    """Read a reconstructed .npz back into {name: fp32 array} (drops the bookkeeping keys)."""
    d = np.load(recon_path)
    return {k: d[k].astype(np.float32) for k in d.files if not k.startswith("__")}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--code"); ap.add_argument("--weights"); ap.add_argument("--out")
    a = ap.parse_args()
    if a.worker:
        _worker(a.code, a.weights, a.out)

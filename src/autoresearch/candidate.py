"""Candidate compression programs for the auto-research loop.

A *candidate* is LLM-written Python defining a per-tensor compression scheme:

    KNOBS = {...}                                  # default hyperparameters (picklable scalars)
    def compress_tensor(w, **knobs) -> payload     # w: float32 np.ndarray (a conv/linear weight)
    def reconstruct_tensor(payload, shape) -> np.ndarray   # dense fp32 reconstruction

The harness (NOT the candidate) owns everything else: it loops over the conv/linear weight
tensors of the baseline, calls compress/reconstruct, **measures honest bytes objectively** as
the zlib-compressed size of the serialized payload (so a candidate cannot fake its byte count),
reconstructs the dense init, and hands it to the vmap-ensemble for recovery scoring.

Integrity: candidate code is statically screened (no I/O, network, eval/exec, pickle/torch/np
load, no giant literals that could smuggle weights), the byte count is measured not self-reported,
and the build runs in a subprocess with a timeout so hangs/crashes can't stall the search.
"""
import argparse
import ast
import json
import os
import pickle
import re
import subprocess
import sys
import tempfile
import zlib

import numpy as np

# ----- the spec we hand the proposer -----------------------------------------
CANDIDATE_SPEC = '''\
Write a per-tensor weight-compression scheme as a single Python snippet using ONLY numpy (as np)
and the standard library (math). Define exactly:

    KNOBS = { ... }    # dict of default hyperparameters (small scalars only)
    def compress_tensor(w, **knobs):
        # w: a float32 numpy array (a conv or linear WEIGHT tensor, shape preserved by the caller)
        # return a small PICKLABLE payload (dict of numpy arrays / ints / floats) that stores the
        # compressed representation. Smaller + lower-entropy payload = better compression ratio
        # (bytes are measured as the zlib-compressed size of the serialized payload).
        ...
    def reconstruct_tensor(payload, shape):
        # return a float32 numpy array of `shape`, the dense reconstruction.
        ...

Rules: deterministic; no file/network/OS access, no eval/exec, no loading external data, no huge
constant arrays. The goal is to reconstruct a good-enough init that retrains to the target loss
in as little compute as possible, at the smallest possible byte size.'''

_BANNED = [r"\bimport\s+os\b", r"\bimport\s+sys\b", r"\bimport\s+subprocess\b",
           r"\bimport\s+socket\b", r"\bimport\s+requests\b", r"\burllib\b", r"\bopen\s*\(",
           r"\beval\s*\(", r"\bexec\s*\(", r"__import__", r"\bglobals\s*\(", r"\blocals\s*\(",
           r"torch\.load", r"np\.load", r"np\.fromfile", r"pickle\.load", r"\bgetattr\s*\("]


def validate_code(code, max_literal=4096):
    """Static screen. Returns (ok, reason)."""
    for pat in _BANNED:
        if re.search(pat, code):
            return False, f"banned pattern: {pat}"
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"syntax error: {e}"
    has_c = has_r = has_k = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "compress_tensor":
            has_c = True
        if isinstance(node, ast.FunctionDef) and node.name == "reconstruct_tensor":
            has_r = True
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", "") == "KNOBS" for t in node.targets):
            has_k = True
        # reject big constant lists/tuples (could smuggle weights into the program)
        if isinstance(node, (ast.List, ast.Tuple)) and len(node.elts) > max_literal:
            return False, "oversized literal"
    if not (has_c and has_r and has_k):
        return False, "must define KNOBS, compress_tensor, reconstruct_tensor"
    return True, "ok"


def _payload_bytes(payload):
    """Objective, hack-resistant byte count: zlib-compressed serialized payload."""
    return len(zlib.compress(pickle.dumps(payload, protocol=4), level=9))


# ----- subprocess worker: build the reconstructed init from a candidate ------
def _worker(code_path, baseline_path, out_path):
    import torch
    sys.path.insert(0, os.getcwd())
    from src.utils import compressible_keys, fp32_bytes_of_keys
    code = open(code_path).read()
    ns = {"np": np, "__builtins__": __builtins__}
    import math
    ns["math"] = math
    exec(code, ns)
    knobs = ns.get("KNOBS", {})
    ctens, rtens = ns["compress_tensor"], ns["reconstruct_tensor"]

    ckpt = torch.load(baseline_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    ckeys = compressible_keys(sd)
    total = 0
    init = {}
    for k, v in sd.items():
        if k in ckeys:
            w = v.detach().float().cpu().numpy()
            payload = ctens(w, **knobs)
            total += _payload_bytes(payload)
            rec = np.asarray(rtens(payload, tuple(w.shape)), dtype=np.float32)
            assert rec.shape == w.shape, f"shape {rec.shape} != {w.shape} for {k}"
            init[k] = torch.from_numpy(rec.copy())
        else:
            init[k] = v.detach().cpu().clone()
    baseline_bytes = fp32_bytes_of_keys(sd, ckeys)
    torch.save({"init": init, "compressed_bytes": int(total),
                "baseline_bytes": int(baseline_bytes)}, out_path)


def run_candidate(code, baseline_path, timeout=120):
    """Build the reconstructed init in a sandboxed subprocess.
    Returns dict(ok, init_path|None, compressed_bytes, baseline_bytes, ratio, error)."""
    ok, reason = validate_code(code)
    if not ok:
        return {"ok": False, "error": f"validate: {reason}"}
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code); code_path = f.name
    out_path = code_path + ".out.pt"
    try:
        r = subprocess.run(
            [sys.executable, "-m", "src.autoresearch.candidate", "--worker",
             "--code", code_path, "--baseline", baseline_path, "--out", out_path],
            cwd=os.getcwd(), capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0 or not os.path.exists(out_path):
            return {"ok": False, "error": (r.stderr or "")[-800:]}
        import torch
        blob = torch.load(out_path, map_location="cpu", weights_only=False)
        return {"ok": True, "init_path": out_path,
                "compressed_bytes": blob["compressed_bytes"],
                "baseline_bytes": blob["baseline_bytes"],
                "ratio": blob["compressed_bytes"] / blob["baseline_bytes"]}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"timeout >{timeout}s"}
    finally:
        if os.path.exists(code_path):
            os.remove(code_path)


# ----- a seed candidate (per-tensor 8-bit symmetric quant) for validation ----
SEED_QUANTIZE = '''
import numpy as np
KNOBS = {"bits": 8}
def compress_tensor(w, bits=8):
    amax = float(np.max(np.abs(w))) or 1.0
    qmax = (1 << (bits - 1)) - 1
    scale = amax / qmax
    q = np.clip(np.round(w / scale), -qmax, qmax).astype(np.int8)
    return {"q": q, "scale": np.float32(scale)}
def reconstruct_tensor(payload, shape):
    return (payload["q"].astype(np.float32) * payload["scale"]).reshape(shape)
'''


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--code"); ap.add_argument("--baseline"); ap.add_argument("--out")
    a = ap.parse_args()
    if a.worker:
        _worker(a.code, a.baseline, a.out)
    else:
        # self-test on the seed candidate
        r = run_candidate(SEED_QUANTIZE, "runs/baseline_gn/best.pt")
        print("seed quantize:", {k: v for k, v in r.items() if k != "init_path"})

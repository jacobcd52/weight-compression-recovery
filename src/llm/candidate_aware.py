"""Activation-aware candidate sandbox (the "attacker is inside the data center" model).

Unlike the numpy-only sandbox, candidates here MAY use torch + the GPU and are HANDED the per-layer
input Hessian H = E[xx^T] (GPTQ/AQLM statistic), so they can do data/activation-aware, compute-heavy
compression (GPU k-means, gradient codebook fitting, GPTQ, AQLM, ...). A per-candidate compute
budget (timeout) bounds GPU spend.

Integrity is still enforced so the experiment stays meaningful:
  * the exfiltrated PAYLOAD is the honest cost = len(zlib(pickle(payload)))  (torch tensors in the
    payload are normalised to numpy first so they can't hide bytes);
  * reconstruct_tensor(payload, shape) is a PURE function of the payload — it does NOT get w, H, or
    any data (else "compression ratio" would be meaningless);
  * file / network / os / eval / load are still banned (so a candidate can't read the eval set or
    smuggle weights via disk); torch + cuda ARE allowed.

Interface the candidate must define:
    KNOBS = {...}
    def compress_tensor(w, H=None, **knobs):   # w: float32 np.ndarray (out,in); H: np (in,in) or None
        return payload                          # picklable: numpy arrays / torch tensors / scalars
    def reconstruct_tensor(payload, shape):     # pure; returns float32 np.ndarray of `shape`
        ...
"""
import argparse
import ast
import inspect
import os
import pickle
import re
import subprocess
import sys
import tempfile
import zlib

import numpy as np

# torch is allowed; these stay banned for integrity (no data peeking / disk smuggling / escape)
_BANNED = [r"\bimport\s+os\b", r"\bimport\s+sys\b", r"\bimport\s+subprocess\b",
           r"\bimport\s+socket\b", r"\brequests\b", r"\burllib\b", r"\bopen\s*\(",
           r"\beval\s*\(", r"\bexec\s*\(", r"__import__", r"\bglobals\s*\(", r"\blocals\s*\(",
           r"torch\.load", r"torch\.save", r"np\.load", r"np\.fromfile", r"\bnp\.save",
           r"pickle\.load", r"\bgetattr\s*\(", r"os\.", r"sys\.", r"\.cpu_count\b"]


def validate_aware(code, max_literal=8192):
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
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "KNOBS" for t in node.targets):
            has_k = True
        if isinstance(node, (ast.List, ast.Tuple)) and len(node.elts) > max_literal:
            return False, "oversized literal"
    if not (has_c and has_r and has_k):
        return False, "must define KNOBS, compress_tensor, reconstruct_tensor"
    return True, "ok"


def _to_numpy(obj):
    """Normalise a payload to numpy/scalars so the zlib byte count is honest (no hidden torch bytes)."""
    import torch
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy()
    if isinstance(obj, dict):
        return {k: _to_numpy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_numpy(v) for v in obj)
    return obj


def _payload_bytes(payload):
    return len(zlib.compress(pickle.dumps(_to_numpy(payload), protocol=4), level=9))


def _call_compress(fn, w, H, knobs):
    """Pass H only if the candidate's compress_tensor accepts it (else stay backward-compatible)."""
    try:
        sig = inspect.signature(fn)
        accepts_H = "H" in sig.parameters or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    except (TypeError, ValueError):
        accepts_H = True
    return fn(w, H=H, **knobs) if accepts_H else fn(w, **knobs)


def _worker(code_path, weights_npz, hess_pt, out_npz):
    import math
    import torch
    code = open(code_path).read()
    ns = {"np": np, "math": math, "torch": torch, "__builtins__": __builtins__}
    exec(code, ns)
    knobs = ns.get("KNOBS", {})
    ctens, rtens = ns["compress_tensor"], ns["reconstruct_tensor"]
    data = np.load(weights_npz)
    H = torch.load(hess_pt, map_location="cpu", weights_only=False)["H"] if os.path.exists(hess_pt) else {}
    recon, total = {}, 0
    for k in data.files:
        w = data[k].astype(np.float32)
        Hk = H.get(k)
        Hk = Hk.numpy().astype(np.float32) if Hk is not None else None
        payload = _call_compress(ctens, w, Hk, knobs)
        total += _payload_bytes(payload)
        payload = _to_numpy(payload)                          # reconstruct sees only normalised bytes
        rec = np.asarray(rtens(payload, tuple(w.shape)), dtype=np.float32)
        assert rec.shape == w.shape, f"shape {rec.shape} != {w.shape} for {k}"
        recon[k] = rec
    fp32 = int(sum(data[k].size for k in data.files) * 4)
    np.savez(out_npz, __compressed_bytes__=np.int64(total), __baseline_bytes__=np.int64(fp32), **recon)


def run_candidate(code, weights_npz, hess_pt, timeout=300):
    ok, reason = validate_aware(code)
    if not ok:
        return {"ok": False, "error": f"validate: {reason}"}
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code); code_path = f.name
    out_path = code_path + ".recon.npz"
    try:
        r = subprocess.run(
            [sys.executable, "-m", "src.llm.candidate_aware", "--worker", "--code", code_path,
             "--weights", weights_npz, "--hess", hess_pt, "--out", out_path],
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
    d = np.load(recon_path)
    return {k: d[k].astype(np.float32) for k in d.files if not k.startswith("__")}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--code"); ap.add_argument("--weights"); ap.add_argument("--hess"); ap.add_argument("--out")
    a = ap.parse_args()
    if a.worker:
        _worker(a.code, a.weights, a.hess, a.out)

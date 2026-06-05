"""Per-linear-layer input Hessians H = E[xx^T] for the Pythia victim (GPT-NeoX), on Pile calibration
data. Accumulated on CPU because the full set is large (the MLP down-proj alone is 8192x8192 per
layer); we compute X^T X on GPU per batch and add into a CPU accumulator.

    python -m src.pythia.activations --revision step143000 --calib-batches 32
"""
import argparse
import os

import torch

from .data import Batcher, load_packed
from .model import load_model, linear_module_for_weight


def compute_hessians(revision, cache_dir, out_path, calib_batches=32, batch_size=8, damp=0.01,
                     device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(revision=revision, device=device, dtype=torch.bfloat16).eval()
    train, _, meta = load_packed(cache_dir)
    tb = Batcher(train, batch_size, device, seed=777)

    wmap = linear_module_for_weight(model)                       # wname -> module name or None
    linears = {n: m for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)}
    mod_to_wn = {m: wn for wn, m in wmap.items() if m is not None}
    H = {wn: None for wn, m in wmap.items() if m is not None}    # CPU accumulators
    counts = {wn: 0 for wn in H}
    handles = []

    def mk_hook(mname):
        def hook(module, inp):
            x = inp[0].detach()
            x = x.reshape(-1, x.shape[-1]).float()               # (tokens, in)
            wn = mod_to_wn[mname]
            g = (x.t() @ x).cpu()                                # accumulate on CPU
            H[wn] = g if H[wn] is None else H[wn] + g
            counts[wn] += x.shape[0]
        return hook

    for mname in mod_to_wn:
        handles.append(linears[mname].register_forward_pre_hook(mk_hook(mname)))
    with torch.no_grad():
        for _ in range(calib_batches):
            model(input_ids=tb.batch())
    for h in handles:
        h.remove()

    out = {}
    for wn, g in H.items():
        if g is None:
            continue
        Hn = g / max(counts[wn], 1)
        d = damp * torch.diag(Hn).mean().clamp_min(1e-8)
        out[wn] = (Hn + d * torch.eye(Hn.shape[0])).to(torch.float32)
    torch.save({"H": out, "revision": revision,
                "calib_tokens": int(sum(counts.values()) / max(len(counts), 1)), "damp": damp},
               out_path)
    tot = sum(v.numel() for v in out.values()) * 4 / 1e9
    print(f"[pythia-hess] {len(out)} layers, ~{tot:.1f} GB, "
          f"~{int(sum(counts.values())/max(len(counts),1))} calib tokens/layer -> {out_path}", flush=True)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--revision", default="step143000")
    ap.add_argument("--cache-dir", default="data/pile")
    ap.add_argument("--out", default="runs/pythia_victim/hessians.pt")
    ap.add_argument("--calib-batches", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--damp", type=float, default=0.01)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    compute_hessians(args.revision, args.cache_dir, args.out, args.calib_batches, args.batch_size,
                     args.damp)


if __name__ == "__main__":
    main()

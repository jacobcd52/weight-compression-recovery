"""GPTQ on the Pythia-1.4B victim, on GPU — the FULL-Hessian method: sequential column
quantization with inverse-Hessian error feedback (the standard GPTQ algorithm), group-wise
low-bit. This is the apples-to-apples upgrade from the diagonal-H AQLM, to test whether the full
Hessian closes the gap toward the paper's 1-2 bpp near-lossless. One-time, no retraining.

    python -m src.pythia.gptq --bits 3 --group-size 128
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from .data import load_packed, Batcher
from .model import load_model, set_weights, eval_loss

DEV = torch.device("cuda")


def gptq_layer(W, H, bits, group_size=128, percdamp=0.01, blocksize=128):
    """W (out,in) float; H (in,in) float (input second moment). Returns dequantized W (out,in) and
    the stored-bytes for this layer (codes + group scales/zeros)."""
    W = W.clone().float(); H = H.clone().float()
    out, inf = W.shape
    maxq = (1 << bits) - 1
    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0; W[:, dead] = 0.0
    H[range(inf), range(inf)] += percdamp * torch.diag(H).mean()
    # inverse-Hessian Cholesky (upper), the GPTQ error-feedback operator
    L = torch.linalg.cholesky(H)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(L), upper=True)
    Q = torch.zeros_like(W)
    scale = zero = None
    ngroups = 0
    for i1 in range(0, inf, blocksize):
        i2 = min(i1 + blocksize, inf); cnt = i2 - i1
        W1 = W[:, i1:i2].clone(); Q1 = torch.zeros_like(W1); Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(cnt):
            col = i1 + i
            if col % group_size == 0:
                grp = W[:, col:col + group_size]
                mn = grp.min(1).values; mx = grp.max(1).values
                scale = ((mx - mn).clamp_min(1e-8) / maxq)
                zero = torch.round(-mn / scale)
                ngroups += 1
            w = W1[:, i]; d = Hinv1[i, i]
            q = torch.clamp(torch.round(w / scale) + zero, 0, maxq)
            dq = scale * (q - zero)
            Q1[:, i] = dq
            err = (w - dq) / d
            W1[:, i:] -= err.unsqueeze(1) * Hinv1[i, i:].unsqueeze(0)
            Err1[:, i] = err
        Q[:, i1:i2] = Q1
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    code_bytes = int(np.ceil(bits * out * inf / 8))
    sz_bytes = 2 * 2 * out * ngroups               # scale+zero, fp16, per (out, group)
    return Q, code_bytes + sz_bytes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--victim-dir", default="runs/pythia_victim")
    ap.add_argument("--revision", default="step143000")
    ap.add_argument("--cache-dir", default="data/pile")
    ap.add_argument("--bits", type=int, default=3)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--eval-batches", type=int, default=20)
    args = ap.parse_args()

    data = np.load(os.path.join(args.victim_dir, "weights.npz"))
    H = torch.load(os.path.join(args.victim_dir, "hessians.pt"), map_location="cpu",
                   weights_only=False)["H"]
    _, ev, _ = load_packed(args.cache_dir)
    victim = load_model(args.revision, device=DEV, dtype=torch.float32)
    eb = Batcher(ev, 4, DEV, seed=1)
    vloss = eval_loss(victim, eb, DEV, max_batches=args.eval_batches)
    print(f"[gptq] victim {args.revision} loss={vloss:.4f} | {args.bits}-bit group={args.group_size} "
          f"FULL-H", flush=True)

    recon, cbytes, fp32 = {}, 0, 0
    t0 = time.time()
    for i, k in enumerate(data.files):
        W = torch.from_numpy(data[k].astype(np.float32)).to(DEV); fp32 += W.numel() * 4
        if k in H:
            Hk = H[k].to(DEV)
        else:                                       # embed_in: no activation H -> identity (plain quant)
            Hk = torch.eye(W.shape[1], device=DEV)
        Q, b = gptq_layer(W, Hk, args.bits, args.group_size)
        recon[k] = Q.cpu().numpy().astype(np.float32); cbytes += b
        del W, Hk, Q; torch.cuda.empty_cache()
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(data.files)} matrices, {time.time()-t0:.0f}s", flush=True)
    build_s = time.time() - t0
    set_weights(victim, recon)
    rloss = eval_loss(victim, eb, DEV, max_batches=args.eval_batches)
    ratio = cbytes / fp32
    print(f"[gptq] DONE {build_s:.0f}s | {args.bits}bit (ratio={ratio:.4f}, ~{ratio*32:.2f}bpp) | "
          f"recon_loss={rloss:.4f} (ppl {np.exp(rloss):.2f}) vs victim {vloss:.4f} "
          f"(ppl {np.exp(vloss):.2f})", flush=True)
    os.makedirs("runs/pythia_aqlm", exist_ok=True)
    json.dump({"method": "gptq", "bits": args.bits, "group_size": args.group_size, "ratio": ratio,
               "recon_loss": rloss, "victim_loss": vloss, "build_s": build_s},
              open(f"runs/pythia_aqlm/gptq_b{args.bits}_g{args.group_size}.json", "w"), indent=1)


if __name__ == "__main__":
    main()

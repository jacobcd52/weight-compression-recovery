"""Full AQLM on the Pythia-1.4B victim, on GPU (port of the ResNet src/compress.py _aqlm_payload
that hit near-lossless at 2-3 bpp without retraining): per-output scale, residual-VQ init,
activation-importance-WEIGHTED greedy encode + WEIGHTED-LEAST-SQUARES codebook update, alternating
for n_iters rounds. Importance = diag(H) (cached Hessians). One-time compression (no time limit);
measures no-retrain reconstruction loss at M bits/weight.

    python -m src.pythia.aqlm_full --M 2 --iters 4
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


def _assign(X, c, chunk=2_000_000):
    out = torch.empty(X.shape[0], dtype=torch.long, device=X.device)
    for i in range(0, X.shape[0], chunk):
        out[i:i + chunk] = torch.cdist(X[i:i + chunk], c).argmin(1)
    return out


def _kmeans(X, K, iters=10):
    c = X[torch.randperm(X.shape[0], device=X.device)[:K]].clone()
    for _ in range(iters):
        lab = _assign(X, c)
        c2 = torch.zeros_like(c); c2.index_add_(0, lab, X)
        cnt = torch.bincount(lab, minlength=K).clamp_min(1).unsqueeze(1).float()
        c = c2 / cnt
    return c


def _rvq_init(X, M, K):
    cbs, res = [], X.clone()
    for _ in range(M):
        c = _kmeans(res, K)
        cbs.append(c); res = res - c[_assign(res, c)]
    return cbs


def _wgreedy_encode(sub, cbs, w, chunk=2_000_000):
    """sub (n,d), w (n,d) per-row dim-weights. Returns codes (n,M)."""
    n, M = sub.shape[0], len(cbs)
    res = sub.clone()
    codes = torch.empty(n, M, dtype=torch.long, device=sub.device)
    for m, cb in enumerate(cbs):
        cb2 = cb * cb                                   # (K,d)
        a = torch.empty(n, dtype=torch.long, device=sub.device)
        for i in range(0, n, chunk):
            wr = w[i:i + chunk]; rr = res[i:i + chunk]
            d2 = -2.0 * (wr * rr) @ cb.t() + wr @ cb2.t()    # (chunk,K)
            a[i:i + chunk] = d2.argmin(1)
        codes[:, m] = a; res = res - cb[a]
    return codes


def _wls_codebooks(Wf, codes, w, M, K, d):
    """Weighted least-squares codebook update. Wf (n,d), codes (n,M), w (n,d)."""
    n = Wf.shape[0]; MK = M * K
    cols = codes + (torch.arange(M, device=Wf.device) * K)        # (n,M)
    cbs = [torch.zeros(K, d, device=Wf.device) for _ in range(M)]
    I = torch.eye(MK, device=Wf.device)
    for e in range(d):
        we = w[:, e]; xe = Wf[:, e]
        AtA = torch.zeros(MK, MK, device=Wf.device)
        Atx = torch.zeros(MK, device=Wf.device)
        for i in range(M):
            ci = cols[:, i]
            Atx.index_add_(0, ci, we * xe)
            for j in range(M):
                flat = ci * MK + cols[:, j]
                AtA.view(-1).index_add_(0, flat, we)
        # damp relative to the matrix scale so unused (zero-row) codebook entries stay invertible
        damp = 1e-3 * AtA.diagonal().mean().clamp_min(1e-8)
        B = torch.linalg.solve(AtA + damp * I, Atx)
        for m in range(M):
            cbs[m][:, e] = B[m * K:(m + 1) * K]
    return cbs


def aqlm_compress(W, imp, M, K, d, iters, Hfull=None, ft_steps=0, ft_lr=3e-3, reassign_every=0):
    """W (out,in) cpu numpy; imp (in,) diag-Hessian; Hfull (in,in) torch on DEV (or None).
    Diagonal weighted-LS+greedy init, then optional gradient codebook fine-tune on the FULL-H
    objective E=(W-Wh)H(W-Wh)^T (the AQLM step). Returns reconstructed (out,in) numpy."""
    Wt = torch.from_numpy(W).float().to(DEV)
    out, inf = Wt.shape
    s = Wt.std(1, keepdim=True).clamp_min(1e-8)                    # per-output scale
    Wn = Wt / s
    pad = (-inf) % d
    if pad:
        Wn = torch.cat([Wn, torch.zeros(out, pad, device=DEV)], 1)
        imp = np.concatenate([imp, np.zeros(pad, np.float32)])
    G = Wn.shape[1] // d
    Wb = Wn.reshape(out, G, d).reshape(out * G, d)                 # (n,d), n=out*G
    impt = torch.from_numpy(imp).float().to(DEV).reshape(G, d)
    impt = impt / (impt.mean() + 1e-12)
    w_rows = impt.repeat(out, 1)
    cbs = _rvq_init(Wb, M, K)
    codes = None
    for _ in range(iters):
        codes = _wgreedy_encode(Wb, cbs, w_rows)
        cbs = _wls_codebooks(Wb, codes, w_rows, M, K, d)
    codes = _wgreedy_encode(Wb, cbs, w_rows)

    if ft_steps > 0 and Hfull is not None:
        # AQLM codebook fine-tuning on the FULL-H output error (in scaled space H is unchanged).
        cb_p = [c.clone().requires_grad_(True) for c in cbs]
        opt = torch.optim.Adam(cb_p, lr=ft_lr)
        for st in range(ft_steps):
            if reassign_every and st > 0 and st % reassign_every == 0:
                with torch.no_grad():
                    codes = _wgreedy_encode(Wb, [c.detach() for c in cb_p], w_rows)
            sub = sum(cb_p[m][codes[:, m]] for m in range(M))      # (n,d)
            Wh = (sub.reshape(out, G * d)[:, :inf]) * s             # (out,in)
            err = Wt - Wh
            loss = ((err @ Hfull) * err).sum() / out
            opt.zero_grad(); loss.backward(); opt.step()
        cbs = [c.detach() for c in cb_p]
        codes = _wgreedy_encode(Wb, cbs, w_rows)

    rec = sum(cbs[m][codes[:, m]] for m in range(M))
    rec = rec.reshape(out, G * d)[:, :inf] * s
    return rec.cpu().numpy().astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--victim-dir", default="runs/pythia_victim")
    ap.add_argument("--revision", default="step143000")
    ap.add_argument("--cache-dir", default="data/pile")
    ap.add_argument("--M", type=int, default=2)
    ap.add_argument("--bits", type=int, default=8)            # K = 2^bits
    ap.add_argument("--dsub", type=int, default=8)
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--ft-steps", type=int, default=0, help="full-H gradient codebook fine-tune steps")
    ap.add_argument("--ft-lr", type=float, default=3e-3)
    ap.add_argument("--reassign-every", type=int, default=0)
    ap.add_argument("--eval-batches", type=int, default=20)
    args = ap.parse_args()
    K = 1 << args.bits
    bpp = args.M * args.bits / args.dsub

    data = np.load(os.path.join(args.victim_dir, "weights.npz"))
    H = torch.load(os.path.join(args.victim_dir, "hessians.pt"), map_location="cpu",
                   weights_only=False)["H"]
    _, ev, _ = load_packed(args.cache_dir);
    victim = load_model(args.revision, device=DEV, dtype=torch.float32)
    eb = Batcher(ev, 4, DEV, seed=1)
    vloss = eval_loss(victim, eb, DEV, max_batches=args.eval_batches)
    print(f"[aqlm-full] victim {args.revision} loss={vloss:.4f} | M={args.M} K={K} d={args.dsub} "
          f"iters={args.iters} -> {bpp:.2f} bits/weight (diag-H importance)", flush=True)

    recon, cbytes, fp32 = {}, 0, 0
    t0 = time.time()
    for i, k in enumerate(data.files):
        W = data[k].astype(np.float32); fp32 += W.size * 4
        imp = (torch.diag(H[k]).numpy().astype(np.float32) if k in H
               else np.ones(W.shape[1], np.float32))
        Hfull = H[k].to(DEV) if (k in H and args.ft_steps > 0) else None
        recon[k] = aqlm_compress(W, imp, args.M, K, args.dsub, args.iters,
                                 Hfull=Hfull, ft_steps=args.ft_steps, ft_lr=args.ft_lr,
                                 reassign_every=args.reassign_every)
        if Hfull is not None:
            del Hfull; torch.cuda.empty_cache()
        G = int(np.ceil(W.shape[1] / args.dsub))
        cbytes += int(np.ceil(W.shape[0] * G * args.M * args.bits / 8) + args.M * K * args.dsub * 2
                      + W.shape[0] * 4)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(data.files)} matrices, {time.time()-t0:.0f}s", flush=True)
    build_s = time.time() - t0
    set_weights(victim, recon)
    rloss = eval_loss(victim, eb, DEV, max_batches=args.eval_batches)
    ratio = cbytes / fp32
    print(f"[aqlm-full] DONE {build_s:.0f}s | {bpp:.2f}bpp | ratio={ratio:.4f} (amortized~{bpp/32:.4f}) "
          f"| recon_loss={rloss:.4f} (ppl {np.exp(rloss):.2f}) vs victim {vloss:.4f} "
          f"(ppl {np.exp(vloss):.2f})", flush=True)
    os.makedirs("runs/pythia_aqlm", exist_ok=True)
    json.dump({"M": args.M, "bits": args.bits, "dsub": args.dsub, "iters": args.iters, "bpp": bpp,
               "ratio": ratio, "recon_loss": rloss, "victim_loss": vloss, "build_s": build_s},
              open(f"runs/pythia_aqlm/M{args.M}_b{args.bits}_d{args.dsub}.json", "w"), indent=1)


if __name__ == "__main__":
    main()

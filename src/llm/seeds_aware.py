"""Activation-aware seeds for the open sandbox (torch + GPU + Hessian H). These seed the
auto-research with the *real* paper machinery so the search starts from it and improves.

seed_aqlm_aware: additive/multi-codebook VQ (AQLM family) with two key activation-aware moves:
  (1) scale each input column by sqrt(diag(H)) so plain Euclidean error in the scaled space
      approximates the activation-weighted (output) error the paper minimises;
  (2) GPU k-means (vectorised scatter update) so it's fast enough to use big codebooks / many iters.
Reconstruct is pure: it stores codebooks, indices, and the per-column scales, and unscales.
"""

SEED_AQLM_AWARE = '''
import numpy as np, torch
KNOBS = {"dsub": 8, "K": 256, "M": 2, "iters": 20, "scale_bits": 0}
def _kmeans(X, K, iters):
    N = X.shape[0]
    c = X[torch.randperm(N, device=X.device)[:K]].clone()
    for _ in range(iters):
        lab = torch.cdist(X, c).argmin(1)
        c2 = torch.zeros_like(c); c2.index_add_(0, lab, X)
        cnt = torch.bincount(lab, minlength=K).clamp_min(1).unsqueeze(1).float()
        c = c2 / cnt
    lab = torch.cdist(X, c).argmin(1)
    return c, lab
def compress_tensor(w, H=None, dsub=8, K=256, M=2, iters=20, scale_bits=0):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    W = torch.from_numpy(np.ascontiguousarray(w)).float().to(dev)
    out_f, in_f = W.shape
    if H is not None:
        s = torch.from_numpy(np.diag(H).copy()).float().to(dev).clamp_min(1e-12).sqrt()
    else:
        s = torch.ones(in_f, device=dev)
    Ws = W * s[None, :]
    pad = (-in_f) % dsub
    if pad: Ws = torch.cat([Ws, torch.zeros(out_f, pad, device=dev)], 1)
    nsub = Ws.shape[1] // dsub
    V = Ws.reshape(out_f * nsub, dsub)
    resid = V.clone(); cbs = []; idxs = []
    for m in range(M):
        c, lab = _kmeans(resid, K, iters)
        cbs.append(c.cpu().numpy().astype(np.float16))
        idxs.append(lab.cpu().numpy().astype(np.uint8 if K <= 256 else np.uint16))
        resid = resid - c[lab]
    return {"cbs": cbs, "idxs": idxs, "s": s.cpu().numpy().astype(np.float16),
            "in": np.int32(in_f), "out": np.int32(out_f), "dsub": np.int32(dsub), "nsub": np.int32(nsub)}
def reconstruct_tensor(payload, shape):
    dsub = int(payload["dsub"]); nsub = int(payload["nsub"])
    out_f = int(payload["out"]); in_f = int(payload["in"])
    rec = np.zeros((out_f * nsub, dsub), np.float32)
    for c, lab in zip(payload["cbs"], payload["idxs"]):
        rec += c.astype(np.float32)[lab.astype(np.int64)]
    Ws = rec.reshape(out_f, nsub * dsub)[:, :in_f]
    s = payload["s"].astype(np.float32)
    s = np.where(s < 1e-12, 1e-12, s)
    return (Ws / s[None, :]).reshape(shape)
'''

# same VQ but WITHOUT the H weighting (s=1) — A/B control to measure what activation-awareness buys
SEED_VQ_PLAIN = SEED_AQLM_AWARE.replace(
    's = torch.from_numpy(np.diag(H).copy()).float().to(dev).clamp_min(1e-12).sqrt()',
    's = torch.ones(in_f, device=dev)')

# Full AQLM essence: k-means init, then GRADIENT fine-tune the codebooks to minimise the full
# H-weighted output error  E = sum_rows (w_r - ŵ_r)^T H (w_r - ŵ_r)  (= what the layer's outputs
# actually see). This is the paper's real mechanism; only possible now that torch is allowed.
SEED_AQLM_FINETUNE = '''
import numpy as np, torch
KNOBS = {"dsub": 8, "K": 256, "M": 2, "kmeans_iters": 12, "ft_steps": 120, "ft_lr": 0.01}
def _assign(X, c, chunk=1000000):   # chunked nearest-centroid so big matrices don't OOM the cdist
    out = torch.empty(X.shape[0], dtype=torch.long, device=X.device)
    for i in range(0, X.shape[0], chunk):
        out[i:i+chunk] = torch.cdist(X[i:i+chunk], c).argmin(1)
    return out
def _kmeans(X, K, iters):
    c = X[torch.randperm(X.shape[0], device=X.device)[:K]].clone()
    for _ in range(iters):
        lab = _assign(X, c)
        c2 = torch.zeros_like(c); c2.index_add_(0, lab, X)
        cnt = torch.bincount(lab, minlength=K).clamp_min(1).unsqueeze(1).float()
        c = c2 / cnt
    return c, _assign(X, c)
def compress_tensor(w, H=None, dsub=8, K=256, M=2, kmeans_iters=12, ft_steps=120, ft_lr=0.01):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    W = torch.from_numpy(np.ascontiguousarray(w)).float().to(dev)
    out_f, in_f = W.shape
    Hm = (torch.from_numpy(H).float().to(dev) if H is not None else torch.eye(in_f, device=dev))
    pad = (-in_f) % dsub
    Wp = torch.cat([W, torch.zeros(out_f, pad, device=dev)], 1) if pad else W
    nsub = Wp.shape[1] // dsub
    V = Wp.reshape(out_f * nsub, dsub)
    resid = V.clone(); cbs = []; idxs = []
    for m in range(M):
        c, lab = _kmeans(resid, K, kmeans_iters)
        cbs.append(c.clone()); idxs.append(lab); resid = resid - c[lab]
    # gradient fine-tune the codebooks (assignments fixed) on the H-weighted error
    cb_p = [c.clone().requires_grad_(True) for c in cbs]
    opt = torch.optim.Adam(cb_p, lr=ft_lr)
    for _ in range(ft_steps):
        sub = sum(cb_p[m][idxs[m]] for m in range(M))            # (out*nsub, dsub)
        What = sub.reshape(out_f, nsub * dsub)[:, :in_f]
        err = W - What
        loss = ((err @ Hm) * err).sum() / out_f
        opt.zero_grad(); loss.backward(); opt.step()
    cbs_np = [c.detach().cpu().numpy().astype(np.float16) for c in cb_p]
    idx_np = [l.cpu().numpy().astype(np.uint8 if K <= 256 else np.uint16) for l in idxs]
    return {"cbs": cbs_np, "idxs": idx_np, "in": np.int32(in_f), "out": np.int32(out_f),
            "dsub": np.int32(dsub), "nsub": np.int32(nsub)}
def reconstruct_tensor(payload, shape):
    dsub = int(payload["dsub"]); nsub = int(payload["nsub"]); out_f = int(payload["out"]); in_f = int(payload["in"])
    rec = np.zeros((out_f * nsub, dsub), np.float32)
    for c, lab in zip(payload["cbs"], payload["idxs"]):
        rec += c.astype(np.float32)[lab.astype(np.int64)]
    return rec.reshape(out_f, nsub * dsub)[:, :in_f].reshape(shape)
'''

AWARE_SEEDS = {
    "seed_aqlm_finetune": ("vq_aware", SEED_AQLM_FINETUNE),
    "seed_aqlm_aware": ("vq_aware", SEED_AQLM_AWARE),
    "seed_vq_plain": ("vq", SEED_VQ_PLAIN),
}

"""Compression library.

Each technique compresses only the conv/linear *weight* tensors of a state_dict
(>=2-D, key ends in "weight"). BatchNorm params, biases, and 1-D tensors are passed
through unchanged and are excluded from the byte ratio.

Public API
----------
    compress(state_dict, technique, knob, **kwargs) -> dict
    reconstruct(compressed) -> state_dict (all keys; compressed ones dense fp32)
    total_bytes(compressed, original_state_dict) -> (compressed_bytes, baseline_fp32_bytes)

The returned `compressed` dict has keys: technique, knob, payload, bytes,
dense_keys, original_shapes. `payload` holds {"compressed": {key: per-key payload},
"passthrough": {key: tensor}}. Each per-key payload is a picklable dict with a "kind"
field so `reconstruct` can dispatch without re-deriving the technique.
"""
import math

import numpy as np
import torch

from .utils import compressible_keys, fp32_bytes_of_keys

TECHNIQUES = {
    "random_sparse": "keep p% of entries at uniformly random positions",
    "magnitude_prune": "keep top-k% by |w| per tensor",
    "snip": "keep top-k% by |w . dL/dw| on one batch",
    "fisher_prune": "keep top-k% by diagonal Fisher (g^2 over 10 batches)",
    "low_rank": "rank-r truncated SVD per tensor",
    "quantize": "uniform per-tensor symmetric scalar quantization",
    "kmeans": "per-tensor k-means weight sharing",
    "magprune_quant": "magnitude-prune then quantize the kept values",
    "additive_vq": "additive multi-codebook VQ with shared global codebooks (simplified)",
    "aqlm": "AQLM: activation-aware per-layer additive quantization (paper reproduction)",
}


# --------------------------------------------------------------------------- #
# index / sparse helpers
# --------------------------------------------------------------------------- #

def _choose_index_encoding(mask_flat_bool, numel, k):
    """Pick the smaller of a packed bitmask (1 bit/param) or int32 indices.

    Returns (encoding, payload_dict, index_bytes).
    """
    bitmask_bytes = math.ceil(numel / 8)
    indices_bytes = 4 * k
    if bitmask_bytes <= indices_bytes:
        bits = np.packbits(mask_flat_bool.astype(np.uint8))
        return "bitmask", {"mask_bits": bits, "numel": int(numel)}, int(bitmask_bytes)
    idx = np.nonzero(mask_flat_bool)[0].astype(np.int32)
    return "indices", {"idx": idx}, int(indices_bytes)


def _decode_mask(payload, numel):
    if payload["encoding"] == "bitmask":
        mask = np.unpackbits(payload["mask_bits"])[:numel].astype(bool)
        return np.nonzero(mask)[0]
    return payload["idx"].astype(np.int64)


def _sparse_payload(w, mask):
    """Store kept fp32 values + smallest index encoding. `mask` is a bool tensor."""
    w_flat = w.reshape(-1).numpy().astype(np.float32)
    mask_flat = mask.reshape(-1).numpy().astype(bool)
    numel = w_flat.size
    k = int(mask_flat.sum())
    enc, idx_payload, index_bytes = _choose_index_encoding(mask_flat, numel, k)
    values = w_flat[mask_flat]  # ascending flat-index order
    pl = {"kind": "sparse", "encoding": enc, "values": values, "k": k,
          "bytes": int(4 * k + index_bytes)}
    pl.update(idx_payload)
    return pl


def _sparse_reconstruct(pl, shape):
    numel = int(np.prod(shape))
    flat = np.zeros(numel, dtype=np.float32)
    pos = _decode_mask(pl, numel)
    flat[pos] = pl["values"]
    return torch.from_numpy(flat.reshape(shape)).float()


# --------------------------------------------------------------------------- #
# quantization helpers (symmetric, per-tensor)
# --------------------------------------------------------------------------- #

def _symmetric_quantize(values, bits):
    """Round-to-nearest symmetric quantization of a 1-D float array.
    Returns (codes_int8, scale). bits==1 is sign quantization."""
    values = values.astype(np.float32)
    if values.size == 0:
        return values.astype(np.int8), 1.0
    if bits == 1:
        scale = float(np.mean(np.abs(values))) or 1.0
        codes = np.where(values >= 0, 1, -1).astype(np.int8)
        return codes, scale
    qmax = (1 << (bits - 1)) - 1
    amax = float(np.max(np.abs(values)))
    scale = amax / qmax if amax > 0 else 1.0
    codes = np.clip(np.round(values / scale), -qmax, qmax).astype(np.int8)
    return codes, scale


def _symmetric_dequantize(codes, scale):
    return codes.astype(np.float32) * np.float32(scale)


# --------------------------------------------------------------------------- #
# importance scores (snip / fisher) — need model + data
# --------------------------------------------------------------------------- #

def _grad_scores(state_dict, ckeys, train_loader, device, n_batches, square):
    """Compute per-key importance from gradients on `n_batches` minibatches.

    snip:   square=False, n_batches=1 -> |w * grad|
    fisher: square=True,  n_batches=10 -> mean(grad^2)
    """
    from .models import resnet20
    import torch.nn as nn

    model = resnet20().to(device)
    model.load_state_dict(state_dict)
    model.eval()  # BN uses running stats; we only want gradients, not BN updates
    ce = nn.CrossEntropyLoss()

    # map state_dict key -> parameter
    name2param = dict(model.named_parameters())
    accum = {k: torch.zeros_like(name2param[k]) for k in ckeys}

    it = iter(train_loader)
    used = 0
    while used < n_batches:
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(train_loader)
            x, y = next(it)
        x, y = x.to(device), y.to(device)
        model.zero_grad(set_to_none=True)
        logits = model(x)
        loss = ce(logits, y)
        loss.backward()
        for k in ckeys:
            g = name2param[k].grad
            if g is None:
                continue
            if square:
                accum[k] += g.detach() ** 2
            else:
                accum[k] += (name2param[k].detach() * g.detach()).abs()
        used += 1

    scores = {}
    for k in ckeys:
        s = accum[k] / used
        if not square:
            s = s.abs()
        scores[k] = s.detach().cpu()
    return scores


def _topk_mask(score_tensor, keep_fraction):
    """Boolean mask keeping the top `keep_fraction` entries by score (per tensor)."""
    numel = score_tensor.numel()
    k = int(round(keep_fraction * numel))
    mask = torch.zeros(numel, dtype=torch.bool)
    if k > 0:
        flat = score_tensor.reshape(-1)
        topk = torch.topk(flat, k, largest=True).indices
        mask[topk] = True
    return mask.reshape(score_tensor.shape)


def _random_mask(shape, keep_fraction, rng):
    numel = int(np.prod(shape))
    k = int(round(keep_fraction * numel))
    mask = np.zeros(numel, dtype=bool)
    if k > 0:
        pos = rng.choice(numel, size=k, replace=False)
        mask[pos] = True
    return torch.from_numpy(mask.reshape(shape))


# --------------------------------------------------------------------------- #
# low-rank
# --------------------------------------------------------------------------- #

def _lowrank_payload(w, rank_fraction):
    shape = tuple(w.shape)
    mat = w.reshape(shape[0], -1).numpy().astype(np.float32)  # (m, n)
    m, n = mat.shape
    r = max(1, int(math.ceil(rank_fraction * min(m, n))))
    r = min(r, min(m, n))
    U, S, Vt = np.linalg.svd(mat, full_matrices=False)
    Us = (U[:, :r] * S[:r]).astype(np.float32)   # (m, r), folds S into U
    V = Vt[:r, :].astype(np.float32)             # (r, n)
    return {"kind": "lowrank", "Us": Us, "V": V, "r": int(r),
            "mat_shape": (int(m), int(n)), "bytes": int(r * (m + n) * 4)}


def _lowrank_reconstruct(pl, shape):
    mat = pl["Us"] @ pl["V"]  # (m, n)
    return torch.from_numpy(mat.reshape(shape)).float()


# --------------------------------------------------------------------------- #
# quantize (dense)
# --------------------------------------------------------------------------- #

def _quantize_payload(w, bits):
    flat = w.reshape(-1).numpy().astype(np.float32)
    codes, scale = _symmetric_quantize(flat, bits)
    numel = flat.size
    return {"kind": "quant", "codes": codes, "scale": float(scale), "bits": int(bits),
            "numel": int(numel),
            "bytes": int(math.ceil(numel * bits / 8) + 4)}


def _quantize_reconstruct(pl, shape):
    deq = _symmetric_dequantize(pl["codes"], pl["scale"])
    return torch.from_numpy(deq.reshape(shape)).float()


# --------------------------------------------------------------------------- #
# k-means weight sharing
# --------------------------------------------------------------------------- #

def _kmeans_payload(w, bits, seed):
    from sklearn.cluster import MiniBatchKMeans
    flat = w.reshape(-1).numpy().astype(np.float32)
    numel = flat.size
    n_clusters = min(1 << bits, numel)
    n_clusters = max(1, n_clusters)
    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed,
                         n_init=3, batch_size=max(256, n_clusters * 4))
    labels = km.fit_predict(flat.reshape(-1, 1))
    centroids = km.cluster_centers_.reshape(-1).astype(np.float32)
    # honest index bits: ceil(log2(n_clusters)) per weight; codebook is fp32
    idx_bits = max(1, int(math.ceil(math.log2(n_clusters)))) if n_clusters > 1 else 0
    labels = labels.astype(np.int32)
    return {"kind": "kmeans", "labels": labels, "codebook": centroids,
            "numel": int(numel),
            "bytes": int(math.ceil(numel * idx_bits / 8) + centroids.size * 4)}


def _kmeans_reconstruct(pl, shape):
    deq = pl["codebook"][pl["labels"]]
    return torch.from_numpy(deq.reshape(shape)).float()


# --------------------------------------------------------------------------- #
# magnitude-prune + quantize (stacked)
# --------------------------------------------------------------------------- #

def _magprune_quant_payload(w, keep_fraction, bits):
    mask = _topk_mask(w.abs(), keep_fraction)
    w_flat = w.reshape(-1).numpy().astype(np.float32)
    mask_flat = mask.reshape(-1).numpy().astype(bool)
    numel = w_flat.size
    k = int(mask_flat.sum())
    enc, idx_payload, index_bytes = _choose_index_encoding(mask_flat, numel, k)
    kept = w_flat[mask_flat]
    codes, scale = _symmetric_quantize(kept, bits)
    value_bytes = math.ceil(k * bits / 8) + 4  # quantized kept values + fp32 scale
    pl = {"kind": "magprune_quant", "encoding": enc, "codes": codes,
          "scale": float(scale), "bits": int(bits), "k": k,
          "bytes": int(value_bytes + index_bytes)}
    pl.update(idx_payload)
    return pl


def _magprune_quant_reconstruct(pl, shape):
    numel = int(np.prod(shape))
    flat = np.zeros(numel, dtype=np.float32)
    pos = _decode_mask(pl, numel)
    flat[pos] = _symmetric_dequantize(pl["codes"], pl["scale"])
    return torch.from_numpy(flat.reshape(shape)).float()


# --------------------------------------------------------------------------- #
# additive / multi-codebook vector quantization (AQLM-style)
# --------------------------------------------------------------------------- #
# Each weight tensor is per-tensor scale-normalized and split into groups of `d`
# consecutive weights. Every group is represented as a SUM of M codewords, one
# drawn from each of M codebooks of 2**b entries (dim d). Codebooks are LEARNED
# GLOBALLY (shared across all compressed tensors) so their byte cost is amortized
# once over the whole network — this is what makes very high compression possible
# on a tiny model. Codes are found by greedy residual encoding; codebooks are
# refined by alternating least squares (a tractable stand-in for AQLM's beam
# search + codebook update). Reference: Egiazarian et al., "Extreme Compression
# of LLMs via Additive Quantization" (AQLM), arXiv 2401.06118.

def _greedy_encode(groups, codebooks):
    """Greedy residual encoding: pick one codeword per codebook to sum toward each group.
    groups (G, d); codebooks list of M arrays (K, d). Returns codes (G, M) int."""
    G = groups.shape[0]
    M = len(codebooks)
    residual = groups.astype(np.float32).copy()
    codes = np.zeros((G, M), dtype=np.int64)
    for m in range(M):
        cb = codebooks[m]                                  # (K, d)
        # squared distance from residual to each codeword
        # ||r||^2 - 2 r.c + ||c||^2 ; drop ||r||^2 (constant per row)
        d2 = (-2.0 * residual @ cb.T) + (cb * cb).sum(1)[None, :]
        a = d2.argmin(1)
        codes[:, m] = a
        residual = residual - cb[a]
    return codes


def _ls_update_codebooks(groups, codes, M, K, d):
    """Least-squares update of all codebooks given fixed assignments.
    Solves min_C || groups - sum_m C_m[codes_m] ||^2 via normal equations on the
    one-hot design (G, M*K)."""
    G = groups.shape[0]
    # Build A^T A (M*K, M*K) and A^T X (M*K, d) without materializing A densely.
    MK = M * K
    AtA = np.zeros((MK, MK), dtype=np.float64)
    AtX = np.zeros((MK, d), dtype=np.float64)
    # column index of group g in codebook m is m*K + codes[g,m]
    cols = codes + (np.arange(M) * K)[None, :]             # (G, M)
    for g in range(0, G, 8192):                            # chunk to bound memory
        cg = cols[g:g + 8192]                              # (n, M)
        xg = groups[g:g + 8192]                            # (n, d)
        n = cg.shape[0]
        for i in range(M):
            ci = cg[:, i]
            np.add.at(AtX, ci, xg)
            for j in range(M):
                cj = cg[:, j]
                # accumulate counts of (ci, cj) co-occurrence
                np.add.at(AtA, (ci, cj), 1.0)
    AtA += 1e-6 * np.eye(MK)
    B = np.linalg.solve(AtA, AtX)                          # (M*K, d)
    return [B[m * K:(m + 1) * K].astype(np.float32) for m in range(M)]


def _additive_vq_compress(state_dict, ckeys, original_shapes, passthrough,
                          M, b, d, seed, knob, refine_iters=2):
    K = 1 << b
    rng = np.random.RandomState(seed)

    # normalize each tensor, split into groups of d (zero-pad tail)
    per_key = {}
    chunks = []
    for k in ckeys:
        w = state_dict[k].detach().float().cpu().reshape(-1).numpy()
        scale = float(w.std()) or 1.0
        wn = w / scale
        n = wn.size
        n_groups = int(math.ceil(n / d))
        pad = n_groups * d - n
        if pad:
            wn = np.concatenate([wn, np.zeros(pad, np.float32)])
        g = wn.reshape(n_groups, d).astype(np.float32)
        per_key[k] = {"scale": scale, "n": int(n), "pad": int(pad),
                      "n_groups": int(n_groups)}
        chunks.append(g)
    X = np.vstack(chunks)                                  # (G_total, d)

    # init codebooks by residual k-means (RVQ)
    from sklearn.cluster import MiniBatchKMeans
    codebooks = []
    residual = X.copy()
    for m in range(M):
        nk = min(K, residual.shape[0])
        km = MiniBatchKMeans(n_clusters=nk, random_state=seed + m, n_init=3,
                             batch_size=max(256, nk * 4))
        km.fit(residual)
        cb = np.zeros((K, d), np.float32)
        cb[:nk] = km.cluster_centers_.astype(np.float32)
        codebooks.append(cb)
        a = km.predict(residual)
        residual = residual - cb[a]

    # refine: alternate (encode, least-squares codebook update)
    codes = _greedy_encode(X, codebooks)
    for _ in range(refine_iters):
        codebooks = _ls_update_codebooks(X, codes, M, K, d)
        codes = _greedy_encode(X, codebooks)

    # split codes back per key
    compressed = {}
    off = 0
    for k in ckeys:
        ng = per_key[k]["n_groups"]
        c = codes[off:off + ng].astype(np.int32)
        off += ng
        compressed[k] = {"kind": "additive_vq", "codes": c,
                         "scale": per_key[k]["scale"], "n": per_key[k]["n"],
                         "pad": per_key[k]["pad"]}

    # honest byte accounting
    code_bits = int(codes.shape[0]) * M * b              # total code bits
    codebook_bytes = M * K * d * 2                       # fp16 codebooks, counted ONCE
    scale_bytes = len(ckeys) * 4                         # one fp32 scale per tensor
    total = int(math.ceil(code_bits / 8) + codebook_bytes + scale_bytes)

    cb_fp16 = [cb.astype(np.float16) for cb in codebooks]
    return {
        "technique": "additive_vq",
        "knob": list(knob),
        "payload": {"compressed": compressed, "passthrough": passthrough,
                    "shared": {"codebooks": cb_fp16, "M": M, "b": b, "d": d}},
        "bytes": total,
        "dense_keys": ckeys,
        "original_shapes": original_shapes,
    }


def _additive_vq_reconstruct(pl, shape, codebooks):
    d = codebooks[0].shape[1]
    codes = pl["codes"].astype(np.int64)                  # (n_groups, M)
    recon = np.zeros((codes.shape[0], d), np.float32)
    for m, cb in enumerate(codebooks):
        recon += cb.astype(np.float32)[codes[:, m]]
    flat = recon.reshape(-1)[: pl["n"]] * np.float32(pl["scale"])
    return torch.from_numpy(flat.reshape(shape)).float()


# --------------------------------------------------------------------------- #
# AQLM — activation-aware additive quantization (Egiazarian et al., arXiv 2401.06118)
# --------------------------------------------------------------------------- #
# Faithful adaptation to a CIFAR conv net:
#   * PER-LAYER codebooks (M codebooks of 2^b codewords over input-groups of d).
#   * ACTIVATION-AWARE: each input dimension j is weighted by its calibration
#     activation energy  H_jj = sum_calib x_j^2  (the diagonal of the layer Hessian
#     X X^T; for conv layers X is the im2col patch matrix). This is the dominant
#     activation-aware effect (cf. AWQ/Wanda) — we use the diagonal, not the full
#     off-diagonal Hessian.
#   * Per-output-channel scales, residual-k-means init, weighted greedy residual
#     code assignment, and alternating weighted-least-squares codebook updates.
# (Simplifications vs. full AQLM: diagonal — not full — Hessian, and greedy — not
#  beam-search — code assignment. Documented; see RESULTS.)

def _aqlm_importance(state_dict, ckeys, train_loader, device, n_batches=4):
    """Per-input-dimension activation energy (diagonal Hessian) for each layer,
    aligned with weight.reshape(out, -1) column order."""
    import torch.nn as nn
    import torch.nn.functional as F
    from .models import resnet20

    model = resnet20().to(device)
    model.load_state_dict(state_dict)
    model.eval()

    name2key = {}
    imp = {}
    handles = []
    for name, m in model.named_modules():
        key = f"{name}.weight"
        if key not in ckeys:
            continue
        name2key[name] = key
        imp[key] = None

        def make_hook(mod, k):
            def hook(module, inp):
                x = inp[0].detach()
                if isinstance(module, nn.Conv2d):
                    cols = F.unfold(x, kernel_size=module.kernel_size,
                                    dilation=module.dilation, padding=module.padding,
                                    stride=module.stride)          # (N, Cin*kh*kw, L)
                    e = (cols ** 2).sum(dim=(0, 2))                 # (Cin*kh*kw,)
                else:  # Linear
                    xf = x.reshape(-1, x.shape[-1])
                    e = (xf ** 2).sum(dim=0)                        # (in,)
                imp[k] = e if imp[k] is None else imp[k] + e
            return hook

        handles.append(m.register_forward_pre_hook(make_hook(m, key)))

    it = iter(train_loader)
    with torch.no_grad():
        for _ in range(n_batches):
            try:
                x, _ = next(it)
            except StopIteration:
                break
            model(x.to(device))
    for h in handles:
        h.remove()
    return {k: v.detach().cpu().numpy().astype(np.float64) for k, v in imp.items()}


def _rvq_init(X, M, K, seed):
    from sklearn.cluster import MiniBatchKMeans
    codebooks = []
    residual = X.astype(np.float32).copy()
    for m in range(M):
        nk = min(K, residual.shape[0])
        km = MiniBatchKMeans(n_clusters=nk, random_state=seed + m, n_init=3,
                             batch_size=max(256, nk * 4))
        km.fit(residual)
        cb = np.zeros((K, X.shape[1]), np.float32)
        cb[:nk] = km.cluster_centers_.astype(np.float32)
        codebooks.append(cb)
        residual = residual - cb[km.predict(residual)]
    return codebooks


def _wgreedy_encode(sub, codebooks, w):
    """Weighted greedy residual encoding. sub (n,d); w (d,) per-dim weights.
    Returns codes (n, M)."""
    n = sub.shape[0]
    M = len(codebooks)
    res = sub.astype(np.float32).copy()
    codes = np.zeros((n, M), np.int64)
    for m in range(M):
        cb = codebooks[m]                                   # (K, d)
        # weighted sq dist (drop the per-row constant ||res||^2_w):
        d2 = (-2.0 * (res * w) @ cb.T) + ((cb * cb) @ w)[None, :]
        a = d2.argmin(1)
        codes[:, m] = a
        res = res - cb[a]
    return codes


def _wls_codebooks(Wb, codes, wts, M, K, d):
    """Weighted least-squares codebook update (per output-dim e).
    Wb (out,G,d); codes (out,G,M); wts (G,d)."""
    out, G, _ = Wb.shape
    n = out * G
    cols = codes.reshape(n, M) + (np.arange(M) * K)[None, :]   # (n, M) flat col idx
    gidx = np.tile(np.arange(G), out)                          # block index per row
    Wf = Wb.reshape(n, d)
    MK = M * K
    codebooks = [np.zeros((K, d), np.float32) for _ in range(M)]
    for e in range(d):
        we = wts[gidx, e].astype(np.float64)                  # (n,) weight for this dim
        xe = Wf[:, e].astype(np.float64)
        AtA = np.zeros((MK, MK)); Atx = np.zeros(MK)
        for i in range(M):
            ci = cols[:, i]
            np.add.at(Atx, ci, we * xe)
            for j in range(M):
                np.add.at(AtA, (ci, cols[:, j]), we)
        AtA += 1e-6 * np.eye(MK)
        B = np.linalg.solve(AtA, Atx)                         # (MK,)
        for m in range(M):
            codebooks[m][:, e] = B[m * K:(m + 1) * K]
    return codebooks


def _aqlm_payload(w, imp, M, b, d, seed, n_iters=3):
    K = 1 << b
    shape = tuple(w.shape)
    out = shape[0]
    W = w.reshape(out, -1).numpy().astype(np.float32)         # (out, in)
    inf = W.shape[1]
    s = np.maximum(W.std(axis=1, keepdims=True), 1e-8)        # (out,1) per-output scale
    Wn = W / s
    inp = int(math.ceil(inf / d) * d)
    pad = inp - inf
    if pad:
        Wn = np.pad(Wn, ((0, 0), (0, pad)))
        imp = np.pad(imp.astype(np.float64), (0, pad))
    G = inp // d
    Wb = Wn.reshape(out, G, d)
    wts = imp.reshape(G, d)
    wts = wts / (wts.mean() + 1e-12)                          # normalize ~O(1)

    codebooks = _rvq_init(Wb.reshape(out * G, d), M, K, seed)
    codes = np.zeros((out, G, M), np.int64)
    for _ in range(n_iters):
        for g in range(G):
            codes[:, g, :] = _wgreedy_encode(Wb[:, g, :], codebooks, wts[g])
        codebooks = _wls_codebooks(Wb, codes, wts, M, K, d)
    for g in range(G):
        codes[:, g, :] = _wgreedy_encode(Wb[:, g, :], codebooks, wts[g])

    code_bits = out * G * M * b
    codebook_bytes = M * K * d * 2                            # fp16, per layer
    scale_bytes = out * 4
    return {"kind": "aqlm",
            "codes": codes.astype(np.int32),
            "codebooks": np.stack(codebooks).astype(np.float16),  # (M,K,d)
            "scale": s.reshape(-1).astype(np.float32),
            "pad": int(pad), "G": int(G),
            "bytes": int(math.ceil(code_bits / 8) + codebook_bytes + scale_bytes)}


def _aqlm_reconstruct(pl, shape):
    cb = pl["codebooks"].astype(np.float32)                   # (M,K,d)
    codes = pl["codes"].astype(np.int64)                      # (out,G,M)
    out, G, M = codes.shape
    d = cb.shape[2]
    recon = np.zeros((out, G, d), np.float32)
    for m in range(M):
        recon += cb[m][codes[:, :, m]]
    flat = recon.reshape(out, G * d)
    inf = int(np.prod(shape[1:]))
    flat = flat[:, :inf] * pl["scale"][:, None]
    return torch.from_numpy(flat.reshape(shape)).float()


# --------------------------------------------------------------------------- #
# top-level dispatch
# --------------------------------------------------------------------------- #

_RECONSTRUCT = {
    "aqlm": _aqlm_reconstruct,
    "sparse": _sparse_reconstruct,
    "lowrank": _lowrank_reconstruct,
    "quant": _quantize_reconstruct,
    "kmeans": _kmeans_reconstruct,
    "magprune_quant": _magprune_quant_reconstruct,
}


def compress(state_dict, technique, knob, *, train_loader=None, device=None,
             seed=42, **kwargs):
    if technique not in TECHNIQUES:
        raise ValueError(f"unknown technique {technique!r}")

    ckeys = compressible_keys(state_dict)
    original_shapes = {k: tuple(state_dict[k].shape) for k in ckeys}
    passthrough = {k: v.detach().cpu().clone()
                   for k, v in state_dict.items() if k not in ckeys}

    if technique == "additive_vq":
        M, b, d = (int(x) for x in knob)
        return _additive_vq_compress(state_dict, ckeys, original_shapes,
                                     passthrough, M, b, d, seed, knob)

    scores = None
    importance = None
    if technique == "snip":
        scores = _grad_scores(state_dict, ckeys, train_loader, device,
                              n_batches=1, square=False)
    elif technique == "fisher_prune":
        scores = _grad_scores(state_dict, ckeys, train_loader, device,
                              n_batches=10, square=True)
    elif technique == "aqlm":
        importance = _aqlm_importance(state_dict, ckeys, train_loader, device)

    rng = np.random.RandomState(seed)
    compressed = {}
    total = 0
    for k in ckeys:
        w = state_dict[k].detach().float().cpu()
        if technique == "random_sparse":
            pl = _sparse_payload(w, _random_mask(w.shape, knob, rng))
        elif technique == "magnitude_prune":
            pl = _sparse_payload(w, _topk_mask(w.abs(), knob))
        elif technique in ("snip", "fisher_prune"):
            pl = _sparse_payload(w, _topk_mask(scores[k], knob))
        elif technique == "low_rank":
            pl = _lowrank_payload(w, knob)
        elif technique == "quantize":
            pl = _quantize_payload(w, int(knob))
        elif technique == "kmeans":
            pl = _kmeans_payload(w, int(knob), seed)
        elif technique == "magprune_quant":
            keep_fraction, bits = knob
            pl = _magprune_quant_payload(w, float(keep_fraction), int(bits))
        elif technique == "aqlm":
            M, b, dd = (int(x) for x in knob)
            pl = _aqlm_payload(w, importance[k], M, b, dd, seed)
        else:  # pragma: no cover
            raise ValueError(technique)
        compressed[k] = pl
        total += pl["bytes"]

    return {
        "technique": technique,
        "knob": knob,
        "payload": {"compressed": compressed, "passthrough": passthrough},
        "bytes": int(total),
        "dense_keys": ckeys,
        "original_shapes": original_shapes,
    }


def reconstruct(compressed):
    payload = compressed["payload"]
    shapes = compressed["original_shapes"]
    sd = {}
    for k, v in payload["passthrough"].items():
        sd[k] = v.detach().cpu().clone().float() if v.is_floating_point() else v.clone()
    shared = payload.get("shared")
    for k, pl in payload["compressed"].items():
        if pl["kind"] == "additive_vq":
            sd[k] = _additive_vq_reconstruct(pl, shapes[k], shared["codebooks"])
        else:
            sd[k] = _RECONSTRUCT[pl["kind"]](pl, shapes[k])
    return sd


def total_bytes(compressed, original_state_dict):
    compressed_bytes = int(compressed["bytes"])
    baseline = fp32_bytes_of_keys(original_state_dict, compressed["dense_keys"])
    return compressed_bytes, baseline


def compression_ratio(compressed, original_state_dict):
    c, b = total_bytes(compressed, original_state_dict)
    return c / b


def overhead_bytes(compressed):
    """Size-independent (or sub-linear) bytes: codebooks and scales. These are a
    fixed/per-layer cost that vanishes as a fraction at large model size. The
    'amortized' ratio subtracts these, counting only the per-weight code/value/index
    bytes that scale ~linearly with the number of weights — approximating the
    compression ratio one would get on a much larger model."""
    payload = compressed["payload"]
    comp = payload["compressed"]
    ov = 0
    shared = payload.get("shared")
    if shared is not None:  # additive_vq: global codebook (fp16), counted once
        ov += int(sum(cb.size for cb in shared["codebooks"]) * 2)
    for pl in comp.values():
        kind = pl["kind"]
        if kind == "quant":
            ov += 4                                   # per-tensor fp32 scale
        elif kind == "kmeans":
            ov += int(pl["codebook"].size * 4)        # per-tensor fp32 codebook
        elif kind == "magprune_quant":
            ov += 4                                   # per-tensor fp32 scale
        elif kind == "additive_vq":
            ov += 4                                   # per-tensor scale (codebook in 'shared')
        elif kind == "aqlm":
            ov += int(pl["codebooks"].size * 2 + pl["scale"].size * 4)  # per-layer cb + scales
        # sparse / lowrank: no size-independent overhead
    return int(ov)


def amortized_bytes(compressed):
    """Per-weight bytes only (total minus size-independent overhead)."""
    return int(compressed["bytes"]) - overhead_bytes(compressed)


# --------------------------------------------------------------------------- #
# unit-style tests
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import torch.utils.data as tud
    from .models import resnet20

    torch.manual_seed(0)
    np.random.seed(0)
    model = resnet20()
    sd = {k: v.clone() for k, v in model.state_dict().items()}
    ckeys = compressible_keys(sd)
    base_bytes = fp32_bytes_of_keys(sd, ckeys)
    print(f"compressible keys={len(ckeys)} baseline_fp32_bytes={base_bytes:,}")

    # tiny synthetic loader for snip/fisher
    x = torch.randn(256, 3, 32, 32)
    y = torch.randint(0, 10, (256,))
    loader = tud.DataLoader(tud.TensorDataset(x, y), batch_size=128, shuffle=True)
    device = torch.device("cpu")

    cases = [
        ("random_sparse", 0.1, None),
        ("magnitude_prune", 0.1, None),
        ("snip", 0.1, None),
        ("fisher_prune", 0.1, None),
        ("low_rank", 0.5, 1.5),   # loose Frobenius rel-error bound at rank_frac 0.5
        ("quantize", 8, 0.2),
        ("kmeans", 6, 0.5),
        ("magprune_quant", (0.1, 4), None),
        ("additive_vq", [2, 8, 8], 0.8),  # 2 codebooks x 256 over dim-8 groups
        ("aqlm", [2, 8, 8], 0.8),         # activation-aware per-layer additive quant
    ]

    all_ok = True
    for tech, knob, err_bound in cases:
        comp = compress(sd, tech, knob, train_loader=loader, device=device, seed=42)
        rec = reconstruct(comp)
        # all keys present
        assert set(rec.keys()) == set(sd.keys()), f"{tech}: key mismatch"
        # shapes preserved
        for k in ckeys:
            assert tuple(rec[k].shape) == tuple(sd[k].shape), f"{tech}: shape {k}"
        # passthrough exact
        for k in sd:
            if k not in ckeys:
                assert torch.allclose(rec[k].float(), sd[k].float()), f"{tech}: passthrough {k}"
        cbytes, bbytes = total_bytes(comp, sd)
        ratio = cbytes / bbytes
        # closeness check (dense techniques only — sparse at low keep is far by design)
        msg = ""
        if err_bound is not None:
            num = den = 0.0
            for k in ckeys:
                num += float(torch.norm(rec[k] - sd[k].float()) ** 2)
                den += float(torch.norm(sd[k].float()) ** 2)
            rel = (num / den) ** 0.5
            ok = rel < err_bound
            all_ok &= ok
            msg = f" rel_fro_err={rel:.3f} (<{err_bound}) {'OK' if ok else 'FAIL'}"
        print(f"[{tech:>15}] knob={knob} ratio={ratio:.4f} "
              f"bytes={cbytes:,}/{bbytes:,}{msg}")

    print("ALL CLOSENESS CHECKS:", "OK" if all_ok else "FAIL")

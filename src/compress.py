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
# top-level dispatch
# --------------------------------------------------------------------------- #

_RECONSTRUCT = {
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

    scores = None
    if technique == "snip":
        scores = _grad_scores(state_dict, ckeys, train_loader, device,
                              n_batches=1, square=False)
    elif technique == "fisher_prune":
        scores = _grad_scores(state_dict, ckeys, train_loader, device,
                              n_batches=10, square=True)

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
    for k, pl in payload["compressed"].items():
        sd[k] = _RECONSTRUCT[pl["kind"]](pl, shapes[k])
    return sd


def total_bytes(compressed, original_state_dict):
    compressed_bytes = int(compressed["bytes"])
    baseline = fp32_bytes_of_keys(original_state_dict, compressed["dense_keys"])
    return compressed_bytes, baseline


def compression_ratio(compressed, original_state_dict):
    c, b = total_bytes(compressed, original_state_dict)
    return c / b


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

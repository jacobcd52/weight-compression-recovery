"""Hand-written seed candidates (pure numpy) that seed the archive across families:
scalar quantization, k-means weight-sharing, low-rank SVD, magnitude sparsity. Each conforms
to the candidate interface (KNOBS, compress_tensor, reconstruct_tensor)."""

_QUANT = '''
import numpy as np
KNOBS = {"bits": %d}
def compress_tensor(w, bits=%d):
    amax = float(np.max(np.abs(w))) or 1.0
    qmax = (1 << (bits - 1)) - 1
    scale = amax / qmax
    q = np.clip(np.round(w / scale), -qmax, qmax).astype(np.int8)
    return {"q": q, "scale": np.float32(scale)}
def reconstruct_tensor(payload, shape):
    return (payload["q"].astype(np.float32) * payload["scale"]).reshape(shape)
'''

_KMEANS = '''
import numpy as np
KNOBS = {"nbits": 4}
def compress_tensor(w, nbits=4):
    x = w.reshape(-1).astype(np.float32); k = 1 << nbits
    c = np.quantile(x, np.linspace(0, 1, k)).astype(np.float32)
    for _ in range(8):
        lab = np.argmin(np.abs(x[:, None] - c[None, :]), axis=1)
        for j in range(k):
            m = lab == j
            if m.any(): c[j] = x[m].mean()
    lab = np.argmin(np.abs(x[:, None] - c[None, :]), axis=1).astype(np.uint8)
    return {"lab": lab, "c": c.astype(np.float32)}
def reconstruct_tensor(payload, shape):
    return payload["c"][payload["lab"]].reshape(shape)
'''

_LOWRANK = '''
import numpy as np, math
KNOBS = {"frac": 0.25}
def compress_tensor(w, frac=0.25):
    m = w.reshape(w.shape[0], -1).astype(np.float32)
    r = max(1, int(math.ceil(frac * min(m.shape))))
    U, S, Vt = np.linalg.svd(m, full_matrices=False)
    return {"U": (U[:, :r] * S[:r]).astype(np.float16), "V": Vt[:r].astype(np.float16),
            "n": m.shape[1]}
def reconstruct_tensor(payload, shape):
    mat = payload["U"].astype(np.float32) @ payload["V"].astype(np.float32)
    return mat.reshape(shape)
'''

_MAGSPARSE = '''
import numpy as np
KNOBS = {"keep": 0.1}
def compress_tensor(w, keep=0.1):
    x = w.reshape(-1).astype(np.float32); n = x.size
    k = max(1, int(round(keep * n)))
    idx = np.argpartition(np.abs(x), n - k)[n - k:].astype(np.int32)
    return {"idx": idx, "vals": x[idx].astype(np.float16), "n": n}
def reconstruct_tensor(payload, shape):
    flat = np.zeros(payload["n"], np.float32)
    flat[payload["idx"]] = payload["vals"].astype(np.float32)
    return flat.reshape(shape)
'''

# --- aggressive sub-0.01 seeds (extreme end of the frontier) ---
_LOWRANK_ABS = '''
import numpy as np, math
KNOBS = {"rank": %d}
def compress_tensor(w, rank=%d):
    m = w.reshape(w.shape[0], -1).astype(np.float32)
    r = max(1, min(rank, min(m.shape)))
    U, S, Vt = np.linalg.svd(m, full_matrices=False)
    return {"U": (U[:, :r] * S[:r]).astype(np.float16), "V": Vt[:r].astype(np.float16),
            "n": m.shape[1]}
def reconstruct_tensor(payload, shape):
    mat = payload["U"].astype(np.float32) @ payload["V"].astype(np.float32)
    return mat.reshape(shape)
'''

# 1-bit: sign per weight (packed) + one fp16 magnitude scale per group
_QUANT1 = '''
import numpy as np
KNOBS = {"group": 64}
def compress_tensor(w, group=64):
    x = w.reshape(-1).astype(np.float32); n = x.size
    pad = (-n) % group
    if pad: x = np.concatenate([x, np.zeros(pad, np.float32)])
    g = x.reshape(-1, group)
    scale = np.abs(g).mean(1).astype(np.float16)
    packed = np.packbits((g >= 0).astype(np.uint8).reshape(-1))
    return {"packed": packed, "scale": scale, "n": np.int32(n), "group": np.int32(group)}
def reconstruct_tensor(payload, shape):
    n = int(payload["n"]); group = int(payload["group"])
    tot = n + ((-n) % group)
    bits = np.unpackbits(payload["packed"])[:tot].reshape(-1, group).astype(np.float32)
    rec = ((bits * 2 - 1) * payload["scale"].astype(np.float32)[:, None]).reshape(-1)[:n]
    return rec.reshape(shape)
'''

SEEDS = {
    "seed_quant8": ("quant", _QUANT % (8, 8)),
    "seed_quant4": ("quant", _QUANT % (4, 4)),
    "seed_quant2": ("quant", _QUANT % (2, 2)),
    "seed_quant1": ("quant", _QUANT1),
    "seed_kmeans4": ("kmeans", _KMEANS),
    "seed_lowrank": ("lowrank", _LOWRANK),
    "seed_lowrank_r8": ("lowrank", _LOWRANK_ABS % (8, 8)),
    "seed_lowrank_r2": ("lowrank", _LOWRANK_ABS % (2, 2)),
    "seed_lowrank_r1": ("lowrank", _LOWRANK_ABS % (1, 1)),
    "seed_magsparse": ("sparse", _MAGSPARSE),
}

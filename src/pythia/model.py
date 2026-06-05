"""Pythia-1.4B (GPT-NeoX) victim: checkpoint loading + the compressible-weight / activation map.

Why Pythia: the Pile is public (+exact training order), and 154 evenly-spaced checkpoints give us
an undertrained-target ladder for FREE — victim = a late checkpoint, target = an earlier one — so we
never train a 1.5B from scratch and we sidestep the converged-knife-edge. ~300B-token denominator
keeps recovery cheap.

GPT-NeoX specifics handled here:
  * fused QKV (attention.query_key_value, 3d x d) is one 2D matrix — fine to quantize as-is;
  * embed_in and embed_out are SEPARATE (not tied) — both compressed;
  * LayerNorm weights AND linear biases are 1D -> kept dense (stolen verbatim), never compressed.
"""
import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "EleutherAI/pythia-1.4b"
TOKENS_PER_STEP = 1024 * 2048   # Pythia batch: 1024 seqs x 2048 ctx ~= 2.1M tokens/step
TOTAL_STEPS = 143000            # -> ~300B tokens


def load_model(revision="step143000", device="cpu", dtype=torch.float32):
    m = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=revision, torch_dtype=dtype)
    return m.to(device)


def get_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_ID)


def compressible_names(model):
    """2D weight matrices: embed_in, embed_out, and per-layer qkv/dense/h_to_4h/4h_to_h."""
    return [n for n, p in model.named_parameters() if p.ndim == 2]


def get_weights(model):
    names = set(compressible_names(model))
    return {n: p.detach().to("cpu", torch.float32).numpy().copy()
            for n, p in model.named_parameters() if n in names}


@torch.no_grad()
def set_weights(model, weights):
    params = dict(model.named_parameters())
    for n, arr in weights.items():
        p = params[n]
        t = torch.from_numpy(np.ascontiguousarray(arr)).to(p.device, p.dtype)
        assert tuple(t.shape) == tuple(p.shape), f"{n}: {t.shape} vs {p.shape}"
        p.copy_(t)


def fp32_bytes(weights):
    return int(sum(a.size for a in weights.values()) * 4)


def linear_module_for_weight(model):
    """Map each compressible weight name -> the nn.Linear module whose INPUT is its activation
    Hessian (for activation-aware compression). embed_in is a lookup (no activation) -> None;
    embed_out (the LM head) uses the final-layernorm output."""
    linears = {n: m for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)}
    out = {}
    for wn in compressible_names(model):
        mod = wn[:-len(".weight")]
        if mod in linears:
            out[wn] = mod
        else:
            out[wn] = None   # embed_in (lookup) — no input-activation Hessian
    return out


@torch.no_grad()
def eval_loss(model, batcher, device, max_batches=None):
    model.eval()
    tot, ntok = 0.0, 0
    for i, x in enumerate(batcher.iter_eval()):
        if max_batches is not None and i >= max_batches:
            break
        out = model(input_ids=x, labels=x)
        nt = x.numel() - x.shape[0]
        tot += out.loss.item() * nt; ntok += nt
    model.train()
    return tot / max(ntok, 1)

"""The 11M SimpleStories Llama, plus the compressible-weight interface that lets the
(model-agnostic) compression candidates operate on this model: extract every 2D weight
matrix as a numpy array, and write reconstructed arrays back. Tied embeddings are handled
by writing embed_tokens once (lm_head follows automatically).
"""
import numpy as np
import torch
from transformers import AutoConfig, LlamaForCausalLM

MODEL_ID = "SimpleStories/SimpleStories-11M"


def build_config():
    return AutoConfig.from_pretrained(MODEL_ID)


def build_model(device="cpu", seed=0):
    """Fresh random-init model matching the released 11M architecture."""
    torch.manual_seed(seed)
    cfg = build_config()
    model = LlamaForCausalLM(cfg)
    return model.to(device)


def compressible_names(model):
    """Names of the 2D weight matrices we compress (attn+MLP projections + tied embedding).
    RMSNorm weights are 1D and kept dense; tied params appear once."""
    return [n for n, p in model.named_parameters() if p.ndim == 2]


def get_weights(model):
    """name -> float32 numpy array (cpu) for every compressible 2D matrix."""
    names = set(compressible_names(model))
    out = {}
    sd = model.state_dict()
    for n, p in model.named_parameters():
        if n in names:
            out[n] = p.detach().to("cpu", torch.float32).numpy().copy()
    return out


@torch.no_grad()
def set_weights(model, weights):
    """Write {name: numpy array} back into the model in place (handles tied embeddings)."""
    params = dict(model.named_parameters())
    for n, arr in weights.items():
        p = params[n]
        t = torch.from_numpy(np.ascontiguousarray(arr)).to(p.device, p.dtype)
        assert tuple(t.shape) == tuple(p.shape), f"{n}: {t.shape} vs {p.shape}"
        p.copy_(t)


def fp32_bytes(weights):
    """Total fp32 byte count of the compressible weights (denominator for compression ratio)."""
    return int(sum(a.size for a in weights.values()) * 4)


@torch.no_grad()
def eval_loss(model, batcher, device, max_batches=None):
    """Mean next-token CE over the eval set (deterministic full pass, optionally truncated)."""
    model.eval()
    tot, ntok = 0.0, 0
    for i, x in enumerate(batcher.iter_eval()):
        if max_batches is not None and i >= max_batches:
            break
        out = model(input_ids=x, labels=x)
        # out.loss is mean over (seqlen-1)*bs tokens; weight by token count for an exact mean
        ntoks = x.numel() - x.shape[0]
        tot += out.loss.item() * ntoks; ntok += ntoks
    model.train()
    return tot / max(ntok, 1)

"""Naive-attacker baselines on SmolLM2-360M: per-output-row uniform quantization (round-to-nearest)
of the transformer-BLOCK linears (embeddings/lm_head/norms kept dense, matching AQLM), evaluated
with AQLM's OWN evaluate_perplexity (get_loaders, seqlen 2048) so every point in the final plot is
on the identical eval. Honest block bits/param includes the per-row scale+zero overhead.
Output: runs/simple_eval.json
"""
import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM
from src.datautils import get_loaders, evaluate_perplexity

MP = "/workspace/models/smollm2-360m"
SEQLEN = 2048
dev = torch.device("cuda")


def block_linear_names(model):
    return [n for n, p in model.named_parameters()
            if p.ndim == 2 and "layers." in n and "embed" not in n and "lm_head" not in n]


def uniform_quant_per_row(w, bits):
    """Per-output-row min/max uniform quant -> dequantized weight + honest bits/param."""
    qmax = (1 << bits) - 1
    wmin = w.amin(dim=1, keepdim=True)
    wmax = w.amax(dim=1, keepdim=True)
    scale = (wmax - wmin).clamp_min(1e-8) / qmax
    q = torch.round((w - wmin) / scale).clamp_(0, qmax)
    deq = q * scale + wmin
    out, inn = w.shape
    # codes: out*inn*bits ; per-row scale+zero: out * 2 * 16 bits
    honest_bits = (out * inn * bits + out * 2 * 16) / (out * inn)
    return deq.to(w.dtype), honest_bits


@torch.no_grad()
def eval_model(model):
    res = {}
    for ds in ["wikitext2", "c4"]:
        data = get_loaders(ds, eval_mode=True, seqlen=SEQLEN, model_path=MP, trust_remote_code=True)
        res[ds] = evaluate_perplexity(model, data, SEQLEN, dev)
    return res


def run(bits):
    model = AutoModelForCausalLM.from_pretrained(MP, torch_dtype=torch.float32, local_files_only=True).to(dev).eval()
    names = block_linear_names(model)
    params = dict(model.named_parameters())
    tot_bits, tot_params = 0.0, 0
    for n in names:
        w = params[n].data
        deq, hb = uniform_quant_per_row(w, bits)
        params[n].data.copy_(deq)
        tot_bits += hb * w.numel(); tot_params += w.numel()
    bpp = tot_bits / tot_params
    ppl = eval_model(model)
    del model; torch.cuda.empty_cache()
    print(f"uniform-{bits}bit  block_bpp={bpp:.3f}  wikitext2={ppl['wikitext2']:.3f}  c4={ppl['c4']:.3f}", flush=True)
    return {"method": f"uniform_{bits}bit", "bpp_block": bpp,
            "wikitext2": ppl["wikitext2"], "c4": ppl["c4"]}


def main():
    out = []
    # baseline (no quant) sanity
    m = AutoModelForCausalLM.from_pretrained(MP, torch_dtype=torch.float32, local_files_only=True).to(dev).eval()
    b = eval_model(m); del m; torch.cuda.empty_cache()
    print(f"baseline  wikitext2={b['wikitext2']:.3f}  c4={b['c4']:.3f}", flush=True)
    out.append({"method": "baseline", "bpp_block": 16.0, "wikitext2": b["wikitext2"], "c4": b["c4"]})
    for bits in [8, 4, 3, 2]:
        out.append(run(bits))
    json.dump(out, open("runs/simple_eval.json", "w"), indent=1)
    print("wrote runs/simple_eval.json", flush=True)


if __name__ == "__main__":
    main()

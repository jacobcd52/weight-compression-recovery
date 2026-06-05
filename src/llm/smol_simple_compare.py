"""Apply the simple per-tensor compressors (quant/kmeans/group-NF2) to SmolLM2-360M's TRANSFORMER-
BLOCK linears (embeddings kept dense, matching AQLM), and eval WikiText-2 perplexity the same way
AQLM does (seqlen-2048 windows, mean nll). Honest ratio is over the block weights only, so it's a
fair head-to-head with AQLM's block ratio. Output: runs/aqlm_frontier/smol_simple.json

    python -m src.llm.smol_simple_compare
"""
import json, os, pickle, zlib
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ..autoresearch.seeds import SEEDS

MODEL = "/workspace/models/smollm2-360m"
SEQLEN = 2048
DEV = torch.device("cuda")


def block_linear_names(model):
    return [n for n, p in model.named_parameters()
            if p.ndim == 2 and "layers." in n and "embed" not in n and "lm_head" not in n]


@torch.no_grad()
def wikitext2_ppl(model):
    from datasets import load_dataset
    tok = AutoTokenizer.from_pretrained(MODEL)
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    enc = tok("\n\n".join(test["text"]), return_tensors="pt").input_ids[0]
    n = enc.numel() // SEQLEN
    nll, ntok = 0.0, 0
    for i in range(n):
        x = enc[i*SEQLEN:(i+1)*SEQLEN].unsqueeze(0).to(DEV)
        out = model(input_ids=x, labels=x)
        nll += out.loss.item() * (SEQLEN - 1); ntok += SEQLEN - 1
    return float(np.exp(nll / ntok))


def compress_eval(method, code):
    ns = {"np": np, "math": __import__("math"), "__builtins__": __builtins__}
    exec(code, ns); knobs = ns.get("KNOBS", {}); ct, rt = ns["compress_tensor"], ns["reconstruct_tensor"]
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).to(DEV)
    names = block_linear_names(model); params = dict(model.named_parameters())
    cbytes = 0; fp32 = 0
    for n in names:
        w = params[n].detach().cpu().float().numpy()
        pl = ct(w, **knobs)
        cbytes += len(zlib.compress(pickle.dumps(pl, protocol=4), level=6))
        rec = np.asarray(rt(pl, tuple(w.shape)), dtype=np.float32)
        params[n].copy_(torch.from_numpy(rec).to(DEV))
        fp32 += w.size * 4
    ppl = wikitext2_ppl(model)
    del model; torch.cuda.empty_cache()
    ratio = cbytes / fp32
    print(f"  {method:14} block-ratio={ratio:.4f} ({ratio*32:.2f}bpp) wikitext2_ppl={ppl:.3f}", flush=True)
    return {"method": method, "ratio_block": ratio, "bpp_block": ratio * 32, "wikitext2_ppl": ppl}


def main():
    # original (no compression) reference
    m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).to(DEV)
    base = wikitext2_ppl(m); del m; torch.cuda.empty_cache()
    print(f"SmolLM2-360M original wikitext2 ppl = {base:.3f}", flush=True)
    results = [{"method": "original", "ratio_block": 1.0, "bpp_block": 32, "wikitext2_ppl": base}]
    methods = ["seed_quant8", "seed_quant4", "seed_quant2", "seed_kmeans4", "seed_quant1"]
    codes = {k: SEEDS[k][1] for k in methods if k in SEEDS}
    arch = "runs/autoresearch/archive.json"
    if os.path.exists(arch):
        e = json.load(open(arch))["entries"]; rec = [x for x in e if x.get("recovered")]
        if rec:
            b = min(rec, key=lambda x: x["ratio"]); codes[f"groupNF2_{b['name']}"] = b["code"]
    for name, code in codes.items():
        try:
            results.append(compress_eval(name, code))
        except Exception as ex:
            print(f"  {name}: FAIL {str(ex)[:100]}", flush=True)
    os.makedirs("runs/aqlm_frontier", exist_ok=True)
    json.dump(results, open("runs/aqlm_frontier/smol_simple.json", "w"), indent=1)


if __name__ == "__main__":
    main()

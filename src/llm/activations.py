"""Compute & cache the per-linear-layer input Hessian  H = (1/N) sum_t x_t x_t^T  for the victim,
on a calibration set. This is the GPTQ/AQLM activation statistic: the quantization error of weight
column j is weighted by H (it captures which input directions actually matter), so activation-aware
schemes minimise OUTPUT error rather than raw weight MSE. Computed once and reused by every
candidate (the attacker is inside the data center and has the activations).

    python -m src.llm.activations --calib-batches 64
"""
import argparse
import os

import torch

from .data import Batcher, load_packed
from .model import build_config, compressible_names


def compute_hessians(victim_pt, cache_dir, calib_batches=64, batch_size=16, damp=0.01, device=None):
    from transformers import LlamaForCausalLM
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sd = torch.load(victim_pt, map_location="cpu", weights_only=False)["state_dict"]
    model = LlamaForCausalLM(build_config()); model.load_state_dict(sd); model.to(device).eval()
    train, _, meta = load_packed(cache_dir)
    tb = Batcher(train, batch_size, device, seed=777)

    # map each compressible weight name -> the nn.Linear module whose INPUT we hook.
    # the tied embedding is quantization-sensitive via the output projection (lm_head), so we use
    # the lm_head input (final hidden state) as its Hessian.
    linears = {name: mod for name, mod in model.named_modules()
               if isinstance(mod, torch.nn.Linear)}
    wname_to_mod = {}
    for wn in compressible_names(model):
        mod = wn[:-len(".weight")]
        if mod in linears:
            wname_to_mod[wn] = mod
        elif wn == "model.embed_tokens.weight" and "lm_head" in linears:
            wname_to_mod[wn] = "lm_head"

    H = {wn: None for wn in wname_to_mod}
    counts = {wn: 0 for wn in wname_to_mod}
    mod_to_wn = {m: wn for wn, m in wname_to_mod.items()}
    handles = []

    def mk_hook(mod_name):
        def hook(module, inp):
            x = inp[0].detach()
            x = x.reshape(-1, x.shape[-1]).float()        # (tokens, in_features)
            wn = mod_to_wn[mod_name]
            g = x.t() @ x                                  # (in, in)
            H[wn] = g if H[wn] is None else H[wn] + g
            counts[wn] += x.shape[0]
        return hook

    for mname in set(wname_to_mod.values()):
        handles.append(linears[mname].register_forward_pre_hook(mk_hook(mname)))

    with torch.no_grad():
        for i in range(calib_batches):
            model(input_ids=tb.batch())
    for h in handles:
        h.remove()

    out = {}
    for wn, g in H.items():
        if g is None:
            continue
        Hn = (g / max(counts[wn], 1)).cpu()
        # GPTQ-style damping for numerical stability when used as a weighting / inverse
        d = damp * torch.diag(Hn).mean().clamp_min(1e-8)
        Hn = Hn + d * torch.eye(Hn.shape[0])
        out[wn] = Hn.to(torch.float32)

    os.makedirs(cache_dir if os.path.isdir(cache_dir) else os.path.dirname(victim_pt), exist_ok=True)
    path = os.path.join(os.path.dirname(victim_pt), "hessians.pt")
    torch.save({"H": out, "calib_tokens": int(sum(counts.values()) / max(len(counts), 1)),
                "damp": damp}, path)
    tot = sum(v.numel() for v in out.values()) * 4 / 1e6
    print(f"[hessians] {len(out)} layers, ~{tot:.0f} MB, "
          f"~{int(sum(counts.values())/max(len(counts),1))} calib tokens/layer -> {path}", flush=True)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--victim-pt", default="runs/llm_victim/victim.pt")
    ap.add_argument("--cache-dir", default="data/simplestories")
    ap.add_argument("--calib-batches", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--damp", type=float, default=0.01)
    args = ap.parse_args()
    compute_hessians(args.victim_pt, args.cache_dir, args.calib_batches, args.batch_size, args.damp)


if __name__ == "__main__":
    main()

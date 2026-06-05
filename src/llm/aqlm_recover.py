"""Take an AQLM-dequantized dense state_dict (from AQLM/reconstruct_aqlm.py), place it as the
recovery init, and produce a frontier point: (honest ratio, amortized ratio, recon loss, recovery
cost) using the SAME recovery setup as the 11M seed frontier (2250-ckpt target, 30% budget).

    python -m src.llm.aqlm_recover --dense /workspace/models/aqlm_2x8_dense.pt --nominal-bpp 2.0 --tag aqlm_2x8
"""
import argparse, json, os
import numpy as np, torch
from transformers import LlamaForCausalLM
from .model import build_config, eval_loss, compressible_names
from .data import load_packed, Batcher
from .recover import _fresh_from, evaluate_recovery

DEV = torch.device("cuda")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense", required=True); ap.add_argument("--nominal-bpp", type=float, required=True)
    ap.add_argument("--tag", required=True); ap.add_argument("--budget-fraction", type=float, default=0.30)
    args = ap.parse_args()
    d = torch.load(args.dense, map_location="cpu", weights_only=False)
    dense, nbits = d["dense"], d["nbits"]
    victim_sd = torch.load("runs/llm_victim/victim.pt", map_location="cpu", weights_only=False)["state_dict"]
    summ = json.load(open("runs/llm_victim/summary.json"))
    train, ev, meta = load_packed("data/simplestories")
    eb = Batcher(ev, 64, DEV, seed=1); tb = Batcher(train, 64, DEV, seed=123)
    c = next(c for c in summ["checkpoints"] if c["step"] == 2250)
    tgt_sd = torch.load(os.path.join("runs/llm_victim", c["path"]), map_location="cpu", weights_only=False)["state_dict"]
    target = eval_loss(_fresh_from(tgt_sd, None, DEV), eb, DEV, max_batches=32)
    torch.cuda.empty_cache()

    # AQLM-quantized model = AQLM reconstructions where available, victim's elsewhere (tied keys)
    aqlm_sd = {k: (dense[k] if k in dense else victim_sd[k]) for k in victim_sd}
    recon_loss = eval_loss(_fresh_from(aqlm_sd, None, DEV), eb, DEV, max_batches=32)
    torch.cuda.empty_cache()

    # ratios over the compressible 2D set (42 AQLM-quantized blocks + the dense fp16 embedding)
    m = LlamaForCausalLM(build_config()); cset = set(compressible_names(m))
    allp = sum(victim_sd[k].numel() for k in cset)
    block_bits = sum(nbits[k] * victim_sd[k].numel() for k in nbits if k in victim_sd)
    embed_p = allp - sum(victim_sd[k].numel() for k in nbits if k in victim_sd)   # the un-quantized embed
    ratio_honest = (block_bits + embed_p * 16) / (32 * allp)
    ratio_amort = args.nominal_bpp / 32.0

    # recovery: retrain the AQLM-quantized init toward the 2250-ckpt target
    res = evaluate_recovery(None, aqlm_sd, target_loss=target, denom_steps=2250,
                            budget_steps=int(round(args.budget_fraction * 2250)),
                            lr_grid=[1e-4, 3e-4, 1e-3, 3e-3], warmup=20, eval_every=25,
                            eval_batches=32, train_batcher=tb, eval_batcher=eb, device=DEV)
    out = dict(tag=args.tag, nominal_bpp=args.nominal_bpp, quant_avg_bits=block_bits / max(1, allp - embed_p),
               ratio_honest=ratio_honest, ratio_amortized=ratio_amort, recon_loss=recon_loss,
               target=target, recovered=res["recovered"], recovery_fraction=res["recovery_fraction"],
               step0_loss=res["step0_loss"])
    os.makedirs("runs/aqlm_frontier", exist_ok=True)
    json.dump(out, open(f"runs/aqlm_frontier/{args.tag}.json", "w"), indent=1)
    print(f"[{args.tag}] bpp={args.nominal_bpp} ratio_honest={ratio_honest:.4f} ratio_amort={ratio_amort:.4f} "
          f"recon_loss={recon_loss:.3f} -> " +
          (f"REC@{res['recovery_fraction']*100:.2f}%" if res["recovered"] else "DNR"))


if __name__ == "__main__":
    main()

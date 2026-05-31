"""Compute exact bpp breakdown for each route."""
import os, sys, json, torch
sys.path.insert(0, "/workspace/projects/CRVQ")
from src.aq import QuantizedLinear

routes = [("X2_g8_mr0.025", 2, 8, 0.025),
          ("X4_g16_mr0.05", 4, 16, 0.05),
          ("X8_g32_mr0.1", 8, 32, 0.1),
          ("X1_g4_mr0.025", 1, 4, 0.025)]
OUT = {}
for tag, X, g, mr in routes:
    DIR = f"/workspace/projects/CRVQ/runs/route_{tag}"
    code_bits = 0; codebook_bits = 0; scale_bits = 0; total_params = 0
    for i in range(6):
        block = torch.load(os.path.join(DIR, f"{i}.pth"), map_location="cpu", weights_only=False)
        for name, mod in block.named_modules():
            if isinstance(mod, QuantizedLinear):
                qw = mod.quantized_weight
                nparams = qw.out_features * qw.in_features
                total_params += nparams
                num_in_groups = qw.in_features // qw.in_group_size
                num_out_groups = qw.out_features // qw.out_group_size
                group_size = qw.in_group_size * qw.out_group_size
                group_bound = int(qw.in_features * qw.multibook_ratio // qw.in_group_size)
                base_bits = qw.nbits_per_codebook[0]
                crit_bits = sum(qw.nbits_per_codebook[1:])
                code_bits += num_out_groups * num_in_groups * base_bits
                code_bits += num_out_groups * group_bound * crit_bits
                for nb in qw.nbits_per_codebook:
                    codebook_bits += (2**nb) * group_size * 16
                if hasattr(qw, "scales") and qw.scales is not None:
                    scale_bits += qw.scales.numel() * 16
    OUT[tag] = {"X": X, "g": g, "mr": mr,
                "code_bpp": code_bits / total_params,
                "codebook_bpp": codebook_bits / total_params,
                "scale_bpp": scale_bits / total_params,
                "asymp_bpp": code_bits / total_params,
                "honest_bpp": (code_bits + codebook_bits + scale_bits) / total_params}
json.dump(OUT, open("/workspace/projects/CRVQ/runs/route_bpp.json", "w"), indent=2)
print(f"{'tag':<20} {'code':>7} {'cbook':>7} {'scale':>7} {'honest':>7}")
for t, d in OUT.items():
    print(f"{t:<20} {d['code_bpp']:>7.4f} {d['codebook_bpp']:>7.4f} {d['scale_bpp']:>7.4f} {d['honest_bpp']:>7.4f}")

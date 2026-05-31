"""Compute exact bpp breakdown (codes / codebooks / scales) for each CRVQ Phase-1 point
by loading the saved per-layer .pth files."""
import os, torch, json
torch.serialization.add_safe_globals = lambda *a, **k: None
import sys; sys.path.insert(0, "/workspace/projects/CRVQ")

from src.aq import QuantizedLinear

OUT = {}
for X in [2, 3, 4, 6, 8]:
    DIR = f"/workspace/projects/CRVQ/runs/bpp_X{X}_X444"
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
                # Codes: base codebook (all groups) + critical codebooks (group_bound per output)
                base_bits = qw.nbits_per_codebook[0]
                crit_bits = sum(qw.nbits_per_codebook[1:])
                code_bits += num_out_groups * num_in_groups * base_bits
                code_bits += num_out_groups * group_bound * crit_bits
                # Codebooks: each codebook = 2^nbits × group_size × 16 fp16
                for nb in qw.nbits_per_codebook:
                    codebook_bits += (2**nb) * group_size * 16
                # Scales: per-output-group, fp16
                if hasattr(qw, "scales") and qw.scales is not None:
                    scale_bits += qw.scales.numel() * 16
    OUT[X] = {"code_bits": code_bits, "codebook_bits": codebook_bits, "scale_bits": scale_bits,
              "total_params": total_params,
              "code_bpp": code_bits / total_params,
              "codebook_bpp": codebook_bits / total_params,
              "scale_bpp": scale_bits / total_params,
              "total_bpp": (code_bits + codebook_bits + scale_bits) / total_params}
json.dump(OUT, open("/workspace/projects/CRVQ/runs/bpp_breakdown.json", "w"), indent=2)
print(f"{'X':>3} {'code':>8} {'cbook':>8} {'scale':>8} {'total':>8} {'ratio':>8}")
for x, d in OUT.items():
    print(f"{x:>3} {d['code_bpp']:>8.4f} {d['codebook_bpp']:>8.4f} {d['scale_bpp']:>8.4f} "
          f"{d['total_bpp']:>8.4f} {d['total_bpp']/16:>8.4f}")

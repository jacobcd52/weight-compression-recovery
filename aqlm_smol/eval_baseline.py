"""Baseline perplexity of the uncompressed SmolLM2-360M, using AQLM's exact eval
(src.datautils.get_loaders + evaluate_perplexity, seqlen 2048) so it's directly
comparable to the stage-1 / stage-2 quantized numbers."""
import torch
from transformers import AutoModelForCausalLM
from src.datautils import get_loaders, evaluate_perplexity

MP = "/workspace/models/smollm2-360m"
dev = torch.device("cuda")
model = AutoModelForCausalLM.from_pretrained(
    MP, torch_dtype=torch.bfloat16, local_files_only=True
).to(dev).eval()
for ds in ["wikitext2", "c4"]:
    data = get_loaders(ds, eval_mode=True, seqlen=2048, model_path=MP, trust_remote_code=True)
    ppl = evaluate_perplexity(model, data, 2048, dev)
    print(f"BASELINE {ds} ppl = {ppl:.4f}", flush=True)

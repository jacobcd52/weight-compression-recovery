"""Pre-tokenize a single (cached) c4 shard with the SmolLM2 tokenizer into 2048-token blocks,
and save_to_disk so finetune.py can load it directly (--dataset_name <path> --split none).
finetune.py treats a dataset with an 'input_ids' column as already-tokenized and asserts each
row has length == model_seqlen, so we emit exactly-2048 blocks."""
import sys
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

MODEL = "/workspace/models/smollm2-360m"
SEQLEN = 2048
N_DOCS = int(sys.argv[1]) if len(sys.argv) > 1 else 12000
OUT = sys.argv[2] if len(sys.argv) > 2 else "/workspace/projects/AQLM/c4_smol_ft_ds"

tok = AutoTokenizer.from_pretrained(MODEL)
# single cached shard (same one get_c4 uses) -> no full c4 download
raw = load_dataset("allenai/c4", "default",
                   data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
                   split="train",
                   revision="607bd4c8450a42878aa9ddc051a65a055450ef87")
raw = raw.select(range(min(N_DOCS, len(raw))))
print(f"loaded {len(raw)} c4 docs", flush=True)

# concatenate all token ids, then chop into exact SEQLEN blocks
ids = []
for i, t in enumerate(raw["text"]):
    ids.extend(tok(t).input_ids)
    ids.append(tok.eos_token_id if tok.eos_token_id is not None else tok.bos_token_id)
    if (i + 1) % 2000 == 0:
        print(f"  tokenized {i+1}/{len(raw)} docs, {len(ids)} tokens", flush=True)
n_blocks = len(ids) // SEQLEN
blocks = [ids[j * SEQLEN:(j + 1) * SEQLEN] for j in range(n_blocks)]
print(f"{n_blocks} blocks of {SEQLEN} tokens ({n_blocks*SEQLEN} tokens total)", flush=True)

ds = Dataset.from_dict({"input_ids": blocks})
ds.save_to_disk(OUT)
print(f"saved -> {OUT}  (rows={len(ds)}, len[0]={len(ds[0]['input_ids'])})", flush=True)

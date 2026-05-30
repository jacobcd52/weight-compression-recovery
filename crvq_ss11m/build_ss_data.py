"""Build CRVQ-format calibration + finetune .pth files for SimpleStories-11M.
CRVQ's get_loaders accepts a file path -> loads via torch.load and takes [:nsamples].
The expected format is a list of [1, seqlen] integer tensors (matching what get_red_pajama
constructs at runtime). We reuse the SimpleStories train.u16 stream + meta.json.

Outputs:
  ss11m_crvq_calib.pth  : 2048 + 256 = 2304 sequences (paper main.py: nsamples + val_size)
  ss11m_crvq_ft.pth     : 1024 + 64 = 1088 sequences (paper finetune.py: nsamples + val_size)
Sequences are non-overlapping windows from the start of train.u16 (deterministic, reproducible).
"""
import json, os, numpy as np, torch

DATA_DIR = "/workspace/projects/weight-compression-recovery/data/simplestories"
OUT_DIR = "/workspace/projects/CRVQ/data"
os.makedirs(OUT_DIR, exist_ok=True)
meta = json.load(open(os.path.join(DATA_DIR, "meta.json")))
SEQLEN = meta["seqlen"]; print(f"seqlen={SEQLEN}, vocab={meta['vocab_size']}, train_seqs={meta['train_shape'][0]}")

train = np.fromfile(os.path.join(DATA_DIR, "train.u16"), dtype=np.uint16)
N_total = train.size // SEQLEN
train = train[:N_total * SEQLEN].reshape(N_total, SEQLEN)
print(f"loaded {N_total} sequences of {SEQLEN} tokens from train.u16")


def save_chunk(start, count, out_path):
    chunk = train[start:start + count].astype(np.int64)
    lst = [torch.from_numpy(chunk[i:i + 1]) for i in range(count)]  # list of [1, seqlen]
    torch.save(lst, out_path)
    print(f"  wrote {out_path}  rows={len(lst)}, shape[0]={tuple(lst[0].shape)}, dtype={lst[0].dtype}")


# Stage-1 (main.py): nsamples=2048, val_size=256
save_chunk(0, 2304, os.path.join(OUT_DIR, "ss11m_crvq_calib.pth"))
# Stage-2 (finetune.py): nsamples=1024, val_size=64.  Use a DIFFERENT slice so train/finetune don't overlap.
save_chunk(2304, 1088, os.path.join(OUT_DIR, "ss11m_crvq_ft.pth"))
print("done")

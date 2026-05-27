"""SimpleStories data: tokenize once, pack into fixed-length sequences, cache as
uint16 memmaps on the volume. The packed arrays are reused across the from-scratch
reference run and every recovery retrain, so tokenization happens only once.

Prepare (one-off, cached):
    python -m src.llm.data --train-tokens 100_000_000 --eval-tokens 2_000_000
"""
import argparse
import json
import os

import numpy as np
from transformers import AutoTokenizer

MODEL_ID = "SimpleStories/SimpleStories-11M"
DEFAULT_CACHE = "data/simplestories"
SEQLEN = 512


def _meta_path(cache_dir):
    return os.path.join(cache_dir, "meta.json")


def _pack_split(tok, split, target_tokens, seqlen, out_path):
    """Stream a split, tokenize stories, separate with EOS, pack into [n, seqlen] uint16."""
    from datasets import load_dataset
    eos = tok.eos_token_id if tok.eos_token_id is not None else (tok.bos_token_id or 0)
    ds = load_dataset(MODEL_ID.split("/")[0] + "/SimpleStories", split=split, streaming=True)
    buf = []          # flat list of token ids
    seqs = []         # list of np.uint16 arrays of length seqlen
    batch, n_tok, n_story = [], 0, 0
    def flush_batch():
        nonlocal buf, n_tok
        enc = tok(batch)["input_ids"]
        for ids in enc:
            buf.extend(ids); buf.append(eos); n_tok += len(ids) + 1
        batch.clear()
    for ex in ds:
        batch.append(ex["story"]); n_story += 1
        if len(batch) >= 1000:
            flush_batch()
            # carve full sequences out of buf to keep memory bounded
            while len(buf) >= seqlen:
                seqs.append(np.asarray(buf[:seqlen], dtype=np.uint16)); del buf[:seqlen]
            if n_tok >= target_tokens:
                break
        if n_story % 200000 == 0:
            print(f"  [{split}] {n_story} stories, {n_tok/1e6:.1f}M tokens, {len(seqs)} seqs", flush=True)
    if batch:
        flush_batch()
    while len(buf) >= seqlen:
        seqs.append(np.asarray(buf[:seqlen], dtype=np.uint16)); del buf[:seqlen]
    arr = np.stack(seqs)
    mm = np.memmap(out_path, dtype=np.uint16, mode="w+", shape=arr.shape)
    mm[:] = arr; mm.flush()
    print(f"  [{split}] wrote {arr.shape} -> {out_path} ({arr.size*2/1e6:.0f} MB, {n_story} stories)",
          flush=True)
    return arr.shape


def prepare(train_tokens, eval_tokens, seqlen=SEQLEN, cache_dir=DEFAULT_CACHE, force=False):
    os.makedirs(cache_dir, exist_ok=True)
    meta_p = _meta_path(cache_dir)
    if os.path.exists(meta_p) and not force:
        meta = json.load(open(meta_p))
        if meta.get("train_tokens") == train_tokens and meta.get("eval_tokens") == eval_tokens \
           and meta.get("seqlen") == seqlen:
            print(f"[data] cache hit: {meta['train_shape']} train / {meta['eval_shape']} eval")
            return meta
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    print(f"[data] tokenizing (vocab={tok.vocab_size}, eos={tok.eos_token_id}) ...", flush=True)
    train_shape = _pack_split(tok, "train", train_tokens, seqlen,
                              os.path.join(cache_dir, "train.u16"))
    eval_shape = _pack_split(tok, "test", eval_tokens, seqlen,
                             os.path.join(cache_dir, "eval.u16"))
    meta = dict(train_tokens=train_tokens, eval_tokens=eval_tokens, seqlen=seqlen,
                train_shape=list(train_shape), eval_shape=list(eval_shape),
                vocab_size=tok.vocab_size, eos=tok.eos_token_id, model_id=MODEL_ID)
    json.dump(meta, open(meta_p, "w"), indent=1)
    print(f"[data] done: {train_shape} train / {eval_shape} eval")
    return meta


def load_packed(cache_dir=DEFAULT_CACHE):
    meta = json.load(open(_meta_path(cache_dir)))
    train = np.memmap(os.path.join(cache_dir, "train.u16"), dtype=np.uint16, mode="r",
                      shape=tuple(meta["train_shape"]))
    ev = np.memmap(os.path.join(cache_dir, "eval.u16"), dtype=np.uint16, mode="r",
                   shape=tuple(meta["eval_shape"]))
    return train, ev, meta


class Batcher:
    """Random minibatches of packed sequences -> (input_ids, labels) int64 tensors on device.
    labels == input_ids (LlamaForCausalLM shifts internally). Sampling is with-replacement over
    a per-epoch shuffle, which is fine for our single-pass undertrained regime."""
    def __init__(self, arr, batch_size, device, seed=0):
        self.arr = arr; self.bs = batch_size; self.device = device
        self.rng = np.random.default_rng(seed); self.n = arr.shape[0]

    def batch(self):
        import torch
        idx = self.rng.integers(0, self.n, size=self.bs)
        x = torch.from_numpy(np.asarray(self.arr[idx], dtype=np.int64))
        return x.to(self.device, non_blocking=True)

    def iter_eval(self, batch_size=None):
        """Deterministic full pass over the (eval) array in order, for eval loss."""
        import torch
        bs = batch_size or self.bs
        for i in range(0, self.n, bs):
            idx = np.arange(i, min(i + bs, self.n))
            yield torch.from_numpy(np.asarray(self.arr[idx], dtype=np.int64)).to(self.device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-tokens", type=int, default=100_000_000)
    ap.add_argument("--eval-tokens", type=int, default=2_000_000)
    ap.add_argument("--seqlen", type=int, default=SEQLEN)
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    prepare(args.train_tokens, args.eval_tokens, args.seqlen, args.cache_dir, args.force)


if __name__ == "__main__":
    main()

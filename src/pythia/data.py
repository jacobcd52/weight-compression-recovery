"""Pile data for Pythia retraining/recovery + Hessian calibration. Streams a public Pile mirror,
tokenizes with the GPT-NeoX tokenizer (Pythia's), packs into ctx-2048 uint16 sequences, caches a
slice on the volume. The attacker "has the training data", so we retrain on the pretraining
distribution (the Pile).

    python -m src.pythia.data --train-tokens 300_000_000 --eval-tokens 5_000_000
"""
import argparse
import json
import os

import numpy as np

from .model import MODEL_ID, get_tokenizer

# public Pile mirror (the original EleutherAI/pile is gated); same distribution Pythia trained on
PILE = "monology/pile-uncopyrighted"
DEFAULT_CACHE = "data/pile"
SEQLEN = 2048


def _meta(cache_dir):
    return os.path.join(cache_dir, "meta.json")


def _pack_split(tok, ds_iter, target_tokens, seqlen, out_path, label):
    eos = tok.eos_token_id if tok.eos_token_id is not None else 0
    buf, seqs, batch, n_tok, n_doc = [], [], [], 0, 0

    def flush():
        nonlocal n_tok
        enc = tok(batch)["input_ids"]
        for ids in enc:
            buf.extend(ids); buf.append(eos); n_tok += len(ids) + 1
        batch.clear()

    for ex in ds_iter:
        batch.append(ex["text"]); n_doc += 1
        if len(batch) >= 256:
            flush()
            while len(buf) >= seqlen:
                seqs.append(np.asarray(buf[:seqlen], dtype=np.uint16)); del buf[:seqlen]
            if n_tok >= target_tokens:
                break
        if n_doc % 50000 == 0:
            print(f"  [{label}] {n_doc} docs, {n_tok/1e6:.1f}M tokens, {len(seqs)} seqs", flush=True)
    if batch:
        flush()
    while len(buf) >= seqlen:
        seqs.append(np.asarray(buf[:seqlen], dtype=np.uint16)); del buf[:seqlen]
    arr = np.stack(seqs)
    mm = np.memmap(out_path, dtype=np.uint16, mode="w+", shape=arr.shape)
    mm[:] = arr; mm.flush()
    print(f"  [{label}] wrote {arr.shape} -> {out_path} ({arr.size*2/1e9:.2f} GB, {n_doc} docs)", flush=True)
    return arr.shape


def prepare(train_tokens, eval_tokens, seqlen=SEQLEN, cache_dir=DEFAULT_CACHE, force=False):
    os.makedirs(cache_dir, exist_ok=True)
    mp = _meta(cache_dir)
    if os.path.exists(mp) and not force:
        meta = json.load(open(mp))
        if meta.get("train_tokens") == train_tokens and meta.get("eval_tokens") == eval_tokens \
           and meta.get("seqlen") == seqlen:
            print(f"[pile] cache hit: {meta['train_shape']} train / {meta['eval_shape']} eval")
            return meta
    from datasets import load_dataset
    tok = get_tokenizer()
    print(f"[pile] tokenizing {PILE} with {MODEL_ID} tokenizer (vocab={tok.vocab_size}, "
          f"eos={tok.eos_token_id}) ...", flush=True)
    ds = load_dataset(PILE, split="train", streaming=True)
    it = iter(ds)
    eval_shape = _pack_split(tok, it, eval_tokens, seqlen, os.path.join(cache_dir, "eval.u16"), "eval")
    train_shape = _pack_split(tok, it, train_tokens, seqlen, os.path.join(cache_dir, "train.u16"), "train")
    meta = dict(train_tokens=train_tokens, eval_tokens=eval_tokens, seqlen=seqlen,
                train_shape=list(train_shape), eval_shape=list(eval_shape),
                vocab_size=tok.vocab_size, eos=tok.eos_token_id, dataset=PILE, model_id=MODEL_ID)
    json.dump(meta, open(mp, "w"), indent=1)
    print(f"[pile] done: {train_shape} train / {eval_shape} eval")
    return meta


def load_packed(cache_dir=DEFAULT_CACHE):
    meta = json.load(open(_meta(cache_dir)))
    train = np.memmap(os.path.join(cache_dir, "train.u16"), dtype=np.uint16, mode="r",
                      shape=tuple(meta["train_shape"]))
    ev = np.memmap(os.path.join(cache_dir, "eval.u16"), dtype=np.uint16, mode="r",
                   shape=tuple(meta["eval_shape"]))
    return train, ev, meta


class Batcher:
    """Random minibatches -> (input_ids) int64 on device; labels==input_ids (HF shifts internally)."""
    def __init__(self, arr, batch_size, device, seed=0):
        self.arr = arr; self.bs = batch_size; self.device = device
        self.rng = np.random.default_rng(seed); self.n = arr.shape[0]

    def batch(self):
        import torch
        idx = self.rng.integers(0, self.n, size=self.bs)
        return torch.from_numpy(np.asarray(self.arr[idx], dtype=np.int64)).to(self.device)

    def iter_eval(self, batch_size=None):
        import torch
        bs = batch_size or self.bs
        for i in range(0, self.n, bs):
            idx = np.arange(i, min(i + bs, self.n))
            yield torch.from_numpy(np.asarray(self.arr[idx], dtype=np.int64)).to(self.device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-tokens", type=int, default=300_000_000)
    ap.add_argument("--eval-tokens", type=int, default=5_000_000)
    ap.add_argument("--seqlen", type=int, default=SEQLEN)
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    prepare(args.train_tokens, args.eval_tokens, args.seqlen, args.cache_dir, args.force)


if __name__ == "__main__":
    main()

"""Shared utilities: config, seeding, device, full-test eval, CSV logging, byte accounting."""
import csv
import os
import random

import numpy as np
import torch
import yaml


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cudnn.benchmark stays on for speed; full bitwise determinism is not required
    # for this single-seed first pass (training is stochastic across runs anyway).
    torch.backends.cudnn.benchmark = True


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def eval_full(model, test_loader, device, use_bf16=True):
    """Top-1 accuracy (percent) on the full test split."""
    model.eval()
    correct = 0
    total = 0
    for x, y in test_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if use_bf16 and device.type == "cuda":
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
        else:
            logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return 100.0 * correct / total


def count_steps_per_epoch(loader):
    return len(loader)


class CSVLogger:
    """Minimal append-only CSV logger with a fixed header."""

    def __init__(self, path, fieldnames):
        self.path = path
        self.fieldnames = fieldnames
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    def log(self, **row):
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow({k: row.get(k, "") for k in self.fieldnames})


# --- byte / compressibility accounting -------------------------------------

def is_compressible_key(key, tensor):
    """We compress only conv/linear *weight* tensors (>=2-D). BN params, biases,
    and any 1-D tensors stay dense and are excluded from byte accounting."""
    return tensor.dim() >= 2 and key.endswith("weight")


def compressible_keys(state_dict):
    return [k for k, v in state_dict.items() if is_compressible_key(k, v)]


def fp32_bytes_of_keys(state_dict, keys):
    return int(sum(state_dict[k].numel() * 4 for k in keys))

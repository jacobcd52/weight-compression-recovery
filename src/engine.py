"""Shared training engine used by both baseline training and retraining.

Keeping the optimizer/scheduler/loop in one place guarantees the retraining runs
use exactly the same "optimizer state machine" as the baseline (AdamW + cosine +
linear warmup, bf16 mixed precision, full-test eval each epoch).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import eval_full


def build_optimizer(model, cfg):
    return torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        betas=tuple(cfg["betas"]),
    )


def build_scheduler(optimizer, total_steps, warmup_steps):
    """Per-step LR multiplier: linear warmup then cosine decay to 0."""
    warmup_steps = max(1, int(warmup_steps))
    total_steps = max(total_steps, warmup_steps + 1)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(total_steps - warmup_steps)
        progress = min(1.0, progress)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def distill_loss(student_logits, teacher_logits, labels, T=4.0, alpha=0.9):
    kl = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction="batchmean",
    ) * (T * T)
    ce = F.cross_entropy(student_logits, labels)
    return alpha * kl + (1.0 - alpha) * ce


def train_loop(model, train_loader, test_loader, device, *,
               optimizer, scheduler, total_steps, tb_writer, csv_logger,
               threshold=None, teacher=None, T=4.0, alpha=0.9,
               use_bf16=True, log_prefix="", on_best=None, eval_every_steps=None):
    """Run training until `total_steps` optimizer steps are taken, evaluating the
    full test split every `eval_every_steps` steps (default: once per epoch).

    A finer cadence resolves the recovery step in the low-budget region (e.g.
    eval_every_steps=78 ≈ 0.1% of baseline steps). If `threshold` is set, stop as
    soon as test accuracy >= threshold and record the step (recovery).
    """
    ce_loss = nn.CrossEntropyLoss()
    steps_per_epoch = len(train_loader)
    max_epochs = max(1, math.ceil(total_steps / steps_per_epoch))
    if not eval_every_steps:
        eval_every_steps = steps_per_epoch

    step = 0
    best_acc = 0.0
    recovery_steps = None
    history = []          # list of (step, test_acc)
    last_eval_step = -1

    if teacher is not None:
        teacher.eval()

    def evaluate(final=False):
        nonlocal best_acc, recovery_steps, last_eval_step
        if step == last_eval_step:
            return False
        last_eval_step = step
        test_acc = eval_full(model, test_loader, device, use_bf16=use_bf16)
        history.append((step, test_acc))
        if tb_writer is not None:
            tb_writer.add_scalar(f"{log_prefix}test_acc", test_acc, step)
        if csv_logger is not None:
            csv_logger.log(epoch=step // steps_per_epoch, step=step,
                           test_acc=test_acc, lr=scheduler.get_last_lr()[0])
        if test_acc > best_acc:
            best_acc = test_acc
            if on_best is not None:
                on_best(model, step // steps_per_epoch, step, test_acc)
        print(f"[{log_prefix or 'train'}] step {step}/{total_steps} "
              f"({100*step/total_steps:.1f}% budget) test_acc {test_acc:.2f} "
              f"best {best_acc:.2f}", flush=True)
        if threshold is not None and test_acc >= threshold and recovery_steps is None:
            recovery_steps = step
            print(f"[{log_prefix or 'train'}] RECOVERED at step {step} "
                  f"(acc {test_acc:.2f} >= {threshold:.2f})", flush=True)
            return True
        return False

    stop = False
    for epoch in range(max_epochs):
        if stop:
            break
        model.train()
        for x, y in train_loader:
            if step >= total_steps:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            if use_bf16 and device.type == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(x)
                    loss = (distill_loss(logits, teacher(x).detach(), y, T=T, alpha=alpha)
                            if teacher is not None else ce_loss(logits, y))
            else:
                logits = model(x)
                loss = (distill_loss(logits, teacher(x).detach(), y, T=T, alpha=alpha)
                        if teacher is not None else ce_loss(logits, y))

            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1

            if tb_writer is not None and step % 20 == 0:
                tb_writer.add_scalar(f"{log_prefix}loss", loss.item(), step)
                tb_writer.add_scalar(f"{log_prefix}lr", scheduler.get_last_lr()[0], step)

            if step % eval_every_steps == 0:
                model.eval()
                if evaluate():
                    stop = True
                    break
                model.train()

    if not stop:           # always do a final eval at the end of the budget
        model.eval()
        evaluate(final=True)

    return {
        "best_acc": best_acc,
        "final_acc": history[-1][1] if history else 0.0,
        "steps_taken": step,
        "recovery_steps": recovery_steps,
        "recovered": recovery_steps is not None,
        "history": history,
    }

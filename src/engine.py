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
               use_bf16=True, log_prefix="", on_best=None):
    """Run training until `total_steps` optimizer steps are taken (counting whole
    epochs), evaluating the full test split after every epoch.

    If `threshold` is set, stop as soon as test accuracy >= threshold and record
    the step at which it happened (recovery). Returns a results dict.
    """
    ce_loss = nn.CrossEntropyLoss()
    steps_per_epoch = len(train_loader)
    max_epochs = max(1, math.ceil(total_steps / steps_per_epoch))

    step = 0
    best_acc = 0.0
    recovery_steps = None
    history = []  # list of (epoch, step, test_acc)

    if teacher is not None:
        teacher.eval()

    for epoch in range(max_epochs):
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
                    if teacher is not None:
                        with torch.no_grad():
                            t_logits = teacher(x)
                        loss = distill_loss(logits, t_logits, y, T=T, alpha=alpha)
                    else:
                        loss = ce_loss(logits, y)
            else:
                logits = model(x)
                if teacher is not None:
                    with torch.no_grad():
                        t_logits = teacher(x)
                    loss = distill_loss(logits, t_logits, y, T=T, alpha=alpha)
                else:
                    loss = ce_loss(logits, y)

            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1

            if tb_writer is not None and step % 20 == 0:
                tb_writer.add_scalar(f"{log_prefix}loss", loss.item(), step)
                tb_writer.add_scalar(f"{log_prefix}lr", scheduler.get_last_lr()[0], step)

        test_acc = eval_full(model, test_loader, device, use_bf16=use_bf16)
        history.append((epoch, step, test_acc))
        if tb_writer is not None:
            tb_writer.add_scalar(f"{log_prefix}test_acc", test_acc, epoch)
        if csv_logger is not None:
            csv_logger.log(epoch=epoch, step=step, test_acc=test_acc,
                           lr=scheduler.get_last_lr()[0])

        if test_acc > best_acc:
            best_acc = test_acc
            if on_best is not None:
                on_best(model, epoch, step, test_acc)

        print(f"[{log_prefix or 'train'}] epoch {epoch+1}/{max_epochs} "
              f"step {step}/{total_steps} test_acc {test_acc:.2f} best {best_acc:.2f}",
              flush=True)

        if threshold is not None and test_acc >= threshold and recovery_steps is None:
            recovery_steps = step
            print(f"[{log_prefix or 'train'}] RECOVERED at step {step} "
                  f"(acc {test_acc:.2f} >= {threshold:.2f})", flush=True)
            break

        if step >= total_steps:
            break

    return {
        "best_acc": best_acc,
        "final_acc": history[-1][2] if history else 0.0,
        "steps_taken": step,
        "recovery_steps": recovery_steps,
        "recovered": recovery_steps is not None,
        "history": history,
    }

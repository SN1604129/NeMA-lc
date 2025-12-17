from __future__ import annotations

import argparse
import os
import csv
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from .models.transformer_lc import TransformerLC


# ======================================================
# DATASETS
# ======================================================

class ToyLongContextDataset(Dataset):
    """
    Toy task: sequence contains a key token somewhere; label depends on whether key appears.
    """
    def __init__(self, n: int = 5000, T: int = 128, vocab: int = 256, key_token: int = 42):
        self.n = n
        self.T = T
        self.vocab = vocab
        self.key = key_token

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        x = torch.randint(0, self.vocab, (self.T,), dtype=torch.long)
        y = torch.tensor(0, dtype=torch.long)

        if random.random() > 0.5:
            pos = random.randint(0, self.T - 1)
            x[pos] = self.key
            y = torch.tensor(1, dtype=torch.long)

        x[0] = 1  # CLS
        return x, y


class LRARetrievalDataset(Dataset):
    """
    LRA-style Retrieval Task
    Long sequence, query token may appear anywhere.
    """
    def __init__(
        self,
        n_samples: int = 8000,
        seq_len: int = 1024,
        vocab_size: int = 512,
        query_token: int = 42,
    ):
        self.n = n_samples
        self.T = seq_len
        self.vocab = vocab_size
        self.query = query_token

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        x = torch.randint(0, self.vocab, (self.T,), dtype=torch.long)
        y = torch.tensor(0, dtype=torch.long)

        if random.random() > 0.5:
            pos = random.randint(0, self.T - 1)
            x[pos] = self.query
            y = torch.tensor(1, dtype=torch.long)

        x[0] = 1  # CLS
        return x, y


# ======================================================
# TRAINING
# ======================================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--task", type=str, default="toy", choices=["toy", "lra"])
    ap.add_argument("--use_memory", action="store_true")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)

    # Logging name (lets you generate: full / noforget / nowrite / nostability)
    ap.add_argument("--run_name", type=str, default=None,
                    help="Optional run name for log file, e.g., full, noforget, nowrite, nostability.")

    # Memory reset mode:
    # - episodic_reset=True  => reset memory each batch (your old behavior)
    # - episodic_reset=False => keep memory across batches within an epoch (needed for meaningful avg_age)
    ap.add_argument("--episodic_reset", action="store_true",
                    help="If set, reset memory at every batch (episodic). Default keeps memory across batches.")

    # Lifecycle loss weights
    ap.add_argument("--lambda_write", type=float, default=1e-2)
    ap.add_argument("--lambda_forget", type=float, default=1e-2)
    ap.add_argument("--lambda_stability", type=float, default=1e-3)

    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------
    # Dataset selection
    # -------------------------
    if args.task == "toy":
        ds = ToyLongContextDataset(n=8000, T=128, vocab=256)
        vocab_size = 256
    else:
        ds = LRARetrievalDataset(
            n_samples=8000,
            seq_len=1024,
            vocab_size=512,
        )
        vocab_size = 512

    # If memory persists across batches, batch size must be consistent for memory tensors.
    # drop_last=True ensures the last smaller batch doesn't break persistent memory state.
    drop_last = bool(args.use_memory and (not args.episodic_reset))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=drop_last)

    # -------------------------
    # Model
    # -------------------------
    model = TransformerLC(
        vocab_size=vocab_size,
        d_model=128,
        n_layers=2,
        n_heads=4,
        n_classes=2,
        mem_slots=64,
        K_ops=4,
        controller_hidden=256,
        use_memory=args.use_memory,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss()

    # -------------------------
    # Logging
    # -------------------------
    os.makedirs("logs", exist_ok=True)

    run_tag = args.run_name if args.run_name is not None else args.task
    log_path = f"logs/memory_dynamics_{run_tag}.csv"

    write_header = not os.path.exists(log_path)
    log_f = open(log_path, "a", newline="")
    logger = csv.writer(log_f)

    if write_header:
        logger.writerow([
            "epoch", "step",
            "loss_total", "loss_task",
            "loss_write", "loss_forget", "loss_stability",
            "accuracy",
            "utilization", "avg_age",
            "writes", "updates", "forgets",
        ])
        log_f.flush()

    global_step = 0

    # -------------------------
    # Training loop
    # -------------------------
    for ep in range(args.epochs):
        model.train()
        correct, total = 0, 0

        # If NOT episodic reset, initialise memory once per epoch with fixed batch_size.
        if args.use_memory and (not args.episodic_reset):
            model.reset_memory(batch_size=args.batch_size, device=device)

        pbar = tqdm(dl, desc=f"[{args.task}|{run_tag}] epoch {ep+1}/{args.epochs}")

        for x, y in pbar:
            x, y = x.to(device), y.to(device)

            # Old behavior (episodic per batch)
            if args.use_memory and args.episodic_reset:
                model.reset_memory(batch_size=x.size(0), device=device)

            logits, aux = model(x)

            loss_task = ce(logits, y)
            loss = loss_task

            # Default values (for logging)
            loss_write = torch.tensor(0.0, device=device)
            loss_forget = torch.tensor(0.0, device=device)
            loss_stability = torch.tensor(0.0, device=device)

            if args.use_memory and (aux is not None):
                # Prefer differentiable lifecycle losses if your model provides them
                # (recommended: implement these inside TransformerLC forward and return in aux).
                if hasattr(aux, "loss_write") and aux.loss_write is not None:
                    loss_write = aux.loss_write
                elif hasattr(aux, "stats") and aux.stats is not None:
                    # Fallback: metric (NOT guaranteed to be differentiable)
                    loss_write = aux.stats.writes

                if hasattr(aux, "loss_forget") and aux.loss_forget is not None:
                    loss_forget = aux.loss_forget
                elif hasattr(aux, "stats") and aux.stats is not None and hasattr(aux, "attn") and aux.attn is not None:
                    loss_forget = aux.stats.forgets * aux.attn.mean()

                if hasattr(aux, "loss_stability") and aux.loss_stability is not None:
                    loss_stability = aux.loss_stability
                elif hasattr(aux, "mem_after") and aux.mem_after is not None and hasattr(aux, "mem_before") and aux.mem_before is not None:
                    loss_stability = (aux.mem_after - aux.mem_before).pow(2).mean()

                loss = (
                    loss_task
                    + args.lambda_write * loss_write
                    + args.lambda_forget * loss_forget
                    + args.lambda_stability * loss_stability
                )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            pred = logits.argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += y.numel()
            acc = correct / max(total, 1)

            # Logging (only if memory stats exist)
            if args.use_memory and (aux is not None) and hasattr(aux, "stats") and aux.stats is not None:
                pbar.set_postfix(
                    loss=float(loss.item()),
                    acc=float(acc),
                    util=float(aux.stats.utilization.item()),
                    age=float(aux.stats.avg_age.item()),
                    w=float(aux.stats.writes.item()),
                )

                logger.writerow([
                    ep, global_step,
                    float(loss.item()), float(loss_task.item()),
                    float(loss_write.item()),
                    float(loss_forget.item()),
                    float(loss_stability.item()),
                    float(acc),
                    float(aux.stats.utilization.item()),
                    float(aux.stats.avg_age.item()),
                    float(aux.stats.writes.item()),
                    float(aux.stats.updates.item()),
                    float(aux.stats.forgets.item()),
                ])
                log_f.flush()

            global_step += 1

        print(f"Epoch {ep+1}: acc={correct / max(total, 1):.4f}")

    log_f.close()
    print("Done.")


if __name__ == "__main__":
    main()

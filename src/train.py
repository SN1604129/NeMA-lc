from __future__ import annotations

import argparse
import os
import csv

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from .models.transformer_lc import TransformerLC


class ToyLongContextDataset(Dataset):
    """
    Toy task: sequence contains a key token somewhere; label depends on whether key appears.
    This is only to verify NeMA-LC wiring runs end-to-end.
    """
    def __init__(self, n: int = 5000, T: int = 128, vocab: int = 256, key_token: int = 42):
        self.n = n
        self.T = T
        self.vocab = vocab
        self.key = key_token

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        x = torch.randint(0, self.vocab, (self.T,), dtype=torch.long)

        # 50% inject key token
        y = torch.tensor(0, dtype=torch.long)
        if torch.rand(()) > 0.5:
            pos = torch.randint(0, self.T, (1,)).item()
            x[pos] = self.key
            y = torch.tensor(1, dtype=torch.long)

        # CLS token at position 0
        x[0] = 1
        return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--use_memory", action="store_true")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)

    # Explicit lifecycle loss weights (Paper 2)
    ap.add_argument("--lambda_write", type=float, default=1e-2)
    ap.add_argument("--lambda_forget", type=float, default=1e-2)
    ap.add_argument("--lambda_stability", type=float, default=1e-3)

    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    ds = ToyLongContextDataset(n=8000, T=128)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # Model
    model = TransformerLC(
        vocab_size=256,
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
    # Memory dynamics logging
    # -------------------------
    os.makedirs("logs", exist_ok=True)
    log_path = "logs/memory_dynamics.csv"
    write_header = not os.path.exists(log_path)

    log_f = open(log_path, "a", newline="")
    logger = csv.writer(log_f)

    if write_header:
        logger.writerow([
            "epoch",
            "step",
            "loss_total",
            "loss_task",
            "loss_write",
            "loss_forget",
            "loss_stability",
            "accuracy",
            "utilization",
            "avg_age",
            "writes",
            "updates",
            "forgets",
        ])

    global_step = 0

    # -------------------------
    # Training loop
    # -------------------------
    for ep in range(args.epochs):
        model.train()
        correct = 0
        total = 0

        pbar = tqdm(dl, desc=f"epoch {ep+1}/{args.epochs}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)

            # Reset memory per batch (safe baseline)
            if args.use_memory:
                model.reset_memory(batch_size=x.size(0), device=x.device)

            logits, aux = model(x)

            # Task loss
            loss_task = ce(logits, y)
            loss = loss_task

            # -------------------------
            # Explicit lifecycle losses
            # -------------------------
            loss_write = torch.tensor(0.0, device=device)
            loss_forget = torch.tensor(0.0, device=device)
            loss_stability = torch.tensor(0.0, device=device)

            if args.use_memory and aux.stats is not None:
                # 1) Write budget loss
                loss_write = aux.stats.writes

                # 2) Forget utility loss (proxy):
                # penalize forgetting when attention is high (mean attention mass)
                attn_mean = aux.attn.mean()
                loss_forget = aux.stats.forgets * attn_mean

                # 3) Stability / churn loss
                loss_stability = (aux.mem_after - aux.mem_before).pow(2).mean()

                # Combine
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

            # Accuracy
            pred = logits.argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += y.numel()
            acc = correct / max(total, 1)

            # Progress bar + CSV logging
            if args.use_memory and aux.stats is not None:
                pbar.set_postfix(
                    loss=float(loss.item()),
                    acc=acc,
                    util=float(aux.stats.utilization.item()),
                    age=float(aux.stats.avg_age.item()),
                    w=float(loss_write.item()),
                    f=float(loss_forget.item()),
                    s=float(loss_stability.item()),
                )

                logger.writerow([
                    ep,
                    global_step,
                    float(loss.item()),
                    float(loss_task.item()),
                    float(loss_write.item()),
                    float(loss_forget.item()),
                    float(loss_stability.item()),
                    acc,
                    float(aux.stats.utilization.item()),
                    float(aux.stats.avg_age.item()),
                    float(aux.stats.writes.item()),
                    float(aux.stats.updates.item()),
                    float(aux.stats.forgets.item()),
                ])
            else:
                pbar.set_postfix(loss=float(loss.item()), acc=acc)

                logger.writerow([
                    ep,
                    global_step,
                    float(loss.item()),
                    float(loss_task.item()),
                    0.0,
                    0.0,
                    0.0,
                    acc,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ])

            global_step += 1

        print(f"Epoch {ep+1}: acc={correct/total:.4f}")

    log_f.close()
    print("Done.")


if __name__ == "__main__":
    main()

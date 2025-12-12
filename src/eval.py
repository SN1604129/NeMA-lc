from __future__ import annotations
import argparse
import torch
from torch.utils.data import DataLoader

from train import ToyLongContextDataset
from models.transformer_lc import TransformerLC


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--use_memory", action="store_true")
    ap.add_argument("--batch_size", type=int, default=128)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = ToyLongContextDataset(n=2000, T=128)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = TransformerLC(use_memory=args.use_memory).to(device)
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            logits, _aux = model(x)
            pred = logits.argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += y.numel()

    print(f"Eval acc: {correct/total:.4f}")


if __name__ == "__main__":
    main()

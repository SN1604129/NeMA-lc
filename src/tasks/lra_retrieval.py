from __future__ import annotations
import torch
from torch.utils.data import Dataset
import random


class LRARetrievalDataset(Dataset):
    """
    Simplified LRA-style retrieval task.
    Sequence length: long (default 1024).
    Label: whether query token appears in context.
    """

    def __init__(
        self,
        n_samples: int = 10000,
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

        # CLS token at position 0
        x[0] = 1
        return x, y

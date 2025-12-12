from __future__ import annotations
import torch
import torch.nn as nn


class TopKAllocator(nn.Module):
    """
    Budgeted selection:
    - per batch, select up to K slot-ops among N slots
    - separately decide whether to write (1 action)
    This is a simple deterministic top-k (works well to start).
    """
    def __init__(self, K_ops: int):
        super().__init__()
        self.K_ops = int(K_ops)

    @torch.no_grad()
    def forward(self, slot_scores: torch.Tensor):
        """
        slot_scores: (B, N) higher = more likely to operate on that slot
        returns op_mask: (B, N) boolean selecting K slots
        """
        B, N = slot_scores.shape
        K = min(self.K_ops, N)
        topk_idx = torch.topk(slot_scores, k=K, dim=-1).indices  # (B,K)
        mask = torch.zeros(B, N, dtype=torch.bool, device=slot_scores.device)
        mask.scatter_(dim=1, index=topk_idx, value=True)
        return mask


class MemoryController(nn.Module):
    """
    Produces lifecycle probabilities per slot (retain/update/forget) and a write decision.

    Inputs:
      mem:  (B, N, D)
      age:  (B, N)
      usage:(B, N)
      ctx:  (B, D)  (CLS/context)

    Outputs:
      p_ops: (B, N, 3) softmax over [retain, update, forget]
      slot_scores: (B, N) utility for choosing which slots to operate on (Top-K)
      write_score: (B,) score for write action
    """
    def __init__(self, mem_dim: int, hidden: int = 256, use_age: bool = True, use_usage: bool = True):
        super().__init__()
        self.mem_dim = int(mem_dim)
        self.use_age = bool(use_age)
        self.use_usage = bool(use_usage)

        in_dim = mem_dim + mem_dim  # mem + ctx
        if self.use_age:
            in_dim += 1
        if self.use_usage:
            in_dim += 1

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.ops_head = nn.Linear(hidden, 3)      # retain/update/forget logits
        self.score_head = nn.Linear(hidden, 1)    # slot utility score

        self.write_head = nn.Sequential(
            nn.Linear(mem_dim + mem_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, mem: torch.Tensor, age: torch.Tensor, usage: torch.Tensor, ctx: torch.Tensor):
        B, N, D = mem.shape
        ctx_rep = ctx.unsqueeze(1).expand(B, N, D)

        feats = [mem, ctx_rep]
        if self.use_age:
            feats.append(age.unsqueeze(-1))
        if self.use_usage:
            feats.append(usage.unsqueeze(-1))

        x = torch.cat(feats, dim=-1)                 # (B,N,feat)
        h = self.mlp(x)                              # (B,N,H)
        ops_logits = self.ops_head(h)                # (B,N,3)
        p_ops = torch.softmax(ops_logits, dim=-1)

        slot_scores = self.score_head(h).squeeze(-1) # (B,N)

        write_in = torch.cat([ctx, ctx], dim=-1)     # simple; replace later with write vector
        write_score = self.write_head(write_in).squeeze(-1)  # (B,)

        return p_ops, slot_scores, write_score

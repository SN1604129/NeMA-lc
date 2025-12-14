from __future__ import annotations
import torch
import torch.nn as nn


class TopKAllocator(nn.Module):
    """
    Budgeted selection over slots only (kept for compatibility).
    """
    def __init__(self, K_ops: int):
        super().__init__()
        self.K_ops = int(K_ops)

    @torch.no_grad()
    def forward(self, slot_scores: torch.Tensor) -> torch.Tensor:
        """
        slot_scores: (B, N)
        returns op_mask: (B, N) bool selecting K slots
        """
        B, N = slot_scores.shape
        K = min(self.K_ops, N)
        topk_idx = torch.topk(slot_scores, k=K, dim=-1).indices  # (B,K)
        mask = torch.zeros(B, N, dtype=torch.bool, device=slot_scores.device)
        mask.scatter_(dim=1, index=topk_idx, value=True)
        return mask


class TopKAllocatorWithWrite(nn.Module):
    """
    Budgeted selection where WRITE competes with slot operations under one K budget.

    We create N+1 "action scores":
      - N slot-op scores
      - 1 write score

    Then select top-K actions. If the write action is selected, write_mask=True.
    """
    def __init__(self, K_total: int):
        super().__init__()
        self.K_total = int(K_total)

    @torch.no_grad()
    def forward(self, slot_scores: torch.Tensor, write_score: torch.Tensor):
        """
        slot_scores: (B, N)
        write_score: (B,)

        returns:
          op_mask: (B, N) bool
          write_mask: (B,) bool
        """
        B, N = slot_scores.shape
        K = min(self.K_total, N + 1)

        # concat write as an extra action
        action_scores = torch.cat([slot_scores, write_score.unsqueeze(-1)], dim=-1)  # (B, N+1)
        topk_idx = torch.topk(action_scores, k=K, dim=-1).indices  # (B, K)

        action_mask = torch.zeros(B, N + 1, dtype=torch.bool, device=slot_scores.device)
        action_mask.scatter_(dim=1, index=topk_idx, value=True)

        op_mask = action_mask[:, :N]
        write_mask = action_mask[:, N]
        return op_mask, write_mask


class MemoryController(nn.Module):
    """
    Produces lifecycle probabilities per slot (retain/update/forget)
    and a write score (write competes with slot ops if you use TopKAllocatorWithWrite).

    Inputs:
      mem:   (B, N, D)
      age:   (B, N)
      usage: (B, N)
      ctx:   (B, D)

    Outputs:
      p_ops:       (B, N, 3) softmax over [retain, update, forget]
      slot_scores: (B, N) slot utility for choosing which slots to operate on
      write_score: (B,) write utility score (compete with slot ops)
    """
    def __init__(self, mem_dim: int, hidden: int = 256, use_age: bool = True, use_usage: bool = True):
        super().__init__()
        self.mem_dim = int(mem_dim)
        self.use_age = bool(use_age)
        self.use_usage = bool(use_usage)

        # slot feature dim = mem + ctx (+ age) (+ usage)
        in_dim = mem_dim + mem_dim
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
        self.ops_head = nn.Linear(hidden, 3)   # retain/update/forget logits
        self.score_head = nn.Linear(hidden, 1) # slot utility

        # write score uses ctx + novelty (ctx - readout proxy); for now use ctx + ctx_norm
        self.write_head = nn.Sequential(
            nn.Linear(mem_dim + 1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, mem: torch.Tensor, age: torch.Tensor, usage: torch.Tensor, ctx: torch.Tensor):
        """
        Returns:
          p_ops: (B,N,3)
          slot_scores: (B,N)
          write_score: (B,)
        """
        B, N, D = mem.shape
        ctx_rep = ctx.unsqueeze(1).expand(B, N, D)

        feats = [mem, ctx_rep]
        if self.use_age:
            feats.append(age.unsqueeze(-1))
        if self.use_usage:
            feats.append(usage.unsqueeze(-1))

        x = torch.cat(feats, dim=-1)              # (B,N,feat)
        h = self.mlp(x)                           # (B,N,H)

        ops_logits = self.ops_head(h)             # (B,N,3)
        p_ops = torch.softmax(ops_logits, dim=-1)

        slot_scores = self.score_head(h).squeeze(-1)  # (B,N)

        # A simple, stable write feature: norm(ctx)
        ctx_norm = torch.norm(ctx, dim=-1, keepdim=True)  # (B,1)
        write_in = torch.cat([ctx, ctx_norm], dim=-1)     # (B, D+1)
        write_score = self.write_head(write_in).squeeze(-1)  # (B,)

        return p_ops, slot_scores, write_score

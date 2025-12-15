from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class MemoryStats:
    utilization: torch.Tensor  # scalar
    avg_age: torch.Tensor      # scalar
    writes: torch.Tensor       # scalar
    updates: torch.Tensor      # scalar
    forgets: torch.Tensor      # scalar


class MemorySlots(nn.Module):
    """
    Fixed-budget external memory slots with metadata.
    Shapes:
      mem:   (B, N, D)
      age:   (B, N)
      usage: (B, N)
      alive: (B, N) boolean
    """
    def __init__(self, n_slots: int, mem_dim: int, device: torch.device | None = None):
        super().__init__()
        self.n_slots = int(n_slots)
        self.mem_dim = int(mem_dim)
        self.device = device

        # "empty slot" content template
        self.empty = nn.Parameter(torch.zeros(mem_dim))

        self._mem: torch.Tensor | None = None
        self._age: torch.Tensor | None = None
        self._usage: torch.Tensor | None = None
        self._alive: torch.Tensor | None = None

    def reset(self, batch_size: int, device: torch.device | None = None):
        dev = device or self.device or torch.device("cpu")
        B, N, D = batch_size, self.n_slots, self.mem_dim
        self._mem = self.empty.view(1, 1, D).repeat(B, N, 1).to(dev)
        self._age = torch.zeros(B, N, device=dev)
        self._usage = torch.zeros(B, N, device=dev)
        self._alive = torch.zeros(B, N, dtype=torch.bool, device=dev)

    @property
    def mem(self) -> torch.Tensor:
        assert self._mem is not None, "Call reset() before using memory."
        return self._mem

    @property
    def age(self) -> torch.Tensor:
        assert self._age is not None, "Call reset() before using memory."
        return self._age

    @property
    def usage(self) -> torch.Tensor:
        assert self._usage is not None, "Call reset() before using memory."
        return self._usage

    @property
    def alive(self) -> torch.Tensor:
        assert self._alive is not None, "Call reset() before using memory."
        return self._alive

    def read(self, query: torch.Tensor):
        """
        Dot-product attention read from memory.
        query: (B, D)
        returns:
          read_vec: (B, D)
          attn: (B, N)
        """
        B, N, D = self.mem.shape
        q = query.view(B, 1, D)

        logits = (self.mem * q).sum(dim=-1)
        logits = logits.masked_fill(~self.alive, float("-inf"))

        attn = torch.softmax(logits, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)

        read_vec = (attn.unsqueeze(-1) * self.mem).sum(dim=1)

        # Exponential moving average usage
        with torch.no_grad():
            self._usage = 0.95 * self._usage + 0.05 * attn

        return read_vec, attn

    def step(
        self,
        retain_mask: torch.Tensor,
        update_mask: torch.Tensor,
        forget_mask: torch.Tensor,
        write_mask: torch.Tensor,
        update_vec: torch.Tensor,
        write_vec: torch.Tensor,
        overwrite_idx: torch.Tensor,
    ) -> MemoryStats:
        """
        Apply lifecycle operations.
        Masks: retain/update/forget: (B,N) bool, write_mask: (B,) bool
        overwrite_idx: (B,) long
        update_vec/write_vec: (B,D)
        """
        B, N, D = self.mem.shape
        dev = self.mem.device

        # Track what actually happened (helps stable logging)
        updated_mask = update_mask.clone()
        forgotten_mask = forget_mask.clone()
        written_mask = write_mask.clone()

        # -------------------------
        # FORGET
        # -------------------------
        if bool(forget_mask.any()):
            self._mem = torch.where(
                forget_mask.unsqueeze(-1),
                self.empty.view(1, 1, D).to(dev).expand(B, N, D),
                self._mem,
            )
            with torch.no_grad():
                self._alive = self._alive & (~forget_mask)
                self._age = torch.where(forget_mask, torch.zeros_like(self._age), self._age)
                self._usage = torch.where(forget_mask, torch.zeros_like(self._usage), self._usage)

        # -------------------------
        # UPDATE
        # -------------------------
        # NOTE: We DO NOT reset age on update by default.
        # Updating means "refine content" not "newly written slot".
        if bool(update_mask.any()):
            u = update_mask.float().unsqueeze(-1)  # (B,N,1)
            upd = update_vec.view(B, 1, D).expand(B, N, D)

            self._mem = (1.0 - u) * self._mem + u * (0.5 * self._mem + 0.5 * upd)

            with torch.no_grad():
                self._alive = self._alive | update_mask
                # Keep age as-is for updated slots (no reset)

        # -------------------------
        # RETAIN (metadata only)
        # -------------------------
        if bool(retain_mask.any()):
            with torch.no_grad():
                self._alive = self._alive | retain_mask

        # -------------------------
        # WRITE (overwrite chosen slot per batch element)
        # -------------------------
        if bool(write_mask.any()):
            b_idx = torch.arange(B, device=dev)
            tgt = overwrite_idx.clamp(min=0, max=N - 1)
            wm = write_mask

            new_mem = self._mem.clone()
            new_mem[b_idx[wm], tgt[wm], :] = write_vec[wm]
            self._mem = new_mem

            with torch.no_grad():
                self._alive[b_idx[wm], tgt[wm]] = True
                self._age[b_idx[wm], tgt[wm]] = 0.0
                self._usage[b_idx[wm], tgt[wm]] = 0.0

        # -------------------------
        # AGE TICK (after lifecycle ops)
        # -------------------------
        # Age increases for slots that are alive at the end of this step.
        # Newly written slots have age=0 already, and will become age=1 on next step.
        with torch.no_grad():
            self._age = self._age + self._alive.float()

        # Detach state so it never carries graphs across steps/batches
        self._mem = self._mem.detach()
        self._age = self._age.detach()
        self._usage = self._usage.detach()
        self._alive = self._alive.detach()

        # -------------------------
        # STATS
        # -------------------------
        utilization = self._alive.float().mean()

        denom = self._alive.float().sum() + 1e-6
        avg_age = torch.where(self._alive, self._age, torch.zeros_like(self._age)).sum() / denom

        # Per-step operation rates
        # - writes: fraction of batch elements that wrote
        # - updates/forgets: fraction of slots operated on (mean over B,N)
        writes = written_mask.float().mean()
        updates = updated_mask.float().mean()
        forgets = forgotten_mask.float().mean()

        return MemoryStats(utilization, avg_age, writes, updates, forgets)

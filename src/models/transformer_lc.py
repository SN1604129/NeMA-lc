from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn

from .memory_slots import MemorySlots
from .memory_controller import MemoryController, TopKAllocatorWithWrite


@dataclass
class AuxOut:
    attn: torch.Tensor                 # (B,N)
    p_ops: torch.Tensor                # (B,N,3)
    stats: object                      # MemoryStats
    mem_before: torch.Tensor           # (B,N,D)
    mem_after: torch.Tensor            # (B,N,D)


class TinyTransformerEncoder(nn.Module):
    """
    Minimal encoder for sanity checks and long-context tasks.
    Input: token ids (B, T)
    Output: ctx (B, D) [CLS-like], token states (B, T, D)
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        max_len: int = 2048,   # ✅ FIX: support LRA (>=1024)
    ):
        super().__init__()

        self.max_len = max_len
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            batch_first=True,
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor):
        B, T = x.shape

        # Safety check (helps debugging & reviewers)
        if T > self.max_len:
            raise ValueError(
                f"Sequence length {T} exceeds max_len {self.max_len}. "
                "Increase positional embedding size."
            )

        pos_ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.emb(x) + self.pos(pos_ids)
        h = self.enc(h)
        h = self.ln(h)
        ctx = h[:, 0, :]  # token 0 as CLS
        return ctx, h


class TransformerLC(nn.Module):
    """
    Transformer + Lifecycle Memory (NeMA-LC).
    Paper-2–aligned version: slot ops + write compete under one budget.
    """
    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        n_classes: int = 2,
        mem_slots: int = 64,
        K_ops: int = 4,
        controller_hidden: int = 256,
        use_memory: bool = True,
    ):
        super().__init__()
        self.use_memory = bool(use_memory)

        self.encoder = TinyTransformerEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            max_len=2048,  # explicitly long-context safe
        )

        self.mem = MemorySlots(n_slots=mem_slots, mem_dim=d_model)
        self.controller = MemoryController(mem_dim=d_model, hidden=controller_hidden)

        # WRITE + SLOT OPS SHARE ONE BUDGET
        # NOTE: allocator now supports write gating via write_tau/util_cap in memory_controller.py
        self.allocator = TopKAllocatorWithWrite(K_total=K_ops)

        self.head = nn.Linear(d_model, n_classes)

    def reset_memory(self, batch_size: int, device: torch.device):
        self.mem.reset(batch_size=batch_size, device=device)

    def forward(self, x: torch.Tensor):
        """
        x: (B,T) token ids
        returns: logits (B,C), aux (AuxOut)
        """
        B, T = x.shape
        device = x.device

        # Initialize / reset memory safely
        if getattr(self.mem, "_mem", None) is None or self.mem.mem.shape[0] != B:
            self.reset_memory(batch_size=B, device=device)

        # Encode input
        ctx, _tok = self.encoder(x)  # (B,D)

        # Defaults
        attn = torch.zeros(B, self.mem.n_slots, device=device)
        p_ops = torch.zeros(B, self.mem.n_slots, 3, device=device)
        stats = None

        # Clone memory snapshot for analysis
        mem_before = self.mem.mem.detach().clone()

        if self.use_memory:
            # READ
            mem_read, attn = self.mem.read(ctx)
            ctx2 = ctx + mem_read

            # CONTROLLER
            p_ops, slot_scores, write_score = self.controller(
                self.mem.mem,
                self.mem.age,
                self.mem.usage,
                ctx2,
            )

            # ✅ UTILIZATION for write gating
            # (B,) fraction of alive slots per batch element
            util = self.mem.alive.float().mean(dim=1)

            # ALLOCATION: slot ops + write under same K, with utilization gate
            op_mask, write_mask = self.allocator(slot_scores, write_score, utilization=util)

            # Lifecycle op per selected slot
            op_choice = torch.argmax(p_ops, dim=-1)  # 0 retain, 1 update, 2 forget

            retain_mask = op_mask & (op_choice == 0)
            update_mask = op_mask & (op_choice == 1)
            forget_mask = op_mask & (op_choice == 2)

            # Choose overwrite target
            with torch.no_grad():
                overwrite_idx = torch.argmin(slot_scores, dim=-1)
                any_forget = forget_mask.any(dim=-1)
                if any_forget.any():
                    first_forget = torch.argmax(forget_mask.float(), dim=-1)
                    overwrite_idx = torch.where(any_forget, first_forget, overwrite_idx)

            # Vectors to write/update
            update_vec = ctx2
            write_vec = ctx2

            # STEP memory
            stats = self.mem.step(
                retain_mask=retain_mask,
                update_mask=update_mask,
                forget_mask=forget_mask,
                write_mask=write_mask,
                update_vec=update_vec,
                write_vec=write_vec,
                overwrite_idx=overwrite_idx,
            )

            ctx_final = ctx2
        else:
            ctx_final = ctx

        logits = self.head(ctx_final)

        aux = AuxOut(
            attn=attn,
            p_ops=p_ops,
            stats=stats,
            mem_before=mem_before,
            mem_after=self.mem.mem.detach().clone(),
        )
        return logits, aux

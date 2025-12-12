from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn

from .memory_slots import MemorySlots
from .memory_controller import MemoryController, TopKAllocator


@dataclass
class AuxOut:
    attn: torch.Tensor                 # (B,N)
    p_ops: torch.Tensor                # (B,N,3)
    stats: object                      # MemoryStats
    mem_before: torch.Tensor           # (B,N,D)
    mem_after: torch.Tensor            # (B,N,D)


class TinyTransformerEncoder(nn.Module):
    """
    Minimal encoder for sanity checks.
    Input: token ids (B, T)
    Output: ctx (B, D) [CLS-like], token states (B, T, D)
    """
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int, max_len: int = 512):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model, batch_first=True
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor):
        B, T = x.shape
        pos_ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.emb(x) + self.pos(pos_ids)
        h = self.enc(h)
        h = self.ln(h)
        ctx = h[:, 0, :]  # take token 0 as CLS
        return ctx, h


class TransformerLC(nn.Module):
    """
    Transformer + Lifecycle Memory (NeMA-LC skeleton).
    For now, this supports classification (toy task).
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

        self.encoder = TinyTransformerEncoder(vocab_size, d_model, n_layers, n_heads)
        self.mem = MemorySlots(n_slots=mem_slots, mem_dim=d_model)
        self.controller = MemoryController(mem_dim=d_model, hidden=controller_hidden)
        self.allocator = TopKAllocator(K_ops=K_ops)

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

        if self.mem._mem is None or self.mem.mem.shape[0] != B:
            self.reset_memory(batch_size=B, device=device)

        ctx, _tok = self.encoder(x)  # (B,D)

        # default aux
        attn = torch.zeros(B, self.mem.n_slots, device=device)
        p_ops = torch.zeros(B, self.mem.n_slots, 3, device=device)
        stats = None
        mem_before = self.mem.mem.detach()

        if self.use_memory:
            # READ
            mem_read, attn = self.mem.read(ctx)
            ctx2 = ctx + mem_read

            # CONTROLLER
            p_ops, slot_scores, write_score = self.controller(self.mem.mem, self.mem.age, self.mem.usage, ctx2)

            # select which slots to operate on
            op_mask = self.allocator(slot_scores)  # (B,N)

            # choose lifecycle op per selected slot (argmax over simplex for now)
            op_choice = torch.argmax(p_ops, dim=-1)  # (B,N) 0 retain 1 update 2 forget

            retain_mask = op_mask & (op_choice == 0)
            update_mask = op_mask & (op_choice == 1)
            forget_mask = op_mask & (op_choice == 2)

            # write decision: write if score > 0 (simple threshold to start)
            write_mask = write_score > 0.0  # (B,)

            # choose overwrite target: prefer any forgotten slot; else lowest score
            with torch.no_grad():
                # if any forget slots exist, pick first forgotten index; else argmin(slot_scores)
                overwrite_idx = torch.argmin(slot_scores, dim=-1)  # (B,)
                any_forget = forget_mask.any(dim=-1)               # (B,)
                if any_forget.any():
                    first_forget = torch.argmax(forget_mask.float(), dim=-1)  # (B,)
                    overwrite_idx = torch.where(any_forget, first_forget, overwrite_idx)

            # update vector and write vector (for now, both = ctx2)
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
            mem_after=self.mem.mem.detach(),
        )
        return logits, aux

# NeMA-LC (Paper 2) — Memory Lifecycle Control

Author- Sudipta Nath


NeMA-LC extends NeMA-Lite by introducing a learned **memory lifecycle** under a fixed budget:
**write → retain → update → forget**, validated on long-context tasks.

## Research Question
How can a neural model dynamically allocate, update, and forget external memory under a fixed budget,
while preserving long-range performance?

## Repository Scope
This repo contains Paper 2 code (NeMA-LC) and Paper 2 baselines (Transformer, Always-Write, FIFO/LRU, NeMA-Lite as imported baseline).

Paper 1 (NeMA-Lite: selective writing) lives in a separate repository and will not be modified here.

## Quickstart
TBD (after skeleton is wired).

## Status
Design + scaffold.
Next: formal problem definition, lifecycle controller, allocation rule, and training objectives.

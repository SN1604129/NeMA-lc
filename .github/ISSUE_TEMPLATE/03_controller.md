---
name: Controller
about: Neural controller network design
title: "Design memory controller network"
labels: ["research","implementation"]
---

## Define
- Inputs per slot and global context
- Outputs: retain/update/forget (simplex) + optional write score
- Allocation mechanism (Top-K / Gumbel-TopK)

## Deliverables
- `memory_controller.py` spec + forward shapes

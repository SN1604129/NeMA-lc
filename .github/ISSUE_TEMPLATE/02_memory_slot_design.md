---
name: Memory slot design
about: Memory slot tensor layout + metadata
title: "Fixed-budget memory slot structure"
labels: ["implementation"]
---

## Decide
- Slot content vector representation
- Metadata: age, usage, importance, alive mask
- Initialization and empty-slot representation

## Deliverables
- `memory_slots.py` spec
- logging of utilization / age / churn

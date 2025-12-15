NeMA-LC — Neural Memory Allocation with Lifecycle Control

Author: Sudipta Nath

📌 Overview

NeMA-LC introduces a learned memory lifecycle framework for memory-augmented Transformers, enabling models to write, retain, update, and forget memory slots under a fixed capacity budget.

Unlike prior approaches that treat memory as a passive buffer or rely on heuristic replacement strategies (e.g., FIFO, LRU), NeMA-LC learns explicit lifecycle decisions through a neural controller, transforming memory into an actively managed computational resource.

This repository contains the full implementation, training pipeline, and analysis code for Paper 2.

🧠 Core Research Question

How can a neural model dynamically manage external memory—deciding when to write, retain, update, or forget—under a fixed budget, while preserving long-range task performance?

✨ Key Contributions

NeMA-LC makes the following contributions:

Memory Lifecycle Formulation
Formalises memory management as a learned lifecycle:
write → retain → update → forget

Neural Memory Controller
A controller network predicts per-slot lifecycle actions conditioned on:

memory content

age

usage

current context

Explicit Budget Constraint
Slot operations and write actions compete under a shared budget, preventing uncontrolled memory growth.

Lifecycle-Aware Training Objective
Introduces explicit losses for:

write budget control

forgetting cost

memory stability (churn)

Empirical Analysis of Memory Dynamics
Goes beyond accuracy by analysing:

memory utilisation

average retention age

write / update / forget rates

stability over long horizons

🧩 Relationship to Paper 1
Paper	Focus	Scope
Paper 1 (NeMA-Lite)	When to write?	Selective memory writing
Paper 2 (NeMA-LC)	How to manage memory over time?	Full lifecycle control

Paper 2 generalises and subsumes Paper 1 by addressing the complete memory lifecycle.

🏗️ Architecture Overview

NeMA-LC consists of four core components:

Transformer Encoder
Produces contextual representations from long sequences.

Fixed-Budget Memory Slots
Each slot stores:

content vector

age

usage signal

alive state

Neural Memory Controller
Predicts per-slot probabilities for:

retain

update

forget
and a global write score.

Budgeted Allocator
Enforces a hard limit on the total number of memory operations per step.

📂 Repository Structure
nema-paper2/
├── README.md
├── src/
│   ├── models/
│   │   ├── memory_slots.py        # Fixed-size memory + metadata
│   │   ├── memory_controller.py   # Lifecycle controller + allocator
│   │   └── transformer_lc.py      # Transformer + lifecycle memory
│   ├── train.py                   # Training loop (toy + LRA)
│   └── eval.py                    # Evaluation utilities
├── scripts/
│   ├── plot_memory_dynamics.py    # Memory dynamics plots
│   └── plot_lra_compare.py        # Baseline vs NeMA-LC comparison
├── logs/                          # (ignored) training logs
├── plots/                         # (ignored) generated figures
└── requirements.txt

🧪 Supported Tasks

Toy Long-Context Classification (sanity checks)

LRA Retrieval (long-range benchmark)

The framework is task-agnostic and designed to extend to:

document QA

continual learning

multimodal memory (Paper 3)

🚀 Running Experiments
1️⃣ Setup Environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

2️⃣ Train LRA Baseline (no memory)
python -m src.train --task lra --batch_size 4

3️⃣ Train NeMA-LC on LRA
python -m src.train --task lra --use_memory --batch_size 4

4️⃣ Plot Memory Dynamics
python scripts/plot_memory_dynamics.py


This generates:

memory utilisation curves

average retention age

lifecycle operation rates

loss component trajectories

5️⃣ Compare Against Baseline
python scripts/plot_lra_compare.py


Produces paper-ready comparison plots.

📊 Logged Metrics

NeMA-LC logs the following per training step:

task loss

write loss

forget loss

stability loss

memory utilisation

average memory age

write / update / forget rates

accuracy

These metrics support interpretability and ablation analysis, not just performance reporting.

🧠 Design Choices (Important Notes)

Memory Reset per Batch
Memory is reset at batch boundaries to model episodic memory.
This avoids cross-batch gradient entanglement and enables controlled analysis.

Explicit Budget Enforcement
Write actions are not free — they compete with slot operations.

Interpretability First
Memory dynamics are treated as first-class experimental results.

🏆 Target Publication Venues

This work is intended for Q1 journals, including:

IEEE Transactions on Neural Networks and Learning Systems (TNNLS)

Machine Learning Journal (Springer)

Neural Computation (MIT Press)

Transactions on Machine Learning Research (TMLR)

📌 Status

✔ Core architecture implemented
✔ Lifecycle losses integrated
✔ Long-context benchmark validated
✔ Memory dynamics analysed

🟡 Additional benchmarks (e.g., Document QA) optional
🟡 Ablation study recommended before submission

📖 Citation (placeholder)
@article{nema_lc,
  title={Neural Memory Allocation with Lifecycle Control},
  author={},
  journal={},
  year={}
}

🧭 Roadmap

Paper 2 (this work): Learned memory lifecycle control

Paper 3: Continual and multimodal memory systems

Thesis: Unified neural memory systems for long-horizon reasoning

✅ Bottom line

This repository contains a complete, reproducible, and journal-ready implementation of NeMA-LC, addressing a fundamental open problem in memory-augmented neural networks.

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

CSV_PATH = Path("logs/memory_dynamics_lra.csv")
OUT = Path("plots")
OUT.mkdir(exist_ok=True)

df = pd.read_csv(CSV_PATH)

# ---- Robust step detection ----
if "step" in df.columns:
    step_col = "step"
elif "global_step" in df.columns:
    step_col = "global_step"
else:
    # fallback: use row index
    df["step_idx"] = range(len(df))
    step_col = "step_idx"

df = df.sort_values(step_col)

def smooth(x, w=50):
    return x.rolling(w, min_periods=1).mean()

# ---- Utilization ----
if "utilization" in df.columns:
    plt.figure(figsize=(7, 5))
    plt.plot(df[step_col], smooth(df["utilization"]))
    plt.xlabel("Training step")
    plt.ylabel("Memory utilization")
    plt.title("Memory Utilization over Training")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUT / "utilization.png", dpi=200)
    plt.close()

# ---- Average Age ----
if "avg_age" in df.columns:
    plt.figure(figsize=(7, 5))
    plt.plot(df[step_col], smooth(df["avg_age"]))
    plt.xlabel("Training step")
    plt.ylabel("Average memory age")
    plt.title("Average Memory Age")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUT / "avg_age.png", dpi=200)
    plt.close()

# ---- Lifecycle Ops ----
if all(c in df.columns for c in ["writes", "updates", "forgets"]):
    plt.figure(figsize=(7, 5))
    plt.plot(df[step_col], smooth(df["writes"]), label="writes")
    plt.plot(df[step_col], smooth(df["updates"]), label="updates")
    plt.plot(df[step_col], smooth(df["forgets"]), label="forgets")
    plt.xlabel("Training step")
    plt.ylabel("Operation rate")
    plt.title("Memory Lifecycle Operations")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUT / "lifecycle_ops.png", dpi=200)
    plt.close()

# ---- Loss components (Paper 2 specific) ----
loss_cols = ["loss_task", "loss_write", "loss_forget", "loss_stability"]
if all(c in df.columns for c in loss_cols):
    plt.figure(figsize=(7, 5))
    for c in loss_cols:
        plt.plot(df[step_col], smooth(df[c]), label=c)
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.title("Lifecycle Loss Components")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUT / "loss_components.png", dpi=200)
    plt.close()

print("Plots saved to:", OUT.resolve())

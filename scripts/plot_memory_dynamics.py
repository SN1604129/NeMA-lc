import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# -------------------------
# Config
# -------------------------
CSV_PATH = Path("logs/memory_dynamics.csv")
OUT_DIR = Path("plots")
OUT_DIR.mkdir(exist_ok=True)

SMOOTH_WINDOW = 50  # moving average window

# -------------------------
# Load data
# -------------------------
df = pd.read_csv(CSV_PATH)

# Safety: sort by step
df = df.sort_values("step")

# -------------------------
# Helper: smooth curves
# -------------------------
def smooth(series, window=SMOOTH_WINDOW):
    return series.rolling(window=window, min_periods=1).mean()

# -------------------------
# Plot 1: Memory Utilization
# -------------------------
plt.figure(figsize=(6, 4))
plt.plot(df["step"], smooth(df["utilization"]), label="Utilization")
plt.xlabel("Training step")
plt.ylabel("Fraction of active slots")
plt.title("Memory Utilization Over Time")
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT_DIR / "utilization.png", dpi=200)
plt.close()

# -------------------------
# Plot 2: Average Memory Age
# -------------------------
plt.figure(figsize=(6, 4))
plt.plot(df["step"], smooth(df["avg_age"]), label="Avg age", color="orange")
plt.xlabel("Training step")
plt.ylabel("Average age")
plt.title("Memory Age Over Time")
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT_DIR / "avg_age.png", dpi=200)
plt.close()

# -------------------------
# Plot 3: Lifecycle Operation Rates
# -------------------------
plt.figure(figsize=(6, 4))
plt.plot(df["step"], smooth(df["writes"]), label="Write")
plt.plot(df["step"], smooth(df["updates"]), label="Update")
plt.plot(df["step"], smooth(df["forgets"]), label="Forget")
plt.xlabel("Training step")
plt.ylabel("Rate")
plt.title("Memory Lifecycle Operations")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT_DIR / "lifecycle_ops.png", dpi=200)
plt.close()

# -------------------------
# Plot 4: Loss components (optional but strong)
# -------------------------
if {"loss_task", "loss_write", "loss_forget", "loss_stability"}.issubset(df.columns):
    plt.figure(figsize=(6, 4))
    plt.plot(df["step"], smooth(df["loss_task"]), label="Task")
    plt.plot(df["step"], smooth(df["loss_write"]), label="Write")
    plt.plot(df["step"], smooth(df["loss_forget"]), label="Forget")
    plt.plot(df["step"], smooth(df["loss_stability"]), label="Stability")
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.title("Loss Components Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "loss_components.png", dpi=200)
    plt.close()

print("Plots saved to:", OUT_DIR.resolve())

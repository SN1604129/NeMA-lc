import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

plt.style.use("seaborn-v0_8-darkgrid")

LOG_DIR = Path("logs")
PLOT_DIR = Path("plots")
PLOT_DIR.mkdir(exist_ok=True)

RUNS = {
    "Full": LOG_DIR / "memory_dynamics_full.csv",
    "Ablate (τ=0.4, cap=0.7)": LOG_DIR / "memory_dynamics_ablate_tau04_cap07.csv",
}

SMOOTH = 50

def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing log file: {path}")
    df = pd.read_csv(path)
    if "step" not in df.columns:
        df["step"] = range(len(df))
    return df.sort_values("step")

dfs = {name: load_csv(path) for name, path in RUNS.items()}

# ---------------------------
# Ablation: Utilization
# ---------------------------
plt.figure(figsize=(8, 5))
for name, df in dfs.items():
    plt.plot(
        df["step"],
        df["utilization"].rolling(SMOOTH, min_periods=1).mean(),
        label=name,
    )

plt.xlabel("Training step")
plt.ylabel("Memory utilization")
plt.title("Ablation: Effect of Write Gate on Memory Utilization")
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_DIR / "ablation_utilization.png", dpi=200)
plt.show()

# ---------------------------
# Ablation: Avg retention age
# ---------------------------
plt.figure(figsize=(8, 5))
for name, df in dfs.items():
    plt.plot(
        df["step"],
        df["avg_age"].rolling(SMOOTH, min_periods=1).mean(),
        label=name,
    )

plt.xlabel("Training step")
plt.ylabel("Average retention age")
plt.title("Ablation: Effect of Write Gate on Memory Retention Age")
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_DIR / "ablation_avg_age.png", dpi=200)
plt.show()

print("Saved:", PLOT_DIR / "ablation_utilization.png")
print("Saved:", PLOT_DIR / "ablation_avg_age.png")

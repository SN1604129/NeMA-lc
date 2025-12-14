import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

FILES = {
    "Full": "logs/memory_dynamics_full.csv",
    "No Write": "logs/memory_dynamics_nowrite.csv",
    "No Forget": "logs/memory_dynamics_noforget.csv",
    "No Stability": "logs/memory_dynamics_nostability.csv",
}

OUT = Path("plots")
OUT.mkdir(exist_ok=True)

def smooth(x, w=50):
    return x.rolling(w, min_periods=1).mean()

plt.figure(figsize=(7, 5))

for label, path in FILES.items():
    df = pd.read_csv(path)
    df = df.sort_values("step")
    plt.plot(df["step"], smooth(df["utilization"]), label=label)

plt.xlabel("Training step")
plt.ylabel("Memory utilization")
plt.title("Ablation: Effect of Lifecycle Losses on Memory Utilization")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT / "ablation_utilization.png", dpi=200)
plt.close()

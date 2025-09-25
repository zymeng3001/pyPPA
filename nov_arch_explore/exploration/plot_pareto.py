import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Reload the newest dataset
file_path = "../data/sweep_data_with_sw_100M_mod.csv"
df = pd.read_csv(file_path)

# Clean column names
df = df.rename(columns=lambda x: x.strip().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_"))

# Normalize EDP and EADP for comparison
df["EDP_norm"] = (df["EDP_mJ_*_ms"] - df["EDP_mJ_*_ms"].min()) / (df["EDP_mJ_*_ms"].max() - df["EDP_mJ_*_ms"].min())
df["EADP_norm"] = (df["EADP_mJ_*_ms_*_um^2"] - df["EADP_mJ_*_ms_*_um^2"].min()) / (df["EADP_mJ_*_ms_*_um^2"].max() - df["EADP_mJ_*_ms_*_um^2"].min())

# Function for Pareto frontier
def pareto_frontier(df, x, y):
    points = df[[x, y]].values
    points = points[np.argsort(points[:, 0])]
    frontier = []
    min_y = np.inf
    for px, py in points:
        if py < min_y:
            frontier.append((px, py))
            min_y = py
    return np.array(frontier)

# Metrics to plot separately
metrics = {
    "Power_W": "Power (W)",
    "Area_um^2": "Area (um^2)",
    "TTFT_ms": "TTFT (ms)",
    "TPOT_ms": "TPOT (ms)",
    "Energy_per_token_mJ_token": "Energy per Token (mJ/token)",
    "EDP_norm": "Normalized EDP",
    "EADP_norm": "Normalized EADP"
}

# Generate 5 separate figures
for metric, label in metrics.items():
    pareto = pareto_frontier(df, metric, "best_val_loss")
    
    plt.figure(figsize=(8,6))
    plt.scatter(df[metric], df["best_val_loss"], s=5, alpha=0.3, label="All Points")
    plt.plot(pareto[:,0], pareto[:,1], c="red", linewidth=2, label="Pareto Frontier")
    plt.xlabel(label)
    plt.ylabel("Best Val Loss")
    plt.title(f"Pareto Frontier: {label} vs Validation Loss")
    plt.legend()
    plt.savefig(f"../plots/pareto/pareto_{metric}.png", dpi=500)


# filter designs with Energy per token < 500 mJ/token
df_2 = df[df["Energy_per_token_mJ_token"] < 500]
df_2 = df_2[df_2["best_val_loss"] < 4]

# for the same value of best_val_loss, keep the one with lowest Energy per token times Area (um^2)
df_2 = df_2.loc[df_2.groupby("best_val_loss")["Energy_per_token_mJ_token"].idxmin()]

# Generate 5 separate figures
for metric, label in metrics.items():
    pareto = pareto_frontier(df_2, metric, "best_val_loss")

    plt.figure(figsize=(8,6))
    plt.scatter(df_2[metric], df_2["best_val_loss"], s=5, alpha=0.3, label="Filtered Points")
    plt.plot(pareto[:,0], pareto[:,1], c="red", linewidth=2, label="Pareto Frontier")
    plt.xlabel(label)
    plt.ylabel("Best Val Loss")
    plt.title(f"Pareto Frontier: {label} vs Validation Loss")
    plt.legend()
    plt.savefig(f"../plots/pareto2/pareto2_{metric}.png", dpi=500)


# generate the figure of energy per token vs latency (TTFT)
plt.figure(figsize=(8,6))
plt.scatter(df["TTFT_ms"], df["Energy_per_token_mJ_token"], s=5, alpha=0.3)
plt.xlabel("TTFT (ms)")
plt.ylabel("Energy per Token (mJ/token)")
plt.title("Energy per Token vs TTFT")
# plt.xscale("log")
# plt.yscale("log")
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.savefig("../plots/pareto/energy_per_token_vs_ttft.png", dpi=500)



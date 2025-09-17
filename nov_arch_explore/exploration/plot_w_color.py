import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Reload the newest dataset
file_path = "../data/sweep_data_with_sw_100M_mod.csv"
df = pd.read_csv(file_path)

# Clean column names
df = df.rename(columns=lambda x: x.strip().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_"))

# Pick the frequency column (after renaming it's F_max_MHz)
freq_col = "F_max_MHz" if "F_max_MHz" in df.columns else "F_max (MHz)"

# Filter to timing-closed points using Clock Slack (ns) >= 0
# slack_col = "Clock_Slack_ns" if "Clock_Slack_ns" in df.columns else "Clock Slack (ns)"
# if slack_col in df.columns:
#     df[slack_col] = pd.to_numeric(df[slack_col], errors='coerce')
#     before_cnt = len(df)
#     df = df[df[slack_col] >= 0].copy()
#     after_cnt = len(df)
#     print(f"Filtered timing: kept {after_cnt}/{before_cnt} rows with {slack_col} >= 0")
# else:
#     print("Warning: Clock slack column not found; no timing filter applied.")

# Normalize EDP and EADP for comparison
df["EDP_norm"] = (df["EDP_mJ_*_ms"] - df["EDP_mJ_*_ms"].min()) / (df["EDP_mJ_*_ms"].max() - df["EDP_mJ_*_ms"].min())
df["EADP_norm"] = (df["EADP_mJ_*_ms_*_um^2"] - df["EADP_mJ_*_ms_*_um^2"].min()) / (df["EADP_mJ_*_ms_*_um^2"].max() - df["EADP_mJ_*_ms_*_um^2"].min())

# filter designs with Energy per token < 800 mJ/token
# df = df[df["Energy_per_token_mJ_token"] < 800]
df = df[df["best_val_loss"] < 4]

# for the same value of best_val_loss, keep the one with lowest Energy per token times Area (um^2)
df = df.loc[df.groupby("best_val_loss")["Energy_per_token_mJ_token"].idxmin()]

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

# Generate 2 separate figures
for metric, label in metrics.items():
    # Select top 10% by the current metric (>= 80th percentile)
    thr = df[metric].quantile(0.20)
    df_plot = df[df[metric] >= thr].copy()
    print(f"{metric}: plotting top 20% (n={len(df_plot)}) of total {len(df)}; threshold={thr:.4g}")

    pareto = pareto_frontier(df_plot, metric, "best_val_loss")

    plt.figure(figsize=(10,6))
    sc = plt.scatter(
        df_plot[metric], df_plot["best_val_loss"],
        s=20, alpha=0.7,
        c=df_plot[freq_col], cmap="viridis",
        label="Data Points"
    )
    # ...existing code...
    if len(pareto) > 0:
        plt.plot(pareto[:,0], pareto[:,1], c="red", linewidth=0.5, label="Pareto Frontier")
    plt.xlabel(label)
    plt.ylabel("Best Val Loss")
    plt.title(f"Pareto Frontier: {label} vs Validation Loss")
    plt.legend(loc="best")
    cbar = plt.colorbar(sc)
    cbar.set_label("Frequency (MHz)")
    plt.tight_layout()
    plt.savefig(f"../plots/pareto_{metric}.png", dpi=500)
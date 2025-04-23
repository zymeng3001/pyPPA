import pandas as pd
import matplotlib.pyplot as plt

# Load updated data
df = pd.read_csv("./data/sweep_data_5ns_with_sw.csv")

# Filter data for Val_Loss < 5.0
filtered_df = df[df["val_loss"] < 5.0]
# filtered_df = filtered_df[filtered_df["Energy per Token(mJ)"] < 2]

small_df = filtered_df[filtered_df["Energy per Token(uJ)"] < 1]

# Count the number of datapoints
num_datapoints = len(filtered_df)
print(f"Number of datapoints with Val_Loss < 5.0: {num_datapoints}")

# Create scatter plot: Energy vs. Token Delay, colored by Perplexity
plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    filtered_df["Token Delay(us)"],
    filtered_df["Energy per Token(uJ)"],
    c=filtered_df["val_loss"],  # Assuming val_loss is used as a proxy for perplexity
    cmap="plasma",
    alpha=0.85,
    edgecolor="k"
)

# Add colorbar for perplexity
cbar = plt.colorbar(scatter)
cbar.set_label("Validation Loss")

# Axis labels and title
plt.ylabel("Energy per Token (uJ)")
plt.xlabel("Token Delay (us)")
plt.title("Energy vs. Token Delay (Colored by Perplexity)")
plt.grid(True)
plt.tight_layout()

# Save the plot
plt.savefig("plots/Energy_vs_TokenDelay_by_val_loss.png", dpi=300)

############################################

# Create scatter plot: Energy vs. Token Delay, colored by Perplexity
plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    filtered_df["Energy per Token(uJ)"],
    filtered_df["val_loss"],
    c=filtered_df["n_embed"],  # Assuming val_loss is used as a proxy for perplexity
    cmap="plasma",
    alpha=0.85,
    edgecolor="k"
)

# Add colorbar for perplexity
cbar = plt.colorbar(scatter)
cbar.set_label("n_embed")

# Axis labels and title
plt.ylabel("Validation Loss")
plt.xlabel("Energy per Token (uJ)")
plt.title("Energy vs. Validation Loss")
plt.grid(True)
plt.tight_layout()

# Save the plot
plt.savefig("plots/Energy_vs_Perplexity_by_n_embed.png", dpi=300)

# Create scatter plot: Energy vs. Token Delay, colored by Perplexity
plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    small_df["Token Delay(us)"],
    small_df["Energy per Token(uJ)"],
    c=small_df["n_embed"],  # Assuming val_loss is used as a proxy for perplexity
    cmap="plasma",
    alpha=0.85,
    edgecolor="k"
)

# Add colorbar for perplexity
cbar = plt.colorbar(scatter)
cbar.set_label("n_embed")

# Axis labels and title
plt.ylabel("Energy per Token (uJ)")
plt.xlabel("Token Delay (us)")
plt.title("Energy vs. Token Delay (Colored by n_embed)")
plt.grid(True)
plt.tight_layout()

# Save the plot
plt.savefig("plots/Energy_vs_TokenDelay_by_n_embed_small.png", dpi=300)


# Create scatter plot: Energy vs. Token Delay, colored by Perplexity
plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    filtered_df["Energy per Token(uJ)"],
    filtered_df["val_loss"],
    c=filtered_df["Total Param Num"],  # Assuming val_loss is used as a proxy for perplexity
    cmap="plasma",
    alpha=0.85,
    edgecolor="k"
)

# Add colorbar for perplexity
cbar = plt.colorbar(scatter)
cbar.set_label("Total Param Num")

# Axis labels and title
plt.ylabel("Validation Loss")
plt.xlabel("Energy per Token (uJ)")
plt.title("Energy vs. Token Delay (Colored by param_size)")
plt.grid(True)
plt.tight_layout()

# Save the plot
plt.savefig("plots/Energy_vs_Perplexity_by_param_size.png", dpi=300)



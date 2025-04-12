import pandas as pd
import matplotlib.pyplot as plt

# Load the re-uploaded sweep data with Token Delay and Energy information
sweep_data_path = './sweep_data.csv'
sweep_df = pd.read_csv(sweep_data_path)

# Define relevant hyperparameters and ensure required metrics are available
factors_to_plot = ['Gbus Width', 'n_embed', 'n_heads', 'n_cols', 'Max Context Length']

# Ensure Token Per Second is computed if only Token Delay exists
if 'Token Per Second' not in sweep_df.columns and 'Token Delay(s)' in sweep_df.columns:
    sweep_df['Token Per Second'] = 1 / sweep_df['Token Delay(s)']

# Create scatter plots color-coded by each hyperparameter
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, factor in enumerate(factors_to_plot):
    scatter = axes[i].scatter(
        sweep_df['Token Per Second'],
        sweep_df['Energy per Token(mJ)'],
        c=sweep_df[factor],
        cmap='viridis',
        alpha=0.6
    )
    axes[i].set_title(f'Token Throughput vs Energy — Colored by {factor}')
    axes[i].set_xlabel('Token Per Second')
    axes[i].set_ylabel('Energy per Token (mJ)')
    fig.colorbar(scatter, ax=axes[i], label=factor)

# Remove unused subplot if necessary
if len(factors_to_plot) < len(axes):
    for j in range(len(factors_to_plot), len(axes)):
        fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig('Token_per_second_vs_energy.png')

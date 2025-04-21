import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv("data/sweep_data_5ns_with_sw.csv")

# Clean column names
df.rename(columns=lambda x: x.strip().lower().replace(' ', '_'), inplace=True)

# Remove duplicated columns if any
df = df.loc[:, ~df.columns.duplicated()]

print("Columns in the DataFrame:")
print(df.columns.tolist())

# Select and clean relevant data
# relevant_cols = ['n_heads', 'n_embed', 'max_context_length', 'val_loss', 'energy_per_token(uj)', 'token_delay(ms)']
# df_filtered = df[relevant_cols].dropna()

# select max_context_length = 128
df = df[df['n_embed'] != 768]

# Group by fixed parameters and collect trends where n_heads varies
# grouped = df_filtered.groupby(['n_embed'])

# trend_data = []
# for (n_embed), group in grouped:
#     if group['n_heads'].nunique() > 1:
#         group_sorted = group.sort_values('n_heads')
#         trend_data.append(group_sorted)

# trend_df = pd.concat(trend_data)

# Group by n_embed and plot val_loss vs n_heads for each group
plt.figure(figsize=(10, 6))

sns.lineplot(
    data=df,
    x='n_heads',
    y='val_loss',
    hue='n_embed',
    marker='o'
)

plt.title('Validation Loss vs Number of Heads (Grouped by n_embed)')
plt.xlabel('Number of Attention Heads')
plt.ylabel('Validation Loss')
plt.grid(True)
plt.legend(title='n_embed', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig("plots/val_loss_vs_n_heads.png", dpi=300)

plt.figure(figsize=(10, 6))

sns.lineplot(
    data=df,
    x='n_heads',
    y='energy_per_token(uj)',
    hue='n_embed',
    marker='o'
)

plt.title('Energy per Token vs Number of Heads (Grouped by n_embed)')
plt.xlabel('Number of Attention Heads')
plt.ylabel('Energy per Token (uJ)')
plt.grid(True)
plt.legend(title='n_embed', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("plots/energy_per_token_vs_n_heads.png", dpi=300)

plt.figure(figsize=(10, 6))

sns.lineplot(
    data=df,
    x='n_heads',
    y='token_delay(ms)',
    hue='n_embed',
    marker='o'
)

plt.title('Token Delay vs Number of Heads (Grouped by n_embed)')
plt.xlabel('Number of Attention Heads')
plt.ylabel('Token Delay (ms)')
plt.grid(True)
plt.legend(title='n_embed', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

plt.savefig("plots/token_delay_vs_n_heads.png", dpi=300)
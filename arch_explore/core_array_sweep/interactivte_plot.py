import pandas as pd
import plotly.express as px

# Load data
df = pd.read_csv("./data/sweep_data_5ns_with_sw.csv")

# Filter data for Val_Loss < 5.0
filtered_df = df[df["val_loss"] < 5.0]

# Create interactive scatter plot
fig = px.scatter(
    filtered_df,
    x="Energy per Token(mJ)",
    y="val_loss",
    color="n_embed",
    labels={
        "Energy per Token(mJ)": "Energy per Token (mJ)",
        "val_loss": "Validation Loss",
        "n_embed": "Embedding Size",
        "Token Delay(ms)": "Token Delay (ms)",
        "Gbus Width": "Gbus Width",
        "n_heads": "n_heads",
        "n_cols": "n_cols",
        "head_dim": "head_dim",
        "max_context_length": "max_context_length"
    },
    title="Energy vs. Validation Loss Colored by Embedding Size",
    hover_data=["Token Delay(ms)", "Energy per Token(mJ)", "val_loss", "n_embed", "Gbus Width", "n_heads", "n_cols", "head_dim", "max_context_length"]
)

fig.update_traces(
    marker=dict(size=8, line=dict(width=1, color='DarkSlateGrey'))
)

fig.show()

import pandas as pd
import plotly.express as px

# Load your data
df = pd.read_csv("Sweeping_sw.csv")

# Rename for clarity
df.rename(columns={
    'Val_loss': 'val_loss',
    'n_head': 'n_heads',
    'block_size': 'max_context_length'
}, inplace=True)

# Filter relevant columns and drop NA
df_subset = df[['val_loss', 'n_embd', 'n_heads', 'max_context_length']].dropna()

# drop the data with n_embd == 768
df_subset = df_subset[df_subset['n_embd'] != 768]

# Create 3D plot with val_loss as both color and size
fig = px.scatter_3d(
    df_subset,
    x='n_embd',
    y='n_heads',
    z='max_context_length',
    color='val_loss',
    size='val_loss',
    opacity=0.8,
    title='3D Projection: val_loss vs n_embd, n_heads (color/size = val_loss)',
    labels={'val_loss': 'Validation Loss', 'n_embd': 'Embedding Dim', 'n_heads': 'Attention Heads'}
)

fig.update_layout(scene=dict(
    xaxis_title='n_embd',
    yaxis_title='n_heads',
    zaxis_title='max_context_length',
))

fig.show()


import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

# Load data
df = pd.read_csv("data/sweep_data_5ns_with_sw.csv")

# Filter for readability
df = df[df['val_loss'] < 5.0]

# Get relevant metrics
x = df['Energy per Token(uJ)'].values
y = df['val_loss'].values
z = df['Token Delay(ms)'].values

# Pareto front detection
def is_pareto_efficient(costs):
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1) | np.all(costs[is_efficient] == c, axis=1)
            is_efficient[i] = True
    return is_efficient

costs = np.vstack((x, y, z)).T
pareto_mask = is_pareto_efficient(costs)
pareto_df = df[pareto_mask]

# Surface fit to Pareto front
X = np.vstack((pareto_df['Energy per Token(uJ)'], pareto_df['Token Delay(ms)'])).T
Y = pareto_df['val_loss']
reg = LinearRegression().fit(X, Y)

# Surface grid
x_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 40)
z_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 40)
xx, zz = np.meshgrid(x_range, z_range)
yy = reg.predict(np.c_[xx.ravel(), zz.ravel()]).reshape(xx.shape)

# Normalize Area for color
scaler = MinMaxScaler()
area_norm = scaler.fit_transform(df[['Area(um^2)']]).flatten()
pareto_area_norm = scaler.transform(pareto_df[['Area(um^2)']]).flatten()

# Build annotations for Pareto points
annotations = [
    f"embd:{int(row['n_embed'])}, head:{int(row['n_heads'])}, col:{int(row['n_cols'])}, gbus:{int(row['Gbus Width'])}, ctx:{int(row['max_context_length'])}"
    for _, row in pareto_df.iterrows()
]

# Create 3D interactive plot
fig = go.Figure()

# Plot all points
fig.add_trace(go.Scatter3d(
    x=df['Energy per Token(uJ)'],
    y=df['val_loss'],
    z=df['Token Delay(ms)'],
    mode='markers',
    marker=dict(size=4, color=area_norm, colorscale='Viridis', opacity=0.3),
    name='All Designs',
    hovertext=df.apply(lambda row: f"embd:{int(row['n_embed'])}, head:{int(row['n_heads'])}, ctx:{int(row['max_context_length'])}", axis=1),
    hoverinfo='text'
))

# Plot Pareto front points
fig.add_trace(go.Scatter3d(
    x=pareto_df['Energy per Token(uJ)'],
    y=pareto_df['val_loss'],
    z=pareto_df['Token Delay(ms)'],
    mode='markers+text',
    marker=dict(size=6, color=pareto_area_norm, colorscale='Plasma', colorbar=dict(title='Area (um²)')),
    text=annotations,
    hoverinfo='text',
    name='Pareto Front'
))

# Add fitted surface to Pareto front
fig.add_trace(go.Surface(
    x=x_range, y=yy, z=z_range, showscale=False,
    opacity=0.5, colorscale='YlGnBu', name='Fitted Surface'
))

# Layout
fig.update_layout(
    title="Design Space: Energy vs Val_loss vs Delay (Interactive)",
    scene=dict(
        xaxis_title="Energy per Token (uJ)",
        yaxis_title="Validation Loss",
        zaxis_title="Token Delay (ms)"
    ),
    width=1000,
    height=800
)

fig.show()

# import pandas as pd
# import numpy as np
# import plotly.graph_objects as go
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import MinMaxScaler

# # Load data
# df = pd.read_csv("data/sweep_data_5ns_with_sw.csv")

# # Filter for better readability
# df = df[df['val_loss'] < 5.0]

# # Get relevant columns
# x = df['Energy per Token(mJ)'].values
# y = df['val_loss'].values
# z = df['Token Delay(ms)'].values

# # Pareto front detection
# def is_pareto_efficient(costs):
#     is_efficient = np.ones(costs.shape[0], dtype=bool)
#     for i, c in enumerate(costs):
#         if is_efficient[i]:
#             is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1) | np.all(costs[is_efficient] == c, axis=1)
#             is_efficient[i] = True
#     return is_efficient

# costs = np.vstack((x, y, z)).T
# pareto_mask = is_pareto_efficient(costs)
# pareto_df = df[pareto_mask]

# # Fit a surface to Pareto points
# X = np.vstack((pareto_df['Energy per Token(mJ)'], pareto_df['Token Delay(ms)'])).T
# Y = pareto_df['val_loss']
# reg = LinearRegression().fit(X, Y)

# # Generate surface grid
# x_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 40)
# z_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 40)
# xx, zz = np.meshgrid(x_range, z_range)
# yy = reg.predict(np.c_[xx.ravel(), zz.ravel()]).reshape(xx.shape)

# # Annotate Pareto points
# annotations = [
#     f"embd:{int(row['n_embed'])}, head:{int(row['n_heads'])}, col:{int(row['n_cols'])}, gbus_width:{int(row['Gbus Width'])}, ctx:{int(row['max_context_length'])}"
#     for _, row in pareto_df.iterrows()
# ]

# # Normalize area for better coloring
# scaler = MinMaxScaler()
# area_norm = scaler.fit_transform(pareto_df[['Area(um^2)']]).flatten()

# # Interactive Plot
# fig = go.Figure()

# # Pareto front points
# fig.add_trace(go.Scatter3d(
#     x=pareto_df['Energy per Token(mJ)'],
#     y=pareto_df['val_loss'],
#     z=pareto_df['Token Delay(ms)'],
#     mode='markers+text',
#     marker=dict(size=6, color=area_norm, colorscale='Plasma', colorbar=dict(title='Area (um²)')),
#     text=annotations,
#     hoverinfo='text'
# ))

# # Surface
# fig.add_trace(go.Surface(
#     x=x_range, y=yy, z=z_range, showscale=False,
#     opacity=0.5, colorscale='Viridis', name='Fitted Surface'
# ))

# fig.update_layout(
#     title="Pareto Front Surface Fit (Interactive)",
#     scene=dict(
#         xaxis_title="Energy per Token(mJ)",
#         yaxis_title="Validation Loss",
#         zaxis_title="Token Delay (ms)"
#     ),
#     width=900,
#     height=700
# )

# fig.show()


# import pandas as pd
# import plotly.express as px

# # Load your data
# df = pd.read_csv("Sweeping_sw.csv")

# # Rename for clarity
# df.rename(columns={
#     'val_loss': 'val_loss',
#     'n_head': 'n_heads',
#     'block_size': 'max_context_length'
# }, inplace=True)

# # Filter relevant columns and drop NA
# df_subset = df[['val_loss', 'n_embd', 'n_heads', 'max_context_length']].dropna()

# # drop the data with n_embd == 768
# df_subset = df_subset[df_subset['n_embd'] != 768]

# # Create 3D plot with val_loss as both color and size
# fig = px.scatter_3d(
#     df_subset,
#     x='n_embd',
#     y='n_heads',
#     z='max_context_length',
#     color='val_loss',
#     size='val_loss',
#     opacity=0.8,
#     title='3D Projection: val_loss vs n_embd, n_heads (color/size = val_loss)',
#     labels={'val_loss': 'Validation Loss', 'n_embd': 'Embedding Dim', 'n_heads': 'Attention Heads'}
# )

# fig.update_layout(scene=dict(
#     xaxis_title='n_embd',
#     yaxis_title='n_heads',
#     zaxis_title='max_context_length',
# ))

# fig.show()


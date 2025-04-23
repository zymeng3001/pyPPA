import pandas as pd
import numpy as np

import numpy as np
import math
import sweep_utils

# Load data again
file_path = '../ppa_core_top_extracted_data.csv'
data = pd.read_csv(file_path)

# Creating a nested dictionary-based database
database = {}

# Populating the database with configuration as keys and relevant metrics as values
for _, row in data.iterrows():
    key = (row['Gbus width'], row['Wmem Depth'], row['Cache Depth'])
    database[key] = {
        'power': row['Power (W)'],
        'slack': row['Clock Slack (ns)'],
        'clk_period': row['Clock Period (ns) Entered'],
        'clk_min_period': row['Clock_Min_Period'],
        'area': row['Area (um^2)']
        # Additional metrics can be added here as needed
    }

df = pd.read_csv("data/Sweeping_sw.csv")

n_cols_range = np.arange(1, 33).tolist()
gbus_width_range = [16, 32, 64, 128]

total_count = 0
feasible_configs = []

for idx, row in df.iterrows():
    # Extract the values for each hyperparameter
    n_model = row['n_embd']
    n_heads = row['n_head']
    max_context_length = row['block_size']
    val_loss = row['Val_loss']
    n_layers = row['n_layer']
    ffn_ratio = 4

    for n_cols in n_cols_range:
        for gbus_width in gbus_width_range:
            if sweep_utils.is_valid_design(n_model, n_heads, n_cols, gbus_width, max_context_length):
                # Calculate wmem_depth and cache_depth
                wmem_depth = sweep_utils.get_wmem_depth(n_model, n_heads, n_cols, gbus_width)
                cache_depth = sweep_utils.get_cache_depth(n_model, n_heads, n_cols, gbus_width, max_context_length)

                # Retrieve core power and area from the database
                core_power = database.get((gbus_width, wmem_depth, cache_depth), {}).get('power', 'N/A') 
                core_area = database.get((gbus_width, wmem_depth, cache_depth), {}).get('area', 'N/A')
                clk_period = database.get((gbus_width, wmem_depth, cache_depth), {}).get('clk_period', 'N/A')
                clk_min_period = database.get((gbus_width, wmem_depth, cache_depth), {}).get('clk_min_period', 'N/A')
                slack = database.get((gbus_width, wmem_depth, cache_depth), {}).get('slack', 'N/A')

                ffn_ratio = 4

                core_wmem_size = wmem_depth * gbus_width / 8
                core_cache_size = cache_depth * gbus_width / 8

                total_wmem_size = core_wmem_size * n_heads * n_cols
                total_cache_size = core_cache_size * n_heads * n_cols

                # calculate perplexity
                perplexity = math.exp(val_loss) if val_loss != 'N/A' else 'N/A'

                if core_power != 'N/A' and core_area != 'N/A' and clk_period != 'N/A':
                    total_power = core_power * n_heads * n_cols
                    total_area = core_area * n_heads * n_cols

                    # token_delay = sweep_utils.get_token_delay(clk_period, n_model, gbus_width, n_heads, n_cols, max_context_length)
                    token_delay = sweep_utils.get_token_delay(clk_min_period, n_model, gbus_width, n_heads, n_cols, max_context_length, n_layers=n_layers, ffn_ratio=ffn_ratio)
                    
                    # calculate total parameter number
                    total_param_num = 4 * n_model * n_model + 2*ffn_ratio*n_model*n_model

                    energy_per_token = total_power * token_delay / 1000
                    total_mac_num = int(gbus_width / 8) * n_heads * n_cols

                    # Store the results in a list or print them
                    feasible_configs.append({
                        'Gbus Width': gbus_width,
                        'n_embed': n_model,
                        'n_heads': n_heads,
                        'n_cols': n_cols,
                        'head_dim': int(n_model / n_heads),
                        'max_context_length': max_context_length,
                        'n_layers': 6,
                        'core_wmem_size': core_wmem_size,
                        'core_cache_size': core_cache_size,
                        'total_wmem_size': total_wmem_size,
                        'total_cache_size': total_cache_size,
                        'clk_period': clk_period,
                        'clk_min_period': clk_min_period,
                        'slack': slack,
                        'Wmem Depth': wmem_depth,
                        'Cache Depth': cache_depth,
                        'Max Context Length': max_context_length,
                        'Total MAC Num': total_mac_num,
                        'Total Param Num': total_param_num,
                        'Power(mW)': total_power,
                        'Area(um^2)': total_area,
                        'Token Delay(us)': token_delay,
                        'Token Per Second': 1000000 / token_delay,
                        'Energy per Token(uJ)': energy_per_token,
                        "val_loss": val_loss,
                        "Perplexity": perplexity
                    })
                    total_count += 1

print(f"Total number of feasible configurations: {total_count}")
feasible_df = pd.DataFrame(feasible_configs)

# Save to CSV
feasible_csv_path = './data/sweep_data_5ns_with_sw.csv'
feasible_df.to_csv(feasible_csv_path, index=True)

    # Print the values
    # print(f"n_embd: {n_embd}, n_heads: {n_heads}, block_size: {block_size}, val_loss: {val_loss}")







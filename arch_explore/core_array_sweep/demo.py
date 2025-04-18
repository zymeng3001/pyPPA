# Re-import necessary libraries as the environment was reset
import pandas as pd
import matplotlib.pyplot as plt
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

# Example of querying the database
query_example = (32, 256, 128)  # (Gbus width, Wmem Depth, Cache Depth)
query_result = database.get(query_example, "Configuration not found")

print(f"Query result for configuration {query_example}: {query_result.get('power', 'N/A')} W")


gbus_width_range = [16, 32, 64, 128]
n_embed_range = [128, 192, 256, 384, 512, 768, 1024]
n_heads_range = np.arange(1, 17).tolist()
n_cols_range = np.arange(1, 33).tolist()
max_context_length_range = [64, 128, 192, 256, 512, 768, 1024]

# count the total number of feasible configurations
total_count = 0
feasible_configs = []

for gbus_width in gbus_width_range:
    for n_model in n_embed_range:
        for n_heads in n_heads_range:
            for n_cols in n_cols_range:
                for max_context_length in max_context_length_range:
                    # Check if the configuration is feasible
                    if sweep_utils.is_valid_design(n_model, n_heads, n_cols, gbus_width, max_context_length):
                        # total_count += 1
                        wmem_depth = sweep_utils.get_wmem_depth(n_model, n_heads, n_cols, gbus_width)
                        cache_depth = sweep_utils.get_cache_depth(n_model, n_heads, n_cols, gbus_width, max_context_length)
                        core_power = database.get((gbus_width, wmem_depth, cache_depth), {}).get('power', 'N/A') 
                        core_area = database.get((gbus_width, wmem_depth, cache_depth), {}).get('area', 'N/A')
                        clk_period = database.get((gbus_width, wmem_depth, cache_depth), {}).get('clk_period', 'N/A')
                        clk_min_period = database.get((gbus_width, wmem_depth, cache_depth), {}).get('clk_min_period', 'N/A')
                        slack = database.get((gbus_width, wmem_depth, cache_depth), {}).get('slack', 'N/A')

                        if core_power != 'N/A' and core_area != 'N/A' and clk_period != 'N/A':
                        
                            total_power = core_power * n_heads *n_cols
                            total_area = core_area * n_heads * n_cols

                            token_delay = sweep_utils.get_token_delay(clk_period, n_model, gbus_width, n_heads, n_cols, max_context_length)
                            energy_per_token = total_power * token_delay / 1000
                            total_mac_num = int(gbus_width / 8) * n_heads * n_cols
                            # print(f"Feasible configuration: Gbus Width: {gbus_width}, n_embed: {n_model}, n_heads: {n_heads}, n_cols: {n_cols}, Max Context Length: {max_context_length}")
                            # print(f"Power: {core_power} W, Area: {core_area} um^2")
                            # print(f"Wmem Depth: {wmem_depth}, Cache Depth: {cache_depth}")
                            # print(f"Total Count: {total_count}")
                            # print()

                            core_wmem_size = wmem_depth * gbus_width / 8
                            core_cache_size = cache_depth * gbus_width / 8

                            total_wmem_size = core_wmem_size * n_heads * n_cols
                            total_cache_size = core_cache_size * n_heads * n_cols
                            
                            # Store the feasible configuration
                            feasible_configs.append({
                                'Gbus Width': gbus_width,
                                'n_embed': n_model,
                                'n_heads': n_heads,
                                'n_cols': n_cols,
                                'head_fim': int(n_model / n_heads),
                                'max_context_length': max_context_length,
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
                                'Power(mW)': total_power,
                                'Area(um^2)': total_area,
                                'Token Delay(ms)': token_delay,
                                'Token Per Second': 1000 / token_delay,
                                'Energy per Token(mJ)': energy_per_token
                            })
                            total_count += 1

print(f"Total number of feasible configurations: {total_count}")

feasible_df = pd.DataFrame(feasible_configs)

# Save to CSV
feasible_csv_path = './sweep_data_5ns_new.csv'
feasible_df.to_csv(feasible_csv_path, index=True)


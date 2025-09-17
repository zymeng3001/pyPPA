# iterate through the sw swweep data and map it to sw_hw data
import pandas as pd
import numpy as np
import math
import sweep_utils

# Load data again
file_path = '../core_top_sweep_data/ppa_nov_core_top_extracted_data_9_1.csv'
data = pd.read_csv(file_path)

# Creating a nested dictionary-based database
database = {}

# Populating the database with configuration as keys and relevant metrics as values
for _, row in data.iterrows():
    key = (row['MAC NUM'], row['Wmem Depth'], row['Cache Depth'], row['Clock Period (ns) Entered'])
    database[key] = {
        'power': row['Power (W)'],
        'slack': row['Clock Slack (ns)'],
        'clk_min_period': row['Clock_Min_Period'],
        'area': row['Area (um^2)']
        # Additional metrics can be added here as needed
    }

df = pd.read_csv("../data/sw_sweep_data.csv")

n_cols_range = np.arange(1, 99).tolist()
mac_num = [4,8,16,32]
clk_periods = [3,4,5,6]
sequence_length = 256

total_count = 0
feasible_configs = []

for idx, row in df.iterrows():
    n_model = row['n_embd']
    n_heads = row['n_head']
    max_context_length = row['block_size']
    val_loss = row['best_val_loss']
    n_layers = row['n_layer']
    ffn_ratio = row['mlp_expansion_factor']
    for clk_period in clk_periods:
        for n_cols in n_cols_range:
            for n_macs in mac_num:
                if sweep_utils.is_feasible(n_model, n_heads, n_cols, n_macs, max_context_length):
                    # Calculate wmem_depth and cache_depth
                    wmem_depth = sweep_utils.get_wmem_depth(n_model, n_heads, n_cols, n_macs, ffn_ratio)
                    cache_depth = sweep_utils.get_cache_depth(n_model, n_heads, n_cols, max_context_length, n_macs)

                    # Retrieve core power and area from the database
                    core_power = database.get((n_macs, wmem_depth, cache_depth, clk_period), {}).get('power', 'N/A') 
                    core_area = database.get((n_macs, wmem_depth, cache_depth, clk_period), {}).get('area', 'N/A')
                    clk_min_period = database.get((n_macs, wmem_depth, cache_depth, clk_period), {}).get('clk_min_period', 'N/A')
                    slack = database.get((n_macs, wmem_depth, cache_depth, clk_period), {}).get('slack', 'N/A')

                    # calculate perplexity
                    perplexity = math.exp(val_loss) if val_loss != 'N/A' else 'N/A'

                    if core_power != 'N/A' and core_area != 'N/A' and clk_min_period != 'N/A':
                        total_count += 1
                        gbus_width = n_macs * 8
                        ttft = sweep_utils.get_TTFT(clk_min_period, n_model, n_macs, n_heads, n_cols, max_context_length, ffn_ratio=ffn_ratio, n_layers=n_layers, sequence_length=sequence_length)
                        tpot = sweep_utils.get_TPOT(clk_min_period, n_model, n_macs, n_heads, n_cols, max_context_length, ffn_ratio=ffn_ratio, n_layers=n_layers, sequence_length=sequence_length)
                        throughput = 1/(tpot) if tpot != 0 else 'N/A'

                        total_area = core_area * n_heads * n_cols
                        total_power = core_power * n_heads * n_cols
                        feasible_configs.append({
                            'n_embd': n_model,
                            'n_head': n_heads,
                            'block_size': max_context_length,
                            'n_layer': n_layers,
                            'mlp_expansion_factor': ffn_ratio,
                            'n_cols': n_cols,
                            'MAC NUM': n_macs,
                            'Wmem Depth': wmem_depth,
                            'Cache Depth': cache_depth,
                            'best_val_loss': val_loss,
                            'perplexity': perplexity,
                            'Power (W)': total_power,
                            'Area (um^2)': total_area,
                            'Clock Period (ns) Entered': clk_period,
                            'Clock_Min_Period': clk_min_period,
                            'F_max (MHz)': 1000/clk_min_period if clk_min_period != 0 else 'N/A',
                            'Clock Slack (ns)': slack,
                            'TTFT (ms)': ttft * 1000,
                            'TPOT (ms)': tpot * 1000,
                            'Throughput (tokens/s)': throughput,
                            'Energy per token (mJ/token)': total_power / throughput * 1000,
                            'EDP (mJ * ms)': (total_power/throughput/1e6) * ttft,
                            'EADP (mJ * ms * um^2)': (total_power/throughput/1e6) * ttft * total_area
                        })

print(f"Total number of feasible configurations: {total_count}")
feasible_df = pd.DataFrame(feasible_configs)

# Save to CSV
feasible_csv_path = '../data/sweep_data_with_sw_100M_mod.csv'
feasible_df.to_csv(feasible_csv_path, index=True)


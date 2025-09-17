import pandas as pd
import matplotlib.pyplot as plt
from sweep_utils import get_token_delay, get_wmem_depth, get_cache_depth, is_feasible, sram_metrics

# Fix n_embed, context length, layer numer and expansion factor
# explore the tradeoffs of add more heads and columns 

data = pd.read_csv("../core_top_sweep_data/ppa_nov_core_top_extracted_data_5ns.csv")

# Creating a nested dictionary-based database
database = {}

# Populating the database with configuration as keys and relevant metrics as values
for _, row in data.iterrows():
    key = (row['MAC NUM'], row['Wmem Depth'], row['Cache Depth'])
    database[key] = {
        'power': row['Power (W)'],
        'slack': row['Clock Slack (ns)'],
        'clk_period': row['Clock Period (ns) Entered'],
        'clk_min_period': row['Clock_Min_Period'],
        'area': row['Area (um^2)']
        # Additional metrics can be added here as needed
    }

n_embed = [384,512,768,1024,1536]
n_heads_range = [2,4,8,16]
max_context_length = [256,512,1024]
n_cols_range = [1,2,4,8,16]
n_macs_range = [8,16,32]
mlp_expansion_factor = 4
n_layers = 6

sequence_length = 256
cnt = 0

feasible_configs = []


# sweep through the design space of the above 5 parameters
# pack them into an object and then sweep
for n_heads in n_heads_range:
    for n_cols in n_cols_range:
        for n_macs in n_macs_range:
            if not is_feasible(n_embed, n_heads, n_cols, n_macs, max_context_length):
                continue

            head_dim = n_embed // n_heads
            wmem_depth = get_wmem_depth(n_embed, n_heads, n_cols, n_macs)
            cache_depth = get_cache_depth(n_embed, n_heads, n_cols, max_context_length, n_macs)

            print("wmem_depth, cache_depth, n_macs:", wmem_depth, cache_depth, n_macs)

            # Retrieve core power and area from the database
            core_power = database.get((n_macs, wmem_depth, cache_depth), {}).get('power', 'N/A') 
            core_area = database.get((n_macs, wmem_depth, cache_depth), {}).get('area', 'N/A')
            clk_period = database.get((n_macs, wmem_depth, cache_depth), {}).get('clk_period', 'N/A')
            clk_min_period = database.get((n_macs, wmem_depth, cache_depth), {}).get('clk_min_period', 'N/A')
            slack = database.get((n_macs, wmem_depth, cache_depth), {}).get('slack', 'N/A')

            if core_power != 'N/A' and core_area != 'N/A' and clk_period != 'N/A':
                cnt += 1
                gbus_width = n_macs * 8
                wmem_p_leak, wmem_e_read = sram_metrics(n_macs * 8, wmem_depth)
                cache_p_leak, cache_e_read = sram_metrics(n_macs * 8, cache_depth)

                total_power = (core_power + wmem_p_leak + cache_p_leak) * n_heads * n_cols
                total_area = core_area * n_heads * n_cols

                token_delay = get_token_delay(clk_min_period, n_embed, gbus_width, n_heads, n_cols, max_context_length, sequence_length=sequence_length, n_layers=n_layers, ffn_ratio=mlp_expansion_factor)
                
                # calculate total parameter number
                total_param_num = 4 * n_embed * n_embed + 2*mlp_expansion_factor*n_embed*n_embed + 50257 * n_embed

                wmem_access_energy = wmem_e_read * wmem_depth * n_heads * n_cols * 2 # (read + write) (pJ)
                cache_access_energy = (sequence_length * n_embed * 2 / n_cols / n_heads / n_macs * (1 + n_macs) ) / sequence_length * n_cols * n_heads # (read + write) (pJ)
                sram_access_energy = wmem_access_energy + cache_access_energy

                energy_per_token = (total_power * token_delay + sram_access_energy) / 1000 # (nJ)
                total_mac_num = int(gbus_width / 8) * n_heads * n_cols

                feasible_configs.append({
                        'Gbus Width': gbus_width,
                        'n_embed': n_embed,
                        'n_heads': n_heads,
                        'n_cols': n_cols,
                        'head_dim': int(n_embed / n_heads),
                        'max_context_length': max_context_length,
                        'ffn_ratio': mlp_expansion_factor,
                        'n_layers': n_layers,
                        'mac_num': n_macs,
                        # 'core_wmem_size': core_wmem_size,
                        # 'core_cache_size': core_cache_size,
                        # 'total_wmem_size': total_wmem_size,
                        # 'total_cache_size': total_cache_size,
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
                        'Energy per Token(nJ)': energy_per_token,
                        'sram_access_energy(pJ)': sram_access_energy,
                        # "val_loss": val_loss,
                        # "Perplexity": perplexity,
                        # "Next Token Accuracy": next_token_accuracy
                    })

            else:
                print(f"Wmem Depth: {wmem_depth}, Cache Depth: {cache_depth}, mac_num: {n_macs} not found in database.")


print(f"Total feasible configurations: {cnt}")
feasible_df = pd.DataFrame(feasible_configs)

# Save to CSV
feasible_csv_path = '../data/sweep_data_5ns_768n_512s_4m.csv'
feasible_df.to_csv(feasible_csv_path, index=True)





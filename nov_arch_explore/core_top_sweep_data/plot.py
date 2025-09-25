# import pandas as pd
# import matplotlib.pyplot as plt
# import os

# # Read the CSV data
# df = pd.read_csv("ppa_nov_core_top_extracted_data_5ns.csv")

# # Filter for widths 8 and 16
# # df = df[df['width_bits'].isin([8, 16])]

# # # Sort by width then depth
# # df = df.sort_values(['Wmem Depth', 'Cache Depth', 'MAC NUM'])

# # Ensure directory for plots
# os.makedirs("figs_coretop", exist_ok=True)


# # # Create a 2x2 subplot grid
# # fig, axs = plt.subplots(2, 2, figsize=(12, 10))
# # axs = axs.flatten()

# # # Plot 1: Frequency vs depth
# # for w, group in grouped:
# #     axs[0].plot(group['depth_words'], group['frequency_MHz'], marker='o', label=f'width {w}')
# # axs[0].set_xlabel('Depth (words)')
# # axs[0].set_ylabel('Frequency (MHz)')
# # axs[0].set_title('Frequency vs Depth')
# # axs[0].legend()
# # axs[0].grid(True)

# # # Plot 2: Read Power vs depth
# # for w, group in grouped:
# #     axs[1].plot(group['depth_words'], group['read_power'], marker='o', label=f'width {w}')
# # axs[1].set_xlabel('Depth (words)')
# # axs[1].set_ylabel('Read Power (mW)')
# # axs[1].set_title('Read Power vs Depth')
# # axs[1].legend()
# # axs[1].grid(True)

# # # Plot 3: Write Power vs depth
# # for w, group in grouped:
# #     axs[2].plot(group['depth_words'], group['write_power'], marker='o', label=f'width {w}')
# # axs[2].set_xlabel('Depth (words)')
# # axs[2].set_ylabel('Write Power (mW)')
# # axs[2].set_title('Write Power vs Depth')
# # axs[2].legend()
# # axs[2].grid(True)

# # # Plot 4: Leakage vs depth
# # for w, group in grouped:
# #     axs[3].plot(group['depth_words'], group['leakage_power'], marker='s', label=f'width {w}')
# # axs[3].set_xlabel('Depth (words)')
# # axs[3].set_ylabel('Leakage Power (mW)')
# # axs[3].set_title('Leakage vs Depth')
# # axs[3].legend()
# # axs[3].grid(True)

# # plt.tight_layout()
# # plt.savefig('fig/sram_summary_plots.png')

# # # Plot 5: energy(pJ) per read vs depth
# # fig, ax = plt.subplots(figsize=(8, 6))
# # for w, group in grouped:
# #     ax.plot(group['depth_words'], group['read_power']/group['frequency_MHz'], marker='x', label=f'width {w}')
# # ax.set_xlabel('Depth (words)')
# # ax.set_ylabel('Energy per read (pJ)')
# # ax.set_title('Energy per Read vs Depth')
# # ax.legend()
# # ax.grid(True)
# # plt.tight_layout()
# # plt.savefig('fig/sram_energy_per_read.png')

# # # Plot 6: energy(pJ) per write vs depth
# # fig, ax = plt.subplots(figsize=(8, 6))
# # for w, group in grouped:
# #     ax.plot(group['depth_words'], group['write_power']/group['frequency_MHz'], marker='x', label=f'width {w}')
# # ax.set_xlabel('Depth (words)')
# # ax.set_ylabel('Energy per Write (pJ)')
# # ax.set_title('Energy per Write vs Depth')
# # ax.legend()
# # ax.grid(True)
# # plt.tight_layout()
# # plt.savefig('fig/sram_energy_per_write.png')

# # Unique sorted kv_cache_depth values
# for cache_depth in sorted(df['kv_cache_depth'].unique()):
#     # Filter for current cache depth
#     subdf = df[df['kv_cache_depth'] == cache_depth]

#     # Group by MAC NUM
#     grouped = subdf.groupby('MAC NUM')

#     # Create plot
#     fig, ax = plt.subplots(figsize=(8, 6))
#     for mac_num, group in grouped:
#         ax.plot(
#             group['Wmem Depth'],
#             group['Power (W)'] * 1000,
#             marker='o',
#             label=f'MAC NUM {mac_num}'
#         )

#     ax.set_title(f'Power vs Wmem Depth @ Cache Depth {cache_depth}')
#     ax.set_xlabel('Wmem Depth')
#     ax.set_ylabel('Power (mW)')
#     ax.legend()
#     ax.grid(True)
#     plt.tight_layout()
#     plt.savefig(f'figs_coretop/coretop_power_cache{cache_depth}.png')
#     plt.close()

# print("✅ All plots saved in `figs_coretop/`.")
import pandas as pd
import matplotlib.pyplot as plt
import os

# Read data
df = pd.read_csv("ppa_nov_core_top_extracted_data_no_sram.csv")

# 🔍 Filter only rows where timing slack is met
# df = df[df['Clock Slack (ns)'] >= 0]

df = df[df['Clock Period (ns) Entered'] == 5]

# Aggregate: average power across duplicate entries
agg_df = df.groupby(['Cache Depth', 'Wmem Depth', 'MAC NUM'])['Power (W)'].mean().reset_index()

# Ensure directory exists
dir_name = "figs_coretop_without_sram"
os.makedirs(dir_name, exist_ok=True)

# Loop over Wmem Depths
for wmem_depth in sorted(agg_df['Wmem Depth'].unique()):
    subdf = agg_df[agg_df['Wmem Depth'] == wmem_depth]

    # Group by Cache Depth
    grouped = subdf.groupby('Cache Depth')

    fig, ax = plt.subplots(figsize=(8, 6))
    for cache_depth, group in grouped:
        group = group.sort_values('MAC NUM')  # Ensure proper x-axis order
        ax.plot(
            group['MAC NUM'],
            group['Power (W)'] * 1000,
            marker='o',
            label=f'Cache Depth {cache_depth}'
        )

    ax.set_title(f'Power vs MAC NUM @ Wmem Depth {wmem_depth}')
    ax.set_xlabel('MAC NUM')
    ax.set_ylabel('Power (mW)')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(f'{dir_name}/coretop_power_vs_mac_wmem{wmem_depth}.png')
    plt.close()

print("✅ Slack-filtered plots saved in `figs_coretop/`.") 

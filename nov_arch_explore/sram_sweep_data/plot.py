import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV data
df = pd.read_csv("ppa_sram_data_3ns.csv")

# Filter for widths 8 and 16
# df = df[df['width_bits'].isin([8, 16])]

# Sort by width then depth
df = df.sort_values(['sram width', 'sram depth'])

# Group by width
grouped = df.groupby('sram width')

# # Create a 2x2 subplot grid
# fig, axs = plt.subplots(2, 2, figsize=(12, 10))
# axs = axs.flatten()

# # Plot 1: Frequency vs depth
# for w, group in grouped:
#     axs[0].plot(group['depth_words'], group['frequency_MHz'], marker='o', label=f'width {w}')
# axs[0].set_xlabel('Depth (words)')
# axs[0].set_ylabel('Frequency (MHz)')
# axs[0].set_title('Frequency vs Depth')
# axs[0].legend()
# axs[0].grid(True)

# # Plot 2: Read Power vs depth
# for w, group in grouped:
#     axs[1].plot(group['depth_words'], group['read_power'], marker='o', label=f'width {w}')
# axs[1].set_xlabel('Depth (words)')
# axs[1].set_ylabel('Read Power (mW)')
# axs[1].set_title('Read Power vs Depth')
# axs[1].legend()
# axs[1].grid(True)

# # Plot 3: Write Power vs depth
# for w, group in grouped:
#     axs[2].plot(group['depth_words'], group['write_power'], marker='o', label=f'width {w}')
# axs[2].set_xlabel('Depth (words)')
# axs[2].set_ylabel('Write Power (mW)')
# axs[2].set_title('Write Power vs Depth')
# axs[2].legend()
# axs[2].grid(True)

# # Plot 4: Leakage vs depth
# for w, group in grouped:
#     axs[3].plot(group['depth_words'], group['leakage_power'], marker='s', label=f'width {w}')
# axs[3].set_xlabel('Depth (words)')
# axs[3].set_ylabel('Leakage Power (mW)')
# axs[3].set_title('Leakage vs Depth')
# axs[3].legend()
# axs[3].grid(True)

# plt.tight_layout()
# plt.savefig('fig/sram_summary_plots.png')

# # Plot 5: energy(pJ) per read vs depth
# fig, ax = plt.subplots(figsize=(8, 6))
# for w, group in grouped:
#     ax.plot(group['depth_words'], group['read_power']/group['frequency_MHz'], marker='x', label=f'width {w}')
# ax.set_xlabel('Depth (words)')
# ax.set_ylabel('Energy per read (pJ)')
# ax.set_title('Energy per Read vs Depth')
# ax.legend()
# ax.grid(True)
# plt.tight_layout()
# plt.savefig('fig/sram_energy_per_read.png')

# # Plot 6: energy(pJ) per write vs depth
# fig, ax = plt.subplots(figsize=(8, 6))
# for w, group in grouped:
#     ax.plot(group['depth_words'], group['write_power']/group['frequency_MHz'], marker='x', label=f'width {w}')
# ax.set_xlabel('Depth (words)')
# ax.set_ylabel('Energy per Write (pJ)')
# ax.set_title('Energy per Write vs Depth')
# ax.legend()
# ax.grid(True)
# plt.tight_layout()
# plt.savefig('fig/sram_energy_per_write.png')

fig, ax = plt.subplots(figsize=(8, 6))
for w, group in grouped:
    ax.plot(group['sram depth'], group['Power (W)']*1000, marker='x', label=f'width {w}')
ax.set_xlabel('Depth (words)')
ax.set_ylabel('Power (mW)')
ax.set_title('Power (mW) vs Depth')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig('sram_power.png')

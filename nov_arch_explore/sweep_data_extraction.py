import os
import json
import csv

# Get the absolute path based on where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Move up to the parent directory
parent_dir = os.path.dirname(script_dir)

# Base directory containing all folders with ppa.json files
base_dir = os.path.join(parent_dir, "runs", "core_top", "1_core_top_sweep")

# Define the output CSV file path
output_csv_path = os.path.join(parent_dir, "nov_arch_explore/core_top_sweep_data", "ppa_nov_core_top_extracted_data_10_1.csv")

# Column headers for CSV
csv_columns = [
    "Trial Number", "Iteration Number", "Clock Period (ns) Entered", "Clock_Min_Period", "Clock Slack (ns)", 
    "Clock_Worst_Slack", "Wmem Depth", "Cache Depth", "MAC NUM", "Total Period (ns)", "ABC Max Fanout", 
    "ABC Map Effort", "SYNTH_HIERARCHICAL", "Power (W)", "Area (um^2)", "Total Cells", "Sequential Cells", 
    "Combinational Cells", "Sequential_Internal", "Sequential_Switching", "Sequential_Leakage", 
    "Sequential_Total", "Combinational_Internal", "Combinational_Switching", "Combinational_Leakage", 
    "Combinational_Total", "Num_Wires", "Num_Wire_Bits", "Num_Public_Wires", 
    "Num_Public_Wire_Bits", "Num_Cells", "Chip_Area", "Num_Memories", "Num_Memory_Bits", 
    "Num_Processes","Total Time(s)",
]

# Open the CSV file for writing
with open(output_csv_path, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=csv_columns)
    writer.writeheader()  # Write the header row

    # Iterate over all folders in base directory
    for i, iteration in enumerate(sorted(os.listdir(base_dir))):
        folder_path = os.path.join(base_dir, iteration)
        ppa_json_path = os.path.join(folder_path, "ppa.json")
        config_json_path = os.path.join(folder_path, "config.json")

        # Check if folder contains a ppa.json file
        if os.path.isdir(folder_path) and os.path.exists(ppa_json_path):
            try:
                # Load the JSON file
                with open(ppa_json_path, "r") as pj:
                    ppa_data = json.load(pj)

                # Load the config.json file to get hyperparameters
                with open(config_json_path, "r") as cj:
                    config_data = json.load(cj)

                # Extract relevant data
                data = {
                    "Trial Number": ppa_data.get("run_number", "N/A"),
                    "Iteration Number": i,
                    "ABC Max Fanout": config_data["flow_config"].get("ABC_MAX_FANOUT", "N/A"),
                    "ABC Map Effort": config_data["flow_config"].get("ABC_MAP_EFFORT", "N/A"),
                    "SYNTH_HIERARCHICAL": config_data["flow_config"].get("SYNTH_HIERARCHICAL", "N/A"),
                    "Power (W)": ppa_data["ppa_stats"]["power_report"]["total"]["total_power"],
                    "Clock Slack (ns)": ppa_data["ppa_stats"]["sta"]["clk"]["clk_slack"],
                    "Clock Period (ns) Entered": ppa_data["hyperparameters"]["clk_period"],
                    "Wmem Depth": ppa_data["hyperparameters"]["wmem_depth"],
                    "Cache Depth": ppa_data["hyperparameters"]["kv_cache_depth"],
                    "MAC NUM": ppa_data["hyperparameters"]["mac_num"],

                    "Total Period (ns)": (
                        ppa_data["ppa_stats"]["sta"]["clk"]["clk_period"] + 
                        ppa_data["ppa_stats"]["sta"]["clk"]["clk_slack"]
                    ),
                    "Total Cells": ppa_data["synth_stats"].get("num_cells", "N/A"),
                    "Area (um^2)": ppa_data["synth_stats"].get("module_area", "N/A"),
                    "Sequential Cells": ppa_data["ppa_stats"].get("num_sequential_cells", "N/A"),
                    "Combinational Cells": ppa_data["ppa_stats"].get("num_combinational_cells", "N/A"),
                    "Clock_Min_Period": ppa_data["ppa_stats"]["sta"]["clk"].get("clk_period", "N/A"),
                    "Clock_Worst_Slack": ppa_data["ppa_stats"]["sta"]["clk"].get("clk_slack", "N/A"),
                    "Sequential_Internal": ppa_data["ppa_stats"]["power_report"]["sequential"]["internal_power"],
                    "Sequential_Switching": ppa_data["ppa_stats"]["power_report"]["sequential"]["switching_power"],
                    "Sequential_Leakage": ppa_data["ppa_stats"]["power_report"]["sequential"]["leakage_power"],
                    "Sequential_Total": ppa_data["ppa_stats"]["power_report"]["sequential"]["total_power"],
                    "Combinational_Internal": ppa_data["ppa_stats"]["power_report"]["combinational"]["internal_power"],
                    "Combinational_Switching": ppa_data["ppa_stats"]["power_report"]["combinational"]["switching_power"],
                    "Combinational_Leakage": ppa_data["ppa_stats"]["power_report"]["combinational"]["leakage_power"],
                    "Combinational_Total": ppa_data["ppa_stats"]["power_report"]["combinational"]["total_power"],
                    "Num_Wires": ppa_data["synth_stats"].get("num_wires", "N/A"),
                    "Num_Wire_Bits": ppa_data["synth_stats"].get("num_wire_bits", "N/A"),
                    "Num_Public_Wires": ppa_data["synth_stats"].get("num_public_wires", "N/A"),
                    "Num_Public_Wire_Bits": ppa_data["synth_stats"].get("num_public_wire_bits", "N/A"),
                    "Num_Cells": ppa_data["synth_stats"].get("num_cells", "N/A"),
                    "Chip_Area": ppa_data["synth_stats"].get("module_area", "N/A"),
                    "Num_Memories": ppa_data["synth_stats"].get("num_memories", "N/A"),
                    "Num_Memory_Bits": ppa_data["synth_stats"].get("num_memory_bits", "N/A"),
                    "Num_Processes": ppa_data["synth_stats"].get("num_processes", "N/A"),
                    "Total Time(s)": ppa_data["total_time_taken"].get("total_seconds", "N/A"),
                    
                }

                # Write data to CSV file
                writer.writerow(data)
                file.flush()
                print(f"✔ Extracted data from: {folder_path}")

            except (json.JSONDecodeError, KeyError) as e:
                print(f"❌ Error processing {folder_path}: {e}")

print(f"\n✅ All data extracted and saved to: {output_csv_path}")
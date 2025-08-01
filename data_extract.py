import yaml
import pandas as pd

# Path to your YAML file
yaml_file = "sweep_data_diff_size.yaml"  # Change this path if necessary
output_excel = "extracted_sweep_data.xlsx"

# Load all documents from the YAML file
with open(yaml_file, "r") as f:
    data = list(yaml.safe_load_all(f))

# Extract desired fields from each record
records = []
for record in data:
    config = record.get("config", {})
    extracted = {
        "block_size": config.get("block_size"),
        "mlp_expansion_factor": config.get("mlp_expansion_factor"),
        "n_embd": config.get("n_embd"),
        "n_head": config.get("n_head"),
        "n_layer": config.get("n_layer"),
        "num_params": record.get("num_params"),
        "best_val_loss": record.get("best_val_loss"),
        "better_than_chance": record.get("better_than_chance"),
        "btc_per_param": record.get("btc_per_param"),
    }
    records.append(extracted)

# Save extracted data to Excel
df = pd.DataFrame(records)
df.to_excel(output_excel, index=False)

print(f"Saved extracted data to {output_excel}")

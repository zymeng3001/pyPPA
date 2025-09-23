from search_space import HeteroSearchSpace, Individual
import csv

space = HeteroSearchSpace(L_max=12)

individual = space.sample()

print(individual)


def individual_to_row(ind: Individual) -> dict:
    row = {
        "n_embd": ind["globals"]["d_model"],
        "block_size": ind["globals"]["block_size"],
        # include only layers where layer_mask is True
        "n_head_layerlist": " ".join(str(ind["layers"][i].get("n_heads",  "")) for i in range(space.L_max) if ind["globals"]["layer_mask"][i]),
        "mlp_size_layerlist": " ".join(str(ind["layers"][i].get("mlp_ratio", "")*ind["globals"]["d_model"]) for i in range(space.L_max) if ind["globals"]["layer_mask"][i]),
    }
    return row


# create a CSV file with just the header row
with open("model_configs.csv", "w", newline="") as csvfile:
    fieldnames = ["n_embd", "block_size", "n_head_layerlist", "mlp_size_layerlist", "param_count"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow(individual_to_row(individual))


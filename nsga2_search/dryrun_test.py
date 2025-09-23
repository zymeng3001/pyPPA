from search_space import HeteroSearchSpace
from nsga2 import *

import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Run dry test
# -----------------------------
space = HeteroSearchSpace(L_max=12)
problem = Problem(space)

# Set up single-file JSON logger (array of generation records)
log_path = "./data/hetero_nsga2_dryrun.json"
logger = make_json_list_logger(log_path)
print(f"Logging generation records to {log_path}")

pop, evals, history = nsga2(problem, pop_size=32, n_gen=6, seed=123, log_fn=logger, log_every=1, verbose=True)

# Extract final Pareto
objs = [e.objs for e in evals]
cons = [e.cons for e in evals]
fronts = fast_non_dominated_sort(objs, cons)
pareto_idx = fronts[0]

# Construct DataFrame for top-10 Pareto solutions
rows = []
for i in pareto_idx[:10]:
    g = pop[i]["globals"]
    mask = pop[i]["globals"].get("layer_mask", [])
    active_layers = sum(1 for v in mask if v)
    rows.append({
        "val_loss": evals[i].objs[0],
        "-throughput": evals[i].objs[1],
        "energy/token": evals[i].objs[2],
        "TTFT_ms": evals[i].objs[3],
        "params(M)": round(evals[i].aux["params"]/1e6, 2),
        "mem(GB)": round(evals[i].aux["mem_bytes"]/1e9, 3),
        "FLOPs(G/tok)": round(evals[i].aux["FLOPs"]/1e9, 3),
        "d_model": g["d_model"],
        "block_size": g["block_size"],
        "quant_bits": g["quant_bits"],
        "active_layers": active_layers,
    })
df = pd.DataFrame(rows).sort_values(["val_loss","TTFT_ms"])

# Save full results
artifact = {
    "history": history,
    "pareto_indices": pareto_idx,
    "solutions": [{"x": pop[i], "objs": evals[i].objs, "cons": evals[i].cons, "aux": evals[i].aux} for i in pareto_idx]
}
with open("./data/hetero_nsga2_dryrun_results.json","w") as f:
    json.dump(artifact, f, indent=2)

# Quick 2D scatter: val_loss vs throughput (negated back)
throughputs = [-evals[i].objs[1] for i in range(len(evals))]
val_losses = [evals[i].objs[0] for i in range(len(evals))]
plt.figure()
plt.scatter(val_losses, throughputs, s=12)
plt.xlabel("Validation loss (minimize)")
plt.ylabel("Throughput (tokens/s, maximize)")
plt.title("NSGA-II Dry Run: val_loss vs throughput")
plt.tight_layout()
plt.savefig("plots/hetero_nsga2_scatter.png")


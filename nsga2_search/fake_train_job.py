# relocated copy of utils/fake_train_job.py
from __future__ import annotations
import argparse, json, random, string, time
from pathlib import Path
import random, math
import os, time
from typing import Any, Dict, List
from search_space import HeteroSearchSpace
import hashlib, json
import csv
from fabric import Connection
from pathlib import Path

HOST="34.61.220.162"
USER="xinting"
KEY_PATH="/home/xinting/.ssh/id_rsa"

def individual_to_row(ind: Individual) -> dict:
    row = {
        "n_embd": ind["globals"]["d_model"],
        "block_size": ind["globals"]["block_size"],
        # include only layers where layer_mask is True
        "n_head_layerlist": " ".join(str(ind["layers"][i].get("n_heads",  "")) for i in range(len(ind["globals"]["layer_mask"])) if ind["globals"]["layer_mask"][i]),
        "mlp_size_layerlist": " ".join(str(ind["layers"][i].get("mlp_ratio", "")*ind["globals"]["d_model"]) for i in range(len(ind["globals"]["layer_mask"])) if ind["globals"]["layer_mask"][i]),
        # "n_layers": sum(1 for v in ind["globals"]["layer_mask"] if v)
    }
    return row

def run(out_path: str):
    print("Starting fake training...")
    for i in range(3):
        acc=random.random(); print(f"epoch {i} acc={acc:.4f}")
    sleep_s=random.randint(2,5); time.sleep(sleep_s)
    result={"val_loss": round(random.uniform(2.0,5.0),4), "throughput": random.randint(1000,5000), "timestamp": time.time(), "run_id": ''.join(random.choices(string.ascii_lowercase+string.digits, k=8)), "sleep_s": sleep_s}
    Path(out_path).write_text(json.dumps(result))
    print("Finished fake training.")

def run_training_eval(group: List[Dict[str, Any]], gen: int) -> List[Dict[str, Any]]:
    n = len(group)
    file_name = f"train/gen{gen}/train{n}.csv"
    # mkdirs if not exist
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    # write CSV file
    with open(file_name, "w") as f:
        fieldnames = ["n_embd", "block_size", "n_head_layerlist", "mlp_size_layerlist", "n_layers"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ind in group:
            writer.writerow(individual_to_row(ind))

    with Connection(host=HOST, user=USER, connect_kwargs={"key_filename": KEY_PATH}) as conn:
        print(f"[{HOST}] starting remote job...")

        remote_path = f"/home/{USER}/nsga/gen{gen}/train{n}.csv"
        # create remote directory if not exist
        conn.run(f"mkdir -p /home/{USER}/nsga/gen{gen}", hide=True)
        print(f"[{HOST}] uploading {file_name} -> {remote_path}")
        conn.put(file_name, remote_path)
        
        # conn.run(f"python train.py --config {remote_path}", hide=True)
       
    results = []
    return results

def main():
    # p=argparse.ArgumentParser(); p.add_argument("--out", default="train_results.json"); a=p.parse_args(); run(a.out)
    random.seed(1234)
    pop_size = 10
    space = HeteroSearchSpace(L_max=12)
    pop = [space.sample() for _ in range(pop_size)]
    run_training_eval(pop, gen=0)

if __name__=="__main__": main()

from nsga2 import Population
from typing import List, Dict, Any
from search_space import Individual
from search_space import HeteroSearchSpace
import yaml
from remote_trainer import RemoteTrainer  
import logging
import time

logging.basicConfig(level=logging.INFO)

#initialize Population class from nsga.py with individuals randomly
search_space = HeteroSearchSpace()
individuals = [search_space.sample() for _ in range(4)]
population = Population(individuals)
population.delete_duplicates()  # Remove duplicates if any

# Convert to YAML format
train_yaml_path = population.to_yaml(save_path="trial")  # Save to file for generation 0

hosts = ["34.85.168.66","34.69.195.101"]
user = "xinting"
key_filename = "/home/xinting/.ssh/id_rsa"

trainer = RemoteTrainer(hosts=hosts, user=user, key_filename=key_filename)
trainer.check_connectivity()
trainer.submit_job(path_to_yaml=train_yaml_path, remote_work_dir=f"/home/{user}/Evo_GPT")

print("&&&&&&&&&&&&&&&&&&&&&&&&& Waiting for all jobs to complete...")

trainer.wait_for_all(poll_interval=10, timeout=3600, verbose=True)


from nsga2 import Population
from typing import List, Dict, Any
from search_space import Individual
from search_space import HeteroSearchSpace
import yaml
from remote_trainer import RemoteTrainer  
import logging
import time

# Configure logging to only show INFO:root messages
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s: %(message)s')
# Disable all other loggers except root
for name in ("paramiko", "paramiko.transport", "fabric", "invoke"):
    logging.getLogger(name).disabled = True

#initialize Population class from nsga.py with individuals randomly
search_space = HeteroSearchSpace(L_max=10)
individuals = [search_space.sample() for _ in range(16)]
population = Population(individuals, search_space=search_space)
population.delete_duplicates()  # Remove duplicates if any

# Convert to YAML format
# train_yaml_path = population.to_yaml(save_path="trial")  # Save to file for generation 0

# hosts = ["34.85.168.66", "34.132.101.194"]
hosts = ["34.85.168.66"]
# hosts = ["34.69.195.101"]

user = "xinting"
key_filename = "/home/xinting/.ssh/id_rsa"

population.sw_eval(hosts=hosts, user=user, key_filename=key_filename)
population.print_summary()
population.n_offspring = 10


population.reorder_by_non_domination()
population.print_summary()
population.generate_offspring()
population.gen += 1
population.sw_eval(hosts=hosts, user=user, key_filename=key_filename)
population.print_summary()


# trainer = RemoteTrainer(hosts=hosts, user=user, key_filename=key_filename)
# trainer.check_connectivity()
# trainer.submit_job(path_to_yaml=train_yaml_path, remote_work_dir=f"/home/{user}/Evo_GPT")

# print("&&&&&&&&&&&&&&&&&&&&&&&&& Waiting for all jobs to complete...")

# trainer.wait_for_all(poll_interval=120, timeout=3600, verbose=True)
# trainer.fetch_results(local_dir="train", gen=0)


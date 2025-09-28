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

# load from checkpoint
population = Population.load_checkpoint("ckpts/927_checkpoint_gen2.pkl")

print("Loaded population from checkpoint.")
population.print_summary()

# Convert to YAML format
# train_yaml_path = population.to_yaml(save_path="trial")  # Save to file for generation 0

hosts = ["34.85.168.66", "34.11.48.206", "34.86.55.236"]
user = "xinting"
key_filename = "/home/xinting/.ssh/id_rsa"





from vizier import service
from vizier.service import clients
from vizier.service import pyvizier as vz
import datetime
import subprocess
import sys
import os
from typing import Dict, Any
import pandas as pd
import sweep_utils


# Configuration for local Vizier database
VIZIER_OWNER = 'xinting'
VIZIER_STUDY_ID = 'optma_vizier_dry_run2'
NUM_TRIALS = 100                           # TODO: Number of parameter combinations to test

trainied_data = 'data/Sweeping_sw.csv'
core_top_path = '../../ppa_core_top_extracted_data.csv'
data = pd.read_csv(core_top_path)

# Creating a nested dictionary-based database
database = {}

# Populating the database with configuration as keys and relevant metrics as values
for _, row in data.iterrows():
    key = (row['Gbus width'], row['Wmem Depth'], row['Cache Depth'])
    database[key] = {
        'power': row['Power (W)'],
        'slack': row['Clock Slack (ns)'],
        'clk_period': row['Clock Period (ns) Entered'],
        'clk_min_period': row['Clock_Min_Period'],
        'area': row['Area (um^2)']
        # Additional metrics can be added here as needed
    }


def create_study_config() -> vz.StudyConfig:
    """Create Vizier study configuration with search space and metrics."""
    
    # explorations/openwebtext_sweep2.json
    problem = vz.ProblemStatement()
    root = problem.search_space.root

    # TODO
    root.add_int_param(name='max_iters', min_value=20000, max_value=20000)
    root.add_discrete_param(name='n_embd', feasible_values=[128, 192, 256, 384, 512])                     # n_embd
    root.add_discrete_param(name='n_head', feasible_values=[1, 2, 4, 6, 8, 12])                 # n_head
    root.add_discrete_param(name='block_size', feasible_values=[64, 128, 256, 512])             # max_context_length
    root.add_int_param(name='n_cols', min_value=1, max_value=32, default_value=4)
    root.add_discrete_param(name='Gbus Width', feasible_values=[16, 32, 64, 128])                 # Gbus Width

    root.add_discrete_param(name='n_layer', feasible_values=[6])        # n_layer
    # root.add_int_param(name='mlp_expansion_factor', min_value=1, max_value=4)                   # ffn_size
    # root.add_categorical_param(                                                                 
    #     name='activation_variant', 
    #     feasible_values=['gelu', 'silu', 'relu', 'softplus']
    # )                                                                                           # Activation Variations
    # root.add_categorical_param(
    #     name='softmax_variant_attn', 
    #     feasible_values=['softmax', 'softermax', 'consmax', 'relumax']
    # )                                                                                           # Softmax Variations
    # root.add_categorical_param(
    #     name='norm_variant_attn', 
    #     feasible_values=['rmsnorm', 'krmsnorm', 'layernorm']
    # )                                                                                           # Normalization Variations

    problem.metric_information.append(
        vz.MetricInformation(
            name='fom',
            goal=vz.ObjectiveMetricGoal.MINIMIZE
        )
    )

    study_config = vz.StudyConfig.from_problem(problem)
    study_config.algorithm = 'NSGA2'   # TODO
    
    return study_config

def fom(mj_per_token, area, token_delay, perplexity):
    """Calculate figure of merit (FOM) based on energy, area, and token delay."""
    w1 = 0.2
    w2 = 0.1
    w3 = 0.2
    w4 = 0.5

    target_engergy_per_token = 20 #mJ
    target_area = 1e8
    target_token_delay = 20    #ms
    target_perplexity = 3

    out =  w1 * (mj_per_token / target_engergy_per_token) + w2 * (area / target_area) + w3 * (token_delay / target_token_delay) + w4 * (perplexity / target_perplexity)
     
    return out

def get_val_loss(n_head, n_embd, block_size, n_layer):
    """
    Search for the validation loss corresponding to the given hyperparameters.
    Returns the val_loss if found, otherwise returns float('inf').
    
    Parameters:
    - n_head: int
    - n_embd: int
    - block_size: int
    - n_layer: int
    
    Returns:
    - float: validation loss or float('inf') if not found
    """

    df = pd.read_csv(trainied_data)

    subset = df[
        (df["n_head"] == n_head) &
        (df["n_embd"] == n_embd) &
        (df["block_size"] == block_size) &
        (df["n_layer"] == n_layer)
    ]
    if subset.empty:
        return float('inf')
    else:
        # Assume unique combination; take the first match
        return float(subset["Val_loss"].iloc[0])
    
def is_feasible(suggestion) -> bool:
    """Check if the suggestion is feasible."""
    n_cols = int(suggestion.parameters['n_cols'])
    n_heads = int(suggestion.parameters['n_head'])
    n_embd = int(suggestion.parameters['n_embd'])
    max_context_length = int(suggestion.parameters['block_size']) # max_context_length
    gbus_width = int(suggestion.parameters['Gbus Width']) # Gbus Width
    mac_num = int(gbus_width/8)

    if n_embd % n_heads != 0:
        print(f"n_embd {n_embd} is not divisible by n_heads {n_heads}. Reject suggestion.")
        return False
    
    head_dim = int(n_embd / n_heads)
    
    if head_dim % n_cols != 0:
        print(f"head_dim {head_dim} is not divisible by n_cols {n_cols}. Reject suggestion.")
        return False
    
    core_dim = int(head_dim/n_cols)
    if core_dim % mac_num != 0:
        print(f"core_dim {core_dim} is not divisible by mac_num {mac_num}. Reject suggestion.")
        return False
    
    if max_context_length % n_cols != 0:
      print(f"max_context_length {max_context_length} is not divisible by n_cols {n_cols}. Reject suggestion.")
      return False

    return True

def get_hw_metrics(suggestion):
    """
    Get hardware metrics from the suggestion.
    
    Parameters:
    - suggestion: vz.Suggestion object
    
    Returns:
    - tuple: (mj_per_token, area, token_delay)
    """
    
    # Extract parameters from the suggestion
    n_model = int(suggestion.parameters['n_embd'])
    n_heads = int(suggestion.parameters['n_head'])
    n_cols = int(suggestion.parameters['n_cols'])
    gbus_width = int(suggestion.parameters['Gbus Width'])
    max_context_length = int(suggestion.parameters['block_size'])

    if sweep_utils.is_valid_design(n_model, n_heads, n_cols, gbus_width, max_context_length):

        wmem_depth = sweep_utils.get_wmem_depth(n_model, n_heads, n_cols, gbus_width)
        cache_depth = sweep_utils.get_cache_depth(n_model, n_heads, n_cols, gbus_width, max_context_length)
        core_power = database.get((gbus_width, wmem_depth, cache_depth), {}).get('power', 'N/A') 
        core_area = database.get((gbus_width, wmem_depth, cache_depth), {}).get('area', 'N/A')
        clk_period = database.get((gbus_width, wmem_depth, cache_depth), {}).get('clk_period', 'N/A')
        clk_min_period = database.get((gbus_width, wmem_depth, cache_depth), {}).get('clk_min_period', 'N/A')
        slack = database.get((gbus_width, wmem_depth, cache_depth), {}).get('slack', 'N/A')
        if core_power != 'N/A' and core_area != 'N/A' and clk_period != 'N/A':
                        
            total_power = core_power * n_heads *n_cols
            total_area = core_area * n_heads * n_cols

            token_delay = sweep_utils.get_token_delay(clk_period, n_model, gbus_width, n_heads, n_cols, max_context_length)
            energy_per_token = total_power * token_delay / 1000

            return energy_per_token, total_area, token_delay


    
    return None, None , None


study_config = create_study_config()
study = clients.Study.from_study_config(
    study_config,
    owner=VIZIER_OWNER,
    study_id=VIZIER_STUDY_ID
)

print(f"Starting optimization study {VIZIER_STUDY_ID}")
print('Local SQL database file located at: ', service.VIZIER_DB_PATH)

for trial_idx in range(NUM_TRIALS):
    print(f"\n=== Trial {trial_idx+1}/{NUM_TRIALS} ===")

    # Get parameter suggestion from Vizier
    suggestions = study.suggest(count=1)
    trial = suggestions[0]

    print(f"Config: n_embd: {trial.parameters['n_embd']}, n_head: {trial.parameters['n_head']}, block_size: {trial.parameters['block_size']}, n_layer: {trial.parameters['n_layer']}")
    print(f"Config: Gbus Width: {trial.parameters['Gbus Width']}, n_cols: {trial.parameters['n_cols']}")

    if not is_feasible(trial):
        print("@@@@@@@Invalid configuration, skipping...")
        trial.complete(vz.Measurement(metrics={'fom': float('inf')}))
        continue

    # Execute training run
    validation_loss = get_val_loss(
        n_head=trial.parameters['n_head'],
        n_embd=trial.parameters['n_embd'],
        block_size=trial.parameters['block_size'],
        n_layer=trial.parameters['n_layer']
    )

    if validation_loss == float('inf'):
        print("Validation loss not found, skipping...")
        print("Invalid configuration, skipping...")
        trial.complete(vz.Measurement(metrics={'fom': float('inf')}))
        continue

    # Get hardware metrics
    mj_per_token, area, token_delay = get_hw_metrics(trial)

    if mj_per_token is None or area is None or token_delay is None:
        print("Cannot give hw metrics...")
        print("Invalid configuration, skipping...")
        trial.complete(vz.Measurement(metrics={'fom': float('inf')}))
        continue

    cost_value = fom(
        mj_per_token=mj_per_token,
        area=area,
        token_delay=token_delay,
        perplexity=validation_loss
    )

    # Report results back to Vizier
    trial.complete(vz.Measurement(metrics={'fom': fom}))

print(f"Completed trial with loss: {fom:.4f}")

# Display best results
best_trials = list(study.optimal_trials())
if best_trials:
    best_trial = best_trials[0].materialize()
    print("\n=== Best Configuration ===")
    print(f"Validation loss: {best_trial.final_measurement.metrics['validation_loss'].value:.4f}")
    print("Parameters:")
for name, value in best_trial.parameters.items():
    print(f"  {name}: {value}")
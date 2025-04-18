"""
Module Name: Optma Software Vizier
Description: This Vizier is designed to optimize the software hyperparameters of ReaLLMASIC/nanoGPT for the OPTMA Project.
Author: Qilong Wang
Date: 2025-03-29
Version: 1.1.0
"""

# This is a more complex and useful optimization example that uses the Vizier optimization tool by Google for running the optimization. See https://github.com/google/vizier for more information on Vizier.
# Install vizier using `pip install google-vizier[jax]`

from vizier import service
from vizier.service import clients
from vizier.service import pyvizier as vz
import datetime
import subprocess
import sys
import os
from typing import Dict, Any

# Configuration for local Vizier database
VIZIER_OWNER = 'qilong'
VIZIER_STUDY_ID = 'optma_vizier_v2'
NUM_TRIALS = 5000                           # TODO: Number of parameter combinations to test
big_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = "vizier_results"               # Base directory for training outputs
REPORT_DIR = "vizier_report"

def parse_validation_loss(run_dir: str) -> float:
    """
    Parse validation loss from training outputs.

    This function first attempts to read from 'best_val_loss_and_iter.txt'.
    If that fails, it tries to load from the checkpoint file 'ckpt.pt'.
    
    Example file content:
    3.1415, 1000, 1.24e6, 2.718e3, 2.192e-3

    Args:
        run_dir (str): Path to the training output directory.
        
    Returns:
        float: Best validation loss, or infinity if all attempts fail.
    """
    # Method 1: Try reading from the text file
    best_val_file = os.path.join(run_dir, "best_val_loss_and_iter.txt")
    if os.path.exists(best_val_file):
        try:
            with open(best_val_file, "r") as f:
                return float(f.readline().split(",")[0].strip())
        except (FileNotFoundError, IndexError, ValueError) as e:
            print(f"[WARNING] Failed to parse {best_val_file}: {str(e)}")
    
    # Method 2: Fallback to checkpoint file
    ckpt_file = os.path.join(run_dir, "ckpt.pt")
    if os.path.exists(ckpt_file):
        try:
            checkpoint = torch.load(ckpt_file, map_location="cpu")
            return checkpoint["best_val_loss"].item()
        except (KeyError, RuntimeError) as e:
            print(f"[WARNING] Failed to load checkpoint: {str(e)}")
    
    return float('inf')

def create_study_config() -> vz.StudyConfig:
    """Create Vizier study configuration with search space and metrics."""
    
    # explorations/openwebtext_sweep2.json
    problem = vz.ProblemStatement()
    root = problem.search_space.root

    # TODO
    root.add_int_param(name='max_iters', min_value=20000, max_value=20000)
    root.add_discrete_param(name='n_embd', feasible_values=[192, 384, 768])                     # n_embd
    root.add_discrete_param(name='n_head', feasible_values=[1, 2, 3, 4, 6, 12])                 # n_head
    root.add_discrete_param(name='block_size', feasible_values=[64, 128, 256, 512])             # max_context_length
    root.add_discrete_param(name='n_layer', feasible_values=[5, 10, 15, 20, 25, 30, 35])        # n_layer
    root.add_int_param(name='mlp_expansion_factor', min_value=1, max_value=4)                   # ffn_size
    root.add_categorical_param(name='device', feasible_values=['cuda'])
    root.add_categorical_param(name='dataset', feasible_values=['openwebtext'])
    root.add_categorical_param(                                                                 
        name='activation_variant', 
        feasible_values=['gelu', 'silu', 'relu', 'softplus']
    )                                                                                           # Activation Variations
    root.add_categorical_param(
        name='softmax_variant_attn', 
        feasible_values=['softmax', 'softermax', 'consmax', 'relumax']
    )                                                                                           # Softmax Variations
    root.add_categorical_param(
        name='norm_variant_attn', 
        feasible_values=['rmsnorm', 'krmsnorm', 'layernorm']
    )                                                                                           # Normalization Variations

    problem.metric_information.append(
        vz.MetricInformation(
            name='validation_loss',
            goal=vz.ObjectiveMetricGoal.MINIMIZE
        )
    )

    study_config = vz.StudyConfig.from_problem(problem)
    study_config.algorithm = 'RANDOM_SEARCH'   # TODO
    
    return study_config

def run_training(params: Dict[str, Any]) -> float:
    """Execute training run with specified parameters and return validation loss."""
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # run_id = f"{timestamp}_L{params['n_layer']}H{params['n_head']}E{params['n_embd']}C{params['block_size']}"  # TODO
    run_id = "2025_04_constant_vizier_outputs"
    run_index = (
            f"{timestamp}_E{params['n_embd']}H{params['n_head']}B{params['block_size']}L{params['n_layer']}"
            f"M{int(params['mlp_expansion_factor'])}_A_{params['activation_variant']}_S_{params['softmax_variant_attn']}"
            f"_N_{params['norm_variant_attn']}"
    )

    run_dir = os.path.join(OUTPUT_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)

    # Report of Best Validation Loss
    report_id = "2025_04_constant_vizier_report"
    report_dir = os.path.join(REPORT_DIR, report_id)
    os.makedirs(report_dir, exist_ok=True)
    report_file_path = os.path.join(report_dir, f"{big_timestamp}_val_loss_report.txt")

    # Build training command
    cmd = [
        "python", "train.py",
        "--max_iters", str(int(params['max_iters'])),
        "--n_embd", str(params['n_embd']),
        "--n_head", str(params['n_head']),
        "--block_size", str(params['block_size']),
        "--n_layer", str(params['n_layer']),
        "--mlp_expansion_factor", str(int(params['mlp_expansion_factor'])),
        "--device", params['device'],
        "--dataset", params['dataset'],
        "--activation_variant", params['activation_variant'],
        "--softmax_variant_attn", params['softmax_variant_attn'],
        "--norm_variant_attn", params['norm_variant_attn'],
        "--out_dir", run_dir
    ]

    print(" ".join(cmd))

    # Execute training
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Training failed: {e.stderr}")
        return float('inf')                          # Return high loss for failed runs
    
    ret = parse_validation_loss(run_dir)
    new_result = f"{ret:.5f}:\t{run_index}\n"
    # Write to the file (append mode)
    with open(report_file_path, "a") as f:
        f.write(new_result)
    
    return ret

def main():
    """Main optimization workflow."""
    
    # Initialize Vizier study
    study_config = create_study_config()
    study = clients.Study.from_study_config(
        study_config,
        owner=VIZIER_OWNER,
        study_id=VIZIER_STUDY_ID
    )

    print(f"Starting optimization study {VIZIER_STUDY_ID}")
    print('Local SQL database file located at: ', service.VIZIER_DB_PATH)

    # Optimization loop
    for trial_idx in range(NUM_TRIALS):
        print(f"\n=== Trial {trial_idx+1}/{NUM_TRIALS} ===")
        
        # Get parameter suggestion from Vizier
        suggestions = study.suggest(count=1)
        trial = suggestions[0]
        
        # Execute training run
        validation_loss = run_training(trial.parameters)
        
        # Report results back to Vizier
        trial.complete(vz.Measurement(metrics={'validation_loss': validation_loss}))
        
        print(f"Completed trial with loss: {validation_loss:.4f}")

    # Display best results
    best_trials = list(study.optimal_trials())
    if best_trials:
        best_trial = best_trials[0].materialize()
        print("\n=== Best Configuration ===")
        print(f"Validation loss: {best_trial.final_measurement.metrics['validation_loss'].value:.4f}")
        print("Parameters:")
        for name, value in best_trial.parameters.items():
            print(f"  {name}: {value}")

if __name__ == "__main__":
    main()

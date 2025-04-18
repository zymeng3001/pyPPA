# This is a more complex and useful optimization example that uses the Vizier optimization tool by Google for running the optimization. See https://github.com/google/vizier for more information on Vizier.
# Install vizier using `pip install google-vizier[jax]`
from vizier import service
from vizier.service import clients
from vizier.service import pyvizier as vz
import numpy as np
import time
import math

import sys
from os import path

sys.path.append(path.join(path.dirname(__file__), '..'))

from pyppa import PPARunner
from pyppa.tools import Yosys, OpenROAD, Iverilog
from pyppa.ppa.ppa_runner import PPARunner
from platforms.sky130hd.config import SKY130HD_PLATFORM_CONFIG

import math


ppa_runner = PPARunner(
    design_name="core_array",
    tools={
        'verilog_sim_tool': Iverilog(scripts_dir=path.join('scripts', 'iverilog')),
        'synth_tool': Yosys(scripts_dir=path.join('scripts', 'synth')),
        'ppa_tool': OpenROAD(scripts_dir=path.join('scripts', 'ppa'))
    },
    platform_config=SKY130HD_PLATFORM_CONFIG,
    threads_per_job=3,
    global_flow_config={
        'VERILOG_FILES': [
            path.join(path.dirname(__file__), 'HW', 'core_array/core_array.v'),
            path.join(path.dirname(__file__), 'HW', 'core/core_acc.v'),
            path.join(path.dirname(__file__), 'HW', 'core/core_buf.v'),
            path.join(path.dirname(__file__), 'HW', 'core/core_mac.v'),
            path.join(path.dirname(__file__), 'HW', 'core/core_mem.v'),
            path.join(path.dirname(__file__), 'HW', 'core/core_quant.v'),
            path.join(path.dirname(__file__), 'HW', 'core/core_top.v'),

            path.join(path.dirname(__file__), 'HW', 'vector_engine/activation/activation.v'),
			path.join(path.dirname(__file__), 'HW', 'vector_engine/activation/relu.v'),
			path.join(path.dirname(__file__), 'HW', 'vector_engine/activation/silu.v'),
			path.join(path.dirname(__file__), 'HW', 'vector_engine/activation/gelu.v'),
			path.join(path.dirname(__file__), 'HW', 'vector_engine/activation/softplus.v'),

            path.join(path.dirname(__file__), 'HW', 'vector_engine/softmax/softmax_wrapper.v'),
			path.join(path.dirname(__file__), 'HW', 'vector_engine/softmax/consmax.v'),
			path.join(path.dirname(__file__), 'HW', 'vector_engine/softmax/softmax.v'),
			path.join(path.dirname(__file__), 'HW', 'vector_engine/softmax/softermax.v')

        ],
        'SDC_FILE': path.join(path.dirname(__file__), 'HW', 'constraint.sdc')
    }
)


problem = vz.ProblemStatement()
problem.search_space.root.add_discrete_param(name='constraint_period', feasible_values=[5], default_value=5) # Guessing that the optimal period is somewhere in between, based on previous results

problem.search_space.root.add_discrete_param(name='n_embd', feasible_values=np.arange(128,544,32).tolist(), default_value=256)
problem.search_space.root.add_int_param(name='n_heads', min_value=1, max_value=16, default_value=4)
problem.search_space.root.add_int_param(name='n_cols', min_value=1, max_value=32, default_value=4)
problem.search_space.root.add_discrete_param(name='max_context_length', feasible_values=np.arange(64,544,32).tolist(), default_value=128)
problem.search_space.root.add_discrete_param(name='gbus_width', feasible_values=[16,32,64,128], default_value=32)
problem.search_space.root.add_int_param(name='mlp_expansion_factor', min_value=1, max_value=4)

problem.search_space.root.add_categorical_param(                                                                 
        name='activation_variant', 
        feasible_values=['gelu', 'silu', 'relu', 'softplus']
    )                                                                                           # Activation Variations
problem.search_space.root.add_categorical_param(
        name='softmax_variant_attn', 
        feasible_values=['softmax', 'softermax', 'consmax', 'relumax']
    )                                                                                           # Softmax Variations
problem.search_space.root.add_categorical_param(
        name='norm_variant_attn', 
        feasible_values=['rmsnorm']
    ) 

problem.search_space.root.add_int_param(name='ABC_MAX_FANOUT', min_value=12, max_value=28, default_value=20) 
problem.search_space.root.add_float_param(name='ABC_MAP_EFFORT', min_value=0, max_value=1, default_value=0.6) 
problem.search_space.root.add_float_param(name='ABC_AREC_EFFORT', min_value=0, max_value=1, default_value=0.6) 


problem.metric_information.append(
    vz.MetricInformation(
        name='fom',
        goal=vz.ObjectiveMetricGoal.MINIMIZE
    )
)


study_config = vz.StudyConfig.from_problem(problem)
study_config.algorithm = 'NSGA-II' # Use random search for random sampling
study_client = clients.Study.from_study_config(
  study_config,
  owner='ppa_runner',
  study_id='ppa_top_optimization'
)
print('Local SQL database file located at: ', service.VIZIER_DB_PATH)


# Store seen configurations (global set)
seen_configs = set()


def is_duplicate(suggestion):
    """Check if the suggestion has already been tried based on unique parameters."""
    config_tuple = (
        int(suggestion.parameters['n_heads']),
        int(suggestion.parameters['n_cols']),
        int(suggestion.parameters['max_context_length']),
        int(suggestion.parameters['gbus_width']),
        int(suggestion.parameters['wmem_depth'])
    )
   
    if config_tuple in seen_configs:
        return True
    seen_configs.add(config_tuple)
    return False


def is_feasible(suggestion) -> bool:
    """Check if the suggestion is feasible."""
    n_cols = int(suggestion.parameters['n_cols'])
    n_heads = int(suggestion.parameters['n_heads'])
    head_dim = int(suggestion.parameters['head_dim'])
    max_context_length = int(suggestion.parameters['max_context_length'])
    gbus_width = int(suggestion.parameters['gbus_width'])
    mac_num = int(gbus_width/8)

    if head_dim * n_heads > 10:
        print(f"head_dim * n_heads {head_dim * n_heads} is greater than 1024. Reject suggestion.")
        return False
    
    if head_dim * n_heads < 128:
        print(f"head_dim * n_heads {head_dim * n_heads} is less than 128. Reject suggestion.")
        return False
    
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


def fom(area: float, period: float, total_power: float):
    w1 = 0.5
    w2 = 0.5
    w3 = 0.6
    w4 = 0.2
    target_energy_per_token = 0.8
    target_area = 3e6
    target_token_delay = 1e4


    out = w1 * (area / target_area) + w2 * (total_power / target_energy_per_token)


    # The objective function/figure of merit (which is minimized), is the product of the area, period, and power attempts to minimize all three.
    return out

def get_cache_depth(suggestion):
    """Get the cache depth based on the suggestion."""
    n_model = int(suggestion.parameters['n_heads']) * int(suggestion.parameters['n_heads'])
    n_cols = int(suggestion.parameters['n_cols'])
    n_heads = int(suggestion.parameters['n_heads'])
    max_context_length = int(suggestion.parameters['max_context_length'])
    gbus_width = int(suggestion.parameters['gbus_width'])
    mac_num = int(gbus_width/8)

    return int(2* n_model * max_context_length/ mac_num / n_cols / n_heads) 

def get_wmem_depth(suggestion):
    """Get the wmem depth based on the suggestion."""
    n_model = int(suggestion.parameters['n_heads']) * int(suggestion.parameters['n_heads'])
    head_dim = int(suggestion.parameters['head_dim'])
    gbus_width = int(suggestion.parameters['gbus_width'])
    raw_wmem_depth = int(n_model * head_dim / gbus_width)
 
    return int(math.ceil(raw_wmem_depth / 256) * 256)

def vizier_optimizer(prev_iter_number, prev_iter_ppa_runs: list[PPARunner], previous_suggestions):
    if prev_iter_ppa_runs is not None:
        if len(prev_iter_ppa_runs) != len(previous_suggestions):
            print("Number of runs does not match number of suggestions. Something went wrong, aborting.")
            return {
                'opt_complete': True
            }


        for i, suggestion in enumerate(previous_suggestions):
            constraint_period = suggestion.parameters['constraint_period']


            run = prev_iter_ppa_runs[i]
            area = run['synth_stats']['module_area']
            period = run['ppa_stats']['sta']['clk']['clk_period']
            total_power = run['ppa_stats']['power_report']['total']['total_power']
            throughput = 1e9 / period
   
            objective = fom(
                area=area,
                period=period,
                total_power=total_power,
            )


            print(f'Iteration {prev_iter_number}, suggestion (constraint_period = {constraint_period})')
           
            print(f'area {area} period {period} total_power {total_power} throughput {throughput} objective value {objective}.')
            final_measurement = vz.Measurement({'fom': objective})
            suggestion.complete(final_measurement)


    if prev_iter_number >= 150: # Run for 10 iterations and then stop
        print("Optimization complete.")
        # Print the optimal Vizier trials
        for optimal_trial in study_client.optimal_trials():
            optimal_trial = optimal_trial.materialize()
            print(
                "Optimal Trial Suggestion and Objective:",
                optimal_trial.parameters,
                optimal_trial.final_measurement
            )

        return {
            'opt_complete': True
        }


    # Assign new suggestions
    feasible_suggestions = []
    suggestions = study_client.suggest(count=1) # Since 3 threads per job
    while len(feasible_suggestions) < 1:
        print("Generating new suggestions")
        for i, suggestion in enumerate(suggestions):
            print(suggestion.parameters)
            if not is_feasible(suggestion):
                print(f"Suggestion {i} is not feasible. Skipping.")
                # suggestion.complete(vz.Measurement(), infeasibility_reason='Infeasible design.')  # mark as completed
                suggestion.complete(vz.Measurement({'fom':math.inf}))  # mark as completed
            else:
                feasible_suggestions.append(suggestion)
        suggestions = study_client.suggest(count=1)
    # feasible_suggestions = suggestions


    for suggestion in feasible_suggestions:
        print("Feasible suggestions:")
        print(suggestion.parameters)


    # report time stamp
    print(f"Time stamp: {time.time()}")
    return {
        'opt_complete': False,
        'next_suggestions': [
            {
                'flow_config': {
                    'ABC_MAX_FANOUT': suggestion.parameters['ABC_MAX_FANOUT'],
                    'ABC_MAP_EFFORT': suggestion.parameters['ABC_MAP_EFFORT'],
                    'ABC_AREC_EFFORT': suggestion.parameters['ABC_MAP_EFFORT']
                },
                'hyperparameters': {
                    'clk_period': suggestion.parameters['constraint_period'],
                    'n_heads': int(suggestion.parameters['n_heads']),
                    'n_cols': int(suggestion.parameters['n_cols']),
                    'head_dim': int(suggestion.parameters['n_embd'] / suggestion.parameters['n_heads']),
                    'max_context_length': int(suggestion.parameters['max_context_length']),
                    'gbus_width': int(suggestion.parameters['gbus_width']),
                    'mac_num': int(suggestion.parameters['gbus_width']/8),
                    'n_model': int(suggestion.parameters['n_heads'] * (suggestion.parameters['head_dim'])),
                    'wmem_depth': get_wmem_depth(suggestion),
                    'cache_depth': get_cache_depth(suggestion),
                    'activation': suggestion.parameters['activation'],
                    'softmax_choice': suggestion.parameters['softmax_choice']
                }
            } for suggestion in feasible_suggestions
        ],
        'context': feasible_suggestions # Send suggestions as context, and they will be sent as arguments for the next run of the optimizer.
    }


ppa_runner.add_job({
    'module_name': 'top_wrapper',
    'mode': 'opt',
    'optimizer': vizier_optimizer
})


ppa_runner.run_all_jobs()
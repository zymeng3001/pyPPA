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
from utils.search_space_utils import *

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
            path.join(path.dirname(__file__), '../HW', 'core_array/core_array.v'),
            path.join(path.dirname(__file__), '../HW', 'core/core_acc.v'),
            path.join(path.dirname(__file__), '../HW', 'core/core_buf.v'),
            path.join(path.dirname(__file__), '../HW', 'core/core_mac.v'),
            path.join(path.dirname(__file__), '../HW', 'core/core_mem.v'),
            path.join(path.dirname(__file__), '../HW', 'core/core_quant.v'),
            path.join(path.dirname(__file__), '../HW', 'core/core_top.v')

        ],
        'SDC_FILE': path.join(path.dirname(__file__), '../HW', 'constraint.sdc')
    }
)

problem = vz.ProblemStatement()
problem.search_space.root.add_discrete_param(name='constraint_period', feasible_values=[4.5,5,6], default_value=6) # Guessing that the optimal period is somewhere in between, based on previous results
# problem.search_space.root.add_int_param(name='ABC_MAX_FANOUT', min_value=12, max_value=28, default_value=20) # Guessing the ABC max fanout is somewhere between 12 and 28
# problem.search_space.root.add_float_param(name='ABC_MAP_EFFORT', min_value=0, max_value=1, default_value=0.6) # Guessing the ABC map effort is somewhere between 0 and 1
problem.search_space.root.add_int_param(name='n_heads', min_value=1, max_value=16, default_value=4)
problem.search_space.root.add_int_param(name='n_cols', min_value=1, max_value=16, default_value=4)
problem.search_space.root.add_discrete_param(name='head_dim', feasible_values=np.arange(16,160,16).tolist(), default_value=32)
problem.search_space.root.add_discrete_param(name='max_context_length', feasible_values=np.arange(64,576,512).tolist(), default_value=128)
problem.search_space.root.add_discrete_param(name='gbus_width', feasible_values=[16,32,64,128], default_value=32)

problem.search_space.root.add_discrete_param(name='n_layer', feasible_values=[2,4,6,8], default_value=6)
problem.search_space.root.add_discrete_param(name='ffn_ratio', feasible_values=[1, 2, 3, 4, 5, 6], default_value=4)




problem.metric_information.append(
    vz.MetricInformation(
        name='fom',
        goal=vz.ObjectiveMetricGoal.MINIMIZE
    )
)

study_config = vz.StudyConfig.from_problem(problem)
study_config.algorithm = 'NSGA2' # Use random search for random sampling
study_client = clients.Study.from_study_config(
  study_config,
  owner='ppa_runner',
  study_id='ppa_core_array_swhw_opt_t1'
)
print('Local SQL database file located at: ', service.VIZIER_DB_PATH)


def fom(mj_per_token, area, token_delay, val_loss):
    """Calculate figure of merit (FOM) based on energy, area, and token delay."""
    w1 = 0.2
    w2 = 0.1
    w3 = 0.2
    w4 = 0.5

    target_engergy_per_token = 20 #mJ
    target_area = 1e8
    target_token_delay = 20    #ms
    target_val_loss = 3

    out =  w1 * (mj_per_token / target_engergy_per_token) + w2 * (area / target_area) + w3 * (token_delay / target_token_delay) + w4 * (val_loss / target_val_loss)
     
    return out

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
            clk_period = run['ppa_stats']['sta']['clk']['clk_period']
            total_power = run['ppa_stats']['power_report']['total']['total_power']

            n_embd = suggestion.parameters['n_heads'] * suggestion.parameters['head_dim']

            token_delay = get_token_delay(clk_period,  n_embd, suggestion.parameters['gbus_width'], suggestion.parameters['n_heads'], 
                                          suggestion.parameters['n_cols'], suggestion.parameters['max_context_length'], 
                                          n_layers=suggestion.parameters['n_layer'], ffn_ratio=suggestion.parameters['ffn_ratio'])
            
            energy_per_token = (total_power * token_delay) / 1000 # Convert to mJ

            # Calculate perplexity
            val_loss = 3
            

            objective = fom(
                mj_per_token=energy_per_token,
                area=area,
                token_delay=token_delay,
                val_loss=val_loss

            )
            print("Iteration: ", prev_iter_number)
            print(f"Objective: {objective}, Area: {area}, Token Delay: {token_delay}, Energy per token: {energy_per_token}, Val Loss: {val_loss}")

            final_measurement = vz.Measurement({'fom': objective})
            suggestion.complete(final_measurement)

    if prev_iter_number >= 5:  # stopping condition
        print("Optimization complete.")
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
    
    print(f'Iteration {prev_iter_number}, Generating new suggestions')

    feasible_suggestions = []
    suggestions = study_client.suggest(count=1)
    while len(feasible_suggestions) < 1:
        print("Suggestions:")
        for suggestion in suggestions:
            if not is_feasible(suggestion):
                suggestion.complete(vz.Measurement({'fom': math.inf}))
                print("Rejecting suggestion due to infeasibility")
            else:
                feasible_suggestions.append(suggestion)
        suggestions = study_client.suggest(count=1)

    for suggestion in feasible_suggestions:
        print("Feasible suggestions:")
        print(suggestion.parameters)


    # start Sw training here

    return {
    'opt_complete': False,
    'next_suggestions': [
        {
            'flow_config': {
                'ABC_AREA': True
            },
            'hyperparameters': {
                'clk_period': suggestion.parameters['constraint_period'],
                'n_heads': int(suggestion.parameters['n_heads']),
                'n_cols': int(suggestion.parameters['n_cols']),
                'gbus_width': int(suggestion.parameters['gbus_width']),
                'wmem_depth': get_wmem_depth(suggestion),
                'cache_depth': get_cache_depth(suggestion),
                'mac_num': int(suggestion.parameters['gbus_width']/8)
            }
        } for suggestion in feasible_suggestions
    ],
    'context': feasible_suggestions # Send suggestions as context, and they will be sent as arguments for the next run of the optimizer.
    }


ppa_runner.add_job({
	'module_name': 'core_array',
	'mode': 'opt',
	'optimizer': vizier_optimizer
})

ppa_runner.run_all_jobs() 
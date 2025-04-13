# This is a more complex and useful optimization example that uses the Vizier optimization tool by Google for running the optimization. See https://github.com/google/vizier for more information on Vizier.
# Install vizier using `pip install google-vizier[jax]`

from vizier import service
from vizier.service import clients
from vizier.service import pyvizier as vz
import numpy as np
import math

import sys
from os import path

sys.path.append(path.join(path.dirname(__file__), '..'))

from pyppa import PPARunner
from pyppa.tools import Yosys, OpenROAD, Iverilog
from pyppa.ppa.ppa_runner import PPARunner
from platforms.sky130hd.config import SKY130HD_PLATFORM_CONFIG


ppa_runner = PPARunner(
	design_name="krms",
	tools={
		'verilog_sim_tool': Iverilog(scripts_dir=path.join('scripts', 'iverilog')),
		'synth_tool': Yosys(scripts_dir=path.join('scripts', 'synth')),
		'ppa_tool': OpenROAD(scripts_dir=path.join('scripts', 'ppa'))
	},
	platform_config=SKY130HD_PLATFORM_CONFIG,
	threads_per_job=3,
	global_flow_config={
		'VERILOG_FILES': [
			path.join(path.dirname(__file__), 'HW', 'vector_engine/layernorm/krms.v'),
			path.join(path.dirname(__file__), 'HW', 'vector_engine/layernorm/fp_div_pipe.v'),
			path.join(path.dirname(__file__), 'HW', 'vector_engine/layernorm/fp_mult_pipe.v'),
			path.join(path.dirname(__file__), 'HW', 'vector_engine/layernorm/fp_invsqrt_pipe.v')
		],
		'SDC_FILE': path.join(path.dirname(__file__), 'HW', 'constraint.sdc')
	}
)

problem = vz.ProblemStatement()
problem.search_space.root.add_discrete_param(name='constraint_period', feasible_values=[10], default_value=10) # Guessing that the optimal period is somewhere in between, based on previous results
problem.metric_information.append(
    vz.MetricInformation(
        name='fom',
		goal=vz.ObjectiveMetricGoal.MINIMIZE
	)
)

study_config = vz.StudyConfig.from_problem(problem)
study_config.algorithm = 'RANDOM_SEARCH'
study_client = clients.Study.from_study_config(
  study_config,
  owner='ppa_runner',
  study_id='ppa_krms'
)
print('Local SQL database file located at: ', service.VIZIER_DB_PATH)

seen_configs = set()

# def is_duplicate(suggestion):
#     """Check if the suggestion has already been tried based on unique parameters."""
#     config_tuple = (
#         int(suggestion.parameters['head_dim']),
# 		suggestion.parameters['activation']
#     )
   
#     if config_tuple in seen_configs:
#         return True
#     seen_configs.add(config_tuple)
#     return False

def fom(area: float, period: float, total_power: float):
    w1 = 0.2
    w2 = 0.2
    w3 = 0.6
    target_power = 0.08
    target_area = 3e6
    target_throughput = 6e7
	
    throughput = 1e9 / period

    out = w1 * (area / target_area) + w2 * (total_power / target_power) - w3 * (throughput / target_throughput)

	# The objective function/figure of merit (which is minimized), is the product of the area, period, and power attempts to minimize all three.
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
			period = run['ppa_stats']['sta']['clk']['clk_period']
			total_power = run['ppa_stats']['power_report']['total']['total_power']
			
			final_measurement = vz.Measurement({'fom': 1})
			suggestion.complete(final_measurement)

	if prev_iter_number >= 128:  # stopping condition
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

	feasible_suggestions = []
	feasible_suggestions = study_client.suggest(count=1)
	# while len(feasible_suggestions) < 1:
	# 	print("Suggestions:")
	# 	for suggestion in suggestions:
	# 		if is_duplicate(suggestion):
	# 			suggestion.complete(vz.Measurement({'fom': math.inf}))
	# 		else:
	# 			feasible_suggestions.append(suggestion)
	# 	suggestions = study_client.suggest(count=10)

	# for suggestion in feasible_suggestions:
	# 	print("Feasible suggestions:")
	# 	print(suggestion.parameters)
	
	return {
        'opt_complete': False,
        'next_suggestions': [
            {
                'flow_config': {
                    'ABC_AREA': True
                },
                'hyperparameters': {
                    'clk_period': suggestion.parameters['constraint_period']
                }
            } for suggestion in feasible_suggestions
        ],
        'context': feasible_suggestions # Send suggestions as context, and they will be sent as arguments for the next run of the optimizer.
    }

ppa_runner.add_job({
	'module_name': 'krms',
	'mode': 'opt',
	'optimizer': vizier_optimizer
})

ppa_runner.run_all_jobs() 
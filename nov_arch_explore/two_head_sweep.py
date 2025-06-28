# This is a more complex and useful optimization example that uses the Vizier optimization tool by Google for running the optimization. See https://github.com/google/vizier for more information on Vizier.
# Install vizier using `pip install google-vizier[jax]`


from vizier import service
from vizier.service import clients
from vizier.service import pyvizier as vz
import numpy as np
import time


import sys
from os import path

import pickle


sys.path.append(path.join(path.dirname(__file__), '..'))


from pyppa import PPARunner
from pyppa.tools import Yosys, OpenROAD, Iverilog
from pyppa.ppa.ppa_runner import PPARunner
from platforms.sky130hd.config import SKY130HD_PLATFORM_CONFIG


import math


ppa_runner = PPARunner(
    design_name="two_heads",
    tools={
        'verilog_sim_tool': Iverilog(scripts_dir=path.join('scripts', 'iverilog')),
        'synth_tool': Yosys(scripts_dir=path.join('scripts', 'synth')),
        'ppa_tool': OpenROAD(scripts_dir=path.join('scripts', 'ppa'))
    },
    platform_config=SKY130HD_PLATFORM_CONFIG,
    threads_per_job=3,
    global_flow_config={
        'VERILOG_FILES': [
            path.join(path.dirname(__file__), 'HW_NOV', 'sysdef.svh'),
            path.join(path.dirname(__file__), 'HW_NOV', 'core/core_acc.v'),
            path.join(path.dirname(__file__), 'HW_NOV', 'core/core_buf.v'),
            path.join(path.dirname(__file__), 'HW_NOV', 'core/core_ctrl.v'),
            path.join(path.dirname(__file__), 'HW_NOV', 'core/core_mac.v'),
            path.join(path.dirname(__file__), 'HW_NOV', 'core/core_mem.v'),
            path.join(path.dirname(__file__), 'HW_NOV', 'core/core_quant.v'),
            path.join(path.dirname(__file__), 'HW_NOV', 'core/core_rc.v'),
            path.join(path.dirname(__file__), 'HW_NOV', 'core/core_top.v'),
            path.join(path.dirname(__file__), 'HW_NOV', 'head/abuf.v'),
            path.join(path.dirname(__file__), 'HW_NOV', 'head/head_core_array.v'),
            path.join(path.dirname(__file__), 'HW_NOV', 'head/head_sram_rd_ctrl.v'),
            path.join(path.dirname(__file__), 'HW_NOV', 'head/head_sram.v'),
            path.join(path.dirname(__file__), 'HW_NOV', 'head/head_top.v'),
            path.join(path.dirname(__file__), 'HW_NOV', 'head/two_heads.v'),
            path.join(path.dirname(__file__), 'HW_NOV', 'util/pe.v'),
            path.join(path.dirname(__file__), 'HW_NOV', 'util/mem.v'),
            path.join(path.dirname(__file__), 'HW_NOV', 'util/align.v'),
            path.join(path.dirname(__file__), 'HW_NOV', 'vector_engine/krms_recompute/rtl/krms.v'),
            path.join(path.dirname(__file__), 'HW_NOV', 'vector_engine/RMSnorm/rtl/RMSnorm.v'),
            path.join(path.dirname(__file__), 'HW_NOV', 'vector_engine/RMSnorm/rtl/fp_div_pipe.v'),
            path.join(path.dirname(__file__), 'HW_NOV', 'vector_engine/RMSnorm/rtl/fp_invsqrt_pipe.v'),
            path.join(path.dirname(__file__), 'HW_NOV', 'vector_engine/RMSnorm/rtl/fp_mult_pipe.v'),
            path.join(path.dirname(__file__), 'HW_NOV', 'vector_engine/softmax_recompute/rtl/softmax_rc.v'),

        ],
        'SDC_FILE': path.join(path.dirname(__file__), 'HW_NOV', 'constraint.sdc')
    }
)


problem = vz.ProblemStatement()
problem.search_space.root.add_discrete_param(name='constraint_period', feasible_values=[10], default_value=10) # Guessing that the optimal period is somewhere in between, based on previous results
# problem.search_space.root.add_int_param(name='ABC_MAX_FANOUT', min_value=12, max_value=28, default_value=20) # Guessing the ABC max fanout is somewhere between 12 and 28
# problem.search_space.root.add_float_param(name='ABC_MAP_EFFORT', min_value=0, max_value=1, default_value=0.6) # Guessing the ABC map effort is somewhere between 0 and 1
problem.search_space.root.add_discrete_param(name='mac_num', feasible_values=[16,32], default_value=32)

problem.metric_information.append(
    vz.MetricInformation(
        name='fom',
        goal=vz.ObjectiveMetricGoal.MINIMIZE
    )
)


study_config = vz.StudyConfig.from_problem(problem)
study_config.algorithm = 'RANDOM_SEARCH' # Use random search for random sampling
study_client = clients.Study.from_study_config(
  study_config,
  owner='ppa_runner',
  study_id='ppa_two_heads_nov'
)
print('Local SQL database file located at: ', service.VIZIER_DB_PATH)

def fom(area: float, period: float, total_power: float):
    w1 = 0.5
    w2 = 0.5
    w3 = 0.6
    target_power = 0.8
    target_area = 3e6
    target_throughput = 6e7

    out = w1 * (area / target_area) + w2 * (total_power / target_power)

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


    if prev_iter_number >= 2: # Run for 10 iterations and then stop
        print("Optimization complete.")

        # Print the optimal Vizier trials
        # for optimal_trial in study_client.optimal_trials():
        #     optimal_trial = optimal_trial.materialize()
        #     print(
        #         "Optimal Trial Suggestion and Objective:",
        #         optimal_trial.parameters,
        #         optimal_trial.final_measurement
        #     )
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
                    'ABC_AREA': True
                },
                'hyperparameters': {
                    'clk_period': suggestion.parameters['constraint_period'],
                    'mac_num': int(suggestion.parameters['mac_num'])
                }
            } for suggestion in feasible_suggestions
        ],
        'context': feasible_suggestions # Send suggestions as context, and they will be sent as arguments for the next run of the optimizer.
    }


ppa_runner.add_job({
    'module_name': 'two_heads',
    'mode': 'opt',
    'optimizer': vizier_optimizer
})

ppa_runner.run_all_jobs()


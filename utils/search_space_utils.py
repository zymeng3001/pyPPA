
from vizier import service
from vizier.service import clients
from vizier.service import pyvizier as vz
import numpy as np
import time
import math

import sys
from os import path


import math

def is_feasible(suggestion) -> bool:
    """Check if the suggestion is feasible."""
    n_cols = int(suggestion.parameters['n_cols'])
    n_heads = int(suggestion.parameters['n_heads'])
    head_dim = int(suggestion.parameters['head_dim'])
    max_context_length = int(suggestion.parameters['max_context_length'])
    gbus_width = int(suggestion.parameters['gbus_width'])
    mac_num = int(gbus_width/8)

    if head_dim * n_heads > 1024:
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

def get_cache_depth(suggestion) -> int:
    """Get the cache depth based on the suggestion."""
    n_model = int(suggestion.parameters['n_heads']) * int(suggestion.parameters['head_dim'])
    n_cols = int(suggestion.parameters['n_cols'])
    n_heads = int(suggestion.parameters['n_heads'])
    max_context_length = int(suggestion.parameters['max_context_length'])
    gbus_width = int(suggestion.parameters['gbus_width'])
    mac_num = int(gbus_width/8)

    raw_cache_depth = int(n_model * max_context_length / mac_num / n_cols / n_heads)
    if raw_cache_depth < 32:    
        return 32
    elif raw_cache_depth < 64:
        return 64
    elif raw_cache_depth < 128:
        return 128
    else:
        # Round up to the nearest 256
        return int(math.ceil(raw_cache_depth / 256) * 256)
    
    # return int(n_model * max_context_length / mac_num / n_cols / n_heads) 

def get_wmem_depth(suggestion) -> int:
    """Get the wmem depth based on the suggestion."""
    n_model = int(suggestion.parameters['n_heads']) * int(suggestion.parameters['head_dim'])
    head_dim = int(suggestion.parameters['head_dim'])
    n_cols = int(suggestion.parameters['n_cols'])
    gbus_width = int(suggestion.parameters['gbus_width'])
    raw_wmem_depth = int(n_model * head_dim / n_cols / gbus_width)

    if raw_wmem_depth < 32:
        return 32
    elif raw_wmem_depth < 64:
        return 64
    elif raw_wmem_depth < 128:
        return 128
    else :
        # Round up to the nearest 256
        return int(math.ceil(raw_wmem_depth / 256) * 256)
    
def get_token_delay(clk_period, n_model, gbus_width, n_heads, n_cols, max_context_length, n_layers=1, ffn_ratio=4, softmax_choice='SOFTMAX', activation_choice='RELU'):
    """
    Calculate the token delay based on the design parameters.
    """
    mac_num = int(gbus_width / 8)
    # Calculate the token delay
    # for GEMM ops

    sequence_length = 128 # set the sequence length to 128

    # Number of cycles = Total MACs / (Number of MAC units)
    num_loading_cycles = (4*n_model*n_model*sequence_length + 2*max_context_length*max_context_length*n_model + 2*ffn_ratio*n_model*n_model*sequence_length) / (n_heads*n_cols*mac_num)
    token_delay = num_loading_cycles * clk_period * 1e-9 # seconds

    # add 2 residual delay
    token_delay += 2 * n_model * 1e-9 * clk_period # residual add delay

    # add memory loading delay
    token_delay += (4*n_model*n_model + 2*ffn_ratio*n_model*n_model) * 1e-9 * clk_period # load and store delay

    # add softmax delay
    if softmax_choice == 'SOFTMAX':
        token_delay += 7 * max_context_length * max_context_length * 1e-9 * clk_period 
    elif softmax_choice == 'SOFTERMAX':
        token_delay += 2 * max_context_length * max_context_length * 1e-9 * clk_period 
    elif softmax_choice == 'CONSMAX':
        token_delay += 4 * 1e-9 * clk_period     #fully pipelined

    # activation delay is negligible

    token_delay *= n_layers

    return token_delay * 1000 # convert to us
    
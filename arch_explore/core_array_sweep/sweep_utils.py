import numpy as np
import math

def is_valid_design(n_model, n_heads, n_cols, gbus_width, max_context_length):
    """
    Check if the design is valid by checking the existence of the design directory.
    """
    mac_num = int(gbus_width / 8)

    if n_model % n_heads != 0:
        return False
    
    head_dim = n_model / n_heads

    if head_dim % n_cols != 0:
        return False
    
    if int(head_dim / n_cols) % mac_num != 0:
        return False
    
    if max_context_length % n_cols != 0:
        return False
    
    return True

def get_cache_depth(n_model, n_heads, n_cols, gbus_width, max_context_length):
    """
    Calculate the cache depth based on the design parameters.
    """
    mac_num = int(gbus_width / 8)
    head_dim = n_model / n_heads
    
    raw_cache_depth = int(head_dim * max_context_length / mac_num / n_cols)
    if raw_cache_depth < 32:    
        return 32
    elif raw_cache_depth < 64:
        return 64
    elif raw_cache_depth < 128:
        return 128
    elif raw_cache_depth < 256:
        return 256
    else:
        # Round up to the nearest 512
        return int(math.ceil(raw_cache_depth / 512) * 512)
    
def get_wmem_depth(n_model, n_heads, n_cols, gbus_width):
    """
    Calculate the wmem depth based on the design parameters.
    """
    mac_num = int(gbus_width / 8)
    head_dim = n_model / n_heads
    
    raw_wmem_depth = int(n_model * head_dim / n_cols / mac_num)
    if raw_wmem_depth < 32:
        return 32
    elif raw_wmem_depth < 64:
        return 64
    elif raw_wmem_depth < 128:
        return 128
    elif raw_wmem_depth < 256:
        return 256
    else :
        # Round up to the nearest 512
        return int(math.ceil(raw_wmem_depth / 512) * 512)
    
def get_token_delay(clk_period, n_model, gbus_width, n_heads, n_cols, max_context_length, n_layers=1, ffn_ratio=4, softmax_choice='SOFTMAX', activation_choice='RELU'):
    """
    Calculate the token delay based on the design parameters.
    """
    mac_num = int(gbus_width / 8)
    # Calculate the token delay
    # for GEMM ops
    sequence_length = 64 # set the sequence length to 64
    efficiency_ratio = 0.8 # assume the efficiency ratio is 0.8

    # Number of cycles = Total MACs / (Number of MAC units)
    # num_loading_cycles = (4*n_model*n_model*sequence_length + 2*max_context_length*max_context_length*n_model + 2*ffn_ratio*n_model*n_model*sequence_length) / (n_heads*n_cols*mac_num)
    # num_loading_cycles = (4*n_model*n_model + 2*max_context_length*max_context_length*n_model + 2*ffn_ratio*n_model*n_model) / (n_heads*n_cols*mac_num)
    num_compute_cycles = (4*n_model*n_model + 2*max_context_length*n_model + 2*ffn_ratio*n_model*n_model) / (n_heads*n_cols*mac_num*efficiency_ratio)
    
    token_delay = num_compute_cycles * clk_period * 1e-9 # seconds

    # add 2 residual delay
    token_delay += 2 * n_model * clk_period * 1e-9 # residual add delay

    # add memory loading delay （loading to the wmem)
    token_delay += (4*n_model*n_model + 2*ffn_ratio*n_model*n_model) / mac_num * 1e-9 * clk_period # load and store delay

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
    

    
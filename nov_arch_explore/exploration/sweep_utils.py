import numpy as np
import math

def is_feasible(n_embd, n_heads, n_cols, n_macs, max_context_length) -> bool:
    """Check if the suggestion is feasible."""

    if n_embd % n_heads != 0:
        # print(f"n_embd {n_embd} is not divisible by n_heads {n_heads}. Reject suggestion.")
        return False
    
    head_dim = int(n_embd/n_heads)
    if head_dim % n_cols != 0:
        # print(f"head_dim {head_dim} is not divisible by n_cols {n_cols}. Reject suggestion.")
        return False
    
    core_dim = int(head_dim/n_cols)
    if core_dim % n_macs != 0:
        # print(f"core_dim {core_dim} is not divisible by mac_num {mac_num}. Reject suggestion.")
        return False
    
    if max_context_length % n_cols != 0:
    #   print(f"max_context_length {max_context_length} is not divisible by n_cols {n_cols}. Reject suggestion.")
      return False

    return True

def get_cache_depth(n_model, n_heads, n_cols, max_context_length, n_macs) -> int:
    """Get the cache depth based on the suggestion."""
    # n_model = int(suggestion.parameters['n_heads']) * int(suggestion.parameters['head_dim'])
    # n_cols = int(suggestion.parameters['n_cols'])
    # n_heads = int(suggestion.parameters['n_heads'])
    # max_context_length = int(suggestion.parameters['max_context_length'])
    # gbus_width = int(suggestion.parameters['gbus_width'])

    raw_cache_depth = int(2 * n_model * max_context_length / n_macs / n_cols / n_heads)
    # if raw_cache_depth < 32:    
    #     return 32
    # elif raw_cache_depth < 64:
    #     return 64
    if raw_cache_depth < 128:
        return 128
    elif raw_cache_depth < 256:
        return 256
    else:
        # Round up to the nearest 512
        return int(math.ceil(raw_cache_depth / 512) * 512)
    
    # return int(n_model * max_context_length / mac_num / n_cols / n_heads) 

def get_wmem_depth(n_model, n_heads, n_cols, n_macs, ffn_ratio) -> int:
    """Get the wmem depth based on the suggestion."""
    # n_model = int(suggestion.parameters['n_heads']) * int(suggestion.parameters['head_dim'])
    # head_dim = int(suggestion.parameters['head_dim'])
    # n_cols = int(suggestion.parameters['n_cols'])
    # gbus_width = int(suggestion.parameters['gbus_width'])
    # gbus_width = int(n_macs * 8)  
    raw_wmem_depth = int((4 + 0*ffn_ratio)*(n_model * n_model)/ n_heads / n_cols / n_macs)

    if raw_wmem_depth < 32:
        return 32
    elif raw_wmem_depth < 64:
        return 64
    elif raw_wmem_depth < 128:
        return 128
    else :
        # Round up to the nearest 512
        return int(math.ceil(raw_wmem_depth / 512) * 512)
    

    
def get_token_delay(clk_period, n_model, gbus_width, n_heads, n_cols, max_context_length, sequence_length=512 ,n_layers=1, ffn_ratio=4, softmax_choice='SOFTMAX', activation_choice='RELU'):
    """
    Calculate the token delay based on the design parameters.
    """
    # coerce inputs to numeric types (some callers pass strings)
    try:
        clk_period = float(clk_period)
    except Exception:
        clk_period = float(str(clk_period))
    n_model = float(n_model)
    gbus_width = float(gbus_width)
    n_heads = int(n_heads)
    n_cols = int(n_cols)
    max_context_length = int(max_context_length)
    sequence_length = int(sequence_length)
    n_layers = int(n_layers)
    ffn_ratio = float(ffn_ratio)

    mac_num = int(gbus_width / 8)
    # Calculate the token delay
    # for GEMM ops

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

    token_delay /= 0.8 # assuming 80% efficiency

    token_delay *= n_layers

    return token_delay * 1000 # convert to us

def get_TTFT(clk_period, n_model, mac_num, n_heads, n_cols, max_context_length, sequence_length=256 ,n_layers=1, ffn_ratio=4, softmax_choice='SOFTMAX', activation_choice='RELU'):
    """
    Calculate the time to first token (TTFT) based on the design parameters.
    """
    # coerce inputs to numeric types (some callers pass strings)
    # try:
    #     clk_period = float(clk_period)
    # except Exception:
    #     clk_period = float(str(clk_period))
    n_model = float(n_model)
    mac_num = float(mac_num)
    n_heads = float(n_heads)
    n_cols = float(n_cols)
    max_context_length = float(max_context_length)
    sequence_length = float(sequence_length)
    n_layers = float(n_layers)
    ffn_ratio = float(ffn_ratio)

    # Calculate the token delay for GEMM ops
    TTFT = (4.0 * n_model * n_model + 2.0 * ffn_ratio * n_model * n_model) * 1e-9 * clk_period / gbus_width # load and store delay

    # Number of cycles = Total MACs / (Number of MAC units)
    num_loading_cycles = (4*n_model*n_model*sequence_length + 2*max_context_length*max_context_length*n_model + 2*ffn_ratio*n_model*n_model*sequence_length) / (n_heads*n_cols*mac_num)
    TTFT += num_loading_cycles * clk_period * 1e-9 # seconds

    # add 2 residual delay
    TTFT += 2 * n_model * sequence_length * 1e-9 * clk_period # residual add delay

    # add memory loading delay

    # add softmax delay
    # if softmax_choice == 'SOFTMAX':
    #     TTFT += 7 * max_context_length * max_context_length * 1e-9 * clk_period 
    # elif softmax_choice == 'SOFTERMAX':
    #     TTFT += 2 * max_context_length * max_context_length * 1e-9 * clk_period 
    # elif softmax_choice == 'CONSMAX':
    #     TTFT += 4 * 1e-9 * clk_period     #fully pipelined

    # activation delay is negligible

    TTFT /= 0.9 # assuming 90% efficiency

    TTFT *= n_layers

    return TTFT # convert to s

def get_TPOT(clk_period, n_model, mac_num, n_heads, n_cols, max_context_length, sequence_length=256 ,n_layers=1, ffn_ratio=4, softmax_choice='SOFTMAX', activation_choice='RELU'):
    """
    Calculate the time per output token (TPOT) based on the design parameters.
    """
    # Calculate the token delay
    # for GEMM ops

    # coerce numeric inputs
    n_model = float(n_model)
    mac_num = float(mac_num)
    n_heads = float(n_heads)
    n_cols = float(n_cols)
    max_context_length = float(max_context_length)
    sequence_length = float(sequence_length)
    n_layers = float(n_layers)
    ffn_ratio = float(ffn_ratio)

    TPOT = (4.0 * n_model * n_model + 2.0 * ffn_ratio * n_model * n_model) * 1e-9 * clk_period # load and store delay

    # Number of cycles = Total MACs / (Number of MAC units)
    num_loading_cycles = (4*n_model*n_model + 2*sequence_length*sequence_length*n_model + 2*ffn_ratio*n_model*n_model) / (n_heads*n_cols*mac_num)
    TPOT += num_loading_cycles * clk_period * 1e-9 # seconds

    # add 2 residual delay
    TPOT += 2 * n_model * 1e-9 * clk_period # residual add delay

    # add memory loading delay

    # add softmax delay
    # if softmax_choice == 'SOFTMAX':
    #     TTFT += 7 * max_context_length * max_context_length * 1e-9 * clk_period 
    # elif softmax_choice == 'SOFTERMAX':
    #     TTFT += 2 * max_context_length * max_context_length * 1e-9 * clk_period 
    # elif softmax_choice == 'CONSMAX':
    #     TTFT += 4 * 1e-9 * clk_period     #fully pipelined

    # activation delay is negligible

    TPOT /= 0.9 # assuming 90% efficiency

    TPOT *= n_layers

    return TPOT # convert to s

def get_EMA_in_byte(clk_period, n_model, mac_num, n_heads, n_cols, max_context_length, sequence_length=256 ,n_layers=1, ffn_ratio=4):
    """
    Calculate the energy memory access (EMA) in bytes based on the design parameters.
    """
    # coerce inputs to numeric types (some callers pass strings)
    n_model = float(n_model)
    mac_num = float(mac_num)
    n_heads = float(n_heads)
    n_cols = float(n_cols)
    max_context_length = float(max_context_length)
    sequence_length = float(sequence_length)
    n_layers = float(n_layers)
    ffn_ratio = float(ffn_ratio)

    layer_size_in_bytes = 4 * n_model * n_model + 2 * ffn_ratio * n_model * n_model # load and store delay



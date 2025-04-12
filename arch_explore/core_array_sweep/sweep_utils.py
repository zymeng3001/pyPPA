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
    
def get_token_delay(clk_period, n_model, gbus_width, n_heads, n_cols, max_context_length):
    """
    Calculate the token delay based on the design parameters.
    """
    efficiency_ratio = 0.7
    mac_num = int(gbus_width / 8)
    tokens_per_second = 1e9 * efficiency_ratio * (n_heads*n_cols*mac_num)/ (clk_period*(4*n_model*n_model + 2*max_context_length*n_model + 2*4*n_model*n_model))
    # Calculate the token delay
    token_delay = 1 / tokens_per_second
    
    return token_delay
    

    
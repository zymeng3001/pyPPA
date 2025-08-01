# 假设这些是你的预定义常量
HEAD_NUM = 8 # 头的数量
HEAD_CORE_NUM = 16  # 每个头的核心数


# 模拟生成代码的函数
def generate_sv_code():
    sv_code = []
    sv_code.append(f"task init_wmem();")
    sv_code.append(f"begin")
    # 循环生成每个 h 和 i 组合
    for h in range(HEAD_NUM):
        for i in range(HEAD_CORE_NUM):
            
            # 生成模块路径
            module_path = f"chip_top.top_array_top_inst_inst_head_array_head_gen_array_{h//2}__two_heads_inst.inst_head_top_{h%2}_inst_head_core_array_head_cores_generate_array_{i}__core_top_inst_mem_inst"
            sv_code.append(f"    // 生成针对 h={h}, i={i} 的路径初始化代码")
            
            # 初始化 Q 权重
            sv_code.append(f"    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin")
            sv_code.append(f"        for(int k = 0; k < `MAC_MULT_NUM; k++) begin")
            sv_code.append(f"            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);")
            sv_code.append(f"            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;")
            sv_code.append(f"            sram_temp_var[k*8 +: 8] = Q_weights_array[{h}][temp_x][temp_y];")
            sv_code.append(f"        end")
            sv_code.append(f"        {module_path}.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];")
            sv_code.append(f"        {module_path}.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];")
            sv_code.append(f"    end")
            
            # 初始化 K 权重
            sv_code.append(f"    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin")
            sv_code.append(f"        for(int k = 0; k < `MAC_MULT_NUM; k++) begin")
            sv_code.append(f"            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);")
            sv_code.append(f"            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;")
            sv_code.append(f"            sram_temp_var[k*8 +: 8] = K_weights_array[{h}][temp_x][temp_y];")
            sv_code.append(f"        end")
            sv_code.append(f"        {module_path}.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];")
            sv_code.append(f"        {module_path}.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];")
            sv_code.append(f"    end")
            
            # 初始化 V 权重
            sv_code.append(f"    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin")
            sv_code.append(f"        for(int k = 0; k < `MAC_MULT_NUM; k++) begin")
            sv_code.append(f"            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);")
            sv_code.append(f"            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;")
            sv_code.append(f"            sram_temp_var[k*8 +: 8] = V_weights_array[{h}][temp_x][temp_y];")
            sv_code.append(f"        end")
            sv_code.append(f"        {module_path}.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];")
            sv_code.append(f"        {module_path}.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];")
            sv_code.append(f"    end")
            
            # 初始化 PROJ 权重
            sv_code.append(f"    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin")
            sv_code.append(f"        for(int k = 0; k < `MAC_MULT_NUM; k++) begin")
            sv_code.append(f"            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);")
            sv_code.append(f"            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;")
            sv_code.append(f"            sram_temp_var[k*8 +: 8] = PROJ_weights_array[{h}][temp_x][temp_y];")
            sv_code.append(f"        end")
            sv_code.append(f"        {module_path}.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];")
            sv_code.append(f"        {module_path}.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];")
            sv_code.append(f"    end")
            
            # 初始化 FFN0 权重
            sv_code.append(f"    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin")
            sv_code.append(f"        for(int k = 0; k < `MAC_MULT_NUM; k++) begin")
            sv_code.append(f"            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);")
            sv_code.append(f"            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;")
            sv_code.append(f"            sram_temp_var[k*8 +: 8] = FFN0_weights_array[{h}][temp_x][temp_y];")
            sv_code.append(f"        end")
            sv_code.append(f"        {module_path}.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];")
            sv_code.append(f"        {module_path}.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];")
            sv_code.append(f"    end")
            
            # 初始化 FFN1 权重
            sv_code.append(f"    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin")
            sv_code.append(f"        for(int k = 0; k < `MAC_MULT_NUM; k++) begin")
            sv_code.append(f"            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);")
            sv_code.append(f"            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;")
            sv_code.append(f"            sram_temp_var[k*8 +: 8] = FFN1_weights_array[{h}][temp_x][temp_y];")
            sv_code.append(f"        end")
            sv_code.append(f"        {module_path}.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];")
            sv_code.append(f"        {module_path}.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];")
            sv_code.append(f"    end")

    sv_code.append(f"end")
    sv_code.append(f"endtask")
    return "\n".join(sv_code)

sv_generated_code = generate_sv_code()


# 可选：将代码输出到日志文件
with open("generated_sv_code.svh", "w") as f:
    f.write(sv_generated_code)

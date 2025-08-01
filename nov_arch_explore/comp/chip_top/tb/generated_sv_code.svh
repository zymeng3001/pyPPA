task init_wmem();
begin
    // 生成针对 h=0, i=0 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=0, i=1 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=0, i=2 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=0, i=3 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=0, i=4 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=0, i=5 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=0, i=6 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=0, i=7 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=0, i=8 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=0, i=9 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=0, i=10 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=0, i=11 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=0, i=12 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=0, i=13 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=0, i=14 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=0, i=15 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[0][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=1, i=0 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=1, i=1 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=1, i=2 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=1, i=3 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=1, i=4 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=1, i=5 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=1, i=6 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=1, i=7 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=1, i=8 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=1, i=9 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=1, i=10 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=1, i=11 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=1, i=12 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=1, i=13 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=1, i=14 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=1, i=15 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[1][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_0__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=2, i=0 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=2, i=1 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=2, i=2 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=2, i=3 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=2, i=4 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=2, i=5 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=2, i=6 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=2, i=7 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=2, i=8 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=2, i=9 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=2, i=10 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=2, i=11 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=2, i=12 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=2, i=13 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=2, i=14 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=2, i=15 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[2][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=3, i=0 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=3, i=1 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=3, i=2 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=3, i=3 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=3, i=4 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=3, i=5 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=3, i=6 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=3, i=7 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=3, i=8 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=3, i=9 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=3, i=10 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=3, i=11 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=3, i=12 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=3, i=13 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=3, i=14 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=3, i=15 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[3][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_1__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=4, i=0 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=4, i=1 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=4, i=2 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=4, i=3 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=4, i=4 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=4, i=5 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=4, i=6 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=4, i=7 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=4, i=8 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=4, i=9 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=4, i=10 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=4, i=11 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=4, i=12 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=4, i=13 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=4, i=14 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=4, i=15 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[4][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=5, i=0 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=5, i=1 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=5, i=2 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=5, i=3 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=5, i=4 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=5, i=5 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=5, i=6 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=5, i=7 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=5, i=8 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=5, i=9 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=5, i=10 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=5, i=11 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=5, i=12 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=5, i=13 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=5, i=14 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=5, i=15 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[5][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_2__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=6, i=0 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=6, i=1 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=6, i=2 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=6, i=3 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=6, i=4 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=6, i=5 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=6, i=6 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=6, i=7 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=6, i=8 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=6, i=9 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=6, i=10 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=6, i=11 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=6, i=12 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=6, i=13 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=6, i=14 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=6, i=15 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[6][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_0_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=7, i=0 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_0__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=7, i=1 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_1__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=7, i=2 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_2__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=7, i=3 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_3__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=7, i=4 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_4__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=7, i=5 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_5__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=7, i=6 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_6__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=7, i=7 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_7__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=7, i=8 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_8__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=7, i=9 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_9__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=7, i=10 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_10__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=7, i=11 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_11__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=7, i=12 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_12__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=7, i=13 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_13__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=7, i=14 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_14__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
    // 生成针对 h=7, i=15 的路径初始化代码
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = Q_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = K_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + K_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = V_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + V_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = PROJ_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j + PROJ_WEIGHT_ADDR_BASE] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4) + j/(TB_EMBD_SIZE/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN0_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j] = sram_temp_var[127:64];
    end
    for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE /`HEAD_NUM  / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++) begin
        for(int k = 0; k < `MAC_MULT_NUM; k++) begin
            temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM) + j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM);
            temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM)) * `MAC_MULT_NUM + k;
            sram_temp_var[k*8 +: 8] = FFN1_weights_array[7][temp_x][temp_y];
        end
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[63:0];
        chip_top.top_array_top_inst_inst_head_array_head_gen_array_3__two_heads_inst.inst_head_top_1_inst_head_core_array_head_cores_generate_array_15__core_top_inst_mem_inst.wmem_inst_true_mem_wrapper_wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j + 512] = sram_temp_var[127:64];
    end
end
endtask
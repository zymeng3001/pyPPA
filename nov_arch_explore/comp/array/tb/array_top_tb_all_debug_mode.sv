`define CHECK_QK
`define CHECK_PV
`define CHECK_PROJ
// `define CHECK_ATT_RESIDUAL
`define CHECK_FFN0
`define CHECK_FFN1
`define CHECK_MLP_RESIDUAL

module array_top_tb_all ();
logic                           clk;
logic                           rst_n;

parameter FORCE_TO_BE_RIGHT = 0;

parameter QKT_MAX_ERR = 6;
parameter PV_MAX_ERR = 6;
parameter PROJ_MAX_ERR = 6;
parameter ATT_RESIDUAL_MAX_ERR = 6;
parameter FFN0_MAX_ERR = 6;
parameter FFN1_MAX_ERR = 6;
parameter MLP_RESIDUAL_MAX_ERR = 6;

parameter GQA_EN = 0;
parameter PMU_CFG_EN = 1;
parameter DEEPSLEEP_EN = 0;

parameter CDATA_ACCU_NUM_WIDTH = `CDATA_ACCU_NUM_WIDTH;
parameter CDATA_SCALE_WIDTH = `CDATA_SCALE_WIDTH;
parameter CDATA_BIAS_WIDTH = `CDATA_BIAS_WIDTH;
parameter CDATA_SHIFT_WIDTH = `CDATA_SHIFT_WIDTH;

localparam K_WEIGHT_ADDR_BASE = (`MAX_EMBD_SIZE * `MAX_EMBD_SIZE / `HEAD_NUM / `HEAD_CORE_NUM / `MAC_MULT_NUM) * 1;
localparam V_WEIGHT_ADDR_BASE = (`MAX_EMBD_SIZE * `MAX_EMBD_SIZE / `HEAD_NUM / `HEAD_CORE_NUM / `MAC_MULT_NUM) * 2;
localparam PROJ_WEIGHT_ADDR_BASE = (`MAX_EMBD_SIZE * `MAX_EMBD_SIZE / `HEAD_NUM / `HEAD_CORE_NUM / `MAC_MULT_NUM) * 3;
localparam FFN0_WEIGHT_ADDR_BASE = (`WMEM_DEPTH/`WMEM_NUM_PER_CORE);
localparam FFN1_WEIGHT_ADDR_BASE = (`WMEM_DEPTH/`WMEM_NUM_PER_CORE*2);

localparam K_CACHE_ADDR_BASE = 0;
localparam V_CACHE_ADDR_BASE = (`MAX_CONTEXT_LENGTH * `MAX_EMBD_SIZE / `HEAD_NUM / `HEAD_CORE_NUM / `MAC_MULT_NUM);

parameter  TB_EMBD_SIZE = `MAX_EMBD_SIZE;
parameter  TB_MAX_CONTEXT_LENGTH = `MAX_CONTEXT_LENGTH; //Hardware configured max context length, NEEDS TO BE Divided by HEAD_CORE_NUM
parameter  TB_TOTAL_CONTEXT_LENGTH = (TB_MAX_CONTEXT_LENGTH/32); //input prompt size + generated prompt size for each user
parameter  TB_NUM_USER = 4;
parameter  TB_QKV_WEIGHT_COLS_PER_CORE = (TB_EMBD_SIZE/`HEAD_NUM/`HEAD_CORE_NUM) ;
parameter  TB_TOKEN_PER_CORE = (TB_MAX_CONTEXT_LENGTH/`HEAD_CORE_NUM);
parameter  TB_TEST_ITER = TB_TOTAL_CONTEXT_LENGTH * TB_NUM_USER; //How many iteration this testbench will test.

parameter RC_SHIFT = 16;

parameter RMS_K = 128;
shortreal attn_rms_dequant_scale_square = 0.0123456;
shortreal mlp_rms_dequant_scale_square =  0.01404802;

parameter Q_GEN_SCALE = 120;
parameter Q_GEN_SHIFT = 15;

parameter K_GEN_SCALE = 135;
parameter K_GEN_SHIFT = 15;

parameter V_GEN_SCALE = 24;
parameter V_GEN_SHIFT = 13;

parameter QKT_SCALE = 86;
parameter QKT_SHIFT = 15;

shortreal SOFTMAX_DEQUANT_SCALE = 0.06123456;
shortreal SOFTMAX_QUANT_SCALE = 128;

parameter PV_SCALE = 247;
parameter PV_SHIFT = 14;

parameter PROJ_SCALE = 247;
parameter PROJ_SHIFT = 19;

parameter ATTN_RESIDUAL_SCALE_A = 320;
parameter ATTN_RESIDUAL_SCALE_B = 279;
parameter ATTN_RESIDUAL_SHIFT = 9;

parameter FFN0_SCALE = 216;
parameter FFN0_SHIFT = 16;

parameter FFN1_SCALE = 201;
parameter FFN1_SHIFT = 18;

parameter MLP_RESIDUAL_SCALE_A = 213;
parameter MLP_RESIDUAL_SCALE_B = 202;
parameter MLP_RESIDUAL_SHIFT = 8;

function automatic int zero_round(real num);
        return int'(num + 0.5);
endfunction


function shortreal bitsbfloat16_to_shortreal;
    input [15:0] x;
    begin
        logic [31:0] x_float;
        x_float = {x,16'b0};
        bitsbfloat16_to_shortreal = $bitstoshortreal(x_float);
    end
endfunction

function [15:0] shortreal_to_bitsbfloat16;
    input real x;
    begin
        logic [31:0] x_float_bits;
        x_float_bits = $shortrealtobits(x);
        shortreal_to_bitsbfloat16 = x_float_bits[31:16] + x_float_bits[15];
    end
endfunction


 logic       [`USER_ID_WIDTH-1:0]                     user_id;// be in same phase


logic [`IDATA_WIDTH-1:0] input_x_array                    [TB_NUM_USER-1:0][TB_TOTAL_CONTEXT_LENGTH-1:0][TB_EMBD_SIZE-1:0];
logic [`IDATA_WIDTH-1:0] Q_weights_array      [`HEAD_NUM-1:0][TB_EMBD_SIZE-1:0][TB_EMBD_SIZE/`HEAD_NUM-1:0];
logic [`IDATA_WIDTH-1:0] K_weights_array      [`HEAD_NUM-1:0][TB_EMBD_SIZE-1:0][TB_EMBD_SIZE/`HEAD_NUM-1:0];
logic [`IDATA_WIDTH-1:0] V_weights_array      [`HEAD_NUM-1:0][TB_EMBD_SIZE-1:0][TB_EMBD_SIZE/`HEAD_NUM-1:0];
logic [`IDATA_WIDTH-1:0] PROJ_weights_array   [`HEAD_NUM-1:0][TB_EMBD_SIZE/`HEAD_NUM-1:0][TB_EMBD_SIZE-1:0];
logic [`IDATA_WIDTH-1:0] FFN0_weights_array   [`HEAD_NUM-1:0][TB_EMBD_SIZE-1:0][4*TB_EMBD_SIZE/`HEAD_NUM-1:0];
logic [`IDATA_WIDTH-1:0] FFN1_weights_array   [`HEAD_NUM-1:0][4*TB_EMBD_SIZE/`HEAD_NUM-1:0][TB_EMBD_SIZE-1:0];
logic [`IDATA_WIDTH-1:0] K_array           [`HEAD_NUM-1:0][TB_NUM_USER-1:0][TB_TOTAL_CONTEXT_LENGTH-1:0][TB_EMBD_SIZE/`HEAD_NUM-1:0];
logic [`IDATA_WIDTH-1:0] Q_array           [`HEAD_NUM-1:0][TB_NUM_USER-1:0][TB_TOTAL_CONTEXT_LENGTH-1:0][TB_EMBD_SIZE/`HEAD_NUM-1:0];
logic [`IDATA_WIDTH-1:0] V_array           [`HEAD_NUM-1:0][TB_NUM_USER-1:0][TB_TOTAL_CONTEXT_LENGTH-1:0][TB_EMBD_SIZE/`HEAD_NUM-1:0];
logic [`IDATA_WIDTH-1:0] ATT_QK_array      [`HEAD_NUM-1:0][TB_NUM_USER-1:0][TB_TOTAL_CONTEXT_LENGTH-1:0][TB_TOTAL_CONTEXT_LENGTH-1:0];
logic [`IDATA_WIDTH-1:0] ATT_PV_array      [`HEAD_NUM-1:0][TB_NUM_USER-1:0][TB_TOTAL_CONTEXT_LENGTH-1:0][TB_EMBD_SIZE/`HEAD_NUM-1:0];
logic [`IDATA_WIDTH-1:0] ATT_PROJ_array                    [TB_NUM_USER-1:0][TB_TOTAL_CONTEXT_LENGTH-1:0][TB_EMBD_SIZE-1:0];
logic [`IDATA_WIDTH-1:0] ATT_RESIDUAL_array                [TB_NUM_USER-1:0][TB_TOTAL_CONTEXT_LENGTH-1:0][TB_EMBD_SIZE-1:0];
logic [`IDATA_WIDTH-1:0] FFN0_array        [`HEAD_NUM-1:0][TB_NUM_USER-1:0][TB_TOTAL_CONTEXT_LENGTH-1:0][4*TB_EMBD_SIZE/`HEAD_NUM-1:0];
logic [`IDATA_WIDTH-1:0] FFN1_array                        [TB_NUM_USER-1:0][TB_TOTAL_CONTEXT_LENGTH-1:0][TB_EMBD_SIZE-1:0];
logic [`IDATA_WIDTH-1:0] MLP_RESIDUAL_array                [TB_NUM_USER-1:0][TB_TOTAL_CONTEXT_LENGTH-1:0][TB_EMBD_SIZE-1:0];
int error_flag=0;
logic signed [`IDATA_WIDTH-1:0] max_value; 
logic signed [`IDATA_WIDTH-1:0] min_value; 

initial begin
    $readmemh("../comp/array/tb/py_head_tb_hex_gen/input_x.hex", input_x_array);
    $readmemh("../comp/array/tb/py_head_tb_hex_gen/Q_weights.hex", Q_weights_array);
    $readmemh("../comp/array/tb/py_head_tb_hex_gen/K_weights.hex", K_weights_array);
    $readmemh("../comp/array/tb/py_head_tb_hex_gen/V_weights.hex", V_weights_array);
    $readmemh("../comp/array/tb/py_head_tb_hex_gen/PROJ_weights.hex", PROJ_weights_array);
    $readmemh("../comp/array/tb/py_head_tb_hex_gen/FFN0_weights.hex", FFN0_weights_array);
    $readmemh("../comp/array/tb/py_head_tb_hex_gen/FFN1_weights.hex", FFN1_weights_array);
end



always begin
    #(`CLOCK_PERIOD/2.0) clk = ~clk;
end

array_top inst_array_top(
    .clk(clk),
    .rst_n(rst_n),

    .interface_addr(22'b0), //保证进来的时序是干净的

    .interface_wen(1'b0),
    .interface_wdata(16'b0),

    .interface_ren(1'b0),
    .interface_rdata(),
    .interface_rvalid() //pulse, when high means that all the config all initial successfully
);



int WMEM_INIT_FLAG = 0;
logic [(`MAC_MULT_NUM * `IDATA_WIDTH)-1:0] sram_temp_var;
integer temp_x;
integer temp_y;
genvar h;
genvar i;
genvar k;
generate
    for(h = 0; h < `HEAD_NUM; h++)begin 
        for(i = 0; i < `HEAD_CORE_NUM; i++)begin
            initial begin
                #1;
                $display("HEAD[%0d].CORE[%0d] CORE WEIGHT SRAM INIT",h,i);
                //Initialize Q weights array
                for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE / `HEAD_NUM / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++)begin //Q_weight depth in one core mem
                    for(int k =0; k <`MAC_MULT_NUM;k++)begin
                        temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE+j/(TB_EMBD_SIZE/`MAC_MULT_NUM); //`EMBD_SIZE是权重矩阵的行数，这里决定第几列,TB_EMBD_SIZE/`MAC_MULT_NUM是存一列权重需要的行数
                        temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM))  * `MAC_MULT_NUM + k;//这里决定那一列中的第几行
                        sram_temp_var[k*8 +: 8] = Q_weights_array[h][temp_x][temp_y];
                        // $display(temp_x);
                    end
                    // $display("%0h",sram_temp_var);
                    if(h%2 == 0)begin
                    `ifndef TRUE_MEM
                    inst_array_top.inst_head_array.head_gen_array[h/2].two_heads_inst.inst_head_top_0.inst_head_core_array.head_cores_generate_array[i].core_top_inst.mem_inst.wmem_inst.inst_mem_sp.mem[j]=sram_temp_var;
                    `else    
                    inst_array_top.inst_head_array.head_gen_array[h/2].two_heads_inst.inst_head_top_0.inst_head_core_array.head_cores_generate_array[i].core_top_inst.mem_inst.wmem_inst.true_mem_wrapper.wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j]=sram_temp_var[63:0];
                    inst_array_top.inst_head_array.head_gen_array[h/2].two_heads_inst.inst_head_top_0.inst_head_core_array.head_cores_generate_array[i].core_top_inst.mem_inst.wmem_inst.true_mem_wrapper.wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j]=sram_temp_var[127:64];
                    `endif
                    end
                    else begin
                    `ifndef TRUE_MEM
                    inst_array_top.inst_head_array.head_gen_array[h/2].two_heads_inst.inst_head_top_1.inst_head_core_array.head_cores_generate_array[i].core_top_inst.mem_inst.wmem_inst.inst_mem_sp.mem[j]=sram_temp_var;
                    `else    
                    inst_array_top.inst_head_array.head_gen_array[h/2].two_heads_inst.inst_head_top_1.inst_head_core_array.head_cores_generate_array[i].core_top_inst.mem_inst.wmem_inst.true_mem_wrapper.wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j]=sram_temp_var[63:0];
                    inst_array_top.inst_head_array.head_gen_array[h/2].two_heads_inst.inst_head_top_1.inst_head_core_array.head_cores_generate_array[i].core_top_inst.mem_inst.wmem_inst.true_mem_wrapper.wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j]=sram_temp_var[127:64];
                    `endif
                    end
                end

                //Initialize K weights array
                for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE / `HEAD_NUM / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++)begin //K_weight depth in one core mem
                    for(int k =0; k <`MAC_MULT_NUM;k++)begin
                        temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE+j/(TB_EMBD_SIZE/`MAC_MULT_NUM); //`EMBD_SIZE是权重矩阵的行数，这里决定第几列,TB_EMBD_SIZE/`MAC_MULT_NUM是存一列权重需要的行数
                        temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM))  * `MAC_MULT_NUM + k;//这里决定那一列中的第几行
                        sram_temp_var[k*8 +: 8] = K_weights_array[h][temp_x][temp_y];
                        // $display(temp_x);
                    end
                    // $display("%0h",sram_temp_var);
                    if(h%2 == 0)begin
                    `ifndef TRUE_MEM
                    inst_array_top.inst_head_array.head_gen_array[h/2].two_heads_inst.inst_head_top_0.inst_head_core_array.head_cores_generate_array[i].core_top_inst.mem_inst.wmem_inst.inst_mem_sp.mem[j+K_WEIGHT_ADDR_BASE]=sram_temp_var;
                    `else    
                    inst_array_top.inst_head_array.head_gen_array[h/2].two_heads_inst.inst_head_top_0.inst_head_core_array.head_cores_generate_array[i].core_top_inst.mem_inst.wmem_inst.true_mem_wrapper.wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j+K_WEIGHT_ADDR_BASE]=sram_temp_var[63:0];
                    inst_array_top.inst_head_array.head_gen_array[h/2].two_heads_inst.inst_head_top_0.inst_head_core_array.head_cores_generate_array[i].core_top_inst.mem_inst.wmem_inst.true_mem_wrapper.wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j+K_WEIGHT_ADDR_BASE]=sram_temp_var[127:64];
                    `endif
                    end
                    else begin
                    `ifndef TRUE_MEM
                    inst_array_top.inst_head_array.head_gen_array[h/2].two_heads_inst.inst_head_top_1.inst_head_core_array.head_cores_generate_array[i].core_top_inst.mem_inst.wmem_inst.inst_mem_sp.mem[j+K_WEIGHT_ADDR_BASE]=sram_temp_var;
                    `else    
                    inst_array_top.inst_head_array.head_gen_array[h/2].two_heads_inst.inst_head_top_1.inst_head_core_array.head_cores_generate_array[i].core_top_inst.mem_inst.wmem_inst.true_mem_wrapper.wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j+K_WEIGHT_ADDR_BASE]=sram_temp_var[63:0];
                    inst_array_top.inst_head_array.head_gen_array[h/2].two_heads_inst.inst_head_top_1.inst_head_core_array.head_cores_generate_array[i].core_top_inst.mem_inst.wmem_inst.true_mem_wrapper.wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j+K_WEIGHT_ADDR_BASE]=sram_temp_var[127:64];
                    `endif
                    end
                end

                
                //Initialize V weights array
                for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE / `HEAD_NUM / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++)begin //V_weight depth in one core mem
                    for(int k =0; k <`MAC_MULT_NUM;k++)begin
                        temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE+j/(TB_EMBD_SIZE/`MAC_MULT_NUM); //`EMBD_SIZE是权重矩阵的行数，这里决定第几列,TB_EMBD_SIZE/`MAC_MULT_NUM是存一列权重需要的行数
                        temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM))  * `MAC_MULT_NUM + k;//这里决定那一列中的第几行
                        sram_temp_var[k*8 +: 8] = V_weights_array[h][temp_x][temp_y];
                        // $display(temp_x);
                    end
                    if(h%2 == 0)begin
                    `ifndef TRUE_MEM
                    inst_array_top.inst_head_array.head_gen_array[h/2].two_heads_inst.inst_head_top_0.inst_head_core_array.head_cores_generate_array[i].core_top_inst.mem_inst.wmem_inst.inst_mem_sp.mem[j+V_WEIGHT_ADDR_BASE]=sram_temp_var;
                    `else    
                    inst_array_top.inst_head_array.head_gen_array[h/2].two_heads_inst.inst_head_top_0.inst_head_core_array.head_cores_generate_array[i].core_top_inst.mem_inst.wmem_inst.true_mem_wrapper.wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j+V_WEIGHT_ADDR_BASE]=sram_temp_var[63:0];
                    inst_array_top.inst_head_array.head_gen_array[h/2].two_heads_inst.inst_head_top_0.inst_head_core_array.head_cores_generate_array[i].core_top_inst.mem_inst.wmem_inst.true_mem_wrapper.wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j+V_WEIGHT_ADDR_BASE]=sram_temp_var[127:64];
                    `endif
                    end
                    else begin
                    `ifndef TRUE_MEM
                    inst_array_top.inst_head_array.head_gen_array[h/2].two_heads_inst.inst_head_top_1.inst_head_core_array.head_cores_generate_array[i].core_top_inst.mem_inst.wmem_inst.inst_mem_sp.mem[j+V_WEIGHT_ADDR_BASE]=sram_temp_var;
                    `else    
                    inst_array_top.inst_head_array.head_gen_array[h/2].two_heads_inst.inst_head_top_1.inst_head_core_array.head_cores_generate_array[i].core_top_inst.mem_inst.wmem_inst.true_mem_wrapper.wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j+V_WEIGHT_ADDR_BASE]=sram_temp_var[63:0];
                    inst_array_top.inst_head_array.head_gen_array[h/2].two_heads_inst.inst_head_top_1.inst_head_core_array.head_cores_generate_array[i].core_top_inst.mem_inst.wmem_inst.true_mem_wrapper.wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j+V_WEIGHT_ADDR_BASE]=sram_temp_var[127:64];
                    `endif
                    end
                end

                for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE / `HEAD_NUM / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++)begin //Proj_weight depth in one core mem
                    for(int k =0; k <`MAC_MULT_NUM;k++)begin
                        temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM)+j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM); //TB_EMBD_SIZE/`HEAD_NUM是权重矩阵的行数，这里决定第几列,TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM是存一列权重需要的行数
                        temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM / `MAC_MULT_NUM))  * `MAC_MULT_NUM + k;//这里决定那一列中的第几行
                        sram_temp_var[k*8 +: 8] = PROJ_weights_array[h][temp_x][temp_y];
                        // $display(temp_x);
                    end
                    if(h%2 == 0)begin
                    `ifndef TRUE_MEM
                    inst_array_top.inst_head_array.head_gen_array[h/2].two_heads_inst.inst_head_top_0.inst_head_core_array.head_cores_generate_array[i].core_top_inst.mem_inst.wmem_inst.inst_mem_sp.mem[j+PROJ_WEIGHT_ADDR_BASE]=sram_temp_var;
                    `else    
                    inst_array_top.inst_head_array.head_gen_array[h/2].two_heads_inst.inst_head_top_0.inst_head_core_array.head_cores_generate_array[i].core_top_inst.mem_inst.wmem_inst.true_mem_wrapper.wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j+PROJ_WEIGHT_ADDR_BASE]=sram_temp_var[63:0];
                    inst_array_top.inst_head_array.head_gen_array[h/2].two_heads_inst.inst_head_top_0.inst_head_core_array.head_cores_generate_array[i].core_top_inst.mem_inst.wmem_inst.true_mem_wrapper.wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j+PROJ_WEIGHT_ADDR_BASE]=sram_temp_var[127:64];
                    `endif
                    end
                    else begin
                    `ifndef TRUE_MEM
                    inst_array_top.inst_head_array.head_gen_array[h/2].two_heads_inst.inst_head_top_1.inst_head_core_array.head_cores_generate_array[i].core_top_inst.mem_inst.wmem_inst.inst_mem_sp.mem[j+PROJ_WEIGHT_ADDR_BASE]=sram_temp_var;
                    `else    
                    inst_array_top.inst_head_array.head_gen_array[h/2].two_heads_inst.inst_head_top_1.inst_head_core_array.head_cores_generate_array[i].core_top_inst.mem_inst.wmem_inst.true_mem_wrapper.wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j+PROJ_WEIGHT_ADDR_BASE]=sram_temp_var[63:0];
                    inst_array_top.inst_head_array.head_gen_array[h/2].two_heads_inst.inst_head_top_1.inst_head_core_array.head_cores_generate_array[i].core_top_inst.mem_inst.wmem_inst.true_mem_wrapper.wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j+PROJ_WEIGHT_ADDR_BASE]=sram_temp_var[127:64];
                    `endif
                    end
                end

                for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE / `HEAD_NUM / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++)begin //ffn0_weight depth in one core mem
                    for(int k =0; k <`MAC_MULT_NUM;k++)begin
                        temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4)+j/(TB_EMBD_SIZE/`MAC_MULT_NUM); //`EMBD_SIZE是权重矩阵的行数，这里决定第几列,TB_EMBD_SIZE/`MAC_MULT_NUM是存一列权重需要的行数
                        temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM))  * `MAC_MULT_NUM + k;//这里决定那一列中的第几行
                        sram_temp_var[k*8 +: 8] = FFN0_weights_array[h][temp_x][temp_y];
                        // $display(temp_x);
                    end
                    if(h%2 == 0)begin
                    `ifndef TRUE_MEM
                    inst_array_top.inst_head_array.head_gen_array[h/2].two_heads_inst.inst_head_top_0.inst_head_core_array.head_cores_generate_array[i].core_top_inst.mem_inst.wmem_inst.inst_mem_sp.mem[j+FFN0_WEIGHT_ADDR_BASE]=sram_temp_var;
                    `else    
                    inst_array_top.inst_head_array.head_gen_array[h/2].two_heads_inst.inst_head_top_0.inst_head_core_array.head_cores_generate_array[i].core_top_inst.mem_inst.wmem_inst.true_mem_wrapper.wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j]=sram_temp_var[63:0];
                    inst_array_top.inst_head_array.head_gen_array[h/2].two_heads_inst.inst_head_top_0.inst_head_core_array.head_cores_generate_array[i].core_top_inst.mem_inst.wmem_inst.true_mem_wrapper.wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j]=sram_temp_var[127:64];
                    `endif
                    end
                    else begin
                    `ifndef TRUE_MEM
                    inst_array_top.inst_head_array.head_gen_array[h/2].two_heads_inst.inst_head_top_1.inst_head_core_array.head_cores_generate_array[i].core_top_inst.mem_inst.wmem_inst.inst_mem_sp.mem[j+FFN0_WEIGHT_ADDR_BASE]=sram_temp_var;
                    `else    
                    inst_array_top.inst_head_array.head_gen_array[h/2].two_heads_inst.inst_head_top_1.inst_head_core_array.head_cores_generate_array[i].core_top_inst.mem_inst.wmem_inst.true_mem_wrapper.wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j]=sram_temp_var[63:0];
                    inst_array_top.inst_head_array.head_gen_array[h/2].two_heads_inst.inst_head_top_1.inst_head_core_array.head_cores_generate_array[i].core_top_inst.mem_inst.wmem_inst.true_mem_wrapper.wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j]=sram_temp_var[127:64];
                    `endif
                    end
                end

                for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE / `HEAD_NUM / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++)begin //ffn1_weight depth in one core mem
                    for(int k =0; k <`MAC_MULT_NUM;k++)begin
                        temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM)+j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM); //4*TB_EMBD_SIZE/`HEAD_NUM是权重矩阵的行数，这里决定第几列,4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM是存一列权重需要的行数
                        temp_x = (j % (4*TB_EMBD_SIZE/`HEAD_NUM / `MAC_MULT_NUM))  * `MAC_MULT_NUM + k;//这里决定那一列中的第几行
                        sram_temp_var[k*8 +: 8] = FFN1_weights_array[h][temp_x][temp_y];
                        // $display(temp_x);
                    end
                    if(h%2 == 0)begin
                    `ifndef TRUE_MEM
                    inst_array_top.inst_head_array.head_gen_array[h/2].two_heads_inst.inst_head_top_0.inst_head_core_array.head_cores_generate_array[i].core_top_inst.mem_inst.wmem_inst.inst_mem_sp.mem[j+FFN1_WEIGHT_ADDR_BASE]=sram_temp_var;
                    `else    
                    inst_array_top.inst_head_array.head_gen_array[h/2].two_heads_inst.inst_head_top_0.inst_head_core_array.head_cores_generate_array[i].core_top_inst.mem_inst.wmem_inst.true_mem_wrapper.wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j+512]=sram_temp_var[63:0];
                    inst_array_top.inst_head_array.head_gen_array[h/2].two_heads_inst.inst_head_top_0.inst_head_core_array.head_cores_generate_array[i].core_top_inst.mem_inst.wmem_inst.true_mem_wrapper.wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j+512]=sram_temp_var[127:64];
                    `endif
                    end
                    else begin
                    `ifndef TRUE_MEM
                    inst_array_top.inst_head_array.head_gen_array[h/2].two_heads_inst.inst_head_top_1.inst_head_core_array.head_cores_generate_array[i].core_top_inst.mem_inst.wmem_inst.inst_mem_sp.mem[j+FFN1_WEIGHT_ADDR_BASE]=sram_temp_var;
                    `else    
                    inst_array_top.inst_head_array.head_gen_array[h/2].two_heads_inst.inst_head_top_1.inst_head_core_array.head_cores_generate_array[i].core_top_inst.mem_inst.wmem_inst.true_mem_wrapper.wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j+512]=sram_temp_var[63:0];
                    inst_array_top.inst_head_array.head_gen_array[h/2].two_heads_inst.inst_head_top_1.inst_head_core_array.head_cores_generate_array[i].core_top_inst.mem_inst.wmem_inst.true_mem_wrapper.wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j+512]=sram_temp_var[127:64];
                    `endif
                    end
                end
                WMEM_INIT_FLAG = 1;
            end
        end
    end
endgenerate




MODEL_CONFIG                                                    tb_model_cfg;
PMU_CONFIG                                                      tb_pmu_cfg;
RC_CONFIG                                                       tb_rc_cfg;
OP_CONFIG_PKT                                                   tb_op_cfg_pkt;
initial begin
    tb_model_cfg.max_context_length = TB_MAX_CONTEXT_LENGTH;
    tb_model_cfg.qkv_weight_cols_per_core = TB_QKV_WEIGHT_COLS_PER_CORE;
    tb_model_cfg.token_per_core = TB_MAX_CONTEXT_LENGTH/`HEAD_CORE_NUM;
    tb_model_cfg.embd_size = TB_EMBD_SIZE;
    tb_model_cfg.gqa_en = GQA_EN;

    tb_pmu_cfg.pmu_cfg_en = PMU_CFG_EN;
    tb_pmu_cfg.pmu_cfg_bc1 = 1;
    tb_pmu_cfg.pmu_cfg_bc2 = 1;
    tb_pmu_cfg.deepsleep_en = DEEPSLEEP_EN;
    
    tb_rc_cfg.rms_rc_shift = RC_SHIFT;
    tb_rc_cfg.rms_K = RMS_K;
    tb_rc_cfg.attn_rms_dequant_scale_square = shortreal_to_bitsbfloat16(attn_rms_dequant_scale_square);
    tb_rc_cfg.mlp_rms_dequant_scale_square = shortreal_to_bitsbfloat16(mlp_rms_dequant_scale_square);

    tb_rc_cfg.softmax_rc_shift = RC_SHIFT;
    tb_rc_cfg.softmax_input_dequant_scale = shortreal_to_bitsbfloat16(SOFTMAX_DEQUANT_SCALE);
    tb_rc_cfg.softmax_exp_quant_scale = shortreal_to_bitsbfloat16(SOFTMAX_QUANT_SCALE);

    tb_op_cfg_pkt.Q_GEN_CFG.cfg_acc_num = TB_EMBD_SIZE / `MAC_MULT_NUM;
    tb_op_cfg_pkt.Q_GEN_CFG.cfg_quant_scale = Q_GEN_SCALE;
    tb_op_cfg_pkt.Q_GEN_CFG.cfg_quant_bias = 0;
    tb_op_cfg_pkt.Q_GEN_CFG.cfg_quant_shift = Q_GEN_SHIFT;

    tb_op_cfg_pkt.K_GEN_CFG.cfg_acc_num = TB_EMBD_SIZE / `MAC_MULT_NUM;
    tb_op_cfg_pkt.K_GEN_CFG.cfg_quant_scale = K_GEN_SCALE;
    tb_op_cfg_pkt.K_GEN_CFG.cfg_quant_bias = 0;
    tb_op_cfg_pkt.K_GEN_CFG.cfg_quant_shift = K_GEN_SHIFT;

    tb_op_cfg_pkt.V_GEN_CFG.cfg_acc_num = TB_EMBD_SIZE / `MAC_MULT_NUM;
    tb_op_cfg_pkt.V_GEN_CFG.cfg_quant_scale = V_GEN_SCALE;
    tb_op_cfg_pkt.V_GEN_CFG.cfg_quant_bias = 0;
    tb_op_cfg_pkt.V_GEN_CFG.cfg_quant_shift = V_GEN_SHIFT;

    tb_op_cfg_pkt.ATT_QK_CFG.cfg_acc_num = TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM;
    tb_op_cfg_pkt.ATT_QK_CFG.cfg_quant_scale = QKT_SCALE;
    tb_op_cfg_pkt.ATT_QK_CFG.cfg_quant_bias = 0;
    tb_op_cfg_pkt.ATT_QK_CFG.cfg_quant_shift = QKT_SHIFT;

    tb_op_cfg_pkt.ATT_PV_CFG.cfg_acc_num = 1; //这个比较特殊，会被global controller 控制
    tb_op_cfg_pkt.ATT_PV_CFG.cfg_quant_scale = PV_SCALE;
    tb_op_cfg_pkt.ATT_PV_CFG.cfg_quant_bias = 0;
    tb_op_cfg_pkt.ATT_PV_CFG.cfg_quant_shift = PV_SHIFT;

    tb_op_cfg_pkt.PROJ_CFG.cfg_acc_num = TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM;
    tb_op_cfg_pkt.PROJ_CFG.cfg_quant_scale = PROJ_SCALE;
    tb_op_cfg_pkt.PROJ_CFG.cfg_quant_bias = 0;
    tb_op_cfg_pkt.PROJ_CFG.cfg_quant_shift = PROJ_SHIFT;

    tb_op_cfg_pkt.ATTN_RESIDUAL_CFG.cfg_acc_num = ATTN_RESIDUAL_SCALE_A;
    tb_op_cfg_pkt.ATTN_RESIDUAL_CFG.cfg_quant_scale = ATTN_RESIDUAL_SCALE_B;
    tb_op_cfg_pkt.ATTN_RESIDUAL_CFG.cfg_quant_bias = 0;
    tb_op_cfg_pkt.ATTN_RESIDUAL_CFG.cfg_quant_shift = ATTN_RESIDUAL_SHIFT;

    tb_op_cfg_pkt.FFN0_CFG.cfg_acc_num = TB_EMBD_SIZE / `MAC_MULT_NUM;
    tb_op_cfg_pkt.FFN0_CFG.cfg_quant_scale = FFN0_SCALE;
    tb_op_cfg_pkt.FFN0_CFG.cfg_quant_bias = 0;
    tb_op_cfg_pkt.FFN0_CFG.cfg_quant_shift = FFN0_SHIFT;

    tb_op_cfg_pkt.FFN1_CFG.cfg_acc_num = 4 * TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM;
    tb_op_cfg_pkt.FFN1_CFG.cfg_quant_scale = FFN1_SCALE;
    tb_op_cfg_pkt.FFN1_CFG.cfg_quant_bias = 0;
    tb_op_cfg_pkt.FFN1_CFG.cfg_quant_shift = FFN1_SHIFT;

    tb_op_cfg_pkt.MLP_RESIDUAL_CFG.cfg_acc_num = MLP_RESIDUAL_SCALE_A;
    tb_op_cfg_pkt.MLP_RESIDUAL_CFG.cfg_quant_scale = MLP_RESIDUAL_SCALE_B;
    tb_op_cfg_pkt.MLP_RESIDUAL_CFG.cfg_quant_bias = 0;
    tb_op_cfg_pkt.MLP_RESIDUAL_CFG.cfg_quant_shift = MLP_RESIDUAL_SHIFT;
end





logic att_qk_done = 0;
logic att_pv_done = 0;
logic proj_done = 0;
logic attn_residual_done = 0;
logic ffn0_done = 0;
logic ffn1_done = 0;
logic mlp_residual_done = 0;
int iter_cnt = 0;
logic iter_done;
integer user_total_token_cnt;
logic [TB_NUM_USER-1:0][$clog2(TB_TOTAL_CONTEXT_LENGTH+1)-1:0]  usr_total_token_cnt_array;
////////////////////////////////////////
//             MAIN                   //
////////////////////////////////////////

assign inst_array_top.user_id = user_id;



initial begin
    // $fsdbDumpfile("tb_all.fsdb");
    // $fsdbDumpvars(0,"+all",array_top_tb_all); //"+all" enables all  signal dumping including Mem, packed array, etc.
    usr_total_token_cnt_array = 0;
    clk = 0;
    rst_n = 0;
    inst_array_top.clean_kv_cache = 0;
    inst_array_top.clean_kv_cache_user_id = 0;
    inst_array_top.cfg_init_success = 0;
    inst_array_top.new_token = 0;
    inst_array_top.user_first_token = 0;
    user_id = 0;

    repeat(100) @(negedge clk);
    rst_n = 1;

    inst_array_top.inst_array_ctrl.model_cfg_reg = tb_model_cfg;
    inst_array_top.inst_array_ctrl.rc_cfg_reg = tb_rc_cfg;
    inst_array_top.inst_array_ctrl.op_cfg_pkt = tb_op_cfg_pkt;
    inst_array_top.inst_array_ctrl.pmu_cfg_reg= tb_pmu_cfg;


    //power mode
    inst_array_top.power_mode_en_in = 0;

    repeat(10) @(negedge clk);
// clean the KV cache
    for(int i = 0; i < TB_NUM_USER; i++)begin
        inst_array_top.clean_kv_cache = 1;
        inst_array_top.clean_kv_cache_user_id = i;
        @(negedge clk);
        inst_array_top.clean_kv_cache = 0;
        repeat(1000) @(negedge clk);
    end

    repeat(10) @(negedge clk);
    inst_array_top.cfg_init_success = 1;
    @(negedge clk);
    inst_array_top.cfg_init_success = 0;

    for(int m = 0; m < TB_EMBD_SIZE/`MAC_MULT_NUM;m++)begin  
        for(int k = 0;k < `MAC_MULT_NUM;k++)begin
            sram_temp_var[k*8 +: 8] = input_x_array[0][0][m*`MAC_MULT_NUM+k];
        end
        inst_array_top.inst_global_sram.inst_mem_dp.mem[m]= sram_temp_var;
    end

///////////////////////////////////////////////////
//                   NEW TOKEN                  //
//////////////////////////////////////////////////

    inst_array_top.debug_mode_en_in = 1;
    inst_array_top.debug_mode_bits_in = 8'b1;


    inst_array_top.new_token = 1;
    user_id = 0;   
    inst_array_top.user_first_token = 1;
    @(negedge clk);
    inst_array_top.new_token = 0;
    inst_array_top.user_first_token = 0;
    

    while(1)  begin
        @(negedge clk);
        if(inst_array_top.current_token_finish)begin
            break;
        end
    end
    repeat(300) @(negedge clk);


///////////////////////////////////////////////////
//                   NEW TOKEN                  //
//////////////////////////////////////////////////


    inst_array_top.debug_mode_en_in = 1;
    inst_array_top.debug_mode_bits_in = 8'b11;

    inst_array_top.new_token = 1;
    user_id = 0;   
    inst_array_top.user_first_token = 1;
    @(negedge clk);
    inst_array_top.new_token = 0;
    inst_array_top.user_first_token = 0;
    

    while(1)  begin
        @(negedge clk);
        if(inst_array_top.current_token_finish)begin
            break;
        end
    end
    repeat(300) @(negedge clk);


///////////////////////////////////////////////////
//                   NEW TOKEN                  //
//////////////////////////////////////////////////


    inst_array_top.debug_mode_en_in = 1;
    inst_array_top.debug_mode_bits_in = 8'b111;

    inst_array_top.new_token = 1;
    user_id = 0;   
    inst_array_top.user_first_token = 1;
    @(negedge clk);
    inst_array_top.new_token = 0;
    inst_array_top.user_first_token = 0;
    

    while(1)  begin
        @(negedge clk);
        if(inst_array_top.current_token_finish)begin
            break;
        end
    end
    repeat(300) @(negedge clk);



///////////////////////////////////////////////////
//                   NEW TOKEN                  //
//////////////////////////////////////////////////


    inst_array_top.debug_mode_en_in = 1;
    inst_array_top.debug_mode_bits_in = 8'b1101_1000;

    inst_array_top.new_token = 1;
    user_id = 0;   
    inst_array_top.user_first_token = 1;
    @(negedge clk);
    inst_array_top.new_token = 0;
    inst_array_top.user_first_token = 0;
    

    while(1)  begin
        @(negedge clk);
        if(inst_array_top.current_token_finish)begin
            break;
        end
    end
    repeat(300) @(negedge clk);



///////////////////////////////////////////////////
//                   NEW TOKEN                  //
//////////////////////////////////////////////////


    inst_array_top.debug_mode_en_in = 1;
    inst_array_top.debug_mode_bits_in = 8'b1111_1111;

    inst_array_top.new_token = 1;
    user_id = 0;   
    inst_array_top.user_first_token = 1;
    @(negedge clk);
    inst_array_top.new_token = 0;
    inst_array_top.user_first_token = 0;
    

    while(1)  begin
        @(negedge clk);
        if(inst_array_top.current_token_finish)begin
            break;
        end
    end
    repeat(300) @(negedge clk);



///////////////////////////////////////////////////
//                   NEW TOKEN                  //
//////////////////////////////////////////////////


    inst_array_top.debug_mode_en_in = 1;
    inst_array_top.debug_mode_bits_in = 8'b1010_1010;

    inst_array_top.new_token = 1;
    user_id = 0;   
    inst_array_top.user_first_token = 1;
    @(negedge clk);
    inst_array_top.new_token = 0;
    inst_array_top.user_first_token = 0;
    

    while(1)  begin
        @(negedge clk);
        if(inst_array_top.current_token_finish)begin
            break;
        end
    end
    repeat(300) @(negedge clk);



///////////////////////////////////////////////////
//                   NEW TOKEN                  //
//////////////////////////////////////////////////


    inst_array_top.debug_mode_en_in = 1;
    inst_array_top.debug_mode_bits_in = 8'b1111_1100;

    inst_array_top.new_token = 1;
    user_id = 0;   
    inst_array_top.user_first_token = 1;
    @(negedge clk);
    inst_array_top.new_token = 0;
    inst_array_top.user_first_token = 0;
    

    while(1)  begin
        @(negedge clk);
        if(inst_array_top.current_token_finish)begin
            break;
        end
    end
    repeat(300) @(negedge clk);




///////////////////////////////////////////////////
//                   NEW TOKEN                  //
//////////////////////////////////////////////////


    inst_array_top.debug_mode_en_in = 1;
    inst_array_top.debug_mode_bits_in = 8'b1100_0000;

    inst_array_top.new_token = 1;
    user_id = 0;   
    inst_array_top.user_first_token = 1;
    @(negedge clk);
    inst_array_top.new_token = 0;
    inst_array_top.user_first_token = 0;
    

    while(1)  begin
        @(negedge clk);
        if(inst_array_top.current_token_finish)begin
            break;
        end
    end
    repeat(300) @(negedge clk);




///////////////////////////////////////////////////
//                   NEW TOKEN                  //
//////////////////////////////////////////////////


    inst_array_top.debug_mode_en_in = 1;
    inst_array_top.debug_mode_bits_in = 8'b0101_0010;

    inst_array_top.new_token = 1;
    user_id = 0;   
    inst_array_top.user_first_token = 1;
    @(negedge clk);
    inst_array_top.new_token = 0;
    inst_array_top.user_first_token = 0;
    

    while(1)  begin
        @(negedge clk);
        if(inst_array_top.current_token_finish)begin
            break;
        end
    end
    repeat(300) @(negedge clk);



///////////////////////////////////////////////////
//                   NEW TOKEN                  //
//////////////////////////////////////////////////


    inst_array_top.debug_mode_en_in = 1;
    inst_array_top.debug_mode_bits_in = 8'b1111_0010;

    inst_array_top.new_token = 1;
    user_id = 0;   
    inst_array_top.user_first_token = 1;
    @(negedge clk);
    inst_array_top.new_token = 0;
    inst_array_top.user_first_token = 0;
    

    while(1)  begin
        @(negedge clk);
        if(inst_array_top.current_token_finish)begin
            break;
        end
    end
    repeat(300) @(negedge clk);



///////////////////////////////////////////////////
//                   NEW TOKEN                  //
//////////////////////////////////////////////////


    inst_array_top.debug_mode_en_in = 1;
    inst_array_top.debug_mode_bits_in = 8'b0001_0010;

    inst_array_top.new_token = 1;
    user_id = 0;   
    inst_array_top.user_first_token = 1;
    @(negedge clk);
    inst_array_top.new_token = 0;
    inst_array_top.user_first_token = 0;
    

    while(1)  begin
        @(negedge clk);
        if(inst_array_top.current_token_finish)begin
            break;
        end
    end
    repeat(300) @(negedge clk);



///////////////////////////////////////////////////
//                   NEW TOKEN                  //
//////////////////////////////////////////////////


    inst_array_top.debug_mode_en_in = 1;
    inst_array_top.debug_mode_bits_in = 8'b0000_0010;

    inst_array_top.new_token = 1;
    user_id = 0;   
    inst_array_top.user_first_token = 1;
    @(negedge clk);
    inst_array_top.new_token = 0;
    inst_array_top.user_first_token = 0;
    

    while(1)  begin
        @(negedge clk);
        if(inst_array_top.current_token_finish)begin
            break;
        end
    end
    repeat(300) @(negedge clk);



///////////////////////////////////////////////////
//                   NEW TOKEN                  //
//////////////////////////////////////////////////


    inst_array_top.debug_mode_en_in = 1;
    inst_array_top.debug_mode_bits_in = 8'b1101_1010;

    inst_array_top.new_token = 1;
    user_id = 0;   
    inst_array_top.user_first_token = 1;
    @(negedge clk);
    inst_array_top.new_token = 0;
    inst_array_top.user_first_token = 0;
    

    while(1)  begin
        @(negedge clk);
        if(inst_array_top.current_token_finish)begin
            break;
        end
    end
    repeat(300) @(negedge clk);



///////////////////////////////////////////////////
//                   NEW TOKEN                  //
//////////////////////////////////////////////////


    inst_array_top.debug_mode_en_in = 1;
    inst_array_top.debug_mode_bits_in = 8'b1101_0010;

    inst_array_top.new_token = 1;
    user_id = 0;   
    inst_array_top.user_first_token = 1;
    @(negedge clk);
    inst_array_top.new_token = 0;
    inst_array_top.user_first_token = 0;
    

    while(1)  begin
        @(negedge clk);
        if(inst_array_top.current_token_finish)begin
            break;
        end
    end
    repeat(300) @(negedge clk);



///////////////////////////////////////////////////
//                   NEW TOKEN                  //
//////////////////////////////////////////////////


    inst_array_top.debug_mode_en_in = 1;
    inst_array_top.debug_mode_bits_in = 8'b0101_0110;

    inst_array_top.new_token = 1;
    user_id = 0;   
    inst_array_top.user_first_token = 1;
    @(negedge clk);
    inst_array_top.new_token = 0;
    inst_array_top.user_first_token = 0;
    

    while(1)  begin
        @(negedge clk);
        if(inst_array_top.current_token_finish)begin
            break;
        end
    end
    repeat(300) @(negedge clk);



///////////////////////////////////////////////////
//                   NEW TOKEN                  //
//////////////////////////////////////////////////


    inst_array_top.debug_mode_en_in = 1;
    inst_array_top.debug_mode_bits_in = 8'b1101_0010;

    inst_array_top.new_token = 1;
    user_id = 0;   
    inst_array_top.user_first_token = 1;
    @(negedge clk);
    inst_array_top.new_token = 0;
    inst_array_top.user_first_token = 0;
    

    while(1)  begin
        @(negedge clk);
        if(inst_array_top.current_token_finish)begin
            break;
        end
    end
    repeat(300) @(negedge clk);


///////////////////////////////////////////////////
//                   NEW TOKEN                  //
//////////////////////////////////////////////////


    inst_array_top.debug_mode_en_in = 1;
    inst_array_top.debug_mode_bits_in = 8'b0100_0010;

    inst_array_top.new_token = 1;
    user_id = 0;   
    inst_array_top.user_first_token = 1;
    @(negedge clk);
    inst_array_top.new_token = 0;
    inst_array_top.user_first_token = 0;
    

    while(1)  begin
        @(negedge clk);
        if(inst_array_top.current_token_finish)begin
            break;
        end
    end
    repeat(300) @(negedge clk);


///////////////////////////////////////////////////
//                   NEW TOKEN                  //
//////////////////////////////////////////////////


    inst_array_top.debug_mode_en_in = 1;
    inst_array_top.debug_mode_bits_in = 8'b0101_1110;

    inst_array_top.new_token = 1;
    user_id = 0;   
    inst_array_top.user_first_token = 1;
    @(negedge clk);
    inst_array_top.new_token = 0;
    inst_array_top.user_first_token = 0;
    

    while(1)  begin
        @(negedge clk);
        if(inst_array_top.current_token_finish)begin
            break;
        end
    end
    repeat(300) @(negedge clk);

///////////////////////////////////////////////////
//                   NEW TOKEN                  //
//////////////////////////////////////////////////


    inst_array_top.debug_mode_en_in = 1;
    inst_array_top.debug_mode_bits_in = 8'b0101_1001;

    inst_array_top.new_token = 1;
    user_id = 0;   
    inst_array_top.user_first_token = 1;
    @(negedge clk);
    inst_array_top.new_token = 0;
    inst_array_top.user_first_token = 0;
    

    while(1)  begin
        @(negedge clk);
        if(inst_array_top.current_token_finish)begin
            break;
        end
    end
    repeat(300) @(negedge clk);


///////////////////////////////////////////////////
//                   NEW TOKEN                  //
//////////////////////////////////////////////////


    inst_array_top.debug_mode_en_in = 1;
    inst_array_top.debug_mode_bits_in = 8'b0101_1100;

    inst_array_top.new_token = 1;
    user_id = 0;   
    inst_array_top.user_first_token = 1;
    @(negedge clk);
    inst_array_top.new_token = 0;
    inst_array_top.user_first_token = 0;
    

    while(1)  begin
        @(negedge clk);
        if(inst_array_top.current_token_finish)begin
            break;
        end
    end
    repeat(300) @(negedge clk);


    // repeat(100000) @(negedge clk);
    $finish();
end




always begin
    #(1000 * `CLOCK_PERIOD);
    $display("Time: %t", $time());
end

endmodule
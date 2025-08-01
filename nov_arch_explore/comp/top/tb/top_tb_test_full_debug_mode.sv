// `define CHECK_QK
// `define CHECK_PV
// `define CHECK_PROJ
// `define CHECK_FFN0
`define CHECK_FFN1

//define了上面这些会让tb跑的变慢。（已经去掉了多余的一层for循环）
//enable true mem 会非常非常慢

module tb ();
parameter FORCE_TO_BE_RIGHT = 0;

parameter real  CHIP_CLK_FREQ  = 500e6; //500M
parameter real  FPGA_CLK_FREQ = 100e6;
parameter real  QSPI_CLK_FREQ = 50e6;

parameter QKT_MAX_ERR = 6;
parameter PV_MAX_ERR = 6;
parameter PROJ_MAX_ERR = 6;
parameter FFN0_MAX_ERR = 6;
parameter FFN1_MAX_ERR = 6;

parameter GQA_EN = 1;
logic PMU_CFG_EN = 1;
logic DEEPSLEEP_EN = 1;


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
parameter  TB_MAX_CONTEXT_LENGTH = `MAX_CONTEXT_LENGTH*2; //Hardware configured max context length, NEEDS TO BE Divided by HEAD_CORE_NUM
parameter  TB_TOTAL_CONTEXT_LENGTH = (TB_MAX_CONTEXT_LENGTH); //input prompt size + generated prompt size for each user
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

parameter PV_SCALE = 227;
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

/////////////////////////////////////////////////
//           read hex file                     //
/////////////////////////////////////////////////

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

logic [`HEAD_NUM-1:0][`HEAD_CORE_NUM-1:0][`WMEM_DEPTH-1:0][`IDATA_WIDTH * `MAC_MULT_NUM-1:0] core_wmem_array      ;
initial core_wmem_array = 0;
logic [`MAX_EMBD_SIZE/`MAC_MULT_NUM-1:0][`IDATA_WIDTH * `MAC_MULT_NUM-1:0] global_mem_array     ;
initial global_mem_array = 0;



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
                    end
                    // `ifndef TRUE_MEM
                    // top.array_top_inst.inst_head_array.head_gen_array[h].inst_head_top.inst_head_core_array.head_cores_generate_array[i].core_top_inst.mem_inst.wmem_inst.inst_mem_sp.mem[j]=sram_temp_var;
                    // `else    
                    // top.array_top_inst.inst_head_array.head_gen_array[h].inst_head_top.inst_head_core_array.head_cores_generate_array[i].core_top_inst.mem_inst.wmem_inst.true_mem_wrapper.wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j]=sram_temp_var[63:0];
                    // top.array_top_inst.inst_head_array.head_gen_array[h].inst_head_top.inst_head_core_array.head_cores_generate_array[i].core_top_inst.mem_inst.wmem_inst.true_mem_wrapper.wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j]=sram_temp_var[127:64];
                    // `endif
                    core_wmem_array[h][i][j] = sram_temp_var;
                end

                //Initialize K weights array
                for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE / `HEAD_NUM / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++)begin //K_weight depth in one core mem
                    for(int k =0; k <`MAC_MULT_NUM;k++)begin
                        temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE+j/(TB_EMBD_SIZE/`MAC_MULT_NUM); //`EMBD_SIZE是权重矩阵的行数，这里决定第几列,TB_EMBD_SIZE/`MAC_MULT_NUM是存一列权重需要的行数
                        temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM))  * `MAC_MULT_NUM + k;//这里决定那一列中的第几行
                        sram_temp_var[k*8 +: 8] = K_weights_array[h][temp_x][temp_y];
                    end
                    // `ifndef TRUE_MEM
                    // top.array_top_inst.inst_head_array.head_gen_array[h].inst_head_top.inst_head_core_array.head_cores_generate_array[i].core_top_inst.mem_inst.wmem_inst.inst_mem_sp.mem[j+K_WEIGHT_ADDR_BASE]=sram_temp_var;
                    // `else
                    // top.array_top_inst.inst_head_array.head_gen_array[h].inst_head_top.inst_head_core_array.head_cores_generate_array[i].core_top_inst.mem_inst.wmem_inst.true_mem_wrapper.wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j+K_WEIGHT_ADDR_BASE]=sram_temp_var[63:0];
                    // top.array_top_inst.inst_head_array.head_gen_array[h].inst_head_top.inst_head_core_array.head_cores_generate_array[i].core_top_inst.mem_inst.wmem_inst.true_mem_wrapper.wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j+K_WEIGHT_ADDR_BASE]=sram_temp_var[127:64];
                    // `endif
                    core_wmem_array[h][i][j+K_WEIGHT_ADDR_BASE] = sram_temp_var;
                end

                
                //Initialize V weights array
                for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE / `HEAD_NUM / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++)begin //V_weight depth in one core mem
                    for(int k =0; k <`MAC_MULT_NUM;k++)begin
                        temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE+j/(TB_EMBD_SIZE/`MAC_MULT_NUM); //`EMBD_SIZE是权重矩阵的行数，这里决定第几列,TB_EMBD_SIZE/`MAC_MULT_NUM是存一列权重需要的行数
                        temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM))  * `MAC_MULT_NUM + k;//这里决定那一列中的第几行
                        sram_temp_var[k*8 +: 8] = V_weights_array[h][temp_x][temp_y];
                    end
                    // `ifndef TRUE_MEM
                    // top.array_top_inst.inst_head_array.head_gen_array[h].inst_head_top.inst_head_core_array.head_cores_generate_array[i].core_top_inst.mem_inst.wmem_inst.inst_mem_sp.mem[j+V_WEIGHT_ADDR_BASE]=sram_temp_var;
                    // `else
                    // top.array_top_inst.inst_head_array.head_gen_array[h].inst_head_top.inst_head_core_array.head_cores_generate_array[i].core_top_inst.mem_inst.wmem_inst.true_mem_wrapper.wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j+V_WEIGHT_ADDR_BASE]=sram_temp_var[63:0];
                    // top.array_top_inst.inst_head_array.head_gen_array[h].inst_head_top.inst_head_core_array.head_cores_generate_array[i].core_top_inst.mem_inst.wmem_inst.true_mem_wrapper.wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j+V_WEIGHT_ADDR_BASE]=sram_temp_var[127:64];
                    // `endif
                    core_wmem_array[h][i][j+V_WEIGHT_ADDR_BASE] = sram_temp_var;
                end

                for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE / `HEAD_NUM / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++)begin //Proj_weight depth in one core mem
                    for(int k =0; k <`MAC_MULT_NUM;k++)begin
                        temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM)+j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM); //TB_EMBD_SIZE/`HEAD_NUM是权重矩阵的行数，这里决定第几列,TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM是存一列权重需要的行数
                        temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM / `MAC_MULT_NUM))  * `MAC_MULT_NUM + k;//这里决定那一列中的第几行
                        sram_temp_var[k*8 +: 8] = PROJ_weights_array[h][temp_x][temp_y];
                    end
                    // `ifndef TRUE_MEM
                    // top.array_top_inst.inst_head_array.head_gen_array[h].inst_head_top.inst_head_core_array.head_cores_generate_array[i].core_top_inst.mem_inst.wmem_inst.inst_mem_sp.mem[j+PROJ_WEIGHT_ADDR_BASE]=sram_temp_var;
                    // `else
                    // top.array_top_inst.inst_head_array.head_gen_array[h].inst_head_top.inst_head_core_array.head_cores_generate_array[i].core_top_inst.mem_inst.wmem_inst.true_mem_wrapper.wmem_512_attn_r.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j+PROJ_WEIGHT_ADDR_BASE]=sram_temp_var[63:0];
                    // top.array_top_inst.inst_head_array.head_gen_array[h].inst_head_top.inst_head_core_array.head_cores_generate_array[i].core_top_inst.mem_inst.wmem_inst.true_mem_wrapper.wmem_512_attn_l.wmem_512_bmod.wmem_512_array.DATA_ARRAY[j+PROJ_WEIGHT_ADDR_BASE]=sram_temp_var[127:64];
                    // `endif
                    core_wmem_array[h][i][j+PROJ_WEIGHT_ADDR_BASE] = sram_temp_var;
                end

                for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE / `HEAD_NUM / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++)begin //ffn0_weight depth in one core mem
                    for(int k =0; k <`MAC_MULT_NUM;k++)begin
                        temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4)+j/(TB_EMBD_SIZE/`MAC_MULT_NUM); //`EMBD_SIZE是权重矩阵的行数，这里决定第几列,TB_EMBD_SIZE/`MAC_MULT_NUM是存一列权重需要的行数
                        temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM))  * `MAC_MULT_NUM + k;//这里决定那一列中的第几行
                        sram_temp_var[k*8 +: 8] = FFN0_weights_array[h][temp_x][temp_y];
                    end
                    // `ifndef TRUE_MEM
                    // top.array_top_inst.inst_head_array.head_gen_array[h].inst_head_top.inst_head_core_array.head_cores_generate_array[i].core_top_inst.mem_inst.wmem_inst.inst_mem_sp.mem[j+FFN0_WEIGHT_ADDR_BASE]=sram_temp_var;
                    // `else
                    // top.array_top_inst.inst_head_array.head_gen_array[h].inst_head_top.inst_head_core_array.head_cores_generate_array[i].core_top_inst.mem_inst.wmem_inst.true_mem_wrapper.wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j]=sram_temp_var[63:0];
                    // top.array_top_inst.inst_head_array.head_gen_array[h].inst_head_top.inst_head_core_array.head_cores_generate_array[i].core_top_inst.mem_inst.wmem_inst.true_mem_wrapper.wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j]=sram_temp_var[127:64];
                    // `endif
                    core_wmem_array[h][i][j+FFN0_WEIGHT_ADDR_BASE] = sram_temp_var;
                end

                for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE / `HEAD_NUM / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++)begin //ffn1_weight depth in one core mem
                    for(int k =0; k <`MAC_MULT_NUM;k++)begin
                        temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM)+j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM); //4*TB_EMBD_SIZE/`HEAD_NUM是权重矩阵的行数，这里决定第几列,4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM是存一列权重需要的行数
                        temp_x = (j % (4*TB_EMBD_SIZE/`HEAD_NUM / `MAC_MULT_NUM))  * `MAC_MULT_NUM + k;//这里决定那一列中的第几行
                        sram_temp_var[k*8 +: 8] = FFN1_weights_array[h][temp_x][temp_y];
                    end
                    // `ifndef TRUE_MEM
                    // top.array_top_inst.inst_head_array.head_gen_array[h].inst_head_top.inst_head_core_array.head_cores_generate_array[i].core_top_inst.mem_inst.wmem_inst.inst_mem_sp.mem[j+FFN1_WEIGHT_ADDR_BASE]=sram_temp_var;
                    // `else
                    // top.array_top_inst.inst_head_array.head_gen_array[h].inst_head_top.inst_head_core_array.head_cores_generate_array[i].core_top_inst.mem_inst.wmem_inst.true_mem_wrapper.wmem_1024_ffn_r.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j+512]=sram_temp_var[63:0];
                    // top.array_top_inst.inst_head_array.head_gen_array[h].inst_head_top.inst_head_core_array.head_cores_generate_array[i].core_top_inst.mem_inst.wmem_inst.true_mem_wrapper.wmem_1024_ffn_l.wmem_1024_bmod.wmem_1024_array.DATA_ARRAY[j+512]=sram_temp_var[127:64];
                    // `endif
                    core_wmem_array[h][i][j+FFN1_WEIGHT_ADDR_BASE] = sram_temp_var;
                end
                WMEM_INIT_FLAG = 1;
            end
        end
    end
endgenerate








/////////////////////////////////////////////////
//              main variables                 //
/////////////////////////////////////////////////


logic       [`USER_ID_WIDTH-1:0]                     user_id;// be in same phase
logic                                                new_token;
logic                                                user_first_token;
logic                chip_clk;
logic                fpga_clk;
logic                asyn_rst;       //rst!, not rst_n
// SPI Domain
logic                spi_clk;       
logic                spi_csn;        // SPI Active Low
logic                spi_mosi;       // Host -> SPI
logic                spi_miso;       // Host <- SPI


logic                qspi_clk;       //qspi clk
logic [15:0]         qspi_mosi;       // Host -> QSPI -> TPU
logic                qspi_mosi_valid;
logic [15:0]         qspi_miso;       // Host <- QSPI <- TPU
logic                qspi_miso_valid;

//FPGA Domain
logic                                                              spi_start; //looks should be in same phase with spi_tx_data
logic                                                              spi_complete;
logic [`INTERFACE_DATA_WIDTH + `INTERFACE_ADDR_WIDTH+2 + 1-1:0]    spi_tx_data;
logic [`INTERFACE_DATA_WIDTH-1:0]                                  spi_rx_data;
logic                                                              spi_rx_valid;

logic                current_token_finish_flag;
logic                current_token_finish_work;

logic                qgen_state_work;
logic                qgen_state_end;

logic                kgen_state_work;
logic                kgen_state_end;

logic                vgen_state_work;
logic                vgen_state_end;

logic                att_qk_state_work;
logic                att_qk_state_end;

logic                att_pv_state_work;
logic                att_pv_state_end;

logic                proj_state_work;
logic                proj_state_end;

logic                ffn0_state_work;
logic                ffn0_state_end;

logic                ffn1_state_work;
logic                ffn1_state_end;


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
logic [7:0] debug_mode_bits; 
logic [TB_NUM_USER-1:0][$clog2(TB_TOTAL_CONTEXT_LENGTH+1)-1:0]  usr_total_token_cnt_array;

logic [`INTERFACE_ADDR_WIDTH-1:0] addr;
logic [`INTERFACE_DATA_WIDTH-1:0] wdata;
logic [`INTERFACE_DATA_WIDTH-1:0] rdata;
logic [255:0][`INTERFACE_DATA_WIDTH-1:0] wdata_array;
logic [255:0][`INTERFACE_DATA_WIDTH-1:0] rdata_array;
logic [255:0][`INTERFACE_DATA_WIDTH-1:0] tpu_state_rdata_array;
logic [7:0] burst_cnt;

//时间刻度和精度都是1ps
initial begin
    chip_clk = 0;
    #100 fpga_clk = 0; //给不同的相位
    #450 qspi_clk = 0;
end
//时间刻度和精度都是1ps
always begin
    #(1e12/CHIP_CLK_FREQ/2);
    chip_clk = ~chip_clk;
end

always begin
    #(1e12/FPGA_CLK_FREQ/2);
    fpga_clk = ~fpga_clk;
end

always begin
    #(1e12/QSPI_CLK_FREQ/2);
    qspi_clk = ~qspi_clk;
end

host_spi #(
    .DW(`INTERFACE_DATA_WIDTH + `INTERFACE_ADDR_WIDTH+2 + 1),
    .TX(22), //useless
    .RX(`INTERFACE_DATA_WIDTH)
)host_spi (
    // Global Signals
    .clk(fpga_clk),
    .rst(asyn_rst),

    // Host Interface
    .spi_start(spi_start),
    .spi_complete(spi_complete),
    .spi_tx_data(spi_tx_data),
    .spi_rx_data(spi_rx_data),
    .spi_rx_valid(spi_rx_valid),

    // SPI Interface
    .spi_sck(spi_clk),
    .spi_csn(spi_csn),
    .spi_mosi(spi_mosi),
    .spi_miso(spi_miso)
);

/////////////////////////////////////////////////
//              instance of top                //
/////////////////////////////////////////////////

top top(
    .chip_clk(chip_clk),
    .asyn_rst(asyn_rst), //rst!, not rst_n

    .spi_clk(spi_clk),        
    .spi_csn(spi_csn),        // SPI Active Low
    .spi_mosi(spi_mosi),       // Host -> SPI
    .spi_miso(spi_miso),       // Host <- SPI

    .qspi_clk(qspi_clk),        //qspi clk
    .qspi_mosi(qspi_mosi),       // Host -> QSPI -> TPU
    .qspi_mosi_valid(qspi_mosi_valid),
    .qspi_miso(qspi_miso),       // Host <- QSPI <- TPU
    .qspi_miso_valid(qspi_miso_valid),


    .current_token_finish_flag(current_token_finish_flag),
    .current_token_finish_work(current_token_finish_work),

    .qgen_state_work(qgen_state_work),
    .qgen_state_end(qgen_state_end),

    .kgen_state_work(kgen_state_work),
    .kgen_state_end(kgen_state_end),

    .vgen_state_work(vgen_state_work),
    .vgen_state_end(vgen_state_end),

    .att_qk_state_work(att_qk_state_work),
    .att_qk_state_end(att_qk_state_end),

    .att_pv_state_work(att_pv_state_work),
    .att_pv_state_end(att_pv_state_end),

    .proj_state_work(proj_state_work),
    .proj_state_end(proj_state_end),

    .ffn0_state_work(ffn0_state_work),
    .ffn0_state_end(ffn0_state_end),

    .ffn1_state_work(ffn1_state_work),
    .ffn1_state_end(ffn1_state_end)
);

task FPGA_SPI_WR(
    input  logic [`INTERFACE_ADDR_WIDTH-1:0] addr,
    input  logic [`INTERFACE_DATA_WIDTH-1:0] wdata
);
    @(negedge fpga_clk);
    spi_start = 1;
    spi_tx_data = {2'b10, addr, 1'b0, wdata};
    @(negedge fpga_clk);
    spi_start = 0;
    while (1)begin
        @(negedge fpga_clk);
        if(spi_complete)
            break;
    end
endtask

task FPGA_SPI_RD(
    input  logic [`INTERFACE_ADDR_WIDTH-1:0] addr,
    output logic [`INTERFACE_DATA_WIDTH-1:0] rdata
);
    @(negedge fpga_clk);
    spi_start = 1;
    spi_tx_data = {2'b01, addr, 1'b0, 16'b0};
    @(negedge fpga_clk);
    spi_start = 0;
    while (1)begin
        @(negedge fpga_clk);
        if(spi_rx_valid)begin
            rdata = spi_rx_data;
            break;
        end
    end
endtask


task FPGA_QSPI_WR(
    input  logic [`INTERFACE_ADDR_WIDTH-1:0]         addr,
    input  logic [7:0]                               burst_cnt,//等于0传递1个数据，等于255传递256个数据
    input  logic [255:0][`INTERFACE_DATA_WIDTH-1:0]  wdata_array
);
    @(posedge qspi_clk);
    qspi_mosi_valid = 1;
    qspi_mosi = {addr[5:0], 2'b10, burst_cnt};
    @(posedge qspi_clk);
    qspi_mosi_valid = 1;
    qspi_mosi = addr[21:6];
    for(int i = 0; i < burst_cnt + 1; i++)begin
        @(posedge qspi_clk);
        qspi_mosi_valid = 1;
        qspi_mosi = wdata_array[i];
    end
    @(posedge qspi_clk);
    qspi_mosi_valid = 0;
    @(posedge qspi_clk);
endtask


task FPGA_QSPI_RD(
    input  logic [`INTERFACE_ADDR_WIDTH-1:0]         addr,
    input  logic [7:0]                               burst_cnt,//等于0传递1个数据，等于63传递64个数据
    output logic [255:0][`INTERFACE_DATA_WIDTH-1:0]  rdata_array
);
    automatic int temp = 0; //不能写成int temp = 0 temp是一个静态变量！！！，所以不会重复定义，导致数据冲突
    @(posedge qspi_clk);
    qspi_mosi_valid = 1;
    qspi_mosi = {addr[5:0], 2'b01, burst_cnt};
    @(posedge qspi_clk);
    qspi_mosi_valid = 1;
    qspi_mosi = addr[21:6];
    @(posedge qspi_clk);
    qspi_mosi_valid = 0;
    while(1)begin
        @(negedge qspi_clk);
        if(qspi_miso_valid)begin
            rdata_array[temp] = qspi_miso;
            temp++;
            if(temp == burst_cnt+1)
                break;
        end
    end
    @(negedge qspi_clk);
endtask

////////////////////////////////////////
//             MAIN                   //
////////////////////////////////////////
initial begin
    usr_total_token_cnt_array = 0;
    asyn_rst = 1;
    spi_start = 0;
    spi_tx_data = 0;
    qspi_mosi_valid = 0;
    qspi_mosi = 0;

    repeat(10000) @(negedge chip_clk);
    asyn_rst = 0;

    repeat(100) @(negedge chip_clk);
    //////////////////////////
    //     Upload WMEM      //
    //////////////////////////
    $display("Upload wmem");
    for(int h = 0; h < `HEAD_NUM; h++)begin
        for(int c = 0; c < `HEAD_CORE_NUM; c++)begin
            for(int i = 0; i < (4096 * 3) / 256; i++)begin  //8-32KB (24KB)
                addr = h * 'h40000 + c * 'h4000 + i * 256 + 4096;
                burst_cnt = 255;
                for(int j = 0; j < 256; j++)begin
                    wdata_array[j] = core_wmem_array[h][c][i*256/(`MAC_MULT_NUM/2) + j/(`MAC_MULT_NUM/2)][(j%(`MAC_MULT_NUM/2))*16 +: 16];
                end
                FPGA_QSPI_WR(addr, burst_cnt, wdata_array);
            end
            repeat(30) @(negedge chip_clk);

            $display("HEAD[%0d]CORE[%0d] WMEM finish.",h,c);
        end
    end


    //////////////////////////
    //     Clean KV Cache   //
    //////////////////////////
    $display("Clean KV CACHE");
    for(int i = 0; i < TB_NUM_USER; i++)begin
        addr = `INSTRUCTION_REGISTERS_BASE_ADDR + 1;
        burst_cnt = 0;
        user_id = i;
        wdata_array[0] = {13'b0, user_id, 1'b1};
        FPGA_QSPI_WR(addr, burst_cnt, wdata_array);
        repeat(1000) @(negedge chip_clk);
    end

    ///////////////////////////////////////////
    //     Upload control registers values   //
    ///////////////////////////////////////////
    $display("Upload control register value");
    burst_cnt = 63;
    addr = `CONTROL_REGISTERS_BASE_ADDR + 0;

    wdata_array[0] = TB_MAX_CONTEXT_LENGTH;
    wdata_array[1] = TB_QKV_WEIGHT_COLS_PER_CORE;
    wdata_array[2] = TB_MAX_CONTEXT_LENGTH/`HEAD_CORE_NUM;
    wdata_array[3] = TB_EMBD_SIZE;
    wdata_array[4] = GQA_EN;

    wdata_array[5] = 1;
    wdata_array[6] = 1;
    wdata_array[7] = {14'b0, DEEPSLEEP_EN, PMU_CFG_EN};   

    wdata_array[8] = RC_SHIFT;
    wdata_array[9] = RMS_K;
    wdata_array[10] = shortreal_to_bitsbfloat16(attn_rms_dequant_scale_square);
    wdata_array[11] = shortreal_to_bitsbfloat16(mlp_rms_dequant_scale_square);

    wdata_array[12] = RC_SHIFT;
    wdata_array[13] = shortreal_to_bitsbfloat16(SOFTMAX_DEQUANT_SCALE);
    wdata_array[14] = shortreal_to_bitsbfloat16(SOFTMAX_QUANT_SCALE);

    wdata_array[15] = TB_EMBD_SIZE / `MAC_MULT_NUM;
    wdata_array[16] = Q_GEN_SCALE;
    wdata_array[17] = 0;
    wdata_array[18] = Q_GEN_SHIFT;

    wdata_array[19] = TB_EMBD_SIZE / `MAC_MULT_NUM;
    wdata_array[20] = K_GEN_SCALE;
    wdata_array[21] = 0;
    wdata_array[22] = K_GEN_SHIFT;   

    wdata_array[23] = TB_EMBD_SIZE / `MAC_MULT_NUM;
    wdata_array[24] = V_GEN_SCALE;
    wdata_array[25] = 0;
    wdata_array[26] = V_GEN_SHIFT;      

    wdata_array[27] = TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM;
    wdata_array[28] = QKT_SCALE;
    wdata_array[29] = 0;
    wdata_array[30] = QKT_SHIFT;   

    wdata_array[31] = 1;//这个比较特殊，会被global controller 控制
    wdata_array[32] = PV_SCALE;
    wdata_array[33] = 0;
    wdata_array[34] = PV_SHIFT;   

    wdata_array[35] = TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM;
    wdata_array[36] = PROJ_SCALE;
    wdata_array[37] = 0;
    wdata_array[38] = PROJ_SHIFT;  

    wdata_array[39] = TB_EMBD_SIZE / `MAC_MULT_NUM;
    wdata_array[40] = FFN0_SCALE;
    wdata_array[41] = 0;
    wdata_array[42] = FFN0_SHIFT;     

    wdata_array[43] = 4 * TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM;
    wdata_array[44] = FFN1_SCALE;
    wdata_array[45] = 0;
    wdata_array[46] = FFN1_SHIFT;     

    wdata_array[47] = ATTN_RESIDUAL_SCALE_A;
    wdata_array[48] = ATTN_RESIDUAL_SCALE_B;
    wdata_array[49] = 0;
    wdata_array[50] = ATTN_RESIDUAL_SHIFT;     

    wdata_array[51] = MLP_RESIDUAL_SCALE_A;
    wdata_array[52] = MLP_RESIDUAL_SCALE_B;
    wdata_array[53] = 0;
    wdata_array[54] = MLP_RESIDUAL_SHIFT;   
    FPGA_QSPI_WR(addr, burst_cnt, wdata_array);

    ///////////////////////////////////////////
    //         INIT CONFIGURATION            //
    ///////////////////////////////////////////
    burst_cnt = 0;
    addr = `INSTRUCTION_REGISTERS_BASE_ADDR + 2;
    wdata_array[0] = 1;
    FPGA_QSPI_WR(addr, burst_cnt, wdata_array);


    ///////////////////////////////////////////
    //           Start Cmputation            //
    ///////////////////////////////////////////

             
    for(int m = 0; m < TB_EMBD_SIZE/`MAC_MULT_NUM;m++)begin  //EMBD_SIZE是`MAC_MULT_NUM的倍数
        for(int k = 0;k < `MAC_MULT_NUM;k++)begin
            sram_temp_var[k*8 +: 8] = input_x_array[0][0][m*`MAC_MULT_NUM+k];
        end
        global_mem_array[m] = sram_temp_var;
    end
    addr = `GLOBAL_MEM_BASE_ADDR;
    burst_cnt = TB_EMBD_SIZE/2-1;
    for(int jj = 0; jj < 256; jj++)begin
        wdata_array[jj] = global_mem_array[jj/(`MAC_MULT_NUM/2)][(jj%(`MAC_MULT_NUM/2))*16 +: 16];
    end
    FPGA_QSPI_WR(addr, burst_cnt, wdata_array);

    debug_mode_bits = 8'b01010101;
    addr = `INSTRUCTION_REGISTERS_BASE_ADDR + 3;
    burst_cnt = 0;
    wdata_array[0] = {debug_mode_bits, 1'b1, 1'b0};
    FPGA_QSPI_WR(addr, burst_cnt, wdata_array);
    

    repeat(10) @(negedge chip_clk);
    new_token = 1;
    user_id = 2;   
    user_first_token = 1;
    burst_cnt = 0;
    addr = `INSTRUCTION_REGISTERS_BASE_ADDR + 0;
    wdata_array[0] = {12'b0, user_id, user_first_token, new_token};
    FPGA_QSPI_WR(addr, burst_cnt, wdata_array);




    repeat(100000) @(negedge chip_clk);
    $finish();
end



always begin
    repeat(10000) @(negedge chip_clk);
    $display("Time: %t", $time());
end

endmodule
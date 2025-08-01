`define CHECK_PV
`define CHECK_PROJ
`define CHECK_FFN0
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
logic PMU_CFG_EN = 0;
logic DEEPSLEEP_EN = 0;


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
parameter  TB_TOTAL_CONTEXT_LENGTH = (TB_MAX_CONTEXT_LENGTH/64); //input prompt size + generated prompt size for each user
parameter  TB_NUM_USER = 1;
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

task QKV_gen_soft(
    input logic [`IDATA_WIDTH-1:0] input_x_array [TB_NUM_USER-1:0][TB_TOTAL_CONTEXT_LENGTH-1:0][TB_EMBD_SIZE-1:0],
    input logic [`IDATA_WIDTH-1:0] weights_array [TB_EMBD_SIZE-1:0][TB_EMBD_SIZE/`HEAD_NUM-1:0],
    input integer scale, //symmetrical quantization, no bias
    input integer shift,
    output logic [`IDATA_WIDTH-1:0] o_array [TB_NUM_USER-1:0][TB_TOTAL_CONTEXT_LENGTH-1:0][TB_EMBD_SIZE/`HEAD_NUM-1:0]
);
longint   temp_sum;
integer   round;
shortreal real_data_array [TB_EMBD_SIZE-1:0];
shortreal square_sum;
shortreal rms;
shortreal one_over_krms;

for(int u = 0; u < TB_NUM_USER; u++) begin
    for(int i = 0; i < TB_TOTAL_CONTEXT_LENGTH; i++)begin
        square_sum = 0;
        rms = 0;
        for(int j = 0; j < TB_EMBD_SIZE; j++)begin
            real_data_array[j] = $itor($signed(input_x_array[u][i][j]));
        end
        for(int j =0;j < RMS_K; j++)begin
            square_sum = square_sum + real_data_array[j] * real_data_array[j];
        end
        square_sum = square_sum * attn_rms_dequant_scale_square;
        rms = $sqrt(square_sum/RMS_K * 1.0);
        one_over_krms = 1/rms;
        for(int j = 0; j < TB_EMBD_SIZE/`HEAD_NUM; j++)begin
            temp_sum = 0;
            round = 0;
            for(int k = 0; k < TB_EMBD_SIZE; k++)begin
                temp_sum = temp_sum + $signed(input_x_array[u][i][k]) * $signed(weights_array[k][j]);
            end
            temp_sum = $rtoi($itor(temp_sum) * one_over_krms);
            temp_sum = temp_sum * scale;
            round = temp_sum[shift-1];
            temp_sum = temp_sum >>> shift;
            temp_sum = temp_sum + round;
            if(temp_sum > 127)
                temp_sum = 127;
            if(temp_sum < -128)
                temp_sum = -128;
            o_array[u][i][j] = temp_sum;
        end
    end
end
endtask

task ATT_QK_soft(
    input logic [`IDATA_WIDTH-1:0] input_q_array [TB_NUM_USER-1:0][TB_TOTAL_CONTEXT_LENGTH-1:0][TB_EMBD_SIZE/`HEAD_NUM-1:0],
    input logic [`IDATA_WIDTH-1:0] input_k_array [TB_NUM_USER-1:0][TB_TOTAL_CONTEXT_LENGTH-1:0][TB_EMBD_SIZE/`HEAD_NUM-1:0],
    input integer scale, //symmetrical quantization, no bias
    input integer shift,
    output logic [`IDATA_WIDTH-1:0] o_array [TB_NUM_USER-1:0][TB_TOTAL_CONTEXT_LENGTH-1:0][TB_TOTAL_CONTEXT_LENGTH-1:0]
);
integer temp_sum;
integer round;
// $display();

for(int u = 0; u < TB_NUM_USER; u++) begin
    for(int i = 0; i < TB_TOTAL_CONTEXT_LENGTH; i++)begin
        for(int j = 0; j < TB_TOTAL_CONTEXT_LENGTH; j++)begin
            temp_sum = 0;
            round = 0;
            for(int k = 0; k < TB_EMBD_SIZE/`HEAD_NUM; k++)begin
                temp_sum = temp_sum + $signed(input_q_array[u][i][k]) * $signed(input_k_array[u][j][k]);
            end
            // $display(temp_sum);
            temp_sum = temp_sum * scale;
            round = temp_sum[shift-1];
            temp_sum = temp_sum >>> shift;
            temp_sum = temp_sum + round;
            if(temp_sum > 127)
                temp_sum = 127;
            if(temp_sum < -128)
                temp_sum = -128;
            o_array[u][i][j] = temp_sum;
        end
    end
end
endtask

task ATT_PV_soft(                   //这里p_array是QKT array
    input logic [`IDATA_WIDTH-1:0] input_p_array [TB_NUM_USER-1:0][TB_TOTAL_CONTEXT_LENGTH-1:0][TB_TOTAL_CONTEXT_LENGTH-1:0],
    input logic [`IDATA_WIDTH-1:0] input_v_array [TB_NUM_USER-1:0][TB_TOTAL_CONTEXT_LENGTH-1:0][TB_EMBD_SIZE/`HEAD_NUM-1:0],
    input integer scale, //symmetrical quantization, no bias
    input integer shift,
    output logic [`IDATA_WIDTH-1:0] o_array [TB_NUM_USER-1:0][TB_TOTAL_CONTEXT_LENGTH-1:0][TB_EMBD_SIZE/`HEAD_NUM-1:0]
);
shortreal exp_max_sum;
int qkt_max;
shortreal softmax_qk;
integer temp_sum;
integer round;
real test_sum;
// $display();

for(int u = 0; u < TB_NUM_USER; u++) begin
    for(int i = 0; i < TB_TOTAL_CONTEXT_LENGTH; i++)begin

        qkt_max = -128;
        exp_max_sum = 0;
        for(int k = 0; k < TB_TOTAL_CONTEXT_LENGTH; k++)begin //find the max
            if(i < TB_MAX_CONTEXT_LENGTH)begin //KV CACHE is not full
                if(k <= i) begin   
                    if(qkt_max <= $signed(input_p_array[u][i][k]))
                        qkt_max = $signed(input_p_array[u][i][k]);
                end
            end
            else begin //KV CACHE is full
                if(k > i - TB_MAX_CONTEXT_LENGTH && k <= i)begin
                    if(qkt_max <= $signed(input_p_array[u][i][k]))
                        qkt_max = $signed(input_p_array[u][i][k]);
                end    
            end
        end
        // $display("USER: %0d  TOKEN_CNT: %0d, TIME: %t, max: %d",u,i,$time, qkt_max);

        //求exp_max_sum
        for(int k = 0; k < TB_TOTAL_CONTEXT_LENGTH; k++)begin
            if(i < TB_MAX_CONTEXT_LENGTH)begin //KV CACHE is not full
                if(k <= i) begin   
                    exp_max_sum = exp_max_sum + $exp($itor($signed(input_p_array[u][i][k]) - qkt_max) * SOFTMAX_DEQUANT_SCALE);
                end
            end
            else begin //KV CACHE 满了
                if(k > i - TB_MAX_CONTEXT_LENGTH && k <= i)begin
                    exp_max_sum = exp_max_sum + $exp($itor($signed(input_p_array[u][i][k]) - qkt_max) * SOFTMAX_DEQUANT_SCALE);
                end    
            end
        end

        // $display("USER: %0d  TOKEN_CNT: %0d, TIME: %t, exp sum: %h",u,i,$time, shortreal_to_bitsbfloat16(exp_max_sum));



        for(int j = 0; j < TB_EMBD_SIZE/`HEAD_NUM; j++)begin
            temp_sum = 0;
            round = 0;
            test_sum = 0;
            for(int k = 0; k < TB_TOTAL_CONTEXT_LENGTH; k++)begin
                if(i < TB_MAX_CONTEXT_LENGTH)begin //KV CACHE is not full
                    if(k <= i) begin
                        softmax_qk = $exp($itor($signed(input_p_array[u][i][k]) - qkt_max) * SOFTMAX_DEQUANT_SCALE) * SOFTMAX_QUANT_SCALE;
                        // if(i==1 && j==0)
                        //     $display(softmax_qk);
                        softmax_qk = zero_round(softmax_qk);
                        if(softmax_qk > 127)
                            softmax_qk = 127;
                        else if(softmax_qk < -128)
                            softmax_qk = -128;
                        softmax_qk = softmax_qk / exp_max_sum;
                        test_sum = test_sum + softmax_qk;
                        // if(i==89 && j==0)  begin
                        //     $display("HHHHH");
                        //     $display(softmax_qk);
                        // end

                        temp_sum = temp_sum + $rtoi(softmax_qk * $itor($signed(input_v_array[u][k][j])));
                        // if(j==0)
                        // $write("+");
                    end
                    else begin
                        temp_sum = temp_sum;
                    end
                end
                else begin //KV CACHE is full
                    if(k > i - TB_MAX_CONTEXT_LENGTH && k <= i)begin
                        softmax_qk = $exp($itor($signed(input_p_array[u][i][k]) - qkt_max) * SOFTMAX_DEQUANT_SCALE) * SOFTMAX_QUANT_SCALE;
                        softmax_qk = zero_round(softmax_qk);
                        if(softmax_qk > 127)
                            softmax_qk = 127;
                        else if(softmax_qk < -128)
                            softmax_qk = -128;
                        // $display(softmax_qk);
                        softmax_qk = softmax_qk / exp_max_sum;
                        temp_sum = temp_sum + $rtoi(softmax_qk * $itor($signed(input_v_array[u][k][j])));
                    end    
                    else begin
                        temp_sum = temp_sum;
                    end
                end
            end
            // $display("HHHHHH, %f",test_sum);
            // if(j==0)begin
            //     $display();
            //     $display("%d",temp_sum);
            // end
            temp_sum = temp_sum * scale;
            round = temp_sum[shift-1];
            temp_sum = temp_sum >>> shift;
            temp_sum = temp_sum + round;
            if(temp_sum > 127)
                temp_sum = 127;
            if(temp_sum < -128)
                temp_sum = -128;
            o_array[u][i][j] = temp_sum;
        end
    end
end
endtask

task PROJ_gen_soft(
    input logic [`IDATA_WIDTH-1:0] ATT_PV_array      [`HEAD_NUM-1:0][TB_NUM_USER-1:0][TB_TOTAL_CONTEXT_LENGTH-1:0][TB_EMBD_SIZE/`HEAD_NUM-1:0],
    input logic [`IDATA_WIDTH-1:0] PROJ_weights_array   [`HEAD_NUM-1:0][TB_EMBD_SIZE/`HEAD_NUM-1:0][TB_EMBD_SIZE-1:0],
    input integer scale, //symmetrical quantization, no bias
    input integer shift,
    output logic [`IDATA_WIDTH-1:0] ATT_PROJ_array     [TB_NUM_USER-1:0][TB_TOTAL_CONTEXT_LENGTH-1:0][TB_EMBD_SIZE-1:0]
);
longint   temp_sum;
integer   round;
int h;
int l;

for(int u = 0; u < TB_NUM_USER; u++) begin
    for(int i = 0; i < TB_TOTAL_CONTEXT_LENGTH; i++)begin
        for(int j = 0; j < TB_EMBD_SIZE; j++)begin
            temp_sum = 0;
            round = 0;
            for(int k = 0; k < TB_EMBD_SIZE; k++)begin
                h = k / (TB_EMBD_SIZE/`HEAD_NUM);
                l = k % (TB_EMBD_SIZE/`HEAD_NUM);
                temp_sum = temp_sum + $signed(ATT_PV_array[h][u][i][l]) * $signed(PROJ_weights_array[h][l][j]);
            end
            temp_sum = temp_sum * scale;
            round = temp_sum[shift-1];
            temp_sum = temp_sum >>> shift;
            temp_sum = temp_sum + round;
            if(temp_sum > 127)
                temp_sum = 127;
            if(temp_sum < -128)
                temp_sum = -128;
            ATT_PROJ_array[u][i][j] = temp_sum;
        end
    end
end
endtask

task RESIDUAL_gen_soft(
    input logic [`IDATA_WIDTH-1:0] input_x_array                    [TB_NUM_USER-1:0][TB_TOTAL_CONTEXT_LENGTH-1:0][TB_EMBD_SIZE-1:0],
    input logic [`IDATA_WIDTH-1:0] ATT_PROJ_array                   [TB_NUM_USER-1:0][TB_TOTAL_CONTEXT_LENGTH-1:0][TB_EMBD_SIZE-1:0],
    input integer scale_A, //symmetrical quantization, no bias
    input integer scale_B,
    input integer shift,
    output logic [`IDATA_WIDTH-1:0] ATT_RESIDUAL_array               [TB_NUM_USER-1:0][TB_TOTAL_CONTEXT_LENGTH-1:0][TB_EMBD_SIZE-1:0]
);
longint   temp;
integer   round;
int h;
int l;

for(int u = 0; u < TB_NUM_USER; u++) begin
    for(int i = 0; i < TB_TOTAL_CONTEXT_LENGTH; i++)begin
        for(int j = 0; j < TB_EMBD_SIZE; j++)begin
            temp = $signed(input_x_array[u][i][j]) * scale_A + $signed(ATT_PROJ_array[u][i][j]) * scale_B;
            round = temp[shift-1];
            temp = temp >>> shift;
            temp = temp + round;
            if(temp > 127)
                temp = 127;
            if(temp < -128)
                temp = -128;
            ATT_RESIDUAL_array[u][i][j] = temp;
        end
    end
end
endtask

task FFN0_SOFT(
    input logic [`IDATA_WIDTH-1:0] ATT_RESIDUAL_array                [TB_NUM_USER-1:0][TB_TOTAL_CONTEXT_LENGTH-1:0][TB_EMBD_SIZE-1:0],
    input logic [`IDATA_WIDTH-1:0] FFN0_weights_array                                 [TB_EMBD_SIZE-1:0][4*TB_EMBD_SIZE/`HEAD_NUM-1:0],
    input integer scale, //symmetrical quantization, no bias
    input integer shift,
    output logic [`IDATA_WIDTH-1:0] FFN0_array                        [TB_NUM_USER-1:0][TB_TOTAL_CONTEXT_LENGTH-1:0][4*TB_EMBD_SIZE/`HEAD_NUM-1:0]
);
longint   temp_sum;
integer   round;
shortreal real_data_array [TB_EMBD_SIZE-1:0];
shortreal square_sum;
shortreal rms;
shortreal one_over_krms;


for(int u = 0; u < TB_NUM_USER; u++) begin
    for(int i = 0; i < TB_TOTAL_CONTEXT_LENGTH; i++)begin
        square_sum = 0;
        rms = 0;
        for(int j = 0; j < TB_EMBD_SIZE; j++)begin
            real_data_array[j] = $itor($signed(ATT_RESIDUAL_array[u][i][j]));
        end
        for(int j =0;j < RMS_K; j++)begin
            square_sum = square_sum + real_data_array[j] * real_data_array[j];
        end
        square_sum = square_sum * mlp_rms_dequant_scale_square;
        rms = $sqrt(square_sum/RMS_K * 1.0);
        one_over_krms = 1/rms;
        for(int j = 0; j < 4*TB_EMBD_SIZE/`HEAD_NUM; j++)begin
            temp_sum = 0;
            round = 0;
            for(int k = 0; k < TB_EMBD_SIZE; k++)begin
                temp_sum = temp_sum + $signed(ATT_RESIDUAL_array[u][i][k]) * $signed(FFN0_weights_array[k][j]);
            end
            temp_sum = $rtoi($itor(temp_sum) * one_over_krms);
            temp_sum = temp_sum * scale;
            round = temp_sum[shift-1];
            temp_sum = temp_sum >>> shift;
            temp_sum = temp_sum + round;
            if(temp_sum > 127)
                temp_sum = 127;
            if(temp_sum < -128)
                temp_sum = -128;
            if(temp_sum < 0)
                temp_sum = 0; //relu
            FFN0_array[u][i][j] = temp_sum;
        end
    end
end
endtask

task FFN1_SOFT(
    input logic [`IDATA_WIDTH-1:0] FFN0_array        [`HEAD_NUM-1:0][TB_NUM_USER-1:0][TB_TOTAL_CONTEXT_LENGTH-1:0][4*TB_EMBD_SIZE/`HEAD_NUM-1:0],
    input logic [`IDATA_WIDTH-1:0] FFN1_weights_array   [`HEAD_NUM-1:0][4*TB_EMBD_SIZE/`HEAD_NUM-1:0][TB_EMBD_SIZE-1:0],
    input integer scale, //symmetrical quantization, no bias
    input integer shift,
    output logic [`IDATA_WIDTH-1:0] FFN1_array   [TB_NUM_USER-1:0][TB_TOTAL_CONTEXT_LENGTH-1:0][TB_EMBD_SIZE-1:0]
);
longint   temp_sum;
integer   round;
int h;
int l;

for(int u = 0; u < TB_NUM_USER; u++) begin
    for(int i = 0; i < TB_TOTAL_CONTEXT_LENGTH; i++)begin
        for(int j = 0; j < TB_EMBD_SIZE; j++)begin
            temp_sum = 0;
            round = 0;
            for(int k = 0; k < 4 * TB_EMBD_SIZE; k++)begin
                h = k / (4 * TB_EMBD_SIZE/`HEAD_NUM);
                l = k % (4 * TB_EMBD_SIZE/`HEAD_NUM);
                temp_sum = temp_sum + $signed(FFN0_array[h][u][i][l]) * $signed(FFN1_weights_array[h][l][j]);
            end
            temp_sum = temp_sum * scale;
            round = temp_sum[shift-1];
            temp_sum = temp_sum >>> shift;
            temp_sum = temp_sum + round;
            if(temp_sum > 127)
                temp_sum = 127;
            if(temp_sum < -128)
                temp_sum = -128;
            FFN1_array[u][i][j] = temp_sum;
        end
    end
end
endtask




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

                    core_wmem_array[h][i][j] = sram_temp_var;
                end

                //Initialize K weights array
                for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE / `HEAD_NUM / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++)begin //K_weight depth in one core mem
                    for(int k =0; k <`MAC_MULT_NUM;k++)begin
                        temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE+j/(TB_EMBD_SIZE/`MAC_MULT_NUM); //`EMBD_SIZE是权重矩阵的行数，这里决定第几列,TB_EMBD_SIZE/`MAC_MULT_NUM是存一列权重需要的行数
                        temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM))  * `MAC_MULT_NUM + k;//这里决定那一列中的第几行
                        sram_temp_var[k*8 +: 8] = K_weights_array[h][temp_x][temp_y];
                    end

                    core_wmem_array[h][i][j+K_WEIGHT_ADDR_BASE] = sram_temp_var;
                end

                
                //Initialize V weights array
                for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE / `HEAD_NUM / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++)begin //V_weight depth in one core mem
                    for(int k =0; k <`MAC_MULT_NUM;k++)begin
                        temp_y = i*TB_QKV_WEIGHT_COLS_PER_CORE+j/(TB_EMBD_SIZE/`MAC_MULT_NUM); //`EMBD_SIZE是权重矩阵的行数，这里决定第几列,TB_EMBD_SIZE/`MAC_MULT_NUM是存一列权重需要的行数
                        temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM))  * `MAC_MULT_NUM + k;//这里决定那一列中的第几行
                        sram_temp_var[k*8 +: 8] = V_weights_array[h][temp_x][temp_y];
                    end

                    core_wmem_array[h][i][j+V_WEIGHT_ADDR_BASE] = sram_temp_var;
                end

                for(int j = 0; j < (TB_EMBD_SIZE * TB_EMBD_SIZE / `HEAD_NUM / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++)begin //Proj_weight depth in one core mem
                    for(int k =0; k <`MAC_MULT_NUM;k++)begin
                        temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM)+j/(TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM); //TB_EMBD_SIZE/`HEAD_NUM是权重矩阵的行数，这里决定第几列,TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM是存一列权重需要的行数
                        temp_x = (j % (TB_EMBD_SIZE/`HEAD_NUM / `MAC_MULT_NUM))  * `MAC_MULT_NUM + k;//这里决定那一列中的第几行
                        sram_temp_var[k*8 +: 8] = PROJ_weights_array[h][temp_x][temp_y];
                    end

                    core_wmem_array[h][i][j+PROJ_WEIGHT_ADDR_BASE] = sram_temp_var;
                end

                for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE / `HEAD_NUM / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++)begin //ffn0_weight depth in one core mem
                    for(int k =0; k <`MAC_MULT_NUM;k++)begin
                        temp_y = i*(TB_QKV_WEIGHT_COLS_PER_CORE*4)+j/(TB_EMBD_SIZE/`MAC_MULT_NUM); //`EMBD_SIZE是权重矩阵的行数，这里决定第几列,TB_EMBD_SIZE/`MAC_MULT_NUM是存一列权重需要的行数
                        temp_x = (j % (TB_EMBD_SIZE / `MAC_MULT_NUM))  * `MAC_MULT_NUM + k;//这里决定那一列中的第几行
                        sram_temp_var[k*8 +: 8] = FFN0_weights_array[h][temp_x][temp_y];
                    end

                    core_wmem_array[h][i][j+FFN0_WEIGHT_ADDR_BASE] = sram_temp_var;
                end

                for(int j = 0; j < (4 * TB_EMBD_SIZE * TB_EMBD_SIZE / `HEAD_NUM / `HEAD_CORE_NUM / `MAC_MULT_NUM); j++)begin //ffn1_weight depth in one core mem
                    for(int k =0; k <`MAC_MULT_NUM;k++)begin
                        temp_y = i*(TB_EMBD_SIZE/`HEAD_CORE_NUM)+j/(4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM); //4*TB_EMBD_SIZE/`HEAD_NUM是权重矩阵的行数，这里决定第几列,4*TB_EMBD_SIZE/`HEAD_NUM/`MAC_MULT_NUM是存一列权重需要的行数
                        temp_x = (j % (4*TB_EMBD_SIZE/`HEAD_NUM / `MAC_MULT_NUM))  * `MAC_MULT_NUM + k;//这里决定那一列中的第几行
                        sram_temp_var[k*8 +: 8] = FFN1_weights_array[h][temp_x][temp_y];
                    end

                    core_wmem_array[h][i][j+FFN1_WEIGHT_ADDR_BASE] = sram_temp_var;
                end
                WMEM_INIT_FLAG = 1;
            end
        end
    end
endgenerate





/////////////////////////////////////////////////
//                 GOLDEN GEN                  //
/////////////////////////////////////////////////
initial begin
    #10;
    for(int h =0; h < `HEAD_NUM; h++)begin
        $display("0");
        QKV_gen_soft(input_x_array, Q_weights_array[h], Q_GEN_SCALE, Q_GEN_SHIFT, Q_array[h]);
        $display("1");
        QKV_gen_soft(input_x_array, K_weights_array[h], K_GEN_SCALE, K_GEN_SHIFT, K_array[h]);
        $display("2");
        QKV_gen_soft(input_x_array, V_weights_array[h], V_GEN_SCALE, V_GEN_SHIFT, V_array[h]);
        $display("3");
        ATT_QK_soft(Q_array[h], K_array[h], QKT_SCALE, QKT_SHIFT, ATT_QK_array[h]);
        $display("4");
        ATT_PV_soft(ATT_QK_array[h], V_array[h], PV_SCALE, PV_SHIFT, ATT_PV_array[h]);
        $display("5");

        max_value = $signed(Q_array[h][0][0][0]);
        min_value = max_value;
        for (int i = 0; i < TB_NUM_USER; i++) begin
            for (int j = 0; j < TB_TOTAL_CONTEXT_LENGTH; j++) begin
                for (int k = 0; k < TB_EMBD_SIZE/`HEAD_NUM; k++) begin
                    if ($signed(Q_array[h][i][j][k]) > $signed(max_value)) begin
                        max_value = $signed(Q_array[h][i][j][k]);
                    end
                    if ($signed(Q_array[h][i][j][k]) < $signed(min_value)) begin
                        min_value = $signed(Q_array[h][i][j][k]);
                    end
                end
            end
        end
        $display("Max value in Q: %0d", $signed(max_value)); 
        $display("Min value in Q: %0d", $signed(min_value)); 


        max_value = $signed(K_array[h][0][0][0]);
        min_value = max_value;
        for (int i = 0; i < TB_NUM_USER; i++) begin
            for (int j = 0; j < TB_TOTAL_CONTEXT_LENGTH; j++) begin
                for (int k = 0; k < TB_EMBD_SIZE/`HEAD_NUM; k++) begin
                    if ($signed(K_array[h][i][j][k]) > $signed(max_value)) begin
                        max_value = $signed(K_array[h][i][j][k]);
                    end
                    if ($signed(K_array[h][i][j][k]) < $signed(min_value)) begin
                        min_value = $signed(K_array[h][i][j][k]);
                    end
                end
            end
        end
        $display("Max value in K: %0d", $signed(max_value)); 
        $display("Min value in K: %0d", $signed(min_value)); 


        max_value = $signed(V_array[h][0][0][0]);
        min_value = max_value;
        for (int i = 0; i < TB_NUM_USER; i++) begin
            for (int j = 0; j < TB_TOTAL_CONTEXT_LENGTH; j++) begin
                for (int k = 0; k < TB_EMBD_SIZE/`HEAD_NUM; k++) begin
                    if ($signed(V_array[h][i][j][k]) > $signed(max_value)) begin
                        max_value = $signed(V_array[h][i][j][k]);
                    end
                    if ($signed(V_array[h][i][j][k]) < $signed(min_value)) begin
                        min_value = $signed(V_array[h][i][j][k]);
                    end
                end
            end
        end
        $display("Max value in V: %0d", $signed(max_value)); 
        $display("Min value in V: %0d", $signed(min_value)); 


        max_value = $signed(ATT_QK_array[h][0][0][0]);
        min_value = max_value;
        for (int i = 0; i < TB_NUM_USER; i++) begin
            for (int j = 0; j < TB_TOTAL_CONTEXT_LENGTH; j++) begin
                for (int k = 0; k < TB_TOTAL_CONTEXT_LENGTH; k++) begin
                    if ($signed(ATT_QK_array[h][i][j][k]) > $signed(max_value)) begin
                        max_value = $signed(ATT_QK_array[h][i][j][k]);
                    end
                    if ($signed(ATT_QK_array[h][i][j][k]) < $signed(min_value)) begin
                        min_value = $signed(ATT_QK_array[h][i][j][k]);
                    end
                end
            end
        end
        $display("Max value in QK: %0d", $signed(max_value)); 
        $display("Min value in QK: %0d", $signed(min_value)); 

        max_value = $signed(ATT_PV_array[h][0][0][0]);
        min_value = max_value;
        for (int i = 0; i < TB_NUM_USER; i++) begin
            for (int j = 0; j < TB_TOTAL_CONTEXT_LENGTH; j++) begin
                for (int k = 0; k < TB_EMBD_SIZE/`HEAD_NUM; k++) begin
                    // if(j==40)
                    // $display("%D",$signed(ATT_PV_array[h][i][j][k]));
                    if ($signed(ATT_PV_array[h][i][j][k]) > $signed(max_value)) begin
                        max_value = $signed(ATT_PV_array[h][i][j][k]);
                    end
                    if ($signed(ATT_PV_array[h][i][j][k]) < $signed(min_value)) begin
                        min_value = $signed(ATT_PV_array[h][i][j][k]);
                    end
                end
            end
        end
        if(h==0)begin
            for(int kk = 0; kk < TB_TOTAL_CONTEXT_LENGTH;kk++)begin
                $display(kk);
                for(int jj =0; jj < TB_EMBD_SIZE/`HEAD_NUM; jj++)
                    $write("%5d, ",$signed(ATT_PV_array[h][0][kk][jj]));
            end
        end
        $display("Max value in PV: %0d", $signed(max_value)); 
        $display("Min value in PV: %0d", $signed(min_value)); 
    end

    PROJ_gen_soft(ATT_PV_array, PROJ_weights_array, PROJ_SCALE, PROJ_SHIFT, ATT_PROJ_array);
    $display("6");
    max_value = $signed(ATT_PROJ_array[0][0][0]);
    min_value = max_value;
    for (int i = 0; i < TB_NUM_USER; i++) begin
        for (int j = 0; j < TB_TOTAL_CONTEXT_LENGTH; j++) begin
            for (int k = 0; k < TB_EMBD_SIZE; k++) begin
                // $display("%d",$signed(ATT_PROJ_array[i][j][k]));
                if ($signed(ATT_PROJ_array[i][j][k]) > $signed(max_value)) begin
                    max_value = $signed(ATT_PROJ_array[i][j][k]);
                end
                if ($signed(ATT_PROJ_array[i][j][k]) < $signed(min_value)) begin
                    min_value = $signed(ATT_PROJ_array[i][j][k]);
                end
            end
        end
    end
    // $display(ATT_PROJ_array);
    $display("Max value in Proj: %0d", $signed(max_value)); 
    $display("Min value in Proj: %0d", $signed(min_value)); 

    RESIDUAL_gen_soft(input_x_array, ATT_PROJ_array, ATTN_RESIDUAL_SCALE_A, ATTN_RESIDUAL_SCALE_B, ATTN_RESIDUAL_SHIFT, ATT_RESIDUAL_array);
    $display("7");
    max_value = $signed(ATT_RESIDUAL_array[0][0][0]);
    min_value = max_value;
    for (int i = 0; i < TB_NUM_USER; i++) begin
        for (int j = 0; j < TB_TOTAL_CONTEXT_LENGTH; j++) begin
            for (int k = 0; k < TB_EMBD_SIZE; k++) begin
                if ($signed(ATT_RESIDUAL_array[i][j][k]) > $signed(max_value)) begin
                    max_value = $signed(ATT_RESIDUAL_array[i][j][k]);
                end
                if ($signed(ATT_RESIDUAL_array[i][j][k]) < $signed(min_value)) begin
                    min_value = $signed(ATT_RESIDUAL_array[i][j][k]);
                end
            end
        end
    end
    $display("Max value in Residual: %0d", $signed(max_value)); 
    $display("Min value in Residual: %0d", $signed(min_value)); 

    for(int h =0; h < `HEAD_NUM; h++)begin
        FFN0_SOFT(ATT_RESIDUAL_array, FFN0_weights_array[h], FFN0_SCALE, FFN0_SHIFT, FFN0_array[h]);
        $display("8");
        max_value = $signed(FFN0_array[h][0][0][0]);
        min_value = max_value;
        for (int i = 0; i < TB_NUM_USER; i++) begin
            for (int j = 0; j < TB_TOTAL_CONTEXT_LENGTH; j++) begin
                for (int k = 0; k < 4*TB_EMBD_SIZE/`HEAD_NUM; k++) begin
                    if ($signed(FFN0_array[h][i][j][k]) > $signed(max_value)) begin
                        max_value = $signed(FFN0_array[h][i][j][k]);
                    end
                    if ($signed(FFN0_array[h][i][j][k]) < $signed(min_value)) begin
                        min_value = $signed(FFN0_array[h][i][j][k]);
                    end
                end
            end
        end
        $display("Max value in FFN0: %0d", $signed(max_value)); 
        $display("Min value in FFN0: %0d", $signed(min_value)); 
    end

    FFN1_SOFT(FFN0_array, FFN1_weights_array, FFN1_SCALE, FFN1_SHIFT, FFN1_array);
    $display("9");
    max_value = $signed(FFN1_array[0][0][0]);
    min_value = max_value;
    for (int i = 0; i < TB_NUM_USER; i++) begin
        for (int j = 0; j < TB_TOTAL_CONTEXT_LENGTH; j++) begin
            // $display("USER[%0d], TOKEN[%0d]",i,j);
            for (int k = 0; k < TB_EMBD_SIZE; k++) begin
                // $write("%0d ",$signed(FFN1_array[i][j][k]));
                // $display("%d",$signed(FFN1_array[i][j][k]));
                if ($signed(FFN1_array[i][j][k]) > $signed(max_value)) begin
                    max_value = $signed(FFN1_array[i][j][k]);
                end
                if ($signed(FFN1_array[i][j][k]) < $signed(min_value)) begin
                    min_value = $signed(FFN1_array[i][j][k]);
                end
            end
            // $display();
        end
    end
    $display("Max value in FFN1: %0d", $signed(max_value)); 
    $display("Min value in FFN1: %0d", $signed(min_value)); 

    RESIDUAL_gen_soft(ATT_RESIDUAL_array, FFN1_array, MLP_RESIDUAL_SCALE_A, MLP_RESIDUAL_SCALE_B, MLP_RESIDUAL_SHIFT, MLP_RESIDUAL_array);
    $display("10");
    max_value = $signed(MLP_RESIDUAL_array[0][0][0]);
    min_value = max_value;
    for (int i = 0; i < TB_NUM_USER; i++) begin
        for (int j = 0; j < TB_TOTAL_CONTEXT_LENGTH; j++) begin
            for (int k = 0; k < TB_EMBD_SIZE; k++) begin
                if ($signed(MLP_RESIDUAL_array[i][j][k]) > $signed(max_value)) begin
                    max_value = $signed(MLP_RESIDUAL_array[i][j][k]);
                end
                if ($signed(MLP_RESIDUAL_array[i][j][k]) < $signed(min_value)) begin
                    min_value = $signed(MLP_RESIDUAL_array[i][j][k]);
                end
            end
        end
    end
    $display("Max value in Residual: %0d", $signed(max_value)); 
    $display("Min value in Residual: %0d", $signed(min_value)); 



end




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
    for(int i=0;i<TB_TOTAL_CONTEXT_LENGTH;i++) begin
        for(int j=0;j<TB_NUM_USER;j++) begin  
             
            for(int m = 0; m < TB_EMBD_SIZE/`MAC_MULT_NUM;m++)begin  //EMBD_SIZE是`MAC_MULT_NUM的倍数
                for(int k = 0;k < `MAC_MULT_NUM;k++)begin
                    sram_temp_var[k*8 +: 8] = input_x_array[j][i][m*`MAC_MULT_NUM+k];
                end
                global_mem_array[m] = sram_temp_var;
            end
            addr = `GLOBAL_MEM_BASE_ADDR;
            burst_cnt = TB_EMBD_SIZE/2-1;
            for(int jj = 0; jj < 256; jj++)begin
                wdata_array[jj] = global_mem_array[jj/(`MAC_MULT_NUM/2)][(jj%(`MAC_MULT_NUM/2))*16 +: 16];
            end
            FPGA_QSPI_WR(addr, burst_cnt, wdata_array);


            repeat(10) @(negedge chip_clk);
            new_token = 1;
            user_id = j;   
            user_first_token = (i==0);
            burst_cnt = 0;
            addr = `INSTRUCTION_REGISTERS_BASE_ADDR + 0;
            wdata_array[0] = {12'b0, user_id, user_first_token, new_token};
            FPGA_QSPI_WR(addr, burst_cnt, wdata_array);


            while(1)  begin
                @(negedge chip_clk);
                if(top.array_top_inst.inst_array_ctrl.control_state == ATT_PV_STATE)begin
                    break;
                end
            end

            $display("USER: %0d DONE Processing, TOKEN_CNT: %0d, TIME: %t",j,usr_total_token_cnt_array[j],$time);
            $display("Iter cnt: %d", iter_cnt);
            $display("QK finish");
            // repeat(10) @(negedge chip_clk);
            att_qk_done = 1;
            @(negedge chip_clk);
            att_qk_done = 0;

            burst_cnt = 0;
            addr = `STATE_REGISTERS_BASE_ADDR + 1;
            FPGA_QSPI_RD(addr, burst_cnt, tpu_state_rdata_array);
            $display("State reg[1] = %d", tpu_state_rdata_array[0]);


            while(1)  begin
                @(negedge chip_clk);
                if(top.array_top_inst.inst_array_ctrl.control_state == PROJ_STATE)begin
                    break;
                end
            end

            $display("USER: %0d DONE Processing, TOKEN_CNT: %0d, TIME: %t",j,usr_total_token_cnt_array[j],$time);
            $display("Iter cnt: %d", iter_cnt);
            $display("PV finish");
            att_pv_done = 1;
            @(negedge chip_clk);
            att_pv_done = 0;

            burst_cnt = 0;
            addr = `STATE_REGISTERS_BASE_ADDR + 1;
            FPGA_QSPI_RD(addr, burst_cnt, tpu_state_rdata_array);
            $display("State reg[1] = %d", tpu_state_rdata_array[0]);

            while(1)  begin
                @(negedge chip_clk);
                if(top.array_top_inst.inst_array_ctrl.control_state == FFN0_STATE)begin
                    break;
                end
            end    

            $display("USER: %0d DONE Processing, TOKEN_CNT: %0d, TIME: %t",j,usr_total_token_cnt_array[j],$time);
            $display("Iter cnt: %d", iter_cnt);
            $display("Proj finish");

            proj_done = 1;
            @(negedge chip_clk);
            proj_done = 0;


            burst_cnt = 0;
            addr = `STATE_REGISTERS_BASE_ADDR + 1;
            FPGA_QSPI_RD(addr, burst_cnt, tpu_state_rdata_array);
            $display("State reg[1] = %d", tpu_state_rdata_array[0]);


            while(1)  begin
                @(negedge chip_clk);
                if(top.array_top_inst.inst_array_ctrl.control_state == FFN1_STATE)begin
                    break;
                end
            end   

            $display("USER: %0d DONE Processing, TOKEN_CNT: %0d, TIME: %t",j,usr_total_token_cnt_array[j],$time);
            $display("Iter cnt: %d", iter_cnt);
            $display("FFN0 finish");
            ffn0_done = 1;
            @(negedge chip_clk);
            ffn0_done = 0;

            burst_cnt = 0;
            addr = `STATE_REGISTERS_BASE_ADDR + 1;
            FPGA_QSPI_RD(addr, burst_cnt, tpu_state_rdata_array);
            $display("State reg[1] = %d", tpu_state_rdata_array[0]);


            // while(1)  begin
            //     @(negedge chip_clk);
            //     if(top.array_top_inst.current_token_finish)begin
            //         break;
            //     end
            // end
            // repeat(300) @(negedge chip_clk);
            while(1)begin
                burst_cnt = 0;
                addr = `STATE_REGISTERS_BASE_ADDR + 0;
                FPGA_QSPI_RD(addr, burst_cnt, rdata_array);
                if(rdata_array[0] == 1)
                    break;
            end

            $display("USER: %0d DONE Processing, TOKEN_CNT: %0d, TIME: %t",j,usr_total_token_cnt_array[j],$time);
            $display("Iter cnt: %d", iter_cnt);
            $display("FFN1 finish");

            
            for(int ii =0; ii < TB_EMBD_SIZE/2/64; ii++)begin
                addr = `RESIDUAL_MEM_BASE_ADDR + 64 * ii;
                burst_cnt = 63;
                FPGA_QSPI_RD(addr, burst_cnt, rdata_array);
                for(int jj = 0; jj < 64; jj++)begin
                    global_mem_array[(ii*64+jj)/(`MAC_MULT_NUM/2)][(jj%(`MAC_MULT_NUM/2))*16 +: 16] = rdata_array[jj];
                end
            end

            ffn1_done = 1;
            @(negedge chip_clk);
            ffn1_done = 0;

            burst_cnt = 0;
            addr = `STATE_REGISTERS_BASE_ADDR + 1;
            FPGA_QSPI_RD(addr, burst_cnt, tpu_state_rdata_array);
            $display("State reg[1] = %d", tpu_state_rdata_array[0]);

            iter_cnt=iter_cnt+1;
            usr_total_token_cnt_array[j] = usr_total_token_cnt_array[j] + 1;

            if(iter_cnt == 1) //here to check true mem and no true mem whether has same result
                `ifndef TRUE_MEM
                $display("%32x",top.array_top_inst.inst_residual_sram.ram_piece.mem[0]);
                `else
                $display("%32x",top.array_top_inst.inst_residual_sram.ram_piece.ARRAY[0]);
                `endif

            $display("\n\n\n\n\n\n\n");
            if(iter_cnt == TB_TEST_ITER) begin
                iter_done=1;
                break;
            end
        end
        if(iter_cnt == TB_TEST_ITER) begin
            iter_done=1;
            break;
        end
    end


    repeat(100) @(negedge chip_clk);
    $finish();
end



///////////////////////////////////////////
//               Check                   //
///////////////////////////////////////////

int temp_k;
genvar iter;

`ifndef TRUE_MEM

// Check ATT PV Result after every iteration
`ifdef CHECK_PV
generate 
    for(h=0; h <`HEAD_NUM; h++)begin
        for(iter=0; iter < `MAC_MULT_NUM; iter++)begin
            logic [7:0] temp_var;
            initial begin
                for(int p=0;p<TB_TEST_ITER;p++) begin
                    #1;
                    while(1)begin
                        @(negedge chip_clk);
                        if(att_pv_done)
                            break;
                    end
                    // $display("***");
                    user_total_token_cnt = usr_total_token_cnt_array[user_id];
                    for(int j = 0; j < TB_EMBD_SIZE/`HEAD_NUM; j++)begin
                        temp_k =j; //the temp_k element
                        if(iter == temp_k%`MAC_MULT_NUM) begin
                            if(h%2==0)begin
                                temp_var = top.array_top_inst.inst_head_array.head_gen_array[h/2].two_heads_inst.inst_head_top_0.inst_head_sram.ram_piece.mem[temp_k/`MAC_MULT_NUM][iter*8 +: 8];
                            end
                            else begin
                                temp_var = top.array_top_inst.inst_head_array.head_gen_array[h/2].two_heads_inst.inst_head_top_1.inst_head_sram.ram_piece.mem[temp_k/`MAC_MULT_NUM][iter*8 +: 8];
                            end
                            if(ATT_PV_array[h][user_id][user_total_token_cnt][j] !== 'bx && $signed(ATT_PV_array[h][user_id][user_total_token_cnt][j]) !== $signed(temp_var))begin
                                if($abs(int'($signed(ATT_PV_array[h][user_id][user_total_token_cnt][j]))-int'($signed(temp_var)))>PV_MAX_ERR) begin
                                    error_flag = 1;
                                    $display("Wrong!!!, Below Case exceed the error limit");
                                    $display("ATT_PV_array[%0d][%0d][%0d][%0d]: %0d != HEAD_SRAM[%0d][%0d][%0d][%0d] = %0d",
                                    h, user_id, user_total_token_cnt, j, $signed(ATT_PV_array[h][user_id][user_total_token_cnt][j]),
                                    h, user_id, temp_k/`MAC_MULT_NUM, iter, $signed(temp_var));
                                end
                                if(FORCE_TO_BE_RIGHT) begin
                                    if(h%2==0)
                                        top.array_top_inst.inst_head_array.head_gen_array[h/2].two_heads_inst.inst_head_top_0.inst_head_sram.ram_piece.mem[temp_k/`MAC_MULT_NUM][iter*8 +: 8] = $signed(ATT_PV_array[h][user_id][user_total_token_cnt][j]);
                                    else 
                                        top.array_top_inst.inst_head_array.head_gen_array[h/2].two_heads_inst.inst_head_top_1.inst_head_sram.ram_piece.mem[temp_k/`MAC_MULT_NUM][iter*8 +: 8] = $signed(ATT_PV_array[h][user_id][user_total_token_cnt][j]);
                                end
                            end
                        end
                    end
                end
            end
        end
    end
endgenerate
`endif 

`ifdef CHECK_PROJ
generate 
        for(iter=0; iter < `MAC_MULT_NUM; iter++)begin
            initial begin
                while(1) begin
                    #1;
                    while(1)begin
                        @(negedge chip_clk);
                        if(proj_done)
                            break;
                    end
                    // $display("***");
                    user_total_token_cnt = usr_total_token_cnt_array[user_id];
                    for(int j = 0; j < TB_EMBD_SIZE; j++)begin
                        temp_k =j; //the temp_k element
                        if(iter == temp_k%`MAC_MULT_NUM) begin
                            if(ATT_PROJ_array[user_id][user_total_token_cnt][j] !== 'bx && $signed(ATT_PROJ_array[user_id][user_total_token_cnt][j]) !== $signed(top.array_top_inst.inst_residual_sram.ram_piece.mem[temp_k/`MAC_MULT_NUM][iter*8 +: 8]))begin
                                if($abs(int'($signed(ATT_PROJ_array[user_id][user_total_token_cnt][j]))-int'($signed(top.array_top_inst.inst_residual_sram.ram_piece.mem[temp_k/`MAC_MULT_NUM][iter*8 +: 8])))>PROJ_MAX_ERR) begin
                                    error_flag = 1;
                                    $display("Wrong!!!, Below Case exceed the error limit");
                                    $display("ATT_PROJ_array[%0d][%0d][%0d]: %0d != Residual SRAM[%0d][%0d][%0d] = %0d",
                                    user_id, user_total_token_cnt, j, $signed(ATT_PROJ_array[user_id][user_total_token_cnt][j]),
                                    user_id, temp_k/`MAC_MULT_NUM, iter, $signed(top.array_top_inst.inst_residual_sram.ram_piece.mem[temp_k/`MAC_MULT_NUM][iter*8 +: 8]));

                                end
                                if(FORCE_TO_BE_RIGHT)
                                    top.array_top_inst.inst_residual_sram.ram_piece.mem[temp_k/`MAC_MULT_NUM][iter*8 +: 8] = $signed(ATT_PROJ_array[user_id][user_total_token_cnt][j]);
                            end
                        end
                    end
                end
            end
        end
endgenerate
`endif


`ifdef CHECK_FFN0
//check ffn0
generate 
    for(h=0; h <`HEAD_NUM; h++)begin
        for(iter=0; iter < `MAC_MULT_NUM; iter++)begin
            logic [7:0] temp_var;
            initial begin
                for(int p=0;p<TB_TEST_ITER;p++) begin
                    #1;
                    while(1)begin
                        @(negedge chip_clk);
                        if(ffn0_done)
                            break;
                    end
                    // $display("***");
                    user_total_token_cnt = usr_total_token_cnt_array[user_id];
                    for(int j = 0; j < 4*TB_EMBD_SIZE/`HEAD_NUM; j++)begin
                        temp_k =j; //the temp_k element
                        if(iter == temp_k%`MAC_MULT_NUM) begin
                            if(h%2==0)begin
                                temp_var = top.array_top_inst.inst_head_array.head_gen_array[h/2].two_heads_inst.inst_head_top_0.inst_head_sram.ram_piece.mem[temp_k/`MAC_MULT_NUM][iter*8 +: 8];
                            end
                            else begin
                                temp_var = top.array_top_inst.inst_head_array.head_gen_array[h/2].two_heads_inst.inst_head_top_1.inst_head_sram.ram_piece.mem[temp_k/`MAC_MULT_NUM][iter*8 +: 8];
                            end
                            if(FFN0_array[h][user_id][user_total_token_cnt][j] !== 'bx && $signed(FFN0_array[h][user_id][user_total_token_cnt][j]) !== $signed(temp_var))begin
                                if($abs(int'($signed(FFN0_array[h][user_id][user_total_token_cnt][j]))-int'($signed(temp_var)))>FFN0_MAX_ERR) begin
                                    error_flag = 1;
                                    $display("Wrong!!!, Below Case exceed the error limit");
                                    $display("FFN0_array[%0d][%0d][%0d][%0d]: %0d != HEAD_SRAM[%0d][%0d][%0d][%0d] = %0d",
                                    h, user_id, user_total_token_cnt, j, $signed(FFN0_array[h][user_id][user_total_token_cnt][j]),
                                    h, user_id, temp_k/`MAC_MULT_NUM, iter, $signed(temp_var));
                                end
                                
                                if(FORCE_TO_BE_RIGHT) begin
                                    if(h%2==0)begin
                                        top.array_top_inst.inst_head_array.head_gen_array[h/2].two_heads_inst.inst_head_top_0.inst_head_sram.ram_piece.mem[temp_k/`MAC_MULT_NUM][iter*8 +: 8] = FFN0_array[h][user_id][user_total_token_cnt][j];
                                    end
                                    else begin
                                        top.array_top_inst.inst_head_array.head_gen_array[h/2].two_heads_inst.inst_head_top_1.inst_head_sram.ram_piece.mem[temp_k/`MAC_MULT_NUM][iter*8 +: 8] = FFN0_array[h][user_id][user_total_token_cnt][j];
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
endgenerate
`endif

`ifdef CHECK_FFN1
generate 
        for(iter=0; iter < `MAC_MULT_NUM; iter++)begin
            initial begin
                while(1) begin
                    #1;
                    while(1)begin
                        @(negedge chip_clk);
                        if(ffn1_done)
                            break;
                    end
                    // $display("***");
                    user_total_token_cnt = usr_total_token_cnt_array[user_id];
                    for(int j = 0; j < TB_EMBD_SIZE; j++)begin
                        temp_k =j; //the temp_k element
                        if(iter == temp_k%`MAC_MULT_NUM) begin
                            if(MLP_RESIDUAL_array[user_id][user_total_token_cnt][j] !== 'bx && $signed(MLP_RESIDUAL_array[user_id][user_total_token_cnt][j]) !== $signed(top.array_top_inst.inst_residual_sram.ram_piece.mem[temp_k/`MAC_MULT_NUM][iter*8 +: 8]))begin
                                if($abs(int'($signed(MLP_RESIDUAL_array[user_id][user_total_token_cnt][j]))-int'($signed(top.array_top_inst.inst_residual_sram.ram_piece.mem[temp_k/`MAC_MULT_NUM][iter*8 +: 8])))>FFN1_MAX_ERR) begin
                                    error_flag = 1;
                                    $display("Wrong!!!, Below Case exceed the error limit");
                                    $display("MLP_RESIDUAL_array[%0d][%0d][%0d]: %0d != Residual SRAM[%0d][%0d][%0d] = %0d",
                                    user_id, user_total_token_cnt, j, $signed(MLP_RESIDUAL_array[user_id][user_total_token_cnt][j]),
                                    user_id, temp_k/`MAC_MULT_NUM, iter, $signed(top.array_top_inst.inst_residual_sram.ram_piece.mem[temp_k/`MAC_MULT_NUM][iter*8 +: 8]));
                                end
                            end
                        end
                    end
                end
            end
        end
endgenerate
`endif 


`ifdef CHECK_FFN1
generate 
        for(iter=0; iter < `MAC_MULT_NUM; iter++)begin
            initial begin
                while(1) begin
                    #1;
                    while(1)begin
                        @(negedge chip_clk);
                        if(ffn1_done)
                            break;
                    end
                    // $display("***");
                    user_total_token_cnt = usr_total_token_cnt_array[user_id];
                    for(int j = 0; j < TB_EMBD_SIZE; j++)begin
                        temp_k =j; //the temp_k element
                        if(iter == temp_k%`MAC_MULT_NUM) begin
                            if(MLP_RESIDUAL_array[user_id][user_total_token_cnt][j] !== 'bx && $signed(MLP_RESIDUAL_array[user_id][user_total_token_cnt][j]) !== $signed(global_mem_array[temp_k/`MAC_MULT_NUM][iter*8 +: 8]))begin
                                if($abs(int'($signed(MLP_RESIDUAL_array[user_id][user_total_token_cnt][j]))-int'($signed(global_mem_array[temp_k/`MAC_MULT_NUM][iter*8 +: 8])))>FFN1_MAX_ERR) begin
                                    error_flag = 1;
                                    $display("Wrong!!!, Below Case exceed the error limit");
                                    $display("MLP_RESIDUAL_array[%0d][%0d][%0d]: %0d != Read out SRAM[%0d][%0d][%0d] = %0d",
                                    user_id, user_total_token_cnt, j, $signed(MLP_RESIDUAL_array[user_id][user_total_token_cnt][j]),
                                    user_id, temp_k/`MAC_MULT_NUM, iter, $signed(global_mem_array[temp_k/`MAC_MULT_NUM][iter*8 +: 8]));
                                end
                            end
                        end
                    end
                end
            end
        end
endgenerate
`endif 


`else //define true mem




// Check ATT PV Result after every iteration
`ifdef CHECK_PV
generate 
    for(h=0; h <`HEAD_NUM; h++)begin
        for(iter=0; iter < `MAC_MULT_NUM; iter++)begin
            logic [7:0] temp_var;
            initial begin
                for(int p=0;p<TB_TEST_ITER;p++) begin
                    #1;
                    while(1)begin
                        @(negedge chip_clk);
                        if(att_pv_done)
                            break;
                    end
                    // $display("***");
                    user_total_token_cnt = usr_total_token_cnt_array[user_id];
                    for(int j = 0; j < TB_EMBD_SIZE/`HEAD_NUM; j++)begin
                        temp_k =j; //the temp_k element
                        if(iter == temp_k%`MAC_MULT_NUM) begin
                            if(h%2==0)begin
                                temp_var = ~top.array_top_inst.inst_head_array.head_gen_array[h/2].two_heads_inst.inst_head_top_0.inst_head_sram.ram_piece.ARRAY[temp_k/`MAC_MULT_NUM][iter*8 +: 8];
                            end
                            else begin
                                temp_var = ~top.array_top_inst.inst_head_array.head_gen_array[h/2].two_heads_inst.inst_head_top_1.inst_head_sram.ram_piece.ARRAY[temp_k/`MAC_MULT_NUM][iter*8 +: 8];
                            end
                            if(ATT_PV_array[h][user_id][user_total_token_cnt][j] !== 'bx && $signed(ATT_PV_array[h][user_id][user_total_token_cnt][j]) !== $signed(temp_var))begin
                                if($abs(int'($signed(ATT_PV_array[h][user_id][user_total_token_cnt][j]))-int'($signed(temp_var)))>PV_MAX_ERR) begin
                                    error_flag = 1;
                                    $display("Wrong!!!, Below Case exceed the error limit");
                                    $display("ATT_PV_array[%0d][%0d][%0d][%0d]: %0d != HEAD_SRAM[%0d][%0d][%0d][%0d] = %0d",
                                    h, user_id, user_total_token_cnt, j, $signed(ATT_PV_array[h][user_id][user_total_token_cnt][j]),
                                    h, user_id, temp_k/`MAC_MULT_NUM, iter, $signed(temp_var));
                                end
                                if(FORCE_TO_BE_RIGHT) begin
                                    if(h%2==0)
                                        top.array_top_inst.inst_head_array.head_gen_array[h/2].two_heads_inst.inst_head_top_0.inst_head_sram.ram_piece.ARRAY[temp_k/`MAC_MULT_NUM][iter*8 +: 8] = ~(ATT_PV_array[h][user_id][user_total_token_cnt][j]);
                                    else 
                                        top.array_top_inst.inst_head_array.head_gen_array[h/2].two_heads_inst.inst_head_top_1.inst_head_sram.ram_piece.ARRAY[temp_k/`MAC_MULT_NUM][iter*8 +: 8] = ~(ATT_PV_array[h][user_id][user_total_token_cnt][j]);
                                end
                            end
                        end
                    end
                end
            end
        end
    end
endgenerate
`endif 

`ifdef CHECK_PROJ
generate 
        for(iter=0; iter < `MAC_MULT_NUM; iter++)begin
            initial begin
                while(1)begin
                    #1;
                    while(1)begin
                        @(negedge chip_clk);
                        if(proj_done)
                            break;
                    end
                    // $display("***");
                    user_total_token_cnt = usr_total_token_cnt_array[user_id];
                    for(int j = 0; j < TB_EMBD_SIZE; j++)begin
                        temp_k =j; //the temp_k element
                        if(iter == temp_k%`MAC_MULT_NUM) begin
                            if(ATT_PROJ_array[user_id][user_total_token_cnt][j] !== 'bx && $signed(ATT_PROJ_array[user_id][user_total_token_cnt][j]) !== $signed(~top.array_top_inst.inst_residual_sram.ram_piece.ARRAY[temp_k/`MAC_MULT_NUM][iter*8 +: 8]))begin
                                if($abs(int'($signed(ATT_PROJ_array[user_id][user_total_token_cnt][j]))-int'($signed(~top.array_top_inst.inst_residual_sram.ram_piece.ARRAY[temp_k/`MAC_MULT_NUM][iter*8 +: 8])))>PROJ_MAX_ERR) begin
                                    error_flag = 1;
                                    $display("Wrong!!!, Below Case exceed the error limit");
                                    $display("ATT_PROJ_array[%0d][%0d][%0d]: %0d != Residual SRAM[%0d][%0d][%0d] = %0d",
                                    user_id, user_total_token_cnt, j, $signed(ATT_PROJ_array[user_id][user_total_token_cnt][j]),
                                    user_id, temp_k/`MAC_MULT_NUM, iter, $signed(~top.array_top_inst.inst_residual_sram.ram_piece.ARRAY[temp_k/`MAC_MULT_NUM][iter*8 +: 8]));

                                end
                                if(FORCE_TO_BE_RIGHT)
                                    top.array_top_inst.inst_residual_sram.ram_piece.ARRAY[temp_k/`MAC_MULT_NUM][iter*8 +: 8] = ~(ATT_PROJ_array[user_id][user_total_token_cnt][j]);
                            end
                        end
                    end
                end
            end
        end
endgenerate
`endif


`ifdef CHECK_FFN0
//check ffn0
generate 
    for(h=0; h <`HEAD_NUM; h++)begin
        for(iter=0; iter < `MAC_MULT_NUM; iter++)begin
            logic [7:0] temp_var;
            initial begin
                for(int p=0;p<TB_TEST_ITER;p++) begin
                    #1;
                    while(1)begin
                        @(negedge chip_clk);
                        if(ffn0_done)
                            break;
                    end
                    // $display("***");
                    user_total_token_cnt = usr_total_token_cnt_array[user_id];
                    for(int j = 0; j < 4*TB_EMBD_SIZE/`HEAD_NUM; j++)begin
                        temp_k =j; //the temp_k element
                        if(iter == temp_k%`MAC_MULT_NUM) begin
                            if(h%2==0)begin
                                temp_var = ~top.array_top_inst.inst_head_array.head_gen_array[h/2].two_heads_inst.inst_head_top_0.inst_head_sram.ram_piece.ARRAY[temp_k/`MAC_MULT_NUM][iter*8 +: 8];
                            end
                            else begin
                                temp_var = ~top.array_top_inst.inst_head_array.head_gen_array[h/2].two_heads_inst.inst_head_top_1.inst_head_sram.ram_piece.ARRAY[temp_k/`MAC_MULT_NUM][iter*8 +: 8];
                            end
                            if(FFN0_array[h][user_id][user_total_token_cnt][j] !== 'bx && $signed(FFN0_array[h][user_id][user_total_token_cnt][j]) !== $signed(temp_var))begin
                                if($abs(int'($signed(FFN0_array[h][user_id][user_total_token_cnt][j]))-int'($signed(temp_var)))>FFN0_MAX_ERR) begin
                                    error_flag = 1;
                                    $display("Wrong!!!, Below Case exceed the error limit");
                                    $display("FFN0_array[%0d][%0d][%0d][%0d]: %0d != HEAD_SRAM[%0d][%0d][%0d][%0d] = %0d",
                                    h, user_id, user_total_token_cnt, j, $signed(FFN0_array[h][user_id][user_total_token_cnt][j]),
                                    h, user_id, temp_k/`MAC_MULT_NUM, iter, $signed(temp_var));
                                end
                                
                                if(FORCE_TO_BE_RIGHT) begin
                                    if(h%2==0)begin
                                        top.array_top_inst.inst_head_array.head_gen_array[h/2].two_heads_inst.inst_head_top_0.inst_head_sram.ram_piece.ARRAY[temp_k/`MAC_MULT_NUM][iter*8 +: 8] = ~FFN0_array[h][user_id][user_total_token_cnt][j];
                                    end
                                    else begin
                                        top.array_top_inst.inst_head_array.head_gen_array[h/2].two_heads_inst.inst_head_top_1.inst_head_sram.ram_piece.ARRAY[temp_k/`MAC_MULT_NUM][iter*8 +: 8] = ~FFN0_array[h][user_id][user_total_token_cnt][j];
                                    end
                                end
                            
                            end
                        end
                    end
                end
            end
        end
    end
endgenerate
`endif

`ifdef CHECK_FFN1
generate 
        for(iter=0; iter < `MAC_MULT_NUM; iter++)begin
            initial begin
                while(1)begin
                    #1;
                    while(1)begin
                        @(negedge chip_clk);
                        if(ffn1_done)
                            break;
                    end
                    // $display("***");
                    user_total_token_cnt = usr_total_token_cnt_array[user_id];
                    for(int j = 0; j < TB_EMBD_SIZE; j++)begin
                        temp_k =j; //the temp_k element
                        if(iter == temp_k%`MAC_MULT_NUM) begin
                            if(MLP_RESIDUAL_array[user_id][user_total_token_cnt][j] !== 'bx && $signed(MLP_RESIDUAL_array[user_id][user_total_token_cnt][j]) !== $signed(~top.array_top_inst.inst_residual_sram.ram_piece.ARRAY[temp_k/`MAC_MULT_NUM][iter*8 +: 8]))begin
                                if($abs(int'($signed(MLP_RESIDUAL_array[user_id][user_total_token_cnt][j]))-int'($signed(~top.array_top_inst.inst_residual_sram.ram_piece.ARRAY[temp_k/`MAC_MULT_NUM][iter*8 +: 8])))>FFN1_MAX_ERR) begin
                                    error_flag = 1;
                                    $display("Wrong!!!, Below Case exceed the error limit");
                                    $display("MLP_RESIDUAL_array[%0d][%0d][%0d]: %0d != Residual SRAM[%0d][%0d][%0d] = %0d",
                                    user_id, user_total_token_cnt, j, $signed(MLP_RESIDUAL_array[user_id][user_total_token_cnt][j]),
                                    user_id, temp_k/`MAC_MULT_NUM, iter, $signed(~top.array_top_inst.inst_residual_sram.ram_piece.ARRAY[temp_k/`MAC_MULT_NUM][iter*8 +: 8]));
                                end
                            end
                        end
                    end
                end
            end
        end
endgenerate
`endif 


`endif

always begin
    repeat(10000) @(negedge chip_clk);
    $display("Time: %t", $time());
end

endmodule
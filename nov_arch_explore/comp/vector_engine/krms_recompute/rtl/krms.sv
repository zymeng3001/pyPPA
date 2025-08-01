// Copyright (c) 2024, Saligane's Group at University of Michigan and Google Research
//
// Licensed under the Apache License, Version 2.0 (the "License");

// you may not use this file except in compliance with the License.

// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


//calculate the One over RMS
//Should be Compatible with TRNG
//TRNG can generate random number, if random number <A FIXED REG VALUE, which means the current input is valid
module krms 
#(
    parameter integer BUS_NUM = 8, //input bus num
    parameter integer DATA_NUM_WIDTH =  10, //layernorm vector length reg width
    parameter integer FIXED_SQUARE_SUM_WIDTH = 24,

    parameter integer sig_width = `SIG_WIDTH,
    parameter integer exp_width = `EXP_WIDTH,
    localparam        isize = 32,
    localparam        isign = 1, //signed integer number
    localparam        ieee_compliance = 0 //No support to NaN adn denormals
)(
    input  logic                                                     clk,
    input  logic                                                     rst_n,

    input  logic                                                     start,

    input  logic                                                     rc_cfg_vld,
    input  RC_CONFIG                                                 rc_cfg,     

    input  CONTROL_STATE                                             control_state,
    input  logic                                                     control_state_update,

    input  logic signed [BUS_NUM-1:0][`IDATA_WIDTH-1:0]              in_fixed_data, //vector的长度一定是BUS NUM的倍数
    input  logic                                                     in_fixed_data_vld,
    
    output logic        [`RECOMPUTE_SCALE_WIDTH-1:0]                 rc_scale,
    output logic                                                     rc_scale_vld
);
logic start_flag;
always_ff@(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        start_flag <= 0;
    end
    else if(start)begin
        start_flag <= 1;
    end
    else if(rc_scale_vld)begin
        start_flag <= 0;
    end
end

RC_CONFIG                                                     rc_cfg_reg;
always_ff@(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        rc_cfg_reg <= 0;
    end
    else if(rc_cfg_vld)begin
        rc_cfg_reg <= rc_cfg;
    end
end

CONTROL_STATE                                                        control_state_reg;
always_ff @(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        control_state_reg <= IDLE_STATE;
    end
    else if(control_state_update) begin
        control_state_reg <=  control_state;
    end
end

//SET K
logic [isize-1:0] K_ext; //always positive
logic                     [sig_width+exp_width : 0]                  float_K;
logic                     [sig_width+exp_width : 0]                  nxt_float_K;
assign K_ext = rc_cfg_reg.rms_K;

always_ff @(posedge clk)begin
    float_K <= nxt_float_K;
end

DW_fp_i2flt #(sig_width, exp_width, isize, isign)
i2flt_K ( 
    .a(K_ext), 
    .rnd(3'b000), 
    .z(nxt_float_K), 
    .status() 
);

logic        [sig_width+exp_width : 0]                    rms_dequant_scale_square_reg;

// assign rms_dequant_scale_square_reg = rc_cfg_reg.attn_rms_dequant_scale_square;
always_comb begin
    if(control_state_reg == FFN0_STATE)begin
        rms_dequant_scale_square_reg = rc_cfg_reg.mlp_rms_dequant_scale_square;
    end
    else begin
        rms_dequant_scale_square_reg = rc_cfg_reg.attn_rms_dequant_scale_square;
    end
end

////////////////////////
//Calculate the Square//
////////////////////////
logic                                                       fixed_data_square_vld;
logic signed [BUS_NUM-1:0][2*DATA_NUM_WIDTH-1:0]            fixed_data_square;

always_ff @(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        fixed_data_square_vld <= 0;
    end
    else begin
        fixed_data_square_vld <= in_fixed_data_vld & start_flag;
    end
end
genvar i; 
generate 
    for(i = 0; i < BUS_NUM; i++)begin : fixed_square_sum_generate_array
        always_ff @(posedge clk ) begin
            if(in_fixed_data_vld && start_flag)begin
                fixed_data_square[i] <= $signed(in_fixed_data[i]) * $signed(in_fixed_data[i]);
            end
            else begin
                fixed_data_square[i] <= 0;      
            end
        end
    end
endgenerate

////////////////////////////
//Calculate the Square Sum//
////////////////////////////
logic                     [FIXED_SQUARE_SUM_WIDTH-1:0]               fixed_square_sum;
logic                     [FIXED_SQUARE_SUM_WIDTH-1:0]               nxt_fixed_square_sum;
logic                     [DATA_NUM_WIDTH-1:0]                       square_sum_cnt;
logic                     [DATA_NUM_WIDTH-1:0]                       nxt_square_sum_cnt;
logic                                                                fixed_square_sum_vld;


always_comb begin
    nxt_square_sum_cnt = square_sum_cnt;
    nxt_fixed_square_sum = fixed_square_sum;
    for(int i = 0;i<BUS_NUM;i++)begin
        if(fixed_data_square_vld)begin
            nxt_square_sum_cnt = nxt_square_sum_cnt + 1;
            nxt_fixed_square_sum = nxt_fixed_square_sum + fixed_data_square[i];
            if(nxt_square_sum_cnt == rc_cfg_reg.rms_K)
                break;
        end
    end
end

always_ff @(posedge clk or negedge rst_n)begin
    if(~rst_n) begin
        square_sum_cnt <= 0;
        fixed_square_sum <= 0;
    end
    else if(rc_scale_vld)begin //RST to zero for next vector calcualtion
        square_sum_cnt <= 0;
        fixed_square_sum <= 0;
    end
    else if(square_sum_cnt == rc_cfg_reg.rms_K) //just update float_square_sum once 
        square_sum_cnt <= rc_cfg_reg.rms_K;
    else if(fixed_data_square_vld)begin
        square_sum_cnt <= nxt_square_sum_cnt;
        fixed_square_sum <= nxt_fixed_square_sum;
    end
end

always_ff @(posedge clk or negedge rst_n)begin
    if(~rst_n)
        fixed_square_sum_vld <= 0;
    else if(fixed_square_sum_vld == 1)
        fixed_square_sum_vld <= 0;
    else if(nxt_square_sum_cnt == rc_cfg_reg.rms_K && square_sum_cnt < rc_cfg_reg.rms_K && rc_cfg_reg.rms_K!=0) //fixed_square_sum_vld should be a pulse
        fixed_square_sum_vld <= 1;
end

////////////////////////////////////////
//Fixed square sum to Float square sum//
////////////////////////////////////////
logic                     [isize-1:0]                                fixed_square_sum_ext;
logic                     [sig_width+exp_width : 0]                  i2flt_square_sum_z;
logic                     [sig_width+exp_width : 0]                  flt_square_sum;
logic                                                                flt_square_sum_vld;//pulse

assign fixed_square_sum_ext = fixed_square_sum;

DW_fp_i2flt #(sig_width, exp_width, isize, isign)
i2flt_square_sum ( 
    .a(fixed_square_sum_ext), 
    .rnd(3'b000), 
    .z(i2flt_square_sum_z), 
    .status() 
);


fp_mult_pipe#(
    .sig_width(sig_width),
    .exp_width(exp_width),
    .ieee_compliance(ieee_compliance),//NaN and denormals
    .stages(5)
)inst_fp_mult_suqare_sum( 
    .clk(clk),
    .rst_n(rst_n),
    .a(i2flt_square_sum_z),
    .b(rms_dequant_scale_square_reg),
    .z(flt_square_sum),
    .ab_valid(fixed_square_sum_vld),
    .z_valid(flt_square_sum_vld) //pulse
);

///////////////////////////////
//Calculate Square Sum Over K//
///////////////////////////////
logic                     [sig_width+exp_width : 0]                  square_sum_over_K;
logic                                                                square_sum_over_K_vld;
fp_div_pipe#(
    .sig_width(sig_width),
    .exp_width(exp_width),
    .ieee_compliance(ieee_compliance),//NaN and denormals
    .stages(5)
) div_pipe_inst ( 
    .clk(clk),
    .rst_n(rst_n),
    .a(flt_square_sum),
    .b(float_K),
    .z(square_sum_over_K),
    .ab_valid(flt_square_sum_vld),
    .z_valid(square_sum_over_K_vld)
);

localparam SLICE_REG_NUM = `RMSNORM_RT_NUM;
logic [sig_width+exp_width : 0] square_sum_over_K_delay;
logic square_sum_over_K_vld_delay;
//retimg registers
generate
    for(i = 0;i < SLICE_REG_NUM;i++)begin : invsqrt_retiming_gen_array
        logic [sig_width+exp_width : 0] timing_register_float;
        logic timing_register;
        if(i == 0)begin
            always_ff@(posedge clk or negedge rst_n)begin
                if(~rst_n)begin
                    timing_register_float <= 0;
                    timing_register <= 0;
                end
                else begin
                    timing_register_float <= square_sum_over_K;
                    timing_register <= square_sum_over_K_vld;
                end
            end
        end
        else begin
            always_ff@(posedge clk or negedge rst_n)begin
                if(~rst_n) begin
                    timing_register_float <= 0;
                    timing_register <= 0;    
                end
                else begin
                    timing_register_float <= invsqrt_retiming_gen_array[i-1].timing_register_float;
                    timing_register <= invsqrt_retiming_gen_array[i-1].timing_register;                        
                end
            end
        end
    end
endgenerate

always_ff@(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        square_sum_over_K_delay <= 0;
        square_sum_over_K_vld_delay <= 0;
    end
    else begin
        square_sum_over_K_delay <= invsqrt_retiming_gen_array[SLICE_REG_NUM-1].timing_register_float;
        square_sum_over_K_vld_delay <= invsqrt_retiming_gen_array[SLICE_REG_NUM-1].timing_register;
    end
end

//////////////////////////
//Calculate One Over RMS//
//////////////////////////
logic                                                                invsqrt_z_vld;
logic                     [sig_width+exp_width : 0]                  invsqrt_z;

logic                     [sig_width+exp_width : 0]                  one_over_rms;
logic                                                                one_over_rms_vld;

fp_invsqrt_pipe inst_fp_invsqrt_pipe(
    .clk(clk),
    .rst_n(rst_n),
 
    .x(square_sum_over_K_delay), //input bfloat x, sign: 1, exp: 8, frac: 7
    .x_vld(square_sum_over_K_vld_delay),

    .y(invsqrt_z),
    .y_vld(invsqrt_z_vld)
);
//难绷，这个IP的误差比我自己写的还大似乎
//////////////////////////
//Calculate One Over RMS//
//////////////////////////
// assign invsqrt_z_vld = square_sum_over_K_vld_delay;
// DW_fp_invsqrt #(sig_width, exp_width, ieee_compliance)
// inst_invsqrt ( 
//     .a(square_sum_over_K_delay), 
//     .rnd(3'b000), 
//     .z(invsqrt_z), 
//     .status() 
// );

always_ff@(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        one_over_rms <= 0;
        one_over_rms_vld <= 0;
    end
    else begin
        one_over_rms <= invsqrt_z;
        one_over_rms_vld <= invsqrt_z_vld;
    end
end

///////////////////////////////////////////
//Calculate One Over RMS to Integer Scale//
///////////////////////////////////////////
logic                     [sig_width+exp_width : 0]                  one_over_rms_exp_shift;
logic                                                                one_over_rms_exp_shift_vld;
always_ff@(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        one_over_rms_exp_shift_vld <= 0;
        one_over_rms_exp_shift <= 0;
    end
    else begin
        one_over_rms_exp_shift_vld <= one_over_rms_vld;
        one_over_rms_exp_shift[sig_width+exp_width]      <= one_over_rms[sig_width+exp_width];
        one_over_rms_exp_shift[sig_width+exp_width-1: 0] <= one_over_rms[sig_width+exp_width-1: 0] + $unsigned({rc_cfg_reg.rms_rc_shift,{sig_width{1'b0}}}); 
    end
end

logic        [`RECOMPUTE_SCALE_WIDTH-1:0]                 nxt_rc_scale;

DW_fp_flt2i #(sig_width, exp_width, `RECOMPUTE_SCALE_WIDTH, isign)
    i2flt_in_data ( 
        .a(one_over_rms_exp_shift), 
        .rnd(3'b000), 
        .z(nxt_rc_scale), 
        .status() 
);

always_ff @(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        rc_scale <= 0;
        rc_scale_vld <= 0;
    end
    else begin
        rc_scale <= nxt_rc_scale;
        rc_scale_vld <= one_over_rms_exp_shift_vld;
    end
end
endmodule
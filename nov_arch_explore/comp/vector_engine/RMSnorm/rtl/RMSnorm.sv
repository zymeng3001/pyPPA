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



//Time sequence, first input data num, in_scale_pos, out_scale_pos and gamma array(This can be done when initialize the whole network)
//Input #DATA_NUM of inputs and wait for the output valid signal to be high
//When the last signal is high, then the next vector input
//The input valid siganl doesn't need to be continuous

//when input gamma and x, the valid bits must be all 1 except the last input
module rms_norm
#(
    parameter integer BUS_NUM = 8, //input bus num
    parameter integer DATA_NUM_WIDTH =  10, //layernorm vector length reg width
    parameter integer SCALA_POS_WIDTH = 5,
    parameter integer FIXED_SQUARE_SUM_WIDTH = 24,
    parameter integer IN_FLOAT_DATA_ARRAY_DEPTH = (384 / BUS_NUM + 1), //Every slot has #BUS_NUM numbers
    parameter integer GAMMA_ARRAY_DEPTH = IN_FLOAT_DATA_ARRAY_DEPTH, //Every slot has #BUS_NUM numbers

    parameter integer sig_width = 7,
    parameter integer exp_width = 8,
    localparam        isize = 32,
    localparam        isign = 1, //signed integer number
    localparam        ieee_compliance = 0 //No support to NaN adn denormals
)
(
    input  logic                                                     clk,
    input  logic                                                     rst_n,

    //input channel     
    input  logic                     [DATA_NUM_WIDTH-1:0]            in_data_num,
    input  logic                                                     in_data_num_vld,

    input  logic        [BUS_NUM-1:0][sig_width+exp_width : 0]       in_gamma,
    input  logic        [BUS_NUM-1:0]                                in_gamma_vld,

    input  logic signed [BUS_NUM-1:0][7:0]                           in_fixed_data, //Here signed seems not work!!! Guess: signed now points to BUS_NUM
    input  logic        [BUS_NUM-1:0]                                in_fixed_data_vld,

    input  logic signed              [SCALA_POS_WIDTH-1:0]           in_scale_pos,  //-16 ~ 15
    input  logic                                                     in_scale_pos_vld, 

    input  logic signed              [SCALA_POS_WIDTH-1:0]           out_scale_pos,  
    input  logic                                                     out_scale_pos_vld,

    input  logic                     [DATA_NUM_WIDTH-1:0]            in_K,
    input  logic                                                     in_K_vld,

    output logic signed [BUS_NUM-1:0][7:0]                           out_fixed_data,
    output logic        [BUS_NUM-1:0]                                out_fixed_data_vld,
    output logic                                                     out_fixed_data_last //This signal is to help to design more efficiently
);      

logic                     [DATA_NUM_WIDTH-1:0]                       data_num;
// logic                     [sig_width+exp_width : 0]                  float_data_num;
// logic                     [sig_width+exp_width : 0]                  nxt_float_data_num;

logic                     [DATA_NUM_WIDTH-1:0]                       K;
logic                     [sig_width+exp_width : 0]                  float_K;
logic                     [sig_width+exp_width : 0]                  nxt_float_K;

logic signed              [SCALA_POS_WIDTH-1:0]                      in_scale_pos_reg;
logic signed              [SCALA_POS_WIDTH-1:0]                      out_scale_pos_reg;

logic signed [BUS_NUM-1:0][2*DATA_NUM_WIDTH-1:0]                     fixed_data_square;
logic        [BUS_NUM-1:0]                                           fixed_data_square_vld;
// logic        [BUS_NUM-1:0]                                           square_K_vld

logic                     [FIXED_SQUARE_SUM_WIDTH-1:0]               fixed_square_sum;
logic                     [FIXED_SQUARE_SUM_WIDTH-1:0]               nxt_fixed_square_sum;
logic                     [DATA_NUM_WIDTH-1:0]                       square_sum_cnt;
logic                     [DATA_NUM_WIDTH-1:0]                       nxt_square_sum_cnt;
logic                                                                fixed_square_sum_vld;


logic                     [sig_width+exp_width : 0]                  i2flt_square_sum_z;
logic                     [sig_width+exp_width : 0]                  float_square_sum;
logic                                                                float_square_sum_vld;

logic                     [sig_width+exp_width : 0]                  square_sum_over_K;
logic                                                                square_sum_over_K_vld;

logic                     [sig_width+exp_width : 0]                  one_over_rms;
logic                     [sig_width+exp_width : 0]                  invsqrt_z;
logic                                                                one_over_rms_vld;

logic        [BUS_NUM-1:0][sig_width+exp_width : 0]                  i2flt_in_data_z;
logic        [BUS_NUM-1:0][sig_width+exp_width : 0]                  in_float_data;
logic        [BUS_NUM-1:0]                                           in_float_data_vld;

logic                     [(sig_width+exp_width+1)*BUS_NUM-1: 0]     in_float_data_array [IN_FLOAT_DATA_ARRAY_DEPTH-1:0];//This can be viewed as fifo
logic                     [BUS_NUM-1:0]                              in_float_data_vld_array [IN_FLOAT_DATA_ARRAY_DEPTH-1:0];
logic                     [(sig_width+exp_width+1)*BUS_NUM-1: 0]     in_float_data_array_wr_slot;
logic                     [(sig_width+exp_width+1)*BUS_NUM-1: 0]     in_float_data_array_rd_slot;
logic                     [BUS_NUM-1:0]                              in_float_data_vld_array_slot; 
logic                     [$clog2(IN_FLOAT_DATA_ARRAY_DEPTH)-1:0]    in_float_data_array_wr_ptr;
logic                     [$clog2(IN_FLOAT_DATA_ARRAY_DEPTH)-1:0]    in_float_data_array_rd_ptr;

logic                     [(sig_width+exp_width+1)*BUS_NUM-1: 0]     gamma_array [GAMMA_ARRAY_DEPTH-1:0];
logic                     [(sig_width+exp_width+1)*BUS_NUM-1: 0]     gamma_array_wr_slot;
logic                     [(sig_width+exp_width+1)*BUS_NUM-1: 0]     gamma_array_rd_slot;
logic                     [BUS_NUM-1:0]                              gamma_vld_array [GAMMA_ARRAY_DEPTH-1:0];
logic                     [$clog2(GAMMA_ARRAY_DEPTH)-1:0]            gamma_array_wr_ptr;
logic                     [$clog2(GAMMA_ARRAY_DEPTH)-1:0]            gamma_array_rd_ptr;

logic                                                                internal_array_rd_en;
logic                                                                internal_array_rd_en_delay1;
logic                     [$clog2(GAMMA_ARRAY_DEPTH*BUS_NUM)-1:0]    internal_array_rd_cnt;

logic        [BUS_NUM-1:0][sig_width+exp_width : 0]                  x_float;
logic        [BUS_NUM-1:0][sig_width+exp_width : 0]                  gamma_float;
logic        [BUS_NUM-1:0]                                           x_and_gamma_vld;//x valid and gamma valid should be same

logic        [BUS_NUM-1:0][sig_width+exp_width : 0]                  x_mult_with_gamma;
logic        [BUS_NUM-1:0]                                           x_mult_with_gamma_vld;

logic        [BUS_NUM-1:0][sig_width+exp_width : 0]                  float_RMSnorm;
logic        [BUS_NUM-1:0]                                           float_RMSnorm_vld;

logic        [BUS_NUM-1:0][sig_width+exp_width : 0]                  float_RMSnorm_flt2i;
logic        [BUS_NUM-1:0]                                           float_RMSnorm_flt2i_vld;

logic signed [BUS_NUM-1:0][isize-1:0]                                fixed_RMSnorm;
logic signed [BUS_NUM-1:0][7:0]                                      nxt_out_fixed_data;
logic                     [15:0]                                     out_fixed_data_last_cnt;


/////////////////////////////////////////////////////
//First set the data num (length of RMSnorm vector)//
//and scale position and gamma array               //
/////////////////////////////////////////////////////
always_ff @(posedge clk or negedge rst_n) begin
    if(~rst_n)
        data_num <=0;
    else if(in_data_num_vld)
        data_num <= in_data_num;
end

always_ff @(posedge clk or negedge rst_n) begin
    if(~rst_n)
        in_scale_pos_reg <=0;
    else if(in_scale_pos_vld)
        in_scale_pos_reg <= in_scale_pos;
end

always_ff @(posedge clk or negedge rst_n) begin
    if(~rst_n)
        out_scale_pos_reg <=0;
    else if(out_scale_pos_vld)
        out_scale_pos_reg <= out_scale_pos;
end


generate
    genvar i;
    for(i = 0; i <BUS_NUM; i++)begin : gamma_array_wr_slot_generate_array
        assign gamma_array_wr_slot[i*(sig_width+exp_width+1)+:(sig_width+exp_width+1)] = in_gamma[i];
    end
endgenerate

always_ff @(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        gamma_array_wr_ptr <= 0;
    end
    else if(out_fixed_data_last)begin
        gamma_array_wr_ptr <= 0; //RST to zero for next vector input
    end
    else if(|in_gamma_vld)begin //At least one valid input
        gamma_array[gamma_array_wr_ptr] <= gamma_array_wr_slot;
        gamma_vld_array[gamma_array_wr_ptr] <= in_gamma_vld;
        gamma_array_wr_ptr <= gamma_array_wr_ptr + 1;
    end
end

/////////////////////////////////////////////////////
//Before each vectore computation, we can set the K//
/////////////////////////////////////////////////////
logic [isize-1:0] K_ext; //always positive
assign K_ext = K;

always @(posedge clk or negedge rst_n) begin
    if(~rst_n)begin
        K <= 0;
    end
    else if (in_K_vld)begin
        K <= in_K;
    end
end

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


//////////////////////////////////////////
//Convert input fixed data to float data//
//And store them in array               //
//////////////////////////////////////////
logic signed [BUS_NUM-1:0][isize-1:0] in_fixed_data_ext;

generate
    // genvar i;
    for(i = 0;i < BUS_NUM;i++)begin : i2flt_in_data_generate_array
        assign in_fixed_data_ext[i] = $signed(in_fixed_data[i]);
        DW_fp_i2flt #(sig_width, exp_width, isize, isign)
        i2flt_in_data ( 
            .a($signed(in_fixed_data_ext[i])), 
            .rnd(3'b000), 
            .z(i2flt_in_data_z[i]), 
            .status() 
        );
    end
endgenerate


generate 
    // genvar i; //i is a cute number
    for(i = 0;i < BUS_NUM; i++)begin :  in_fix2float_generate_array
        always_ff @(posedge clk or negedge rst_n)begin
            if(~rst_n)begin
                in_float_data_vld[i] <= 0;
                in_float_data[i] <= 0;
            end
            else begin
                in_float_data_vld[i] <= in_fixed_data_vld[i];
                if(i2flt_in_data_z[i] == 0)begin
                    in_float_data[i] <= 0;
                end
                else begin
                    in_float_data[i][sig_width+exp_width-1: 0] <= i2flt_in_data_z[i][sig_width+exp_width-1: 0] - $signed({in_scale_pos_reg,{sig_width{1'b0}}}); // i2flt_z_inst * 2^(-in_scale_pos_reg) ;
                    in_float_data[i][sig_width+exp_width] <= i2flt_in_data_z[i][sig_width+exp_width];
                end
            end
        end
        assign in_float_data_array_wr_slot[i*(sig_width+exp_width+1)+:(sig_width+exp_width+1)] = in_float_data[i];
        assign in_float_data_vld_array_slot[i] = in_float_data_vld[i];
    end
endgenerate

always_ff @(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        in_float_data_array_wr_ptr <= 0;
    end
    else if(out_fixed_data_last)begin
        in_float_data_array_wr_ptr <= 0; //RST to zero for next vector input
    end
    else if(|in_float_data_vld)begin //At least one valid input
        in_float_data_array[in_float_data_array_wr_ptr] <= in_float_data_array_wr_slot;
        in_float_data_vld_array[in_float_data_array_wr_ptr] <= in_float_data_vld_array_slot;
        in_float_data_array_wr_ptr <= in_float_data_array_wr_ptr + 1;
    end

end
////////////////////////
//Calculate the Square//
////////////////////////
logic fixed_data_square_input_gating; //save power
always_ff@(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        fixed_data_square_input_gating <= 0;
    end
    else if(out_fixed_data_last)begin
        fixed_data_square_input_gating <= 0;
    end
    else if(fixed_square_sum_vld)begin
        fixed_data_square_input_gating <= 1;
    end
end

generate 
    // genvar i; //i is a cute number
    for(i = 0; i < BUS_NUM; i++)begin : fixed_square_sum_generate_array
        always_ff @(posedge clk ) begin
            if(~fixed_data_square_input_gating)begin
                fixed_data_square[i] <= $signed(in_fixed_data[i]) * $signed(in_fixed_data[i]);
                fixed_data_square_vld[i] <= in_fixed_data_vld[i];                
            end
            else begin
                fixed_data_square[i] <= 0;
                fixed_data_square_vld[i] <= 0;                
            end
        end
    end
endgenerate


////////////////////////////
//Calculate the Square Sum//
////////////////////////////
always_comb begin
    nxt_square_sum_cnt = square_sum_cnt;
    nxt_fixed_square_sum = fixed_square_sum;
    for(int i = 0;i<BUS_NUM;i++)begin
        if(fixed_data_square_vld[i])begin
            nxt_square_sum_cnt = nxt_square_sum_cnt + 1;
            nxt_fixed_square_sum = nxt_fixed_square_sum + fixed_data_square[i];
            if(nxt_square_sum_cnt == K)
                break;
        end
    end
end

always_ff @(posedge clk or negedge rst_n)begin
    if(~rst_n) begin
        square_sum_cnt <= 0;
        fixed_square_sum <= 0;
    end
    else if(out_fixed_data_last)begin //RST to zero for next vector calcualtion
        square_sum_cnt <= 0;
        fixed_square_sum <= 0;
    end
    else if(square_sum_cnt == K) //just update float_square_sum once
        square_sum_cnt <= K;
    else if(|fixed_data_square_vld)begin
        square_sum_cnt <= nxt_square_sum_cnt;
        fixed_square_sum <= nxt_fixed_square_sum;
    end
end

always_ff @(posedge clk or negedge rst_n)begin
    if(~rst_n)
        fixed_square_sum_vld <= 0;
    else if(fixed_square_sum_vld == 1)
        fixed_square_sum_vld <= 0;
    else if(nxt_square_sum_cnt == K && square_sum_cnt < K && K!=0) //fixed_square_sum_vld should be a pulse
        fixed_square_sum_vld <= 1;
end

////////////////////////////////////////
//Fixed square sum to Float square sum//
////////////////////////////////////////
logic    [isize-1:0]  fixed_square_sum_ext;//always positive
assign fixed_square_sum_ext = fixed_square_sum;
DW_fp_i2flt #(sig_width, exp_width, isize, isign)
i2flt_square_sum ( 
    .a(fixed_square_sum_ext), 
    .rnd(3'b000), 
    .z(i2flt_square_sum_z), 
    .status() 
);

always_ff @(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        float_square_sum <= 0;
        float_square_sum_vld <= 0;
    end
    else if(out_fixed_data_last) begin //RST for next vector calculation
        float_square_sum_vld <= 0;
        float_square_sum <= 0;
    end
    else if(fixed_square_sum_vld)begin
        float_square_sum_vld <= 1;
        if(i2flt_square_sum_z == 0)begin //float zero
            float_square_sum <= 0;
        end
        else begin
            float_square_sum[sig_width+exp_width-1: 0] <= i2flt_square_sum_z[sig_width+exp_width-1: 0] - $signed({in_scale_pos_reg,1'b0,{sig_width{1'b0}}}); // i2flt_z_inst * 2^(-2*in_scale_pos_reg)
            float_square_sum[sig_width+exp_width] <= i2flt_square_sum_z[sig_width+exp_width];
        end
    end
    else
        float_square_sum_vld <= 0;
end


///////////////////////////////
//Calculate Square Sum Over K//
///////////////////////////////
fp_div_pipe#(
    .sig_width(sig_width),
    .exp_width(exp_width),
    .ieee_compliance(ieee_compliance),//NaN and denormals
    .stages(4)
) div_pipe_inst ( 
    .clk(clk),
    .rst_n(rst_n),
    .a(float_square_sum),
    .b(float_K),
    .z(square_sum_over_K),
    .ab_valid(float_square_sum_vld),
    .z_valid(square_sum_over_K_vld)
);
localparam SLICE_REG_NUM = 1;
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
// DW_fp_invsqrt #(sig_width, exp_width, ieee_compliance)
// inst_invsqrt ( 
//     .a(square_sum_over_K_delay), 
//     .rnd(3'b001), 
//     .z(invsqrt_z), 
//     .status() 
// );

logic invsqrt_z_vld;

fp_invsqrt_pipe inst_fp_invsqrt_pipe(
    .clk(clk),
    .rst_n(rst_n),
 
    .x(square_sum_over_K_delay), //input bfloat x, sign: 1, exp: 8, frac: 7
    .x_vld(square_sum_over_K_vld_delay),

    .y(invsqrt_z),
    .y_vld(invsqrt_z_vld)
);

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

///////////////////////////////
//Calculate x mult with gamma//
///////////////////////////////

always_ff@(posedge clk or negedge rst_n)begin
    if(~rst_n) begin
        internal_array_rd_en <= 0;
    end
    else if(one_over_rms_vld)
        internal_array_rd_en <= 1;
    else if(internal_array_rd_cnt >= data_num-BUS_NUM && data_num !=0 && internal_array_rd_en)begin
        internal_array_rd_en <= 0;
    end
end

always_ff @(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        internal_array_rd_cnt <= 0;
    end
    else if (out_fixed_data_last)begin
        internal_array_rd_cnt <= 0; //RST to 0 for next vector input
    end
    else if(internal_array_rd_en)begin
        internal_array_rd_cnt <= internal_array_rd_cnt + BUS_NUM;
    end
end

always_ff @(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        in_float_data_array_rd_ptr <= 0;
        gamma_array_rd_ptr <= 0;
        in_float_data_array_rd_slot <= 0;
        gamma_array_rd_slot <= 0;
    end
    else if(out_fixed_data_last) begin
        in_float_data_array_rd_ptr <= 0;
        gamma_array_rd_ptr <= 0;
    end
    else if(internal_array_rd_en) begin
        in_float_data_array_rd_ptr <= in_float_data_array_rd_ptr + 1;
        gamma_array_rd_ptr <= gamma_array_rd_ptr + 1;
        in_float_data_array_rd_slot <= in_float_data_array[in_float_data_array_rd_ptr];
        gamma_array_rd_slot <= gamma_array[gamma_array_rd_ptr];
    end
end

always_ff @(posedge clk or negedge rst_n)begin
    if(~rst_n) begin
        x_and_gamma_vld <= 0;
    end
    else begin
        x_and_gamma_vld <= in_float_data_vld_array[in_float_data_array_rd_ptr] & {BUS_NUM{internal_array_rd_en}};
    end
end
generate
    // genvar i;
    for(i = 0;i < BUS_NUM; i++)begin : x_gamma_float_generate_array
        assign x_float[i] = in_float_data_array_rd_slot[i*(sig_width+exp_width+1)+:(sig_width+exp_width+1)];
        assign gamma_float[i] = gamma_array_rd_slot[i*(sig_width+exp_width+1)+:(sig_width+exp_width+1)];
    end
endgenerate

generate
// genvar i;
    for (i =0; i <BUS_NUM; i++)begin : x_mult_with_gamma_generate_array
        fp_mult_pipe#(
            .sig_width(sig_width),
            .exp_width(exp_width),
            .ieee_compliance(ieee_compliance),//NaN and denormals
            .stages(3)
        )inst_fp_mult_x_gamma( 
            .clk(clk),
            .rst_n(rst_n),
            .a(x_float[i]),
            .b(gamma_float[i]),
            .z(x_mult_with_gamma[i]),
            .ab_valid(x_and_gamma_vld[i]),
            .z_valid(x_mult_with_gamma_vld[i])
        );
    end
endgenerate

///////////////////////////
//Calculate float RMSnorm//
///////////////////////////
generate
// genvar i;
    for (i =0; i <BUS_NUM; i++)begin : float_RMSnorm_generate_array
        fp_mult_pipe#(
            .sig_width(sig_width),
            .exp_width(exp_width),
            .ieee_compliance(ieee_compliance),//NaN and denormals
            .stages(3)
        )inst_fp_mult_float_RMSnorm( 
            .clk(clk),
            .rst_n(rst_n),
            .a(x_mult_with_gamma[i]),
            .b(one_over_rms),
            .z(float_RMSnorm[i]),
            .ab_valid(x_mult_with_gamma_vld[i]),
            .z_valid(float_RMSnorm_vld[i])
        );
    end
endgenerate

///////////////////////////
//Calculate fixed RMSnorm//
///////////////////////////
generate
    // genvar i;
    for(i = 0;i < BUS_NUM; i++)begin : fixed_RMSnorm_generate_array
        always_ff @(posedge clk or negedge rst_n)begin
            if(~rst_n)begin
                float_RMSnorm_flt2i[i] <= 0;
                float_RMSnorm_flt2i_vld[i] <= 0;
            end
            else begin
                if(float_RMSnorm == 0)begin
                    float_RMSnorm_flt2i[i] <= 0;
                end
                else begin
                    float_RMSnorm_flt2i[i][sig_width+exp_width-1: 0] <= float_RMSnorm[i][sig_width+exp_width-1: 0] + $signed({out_scale_pos_reg,{sig_width{1'b0}}});// float_RMSnorm * 2^(out_scale_pos_reg)
                    float_RMSnorm_flt2i[i][sig_width+exp_width] <= float_RMSnorm[i][sig_width+exp_width];    
                end
                float_RMSnorm_flt2i_vld[i] <= float_RMSnorm_vld[i];
            end
        end

    
        DW_fp_flt2i #(sig_width, exp_width, isize, isign)
            i2flt_in_data ( 
                .a(float_RMSnorm_flt2i[i]), 
                .rnd(3'b000), 
                .z(fixed_RMSnorm[i]), 
                .status() 
        );

        always_comb begin
            nxt_out_fixed_data[i] = fixed_RMSnorm[i];
            if($signed(fixed_RMSnorm[i]) > 127) begin
                nxt_out_fixed_data[i] = 8'b0111_1111; //127
            end
            else if($signed(fixed_RMSnorm[i]) < -128) begin
                nxt_out_fixed_data[i] = 8'b1000_0000; //-128
            end
        end

        always_ff@ (posedge clk or negedge rst_n)begin
            if(~rst_n)begin
                out_fixed_data_vld[i] <= 0;
                out_fixed_data[i] <= 0;
            end
            else begin
                out_fixed_data_vld[i] <= float_RMSnorm_flt2i_vld[i];
                out_fixed_data[i] <= nxt_out_fixed_data[i];
            end
        end

    end
endgenerate


always_ff@ (posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        out_fixed_data_last_cnt <= 0;
        out_fixed_data_last <= 0;
    end
    else if(out_fixed_data_last_cnt >= data_num -BUS_NUM && data_num != 0 && |float_RMSnorm_flt2i_vld)begin
        out_fixed_data_last <= 1;
        out_fixed_data_last_cnt <= 0;
    end
    else if(|float_RMSnorm_flt2i_vld) begin
        out_fixed_data_last_cnt <= out_fixed_data_last_cnt + BUS_NUM;
    end
    else begin
        out_fixed_data_last <= 0;
    end
end

endmodule
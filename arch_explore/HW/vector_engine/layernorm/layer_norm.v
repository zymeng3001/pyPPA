module layer_norm #(
    parameter integer BUS_NUM = 8,
    parameter integer DATA_NUM_WIDTH = 10,
    parameter integer SCALA_POS_WIDTH = 5,
    parameter integer FIXED_ACC_WIDTH = 32,
    parameter integer IN_FLOAT_DATA_ARRAY_DEPTH = (${n_embd} / BUS_NUM + 1), 
    parameter integer sig_width = 7,
    parameter integer exp_width = 8,
    parameter integer isize     = 32,
    parameter integer isign     = 1,
    parameter integer ieee_compliance = 0
)(
    input clk,
    input rst_n,

    input [DATA_NUM_WIDTH-1:0] in_data_num,
    input                      in_data_num_vld,

    input signed [BUS_NUM*8-1:0] in_fixed_data,
    input [BUS_NUM-1:0] in_fixed_data_vld,

    input [BUS_NUM*(sig_width+exp_width+1)-1:0] in_gamma,
    input [BUS_NUM*(sig_width+exp_width+1)-1:0] in_beta,
    input [BUS_NUM-1:0] in_gamma_vld,
    input [BUS_NUM-1:0] in_beta_vld,

    input signed [SCALA_POS_WIDTH-1:0] in_scale_pos,
    input                              in_scale_pos_vld,

    input signed [SCALA_POS_WIDTH-1:0] out_scale_pos,
    input                              out_scale_pos_vld,

    output reg signed [BUS_NUM*8-1:0] out_fixed_data,
    output reg [BUS_NUM-1:0] out_fixed_data_vld,
    output reg out_fixed_data_last
);

reg [DATA_NUM_WIDTH-1:0] data_num;

reg signed [SCALA_POS_WIDTH-1:0] in_scale_pos_reg;
reg signed [SCALA_POS_WIDTH-1:0] out_scale_pos_reg;

// data_num register
always @(posedge clk or negedge rst_n) begin
if (!rst_n)
    data_num <= 0;
else if (in_data_num_vld)
    data_num <= in_data_num;
end

// in_scale_pos_reg
always @(posedge clk or negedge rst_n) begin
if (!rst_n)
    in_scale_pos_reg <= 0;
else if (in_scale_pos_vld)
    in_scale_pos_reg <= in_scale_pos;
end

// out_scale_pos_reg
always @(posedge clk or negedge rst_n) begin
if (!rst_n)
    out_scale_pos_reg <= 0;
else if (out_scale_pos_vld)
    out_scale_pos_reg <= out_scale_pos;
end





endmodule
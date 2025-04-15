module layer_norm #(
    parameter integer BUS_NUM = 8,
    parameter integer DATA_NUM_WIDTH = 10,
    parameter integer SCALA_POS_WIDTH = 5,
    parameter integer FIXED_ACC_WIDTH = 32,
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



endmodule
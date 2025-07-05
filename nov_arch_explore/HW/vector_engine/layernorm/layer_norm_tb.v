`timescale 1ns / 1ps

module layer_norm_tb;

  // Parameters
  parameter BUS_NUM = 8;
  parameter DATA_NUM_WIDTH = 10;
  parameter SCALA_POS_WIDTH = 5;
  parameter FIXED_ACC_WIDTH = 32;
  parameter IN_FIXED_DATA_ARRAY_DEPTH = 10;
  parameter sig_width = 7;
  parameter exp_width = 8;
  parameter isize = 32;
  parameter isign = 1;
  parameter ieee_compliance = 0;

  // Clock and reset
  reg clk;
  reg rstn;

  // Inputs
  reg [DATA_NUM_WIDTH-1:0] in_data_num;
  reg in_data_num_vld;
  reg signed [BUS_NUM*8-1:0] in_fixed_data;
  reg in_fixed_data_vld;
  reg [BUS_NUM*(sig_width+exp_width+1)-1:0] in_gamma;
  reg [BUS_NUM*(sig_width+exp_width+1)-1:0] in_beta;
  reg in_gamma_vld;
  reg in_beta_vld;
  reg signed [SCALA_POS_WIDTH-1:0] in_scale_pos;
  reg in_scale_pos_vld;
  reg signed [SCALA_POS_WIDTH-1:0] out_scale_pos;
  reg out_scale_pos_vld;

  // Outputs
  wire signed [BUS_NUM*8-1:0] out_fixed_data;
  wire out_fixed_data_vld;
  wire out_fixed_data_last;

  // Instantiate DUT
  layer_norm #(
    .BUS_NUM(BUS_NUM),
    .DATA_NUM_WIDTH(DATA_NUM_WIDTH),
    .SCALA_POS_WIDTH(SCALA_POS_WIDTH),
    .FIXED_ACC_WIDTH(FIXED_ACC_WIDTH),
    .IN_FIXED_DATA_ARRAY_DEPTH(IN_FIXED_DATA_ARRAY_DEPTH),
    .sig_width(sig_width),
    .exp_width(exp_width),
    .isize(isize),
    .isign(isign),
    .ieee_compliance(ieee_compliance)
  ) dut (
    .clk(clk),
    .rstn(rstn),
    .in_data_num(in_data_num),
    .in_data_num_vld(in_data_num_vld),
    .in_fixed_data(in_fixed_data),
    .in_fixed_data_vld(in_fixed_data_vld),
    .in_gamma(in_gamma),
    .in_beta(in_beta),
    .in_gamma_vld(in_gamma_vld),
    .in_beta_vld(in_beta_vld),
    .in_scale_pos(in_scale_pos),
    .in_scale_pos_vld(in_scale_pos_vld),
    .out_scale_pos(out_scale_pos),
    .out_scale_pos_vld(out_scale_pos_vld),
    .out_fixed_data(out_fixed_data),
    .out_fixed_data_vld(out_fixed_data_vld),
    .out_fixed_data_last(out_fixed_data_last)
  );

  // Clock generation
  initial clk = 0;
  always #5 clk = ~clk;

  // Stimulus
  initial begin
    // Reset
    rstn = 0;
    #20 rstn = 1;

    // Initial values
    in_data_num = 32;
    in_data_num_vld = 1;
    in_fixed_data = 64'h0102030405060708;
    in_fixed_data_vld = 1;
    in_gamma = {128{1'b1}}; // simple test value
    in_beta = 128'h01010101010101010202020202020202; // simple test value
    in_gamma_vld = 1;
    in_beta_vld = 1;
    in_scale_pos = 0;
    in_scale_pos_vld = 1;
    out_scale_pos = 0;
    out_scale_pos_vld = 1;

    #50;
    in_data_num_vld = 0;
    in_fixed_data_vld = 0;
    in_gamma_vld = 0;
    in_beta_vld = 0;
    in_scale_pos_vld = 0;
    out_scale_pos_vld = 0;

    // Wait for outputs
    #500;
    $stop;
  end

endmodule

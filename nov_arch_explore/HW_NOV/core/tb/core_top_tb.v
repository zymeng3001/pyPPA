`timescale 1ns / 1ps

module tb_core_top;

  // Parameters
  localparam GBUS_DATA_WIDTH = 32;
  localparam GBUS_ADDR_WIDTH = 19;
  localparam CMEM_ADDR_WIDTH = 13;
  localparam CMEM_DATA_WIDTH = 128;
  localparam VLINK_DATA_WIDTH = 128;
  localparam HLINK_DATA_WIDTH = 128;

  // Clock and reset
  reg clk;
  reg rstn;

  // DUT I/Os
  reg clean_kv_cache;
  reg [1:0] clean_kv_cache_user_id;
  reg [13:0] core_mem_addr;
  reg [15:0] core_mem_wdata;
  reg core_mem_wen;
  wire [15:0] core_mem_rdata;
  reg core_mem_ren;
  wire core_mem_rvld;
  reg op_cfg_vld;
  reg [40:0] op_cfg;
  reg usr_cfg_vld;
  reg [11:0] usr_cfg;
  reg model_cfg_vld;
  reg [29:0] model_cfg;
  reg pmu_cfg_vld;
  reg [3:0] pmu_cfg;
  reg rc_cfg_vld;
  reg [83:0] rc_cfg;
  reg [31:0] control_state;
  reg control_state_update;
  reg start;
  wire finish;
  reg [(2 + 4 + 13)-1:0] in_gbus_addr;
  reg in_gbus_wen;
  reg [GBUS_DATA_WIDTH - 1:0] in_gbus_wdata;
  wire [(2 + 4 + 13)-1:0] out_gbus_addr;
  wire out_gbus_wen;
  wire [GBUS_DATA_WIDTH - 1:0] out_gbus_wdata;
  reg [23:0] rc_scale;
  reg rc_scale_vld;
  reg rc_scale_clear;
  reg [VLINK_DATA_WIDTH - 1:0] vlink_data_in;
  reg vlink_data_in_vld;
  wire [VLINK_DATA_WIDTH - 1:0] vlink_data_out;
  wire vlink_data_out_vld;
  reg [HLINK_DATA_WIDTH - 1:0] hlink_wdata;
  reg hlink_wen;
  wire [HLINK_DATA_WIDTH - 1:0] hlink_rdata;
  wire hlink_rvalid;

  // Clock generation
  initial clk = 0;
  always #5 clk = ~clk;  // 100 MHz

  // DUT instantiation
  core_top dut (
    .clk(clk),
    .rstn(rstn),
    .clean_kv_cache(clean_kv_cache),
    .clean_kv_cache_user_id(clean_kv_cache_user_id),
    .core_mem_addr(core_mem_addr),
    .core_mem_wdata(core_mem_wdata),
    .core_mem_wen(core_mem_wen),
    .core_mem_rdata(core_mem_rdata),
    .core_mem_ren(core_mem_ren),
    .core_mem_rvld(core_mem_rvld),
    .op_cfg_vld(op_cfg_vld),
    .op_cfg(op_cfg),
    .usr_cfg_vld(usr_cfg_vld),
    .usr_cfg(usr_cfg),
    .model_cfg_vld(model_cfg_vld),
    .model_cfg(model_cfg),
    .pmu_cfg_vld(pmu_cfg_vld),
    .pmu_cfg(pmu_cfg),
    .rc_cfg_vld(rc_cfg_vld),
    .rc_cfg(rc_cfg),
    .control_state(control_state),
    .control_state_update(control_state_update),
    .start(start),
    .finish(finish),
    .in_gbus_addr(in_gbus_addr),
    .in_gbus_wen(in_gbus_wen),
    .in_gbus_wdata(in_gbus_wdata),
    .out_gbus_addr(out_gbus_addr),
    .out_gbus_wen(out_gbus_wen),
    .out_gbus_wdata(out_gbus_wdata),
    .rc_scale(rc_scale),
    .rc_scale_vld(rc_scale_vld),
    .rc_scale_clear(rc_scale_clear),
    .vlink_data_in(vlink_data_in),
    .vlink_data_in_vld(vlink_data_in_vld),
    .vlink_data_out(vlink_data_out),
    .vlink_data_out_vld(vlink_data_out_vld),
    .hlink_wdata(hlink_wdata),
    .hlink_wen(hlink_wen),
    .hlink_rdata(hlink_rdata),
    .hlink_rvalid(hlink_rvalid)
  );

  // Test procedure
  initial begin
    $dumpfile("core_top_tb.vcd");       
    $dumpvars(0, tb_core_top);  
    // Initialize signals
    rstn = 0;
    clean_kv_cache = 0;
    clean_kv_cache_user_id = 0;
    core_mem_addr = 0;
    core_mem_wdata = 0;
    core_mem_wen = 0;
    core_mem_ren = 0;
    op_cfg_vld = 0;
    op_cfg = 41'h123456789A;
    usr_cfg_vld = 0;
    usr_cfg = 12'habc;
    model_cfg_vld = 0;
    model_cfg = 30'h3FFFFFFF;
    pmu_cfg_vld = 0;
    pmu_cfg = 4'hA;
    rc_cfg_vld = 0;
    rc_cfg = 84'h0123456789ABCDEF012;
    control_state = 0;
    control_state_update = 0;
    start = 0;
    in_gbus_addr = 0;
    in_gbus_wen = 0;
    in_gbus_wdata = 0;
    rc_scale = 0;
    rc_scale_vld = 0;
    rc_scale_clear = 0;
    vlink_data_in = 0;
    vlink_data_in_vld = 0;
    hlink_wdata = 0;
    hlink_wen = 0;

    // Reset sequence
    #20;
    rstn = 1;

    // Load config
    #10 op_cfg_vld = 1;
    #10 op_cfg_vld = 0;
    #10 usr_cfg_vld = 1;
    #10 usr_cfg_vld = 0;
    #10 model_cfg_vld = 1;
    #10 model_cfg_vld = 0;
    #10 pmu_cfg_vld = 1;
    #10 pmu_cfg_vld = 0;
    #10 rc_cfg_vld = 1;
    #10 rc_cfg_vld = 0;

    // Start operation
    #20;
    start = 1;
    #10;
    start = 0;

    // Wait for finish
    wait(finish);
    $display("Core finished execution at time %t", $time);

    #50 $finish;
  end

endmodule

`include "sys_defs.svh"
`timescale 1ns/1ps

module Q_GEN_tb;
  // Clock & reset
  reg clk = 0;
  reg rst_n = 0;
  localparam CLK_PERIOD = 10;
  always #(CLK_PERIOD/2) clk = ~clk;

  // Interface to array_top
  reg [21:0] interface_addr;
  reg interface_wen;
  reg [15:0] interface_wdata;
  reg interface_ren;
  wire [15:0] interface_rdata;
  wire interface_rvalid;

  wire current_token_finish_flag;
  wire current_token_finish_work;
  wire qgen_state_work;
  wire qgen_state_end;
  wire kgen_state_work;
  wire kgen_state_end;
  wire vgen_state_work;
  wire vgen_state_end;
  wire att_qk_state_work;
  wire att_qk_state_end;
  wire att_pv_state_work;
  wire att_pv_state_end;
  wire proj_state_work;
  wire proj_state_end;
  wire ffn0_state_work;
  wire ffn0_state_end;
  wire ffn1_state_work;
  wire ffn1_state_end;

  // Instantiate core_top (single core focused test)
  // Declare core_top interface signals used by this TB
  reg clean_kv_cache = 0;
  reg [(`USER_ID_WIDTH)-1:0] clean_kv_cache_user_id = 0;
  reg [(`CORE_MEM_ADDR_WIDTH)-1:0] core_mem_addr = 0;
  reg [(`INTERFACE_DATA_WIDTH)-1:0] core_mem_wdata = 0;
  reg core_mem_wen = 0;
  reg core_mem_ren = 0;
  wire [(`INTERFACE_DATA_WIDTH)-1:0] core_mem_rdata;
  wire core_mem_rvld;
  reg op_cfg_vld = 0;
  reg [40:0] op_cfg = 0;
  reg usr_cfg_vld = 0;
  reg [11:0] usr_cfg = 0;
  reg model_cfg_vld = 0;
  reg [29:0] model_cfg = 0;
  reg pmu_cfg_vld = 0;
  reg [3:0] pmu_cfg = 0;
  reg rc_cfg_vld = 0;
  reg [83:0] rc_cfg = 0;
  reg [31:0] control_state = 0;
  reg control_state_update = 0;
  reg start = 0;
  wire finish;

  // GBUS and recompute/link signals (kept simple)
  // core_top expects 19-bit GBUS addr and 128-bit GBUS/vlink/hlink data (MAC_MULT_NUM*IDATA_WIDTH)
  reg [18:0] in_gbus_addr = 0;
  reg in_gbus_wen = 0;
  reg [127:0] in_gbus_wdata = 0;
  wire [18:0] out_gbus_addr;
  wire out_gbus_wen;
  wire [127:0] out_gbus_wdata;
  reg [(`RECOMPUTE_SCALE_WIDTH)-1:0] rc_scale = 0;
  reg rc_scale_vld = 0;
  reg rc_scale_clear = 0;
  reg [127:0] vlink_data_in = 0;
  reg vlink_data_in_vld = 0;
  wire [127:0] vlink_data_out;
  wire vlink_data_out_vld;
  reg [127:0] hlink_wdata = 0;
  reg hlink_wen = 0;
  wire [127:0] hlink_rdata;
  wire hlink_rvalid;

  core_top uut (
    .clk(clk),
    .rstn(rst_n),
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

  // Helper to compute base address for instruction registers (from sys_defs)
  localparam [21:0] INSTR_BASE = (22'h200000 + 22'h001000 + 22'h000100);

  // Monitor qgen transitions
  reg prev_qgen_work = 0;
  reg prev_qgen_end = 0;

  always @(posedge clk) begin
    if (qgen_state_work && !prev_qgen_work) $display("[Q_GEN_tb] Q_GEN started at time=%0t", $time);
    if (qgen_state_end && !prev_qgen_end) $display("[Q_GEN_tb] Q_GEN ended at time=%0t", $time);
    prev_qgen_work <= qgen_state_work;
    prev_qgen_end <= qgen_state_end;
  end
  // Monitor key DUT outputs and print when they become valid
  always @(posedge clk) begin
    if (out_gbus_wen) $display("[Q_GEN_tb] out_gbus_wen at %0t addr=0x%h data=0x%h", $time, out_gbus_addr, out_gbus_wdata);
    if (core_mem_rvld) $display("[Q_GEN_tb] core_mem_rdata at %0t data=0x%h", $time, core_mem_rdata);
    if (vlink_data_out_vld) $display("[Q_GEN_tb] vlink_data_out at %0t data=0x%h", $time, vlink_data_out);
    if (hlink_rvalid) $display("[Q_GEN_tb] hlink_rdata at %0t data=0x%h", $time, hlink_rdata);
    if (finish) $display("[Q_GEN_tb] DUT finish asserted at %0t", $time);
  end

  initial begin
    $display("\n[Q_GEN_tb] Starting focused Q generation testbench");
    $dumpfile("Q_GEN_tb.vcd");
    $dumpvars(0, Q_GEN_tb);

    // reset
    rst_n = 0;
    interface_wen = 0;
    interface_ren = 0;
    interface_addr = 0;
    interface_wdata = 0;
    op_cfg_vld = 0;
    in_gbus_wen = 0;
    vlink_data_in_vld = 0;
    #20; rst_n = 1;
    @(posedge clk);

    // Provide op_cfg: {cfg_acc_num[9:0], cfg_quant_scale[9:0], cfg_quant_bias[15:0], cfg_quant_shift[4:0]}
    op_cfg = {10'd4, 10'd8, 16'd100, 5'd2};
    op_cfg_vld = 1; @(posedge clk); op_cfg_vld = 0;
    $display("[Q_GEN_tb] op_cfg sent at %0t: 0x%h", $time, op_cfg);

    // Load weights via GBUS (single beat write)
    in_gbus_addr = 0;
    in_gbus_wdata = {`MAC_MULT_NUM{8'h01}}; // replicate 0x01 per MAC
    in_gbus_wen = 1; @(posedge clk); in_gbus_wen = 0;
    $display("[Q_GEN_tb] Loaded weights at %0t", $time);

    // Feed activation data on vlink
    vlink_data_in = {`MAC_MULT_NUM{8'h10}};
    vlink_data_in_vld = 1; @(posedge clk); vlink_data_in_vld = 0;
    $display("[Q_GEN_tb] Sent vlink activations at %0t", $time);

    // Trigger Q generation by setting control_state to Q_GEN (1)
    control_state = 32'd1; control_state_update = 1; @(posedge clk); control_state_update = 0;
    $display("[Q_GEN_tb] control_state set to Q_GEN at %0t", $time);

    // Optionally trigger recompute scale if needed
    rc_scale = 5'd4; rc_scale_vld = 1; @(posedge clk); rc_scale_vld = 0;

    // Run for some cycles while monitoring
    repeat (400) @(posedge clk);

    $display("[Q_GEN_tb] Done, finishing at time=%0t\n", $time);
    $finish;
  end
endmodule

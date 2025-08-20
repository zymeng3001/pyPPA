`timescale 1ps/1ps
`include "sys_defs.svh"

module tb_core_top_power;

  // -----------------------
  // Clock / Reset
  // -----------------------
  integer CLK_PERIOD_PS = 2000; // 500 MHz default
  reg clk = 0;
  always #(CLK_PERIOD_PS/2) clk = ~clk;
  reg rstn;

  // -----------------------
  // Widths (use $clog2 as requested)
  // -----------------------
  localparam integer CORE_ADDR_W   = $clog2(`HEAD_CORE_NUM);
  localparam integer GBUS_ADDR_W   = (2 + CORE_ADDR_W + `CMEM_ADDR_WIDTH);
  localparam integer LANES_W       = (`MAC_MULT_NUM*`IDATA_WIDTH);

  // -----------------------
  // DUT I/O
  // -----------------------
  reg                           clean_kv_cache;
  reg [`USER_ID_WIDTH-1:0]      clean_kv_cache_user_id;

  reg  [`CORE_MEM_ADDR_WIDTH-1:0]  core_mem_addr;
  reg  [`INTERFACE_DATA_WIDTH-1:0] core_mem_wdata;
  reg                               core_mem_wen;
  wire [`INTERFACE_DATA_WIDTH-1:0]  core_mem_rdata;
  reg                               core_mem_ren;
  wire                              core_mem_rvld;

  reg        op_cfg_vld; reg  [40:0] op_cfg;
  reg        usr_cfg_vld; reg [11:0] usr_cfg;
  reg      model_cfg_vld; reg [29:0] model_cfg;
  reg        pmu_cfg_vld; reg  [3:0] pmu_cfg;
  reg         rc_cfg_vld; reg [83:0] rc_cfg;

  reg  [31:0] control_state;
  reg         control_state_update;
  reg         start;
  wire        finish;

  reg  [GBUS_ADDR_W-1:0]        in_gbus_addr;
  reg                           in_gbus_wen;
  reg  [`GBUS_DATA_WIDTH-1:0]   in_gbus_wdata;

  wire [GBUS_ADDR_W-1:0]        out_gbus_addr;
  wire                          out_gbus_wen;
  wire [`GBUS_DATA_WIDTH-1:0]   out_gbus_wdata;

  reg  [`RECOMPUTE_SCALE_WIDTH-1:0] rc_scale;
  reg                               rc_scale_vld;
  reg                               rc_scale_clear;

  reg  [LANES_W-1:0] vlink_data_in;
  reg                vlink_data_in_vld;
  wire [LANES_W-1:0] vlink_data_out;
  wire               vlink_data_out_vld;

  reg  [LANES_W-1:0] hlink_wdata;
  reg                hlink_wen;
  wire [LANES_W-1:0] hlink_rdata;
  wire               hlink_rvalid;

  // -----------------------
  // Instantiate DUT
  // -----------------------
  core_top dut (
    .clk(clk), .rstn(rstn),
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

  // -----------------------
  // Stage codes (match your gates)
  // -----------------------
  localparam [31:0] S_IDLE   = 32'd0;
  localparam [31:0] S_Q_GEN  = 32'd1;
  localparam [31:0] S_K_GEN  = 32'd2;
  localparam [31:0] S_V_GEN  = 32'd3;
  localparam [31:0] S_ATT_QK = 32'd4; // S2P ON
  localparam [31:0] S_ATT_PV = 32'd5;
  localparam [31:0] S_PROJ   = 32'd6; // QUANT OFF
  localparam [31:0] S_FFN0   = 32'd7;
  localparam [31:0] S_FFN1   = 32'd8; // QUANT OFF

  // -----------------------
  // VCD (single file, per-stage windows)
  // -----------------------
  initial begin
    $dumpfile("core_top_power.vcd");
    $dumpvars(0, dut); // dump DUT hierarchy
    $dumpoff;
  end

  // -----------------------
  // Tasks / Functions
  // -----------------------
  task reset_dut; integer i; begin
    rstn = 0;
    clean_kv_cache = 0; clean_kv_cache_user_id = {`USER_ID_WIDTH{1'b0}};
    core_mem_addr = 0; core_mem_wdata = 0; core_mem_wen = 0; core_mem_ren = 0;
    op_cfg_vld = 0; op_cfg = 0;
    usr_cfg_vld = 0; usr_cfg = 0;
    model_cfg_vld = 0; model_cfg = 0;
    pmu_cfg_vld = 0; pmu_cfg = 0;
    rc_cfg_vld = 0; rc_cfg = 0;
    control_state = S_IDLE; control_state_update = 0;
    start = 0;
    in_gbus_addr = 0; in_gbus_wdata = 0; in_gbus_wen = 0;
    rc_scale = 0; rc_scale_vld = 0; rc_scale_clear = 0;
    vlink_data_in = 0; vlink_data_in_vld = 0;
    hlink_wdata = 0; hlink_wen = 0;
    for (i=0; i<10; i=i+1) @(posedge clk);
    rstn = 1;
    for (i=0; i<10; i=i+1) @(posedge clk);
  end endtask

  function [40:0] make_op_cfg;
    input integer acc_num, scale, bias, shift;
    reg [40:0] x;
    begin
      x = 41'd0;
      x[40:31] = acc_num[9:0];
      x[30:21] = scale[9:0];
      x[20:5]  = bias[15:0];
      x[4:0]   = shift[4:0];
      make_op_cfg = x;
    end
  endfunction

  task push_state; input [31:0] st; begin
    control_state       <= st;
    control_state_update<= 1'b1;
    @(posedge clk);
    control_state_update<= 1'b0;
  end endtask

  task poke_rc_scale; input integer pulses, gap; integer i,j; begin
    for (i=0; i<pulses; i=i+1) begin
      rc_scale     <= $urandom % (1<<`RECOMPUTE_SCALE_WIDTH);
      rc_scale_vld <= 1'b1;
      @(posedge clk);
      rc_scale_vld <= 1'b0;
      for (j=0; j<gap; j=j+1) @(posedge clk);
    end
  end endtask

  task drive_links; input integer cycles; integer i; begin
    for (i=0; i<cycles; i=i+1) begin
      hlink_wen         <= (i%3==0);
      hlink_wdata       <= $urandom;
      vlink_data_in_vld <= (i%5==0);
      vlink_data_in     <= $urandom;
      @(posedge clk);
    end
    hlink_wen <= 1'b0; vlink_data_in_vld <= 1'b0;
  end endtask

  task preload_cmem; input integer words; integer i; begin
    for (i=0; i<words; i=i+1) begin
      // {hs_bias(2'b00=core CMEM), core_addr(CORE_INDEX=0), cmem_addr(i)}
      in_gbus_addr  = {2'b00, {CORE_ADDR_W{1'b0}}, i[`CMEM_ADDR_WIDTH-1:0]};
      in_gbus_wdata = $urandom;
      in_gbus_wen   = 1'b1;
      @(posedge clk);
      in_gbus_wen   = 1'b0;
      @(posedge clk);
    end
  end endtask

  task measure_window; input [159:0] tag; input integer cycles; integer i; begin
    $display("[TB] VCD  ON   %0s @ %0t ps", tag, $time);
    $dumpon;
    for (i=0; i<cycles; i=i+1) @(posedge clk);
    $dumpoff;
    $display("[TB] VCD  OFF  %0s @ %0t ps", tag, $time);
  end endtask

  task run_stage;
    input [159:0] tag;
    input [31:0]  st;
    input integer warm, meas, cool;
    begin
      push_state(st);
      // pulse start per stage (safe)
      start <= 1'b1; @(posedge clk); start <= 1'b0;

      // warm up
      drive_links(warm/2);
      poke_rc_scale(2, 3);
      drive_links(warm - (warm/2));

      // measure
      measure_window(tag, meas);

      // cool down
      drive_links(cool);
    end
  endtask

  // -----------------------
  // Sequence
  // -----------------------
  integer warm_cycles, meas_cycles, cool_cycles;
  initial begin
    warm_cycles = 400;
    meas_cycles = 4000;
    cool_cycles = 200;
    // overrides via +args
    void'($value$plusargs("WARM=%d",  warm_cycles));
    void'($value$plusargs("MEAS=%d",  meas_cycles));
    void'($value$plusargs("COOL=%d",  cool_cycles));
    void'($value$plusargs("CLKPS=%d", CLK_PERIOD_PS));

    reset_dut();

    // Program OP config (adjust as needed)
    op_cfg     = make_op_cfg(/*acc*/16, /*scale*/1, /*bias*/0, /*shift*/0);
    op_cfg_vld = 1'b1; @(posedge clk); op_cfg_vld = 1'b0;

    // Minimal configs (plug real values if needed)
    usr_cfg_vld=1'b1; usr_cfg=12'd0; @(posedge clk); usr_cfg_vld=1'b0;
    model_cfg_vld=1'b1; model_cfg=30'd0; @(posedge clk); model_cfg_vld=1'b0;
    rc_cfg_vld=1'b1; rc_cfg=84'd0; @(posedge clk); rc_cfg_vld=1'b0;
    pmu_cfg_vld=1'b1; pmu_cfg=4'd0; @(posedge clk); pmu_cfg_vld=1'b0;

    // Preload some CMEM words so reads hit non-zero data
    preload_cmem(32);

    // Respect KV clean spacing if you use it later (>=1000 chip clocks)
    clean_kv_cache = 1'b0;

    // Walk stages; capture VCD windows
    run_stage("Q_GEN",  S_Q_GEN,  warm_cycles, meas_cycles, cool_cycles);
    run_stage("K_GEN",  S_K_GEN,  warm_cycles, meas_cycles, cool_cycles);
    run_stage("V_GEN",  S_V_GEN,  warm_cycles, meas_cycles, cool_cycles);
    run_stage("ATT_QK", S_ATT_QK, warm_cycles, meas_cycles, cool_cycles); // S2P on
    run_stage("ATT_PV", S_ATT_PV, warm_cycles, meas_cycles, cool_cycles);
    run_stage("PROJ",   S_PROJ,   warm_cycles, meas_cycles, cool_cycles); // quant off
    run_stage("FFN0",   S_FFN0,   warm_cycles, meas_cycles, cool_cycles);
    run_stage("FFN1",   S_FFN1,   warm_cycles, meas_cycles, cool_cycles); // quant off

    $display("\n[TB] Use the printed VCD ON/OFF timestamps as start/end windows in your power tool.\n");
    #(10*CLK_PERIOD_PS) $finish;
  end

endmodule

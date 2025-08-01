`timescale 1ns/1ps

module tb_head_top;
  reg clk;
  reg rst_n;

 
  initial clk = 0;
  always #50 clk = ~clk;

  // DUT inputs
  reg clean_kv_cache;
  reg [1:0] clean_kv_cache_user_id;
  reg [18:0] head_mem_addr;
  reg [15:0] head_mem_wdata;
  reg head_mem_wen;
  reg head_mem_ren;
  reg [127:0] global_sram_rd_data;
  reg global_sram_rd_data_vld;
  reg [2047:0] vlink_data_in_array;
  reg [15:0] vlink_data_in_vld_array;
  reg [31:0] control_state;
  reg control_state_update;
  reg start;
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

  // DUT outputs
  wire [15:0] head_mem_rdata;
  wire head_mem_rvld;
  wire [2047:0] vlink_data_out_array;
  wire [15:0] vlink_data_out_vld_array;
  wire finish;
  wire [18:0] gbus_addr_delay1;
  wire gbus_wen_delay1;
  wire [31:0] gbus_wdata_delay1;

  // Instantiate DUT
  head_top uut (
    .clk(clk), .rst_n(rst_n),
    .clean_kv_cache(clean_kv_cache),
    .clean_kv_cache_user_id(clean_kv_cache_user_id),
    .head_mem_addr(head_mem_addr),
    .head_mem_wdata(head_mem_wdata),
    .head_mem_wen(head_mem_wen),
    .head_mem_rdata(head_mem_rdata),
    .head_mem_ren(head_mem_ren),
    .head_mem_rvld(head_mem_rvld),
    .global_sram_rd_data(global_sram_rd_data),
    .global_sram_rd_data_vld(global_sram_rd_data_vld),
    .vlink_data_in_array(vlink_data_in_array),
    .vlink_data_in_vld_array(vlink_data_in_vld_array),
    .vlink_data_out_array(vlink_data_out_array),
    .vlink_data_out_vld_array(vlink_data_out_vld_array),
    .control_state(control_state),
    .control_state_update(control_state_update),
    .start(start),
    .finish(finish),
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
    .gbus_addr_delay1(gbus_addr_delay1),
    .gbus_wen_delay1(gbus_wen_delay1),
    .gbus_wdata_delay1(gbus_wdata_delay1)
  );

  // Stimulus
  initial begin
    $dumpfile("tb_head_top.vcd");
    $dumpvars(0, tb_head_top);

    // Reset
    rst_n = 0;
    start = 0;
    control_state = 0;
    control_state_update = 0;
    clean_kv_cache = 0;
    clean_kv_cache_user_id = 0;
    head_mem_addr = 0;
    head_mem_wdata = 16'hABCD;
    head_mem_wen = 0;
    head_mem_ren = 0;
    global_sram_rd_data = 128'h12345678_9ABCDEF0_0F1E2D3C_4B5A6978;
    global_sram_rd_data_vld = 0;
    vlink_data_in_array = 0;
    vlink_data_in_vld_array = 0;
    op_cfg = 41'h1;
    op_cfg_vld = 0;
    usr_cfg = 12'hA5A;
    usr_cfg_vld = 0;
    model_cfg = 30'h3FF_FFFF;
    model_cfg_vld = 0;
    pmu_cfg = 4'hF;
    pmu_cfg_vld = 0;
    rc_cfg = 84'hABC_DEF_123_456_789;
    rc_cfg_vld = 0;

    #100 rst_n = 1;

    // Step 1: Assert control_state and start
    #200;
    control_state = 32'd4;
    control_state_update = 1;
    start = 1;

    #200;
    control_state_update = 0;
    start = 0;

    // Step 2: Feed in global_sram_rd_data
    #200;
    global_sram_rd_data_vld = 1;
    #200;
    global_sram_rd_data_vld = 0;

    // Step 3: Trigger head_mem read and write separately
    #200;
    head_mem_wen = 1;
    head_mem_addr = 19'h1;
    #200;
    head_mem_wen = 0;
    head_mem_ren = 1;
    #200;
    head_mem_ren = 0;

    // Step 4: Enable configuration pulses
    #200;
    op_cfg_vld = 1;
    usr_cfg_vld = 1;
    model_cfg_vld = 1;
    pmu_cfg_vld = 1;
    rc_cfg_vld = 1;
    #200;
    op_cfg_vld = 0;
    usr_cfg_vld = 0;
    model_cfg_vld = 0;
    pmu_cfg_vld = 0;
    rc_cfg_vld = 0;
    #1000 $display("SIMULATION TIME = %t", $time);

    // Step 5: Wait for finish
    #5000;
    $display("Finish = %b", finish);

    #10000 $finish;
  end
endmodule

// `timescale 1ns / 1ps

// module core_top_tb;
// 	parameter GBUS_DATA_WIDTH = 32;
// 	parameter GBUS_ADDR_WIDTH = 19;
// 	parameter WMEM_DEPTH = ${wmem_depth};
// 	parameter GSRAM_DEPTH = 32;
// 	parameter CACHE_DEPTH = ${kv_cache_depth};
// 	parameter CACHE_NUM = 16;
// 	parameter CACHE_ADDR_WIDTH = $clog2(CACHE_NUM) + $clog2(CACHE_DEPTH);
// 	parameter MAC_MULT_NUM = ${mac_num};
// 	parameter IDATA_WIDTH = 8;
// 	parameter ODATA_BIT = 25;
// 	parameter CDATA_ACCU_NUM_WIDTH = 10;
// 	parameter CDATA_SCALE_WIDTH = 10;
// 	parameter CDATA_BIAS_WIDTH = 16;
// 	parameter CDATA_SHIFT_WIDTH = 5;
// 	parameter VLINK_DATA_WIDTH = 32;
// 	parameter HLINK_DATA_WIDTH = 32;
// 	parameter CMEM_ADDR_WIDTH = 13;
// 	parameter CMEM_DATA_WIDTH = 32;
// 	parameter CORE_INDEX = 0;
// 	parameter CFG_ACC_NUM = 32;
// 	parameter CFG_QUANT_SCALE = 1;
// 	parameter CFG_QUANT_BIAS = 0;
// 	parameter CFG_QUANT_SHIFT = 9;
// 	parameter COMPUTE_LENGTH = 128;
// 	parameter WEIGHT_BASE_ADDR = 0;
// 	parameter IN_BASE_ADDR = 0;
// 	reg clk;
// 	reg rstn;
// 	reg cfg_vld;
// 	reg [CDATA_ACCU_NUM_WIDTH - 1:0] cfg_acc_num;
// 	reg [CDATA_SCALE_WIDTH - 1:0] cfg_quant_scale;
// 	reg [CDATA_BIAS_WIDTH - 1:0] cfg_quant_bias;
// 	reg [CDATA_SHIFT_WIDTH - 1:0] cfg_quant_shift;
// 	reg [31:0] control_state;
// 	reg control_state_update;
// 	reg start;
// 	wire finish;
// 	reg [GBUS_ADDR_WIDTH - 1:0] in_gbus_addr;
// 	reg in_gbus_wen;
// 	reg [GBUS_DATA_WIDTH - 1:0] in_gbus_wdata;
// 	wire [GBUS_ADDR_WIDTH - 1:0] out_gbus_addr;
// 	wire out_gbus_wen;
// 	wire [GBUS_DATA_WIDTH - 1:0] out_gbus_wdata;
// 	reg vlink_enable;
// 	reg [GBUS_DATA_WIDTH - 1:0] vlink_wdata;
// 	reg vlink_wen;
// 	wire [GBUS_DATA_WIDTH - 1:0] vlink_rdata;
// 	wire vlink_rvalid;
// 	reg [GBUS_DATA_WIDTH - 1:0] hlink_wdata;
// 	reg hlink_wen;
// 	wire [GBUS_DATA_WIDTH - 1:0] hlink_rdata;
// 	wire hlink_rvalid;
// 	core_top #(
// 		.GBUS_DATA_WIDTH(GBUS_DATA_WIDTH),
// 		.GBUS_ADDR_WIDTH(GBUS_ADDR_WIDTH),
// 		.WMEM_DEPTH(WMEM_DEPTH),
// 		.CACHE_DEPTH(CACHE_DEPTH),
// 		.CACHE_NUM(CACHE_NUM),
// 		.CMEM_ADDR_WIDTH(CMEM_ADDR_WIDTH),
// 		.CMEM_DATA_WIDTH(CMEM_DATA_WIDTH),
// 		.VLINK_DATA_WIDTH(VLINK_DATA_WIDTH),
// 		.HLINK_DATA_WIDTH(HLINK_DATA_WIDTH),
// 		.MAC_MULT_NUM(MAC_MULT_NUM),
// 		.IDATA_WIDTH(IDATA_WIDTH),
// 		.ODATA_BIT(ODATA_BIT),
// 		.CORE_INDEX(CORE_INDEX)
// 	) core_inst(.*);
// 	always #(50) clk = ~clk;
// 	wire [GBUS_DATA_WIDTH - 1:0] activation;
// 	wire [GBUS_ADDR_WIDTH - 1:0] weight_base_addr;
// 	wire [GBUS_ADDR_WIDTH - 1:0] in_base_addr;
// 	wire [GBUS_ADDR_WIDTH - 1:0] weight_raddr;
// 	integer compute_length;
// 	reg [GBUS_DATA_WIDTH - 1:0] wmem_mem [WMEM_DEPTH - 1:0];
// 	reg [GBUS_DATA_WIDTH - 1:0] in_mem [GSRAM_DEPTH - 1:0];
// 	integer result_file;
// 	integer expected_file;
// 	integer write_back_file;
// 	task single_compute_wmem;
// 		input [GBUS_DATA_WIDTH - 1:0] activation;
// 		input [GBUS_ADDR_WIDTH - 1:0] weight_raddr;
// 		begin
// 			@(negedge clk) hlink_wen = 1'b1;
// 			hlink_wdata = activation;
// 		end
// 	endtask
// 	task burst_compute_wmem;
// 		input [GBUS_ADDR_WIDTH - 1:0] weight_base_addr;
// 		input [GBUS_ADDR_WIDTH - 1:0] in_base_addr;
// 		input integer compute_length;
// 		integer w_addr;
// 		integer in_addr;
// 		reg [GBUS_DATA_WIDTH - 1:0] data;
// 		begin
// 			w_addr = weight_base_addr;
// 			in_addr = in_base_addr;
// 			begin : sv2v_autoblock_1
// 				reg signed [31:0] i;
// 				for (i = 0; i < compute_length; i = i + 1)
// 					begin
// 						data = in_mem[in_addr];
// 						single_compute_wmem(data, w_addr);
// 						w_addr = w_addr + 1;
// 						in_addr = in_addr + 1;
// 					end
// 			end
// 			@(negedge clk) hlink_wen = 1'b0;
// 			hlink_wdata = 'b0;
// 		end
// 	endtask
// 	integer expected_result_array [0:(COMPUTE_LENGTH / CFG_ACC_NUM) + 0];
// 	integer expected_result_array_wr_ptr = 0;
// 	task expected_compute_wmem;
// 		input [GBUS_ADDR_WIDTH - 1:0] weight_base_addr;
// 		input [GBUS_ADDR_WIDTH - 1:0] in_base_addr;
// 		input integer compute_length;
// 		integer offset;
// 		integer mac_result;
// 		reg [31:0] quant_result;
// 		reg [31:0] quant_bias;
// 		integer rnd_result;
// 		integer expected_result;
// 		integer quant_shift;
// 		reg [GBUS_DATA_WIDTH - 1:0] weight;
// 		reg [(MAC_MULT_NUM * IDATA_WIDTH) - 1:0] data_split;
// 		reg [(MAC_MULT_NUM * IDATA_WIDTH) - 1:0] weight_split;
// 		reg quant_round;
// 		reg [0:1] _sv2v_jump;
// 		begin
// 			_sv2v_jump = 2'b00;
// 			offset = 0;
// 			mac_result = 0;
// 			begin : sv2v_autoblock_2
// 				reg signed [31:0] i;
// 				begin : sv2v_autoblock_3
// 					reg signed [31:0] _sv2v_value_on_break;
// 					for (i = 0; i < compute_length; i = i + 1)
// 						if (_sv2v_jump < 2'b10) begin
// 							_sv2v_jump = 2'b00;
// 							data_split = in_mem[in_base_addr + offset];
// 							weight_split = wmem_mem[weight_base_addr + offset];
// 							if (((i + 1) % CFG_ACC_NUM) != 0) begin
// 								begin : sv2v_autoblock_4
// 									reg signed [31:0] k;
// 									for (k = 0; k < MAC_MULT_NUM; k = k + 1)
// 										mac_result = $signed(mac_result) + ($signed(data_split[k * IDATA_WIDTH+:IDATA_WIDTH]) * $signed(weight_split[k * IDATA_WIDTH+:IDATA_WIDTH]));
// 								end
// 								offset = offset + 1;
// 								_sv2v_jump = 2'b01;
// 							end
// 							if (_sv2v_jump == 2'b00) begin
// 								begin : sv2v_autoblock_5
// 									reg signed [31:0] k;
// 									for (k = 0; k < MAC_MULT_NUM; k = k + 1)
// 										mac_result = $signed(mac_result) + ($signed(data_split[k * IDATA_WIDTH+:IDATA_WIDTH]) * $signed(weight_split[k * IDATA_WIDTH+:IDATA_WIDTH]));
// 								end
// 								quant_bias = ($signed(mac_result) * $signed(CFG_QUANT_SCALE)) + $signed(CFG_QUANT_BIAS);
// 								quant_round = quant_bias[CFG_QUANT_SHIFT - 1];
// 								quant_shift = CFG_QUANT_SHIFT;
// 								quant_result = $signed(quant_bias) >>> CFG_QUANT_SHIFT;
// 								rnd_result = quant_result + quant_round;
// 								if (rnd_result > 127)
// 									expected_result = 127;
// 								else if (rnd_result < -128)
// 									expected_result = -128;
// 								else
// 									expected_result = rnd_result;
// 								$fdisplay(expected_file, "acc_result = %d, quant_bias = %d, quant_result = %d, rnd_result = %d, Result = %d,i=%d", mac_result, $signed(quant_bias), $signed(quant_result), rnd_result, expected_result, i);
// 								expected_result_array[expected_result_array_wr_ptr] = expected_result;
// 								expected_result_array_wr_ptr = expected_result_array_wr_ptr + 1;
// 								mac_result = 0;
// 								quant_result = 0;
// 								rnd_result = 0;
// 								expected_result = 0;
// 								offset = offset + 1;
// 							end
// 							_sv2v_value_on_break = i;
// 						end
// 					if (!(_sv2v_jump < 2'b10))
// 						i = _sv2v_value_on_break;
// 					if (_sv2v_jump != 2'b11)
// 						_sv2v_jump = 2'b00;
// 				end
// 			end
// 		end
// 	endtask
// 	initial @(posedge rstn)
// 		$readmemh("../mems/hex/wmem_512.hex", core_inst.mem_inst.wmem_inst.inst_mem_sp.mem);
// 	reg [IDATA_WIDTH - 1:0] quant_odata_array [0:(COMPUTE_LENGTH / CFG_ACC_NUM) + 0];
// 	integer quant_odata_array_wr_ptr = 0;
// 	always begin
// 		@(negedge clk)
// 			;
// 		#(0.1)
// 			;
// 		if (core_inst.quant_odata_valid) begin
// 			$fdisplay(result_file, "Result = %d, time: %0t", $signed(core_inst.quant_odata), $time);
// 			quant_odata_array[quant_odata_array_wr_ptr] = core_inst.quant_odata;
// 			quant_odata_array_wr_ptr = quant_odata_array_wr_ptr + 1;
// 		end
// 	end
// 	reg signed [31:0] error_flag = 0;
// 	initial begin
// 		$fsdbDumpfile("core.fsdb");
// 		$fsdbDumpvars(0, core_tb);
// 		result_file = $fopen("../comp/core/tb/results.txt", "w");
// 		expected_file = $fopen("../comp/core/tb/expected.txt", "w");
// 		write_back_file = $fopen("../comp/core/tb/write_back.txt", "w");
// 		cfg_vld = 0;
// 		clk = 0;
// 		rstn = 0;
// 		control_state = 32'd0;
// 		control_state_update = 0;
// 		start = 0;
// 		hlink_wen = 0;
// 		cfg_acc_num = CFG_ACC_NUM;
// 		cfg_quant_scale = CFG_QUANT_SCALE;
// 		cfg_quant_bias = CFG_QUANT_BIAS;
// 		cfg_quant_shift = CFG_QUANT_SHIFT;
// 		@(negedge clk)
// 			;
// 		@(negedge clk)
// 			;
// 		rstn = 1;
// 		repeat (10) @(negedge clk)
// 			;
// 		control_state = 32'd1;
// 		control_state_update = 1;
// 		start = 1;
// 		cfg_vld = 1;
// 		@(negedge clk)
// 			;
// 		control_state_update = 0;
// 		start = 0;
// 		cfg_vld = 0;
// 		$readmemh("../mems/hex/wmem_512.hex", wmem_mem);
// 		$readmemh("../mems/hex/global_sram.hex", in_mem);
// 		in_gbus_addr = 'b0;
// 		in_gbus_wen = 1'b0;
// 		in_gbus_wdata = 'b0;
// 		vlink_enable = 1'b1;
// 		vlink_wdata = 'b0;
// 		vlink_wen = 1'b0;
// 		hlink_wen = 1'b0;
// 		hlink_wdata = 'b0;
// 		expected_compute_wmem(WEIGHT_BASE_ADDR, IN_BASE_ADDR, COMPUTE_LENGTH);
// 		burst_compute_wmem(WEIGHT_BASE_ADDR, IN_BASE_ADDR, COMPUTE_LENGTH);
// 		repeat (5000) @(negedge clk)
// 			;
// 		$fclose(expected_file);
// 		$fclose(result_file);
// 		$fclose(write_back_file);
// 		$display("Start comparing");
// 		begin : sv2v_autoblock_6
// 			reg signed [31:0] i;
// 			for (i = 0; i < (COMPUTE_LENGTH / CFG_ACC_NUM); i = i + 1)
// 				if ($signed(quant_odata_array[i]) !== expected_result_array[i]) begin
// 					$display("quant_odata_array[%0d]:%0d != expected_result_array[%0d]:%0d", i, $signed(quant_odata_array[i]), i, expected_result_array[i]);
// 					error_flag = 1;
// 				end
// 		end
// 		quant_odata_array_wr_ptr = 0;
// 		expected_result_array_wr_ptr = 0;
// 		if (error_flag == 0)
// 			$display("NO ERRROR");
// 		else
// 			$display("ERROR!!!");
// 		$display("Finish successfully");
// 		$finish;
// 	end
// endmodule


`timescale 1ns / 1ps

module core_top_tb;

  // Parameters
  parameter MAC_MULT_NUM = ${mac_num};
  parameter IDATA_WIDTH = 8;
  parameter HEAD_CORE_NUM = 16;
  parameter KV_CACHE_DEPTH_SINGLE_USER = ${kv_cache_depth};
  parameter KV_CACHE_DEPTH_SINGLE_USER_WITH_GQA = (KV_CACHE_DEPTH_SINGLE_USER*2);
  parameter CMEM_ADDR_WIDTH = (1 + ($clog2(MAC_MULT_NUM) +  $clog2(KV_CACHE_DEPTH_SINGLE_USER_WITH_GQA)));
  parameter GBUS_DATA_WIDTH = (MAC_MULT_NUM * IDATA_WIDTH);
  parameter GBUS_ADDR_WIDTH = (2 + $clog2(HEAD_CORE_NUM) + CMEM_ADDR_WIDTH);
  parameter CORE_MEM_ADDR_WIDTH = 14;
  parameter INTERFACE_DATA_WIDTH = 16;
  parameter MAX_NUM_USER = 1;
  parameter USER_ID_WIDTH = $clog2(MAX_NUM_USER);
  parameter RECOMPUTE_SCALE_WIDTH = 24;
  // Clocks & Reset
  reg clk = 0;
  reg rstn = 0;
  localparam CLK_PERIOD = ${clk_period};
  always #(CLK_PERIOD/2) clk = ~clk;

  // DUT inputs
  reg clean_kv_cache;
  reg [USER_ID_WIDTH-1:0] clean_kv_cache_user_id;
  reg [GBUS_ADDR_WIDTH-1:0] in_gbus_addr;
  reg in_gbus_wen;
  reg [GBUS_DATA_WIDTH-1:0] in_gbus_wdata;
  reg [INTERFACE_DATA_WIDTH-1:0] core_mem_wdata;
  reg [CORE_MEM_ADDR_WIDTH-1:0] core_mem_addr;
  reg core_mem_wen, core_mem_ren;
  reg op_cfg_vld, usr_cfg_vld, model_cfg_vld, pmu_cfg_vld, rc_cfg_vld;
  reg [40:0] op_cfg;
  reg [11:0] usr_cfg;
  reg [29:0] model_cfg;
  reg [3:0] pmu_cfg;
  reg [83:0] rc_cfg;
  reg control_state_update, start;
  reg [31:0] control_state;
  reg rc_scale_vld, rc_scale_clear;
  reg [RECOMPUTE_SCALE_WIDTH-1:0] rc_scale;
  reg vlink_data_in_vld;
  reg [GBUS_DATA_WIDTH-1:0] vlink_data_in;
  reg [GBUS_DATA_WIDTH-1:0] hlink_wdata;
  reg hlink_wen;

  // DUT outputs
  wire [GBUS_DATA_WIDTH-1:0] out_gbus_wdata;
  wire [GBUS_ADDR_WIDTH-1:0] out_gbus_addr;
  wire out_gbus_wen;
  wire finish;
  wire [INTERFACE_DATA_WIDTH-1:0] core_mem_rdata;
  wire core_mem_rvld;
  wire [GBUS_DATA_WIDTH-1:0] vlink_data_out;
  wire vlink_data_out_vld;
  wire [GBUS_DATA_WIDTH-1:0] hlink_rdata;
  wire hlink_rvalid;
  // Instantiate DUT
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

  // Simulation control
  initial begin
    $dumpfile("core_top.vcd");
    $dumpvars(0, core_top_tb);

    // Reset sequence
    rstn = 0;
    #30 rstn = 1;

    // Config setup
    op_cfg_vld = 1;
    op_cfg = {10'd32, 10'd1, 16'd0, 5'd9};
    usr_cfg_vld = 1;
    usr_cfg = 12'd0;
    model_cfg_vld = 1;
    model_cfg = 30'd0;
    pmu_cfg_vld = 1;
    pmu_cfg = 4'd0;
    rc_cfg_vld = 1;
    rc_cfg = 84'd0;
    @(negedge clk);
    op_cfg_vld = 0;
    usr_cfg_vld = 0;
    model_cfg_vld = 0;
    pmu_cfg_vld = 0;
    rc_cfg_vld = 0;

    // Start control FSM
    control_state = 32'd1;
    control_state_update = 1;
    start = 1;
    @(negedge clk);
    control_state_update = 0;
    start = 0;

    // Load weights
    in_gbus_wen = 1;
    in_gbus_addr = 0;
    in_gbus_wdata = {MAC_MULT_NUM{8'h01}};
    @(negedge clk);
    in_gbus_wen = 0;

    // Feed activation data
    vlink_data_in_vld = 1;
    vlink_data_in = {MAC_MULT_NUM{8'h10}};
    repeat (10) @(negedge clk);
    vlink_data_in_vld = 0;

    // Trigger MAC pipeline
    control_state = 32'd4;
    control_state_update = 1;
    @(negedge clk);
    control_state_update = 0;

    // Trigger recompute
    rc_scale = 5'd4;
    rc_scale_vld = 1;
    @(negedge clk);
    rc_scale_vld = 0;

    // Wait for finish or timeout
    repeat (1000) @(negedge clk);
    if (finish)
      $display("core_top completed");
    else
      $display("Timeout!");

    $finish;
  end

endmodule

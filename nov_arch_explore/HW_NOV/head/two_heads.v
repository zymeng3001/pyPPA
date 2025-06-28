module two_heads (
	clk,
	rst_n,
	clean_kv_cache,
	clean_kv_cache_user_id,
	head_mem_addr_0,
	head_mem_wdata_0,
	head_mem_wen_0,
	head_mem_rdata_0,
	head_mem_ren_0,
	head_mem_rvld_0,
	head_mem_addr_1,
	head_mem_wdata_1,
	head_mem_wen_1,
	head_mem_rdata_1,
	head_mem_ren_1,
	head_mem_rvld_1,
	global_sram_rd_data,
	global_sram_rd_data_vld,
	control_state,
	control_state_update,
	start,
	finish_0,
	finish_1,
	op_cfg_vld,
	op_cfg,
	usr_cfg_vld,
	usr_cfg,
	model_cfg_vld,
	model_cfg,
	pmu_cfg_vld,
	pmu_cfg,
	rc_cfg_vld,
	rc_cfg,
	gbus_addr_delay1_0,
	gbus_wen_delay1_0,
	gbus_wdata_delay1_0,
	gbus_addr_delay1_1,
	gbus_wen_delay1_1,
	gbus_wdata_delay1_1
);
	reg _sv2v_0;
	parameter VLINK_DATA_WIDTH = 128;
	parameter GLOBAL_SRAM_DATA_BIT = 128;
	input wire clk;
	input wire rst_n;
	input wire clean_kv_cache;
	input wire [1:0] clean_kv_cache_user_id;
	input wire [18:0] head_mem_addr_0;
	input wire [15:0] head_mem_wdata_0;
	input wire head_mem_wen_0;
	output wire [15:0] head_mem_rdata_0;
	input wire head_mem_ren_0;
	output wire head_mem_rvld_0;
	input wire [18:0] head_mem_addr_1;
	input wire [15:0] head_mem_wdata_1;
	input wire head_mem_wen_1;
	output wire [15:0] head_mem_rdata_1;
	input wire head_mem_ren_1;
	output wire head_mem_rvld_1;
	input wire [GLOBAL_SRAM_DATA_BIT - 1:0] global_sram_rd_data;
	input wire global_sram_rd_data_vld;
	input wire [31:0] control_state;
	input wire control_state_update;
	input wire start;
	output wire finish_0;
	output wire finish_1;
	input wire op_cfg_vld;
	input wire [40:0] op_cfg;
	input usr_cfg_vld;
	input wire [11:0] usr_cfg;
	input model_cfg_vld;
	input wire [29:0] model_cfg;
	input wire pmu_cfg_vld;
	input wire [3:0] pmu_cfg;
	input wire rc_cfg_vld;
	input wire [83:0] rc_cfg;
	localparam integer BUS_CMEM_ADDR_WIDTH = 13;
	localparam integer BUS_CORE_ADDR_WIDTH = 4;
	localparam integer HEAD_SRAM_BIAS_WIDTH = 2;
	output wire [((HEAD_SRAM_BIAS_WIDTH + BUS_CORE_ADDR_WIDTH) + BUS_CMEM_ADDR_WIDTH) - 1:0] gbus_addr_delay1_0;
	output wire gbus_wen_delay1_0;
	output wire [31:0] gbus_wdata_delay1_0;
	output wire [((HEAD_SRAM_BIAS_WIDTH + BUS_CORE_ADDR_WIDTH) + BUS_CMEM_ADDR_WIDTH) - 1:0] gbus_addr_delay1_1;
	output wire gbus_wen_delay1_1;
	output wire [31:0] gbus_wdata_delay1_1;
	reg [(16 * VLINK_DATA_WIDTH) - 1:0] vlink_data_in_array_0;
	reg [15:0] vlink_data_in_vld_array_0;
	wire [(16 * VLINK_DATA_WIDTH) - 1:0] vlink_data_out_array_0;
	wire [15:0] vlink_data_out_vld_array_0;
	reg [(16 * VLINK_DATA_WIDTH) - 1:0] vlink_data_in_array_1;
	reg [15:0] vlink_data_in_vld_array_1;
	wire [(16 * VLINK_DATA_WIDTH) - 1:0] vlink_data_out_array_1;
	wire [15:0] vlink_data_out_vld_array_1;
	always @(*) begin
		if (_sv2v_0)
			;
		vlink_data_in_array_0 = vlink_data_out_array_1;
		vlink_data_in_vld_array_0 = vlink_data_out_vld_array_1;
		vlink_data_in_array_1 = vlink_data_out_array_0;
		vlink_data_in_vld_array_1 = vlink_data_out_vld_array_0;
	end
	head_top #(.HEAD_INDEX(0)) inst_head_top_0(
		.clk(clk),
		.rst_n(rst_n),
		.clean_kv_cache(clean_kv_cache),
		.clean_kv_cache_user_id(clean_kv_cache_user_id),
		.head_mem_addr(head_mem_addr_0),
		.head_mem_wdata(head_mem_wdata_0),
		.head_mem_wen(head_mem_wen_0),
		.head_mem_rdata(head_mem_rdata_0),
		.head_mem_ren(head_mem_ren_0),
		.head_mem_rvld(head_mem_rvld_0),
		.global_sram_rd_data(global_sram_rd_data),
		.global_sram_rd_data_vld(global_sram_rd_data_vld),
		.vlink_data_in_array(vlink_data_in_array_0),
		.vlink_data_in_vld_array(vlink_data_in_vld_array_0),
		.vlink_data_out_array(vlink_data_out_array_0),
		.vlink_data_out_vld_array(vlink_data_out_vld_array_0),
		.control_state(control_state),
		.control_state_update(control_state_update),
		.start(start),
		.finish(finish_0),
		.op_cfg_vld(op_cfg_vld),
		.op_cfg(op_cfg),
		.model_cfg_vld(model_cfg_vld),
		.model_cfg(model_cfg),
		.pmu_cfg_vld(pmu_cfg_vld),
		.pmu_cfg(pmu_cfg),
		.usr_cfg_vld(usr_cfg_vld),
		.usr_cfg(usr_cfg),
		.rc_cfg_vld(rc_cfg_vld),
		.rc_cfg(rc_cfg),
		.gbus_addr_delay1(gbus_addr_delay1_0),
		.gbus_wen_delay1(gbus_wen_delay1_0),
		.gbus_wdata_delay1(gbus_wdata_delay1_0)
	);
	head_top #(.HEAD_INDEX(1)) inst_head_top_1(
		.clk(clk),
		.rst_n(rst_n),
		.clean_kv_cache(clean_kv_cache),
		.clean_kv_cache_user_id(clean_kv_cache_user_id),
		.head_mem_addr(head_mem_addr_1),
		.head_mem_wdata(head_mem_wdata_1),
		.head_mem_wen(head_mem_wen_1),
		.head_mem_rdata(head_mem_rdata_1),
		.head_mem_ren(head_mem_ren_1),
		.head_mem_rvld(head_mem_rvld_1),
		.global_sram_rd_data(global_sram_rd_data),
		.global_sram_rd_data_vld(global_sram_rd_data_vld),
		.vlink_data_in_array(vlink_data_in_array_1),
		.vlink_data_in_vld_array(vlink_data_in_vld_array_1),
		.vlink_data_out_array(vlink_data_out_array_1),
		.vlink_data_out_vld_array(vlink_data_out_vld_array_1),
		.control_state(control_state),
		.control_state_update(control_state_update),
		.start(start),
		.finish(finish_1),
		.op_cfg_vld(op_cfg_vld),
		.op_cfg(op_cfg),
		.model_cfg_vld(model_cfg_vld),
		.model_cfg(model_cfg),
		.pmu_cfg_vld(pmu_cfg_vld),
		.pmu_cfg(pmu_cfg),
		.usr_cfg_vld(usr_cfg_vld),
		.usr_cfg(usr_cfg),
		.rc_cfg_vld(rc_cfg_vld),
		.rc_cfg(rc_cfg),
		.gbus_addr_delay1(gbus_addr_delay1_1),
		.gbus_wen_delay1(gbus_wen_delay1_1),
		.gbus_wdata_delay1(gbus_wdata_delay1_1)
	);
	initial _sv2v_0 = 0;
endmodule

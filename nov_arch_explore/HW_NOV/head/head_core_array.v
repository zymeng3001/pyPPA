module head_core_array (
	clk,
	rst_n,
	clean_kv_cache,
	clean_kv_cache_user_id,
	core_array_mem_addr,
	core_array_mem_wdata,
	core_array_mem_wen,
	core_array_mem_rdata,
	core_array_mem_ren,
	core_array_mem_rvld,
	in_gbus_addr,
	in_gbus_wen,
	in_gbus_wdata,
	out_gbus_addr_array,
	out_gbus_wen_array,
	out_gbus_wdata_array,
	hlink_wdata,
	hlink_wen,
	vlink_data_in_array,
	vlink_data_in_vld_array,
	vlink_data_out_array,
	vlink_data_out_vld_array,
	rc_scale,
	rc_scale_vld,
	rc_scale_clear,
	op_cfg_vld,
	op_cfg,
	usr_cfg_vld,
	usr_cfg,
	model_cfg_vld,
	model_cfg,
	rc_cfg_vld,
	rc_cfg,
	pmu_cfg_vld,
	pmu_cfg,
	control_state,
	control_state_update,
	start,
	finish
);
	reg _sv2v_0;
	parameter HEAD_CORE_NUM = 16;
	parameter GBUS_ADDR_WIDTH = 19;
	parameter GBUS_DATA_WIDTH = 32;
	parameter HLINK_DATA_WIDTH = 128;
	parameter VLINK_DATA_WIDTH = 128;
	parameter CDATA_ACCU_NUM_WIDTH = 10;
	parameter CDATA_SCALE_WIDTH = 10;
	parameter CDATA_BIAS_WIDTH = 16;
	parameter CDATA_SHIFT_WIDTH = 5;
	parameter HEAD_INDEX = 0;
	input wire clk;
	input wire rst_n;
	input wire clean_kv_cache;
	input wire [1:0] clean_kv_cache_user_id;
	input wire [18:0] core_array_mem_addr;
	input wire [15:0] core_array_mem_wdata;
	input wire core_array_mem_wen;
	output reg [15:0] core_array_mem_rdata;
	input wire core_array_mem_ren;
	output reg core_array_mem_rvld;
	localparam integer BUS_CMEM_ADDR_WIDTH = 13;
	localparam integer BUS_CORE_ADDR_WIDTH = 4;
	localparam integer HEAD_SRAM_BIAS_WIDTH = 2;
	input wire [((HEAD_SRAM_BIAS_WIDTH + BUS_CORE_ADDR_WIDTH) + BUS_CMEM_ADDR_WIDTH) - 1:0] in_gbus_addr;
	input wire in_gbus_wen;
	input wire [GBUS_DATA_WIDTH - 1:0] in_gbus_wdata;
	output wire [(HEAD_CORE_NUM * ((HEAD_SRAM_BIAS_WIDTH + BUS_CORE_ADDR_WIDTH) + BUS_CMEM_ADDR_WIDTH)) - 1:0] out_gbus_addr_array;
	output wire [HEAD_CORE_NUM - 1:0] out_gbus_wen_array;
	output wire [(HEAD_CORE_NUM * GBUS_DATA_WIDTH) - 1:0] out_gbus_wdata_array;
	input wire [HLINK_DATA_WIDTH - 1:0] hlink_wdata;
	input wire hlink_wen;
	input wire [(HEAD_CORE_NUM * VLINK_DATA_WIDTH) - 1:0] vlink_data_in_array;
	input wire [HEAD_CORE_NUM - 1:0] vlink_data_in_vld_array;
	output wire [(HEAD_CORE_NUM * VLINK_DATA_WIDTH) - 1:0] vlink_data_out_array;
	output wire [HEAD_CORE_NUM - 1:0] vlink_data_out_vld_array;
	input wire [23:0] rc_scale;
	input wire rc_scale_vld;
	input wire rc_scale_clear;
	input wire op_cfg_vld;
	input wire [40:0] op_cfg;
	input wire usr_cfg_vld;
	input wire [11:0] usr_cfg;
	input wire model_cfg_vld;
	input wire [29:0] model_cfg;
	input wire rc_cfg_vld;
	input wire [83:0] rc_cfg;
	input wire pmu_cfg_vld;
	input wire [3:0] pmu_cfg;
	input wire [31:0] control_state;
	input wire control_state_update;
	input wire start;
	output reg finish;
	genvar _gv_i_1;
	reg [(HEAD_CORE_NUM * HLINK_DATA_WIDTH) - 1:0] hlink_wdata_array;
	reg [HEAD_CORE_NUM - 1:0] hlink_wen_array;
	wire [(HEAD_CORE_NUM * HLINK_DATA_WIDTH) - 1:0] hlink_rdata_array;
	wire [HEAD_CORE_NUM - 1:0] hlink_rvalid_array;
	always @(*) begin
		if (_sv2v_0)
			;
		begin : sv2v_autoblock_1
			reg signed [31:0] i;
			for (i = 0; i < HEAD_CORE_NUM; i = i + 1)
				if (i == 0) begin
					hlink_wdata_array[i * HLINK_DATA_WIDTH+:HLINK_DATA_WIDTH] = hlink_wdata;
					hlink_wen_array[i] = hlink_wen;
				end
				else begin
					hlink_wdata_array[i * HLINK_DATA_WIDTH+:HLINK_DATA_WIDTH] = hlink_rdata_array[(i - 1) * HLINK_DATA_WIDTH+:HLINK_DATA_WIDTH];
					hlink_wen_array[i] = hlink_rvalid_array[i - 1];
				end
		end
	end
	wire [HEAD_CORE_NUM - 1:0] core_finish_array;
	reg [HEAD_CORE_NUM - 1:0] core_finish_array_flag;
	generate
		for (_gv_i_1 = 0; _gv_i_1 < HEAD_CORE_NUM; _gv_i_1 = _gv_i_1 + 1) begin : core_finish_gen_array
			localparam i = _gv_i_1;
			always @(posedge clk or negedge rst_n)
				if (~rst_n)
					core_finish_array_flag[i] <= 0;
				else if (finish)
					core_finish_array_flag[i] <= 0;
				else if (core_finish_array[i])
					core_finish_array_flag[i] <= 1;
		end
	endgenerate
	always @(posedge clk or negedge rst_n)
		if (~rst_n)
			finish <= 0;
		else if (finish)
			finish <= 0;
		else if (&core_finish_array_flag)
			finish <= 1;
	reg [223:0] nxt_core_mem_addr_array;
	reg [255:0] nxt_core_mem_wdata_array;
	reg [15:0] nxt_core_mem_wen_array;
	reg [15:0] nxt_core_mem_ren_array;
	reg [223:0] core_mem_addr_array;
	reg [255:0] core_mem_wdata_array;
	reg [15:0] core_mem_wen_array;
	wire [255:0] core_mem_rdata_array;
	reg [15:0] core_mem_ren_array;
	wire [15:0] core_mem_rvld_array;
	always @(*) begin
		if (_sv2v_0)
			;
		nxt_core_mem_addr_array = core_mem_addr_array;
		nxt_core_mem_wdata_array = core_mem_wdata_array;
		nxt_core_mem_wen_array = 0;
		nxt_core_mem_ren_array = 0;
		begin : sv2v_autoblock_2
			reg signed [31:0] i;
			for (i = 0; i < 16; i = i + 1)
				if (~core_array_mem_addr[18]) begin
					if (i == core_array_mem_addr[17:14]) begin
						nxt_core_mem_addr_array[i * 14+:14] = core_array_mem_addr[13:0];
						nxt_core_mem_wdata_array[i * 16+:16] = core_array_mem_wdata;
						nxt_core_mem_wen_array[i] = core_array_mem_wen;
						nxt_core_mem_ren_array[i] = core_array_mem_ren;
					end
				end
		end
	end
	always @(posedge clk or negedge rst_n)
		if (~rst_n) begin
			core_mem_addr_array <= 0;
			core_mem_wdata_array <= 0;
			core_mem_wen_array <= 0;
			core_mem_ren_array <= 0;
		end
		else begin
			core_mem_addr_array <= nxt_core_mem_addr_array;
			core_mem_wdata_array <= nxt_core_mem_wdata_array;
			core_mem_wen_array <= nxt_core_mem_wen_array;
			core_mem_ren_array <= nxt_core_mem_ren_array;
		end
	reg [15:0] nxt_core_array_mem_rdata;
	reg nxt_core_array_mem_rvld;
	always @(posedge clk or negedge rst_n)
		if (~rst_n) begin
			core_array_mem_rvld <= 0;
			core_array_mem_rdata <= 0;
		end
		else begin
			core_array_mem_rvld <= nxt_core_array_mem_rvld;
			core_array_mem_rdata <= nxt_core_array_mem_rdata;
		end
	always @(*) begin
		if (_sv2v_0)
			;
		nxt_core_array_mem_rvld = 0;
		nxt_core_array_mem_rdata = 0;
		begin : sv2v_autoblock_3
			reg signed [31:0] i;
			for (i = 0; i < 16; i = i + 1)
				if (core_mem_rvld_array[i]) begin
					nxt_core_array_mem_rvld = 1;
					nxt_core_array_mem_rdata = core_mem_rdata_array[i * 16+:16];
				end
		end
	end
	generate
		for (_gv_i_1 = 0; _gv_i_1 < HEAD_CORE_NUM; _gv_i_1 = _gv_i_1 + 1) begin : head_cores_generate_array
			localparam i = _gv_i_1;
			core_top #(
				.CORE_INDEX(i),
				.HEAD_INDEX(HEAD_INDEX)
			) core_top_inst(
				.clk(clk),
				.rstn(rst_n),
				.clean_kv_cache(clean_kv_cache),
				.clean_kv_cache_user_id(clean_kv_cache_user_id),
				.core_mem_addr(core_mem_addr_array[i * 14+:14]),
				.core_mem_wdata(core_mem_wdata_array[i * 16+:16]),
				.core_mem_wen(core_mem_wen_array[i]),
				.core_mem_rdata(core_mem_rdata_array[i * 16+:16]),
				.core_mem_ren(core_mem_ren_array[i]),
				.core_mem_rvld(core_mem_rvld_array[i]),
				.op_cfg_vld(op_cfg_vld),
				.op_cfg(op_cfg),
				.model_cfg_vld(model_cfg_vld),
				.model_cfg(model_cfg),
				.usr_cfg_vld(usr_cfg_vld),
				.usr_cfg(usr_cfg),
				.rc_cfg_vld(rc_cfg_vld),
				.rc_cfg(rc_cfg),
				.pmu_cfg_vld(pmu_cfg_vld),
				.pmu_cfg(pmu_cfg),
				.in_gbus_addr(in_gbus_addr),
				.in_gbus_wen(in_gbus_wen),
				.in_gbus_wdata(in_gbus_wdata),
				.out_gbus_addr(out_gbus_addr_array[i * ((HEAD_SRAM_BIAS_WIDTH + BUS_CORE_ADDR_WIDTH) + BUS_CMEM_ADDR_WIDTH)+:(HEAD_SRAM_BIAS_WIDTH + BUS_CORE_ADDR_WIDTH) + BUS_CMEM_ADDR_WIDTH]),
				.out_gbus_wen(out_gbus_wen_array[i]),
				.out_gbus_wdata(out_gbus_wdata_array[i * GBUS_DATA_WIDTH+:GBUS_DATA_WIDTH]),
				.rc_scale(rc_scale),
				.rc_scale_vld(rc_scale_vld),
				.rc_scale_clear(rc_scale_clear),
				.vlink_data_in(vlink_data_in_array[i * VLINK_DATA_WIDTH+:VLINK_DATA_WIDTH]),
				.vlink_data_in_vld(vlink_data_in_vld_array[i]),
				.vlink_data_out(vlink_data_out_array[i * VLINK_DATA_WIDTH+:VLINK_DATA_WIDTH]),
				.vlink_data_out_vld(vlink_data_out_vld_array[i]),
				.hlink_wdata(hlink_wdata_array[i * HLINK_DATA_WIDTH+:HLINK_DATA_WIDTH]),
				.hlink_wen(hlink_wen_array[i]),
				.hlink_rdata(hlink_rdata_array[i * HLINK_DATA_WIDTH+:HLINK_DATA_WIDTH]),
				.hlink_rvalid(hlink_rvalid_array[i]),
				.control_state(control_state),
				.control_state_update(control_state_update),
				.start(start),
				.finish(core_finish_array[i])
			);
		end
	endgenerate
	initial _sv2v_0 = 0;
endmodule

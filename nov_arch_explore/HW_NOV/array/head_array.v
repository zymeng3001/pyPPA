module head_array (
	clk,
	rst_n,
	clean_kv_cache,
	clean_kv_cache_user_id,
	array_mem_addr,
	array_mem_wdata,
	array_mem_wen,
	array_mem_rdata,
	array_mem_ren,
	array_mem_rvld,
	global_sram_rvld,
	global_sram_rdata,
	control_state,
	control_state_update,
	start,
	finish,
	op_cfg,
	op_cfg_vld,
	usr_cfg,
	usr_cfg_vld,
	model_cfg,
	model_cfg_vld,
	pmu_cfg_vld,
	pmu_cfg,
	rc_cfg_vld,
	rc_cfg,
	gbus_addr_delay1_array,
	gbus_wen_delay1_array,
	gbus_wdata_delay1_array
);
	reg _sv2v_0;
	input wire clk;
	input wire rst_n;
	input wire clean_kv_cache;
	input wire [1:0] clean_kv_cache_user_id;
	input wire [21:0] array_mem_addr;
	input wire [15:0] array_mem_wdata;
	input wire array_mem_wen;
	output reg [15:0] array_mem_rdata;
	input wire array_mem_ren;
	output reg array_mem_rvld;
	input wire global_sram_rvld;
	input wire [127:0] global_sram_rdata;
	input wire [31:0] control_state;
	input wire control_state_update;
	input wire start;
	output reg finish;
	input wire [40:0] op_cfg;
	input wire op_cfg_vld;
	input wire [11:0] usr_cfg;
	input wire usr_cfg_vld;
	input wire [29:0] model_cfg;
	input wire model_cfg_vld;
	input wire pmu_cfg_vld;
	input wire [3:0] pmu_cfg;
	input wire rc_cfg_vld;
	input wire [83:0] rc_cfg;
	localparam integer BUS_CMEM_ADDR_WIDTH = 13;
	localparam integer BUS_CORE_ADDR_WIDTH = 4;
	localparam integer HEAD_SRAM_BIAS_WIDTH = 2;
	output wire [(8 * ((HEAD_SRAM_BIAS_WIDTH + BUS_CORE_ADDR_WIDTH) + BUS_CMEM_ADDR_WIDTH)) - 1:0] gbus_addr_delay1_array;
	output wire [7:0] gbus_wen_delay1_array;
	output wire [255:0] gbus_wdata_delay1_array;
	genvar _gv_i_1;
	wire [7:0] head_finish_array;
	reg [7:0] head_finish_flag_array;
	always @(posedge clk or negedge rst_n)
		if (~rst_n)
			finish <= 0;
		else if (finish)
			finish <= 0;
		else
			finish <= &head_finish_flag_array;
	generate
		for (_gv_i_1 = 0; _gv_i_1 < 8; _gv_i_1 = _gv_i_1 + 1) begin : genblk1
			localparam i = _gv_i_1;
			always @(posedge clk or negedge rst_n)
				if (~rst_n)
					head_finish_flag_array[i] <= 0;
				else if (finish)
					head_finish_flag_array[i] <= 0;
				else if (head_finish_array[i])
					head_finish_flag_array[i] <= 1;
		end
	endgenerate
	reg [151:0] nxt_head_mem_addr_array;
	reg [127:0] nxt_head_mem_wdata_array;
	reg [7:0] nxt_head_mem_wen_array;
	reg [7:0] nxt_head_mem_ren_array;
	reg [151:0] head_mem_addr_array;
	reg [127:0] head_mem_wdata_array;
	reg [7:0] head_mem_wen_array;
	wire [127:0] head_mem_rdata_array;
	reg [7:0] head_mem_ren_array;
	wire [7:0] head_mem_rvld_array;
	always @(*) begin
		if (_sv2v_0)
			;
		nxt_head_mem_addr_array = head_mem_addr_array;
		nxt_head_mem_wdata_array = head_mem_wdata_array;
		nxt_head_mem_wen_array = 0;
		nxt_head_mem_ren_array = 0;
		begin : sv2v_autoblock_1
			reg signed [31:0] i;
			for (i = 0; i < 8; i = i + 1)
				if (~array_mem_addr[21]) begin
					if (i == array_mem_addr[18+:3]) begin
						nxt_head_mem_addr_array[i * 19+:19] = {1'b0, array_mem_addr[17:0]};
						nxt_head_mem_wdata_array[i * 16+:16] = array_mem_wdata;
						nxt_head_mem_wen_array[i] = array_mem_wen;
						nxt_head_mem_ren_array[i] = array_mem_ren;
					end
				end
				else if (i == array_mem_addr[10:8]) begin
					nxt_head_mem_addr_array[i * 19+:19] = {11'b10000000000, array_mem_addr[7:0]};
					nxt_head_mem_wdata_array[i * 16+:16] = array_mem_wdata;
					nxt_head_mem_wen_array[i] = array_mem_wen;
					nxt_head_mem_ren_array[i] = array_mem_ren;
				end
		end
	end
	always @(posedge clk or negedge rst_n)
		if (~rst_n) begin
			head_mem_addr_array <= 0;
			head_mem_wdata_array <= 0;
			head_mem_wen_array <= 0;
			head_mem_ren_array <= 0;
		end
		else begin
			head_mem_addr_array <= nxt_head_mem_addr_array;
			head_mem_wdata_array <= nxt_head_mem_wdata_array;
			head_mem_wen_array <= nxt_head_mem_wen_array;
			head_mem_ren_array <= nxt_head_mem_ren_array;
		end
	reg [15:0] nxt_array_mem_rdata;
	reg nxt_array_mem_rvld;
	always @(*) begin
		if (_sv2v_0)
			;
		nxt_array_mem_rvld = |head_mem_rvld_array;
		nxt_array_mem_rdata = 0;
		begin : sv2v_autoblock_2
			reg signed [31:0] i;
			for (i = 0; i < 8; i = i + 1)
				if (head_mem_rvld_array[i])
					nxt_array_mem_rdata = head_mem_rdata_array[i * 16+:16];
		end
	end
	always @(posedge clk or negedge rst_n)
		if (~rst_n) begin
			array_mem_rvld <= 0;
			array_mem_rdata <= 0;
		end
		else begin
			array_mem_rvld <= nxt_array_mem_rvld;
			array_mem_rdata <= nxt_array_mem_rdata;
		end
	generate
		for (_gv_i_1 = 0; _gv_i_1 < 4; _gv_i_1 = _gv_i_1 + 1) begin : head_gen_array
			localparam i = _gv_i_1;
			two_heads two_heads_inst(
				.clk(clk),
				.rst_n(rst_n),
				.clean_kv_cache(clean_kv_cache),
				.clean_kv_cache_user_id(clean_kv_cache_user_id),
				.head_mem_addr_0(head_mem_addr_array[(2 * i) * 19+:19]),
				.head_mem_wdata_0(head_mem_wdata_array[(2 * i) * 16+:16]),
				.head_mem_wen_0(head_mem_wen_array[2 * i]),
				.head_mem_rdata_0(head_mem_rdata_array[(2 * i) * 16+:16]),
				.head_mem_ren_0(head_mem_ren_array[2 * i]),
				.head_mem_rvld_0(head_mem_rvld_array[2 * i]),
				.head_mem_addr_1(head_mem_addr_array[((2 * i) + 1) * 19+:19]),
				.head_mem_wdata_1(head_mem_wdata_array[((2 * i) + 1) * 16+:16]),
				.head_mem_wen_1(head_mem_wen_array[(2 * i) + 1]),
				.head_mem_rdata_1(head_mem_rdata_array[((2 * i) + 1) * 16+:16]),
				.head_mem_ren_1(head_mem_ren_array[(2 * i) + 1]),
				.head_mem_rvld_1(head_mem_rvld_array[(2 * i) + 1]),
				.global_sram_rd_data(global_sram_rdata),
				.global_sram_rd_data_vld(global_sram_rvld),
				.control_state(control_state),
				.control_state_update(control_state_update),
				.start(start),
				.finish_0(head_finish_array[2 * i]),
				.finish_1(head_finish_array[(2 * i) + 1]),
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
				.gbus_addr_delay1_0(gbus_addr_delay1_array[(2 * i) * ((HEAD_SRAM_BIAS_WIDTH + BUS_CORE_ADDR_WIDTH) + BUS_CMEM_ADDR_WIDTH)+:(HEAD_SRAM_BIAS_WIDTH + BUS_CORE_ADDR_WIDTH) + BUS_CMEM_ADDR_WIDTH]),
				.gbus_wen_delay1_0(gbus_wen_delay1_array[2 * i]),
				.gbus_wdata_delay1_0(gbus_wdata_delay1_array[(2 * i) * 32+:32]),
				.gbus_addr_delay1_1(gbus_addr_delay1_array[((2 * i) + 1) * ((HEAD_SRAM_BIAS_WIDTH + BUS_CORE_ADDR_WIDTH) + BUS_CMEM_ADDR_WIDTH)+:(HEAD_SRAM_BIAS_WIDTH + BUS_CORE_ADDR_WIDTH) + BUS_CMEM_ADDR_WIDTH]),
				.gbus_wen_delay1_1(gbus_wen_delay1_array[(2 * i) + 1]),
				.gbus_wdata_delay1_1(gbus_wdata_delay1_array[((2 * i) + 1) * 32+:32])
			);
		end
	endgenerate
	initial _sv2v_0 = 0;
endmodule

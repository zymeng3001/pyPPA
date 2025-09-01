module head_top (
	clk,
	rst_n,
	clean_kv_cache,
	clean_kv_cache_user_id,
	head_mem_addr,
	head_mem_wdata,
	head_mem_wen,
	head_mem_rdata,
	head_mem_ren,
	head_mem_rvld,
	global_sram_rd_data,
	global_sram_rd_data_vld,
	vlink_data_in_array,
	vlink_data_in_vld_array,
	vlink_data_out_array,
	vlink_data_out_vld_array,
	control_state,
	control_state_update,
	start,
	finish,
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
	gbus_addr_delay1,
	gbus_wen_delay1,
	gbus_wdata_delay1
);
	reg _sv2v_0;
	parameter HEAD_INDEX = 0;
	parameter GLOBAL_SRAM_DATA_BIT = 128;
	parameter HEAD_CORE_NUM = 16;
	parameter GBUS_ADDR_WIDTH = 19;
	parameter GBUS_DATA_WIDTH = 32;
	parameter HLINK_DATA_WIDTH = 128;
	parameter VLINK_DATA_WIDTH = 128;
	parameter CDATA_ACCU_NUM_WIDTH = 10;
	parameter CDATA_SCALE_WIDTH = 10;
	parameter CDATA_BIAS_WIDTH = 16;
	parameter CDATA_SHIFT_WIDTH = 5;
	input wire clk;
	input wire rst_n;
	input wire clean_kv_cache;
	input wire [1:0] clean_kv_cache_user_id;
	input wire [18:0] head_mem_addr;
	input wire [15:0] head_mem_wdata;
	input wire head_mem_wen;
	output reg [15:0] head_mem_rdata;
	input wire head_mem_ren;
	output reg head_mem_rvld;
	input wire [GLOBAL_SRAM_DATA_BIT - 1:0] global_sram_rd_data;
	input wire global_sram_rd_data_vld;
	input wire [(HEAD_CORE_NUM * VLINK_DATA_WIDTH) - 1:0] vlink_data_in_array;
	input wire [HEAD_CORE_NUM - 1:0] vlink_data_in_vld_array;
	output wire [(HEAD_CORE_NUM * VLINK_DATA_WIDTH) - 1:0] vlink_data_out_array;
	output wire [HEAD_CORE_NUM - 1:0] vlink_data_out_vld_array;
	input wire [31:0] control_state;
	input wire control_state_update;
	input wire start;
	output reg finish;
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
	output reg [((HEAD_SRAM_BIAS_WIDTH + BUS_CORE_ADDR_WIDTH) + BUS_CMEM_ADDR_WIDTH) - 1:0] gbus_addr_delay1;
	output reg gbus_wen_delay1;
	output reg [GBUS_DATA_WIDTH - 1:0] gbus_wdata_delay1;
	wire [((HEAD_SRAM_BIAS_WIDTH + BUS_CORE_ADDR_WIDTH) + BUS_CMEM_ADDR_WIDTH) - 1:0] gbus_addr;
	wire gbus_wen;
	wire [GBUS_DATA_WIDTH - 1:0] gbus_wdata;
	always @(posedge clk or negedge rst_n)
		if (~rst_n) begin
			gbus_addr_delay1 <= 0;
			gbus_wen_delay1 <= 0;
			gbus_wdata_delay1 <= 0;
		end
		else begin
			gbus_addr_delay1 <= gbus_addr;
			gbus_wen_delay1 <= gbus_wen;
			gbus_wdata_delay1 <= gbus_wdata;
		end
	reg clean_kv_cache_delay1;
	reg [1:0] clean_kv_cache_user_id_delay1;
	always @(posedge clk or negedge rst_n)
		if (~rst_n) begin
			clean_kv_cache_delay1 <= 0;
			clean_kv_cache_user_id_delay1 <= 0;
		end
		else begin
			clean_kv_cache_delay1 <= clean_kv_cache;
			clean_kv_cache_user_id_delay1 <= clean_kv_cache_user_id;
		end
	reg [127:0] abuf_in_data;
	reg abuf_in_data_vld;
	wire [127:0] abuf_out_data;
	wire abuf_out_data_vld;
	wire [(HEAD_CORE_NUM * ((HEAD_SRAM_BIAS_WIDTH + BUS_CORE_ADDR_WIDTH) + BUS_CMEM_ADDR_WIDTH)) - 1:0] hca_out_gbus_addr_array;
	wire [HEAD_CORE_NUM - 1:0] hca_out_gbus_wen_array;
	wire [(HEAD_CORE_NUM * GBUS_DATA_WIDTH) - 1:0] hca_out_gbus_wdata_array;
	reg op_cfg_vld_reg;
	reg [40:0] op_cfg_reg;
	reg usr_cfg_vld_reg;
	reg [11:0] usr_cfg_reg;
	reg model_cfg_vld_reg;
	reg [29:0] model_cfg_reg;
	reg [31:0] control_state_reg;
	reg control_state_update_reg;
	reg rc_cfg_vld_reg;
	reg [83:0] rc_cfg_reg;
	reg pmu_cfg_vld_reg;
	reg [3:0] pmu_cfg_reg;
	reg start_reg;
	reg [8:0] head_sram_waddr;
	reg head_sram_wen;
	reg [127:0] head_sram_wdata;
	wire [8:0] head_sram_raddr;
	wire head_sram_ren;
	wire [127:0] head_sram_rdata;
	reg head_sram_rvld;
	wire hca_finish;
	reg nxt_hca_finish_flag;
	reg hca_finish_flag;
	wire head_sram_finish;
	reg nxt_head_sram_finish_flag;
	reg head_sram_finish_flag;
	reg nxt_finish;
	always @(*) begin
		if (_sv2v_0)
			;
		nxt_finish = 0;
		nxt_hca_finish_flag = hca_finish_flag;
		nxt_head_sram_finish_flag = head_sram_finish_flag;
		if (finish) begin
			nxt_finish = 0;
			nxt_hca_finish_flag = 0;
			nxt_head_sram_finish_flag = 0;
		end
		else begin
			if (((((control_state_reg == 32'd4) || (control_state_reg == 32'd5)) || (control_state_reg == 32'd7)) || (control_state_reg == 32'd6)) || (control_state_reg == 32'd8)) begin
				if ((hca_finish_flag && head_sram_finish_flag) && ~gbus_wen_delay1)
					nxt_finish = 1;
			end
			else if (hca_finish_flag && head_sram_finish_flag)
				nxt_finish = 1;
			if (hca_finish)
				nxt_hca_finish_flag = 1;
			if (head_sram_finish)
				nxt_head_sram_finish_flag = 1;
		end
	end
	always @(posedge clk or negedge rst_n)
		if (~rst_n) begin
			finish <= 0;
			head_sram_finish_flag <= 0;
			hca_finish_flag <= 0;
		end
		else begin
			finish <= nxt_finish;
			head_sram_finish_flag <= nxt_head_sram_finish_flag;
			hca_finish_flag <= nxt_hca_finish_flag;
		end
	wire [127:0] softmax_rc_out_bus_data;
	wire softmax_rc_out_bus_data_vld;
	always @(*) begin
		if (_sv2v_0)
			;
		abuf_in_data = 0;
		abuf_in_data_vld = 0;
		if (global_sram_rd_data_vld) begin
			abuf_in_data_vld = 1;
			abuf_in_data = global_sram_rd_data;
		end
		else if ((control_state_reg == 32'd5) && softmax_rc_out_bus_data_vld) begin
			abuf_in_data_vld = 1;
			abuf_in_data = softmax_rc_out_bus_data;
		end
		else if (head_sram_rvld && (control_state_reg != 32'd5)) begin
			abuf_in_data_vld = 1;
			abuf_in_data = head_sram_rdata;
		end
	end
	abuf inst_abuf(
		.clk(clk),
		.rst_n(rst_n),
		.start(start_reg),
		.in_data(abuf_in_data),
		.in_data_vld(abuf_in_data_vld),
		.control_state(control_state_reg),
		.control_state_update(control_state_update_reg),
		.model_cfg_vld(model_cfg_vld_reg),
		.model_cfg(model_cfg_reg),
		.usr_cfg_vld(usr_cfg_vld_reg),
		.usr_cfg(usr_cfg_reg),
		.out_data(abuf_out_data),
		.out_data_vld(abuf_out_data_vld),
		.finish_row()
	);
	reg [1:0] wdata_byte_flag;
	always @(*) begin
		if (_sv2v_0)
			;
		wdata_byte_flag = 1;
		head_sram_waddr = gbus_addr[BUS_CMEM_ADDR_WIDTH - 1-:BUS_CMEM_ADDR_WIDTH];
		head_sram_wen = 0;
		head_sram_wdata = gbus_wdata;
		if (control_state_reg == 32'd4)
			wdata_byte_flag = 2;
		if (gbus_wen && (gbus_addr[HEAD_SRAM_BIAS_WIDTH + (BUS_CORE_ADDR_WIDTH + (BUS_CMEM_ADDR_WIDTH - 1))-:((HEAD_SRAM_BIAS_WIDTH + (BUS_CORE_ADDR_WIDTH + (BUS_CMEM_ADDR_WIDTH - 1))) >= (BUS_CORE_ADDR_WIDTH + (BUS_CMEM_ADDR_WIDTH + 0)) ? ((HEAD_SRAM_BIAS_WIDTH + (BUS_CORE_ADDR_WIDTH + (BUS_CMEM_ADDR_WIDTH - 1))) - (BUS_CORE_ADDR_WIDTH + (BUS_CMEM_ADDR_WIDTH + 0))) + 1 : ((BUS_CORE_ADDR_WIDTH + (BUS_CMEM_ADDR_WIDTH + 0)) - (HEAD_SRAM_BIAS_WIDTH + (BUS_CORE_ADDR_WIDTH + (BUS_CMEM_ADDR_WIDTH - 1)))) + 1)] == 1))
			head_sram_wen = 1;
		if (control_state_reg == 32'd7) begin
			if (gbus_wdata[7])
				head_sram_wdata = 0;
		end
	end
	always @(posedge clk or negedge rst_n)
		if (~rst_n)
			head_sram_rvld <= 0;
		else if (head_sram_ren)
			head_sram_rvld <= 1;
		else
			head_sram_rvld <= 0;
	reg [7:0] hs_interface_addr;
	reg hs_interface_ren;
	wire hs_interface_rvalid;
	wire [15:0] hs_interface_rdata;
	reg hs_interface_wen;
	reg [15:0] hs_interface_wdata;
	wire [15:0] core_array_mem_rdata;
	wire core_array_mem_rvld;
	always @(*) begin
		if (_sv2v_0)
			;
		head_mem_rvld = 0;
		head_mem_rdata = 0;
		if (hs_interface_rvalid) begin
			head_mem_rvld = 1;
			head_mem_rdata = hs_interface_rdata;
		end
		else if (core_array_mem_rvld) begin
			head_mem_rvld = 1;
			head_mem_rdata = core_array_mem_rdata;
		end
	end
	always @(*) begin
		if (_sv2v_0)
			;
		hs_interface_addr = head_mem_addr[7:0];
		hs_interface_ren = head_mem_ren && head_mem_addr[18];
		hs_interface_wen = head_mem_wen && head_mem_addr[18];
		hs_interface_wdata = head_mem_wdata;
	end
	head_sram inst_head_sram(
		.clk(clk),
		.rstn(rst_n),
		.interface_addr(hs_interface_addr),
		.interface_ren(hs_interface_ren),
		.interface_rvalid(hs_interface_rvalid),
		.interface_rdata(hs_interface_rdata),
		.interface_wen(hs_interface_wen),
		.interface_wdata(hs_interface_wdata),
		.waddr(head_sram_waddr),
		.wen(head_sram_wen),
		.wdata(head_sram_wdata),
		.wdata_byte_flag(wdata_byte_flag),
		.raddr(head_sram_raddr),
		.ren(head_sram_ren),
		.rdata(head_sram_rdata)
	);
	head_sram_rd_ctrl inst_head_sram_rd_ctrl(
		.clk(clk),
		.rst_n(rst_n),
		.control_state(control_state_reg),
		.control_state_update(control_state_update_reg),
		.model_cfg_vld(model_cfg_vld_reg),
		.model_cfg(model_cfg_reg),
		.usr_cfg_vld(usr_cfg_vld_reg),
		.usr_cfg(usr_cfg_reg),
		.start(start_reg),
		.finish(head_sram_finish),
		.head_sram_ren(head_sram_ren),
		.head_sram_raddr(head_sram_raddr)
	);
	always @(posedge clk or negedge rst_n)
		if (~rst_n) begin
			op_cfg_vld_reg <= 0;
			op_cfg_reg <= 0;
		end
		else if (op_cfg_vld) begin
			op_cfg_vld_reg <= 1;
			op_cfg_reg <= op_cfg;
		end
		else
			op_cfg_vld_reg <= 0;
	always @(posedge clk or negedge rst_n)
		if (~rst_n) begin
			usr_cfg_vld_reg <= 0;
			usr_cfg_reg <= 0;
		end
		else if (usr_cfg_vld) begin
			usr_cfg_vld_reg <= 1;
			usr_cfg_reg <= usr_cfg;
		end
		else
			usr_cfg_vld_reg <= 0;
	always @(posedge clk or negedge rst_n)
		if (~rst_n) begin
			model_cfg_vld_reg <= 0;
			model_cfg_reg <= 0;
		end
		else if (model_cfg_vld) begin
			model_cfg_vld_reg <= 1;
			model_cfg_reg <= model_cfg;
		end
		else
			model_cfg_vld_reg <= 0;
	always @(posedge clk or negedge rst_n)
		if (~rst_n) begin
			rc_cfg_vld_reg <= 0;
			rc_cfg_reg <= 0;
		end
		else if (rc_cfg_vld) begin
			rc_cfg_vld_reg <= 1;
			rc_cfg_reg <= rc_cfg;
		end
		else
			rc_cfg_vld_reg <= 0;
	always @(posedge clk or negedge rst_n)
		if (~rst_n) begin
			pmu_cfg_vld_reg <= 0;
			pmu_cfg_reg <= 0;
		end
		else if (pmu_cfg_vld) begin
			pmu_cfg_vld_reg <= 1;
			pmu_cfg_reg <= pmu_cfg;
		end
		else
			pmu_cfg_vld_reg <= 0;
	always @(posedge clk or negedge rst_n)
		if (~rst_n) begin
			control_state_reg <= 32'd0;
			control_state_update_reg <= 0;
		end
		else if (control_state_update) begin
			control_state_update_reg <= 1;
			control_state_reg <= control_state;
		end
		else
			control_state_update_reg <= 0;
	always @(posedge clk or negedge rst_n)
		if (~rst_n)
			start_reg <= 0;
		else if (start)
			start_reg <= 1;
		else
			start_reg <= 0;
	reg [23:0] rc_scale;
	reg rc_scale_vld;
	wire [23:0] krms_rc_scale;
	wire krms_rc_scale_vld;
	wire [23:0] softmax_rc_scale;
	wire softmax_rc_scale_vld;
	always @(*) begin
		if (_sv2v_0)
			;
		rc_scale_vld = 0;
		rc_scale = 0;
		if (krms_rc_scale_vld) begin
			rc_scale_vld = 1;
			rc_scale = krms_rc_scale;
		end
		else if (softmax_rc_scale_vld) begin
			rc_scale = softmax_rc_scale;
			rc_scale_vld = 1;
		end
	end
	head_core_array #(
		.HEAD_CORE_NUM(HEAD_CORE_NUM),
		.GBUS_ADDR_WIDTH(GBUS_ADDR_WIDTH),
		.GBUS_DATA_WIDTH(GBUS_DATA_WIDTH),
		.HLINK_DATA_WIDTH(HLINK_DATA_WIDTH),
		.HEAD_INDEX(HEAD_INDEX)
	) inst_head_core_array(
		.clk(clk),
		.rst_n(rst_n),
		.clean_kv_cache(clean_kv_cache_delay1),
		.clean_kv_cache_user_id(clean_kv_cache_user_id_delay1),
		.core_array_mem_addr(head_mem_addr),
		.core_array_mem_wdata(head_mem_wdata),
		.core_array_mem_wen(head_mem_wen),
		.core_array_mem_rdata(core_array_mem_rdata),
		.core_array_mem_ren(head_mem_ren),
		.core_array_mem_rvld(core_array_mem_rvld),
		.in_gbus_addr(gbus_addr),
		.in_gbus_wen(gbus_wen),
		.in_gbus_wdata(gbus_wdata),
		.out_gbus_addr_array(hca_out_gbus_addr_array),
		.out_gbus_wen_array(hca_out_gbus_wen_array),
		.out_gbus_wdata_array(hca_out_gbus_wdata_array),
		.hlink_wdata(abuf_out_data),
		.hlink_wen(abuf_out_data_vld),
		.vlink_data_in_array(vlink_data_in_array),
		.vlink_data_in_vld_array(vlink_data_in_vld_array),
		.vlink_data_out_array(vlink_data_out_array),
		.vlink_data_out_vld_array(vlink_data_out_vld_array),
		.rc_scale(rc_scale),
		.rc_scale_vld(rc_scale_vld),
		.rc_scale_clear(finish),
		.op_cfg_vld(op_cfg_vld_reg),
		.op_cfg(op_cfg_reg),
		.pmu_cfg_vld(pmu_cfg_vld_reg),
		.pmu_cfg(pmu_cfg_reg),
		.model_cfg_vld(model_cfg_vld_reg),
		.model_cfg(model_cfg_reg),
		.rc_cfg_vld(rc_cfg_vld_reg),
		.rc_cfg(rc_cfg_reg),
		.usr_cfg_vld(usr_cfg_vld_reg),
		.usr_cfg(usr_cfg_reg),
		.control_state(control_state_reg),
		.control_state_update(control_state_update_reg),
		.start(start_reg),
		.finish(hca_finish)
	);
	gbus_top #(.HEAD_CORE_NUM(HEAD_CORE_NUM)) inst_gbus_top(
		.clk(clk),
		.rst_n(rst_n),
		.hca_gbus_addr_array(hca_out_gbus_addr_array),
		.hca_gbus_wen_array(hca_out_gbus_wen_array),
		.hca_gbus_wdata_array(hca_out_gbus_wdata_array),
		.gbus_addr(gbus_addr),
		.gbus_wen(gbus_wen),
		.gbus_wdata(gbus_wdata)
	);
	reg [127:0] nxt_krms_in_fixed_data;
	reg nxt_krms_in_fixed_data_vld;
	reg [127:0] krms_in_fixed_data;
	reg krms_in_fixed_data_vld;
	always @(*) begin
		if (_sv2v_0)
			;
		nxt_krms_in_fixed_data = 0;
		nxt_krms_in_fixed_data_vld = 0;
		if ((((control_state_reg == 32'd1) || (control_state_reg == 32'd2)) || (control_state_reg == 32'd3)) || (control_state_reg == 32'd7)) begin
			nxt_krms_in_fixed_data = global_sram_rd_data;
			nxt_krms_in_fixed_data_vld = global_sram_rd_data_vld;
		end
	end
	always @(posedge clk or negedge rst_n)
		if (~rst_n) begin
			krms_in_fixed_data <= 0;
			krms_in_fixed_data_vld <= 0;
		end
		else begin
			krms_in_fixed_data <= nxt_krms_in_fixed_data;
			krms_in_fixed_data_vld <= nxt_krms_in_fixed_data_vld;
		end
	krms #(
		.BUS_NUM(16),
		.DATA_NUM_WIDTH(10),
		.FIXED_SQUARE_SUM_WIDTH(24)
	) inst_krms(
		.clk(clk),
		.rst_n(rst_n),
		.start(start_reg),
		.rc_cfg_vld(rc_cfg_vld_reg),
		.rc_cfg(rc_cfg_reg),
		.control_state(control_state_reg),
		.control_state_update(control_state_update_reg),
		.in_fixed_data(krms_in_fixed_data),
		.in_fixed_data_vld(krms_in_fixed_data_vld),
		.rc_scale(krms_rc_scale),
		.rc_scale_vld(krms_rc_scale_vld)
	);
	wire softmax_clear;
	assign softmax_clear = finish && (control_state_reg == 32'd5);
	softmax_rc #(
		.GBUS_DATA_WIDTH(32),
		.BUS_NUM(16),
		.FIXED_DATA_WIDTH(8),
		.EXP_STAGE_DELAY(8)
	) softmax_rc_inst(
		.clk(clk),
		.rst_n(rst_n),
		.control_state(control_state_reg),
		.control_state_update(control_state_update_reg),
		.rc_cfg_vld(rc_cfg_vld_reg),
		.rc_cfg(rc_cfg_reg),
		.model_cfg_vld(model_cfg_vld_reg),
		.model_cfg(model_cfg_reg),
		.usr_cfg(usr_cfg_reg),
		.usr_cfg_vld(usr_cfg_vld_reg),
		.in_bus_data(head_sram_rdata),
		.in_bus_data_vld(head_sram_rvld),
		.gbus_wen(gbus_wen),
		.gbus_wdata(gbus_wdata),
		.gbus_addr(gbus_addr),
		.clear(softmax_clear),
		.rc_scale(softmax_rc_scale),
		.rc_scale_vld(softmax_rc_scale_vld),
		.out_bus_data(softmax_rc_out_bus_data),
		.out_bus_data_vld(softmax_rc_out_bus_data_vld)
	);
	initial _sv2v_0 = 0;
endmodule

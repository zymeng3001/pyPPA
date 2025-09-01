module array_top (
	clk,
	rst_n,
	interface_addr,
	interface_wen,
	interface_wdata,
	interface_ren,
	interface_rdata,
	interface_rvalid,
	current_token_finish_flag,
	current_token_finish_work,
	qgen_state_work,
	qgen_state_end,
	kgen_state_work,
	kgen_state_end,
	vgen_state_work,
	vgen_state_end,
	att_qk_state_work,
	att_qk_state_end,
	att_pv_state_work,
	att_pv_state_end,
	proj_state_work,
	proj_state_end,
	ffn0_state_work,
	ffn0_state_end,
	ffn1_state_work,
	ffn1_state_end
);
	reg _sv2v_0;
	input wire clk;
	input wire rst_n;
	input wire [21:0] interface_addr;
	input wire interface_wen;
	input wire [15:0] interface_wdata;
	input wire interface_ren;
	output reg [15:0] interface_rdata;
	output reg interface_rvalid;
	output reg current_token_finish_flag;
	output reg current_token_finish_work;
	output reg qgen_state_work;
	output reg qgen_state_end;
	output reg kgen_state_work;
	output reg kgen_state_end;
	output reg vgen_state_work;
	output reg vgen_state_end;
	output reg att_qk_state_work;
	output reg att_qk_state_end;
	output reg att_pv_state_work;
	output reg att_pv_state_end;
	output reg proj_state_work;
	output reg proj_state_end;
	output reg ffn0_state_work;
	output reg ffn0_state_end;
	output reg ffn1_state_work;
	output reg ffn1_state_end;
	reg clean_kv_cache;
	reg [1:0] clean_kv_cache_user_id;
	reg new_token;
	reg user_first_token;
	reg [1:0] user_id;
	wire current_token_finish;
	reg cfg_init_success;
	wire control_state_update;
	reg [31:0] control_state_delay1;
	wire [31:0] control_state;
	reg power_mode_en_in;
	reg debug_mode_en_in;
	reg [7:0] debug_mode_bits_in;
	always @(posedge clk or negedge rst_n)
		if (~rst_n)
			current_token_finish_flag <= 1;
		else if (new_token)
			current_token_finish_flag <= 0;
		else if (current_token_finish)
			current_token_finish_flag <= 1;
	always @(posedge clk or negedge rst_n)
		if (~rst_n)
			current_token_finish_work <= 0;
		else
			current_token_finish_work <= control_state_delay1 != 32'd0;
	reg nxt_qgen_state_work;
	reg nxt_qgen_state_end;
	reg nxt_kgen_state_work;
	reg nxt_kgen_state_end;
	reg nxt_vgen_state_work;
	reg nxt_vgen_state_end;
	reg nxt_att_qk_state_work;
	reg nxt_att_qk_state_end;
	reg nxt_att_pv_state_work;
	reg nxt_att_pv_state_end;
	reg nxt_proj_state_work;
	reg nxt_proj_state_end;
	reg nxt_ffn0_state_work;
	reg nxt_ffn0_state_end;
	reg nxt_ffn1_state_work;
	reg nxt_ffn1_state_end;
	always @(*) begin
		if (_sv2v_0)
			;
		nxt_qgen_state_work = control_state_delay1 == 32'd1;
		nxt_kgen_state_work = control_state_delay1 == 32'd2;
		nxt_vgen_state_work = control_state_delay1 == 32'd3;
		nxt_att_qk_state_work = control_state_delay1 == 32'd4;
		nxt_att_pv_state_work = control_state_delay1 == 32'd5;
		nxt_proj_state_work = control_state_delay1 == 32'd6;
		nxt_ffn0_state_work = control_state_delay1 == 32'd7;
		nxt_ffn1_state_work = control_state_delay1 == 32'd8;
		nxt_qgen_state_end = qgen_state_end;
		if ((control_state == 32'd2) && (control_state_delay1 == 32'd1))
			nxt_qgen_state_end = 1;
		else if (new_token)
			nxt_qgen_state_end = 0;
		nxt_kgen_state_end = kgen_state_end;
		if ((control_state == 32'd3) && (control_state_delay1 == 32'd2))
			nxt_kgen_state_end = 1;
		else if (new_token)
			nxt_kgen_state_end = 0;
		nxt_vgen_state_end = vgen_state_end;
		if ((control_state == 32'd4) && (control_state_delay1 == 32'd3))
			nxt_vgen_state_end = 1;
		else if (new_token)
			nxt_vgen_state_end = 0;
		nxt_att_qk_state_end = att_qk_state_end;
		if ((control_state == 32'd5) && (control_state_delay1 == 32'd4))
			nxt_att_qk_state_end = 1;
		else if (new_token)
			nxt_att_qk_state_end = 0;
		nxt_att_pv_state_end = att_pv_state_end;
		if ((control_state == 32'd6) && (control_state_delay1 == 32'd5))
			nxt_att_pv_state_end = 1;
		else if (new_token)
			nxt_att_pv_state_end = 0;
		nxt_proj_state_end = proj_state_end;
		if ((control_state == 32'd7) && (control_state_delay1 == 32'd6))
			nxt_proj_state_end = 1;
		else if (new_token)
			nxt_proj_state_end = 0;
		nxt_ffn0_state_end = ffn0_state_end;
		if ((control_state == 32'd8) && (control_state_delay1 == 32'd7))
			nxt_ffn0_state_end = 1;
		else if (new_token)
			nxt_ffn0_state_end = 0;
		nxt_ffn1_state_end = ffn1_state_end;
		if ((control_state == 32'd0) && (control_state_delay1 == 32'd8))
			nxt_ffn1_state_end = 1;
		else if (new_token)
			nxt_ffn1_state_end = 0;
	end
	always @(posedge clk or negedge rst_n)
		if (~rst_n) begin
			qgen_state_work <= 0;
			qgen_state_end <= 0;
			kgen_state_work <= 0;
			kgen_state_end <= 0;
			vgen_state_work <= 0;
			vgen_state_end <= 0;
			att_qk_state_work <= 0;
			att_qk_state_end <= 0;
			att_pv_state_work <= 0;
			att_pv_state_end <= 0;
			proj_state_work <= 0;
			proj_state_end <= 0;
			ffn0_state_work <= 0;
			ffn0_state_end <= 0;
			ffn1_state_work <= 0;
			ffn1_state_end <= 0;
		end
		else begin
			qgen_state_work <= nxt_qgen_state_work;
			qgen_state_end <= nxt_qgen_state_end;
			kgen_state_work <= nxt_kgen_state_work;
			kgen_state_end <= nxt_kgen_state_end;
			vgen_state_work <= nxt_vgen_state_work;
			vgen_state_end <= nxt_vgen_state_end;
			att_qk_state_work <= nxt_att_qk_state_work;
			att_qk_state_end <= nxt_att_qk_state_end;
			att_pv_state_work <= nxt_att_pv_state_work;
			att_pv_state_end <= nxt_att_pv_state_end;
			proj_state_work <= nxt_proj_state_work;
			proj_state_end <= nxt_proj_state_end;
			ffn0_state_work <= nxt_ffn0_state_work;
			ffn0_state_end <= nxt_ffn0_state_end;
			ffn1_state_work <= nxt_ffn1_state_work;
			ffn1_state_end <= nxt_ffn1_state_end;
		end
	reg [15:0] instruction_in_data_0;
	reg [15:0] instruction_in_data_1;
	reg [15:0] instruction_in_data_2;
	reg [15:0] instruction_in_data_3;
	reg signed [23:0] interface_instruction_bias_addr;
	reg [1:0] instruction_reg_addr;
	reg [15:0] instruction_reg_wdata;
	reg instruction_reg_wen;
	reg [15:0] instruction_reg_rdata;
	reg instruction_reg_ren;
	reg instruction_reg_rvld;
	always @(*) begin
		if (_sv2v_0)
			;
		interface_instruction_bias_addr = {1'b0, interface_addr} - {1'b0, (22'h200000 + 22'h001000) + 22'h000100};
		instruction_reg_addr = interface_instruction_bias_addr[1:0];
		instruction_reg_wdata = interface_wdata;
		instruction_reg_wen = interface_wen && (($signed(interface_instruction_bias_addr) >= 0) && ($signed(interface_instruction_bias_addr) <= 3));
		instruction_reg_ren = interface_ren && (($signed(interface_instruction_bias_addr) >= 0) && ($signed(interface_instruction_bias_addr) <= 3));
	end
	always @(posedge clk or negedge rst_n)
		if (~rst_n) begin
			instruction_in_data_0 <= 0;
			instruction_in_data_1 <= 0;
			instruction_in_data_2 <= 0;
			instruction_in_data_3 <= 0;
		end
		else if (instruction_reg_wen)
			case (instruction_reg_addr)
				2'b00: instruction_in_data_0 <= instruction_reg_wdata;
				2'b01: instruction_in_data_1 <= instruction_reg_wdata;
				2'b10: instruction_in_data_2 <= instruction_reg_wdata;
				2'b11: instruction_in_data_3 <= instruction_reg_wdata;
			endcase
		else begin
			if (new_token)
				instruction_in_data_0 <= 0;
			if (clean_kv_cache)
				instruction_in_data_1 <= 0;
			if (cfg_init_success)
				instruction_in_data_2 <= 0;
		end
	always @(posedge clk or negedge rst_n)
		if (~rst_n) begin
			instruction_reg_rvld <= 0;
			instruction_reg_rdata <= 0;
		end
		else if (instruction_reg_ren) begin
			instruction_reg_rvld <= 1;
			case (instruction_reg_addr)
				2'b00: instruction_reg_rdata <= instruction_in_data_0;
				2'b01: instruction_reg_rdata <= instruction_in_data_1;
				2'b10: instruction_reg_rdata <= instruction_in_data_2;
				2'b11: instruction_reg_rdata <= instruction_in_data_3;
			endcase
		end
		else
			instruction_reg_rvld <= 0;
	reg [15:0] instruction_in_data_0_delay1;
	reg [15:0] instruction_in_data_1_delay1;
	reg [15:0] instruction_in_data_2_delay1;
	always @(posedge clk or negedge rst_n)
		if (~rst_n) begin
			instruction_in_data_0_delay1 <= 0;
			instruction_in_data_1_delay1 <= 0;
			instruction_in_data_2_delay1 <= 0;
		end
		else begin
			instruction_in_data_0_delay1 <= instruction_in_data_0;
			instruction_in_data_1_delay1 <= instruction_in_data_1;
			instruction_in_data_2_delay1 <= instruction_in_data_2;
		end
	reg nxt_new_token;
	reg nxt_user_first_token;
	reg [1:0] nxt_user_id;
	always @(*) begin
		if (_sv2v_0)
			;
		nxt_new_token = 0;
		nxt_user_first_token = 0;
		nxt_user_id = instruction_in_data_0[3:2];
		if (new_token) begin
			nxt_new_token = 0;
			nxt_user_first_token = 0;
		end
		else begin
			nxt_new_token = instruction_in_data_0[0] & ~instruction_in_data_0_delay1[0];
			nxt_user_first_token = instruction_in_data_0[1] & ~instruction_in_data_0_delay1[1];
		end
	end
	always @(posedge clk or negedge rst_n)
		if (~rst_n) begin
			new_token <= 0;
			user_first_token <= 0;
			user_id <= 0;
		end
		else begin
			new_token <= nxt_new_token;
			user_first_token <= nxt_user_first_token;
			user_id <= nxt_user_id;
		end
	reg nxt_clean_kv_cache;
	reg [1:0] nxt_clean_kv_cache_user_id;
	always @(*) begin
		if (_sv2v_0)
			;
		nxt_clean_kv_cache = 0;
		nxt_clean_kv_cache_user_id = instruction_in_data_1[2:1];
		if (clean_kv_cache)
			nxt_clean_kv_cache = 0;
		else
			nxt_clean_kv_cache = instruction_in_data_1[0] & ~instruction_in_data_1_delay1[0];
	end
	always @(posedge clk or negedge rst_n)
		if (~rst_n) begin
			clean_kv_cache <= 0;
			clean_kv_cache_user_id <= 0;
		end
		else begin
			clean_kv_cache <= nxt_clean_kv_cache;
			clean_kv_cache_user_id <= nxt_clean_kv_cache_user_id;
		end
	reg nxt_cfg_init_success;
	always @(*) begin
		if (_sv2v_0)
			;
		nxt_cfg_init_success = 0;
		if (cfg_init_success)
			nxt_cfg_init_success = 0;
		else
			nxt_cfg_init_success = instruction_in_data_2[0] & ~instruction_in_data_2_delay1[0];
	end
	always @(posedge clk or negedge rst_n)
		if (~rst_n)
			cfg_init_success <= 0;
		else
			cfg_init_success <= nxt_cfg_init_success;
	always @(posedge clk or negedge rst_n)
		if (~rst_n)
			power_mode_en_in <= 0;
		else
			power_mode_en_in <= instruction_in_data_3[0];
	always @(posedge clk or negedge rst_n)
		if (~rst_n) begin
			debug_mode_en_in <= 0;
			debug_mode_bits_in <= 0;
		end
		else begin
			debug_mode_en_in <= instruction_in_data_3[1];
			debug_mode_bits_in <= instruction_in_data_3[9:2];
		end
	reg [15:0] state_out_data_0;
	reg [15:0] state_out_data_1;
	reg [15:0] state_out_data_2;
	reg [15:0] state_out_data_3;
	reg signed [23:0] interface_state_bias_addr;
	reg [1:0] state_reg_addr;
	reg [15:0] state_reg_wdata;
	reg state_reg_wen;
	reg [15:0] state_reg_rdata;
	reg state_reg_ren;
	reg state_reg_rvld;
	always @(*) begin
		if (_sv2v_0)
			;
		interface_state_bias_addr = {1'b0, interface_addr} - {1'b0, (22'h200000 + 22'h001000) + 22'h000200};
		state_reg_addr = interface_state_bias_addr[1:0];
		state_reg_wdata = interface_wdata;
		state_reg_wen = interface_wen && (($signed(interface_state_bias_addr) >= 0) && ($signed(interface_state_bias_addr) <= 3));
		state_reg_ren = interface_ren && (($signed(interface_state_bias_addr) >= 0) && ($signed(interface_state_bias_addr) <= 3));
	end
	always @(posedge clk or negedge rst_n)
		if (~rst_n) begin
			state_out_data_0 <= 0;
			state_out_data_1 <= 0;
			state_out_data_2 <= 0;
			state_out_data_3 <= 0;
		end
		else if (new_token) begin
			state_out_data_0 <= 0;
			state_out_data_1 <= 0;
			state_out_data_2 <= 0;
			state_out_data_3 <= 0;
		end
		else if (state_reg_wen)
			case (state_reg_addr)
				2'b00: state_out_data_0 <= state_reg_wdata;
				2'b01: state_out_data_1 <= state_reg_wdata;
				2'b10: state_out_data_2 <= state_reg_wdata;
				2'b11: state_out_data_3 <= state_reg_wdata;
			endcase
		else begin
			if (current_token_finish)
				state_out_data_0 <= 1;
			else if (new_token)
				state_out_data_0 <= 0;
			if (control_state_update)
				state_out_data_1 <= control_state;
		end
	always @(posedge clk or negedge rst_n)
		if (~rst_n) begin
			state_reg_rvld <= 0;
			state_reg_rdata <= 0;
		end
		else if (state_reg_ren) begin
			state_reg_rvld <= 1;
			case (state_reg_addr)
				2'b00: state_reg_rdata <= state_out_data_0;
				2'b01: state_reg_rdata <= state_out_data_1;
				2'b10: state_reg_rdata <= state_out_data_2;
				2'b11: state_reg_rdata <= state_out_data_3;
			endcase
		end
		else
			state_reg_rvld <= 0;
	reg [5:0] control_reg_addr;
	reg signed [23:0] interface_control_bias_addr;
	reg [15:0] control_reg_wdata;
	reg control_reg_wen;
	wire [15:0] control_reg_rdata;
	reg control_reg_ren;
	wire control_reg_rvld;
	always @(*) begin
		if (_sv2v_0)
			;
		interface_control_bias_addr = {1'b0, interface_addr} - {1'b0, (22'h200000 + 22'h001000) + 22'h000000};
		control_reg_addr = interface_control_bias_addr[5:0];
		control_reg_wdata = interface_wdata;
		control_reg_wen = interface_wen && (($signed(interface_control_bias_addr) >= 0) && ($signed(interface_control_bias_addr) <= 63));
		control_reg_ren = interface_ren && (($signed(interface_control_bias_addr) >= 0) && ($signed(interface_control_bias_addr) <= 63));
	end
	wire [4:0] global_sram_waddr;
	wire global_sram_wen;
	reg [127:0] global_sram_bwe;
	wire [127:0] global_sram_wdata;
	reg [4:0] global_sram_waddr_delay1;
	reg global_sram_wen_delay1;
	reg [127:0] global_sram_bwe_delay1;
	reg [127:0] global_sram_wdata_delay1;
	always @(*) begin
		if (_sv2v_0)
			;
		global_sram_bwe = {128 {1'b1}};
	end
	always @(posedge clk or negedge rst_n)
		if (~rst_n) begin
			global_sram_waddr_delay1 <= 0;
			global_sram_wen_delay1 <= 0;
			global_sram_wdata_delay1 <= 0;
			global_sram_bwe_delay1 <= 0;
		end
		else begin
			global_sram_waddr_delay1 <= global_sram_waddr;
			global_sram_wen_delay1 <= global_sram_wen;
			global_sram_wdata_delay1 <= global_sram_wdata;
			global_sram_bwe_delay1 <= global_sram_bwe;
		end
	wire [4:0] global_sram_raddr;
	wire global_sram_ren;
	wire global_sram_rvld;
	wire [127:0] global_sram_rdata;
	wire [8:0] residual_sram_waddr;
	wire residual_sram_wen;
	wire [127:0] residual_sram_wdata;
	wire residual_sram_wdata_byte_flag;
	reg [8:0] residual_sram_raddr;
	reg [8:0] residual_sram_raddr_delay1;
	reg residual_sram_ren;
	wire [127:0] residual_sram_rdata;
	reg residual_sram_rvld;
	reg [127:0] head_input_sram_rdata;
	reg head_input_sram_rdata_vld;
	reg op_start_delay1;
	wire global_sram_finish;
	wire op_start;
	wire residual_finish;
	wire head_array_finish;
	reg op_finish;
	wire [40:0] op_cfg;
	wire op_cfg_vld;
	wire [11:0] usr_cfg;
	wire usr_cfg_vld;
	wire [29:0] model_cfg;
	wire model_cfg_vld;
	wire [3:0] pmu_cfg;
	wire pmu_cfg_vld;
	wire rc_cfg_vld;
	wire [83:0] rc_cfg;
	localparam integer BUS_CMEM_ADDR_WIDTH = 13;
	localparam integer BUS_CORE_ADDR_WIDTH = 4;
	localparam integer HEAD_SRAM_BIAS_WIDTH = 2;
	wire [(8 * ((HEAD_SRAM_BIAS_WIDTH + BUS_CORE_ADDR_WIDTH) + BUS_CMEM_ADDR_WIDTH)) - 1:0] gbus_addr_delay1_array;
	wire [7:0] gbus_wen_delay1_array;
	wire [255:0] gbus_wdata_delay1_array;
	reg [(8 * ((HEAD_SRAM_BIAS_WIDTH + BUS_CORE_ADDR_WIDTH) + BUS_CMEM_ADDR_WIDTH)) - 1:0] gbus_addr_delay2_array;
	reg [7:0] gbus_wen_delay2_array;
	reg [255:0] gbus_wdata_delay2_array;
	always @(posedge clk or negedge rst_n)
		if (~rst_n) begin
			gbus_addr_delay2_array <= 0;
			gbus_wen_delay2_array <= 0;
			gbus_wdata_delay2_array <= 0;
		end
		else begin
			gbus_addr_delay2_array <= gbus_addr_delay1_array;
			gbus_wen_delay2_array <= gbus_wen_delay1_array;
			gbus_wdata_delay2_array <= gbus_wdata_delay1_array;
		end
	always @(posedge clk or negedge rst_n)
		if (~rst_n)
			op_start_delay1 <= 0;
		else
			op_start_delay1 <= op_start;
	always @(*) begin
		if (_sv2v_0)
			;
		if (control_state_delay1 == 32'd7) begin
			head_input_sram_rdata_vld = global_sram_wen;
			head_input_sram_rdata = global_sram_wdata;
		end
		else if (control_state_delay1 == 32'd8) begin
			head_input_sram_rdata_vld = 0;
			head_input_sram_rdata = 0;
		end
		else begin
			head_input_sram_rdata_vld = global_sram_rvld;
			head_input_sram_rdata = global_sram_rdata;
		end
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
	reg [21:0] array_mem_addr;
	reg [15:0] array_mem_wdata;
	reg array_mem_wen;
	wire [15:0] array_mem_rdata;
	reg array_mem_ren;
	wire array_mem_rvld;
	reg [21:0] nxt_array_mem_addr;
	reg [15:0] nxt_array_mem_wdata;
	reg nxt_array_mem_wen;
	reg nxt_array_mem_ren;
	always @(*) begin
		if (_sv2v_0)
			;
		nxt_array_mem_addr = interface_addr;
		nxt_array_mem_wdata = interface_wdata;
		nxt_array_mem_wen = interface_wen && (interface_addr < 2099200);
		nxt_array_mem_ren = interface_ren && (interface_addr < 2099200);
	end
	always @(posedge clk or negedge rst_n)
		if (~rst_n) begin
			array_mem_addr <= 0;
			array_mem_wdata <= 0;
			array_mem_wen <= 0;
			array_mem_ren <= 0;
		end
		else begin
			array_mem_addr <= nxt_array_mem_addr;
			array_mem_wdata <= nxt_array_mem_wdata;
			array_mem_wen <= nxt_array_mem_wen;
			array_mem_ren <= nxt_array_mem_ren;
		end
	reg [7:0] global_mem_addr;
	reg [15:0] global_mem_wdata;
	reg global_mem_wen;
	wire [15:0] global_mem_rdata;
	reg global_mem_ren;
	wire global_mem_rvld;
	reg signed [23:0] interface_global_bias_addr;
	reg [7:0] nxt_global_mem_addr;
	reg [15:0] nxt_global_mem_wdata;
	reg nxt_global_mem_wen;
	reg nxt_global_mem_ren;
	always @(*) begin
		if (_sv2v_0)
			;
		interface_global_bias_addr = {1'b0, interface_addr} - {1'b0, 22'h200000 + 22'h000800};
		nxt_global_mem_addr = interface_global_bias_addr[7:0];
		nxt_global_mem_wdata = interface_wdata;
		nxt_global_mem_wen = interface_wen && (($signed(interface_global_bias_addr) >= 0) && ($signed(interface_global_bias_addr) <= 255));
		nxt_global_mem_ren = interface_ren && (($signed(interface_global_bias_addr) >= 0) && ($signed(interface_global_bias_addr) <= 255));
	end
	always @(posedge clk or negedge rst_n)
		if (~rst_n) begin
			global_mem_addr <= 0;
			global_mem_wdata <= 0;
			global_mem_wen <= 0;
			global_mem_ren <= 0;
		end
		else begin
			global_mem_addr <= nxt_global_mem_addr;
			global_mem_wdata <= nxt_global_mem_wdata;
			global_mem_wen <= nxt_global_mem_wen;
			global_mem_ren <= nxt_global_mem_ren;
		end
	reg [7:0] residual_mem_addr;
	reg [15:0] residual_mem_wdata;
	reg residual_mem_wen;
	wire [15:0] residual_mem_rdata;
	reg residual_mem_ren;
	wire residual_mem_rvld;
	reg signed [23:0] interface_residual_bias_addr;
	reg [15:0] nxt_residual_mem_addr;
	reg [15:0] nxt_residual_mem_wdata;
	reg nxt_residual_mem_wen;
	reg nxt_residual_mem_ren;
	always @(*) begin
		if (_sv2v_0)
			;
		interface_residual_bias_addr = {1'b0, interface_addr} - {1'b0, 22'h200000 + 22'h000900};
		nxt_residual_mem_addr = interface_residual_bias_addr[7:0];
		nxt_residual_mem_wdata = interface_wdata;
		nxt_residual_mem_wen = interface_wen && (($signed(interface_residual_bias_addr) >= 0) && ($signed(interface_residual_bias_addr) <= 255));
		nxt_residual_mem_ren = interface_ren && (($signed(interface_residual_bias_addr) >= 0) && ($signed(interface_residual_bias_addr) <= 255));
	end
	always @(posedge clk or negedge rst_n)
		if (~rst_n) begin
			residual_mem_addr <= 0;
			residual_mem_wdata <= 0;
			residual_mem_wen <= 0;
			residual_mem_ren <= 0;
		end
		else begin
			residual_mem_addr <= nxt_residual_mem_addr;
			residual_mem_wdata <= nxt_residual_mem_wdata;
			residual_mem_wen <= nxt_residual_mem_wen;
			residual_mem_ren <= nxt_residual_mem_ren;
		end
	always @(*) begin
		if (_sv2v_0)
			;
		if (control_reg_rvld) begin
			interface_rvalid = 1;
			interface_rdata = control_reg_rdata;
		end
		else if (state_reg_rvld) begin
			interface_rvalid = 1;
			interface_rdata = state_reg_rdata;
		end
		else if (instruction_reg_rvld) begin
			interface_rvalid = 1;
			interface_rdata = instruction_reg_rdata;
		end
		else if (array_mem_rvld) begin
			interface_rvalid = 1;
			interface_rdata = array_mem_rdata;
		end
		else if (global_mem_rvld) begin
			interface_rvalid = 1;
			interface_rdata = global_mem_rdata;
		end
		else if (residual_mem_rvld) begin
			interface_rvalid = 1;
			interface_rdata = residual_mem_rdata;
		end
		else begin
			interface_rdata = 0;
			interface_rvalid = 0;
		end
	end
	head_array inst_head_array(
		.clk(clk),
		.rst_n(rst_n),
		.clean_kv_cache(clean_kv_cache_delay1),
		.clean_kv_cache_user_id(clean_kv_cache_user_id_delay1),
		.array_mem_addr(array_mem_addr),
		.array_mem_wdata(array_mem_wdata),
		.array_mem_wen(array_mem_wen),
		.array_mem_rdata(array_mem_rdata),
		.array_mem_ren(array_mem_ren),
		.array_mem_rvld(array_mem_rvld),
		.global_sram_rvld(head_input_sram_rdata_vld),
		.global_sram_rdata(head_input_sram_rdata),
		.control_state(control_state),
		.control_state_update(control_state_update),
		.start(op_start),
		.finish(head_array_finish),
		.op_cfg(op_cfg),
		.op_cfg_vld(op_cfg_vld),
		.pmu_cfg_vld(pmu_cfg_vld),
		.pmu_cfg(pmu_cfg),
		.usr_cfg(usr_cfg),
		.usr_cfg_vld(usr_cfg_vld),
		.model_cfg(model_cfg),
		.model_cfg_vld(model_cfg_vld),
		.rc_cfg_vld(rc_cfg_vld),
		.rc_cfg(rc_cfg),
		.gbus_addr_delay1_array(gbus_addr_delay1_array),
		.gbus_wen_delay1_array(gbus_wen_delay1_array),
		.gbus_wdata_delay1_array(gbus_wdata_delay1_array)
	);
	reg [7:0] vector_in_data_vld;
	reg [7:0] vector_in_data_vld_delay1;
	wire [7:0] vector_out_data;
	wire vector_out_data_vld;
	wire [12:0] vector_out_data_addr;
	reg [7:0] vector_out_data_delay1;
	reg vector_out_data_vld_delay1;
	reg [12:0] vector_out_data_addr_delay1;
	reg [7:0] vector_out_data_delay2;
	reg vector_out_data_vld_delay2;
	reg [12:0] vector_out_data_addr_delay2;
	always @(posedge clk or negedge rst_n)
		if (~rst_n)
			vector_in_data_vld_delay1 <= 0;
		else
			vector_in_data_vld_delay1 <= vector_in_data_vld;
	always @(posedge clk or negedge rst_n)
		if (~rst_n) begin
			control_state_delay1 <= 32'd0;
			vector_out_data_vld_delay1 <= 0;
			vector_out_data_vld_delay2 <= 0;
			vector_out_data_addr_delay1 <= 0;
			vector_out_data_addr_delay2 <= 0;
			vector_out_data_delay1 <= 0;
			vector_out_data_delay2 <= 0;
		end
		else begin
			control_state_delay1 <= control_state;
			vector_out_data_vld_delay1 <= vector_out_data_vld;
			vector_out_data_vld_delay2 <= vector_out_data_vld_delay1;
			vector_out_data_addr_delay1 <= vector_out_data_addr;
			vector_out_data_addr_delay2 <= vector_out_data_addr_delay1;
			vector_out_data_delay1 <= vector_out_data;
			vector_out_data_delay2 <= vector_out_data_delay1;
		end
	always @(*) begin
		if (_sv2v_0)
			;
		if ((control_state_delay1 == 32'd6) || (control_state_delay1 == 32'd8))
			vector_in_data_vld = gbus_wen_delay1_array;
		else
			vector_in_data_vld = 0;
	end
	wire vec_adder_finish;
	always @(*) begin
		if (_sv2v_0)
			;
		if (control_state_delay1 == 32'd8)
			op_finish = residual_finish;
		else if (control_state_delay1 == 32'd6)
			op_finish = vec_adder_finish;
		else
			op_finish = head_array_finish;
	end
	array_ctrl inst_array_ctrl(
		.clk(clk),
		.rst_n(rst_n),
		.control_reg_addr(control_reg_addr),
		.control_reg_wdata(control_reg_wdata),
		.control_reg_wen(control_reg_wen),
		.control_reg_rdata(control_reg_rdata),
		.control_reg_ren(control_reg_ren),
		.control_reg_rvld(control_reg_rvld),
		.control_state(control_state),
		.control_state_update(control_state_update),
		.op_start(op_start),
		.op_finish(op_finish),
		.op_cfg(op_cfg),
		.op_cfg_vld(op_cfg_vld),
		.new_token(new_token),
		.user_first_token(user_first_token),
		.user_id(user_id),
		.current_token_finish(current_token_finish),
		.usr_cfg(usr_cfg),
		.usr_cfg_vld(usr_cfg_vld),
		.pmu_cfg_vld(pmu_cfg_vld),
		.pmu_cfg(pmu_cfg),
		.cfg_init_success(cfg_init_success),
		.model_cfg(model_cfg),
		.model_cfg_vld(model_cfg_vld),
		.rc_cfg_vld(rc_cfg_vld),
		.rc_cfg(rc_cfg),
		.power_mode_en_in(power_mode_en_in),
		.debug_mode_en_in(debug_mode_en_in),
		.debug_mode_bits_in(debug_mode_bits_in)
	);
	global_sram_rd_ctrl inst_global_sram_rd_ctrl(
		.clk(clk),
		.rst_n(rst_n),
		.control_state(control_state),
		.control_state_update(control_state_update),
		.model_cfg_vld(model_cfg_vld),
		.model_cfg(model_cfg),
		.start(op_start_delay1),
		.finish(global_sram_finish),
		.vector_out_data_addr(vector_out_data_addr),
		.vector_out_data_vld(vector_out_data_vld),
		.global_sram_ren(global_sram_ren),
		.global_sram_raddr(global_sram_raddr)
	);
	global_sram_wrapper #(
		.DATA_BIT(128),
		.DEPTH(32)
	) inst_global_sram(
		.clk(clk),
		.rst_n(rst_n),
		.global_sram_waddr(global_sram_waddr_delay1),
		.global_sram_wen(global_sram_wen_delay1),
		.global_sram_bwe(global_sram_bwe_delay1),
		.global_sram_wdata(global_sram_wdata_delay1),
		.global_sram_raddr(global_sram_raddr),
		.global_sram_ren(global_sram_ren),
		.global_sram_rdata(global_sram_rdata),
		.global_sram_rvld(global_sram_rvld),
		.global_mem_addr(global_mem_addr),
		.global_mem_wdata(global_mem_wdata),
		.global_mem_wen(global_mem_wen),
		.global_mem_rdata(global_mem_rdata),
		.global_mem_ren(global_mem_ren),
		.global_mem_rvld(global_mem_rvld)
	);
	vector_adder inst_vector_adder(
		.clk(clk),
		.rst_n(rst_n),
		.in_data(gbus_wdata_delay2_array),
		.in_data_vld(vector_in_data_vld_delay1),
		.in_data_addr(gbus_addr_delay2_array[(3 * ((HEAD_SRAM_BIAS_WIDTH + BUS_CORE_ADDR_WIDTH) + BUS_CMEM_ADDR_WIDTH)) + (BUS_CMEM_ADDR_WIDTH - 1)-:BUS_CMEM_ADDR_WIDTH]),
		.op_cfg_vld(op_cfg_vld),
		.op_cfg(op_cfg),
		.out_data(vector_out_data),
		.out_data_vld(vector_out_data_vld),
		.out_data_addr(vector_out_data_addr),
		.in_finish(head_array_finish && ((control_state_delay1 == 32'd6) || (control_state_delay1 == 32'd8))),
		.out_finish(vec_adder_finish)
	);
	always @(*) begin
		if (_sv2v_0)
			;
		residual_sram_ren = 0;
		residual_sram_raddr = 0;
		if (control_state_delay1 == 32'd7) begin
			residual_sram_ren = global_sram_ren;
			residual_sram_raddr = global_sram_raddr;
		end
	end
	always @(posedge clk) residual_sram_raddr_delay1 <= residual_sram_raddr;
	always @(posedge clk or negedge rst_n)
		if (~rst_n)
			residual_sram_rvld <= 0;
		else if (residual_sram_ren)
			residual_sram_rvld <= 1;
		else
			residual_sram_rvld <= 0;
	reg rs_adder_scale_vld;
	reg [9:0] rs_adder_scale_a;
	reg [9:0] rs_adder_scale_b;
	reg rs_adder_shift_vld;
	reg [4:0] rs_adder_shift;
	reg signed [127:0] rs_adder_in_data_a;
	reg signed [127:0] rs_adder_in_data_b;
	reg rs_adder_in_data_vld;
	reg [8:0] rs_adder_in_addr;
	wire signed [127:0] rs_adder_out_data;
	wire rs_adder_out_data_vld;
	wire [8:0] rs_adder_out_addr;
	reg [40:0] ATTN_RESIDUAL_CFG_REG;
	reg [40:0] MLP_RESIDUAL_CFG_REG;
	always @(posedge clk or negedge rst_n)
		if (~rst_n) begin
			ATTN_RESIDUAL_CFG_REG <= 0;
			MLP_RESIDUAL_CFG_REG <= 0;
		end
		else begin
			ATTN_RESIDUAL_CFG_REG <= inst_array_ctrl.op_cfg_pkt.ATTN_RESIDUAL_CFG;
			MLP_RESIDUAL_CFG_REG <= inst_array_ctrl.op_cfg_pkt.MLP_RESIDUAL_CFG;
		end
	always @(*) begin
		if (_sv2v_0)
			;
		rs_adder_scale_vld = 0;
		rs_adder_scale_a = 0;
		rs_adder_scale_b = 0;
		rs_adder_shift_vld = 0;
		rs_adder_shift = 0;
		if (op_cfg_vld && (control_state_delay1 == 32'd8)) begin
			rs_adder_scale_vld = 1;
			rs_adder_shift_vld = 1;
			rs_adder_scale_a = MLP_RESIDUAL_CFG_REG[40:31];
			rs_adder_scale_b = MLP_RESIDUAL_CFG_REG[30-:10];
			rs_adder_shift = MLP_RESIDUAL_CFG_REG[4-:5];
		end
		else if (op_cfg_vld && (control_state_delay1 == 32'd7)) begin
			rs_adder_scale_vld = 1;
			rs_adder_shift_vld = 1;
			rs_adder_scale_a = ATTN_RESIDUAL_CFG_REG[40:31];
			rs_adder_scale_b = ATTN_RESIDUAL_CFG_REG[30-:10];
			rs_adder_shift = ATTN_RESIDUAL_CFG_REG[4-:5];
		end
	end
	always @(*) begin
		if (_sv2v_0)
			;
		rs_adder_in_data_a = 0;
		rs_adder_in_data_b = 0;
		rs_adder_in_data_vld = 0;
		rs_adder_in_addr = 0;
		if ((control_state_delay1 == 32'd7) && residual_sram_rvld) begin
			rs_adder_in_data_a = global_sram_rdata;
			rs_adder_in_data_b = residual_sram_rdata;
			rs_adder_in_data_vld = residual_sram_rvld;
			rs_adder_in_addr = residual_sram_raddr_delay1;
		end
		else if ((control_state_delay1 == 32'd8) && vector_out_data_vld_delay2) begin
			rs_adder_in_data_a = global_sram_rdata[vector_out_data_addr_delay2[5+:4] * 8+:8];
			rs_adder_in_data_b = vector_out_data_delay2;
			rs_adder_in_data_vld = vector_out_data_vld_delay2;
			rs_adder_in_addr = vector_out_data_addr_delay2;
		end
	end
	residual_adder inst_residual_adder(
		.clk(clk),
		.rst_n(rst_n),
		.in_finish(vec_adder_finish && (control_state_delay1 == 32'd8)),
		.out_finish(residual_finish),
		.scale_vld(rs_adder_scale_vld),
		.scale_a(rs_adder_scale_a),
		.scale_b(rs_adder_scale_b),
		.shift_vld(rs_adder_shift_vld),
		.shift(rs_adder_shift),
		.in_data_a(rs_adder_in_data_a),
		.in_data_b(rs_adder_in_data_b),
		.in_data_vld(rs_adder_in_data_vld),
		.in_addr(rs_adder_in_addr),
		.out_data(rs_adder_out_data),
		.out_data_vld(rs_adder_out_data_vld),
		.out_addr(rs_adder_out_addr)
	);
	residual_global_sram_wr_ctrl inst_residual_sram_wr_ctrl(
		.clk(clk),
		.rst_n(rst_n),
		.control_state(control_state),
		.control_state_update(control_state_update),
		.model_cfg(model_cfg),
		.model_cfg_vld(model_cfg_vld),
		.vector_out_data(vector_out_data),
		.vector_out_data_vld(vector_out_data_vld),
		.vector_out_data_addr(vector_out_data_addr),
		.rs_adder_out_data(rs_adder_out_data),
		.rs_adder_out_data_vld(rs_adder_out_data_vld),
		.rs_adder_out_addr(rs_adder_out_addr),
		.residual_sram_wdata_byte_flag(residual_sram_wdata_byte_flag),
		.residual_sram_wen(residual_sram_wen),
		.residual_sram_waddr(residual_sram_waddr),
		.residual_sram_wdata(residual_sram_wdata),
		.global_sram_waddr(global_sram_waddr),
		.global_sram_wen(global_sram_wen),
		.global_sram_wdata(global_sram_wdata)
	);
	residual_sram inst_residual_sram(
		.clk(clk),
		.rstn(rst_n),
		.interface_addr(residual_mem_addr),
		.interface_ren(residual_mem_ren),
		.interface_rdata(residual_mem_rdata),
		.interface_rvalid(residual_mem_rvld),
		.interface_wen(residual_mem_wen),
		.interface_wdata(residual_mem_wdata),
		.raddr(residual_sram_raddr),
		.ren(residual_sram_ren),
		.rdata(residual_sram_rdata),
		.waddr(residual_sram_waddr),
		.wen(residual_sram_wen),
		.wdata(residual_sram_wdata),
		.wdata_byte_flag(residual_sram_wdata_byte_flag)
	);
	initial _sv2v_0 = 0;
endmodule

module softmax_rc (
	clk,
	rst_n,
	control_state,
	control_state_update,
	rc_cfg_vld,
	rc_cfg,
	usr_cfg,
	usr_cfg_vld,
	model_cfg_vld,
	model_cfg,
	in_bus_data,
	in_bus_data_vld,
	gbus_wen,
	gbus_wdata,
	gbus_addr,
	clear,
	rc_scale,
	rc_scale_vld,
	out_bus_data,
	out_bus_data_vld
);
	reg _sv2v_0;
	parameter integer GBUS_DATA_WIDTH = 32;
	parameter integer BUS_NUM = 16;
	parameter integer FIXED_DATA_WIDTH = 8;
	parameter integer EXP_STAGE_DELAY = 1;
	parameter integer sig_width = 7;
	parameter integer exp_width = 8;
	localparam isize = 32;
	localparam isign = 1;
	localparam ieee_compliance = 0;
	localparam exp_arch = 1;
	input wire clk;
	input wire rst_n;
	input wire [31:0] control_state;
	input wire control_state_update;
	input wire rc_cfg_vld;
	input wire [83:0] rc_cfg;
	input wire [11:0] usr_cfg;
	input usr_cfg_vld;
	input wire model_cfg_vld;
	input wire [29:0] model_cfg;
	input wire [(16 * FIXED_DATA_WIDTH) - 1:0] in_bus_data;
	input wire in_bus_data_vld;
	input wire gbus_wen;
	input wire [GBUS_DATA_WIDTH - 1:0] gbus_wdata;
	localparam integer BUS_CMEM_ADDR_WIDTH = 13;
	localparam integer BUS_CORE_ADDR_WIDTH = 4;
	localparam integer HEAD_SRAM_BIAS_WIDTH = 2;
	input wire [((HEAD_SRAM_BIAS_WIDTH + BUS_CORE_ADDR_WIDTH) + BUS_CMEM_ADDR_WIDTH) - 1:0] gbus_addr;
	input wire clear;
	output reg [23:0] rc_scale;
	output reg rc_scale_vld;
	output reg [(16 * FIXED_DATA_WIDTH) - 1:0] out_bus_data;
	output reg out_bus_data_vld;
	reg [83:0] rc_cfg_reg;
	reg [31:0] control_state_reg;
	reg [29:0] model_cfg_reg;
	reg [11:0] usr_cfg_reg;
	reg signed [GBUS_DATA_WIDTH - 1:0] gbus_data_reg;
	reg [((HEAD_SRAM_BIAS_WIDTH + BUS_CORE_ADDR_WIDTH) + BUS_CMEM_ADDR_WIDTH) - 1:0] gbus_addr_reg;
	reg gbus_data_reg_vld;
	reg signed [(16 * FIXED_DATA_WIDTH) - 1:0] in_bus_data_reg;
	reg in_bus_data_vld_reg;
	reg signed [FIXED_DATA_WIDTH - 1:0] max_reg;
	reg signed [FIXED_DATA_WIDTH - 1:0] max_reg_w;
	reg signed [(FIXED_DATA_WIDTH >= 0 ? (BUS_NUM * (FIXED_DATA_WIDTH + 1)) - 1 : (BUS_NUM * (1 - FIXED_DATA_WIDTH)) + (FIXED_DATA_WIDTH - 1)):(FIXED_DATA_WIDTH >= 0 ? 0 : FIXED_DATA_WIDTH + 0)] xi_max;
	reg xi_max_vld;
	wire signed [(BUS_NUM * FIXED_DATA_WIDTH) - 1:0] xi_data_split;
	wire [((sig_width + exp_width) >= 0 ? (BUS_NUM * ((sig_width + exp_width) + 1)) - 1 : (BUS_NUM * (1 - (sig_width + exp_width))) + ((sig_width + exp_width) - 1)):((sig_width + exp_width) >= 0 ? 0 : (sig_width + exp_width) + 0)] i2flt_xi_max_dequant;
	wire [BUS_NUM - 1:0] i2flt_xi_max_dequant_vld;
	wire i2flt_xi_max_dequant_vld_out;
	wire [sig_width + exp_width:0] softmax_input_dequant_scale_reg;
	reg i2flt_xi_max_vld;
	reg [((sig_width + exp_width) >= 0 ? (BUS_NUM * ((sig_width + exp_width) + 1)) - 1 : (BUS_NUM * (1 - (sig_width + exp_width))) + ((sig_width + exp_width) - 1)):((sig_width + exp_width) >= 0 ? 0 : (sig_width + exp_width) + 0)] exp_xi_max;
	wire [((sig_width + exp_width) >= 0 ? (BUS_NUM * ((sig_width + exp_width) + 1)) - 1 : (BUS_NUM * (1 - (sig_width + exp_width))) + ((sig_width + exp_width) - 1)):((sig_width + exp_width) >= 0 ? 0 : (sig_width + exp_width) + 0)] exp_xi_max_w;
	reg exp_xi_max_vld;
	reg [((sig_width + exp_width) >= 0 ? (BUS_NUM * ((sig_width + exp_width) + 1)) - 1 : (BUS_NUM * (1 - (sig_width + exp_width))) + ((sig_width + exp_width) - 1)):((sig_width + exp_width) >= 0 ? 0 : (sig_width + exp_width) + 0)] exp_xi_max_delay;
	reg exp_xi_max_vld_delay;
	wire [sig_width + exp_width:0] softmax_exp_quant_scale_reg;
	wire [BUS_NUM - 1:0] exp_xi_max_scaled_vld;
	wire exp_xi_max_scaled_vld_out;
	reg [9:0] nxt_exp_sum_cnt;
	reg [9:0] exp_sum_cnt;
	reg [BUS_NUM - 1:0] nxt_fadd_tree_vld;
	reg [BUS_NUM - 1:0] fadd_tree_vld;
	reg [((sig_width + exp_width) >= 0 ? (BUS_NUM * ((sig_width + exp_width) + 1)) - 1 : (BUS_NUM * (1 - (sig_width + exp_width))) + ((sig_width + exp_width) - 1)):((sig_width + exp_width) >= 0 ? 0 : (sig_width + exp_width) + 0)] nxt_fadd_tree_in;
	reg [((sig_width + exp_width) >= 0 ? (BUS_NUM * ((sig_width + exp_width) + 1)) - 1 : (BUS_NUM * (1 - (sig_width + exp_width))) + ((sig_width + exp_width) - 1)):((sig_width + exp_width) >= 0 ? 0 : (sig_width + exp_width) + 0)] fadd_tree_in;
	reg nxt_fadd_tree_in_last;
	reg fadd_tree_in_last;
	wire signed [sig_width + exp_width:0] adder_out;
	wire adder_out_vld;
	reg signed [sig_width + exp_width:0] acc_out;
	wire signed [sig_width + exp_width:0] acc_out_w;
	reg acc_vld;
	wire adder_out_last;
	reg [sig_width + exp_width:0] one_over_exp_sum;
	wire [sig_width + exp_width:0] one_over_exp_sum_w;
	reg one_over_exp_sum_vld;
	wire [23:0] nxt_rc_scale;
	reg [sig_width + exp_width:0] one_over_exp_sum_shift;
	reg one_over_exp_sum_shift_vld;
	always @(posedge clk or negedge rst_n)
		if (~rst_n)
			rc_cfg_reg <= 0;
		else if (rc_cfg_vld)
			rc_cfg_reg <= rc_cfg;
	always @(posedge clk or negedge rst_n)
		if (~rst_n)
			usr_cfg_reg <= 0;
		else if (usr_cfg_vld)
			usr_cfg_reg <= usr_cfg;
	always @(posedge clk or negedge rst_n)
		if (~rst_n)
			model_cfg_reg <= 0;
		else if (model_cfg_vld)
			model_cfg_reg <= model_cfg;
	always @(posedge clk or negedge rst_n)
		if (~rst_n)
			control_state_reg <= 32'd0;
		else if (control_state_update)
			control_state_reg <= control_state;
	always @(posedge clk or negedge rst_n)
		if (~rst_n)
			gbus_data_reg_vld <= 0;
		else if (control_state_reg == 32'd4)
			gbus_data_reg_vld <= gbus_wen;
	always @(posedge clk or negedge rst_n)
		if (~rst_n) begin
			gbus_data_reg <= 0;
			gbus_addr_reg <= 0;
		end
		else if (gbus_wen & (control_state_reg == 32'd4)) begin
			gbus_data_reg <= gbus_wdata;
			gbus_addr_reg <= gbus_addr;
		end
	always @(posedge clk or negedge rst_n)
		if (~rst_n)
			in_bus_data_vld_reg <= 0;
		else if (control_state_reg == 32'd5)
			in_bus_data_vld_reg <= in_bus_data_vld;
	always @(posedge clk or negedge rst_n)
		if (~rst_n)
			in_bus_data_reg <= 0;
		else if (in_bus_data_vld & (control_state_reg == 32'd5))
			in_bus_data_reg <= in_bus_data;
	always @(*) begin
		if (_sv2v_0)
			;
		max_reg_w = max_reg;
		begin : sv2v_autoblock_1
			reg signed [31:0] i;
			for (i = 0; i < 4; i = i + 1)
				if (gbus_data_reg_vld) begin
					if ((usr_cfg_reg[0] && ((((gbus_addr_reg[(BUS_CMEM_ADDR_WIDTH - 1) - (BUS_CMEM_ADDR_WIDTH - 1)+:5] * 16) + gbus_addr_reg[(BUS_CMEM_ADDR_WIDTH - 1) - (BUS_CMEM_ADDR_WIDTH - 6)+:4]) + i) <= usr_cfg_reg[9-:9])) || ~usr_cfg_reg[0])
						max_reg_w = ($signed(gbus_data_reg[FIXED_DATA_WIDTH * i+:FIXED_DATA_WIDTH]) > $signed(max_reg_w) ? gbus_data_reg[FIXED_DATA_WIDTH * i+:FIXED_DATA_WIDTH] : max_reg_w);
				end
		end
	end
	always @(posedge clk or negedge rst_n)
		if (~rst_n)
			max_reg <= {1'b1, {FIXED_DATA_WIDTH - 1 {1'b0}}};
		else if (clear)
			max_reg <= {1'b1, {FIXED_DATA_WIDTH - 1 {1'b0}}};
		else if (gbus_data_reg_vld)
			max_reg <= max_reg_w;
	assign xi_data_split = in_bus_data_reg;
	genvar _gv_i_1;
	generate
		for (_gv_i_1 = 0; _gv_i_1 < BUS_NUM; _gv_i_1 = _gv_i_1 + 1) begin : genblk1
			localparam i = _gv_i_1;
			always @(posedge clk or negedge rst_n)
				if (~rst_n)
					xi_max[(FIXED_DATA_WIDTH >= 0 ? 0 : FIXED_DATA_WIDTH) + (i * (FIXED_DATA_WIDTH >= 0 ? FIXED_DATA_WIDTH + 1 : 1 - FIXED_DATA_WIDTH))+:(FIXED_DATA_WIDTH >= 0 ? FIXED_DATA_WIDTH + 1 : 1 - FIXED_DATA_WIDTH)] <= 0;
				else if (in_bus_data_vld_reg)
					xi_max[(FIXED_DATA_WIDTH >= 0 ? 0 : FIXED_DATA_WIDTH) + (i * (FIXED_DATA_WIDTH >= 0 ? FIXED_DATA_WIDTH + 1 : 1 - FIXED_DATA_WIDTH))+:(FIXED_DATA_WIDTH >= 0 ? FIXED_DATA_WIDTH + 1 : 1 - FIXED_DATA_WIDTH)] <= $signed(xi_data_split[i * FIXED_DATA_WIDTH+:FIXED_DATA_WIDTH]) - $signed(max_reg);
		end
	endgenerate
	always @(posedge clk or negedge rst_n)
		if (~rst_n)
			xi_max_vld <= 0;
		else
			xi_max_vld <= in_bus_data_vld_reg;
	assign softmax_input_dequant_scale_reg = rc_cfg_reg[31-:16];
	assign i2flt_xi_max_dequant_vld_out = &i2flt_xi_max_dequant_vld;
	generate
		for (_gv_i_1 = 0; _gv_i_1 < BUS_NUM; _gv_i_1 = _gv_i_1 + 1) begin : i2flt_xi_max_stage
			localparam i = _gv_i_1;
			wire signed [31:0] xi_max_ext;
			wire [sig_width + exp_width:0] i2flt_xi_max_w;
			reg [sig_width + exp_width:0] i2flt_xi_max;
			assign xi_max_ext = $signed(xi_max[(FIXED_DATA_WIDTH >= 0 ? 0 : FIXED_DATA_WIDTH) + (i * (FIXED_DATA_WIDTH >= 0 ? FIXED_DATA_WIDTH + 1 : 1 - FIXED_DATA_WIDTH))+:(FIXED_DATA_WIDTH >= 0 ? FIXED_DATA_WIDTH + 1 : 1 - FIXED_DATA_WIDTH)]);
			fp_i2flt #(
				sig_width,
				exp_width,
				isize,
				isign
			) i2flt_xi_max_inst(
				.a(xi_max_ext),
				.z(i2flt_xi_max_w)
			);
			always @(posedge clk or negedge rst_n)
				if (~rst_n)
					i2flt_xi_max <= 0;
				else if (xi_max_vld)
					i2flt_xi_max <= i2flt_xi_max_w;
			fp_mult_pipe #(
				.sig_width(sig_width),
				.exp_width(exp_width),
				.ieee_compliance(ieee_compliance),
				.stages(4)
			) inst_fp_dequant(
				.clk(clk),
				.rst_n(rst_n),
				.a(i2flt_xi_max),
				.b(softmax_input_dequant_scale_reg),
				.z(i2flt_xi_max_dequant[((sig_width + exp_width) >= 0 ? 0 : sig_width + exp_width) + (i * ((sig_width + exp_width) >= 0 ? (sig_width + exp_width) + 1 : 1 - (sig_width + exp_width)))+:((sig_width + exp_width) >= 0 ? (sig_width + exp_width) + 1 : 1 - (sig_width + exp_width))]),
				.ab_valid(i2flt_xi_max_vld),
				.z_valid(i2flt_xi_max_dequant_vld[i])
			);
		end
	endgenerate
	always @(posedge clk or negedge rst_n)
		if (~rst_n)
			i2flt_xi_max_vld <= 0;
		else
			i2flt_xi_max_vld <= xi_max_vld;
	generate
		for (_gv_i_1 = 0; _gv_i_1 < BUS_NUM; _gv_i_1 = _gv_i_1 + 1) begin : exp_xi_max_stage
			localparam i = _gv_i_1;
			// DW_fp_exp #(
			// 	sig_width,
			// 	exp_width,
			// 	ieee_compliance,
			// 	exp_arch
			// ) iexp_xi_max(
			// 	.a(i2flt_xi_max_dequant[((sig_width + exp_width) >= 0 ? 0 : sig_width + exp_width) + (i * ((sig_width + exp_width) >= 0 ? (sig_width + exp_width) + 1 : 1 - (sig_width + exp_width)))+:((sig_width + exp_width) >= 0 ? (sig_width + exp_width) + 1 : 1 - (sig_width + exp_width))]),
			// 	.z(exp_xi_max_w[((sig_width + exp_width) >= 0 ? 0 : sig_width + exp_width) + (i * ((sig_width + exp_width) >= 0 ? (sig_width + exp_width) + 1 : 1 - (sig_width + exp_width)))+:((sig_width + exp_width) >= 0 ? (sig_width + exp_width) + 1 : 1 - (sig_width + exp_width))]),
			// 	.status()
			// );
			fp_exp #(
				sig_width,
				exp_width
			) iexp_xi_max(
				.a(i2flt_xi_max_dequant[((sig_width + exp_width) >= 0 ? 0 : sig_width + exp_width) + (i * ((sig_width + exp_width) >= 0 ? (sig_width + exp_width) + 1 : 1 - (sig_width + exp_width)))+:((sig_width + exp_width) >= 0 ? (sig_width + exp_width) + 1 : 1 - (sig_width + exp_width))]),
				.z(exp_xi_max_w[((sig_width + exp_width) >= 0 ? 0 : sig_width + exp_width) + (i * ((sig_width + exp_width) >= 0 ? (sig_width + exp_width) + 1 : 1 - (sig_width + exp_width)))+:((sig_width + exp_width) >= 0 ? (sig_width + exp_width) + 1 : 1 - (sig_width + exp_width))])
			);
			always @(posedge clk or negedge rst_n)
				if (~rst_n)
					exp_xi_max[((sig_width + exp_width) >= 0 ? 0 : sig_width + exp_width) + (i * ((sig_width + exp_width) >= 0 ? (sig_width + exp_width) + 1 : 1 - (sig_width + exp_width)))+:((sig_width + exp_width) >= 0 ? (sig_width + exp_width) + 1 : 1 - (sig_width + exp_width))] <= 0;
				else if (i2flt_xi_max_dequant_vld_out)
					exp_xi_max[((sig_width + exp_width) >= 0 ? 0 : sig_width + exp_width) + (i * ((sig_width + exp_width) >= 0 ? (sig_width + exp_width) + 1 : 1 - (sig_width + exp_width)))+:((sig_width + exp_width) >= 0 ? (sig_width + exp_width) + 1 : 1 - (sig_width + exp_width))] <= exp_xi_max_w[((sig_width + exp_width) >= 0 ? 0 : sig_width + exp_width) + (i * ((sig_width + exp_width) >= 0 ? (sig_width + exp_width) + 1 : 1 - (sig_width + exp_width)))+:((sig_width + exp_width) >= 0 ? (sig_width + exp_width) + 1 : 1 - (sig_width + exp_width))];
		end
	endgenerate
	always @(posedge clk or negedge rst_n)
		if (~rst_n)
			exp_xi_max_vld <= 0;
		else
			exp_xi_max_vld <= i2flt_xi_max_dequant_vld_out;
	generate
		for (_gv_i_1 = 0; _gv_i_1 < EXP_STAGE_DELAY; _gv_i_1 = _gv_i_1 + 1) begin : exp_xi_max_retiming_stage
			localparam i = _gv_i_1;
			reg [((sig_width + exp_width) >= 0 ? (BUS_NUM * ((sig_width + exp_width) + 1)) - 1 : (BUS_NUM * (1 - (sig_width + exp_width))) + ((sig_width + exp_width) - 1)):((sig_width + exp_width) >= 0 ? 0 : (sig_width + exp_width) + 0)] timing_register_float;
			reg [BUS_NUM - 1:0] timing_register_vld;
			if (i == 0) begin : genblk1
				always @(posedge clk or negedge rst_n)
					if (~rst_n) begin
						timing_register_float <= 0;
						timing_register_vld <= 0;
					end
					else begin
						timing_register_float <= exp_xi_max;
						timing_register_vld <= exp_xi_max_vld;
					end
			end
			else begin : genblk1
				always @(posedge clk or negedge rst_n)
					if (~rst_n) begin
						timing_register_float <= 0;
						timing_register_vld <= 0;
					end
					else begin
						timing_register_float <= exp_xi_max_retiming_stage[i - 1].timing_register_float;
						timing_register_vld <= exp_xi_max_retiming_stage[i - 1].timing_register_vld;
					end
			end
		end
	endgenerate
	always @(*) begin
		if (_sv2v_0)
			;
		exp_xi_max_delay = exp_xi_max_retiming_stage[EXP_STAGE_DELAY - 1].timing_register_float;
		exp_xi_max_vld_delay = exp_xi_max_retiming_stage[EXP_STAGE_DELAY - 1].timing_register_vld;
	end
	assign softmax_exp_quant_scale_reg = rc_cfg_reg[15-:16];
	assign exp_xi_max_scaled_vld_out = &exp_xi_max_scaled_vld;
	generate
		for (_gv_i_1 = 0; _gv_i_1 < BUS_NUM; _gv_i_1 = _gv_i_1 + 1) begin : flt2i_quant_stage
			localparam i = _gv_i_1;
			wire [sig_width + exp_width:0] exp_xi_max_scaled;
			fp_mult_pipe #(
				.sig_width(sig_width),
				.exp_width(exp_width),
				.ieee_compliance(ieee_compliance),
				.stages(5)
			) inst_fp_quant(
				.clk(clk),
				.rst_n(rst_n),
				.a(exp_xi_max_delay[((sig_width + exp_width) >= 0 ? 0 : sig_width + exp_width) + (i * ((sig_width + exp_width) >= 0 ? (sig_width + exp_width) + 1 : 1 - (sig_width + exp_width)))+:((sig_width + exp_width) >= 0 ? (sig_width + exp_width) + 1 : 1 - (sig_width + exp_width))]),
				.b(softmax_exp_quant_scale_reg),
				.z(exp_xi_max_scaled),
				.ab_valid(exp_xi_max_vld_delay),
				.z_valid(exp_xi_max_scaled_vld[i])
			);
			wire [31:0] exp_xi_max_flt2i;
			fp_flt2i #(
				sig_width,
				exp_width,
				isize,
				isign
			) exp_flt2i_inst(
				.a(exp_xi_max_scaled),
				.z(exp_xi_max_flt2i)
			);
			reg signed [FIXED_DATA_WIDTH - 1:0] nxt_out_fixed_data;
			always @(*) begin
				if (_sv2v_0)
					;
				nxt_out_fixed_data = exp_xi_max_flt2i;
				if ($signed(exp_xi_max_flt2i) > 127)
					nxt_out_fixed_data = 8'b01111111;
				else if ($signed(exp_xi_max_flt2i) < -128)
					nxt_out_fixed_data = 8'b10000000;
			end
			always @(posedge clk or negedge rst_n)
				if (~rst_n)
					out_bus_data[i * FIXED_DATA_WIDTH+:FIXED_DATA_WIDTH] <= 0;
				else if (exp_xi_max_scaled_vld[i])
					out_bus_data[i * FIXED_DATA_WIDTH+:FIXED_DATA_WIDTH] <= nxt_out_fixed_data;
		end
	endgenerate
	reg [1:0] _sv2v_jump;
	always @(posedge clk or negedge rst_n)
		if (~rst_n)
			out_bus_data_vld <= 0;
		else
			out_bus_data_vld <= exp_xi_max_scaled_vld_out;
	always @(*) begin: sv2v_autoblock_2
		// 默认赋值（避免锁存器）
		integer m;
		// reg [31:0] _sv2v_value_on_break;
		m = 0;                         // 初始化循环变量
		// _sv2v_value_on_break = 0;      // 初始化临时变量

		// 原有逻辑（保持 _sv2v_jump 结构）
		nxt_exp_sum_cnt = exp_sum_cnt;
		nxt_fadd_tree_in = {((sig_width + exp_width + 1) * BUS_NUM){1'b0}};
		nxt_fadd_tree_vld = {BUS_NUM{1'b0}};
		nxt_fadd_tree_in_last = 1'b0;
		_sv2v_jump = 2'b00;

		if (clear) begin
			nxt_exp_sum_cnt = 10'b0;
		end
		else if (exp_xi_max_vld_delay) begin
			nxt_fadd_tree_in = exp_xi_max_delay;
			for (m = 0; m < BUS_NUM; m = m + 1) begin
				if (_sv2v_jump < 2'b10) begin
					_sv2v_jump = 2'b00;
					nxt_fadd_tree_vld[m] = 1'b1;
					nxt_exp_sum_cnt = exp_sum_cnt + 1;
					// _sv2v_value_on_break = m;  // 记录当前 m 的值

					if (usr_cfg_reg[0] && (nxt_exp_sum_cnt == usr_cfg_reg[9-:9] + 1)) begin
						nxt_fadd_tree_in_last = 1'b1;
						_sv2v_jump = 2'b10;     // 标记跳出循环
					end
					else if (!usr_cfg_reg[0] && (nxt_exp_sum_cnt == model_cfg_reg[29-:10])) begin
						nxt_fadd_tree_in_last = 1'b1;
						_sv2v_jump = 2'b10;     // 标记跳出循环
					end
				end
			end
		end
	end
	always @(posedge clk or negedge rst_n) begin
		if (~rst_n) begin
			fadd_tree_in <= 0;
			fadd_tree_vld <= 0;
			exp_sum_cnt <= 0;
			fadd_tree_in_last <= 0;
		end
		else begin
			fadd_tree_in <= nxt_fadd_tree_in;
			fadd_tree_vld <= nxt_fadd_tree_vld;
			exp_sum_cnt <= nxt_exp_sum_cnt;
			fadd_tree_in_last <= nxt_fadd_tree_in_last;
		end
	end
	fadd_tree #(
		.sig_width(sig_width),
		.exp_width(exp_width),
		.MAC_NUM(BUS_NUM)
	) fadd_tree_inst(
		.clk(clk),
		.rstn(rst_n),
		.idata(fadd_tree_in),
		.idata_valid(fadd_tree_vld),
		.last_in(fadd_tree_in_last),
		.odata(adder_out),
		.odata_valid(adder_out_vld),
		.last_out(adder_out_last)
	);
	fp_add_DG #(
		sig_width,
		exp_width
	) acc_inst(
		.a(acc_out),
		.b(adder_out),
		.rnd(3'b000),
		.DG_ctrl(adder_out_vld),
		.z(acc_out_w),
		.status()
	);
	always @(posedge clk or negedge rst_n) begin
		if (~rst_n)
			acc_out <= 0;
		else if (clear)
			acc_out <= 0;
		else if (adder_out_vld)
			acc_out <= acc_out_w;
	end
	always @(posedge clk or negedge rst_n) begin
		if (~rst_n)
			acc_vld <= 0;
		else
			acc_vld <= adder_out_last;
	end
	// DW_fp_recip #(
	// 	sig_width,
	// 	exp_width,
	// 	ieee_compliance,
	// 	0
	// ) inst_recip(
	// 	.a(acc_out),
	// 	.rnd(3'b000),
	// 	.z(one_over_exp_sum_w),
	// 	.status()
	// );
	bf16_fp_recip fp_recip_inst (
		.a(acc_out),
		.z(one_over_exp_sum_w)
	);
	always @(posedge clk or negedge rst_n)
		if (~rst_n) begin
			one_over_exp_sum_vld <= 0;
			one_over_exp_sum <= 0;
		end
		else begin
			one_over_exp_sum_vld <= acc_vld;
			if (acc_vld)
				one_over_exp_sum <= one_over_exp_sum_w;
		end
	always @(posedge clk or negedge rst_n)
		if (~rst_n) begin
			one_over_exp_sum_shift_vld <= 0;
			one_over_exp_sum_shift <= 0;
		end
		else begin
			one_over_exp_sum_shift_vld <= one_over_exp_sum_vld;
			if (one_over_exp_sum_vld) begin
				one_over_exp_sum_shift[sig_width + exp_width] <= one_over_exp_sum[sig_width + exp_width];
				one_over_exp_sum_shift[(sig_width + exp_width) - 1:0] <= one_over_exp_sum[(sig_width + exp_width) - 1:0] + $unsigned({rc_cfg_reg[36-:5], {sig_width {1'b0}}});
			end
		end
	// DW_fp_flt2i #(
	// 	sig_width,
	// 	exp_width,
	// 	24,
	// 	isign
	// ) i2flt_in_data(
	// 	.a(one_over_exp_sum_shift),
	// 	.rnd(3'b000),
	// 	.z(nxt_rc_scale),
	// 	.status()
	// );
	fp_flt2i #(
		sig_width,
		exp_width,
		24,
		isign
	) i2flt_in_data(
		.a(one_over_exp_sum_shift),
		.z(nxt_rc_scale)
	);
	always @(posedge clk or negedge rst_n)
		if (~rst_n) begin
			rc_scale <= 0;
			rc_scale_vld <= 0;
		end
		else begin
			rc_scale_vld <= one_over_exp_sum_shift_vld;
			if (one_over_exp_sum_shift_vld)
				rc_scale <= nxt_rc_scale;
		end
	initial _sv2v_0 = 0;
endmodule

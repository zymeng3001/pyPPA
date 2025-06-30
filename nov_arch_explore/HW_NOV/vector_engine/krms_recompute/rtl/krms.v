module krms (
	clk,
	rst_n,
	start,
	rc_cfg_vld,
	rc_cfg,
	control_state,
	control_state_update,
	in_fixed_data,
	in_fixed_data_vld,
	rc_scale,
	rc_scale_vld
);
	reg _sv2v_0;
	parameter integer BUS_NUM = 8;
	parameter integer DATA_NUM_WIDTH = 10;
	parameter integer FIXED_SQUARE_SUM_WIDTH = 24;
	parameter integer sig_width = 7;
	parameter integer exp_width = 8;
	localparam isize = 32;
	localparam isign = 1;
	localparam ieee_compliance = 0;
	input wire clk;
	input wire rst_n;
	input wire start;
	input wire rc_cfg_vld;
	input wire [83:0] rc_cfg;
	input wire [31:0] control_state;
	input wire control_state_update;
	input wire signed [(BUS_NUM * 8) - 1:0] in_fixed_data;
	input wire in_fixed_data_vld;
	output reg [23:0] rc_scale;
	output reg rc_scale_vld;
	reg start_flag;
	always @(posedge clk or negedge rst_n)
		if (~rst_n)
			start_flag <= 0;
		else if (start)
			start_flag <= 1;
		else if (rc_scale_vld)
			start_flag <= 0;
	reg [83:0] rc_cfg_reg;
	always @(posedge clk or negedge rst_n)
		if (~rst_n)
			rc_cfg_reg <= 0;
		else if (rc_cfg_vld)
			rc_cfg_reg <= rc_cfg;
	reg [31:0] control_state_reg;
	always @(posedge clk or negedge rst_n)
		if (~rst_n)
			control_state_reg <= 32'd0;
		else if (control_state_update)
			control_state_reg <= control_state;
	wire [31:0] K_ext;
	reg [sig_width + exp_width:0] float_K;
	wire [sig_width + exp_width:0] nxt_float_K;
	assign K_ext = rc_cfg_reg[78-:10];
	always @(posedge clk) float_K <= nxt_float_K;
	// DW_fp_i2flt #(
	// 	sig_width,
	// 	exp_width,
	// 	isize,
	// 	isign
	// ) i2flt_K(
	// 	.a(K_ext),
	// 	.rnd(3'b000),
	// 	.z(nxt_float_K),
	// 	.status()
	// );
	fp_i2flt #(
		sig_width,
		exp_width,
		isize,
		isign
	) i2flt_K(
		.a(K_ext),
		.z(nxt_float_K),
	);
	reg [sig_width + exp_width:0] rms_dequant_scale_square_reg;
	always @(*) begin
		if (_sv2v_0)
			;
		if (control_state_reg == 32'd7)
			rms_dequant_scale_square_reg = rc_cfg_reg[52-:16];
		else
			rms_dequant_scale_square_reg = rc_cfg_reg[68-:16];
	end
	reg fixed_data_square_vld;
	reg signed [(BUS_NUM * (2 * DATA_NUM_WIDTH)) - 1:0] fixed_data_square;
	always @(posedge clk or negedge rst_n)
		if (~rst_n)
			fixed_data_square_vld <= 0;
		else
			fixed_data_square_vld <= in_fixed_data_vld & start_flag;
	genvar _gv_i_1;
	generate
		for (_gv_i_1 = 0; _gv_i_1 < BUS_NUM; _gv_i_1 = _gv_i_1 + 1) begin : fixed_square_sum_generate_array
			localparam i = _gv_i_1;
			always @(posedge clk)
				if (in_fixed_data_vld && start_flag)
					fixed_data_square[i * (2 * DATA_NUM_WIDTH)+:2 * DATA_NUM_WIDTH] <= $signed(in_fixed_data[i * 8+:8]) * $signed(in_fixed_data[i * 8+:8]);
				else
					fixed_data_square[i * (2 * DATA_NUM_WIDTH)+:2 * DATA_NUM_WIDTH] <= 0;
		end
	endgenerate
	reg [FIXED_SQUARE_SUM_WIDTH - 1:0] fixed_square_sum;
	reg [FIXED_SQUARE_SUM_WIDTH - 1:0] nxt_fixed_square_sum;
	reg [DATA_NUM_WIDTH - 1:0] square_sum_cnt;
	reg [DATA_NUM_WIDTH - 1:0] nxt_square_sum_cnt;
	reg fixed_square_sum_vld;
	always @(*) begin : sv2v_autoblock_1
		reg [0:1] _sv2v_jump;
		_sv2v_jump = 2'b00;
		if (_sv2v_0)
			;
		nxt_square_sum_cnt = square_sum_cnt;
		nxt_fixed_square_sum = fixed_square_sum;
		begin : sv2v_autoblock_2
			reg signed [31:0] i;
			begin : sv2v_autoblock_3
				reg signed [31:0] _sv2v_value_on_break;
				for (i = 0; i < BUS_NUM; i = i + 1)
					if (_sv2v_jump < 2'b10) begin
						_sv2v_jump = 2'b00;
						if (fixed_data_square_vld) begin
							nxt_square_sum_cnt = nxt_square_sum_cnt + 1;
							nxt_fixed_square_sum = nxt_fixed_square_sum + fixed_data_square[i * (2 * DATA_NUM_WIDTH)+:2 * DATA_NUM_WIDTH];
							if (nxt_square_sum_cnt == rc_cfg_reg[78-:10])
								_sv2v_jump = 2'b10;
						end
						_sv2v_value_on_break = i;
					end
				if (!(_sv2v_jump < 2'b10))
					i = _sv2v_value_on_break;
				if (_sv2v_jump != 2'b11)
					_sv2v_jump = 2'b00;
			end
		end
	end
	always @(posedge clk or negedge rst_n)
		if (~rst_n) begin
			square_sum_cnt <= 0;
			fixed_square_sum <= 0;
		end
		else if (rc_scale_vld) begin
			square_sum_cnt <= 0;
			fixed_square_sum <= 0;
		end
		else if (square_sum_cnt == rc_cfg_reg[78-:10])
			square_sum_cnt <= rc_cfg_reg[78-:10];
		else if (fixed_data_square_vld) begin
			square_sum_cnt <= nxt_square_sum_cnt;
			fixed_square_sum <= nxt_fixed_square_sum;
		end
	always @(posedge clk or negedge rst_n)
		if (~rst_n)
			fixed_square_sum_vld <= 0;
		else if (fixed_square_sum_vld == 1)
			fixed_square_sum_vld <= 0;
		else if (((nxt_square_sum_cnt == rc_cfg_reg[78-:10]) && (square_sum_cnt < rc_cfg_reg[78-:10])) && (rc_cfg_reg[78-:10] != 0))
			fixed_square_sum_vld <= 1;
	wire [31:0] fixed_square_sum_ext;
	wire [sig_width + exp_width:0] i2flt_square_sum_z;
	wire [sig_width + exp_width:0] flt_square_sum;
	wire flt_square_sum_vld;
	assign fixed_square_sum_ext = fixed_square_sum;
	// DW_fp_i2flt #(
	// 	sig_width,
	// 	exp_width,
	// 	isize,
	// 	isign
	// ) i2flt_square_sum(
	// 	.a(fixed_square_sum_ext),
	// 	.rnd(3'b000),
	// 	.z(i2flt_square_sum_z),
	// 	.status()
	// );
	fp_i2flt #(
		sig_width,
		exp_width,
		isize,
		isign
	) i2flt_square_sum(
		.a(fixed_square_sum_ext),
		.z(i2flt_square_sum_z),
	);
	fp_mult_pipe #(
		.sig_width(sig_width),
		.exp_width(exp_width),
		.ieee_compliance(ieee_compliance),
		.stages(5)
	) inst_fp_mult_suqare_sum(
		.clk(clk),
		.rst_n(rst_n),
		.a(i2flt_square_sum_z),
		.b(rms_dequant_scale_square_reg),
		.z(flt_square_sum),
		.ab_valid(fixed_square_sum_vld),
		.z_valid(flt_square_sum_vld)
	);
	wire [sig_width + exp_width:0] square_sum_over_K;
	wire square_sum_over_K_vld;
	fp_div_pipe #(
		.sig_width(sig_width),
		.exp_width(exp_width),
		.ieee_compliance(ieee_compliance),
		.stages(5)
	) div_pipe_inst(
		.clk(clk),
		.rst_n(rst_n),
		.a(flt_square_sum),
		.b(float_K),
		.z(square_sum_over_K),
		.ab_valid(flt_square_sum_vld),
		.z_valid(square_sum_over_K_vld)
	);
	localparam SLICE_REG_NUM = 4;
	reg [sig_width + exp_width:0] square_sum_over_K_delay;
	reg square_sum_over_K_vld_delay;
	generate
		for (_gv_i_1 = 0; _gv_i_1 < SLICE_REG_NUM; _gv_i_1 = _gv_i_1 + 1) begin : invsqrt_retiming_gen_array
			localparam i = _gv_i_1;
			reg [sig_width + exp_width:0] timing_register_float;
			reg timing_register;
			if (i == 0) begin : genblk1
				always @(posedge clk or negedge rst_n)
					if (~rst_n) begin
						timing_register_float <= 0;
						timing_register <= 0;
					end
					else begin
						timing_register_float <= square_sum_over_K;
						timing_register <= square_sum_over_K_vld;
					end
			end
			else begin : genblk1
				always @(posedge clk or negedge rst_n)
					if (~rst_n) begin
						timing_register_float <= 0;
						timing_register <= 0;
					end
					else begin
						timing_register_float <= invsqrt_retiming_gen_array[i - 1].timing_register_float;
						timing_register <= invsqrt_retiming_gen_array[i - 1].timing_register;
					end
			end
		end
	endgenerate
	always @(posedge clk or negedge rst_n)
		if (~rst_n) begin
			square_sum_over_K_delay <= 0;
			square_sum_over_K_vld_delay <= 0;
		end
		else begin
			square_sum_over_K_delay <= invsqrt_retiming_gen_array[3].timing_register_float;
			square_sum_over_K_vld_delay <= invsqrt_retiming_gen_array[3].timing_register;
		end
	wire invsqrt_z_vld;
	wire [sig_width + exp_width:0] invsqrt_z;
	reg [sig_width + exp_width:0] one_over_rms;
	reg one_over_rms_vld;
	fp_invsqrt_pipe inst_fp_invsqrt_pipe(
		.clk(clk),
		.rst_n(rst_n),
		.x(square_sum_over_K_delay),
		.x_vld(square_sum_over_K_vld_delay),
		.y(invsqrt_z),
		.y_vld(invsqrt_z_vld)
	);
	always @(posedge clk or negedge rst_n)
		if (~rst_n) begin
			one_over_rms <= 0;
			one_over_rms_vld <= 0;
		end
		else begin
			one_over_rms <= invsqrt_z;
			one_over_rms_vld <= invsqrt_z_vld;
		end
	reg [sig_width + exp_width:0] one_over_rms_exp_shift;
	reg one_over_rms_exp_shift_vld;
	always @(posedge clk or negedge rst_n)
		if (~rst_n) begin
			one_over_rms_exp_shift_vld <= 0;
			one_over_rms_exp_shift <= 0;
		end
		else begin
			one_over_rms_exp_shift_vld <= one_over_rms_vld;
			one_over_rms_exp_shift[sig_width + exp_width] <= one_over_rms[sig_width + exp_width];
			one_over_rms_exp_shift[(sig_width + exp_width) - 1:0] <= one_over_rms[(sig_width + exp_width) - 1:0] + $unsigned({rc_cfg_reg[83-:5], {sig_width {1'b0}}});
		end
	wire [23:0] nxt_rc_scale;
	// DW_fp_flt2i #(
	// 	sig_width,
	// 	exp_width,
	// 	24,
	// 	isign
	// ) i2flt_in_data(
	// 	.a(one_over_rms_exp_shift),
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
		.a(one_over_rms_exp_shift),
		.z(nxt_rc_scale)
	);
	always @(posedge clk or negedge rst_n)
		if (~rst_n) begin
			rc_scale <= 0;
			rc_scale_vld <= 0;
		end
		else begin
			rc_scale <= nxt_rc_scale;
			rc_scale_vld <= one_over_rms_exp_shift_vld;
		end
	initial _sv2v_0 = 0;
endmodule

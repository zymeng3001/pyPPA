module rms_norm (
	clk,
	rst_n,
	in_data_num,
	in_data_num_vld,
	in_gamma,
	in_gamma_vld,
	in_fixed_data,
	in_fixed_data_vld,
	in_scale_pos,
	in_scale_pos_vld,
	out_scale_pos,
	out_scale_pos_vld,
	in_K,
	in_K_vld,
	out_fixed_data,
	out_fixed_data_vld,
	out_fixed_data_last
);
	reg _sv2v_0;
	parameter integer BUS_NUM = 8;
	parameter integer DATA_NUM_WIDTH = 10;
	parameter integer SCALA_POS_WIDTH = 5;
	parameter integer FIXED_SQUARE_SUM_WIDTH = 24;
	parameter integer IN_FLOAT_DATA_ARRAY_DEPTH = (384 / BUS_NUM) + 1;
	parameter integer GAMMA_ARRAY_DEPTH = IN_FLOAT_DATA_ARRAY_DEPTH;
	parameter integer sig_width = 7;
	parameter integer exp_width = 8;
	localparam isize = 32;
	localparam isign = 1;
	localparam ieee_compliance = 0;
	input wire clk;
	input wire rst_n;
	input wire [DATA_NUM_WIDTH - 1:0] in_data_num;
	input wire in_data_num_vld;
	input wire [((sig_width + exp_width) >= 0 ? (BUS_NUM * ((sig_width + exp_width) + 1)) - 1 : (BUS_NUM * (1 - (sig_width + exp_width))) + ((sig_width + exp_width) - 1)):((sig_width + exp_width) >= 0 ? 0 : (sig_width + exp_width) + 0)] in_gamma;
	input wire [BUS_NUM - 1:0] in_gamma_vld;
	input wire signed [(BUS_NUM * 8) - 1:0] in_fixed_data;
	input wire [BUS_NUM - 1:0] in_fixed_data_vld;
	input wire signed [SCALA_POS_WIDTH - 1:0] in_scale_pos;
	input wire in_scale_pos_vld;
	input wire signed [SCALA_POS_WIDTH - 1:0] out_scale_pos;
	input wire out_scale_pos_vld;
	input wire [DATA_NUM_WIDTH - 1:0] in_K;
	input wire in_K_vld;
	output reg signed [(BUS_NUM * 8) - 1:0] out_fixed_data;
	output reg [BUS_NUM - 1:0] out_fixed_data_vld;
	output reg out_fixed_data_last;
	reg [DATA_NUM_WIDTH - 1:0] data_num;
	reg [DATA_NUM_WIDTH - 1:0] K;
	reg [sig_width + exp_width:0] float_K;
	wire [sig_width + exp_width:0] nxt_float_K;
	reg signed [SCALA_POS_WIDTH - 1:0] in_scale_pos_reg;
	reg signed [SCALA_POS_WIDTH - 1:0] out_scale_pos_reg;
	reg signed [(BUS_NUM * (2 * DATA_NUM_WIDTH)) - 1:0] fixed_data_square;
	reg [BUS_NUM - 1:0] fixed_data_square_vld;
	reg [FIXED_SQUARE_SUM_WIDTH - 1:0] fixed_square_sum;
	reg [FIXED_SQUARE_SUM_WIDTH - 1:0] nxt_fixed_square_sum;
	reg [DATA_NUM_WIDTH - 1:0] square_sum_cnt;
	reg [DATA_NUM_WIDTH - 1:0] nxt_square_sum_cnt;
	reg fixed_square_sum_vld;
	wire [sig_width + exp_width:0] i2flt_square_sum_z;
	reg [sig_width + exp_width:0] float_square_sum;
	reg float_square_sum_vld;
	wire [sig_width + exp_width:0] square_sum_over_K;
	wire square_sum_over_K_vld;
	reg [sig_width + exp_width:0] one_over_rms;
	wire [sig_width + exp_width:0] invsqrt_z;
	reg one_over_rms_vld;
	wire [((sig_width + exp_width) >= 0 ? (BUS_NUM * ((sig_width + exp_width) + 1)) - 1 : (BUS_NUM * (1 - (sig_width + exp_width))) + ((sig_width + exp_width) - 1)):((sig_width + exp_width) >= 0 ? 0 : (sig_width + exp_width) + 0)] i2flt_in_data_z;
	reg [((sig_width + exp_width) >= 0 ? (BUS_NUM * ((sig_width + exp_width) + 1)) - 1 : (BUS_NUM * (1 - (sig_width + exp_width))) + ((sig_width + exp_width) - 1)):((sig_width + exp_width) >= 0 ? 0 : (sig_width + exp_width) + 0)] in_float_data;
	reg [BUS_NUM - 1:0] in_float_data_vld;
	reg [(((sig_width + exp_width) + 1) * BUS_NUM) - 1:0] in_float_data_array [IN_FLOAT_DATA_ARRAY_DEPTH - 1:0];
	reg [BUS_NUM - 1:0] in_float_data_vld_array [IN_FLOAT_DATA_ARRAY_DEPTH - 1:0];
	wire [(((sig_width + exp_width) + 1) * BUS_NUM) - 1:0] in_float_data_array_wr_slot;
	reg [(((sig_width + exp_width) + 1) * BUS_NUM) - 1:0] in_float_data_array_rd_slot;
	wire [BUS_NUM - 1:0] in_float_data_vld_array_slot;
	reg [$clog2(IN_FLOAT_DATA_ARRAY_DEPTH) - 1:0] in_float_data_array_wr_ptr;
	reg [$clog2(IN_FLOAT_DATA_ARRAY_DEPTH) - 1:0] in_float_data_array_rd_ptr;
	reg [(((sig_width + exp_width) + 1) * BUS_NUM) - 1:0] gamma_array [GAMMA_ARRAY_DEPTH - 1:0];
	wire [(((sig_width + exp_width) + 1) * BUS_NUM) - 1:0] gamma_array_wr_slot;
	reg [(((sig_width + exp_width) + 1) * BUS_NUM) - 1:0] gamma_array_rd_slot;
	reg [BUS_NUM - 1:0] gamma_vld_array [GAMMA_ARRAY_DEPTH - 1:0];
	reg [$clog2(GAMMA_ARRAY_DEPTH) - 1:0] gamma_array_wr_ptr;
	reg [$clog2(GAMMA_ARRAY_DEPTH) - 1:0] gamma_array_rd_ptr;
	reg internal_array_rd_en;
	wire internal_array_rd_en_delay1;
	reg [$clog2(GAMMA_ARRAY_DEPTH * BUS_NUM) - 1:0] internal_array_rd_cnt;
	wire [((sig_width + exp_width) >= 0 ? (BUS_NUM * ((sig_width + exp_width) + 1)) - 1 : (BUS_NUM * (1 - (sig_width + exp_width))) + ((sig_width + exp_width) - 1)):((sig_width + exp_width) >= 0 ? 0 : (sig_width + exp_width) + 0)] x_float;
	wire [((sig_width + exp_width) >= 0 ? (BUS_NUM * ((sig_width + exp_width) + 1)) - 1 : (BUS_NUM * (1 - (sig_width + exp_width))) + ((sig_width + exp_width) - 1)):((sig_width + exp_width) >= 0 ? 0 : (sig_width + exp_width) + 0)] gamma_float;
	reg [BUS_NUM - 1:0] x_and_gamma_vld;
	wire [((sig_width + exp_width) >= 0 ? (BUS_NUM * ((sig_width + exp_width) + 1)) - 1 : (BUS_NUM * (1 - (sig_width + exp_width))) + ((sig_width + exp_width) - 1)):((sig_width + exp_width) >= 0 ? 0 : (sig_width + exp_width) + 0)] x_mult_with_gamma;
	wire [BUS_NUM - 1:0] x_mult_with_gamma_vld;
	wire [((sig_width + exp_width) >= 0 ? (BUS_NUM * ((sig_width + exp_width) + 1)) - 1 : (BUS_NUM * (1 - (sig_width + exp_width))) + ((sig_width + exp_width) - 1)):((sig_width + exp_width) >= 0 ? 0 : (sig_width + exp_width) + 0)] float_RMSnorm;
	wire [BUS_NUM - 1:0] float_RMSnorm_vld;
	reg [((sig_width + exp_width) >= 0 ? (BUS_NUM * ((sig_width + exp_width) + 1)) - 1 : (BUS_NUM * (1 - (sig_width + exp_width))) + ((sig_width + exp_width) - 1)):((sig_width + exp_width) >= 0 ? 0 : (sig_width + exp_width) + 0)] float_RMSnorm_flt2i;
	reg [BUS_NUM - 1:0] float_RMSnorm_flt2i_vld;
	wire signed [(BUS_NUM * 32) - 1:0] fixed_RMSnorm;
	reg signed [(BUS_NUM * 8) - 1:0] nxt_out_fixed_data;
	reg [15:0] out_fixed_data_last_cnt;
	always @(posedge clk or negedge rst_n)
		if (~rst_n)
			data_num <= 0;
		else if (in_data_num_vld)
			data_num <= in_data_num;
	always @(posedge clk or negedge rst_n)
		if (~rst_n)
			in_scale_pos_reg <= 0;
		else if (in_scale_pos_vld)
			in_scale_pos_reg <= in_scale_pos;
	always @(posedge clk or negedge rst_n)
		if (~rst_n)
			out_scale_pos_reg <= 0;
		else if (out_scale_pos_vld)
			out_scale_pos_reg <= out_scale_pos;
	genvar _gv_i_1;
	generate
		for (_gv_i_1 = 0; _gv_i_1 < BUS_NUM; _gv_i_1 = _gv_i_1 + 1) begin : gamma_array_wr_slot_generate_array
			localparam i = _gv_i_1;
			assign gamma_array_wr_slot[i * ((sig_width + exp_width) + 1)+:(sig_width + exp_width) + 1] = in_gamma[((sig_width + exp_width) >= 0 ? 0 : sig_width + exp_width) + (i * ((sig_width + exp_width) >= 0 ? (sig_width + exp_width) + 1 : 1 - (sig_width + exp_width)))+:((sig_width + exp_width) >= 0 ? (sig_width + exp_width) + 1 : 1 - (sig_width + exp_width))];
		end
	endgenerate
	always @(posedge clk or negedge rst_n)
		if (~rst_n)
			gamma_array_wr_ptr <= 0;
		else if (out_fixed_data_last)
			gamma_array_wr_ptr <= 0;
		else if (|in_gamma_vld) begin
			gamma_array[gamma_array_wr_ptr] <= gamma_array_wr_slot;
			gamma_vld_array[gamma_array_wr_ptr] <= in_gamma_vld;
			gamma_array_wr_ptr <= gamma_array_wr_ptr + 1;
		end
	wire [31:0] K_ext;
	assign K_ext = K;
	always @(posedge clk or negedge rst_n)
		if (~rst_n)
			K <= 0;
		else if (in_K_vld)
			K <= in_K;
	always @(posedge clk) float_K <= nxt_float_K;
	DW_fp_i2flt #(
		sig_width,
		exp_width,
		isize,
		isign
	) i2flt_K(
		.a(K_ext),
		.rnd(3'b000),
		.z(nxt_float_K),
		.status()
	);
	wire signed [(BUS_NUM * 32) - 1:0] in_fixed_data_ext;
	generate
		for (_gv_i_1 = 0; _gv_i_1 < BUS_NUM; _gv_i_1 = _gv_i_1 + 1) begin : i2flt_in_data_generate_array
			localparam i = _gv_i_1;
			assign in_fixed_data_ext[i * 32+:32] = $signed(in_fixed_data[i * 8+:8]);
			DW_fp_i2flt #(
				sig_width,
				exp_width,
				isize,
				isign
			) i2flt_in_data(
				.a($signed(in_fixed_data_ext[i * 32+:32])),
				.rnd(3'b000),
				.z(i2flt_in_data_z[((sig_width + exp_width) >= 0 ? 0 : sig_width + exp_width) + (i * ((sig_width + exp_width) >= 0 ? (sig_width + exp_width) + 1 : 1 - (sig_width + exp_width)))+:((sig_width + exp_width) >= 0 ? (sig_width + exp_width) + 1 : 1 - (sig_width + exp_width))]),
				.status()
			);
		end
		for (_gv_i_1 = 0; _gv_i_1 < BUS_NUM; _gv_i_1 = _gv_i_1 + 1) begin : in_fix2float_generate_array
			localparam i = _gv_i_1;
			always @(posedge clk or negedge rst_n)
				if (~rst_n) begin
					in_float_data_vld[i] <= 0;
					in_float_data[((sig_width + exp_width) >= 0 ? 0 : sig_width + exp_width) + (i * ((sig_width + exp_width) >= 0 ? (sig_width + exp_width) + 1 : 1 - (sig_width + exp_width)))+:((sig_width + exp_width) >= 0 ? (sig_width + exp_width) + 1 : 1 - (sig_width + exp_width))] <= 0;
				end
				else begin
					in_float_data_vld[i] <= in_fixed_data_vld[i];
					if (i2flt_in_data_z[((sig_width + exp_width) >= 0 ? 0 : sig_width + exp_width) + (i * ((sig_width + exp_width) >= 0 ? (sig_width + exp_width) + 1 : 1 - (sig_width + exp_width)))+:((sig_width + exp_width) >= 0 ? (sig_width + exp_width) + 1 : 1 - (sig_width + exp_width))] == 0)
						in_float_data[((sig_width + exp_width) >= 0 ? 0 : sig_width + exp_width) + (i * ((sig_width + exp_width) >= 0 ? (sig_width + exp_width) + 1 : 1 - (sig_width + exp_width)))+:((sig_width + exp_width) >= 0 ? (sig_width + exp_width) + 1 : 1 - (sig_width + exp_width))] <= 0;
					else begin
						in_float_data[((sig_width + exp_width) >= 0 ? (i * ((sig_width + exp_width) >= 0 ? (sig_width + exp_width) + 1 : 1 - (sig_width + exp_width))) + ((sig_width + exp_width) >= 0 ? (sig_width + exp_width) - 1 : (sig_width + exp_width) - ((sig_width + exp_width) - 1)) : (((i * ((sig_width + exp_width) >= 0 ? (sig_width + exp_width) + 1 : 1 - (sig_width + exp_width))) + ((sig_width + exp_width) >= 0 ? (sig_width + exp_width) - 1 : (sig_width + exp_width) - ((sig_width + exp_width) - 1))) + (sig_width + exp_width)) - 1)-:sig_width + exp_width] <= i2flt_in_data_z[((sig_width + exp_width) >= 0 ? (i * ((sig_width + exp_width) >= 0 ? (sig_width + exp_width) + 1 : 1 - (sig_width + exp_width))) + ((sig_width + exp_width) >= 0 ? (sig_width + exp_width) - 1 : (sig_width + exp_width) - ((sig_width + exp_width) - 1)) : (((i * ((sig_width + exp_width) >= 0 ? (sig_width + exp_width) + 1 : 1 - (sig_width + exp_width))) + ((sig_width + exp_width) >= 0 ? (sig_width + exp_width) - 1 : (sig_width + exp_width) - ((sig_width + exp_width) - 1))) + (sig_width + exp_width)) - 1)-:sig_width + exp_width] - $signed({in_scale_pos_reg, {sig_width {1'b0}}});
						in_float_data[(i * ((sig_width + exp_width) >= 0 ? (sig_width + exp_width) + 1 : 1 - (sig_width + exp_width))) + ((sig_width + exp_width) >= 0 ? sig_width + exp_width : (sig_width + exp_width) - (sig_width + exp_width))] <= i2flt_in_data_z[(i * ((sig_width + exp_width) >= 0 ? (sig_width + exp_width) + 1 : 1 - (sig_width + exp_width))) + ((sig_width + exp_width) >= 0 ? sig_width + exp_width : (sig_width + exp_width) - (sig_width + exp_width))];
					end
				end
			assign in_float_data_array_wr_slot[i * ((sig_width + exp_width) + 1)+:(sig_width + exp_width) + 1] = in_float_data[((sig_width + exp_width) >= 0 ? 0 : sig_width + exp_width) + (i * ((sig_width + exp_width) >= 0 ? (sig_width + exp_width) + 1 : 1 - (sig_width + exp_width)))+:((sig_width + exp_width) >= 0 ? (sig_width + exp_width) + 1 : 1 - (sig_width + exp_width))];
			assign in_float_data_vld_array_slot[i] = in_float_data_vld[i];
		end
	endgenerate
	always @(posedge clk or negedge rst_n)
		if (~rst_n)
			in_float_data_array_wr_ptr <= 0;
		else if (out_fixed_data_last)
			in_float_data_array_wr_ptr <= 0;
		else if (|in_float_data_vld) begin
			in_float_data_array[in_float_data_array_wr_ptr] <= in_float_data_array_wr_slot;
			in_float_data_vld_array[in_float_data_array_wr_ptr] <= in_float_data_vld_array_slot;
			in_float_data_array_wr_ptr <= in_float_data_array_wr_ptr + 1;
		end
	reg fixed_data_square_input_gating;
	always @(posedge clk or negedge rst_n)
		if (~rst_n)
			fixed_data_square_input_gating <= 0;
		else if (out_fixed_data_last)
			fixed_data_square_input_gating <= 0;
		else if (fixed_square_sum_vld)
			fixed_data_square_input_gating <= 1;
	generate
		for (_gv_i_1 = 0; _gv_i_1 < BUS_NUM; _gv_i_1 = _gv_i_1 + 1) begin : fixed_square_sum_generate_array
			localparam i = _gv_i_1;
			always @(posedge clk)
				if (~fixed_data_square_input_gating) begin
					fixed_data_square[i * (2 * DATA_NUM_WIDTH)+:2 * DATA_NUM_WIDTH] <= $signed(in_fixed_data[i * 8+:8]) * $signed(in_fixed_data[i * 8+:8]);
					fixed_data_square_vld[i] <= in_fixed_data_vld[i];
				end
				else begin
					fixed_data_square[i * (2 * DATA_NUM_WIDTH)+:2 * DATA_NUM_WIDTH] <= 0;
					fixed_data_square_vld[i] <= 0;
				end
		end
	endgenerate
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
						if (fixed_data_square_vld[i]) begin
							nxt_square_sum_cnt = nxt_square_sum_cnt + 1;
							nxt_fixed_square_sum = nxt_fixed_square_sum + fixed_data_square[i * (2 * DATA_NUM_WIDTH)+:2 * DATA_NUM_WIDTH];
							if (nxt_square_sum_cnt == K)
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
		else if (out_fixed_data_last) begin
			square_sum_cnt <= 0;
			fixed_square_sum <= 0;
		end
		else if (square_sum_cnt == K)
			square_sum_cnt <= K;
		else if (|fixed_data_square_vld) begin
			square_sum_cnt <= nxt_square_sum_cnt;
			fixed_square_sum <= nxt_fixed_square_sum;
		end
	always @(posedge clk or negedge rst_n)
		if (~rst_n)
			fixed_square_sum_vld <= 0;
		else if (fixed_square_sum_vld == 1)
			fixed_square_sum_vld <= 0;
		else if (((nxt_square_sum_cnt == K) && (square_sum_cnt < K)) && (K != 0))
			fixed_square_sum_vld <= 1;
	wire [31:0] fixed_square_sum_ext;
	assign fixed_square_sum_ext = fixed_square_sum;
	DW_fp_i2flt #(
		sig_width,
		exp_width,
		isize,
		isign
	) i2flt_square_sum(
		.a(fixed_square_sum_ext),
		.rnd(3'b000),
		.z(i2flt_square_sum_z),
		.status()
	);
	always @(posedge clk or negedge rst_n)
		if (~rst_n) begin
			float_square_sum <= 0;
			float_square_sum_vld <= 0;
		end
		else if (out_fixed_data_last) begin
			float_square_sum_vld <= 0;
			float_square_sum <= 0;
		end
		else if (fixed_square_sum_vld) begin
			float_square_sum_vld <= 1;
			if (i2flt_square_sum_z == 0)
				float_square_sum <= 0;
			else begin
				float_square_sum[(sig_width + exp_width) - 1:0] <= i2flt_square_sum_z[(sig_width + exp_width) - 1:0] - $signed({in_scale_pos_reg, 1'b0, {sig_width {1'b0}}});
				float_square_sum[sig_width + exp_width] <= i2flt_square_sum_z[sig_width + exp_width];
			end
		end
		else
			float_square_sum_vld <= 0;
	fp_div_pipe #(
		.sig_width(sig_width),
		.exp_width(exp_width),
		.ieee_compliance(ieee_compliance),
		.stages(4)
	) div_pipe_inst(
		.clk(clk),
		.rst_n(rst_n),
		.a(float_square_sum),
		.b(float_K),
		.z(square_sum_over_K),
		.ab_valid(float_square_sum_vld),
		.z_valid(square_sum_over_K_vld)
	);
	localparam SLICE_REG_NUM = 1;
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
			square_sum_over_K_delay <= invsqrt_retiming_gen_array[0].timing_register_float;
			square_sum_over_K_vld_delay <= invsqrt_retiming_gen_array[0].timing_register;
		end
	wire invsqrt_z_vld;
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
	always @(posedge clk or negedge rst_n)
		if (~rst_n)
			internal_array_rd_en <= 0;
		else if (one_over_rms_vld)
			internal_array_rd_en <= 1;
		else if (((internal_array_rd_cnt >= (data_num - BUS_NUM)) && (data_num != 0)) && internal_array_rd_en)
			internal_array_rd_en <= 0;
	always @(posedge clk or negedge rst_n)
		if (~rst_n)
			internal_array_rd_cnt <= 0;
		else if (out_fixed_data_last)
			internal_array_rd_cnt <= 0;
		else if (internal_array_rd_en)
			internal_array_rd_cnt <= internal_array_rd_cnt + BUS_NUM;
	always @(posedge clk or negedge rst_n)
		if (~rst_n) begin
			in_float_data_array_rd_ptr <= 0;
			gamma_array_rd_ptr <= 0;
			in_float_data_array_rd_slot <= 0;
			gamma_array_rd_slot <= 0;
		end
		else if (out_fixed_data_last) begin
			in_float_data_array_rd_ptr <= 0;
			gamma_array_rd_ptr <= 0;
		end
		else if (internal_array_rd_en) begin
			in_float_data_array_rd_ptr <= in_float_data_array_rd_ptr + 1;
			gamma_array_rd_ptr <= gamma_array_rd_ptr + 1;
			in_float_data_array_rd_slot <= in_float_data_array[in_float_data_array_rd_ptr];
			gamma_array_rd_slot <= gamma_array[gamma_array_rd_ptr];
		end
	always @(posedge clk or negedge rst_n)
		if (~rst_n)
			x_and_gamma_vld <= 0;
		else
			x_and_gamma_vld <= in_float_data_vld_array[in_float_data_array_rd_ptr] & {BUS_NUM {internal_array_rd_en}};
	generate
		for (_gv_i_1 = 0; _gv_i_1 < BUS_NUM; _gv_i_1 = _gv_i_1 + 1) begin : x_gamma_float_generate_array
			localparam i = _gv_i_1;
			assign x_float[((sig_width + exp_width) >= 0 ? 0 : sig_width + exp_width) + (i * ((sig_width + exp_width) >= 0 ? (sig_width + exp_width) + 1 : 1 - (sig_width + exp_width)))+:((sig_width + exp_width) >= 0 ? (sig_width + exp_width) + 1 : 1 - (sig_width + exp_width))] = in_float_data_array_rd_slot[i * ((sig_width + exp_width) + 1)+:(sig_width + exp_width) + 1];
			assign gamma_float[((sig_width + exp_width) >= 0 ? 0 : sig_width + exp_width) + (i * ((sig_width + exp_width) >= 0 ? (sig_width + exp_width) + 1 : 1 - (sig_width + exp_width)))+:((sig_width + exp_width) >= 0 ? (sig_width + exp_width) + 1 : 1 - (sig_width + exp_width))] = gamma_array_rd_slot[i * ((sig_width + exp_width) + 1)+:(sig_width + exp_width) + 1];
		end
		for (_gv_i_1 = 0; _gv_i_1 < BUS_NUM; _gv_i_1 = _gv_i_1 + 1) begin : x_mult_with_gamma_generate_array
			localparam i = _gv_i_1;
			fp_mult_pipe #(
				.sig_width(sig_width),
				.exp_width(exp_width),
				.ieee_compliance(ieee_compliance),
				.stages(3)
			) inst_fp_mult_x_gamma(
				.clk(clk),
				.rst_n(rst_n),
				.a(x_float[((sig_width + exp_width) >= 0 ? 0 : sig_width + exp_width) + (i * ((sig_width + exp_width) >= 0 ? (sig_width + exp_width) + 1 : 1 - (sig_width + exp_width)))+:((sig_width + exp_width) >= 0 ? (sig_width + exp_width) + 1 : 1 - (sig_width + exp_width))]),
				.b(gamma_float[((sig_width + exp_width) >= 0 ? 0 : sig_width + exp_width) + (i * ((sig_width + exp_width) >= 0 ? (sig_width + exp_width) + 1 : 1 - (sig_width + exp_width)))+:((sig_width + exp_width) >= 0 ? (sig_width + exp_width) + 1 : 1 - (sig_width + exp_width))]),
				.z(x_mult_with_gamma[((sig_width + exp_width) >= 0 ? 0 : sig_width + exp_width) + (i * ((sig_width + exp_width) >= 0 ? (sig_width + exp_width) + 1 : 1 - (sig_width + exp_width)))+:((sig_width + exp_width) >= 0 ? (sig_width + exp_width) + 1 : 1 - (sig_width + exp_width))]),
				.ab_valid(x_and_gamma_vld[i]),
				.z_valid(x_mult_with_gamma_vld[i])
			);
		end
		for (_gv_i_1 = 0; _gv_i_1 < BUS_NUM; _gv_i_1 = _gv_i_1 + 1) begin : float_RMSnorm_generate_array
			localparam i = _gv_i_1;
			fp_mult_pipe #(
				.sig_width(sig_width),
				.exp_width(exp_width),
				.ieee_compliance(ieee_compliance),
				.stages(3)
			) inst_fp_mult_float_RMSnorm(
				.clk(clk),
				.rst_n(rst_n),
				.a(x_mult_with_gamma[((sig_width + exp_width) >= 0 ? 0 : sig_width + exp_width) + (i * ((sig_width + exp_width) >= 0 ? (sig_width + exp_width) + 1 : 1 - (sig_width + exp_width)))+:((sig_width + exp_width) >= 0 ? (sig_width + exp_width) + 1 : 1 - (sig_width + exp_width))]),
				.b(one_over_rms),
				.z(float_RMSnorm[((sig_width + exp_width) >= 0 ? 0 : sig_width + exp_width) + (i * ((sig_width + exp_width) >= 0 ? (sig_width + exp_width) + 1 : 1 - (sig_width + exp_width)))+:((sig_width + exp_width) >= 0 ? (sig_width + exp_width) + 1 : 1 - (sig_width + exp_width))]),
				.ab_valid(x_mult_with_gamma_vld[i]),
				.z_valid(float_RMSnorm_vld[i])
			);
		end
		for (_gv_i_1 = 0; _gv_i_1 < BUS_NUM; _gv_i_1 = _gv_i_1 + 1) begin : fixed_RMSnorm_generate_array
			localparam i = _gv_i_1;
			always @(posedge clk or negedge rst_n)
				if (~rst_n) begin
					float_RMSnorm_flt2i[((sig_width + exp_width) >= 0 ? 0 : sig_width + exp_width) + (i * ((sig_width + exp_width) >= 0 ? (sig_width + exp_width) + 1 : 1 - (sig_width + exp_width)))+:((sig_width + exp_width) >= 0 ? (sig_width + exp_width) + 1 : 1 - (sig_width + exp_width))] <= 0;
					float_RMSnorm_flt2i_vld[i] <= 0;
				end
				else begin
					if (float_RMSnorm == 0)
						float_RMSnorm_flt2i[((sig_width + exp_width) >= 0 ? 0 : sig_width + exp_width) + (i * ((sig_width + exp_width) >= 0 ? (sig_width + exp_width) + 1 : 1 - (sig_width + exp_width)))+:((sig_width + exp_width) >= 0 ? (sig_width + exp_width) + 1 : 1 - (sig_width + exp_width))] <= 0;
					else begin
						float_RMSnorm_flt2i[((sig_width + exp_width) >= 0 ? (i * ((sig_width + exp_width) >= 0 ? (sig_width + exp_width) + 1 : 1 - (sig_width + exp_width))) + ((sig_width + exp_width) >= 0 ? (sig_width + exp_width) - 1 : (sig_width + exp_width) - ((sig_width + exp_width) - 1)) : (((i * ((sig_width + exp_width) >= 0 ? (sig_width + exp_width) + 1 : 1 - (sig_width + exp_width))) + ((sig_width + exp_width) >= 0 ? (sig_width + exp_width) - 1 : (sig_width + exp_width) - ((sig_width + exp_width) - 1))) + (sig_width + exp_width)) - 1)-:sig_width + exp_width] <= float_RMSnorm[((sig_width + exp_width) >= 0 ? (i * ((sig_width + exp_width) >= 0 ? (sig_width + exp_width) + 1 : 1 - (sig_width + exp_width))) + ((sig_width + exp_width) >= 0 ? (sig_width + exp_width) - 1 : (sig_width + exp_width) - ((sig_width + exp_width) - 1)) : (((i * ((sig_width + exp_width) >= 0 ? (sig_width + exp_width) + 1 : 1 - (sig_width + exp_width))) + ((sig_width + exp_width) >= 0 ? (sig_width + exp_width) - 1 : (sig_width + exp_width) - ((sig_width + exp_width) - 1))) + (sig_width + exp_width)) - 1)-:sig_width + exp_width] + $signed({out_scale_pos_reg, {sig_width {1'b0}}});
						float_RMSnorm_flt2i[(i * ((sig_width + exp_width) >= 0 ? (sig_width + exp_width) + 1 : 1 - (sig_width + exp_width))) + ((sig_width + exp_width) >= 0 ? sig_width + exp_width : (sig_width + exp_width) - (sig_width + exp_width))] <= float_RMSnorm[(i * ((sig_width + exp_width) >= 0 ? (sig_width + exp_width) + 1 : 1 - (sig_width + exp_width))) + ((sig_width + exp_width) >= 0 ? sig_width + exp_width : (sig_width + exp_width) - (sig_width + exp_width))];
					end
					float_RMSnorm_flt2i_vld[i] <= float_RMSnorm_vld[i];
				end
			DW_fp_flt2i #(
				sig_width,
				exp_width,
				isize,
				isign
			) i2flt_in_data(
				.a(float_RMSnorm_flt2i[((sig_width + exp_width) >= 0 ? 0 : sig_width + exp_width) + (i * ((sig_width + exp_width) >= 0 ? (sig_width + exp_width) + 1 : 1 - (sig_width + exp_width)))+:((sig_width + exp_width) >= 0 ? (sig_width + exp_width) + 1 : 1 - (sig_width + exp_width))]),
				.rnd(3'b000),
				.z(fixed_RMSnorm[i * 32+:32]),
				.status()
			);
			always @(*) begin
				if (_sv2v_0)
					;
				nxt_out_fixed_data[i * 8+:8] = fixed_RMSnorm[i * 32+:32];
				if ($signed(fixed_RMSnorm[i * 32+:32]) > 127)
					nxt_out_fixed_data[i * 8+:8] = 8'b01111111;
				else if ($signed(fixed_RMSnorm[i * 32+:32]) < -128)
					nxt_out_fixed_data[i * 8+:8] = 8'b10000000;
			end
			always @(posedge clk or negedge rst_n)
				if (~rst_n) begin
					out_fixed_data_vld[i] <= 0;
					out_fixed_data[i * 8+:8] <= 0;
				end
				else begin
					out_fixed_data_vld[i] <= float_RMSnorm_flt2i_vld[i];
					out_fixed_data[i * 8+:8] <= nxt_out_fixed_data[i * 8+:8];
				end
		end
	endgenerate
	always @(posedge clk or negedge rst_n)
		if (~rst_n) begin
			out_fixed_data_last_cnt <= 0;
			out_fixed_data_last <= 0;
		end
		else if (((out_fixed_data_last_cnt >= (data_num - BUS_NUM)) && (data_num != 0)) && |float_RMSnorm_flt2i_vld) begin
			out_fixed_data_last <= 1;
			out_fixed_data_last_cnt <= 0;
		end
		else if (|float_RMSnorm_flt2i_vld)
			out_fixed_data_last_cnt <= out_fixed_data_last_cnt + BUS_NUM;
		else
			out_fixed_data_last <= 0;
	initial _sv2v_0 = 0;
endmodule

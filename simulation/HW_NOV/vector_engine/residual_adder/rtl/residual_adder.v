module residual_adder (
	clk,
	rst_n,
	scale_vld,
	scale_a,
	scale_b,
	in_finish,
	shift_vld,
	shift,
	in_data_a,
	in_data_b,
	in_data_vld,
	in_addr,
	out_data,
	out_data_vld,
	out_addr,
	out_finish
);
	input wire clk;
	input wire rst_n;
	input wire scale_vld;
	input wire [9:0] scale_a;
	input wire [9:0] scale_b;
	input wire in_finish;
	input wire shift_vld;
	input wire [4:0] shift;
	input wire [127:0] in_data_a;
	input wire [127:0] in_data_b;
	input wire in_data_vld;
	input wire [8:0] in_addr;
	output wire [127:0] out_data;
	output wire out_data_vld;
	output wire [8:0] out_addr;
	output wire out_finish;
	wire [15:0] out_data_vld_array;
	assign out_data_vld = |out_data_vld_array;
	localparam addr_delay = 13;
	genvar _gv_i_1;
	generate
		for (_gv_i_1 = 0; _gv_i_1 < addr_delay; _gv_i_1 = _gv_i_1 + 1) begin : adddr_delay_gen_array
			localparam i = _gv_i_1;
			reg [8:0] temp_addr;
			reg temp_finish;
			if (i == 0) begin : genblk1
				always @(posedge clk or negedge rst_n)
					if (~rst_n) begin
						temp_addr <= 0;
						temp_finish <= 0;
					end
					else begin
						temp_addr <= in_addr;
						temp_finish <= in_finish;
					end
			end
			else begin : genblk1
				always @(posedge clk or negedge rst_n)
					if (~rst_n) begin
						temp_addr <= 0;
						temp_finish <= 0;
					end
					else begin
						temp_addr <= adddr_delay_gen_array[i - 1].temp_addr;
						temp_finish <= adddr_delay_gen_array[i - 1].temp_finish;
					end
			end
		end
	endgenerate
	assign out_addr = adddr_delay_gen_array[12].temp_addr;
	assign out_finish = adddr_delay_gen_array[12].temp_finish;
	generate
		for (_gv_i_1 = 0; _gv_i_1 < 16; _gv_i_1 = _gv_i_1 + 1) begin : single_residual_adder_gen_array
			localparam i = _gv_i_1;
			residual_adder_single inst_single_adder(
				.clk(clk),
				.rst_n(rst_n),
				.scale_vld(scale_vld),
				.scale_a(scale_a),
				.scale_b(scale_b),
				.shift_vld(shift_vld),
				.shift(shift),
				.in_data_a(in_data_a[i * 8+:8]),
				.in_data_b(in_data_b[i * 8+:8]),
				.in_data_vld(in_data_vld),
				.out_data(out_data[i * 8+:8]),
				.out_data_vld(out_data_vld_array[i])
			);
		end
	endgenerate
endmodule
module residual_adder_single (
	clk,
	rst_n,
	scale_vld,
	scale_a,
	scale_b,
	shift_vld,
	shift,
	in_data_a,
	in_data_b,
	in_data_vld,
	out_data,
	out_data_vld
);
	reg _sv2v_0;
	input wire clk;
	input wire rst_n;
	input wire scale_vld;
	input wire [9:0] scale_a;
	input wire [9:0] scale_b;
	input wire shift_vld;
	input wire [4:0] shift;
	input wire signed [7:0] in_data_a;
	input wire signed [7:0] in_data_b;
	input wire in_data_vld;
	output reg signed [7:0] out_data;
	output reg out_data_vld;
	reg [9:0] scale_a_reg;
	reg [9:0] scale_b_reg;
	reg [4:0] shift_reg;
	always @(posedge clk or negedge rst_n)
		if (~rst_n) begin
			scale_a_reg <= 0;
			scale_b_reg <= 0;
		end
		else if (scale_vld) begin
			scale_a_reg <= scale_a;
			scale_b_reg <= scale_b;
		end
	always @(posedge clk or negedge rst_n)
		if (~rst_n)
			shift_reg <= 0;
		else if (shift_vld)
			shift_reg <= shift;
	reg signed [7:0] in_data_a_delay1;
	reg signed [7:0] in_data_b_delay1;
	reg in_data_vld_delay1;
	reg signed [17:0] dequant_a;
	reg signed [17:0] dequant_b;
	reg dequant_vld;
	wire signed [17:0] dequant_a_delay;
	wire signed [17:0] dequant_b_delay;
	wire dequant_vld_delay;
	always @(posedge clk or negedge rst_n)
		if (~rst_n) begin
			in_data_a_delay1 <= 0;
			in_data_b_delay1 <= 0;
			in_data_vld_delay1 <= 0;
		end
		else if (in_data_vld) begin
			in_data_a_delay1 <= in_data_a;
			in_data_b_delay1 <= in_data_b;
			in_data_vld_delay1 <= 1;
		end
		else
			in_data_vld_delay1 <= 0;
	always @(posedge clk or negedge rst_n)
		if (~rst_n) begin
			dequant_a <= 0;
			dequant_b <= 0;
			dequant_vld <= 0;
		end
		else if (in_data_vld_delay1) begin
			dequant_a <= $signed(in_data_a_delay1) * $signed({1'b0, scale_a_reg});
			dequant_b <= $signed(in_data_b_delay1) * $signed({1'b0, scale_b_reg});
			dequant_vld <= 1;
		end
		else
			dequant_vld <= 0;
	genvar _gv_i_2;
	generate
		for (_gv_i_2 = 0; _gv_i_2 < 7; _gv_i_2 = _gv_i_2 + 1) begin : deqaunt_rt_array
			localparam i = _gv_i_2;
			reg [17:0] dequant_a_temp;
			reg [17:0] dequant_b_temp;
			reg dequant_vld_temp;
			if (i == 0) begin : genblk1
				always @(posedge clk or negedge rst_n)
					if (~rst_n) begin
						dequant_vld_temp <= 0;
						dequant_a_temp <= 0;
						dequant_b_temp <= 0;
					end
					else begin
						dequant_vld_temp <= dequant_vld;
						dequant_a_temp <= dequant_a;
						dequant_b_temp <= dequant_b;
					end
			end
			else begin : genblk1
				always @(posedge clk or negedge rst_n)
					if (~rst_n) begin
						dequant_vld_temp <= 0;
						dequant_a_temp <= 0;
						dequant_b_temp <= 0;
					end
					else begin
						dequant_vld_temp <= deqaunt_rt_array[i - 1].dequant_vld_temp;
						dequant_a_temp <= deqaunt_rt_array[i - 1].dequant_a_temp;
						dequant_b_temp <= deqaunt_rt_array[i - 1].dequant_b_temp;
					end
			end
		end
	endgenerate
	assign dequant_a_delay = deqaunt_rt_array[6].dequant_a_temp;
	assign dequant_b_delay = deqaunt_rt_array[6].dequant_b_temp;
	assign dequant_vld_delay = deqaunt_rt_array[6].dequant_vld_temp;
	reg signed [18:0] sum_ab;
	reg sum_ab_vld;
	always @(posedge clk or negedge rst_n)
		if (~rst_n) begin
			sum_ab <= 0;
			sum_ab_vld <= 0;
		end
		else if (dequant_vld_delay) begin
			sum_ab_vld <= 1;
			sum_ab <= dequant_a_delay + dequant_b_delay;
		end
		else
			sum_ab_vld <= 0;
	wire round;
	assign round = (shift_reg > 0 ? sum_ab[shift_reg - 1] : 0);
	reg signed [18:0] shift_sum_ab;
	reg shift_sum_ab_vld;
	reg round_delay1;
	always @(posedge clk or negedge rst_n)
		if (~rst_n) begin
			shift_sum_ab_vld <= 0;
			shift_sum_ab <= 0;
			round_delay1 <= 0;
		end
		else if (sum_ab_vld) begin
			shift_sum_ab <= sum_ab >>> shift_reg;
			shift_sum_ab_vld <= 1;
			round_delay1 <= round;
		end
		else
			shift_sum_ab_vld <= 0;
	reg signed [18:0] round_sum_ab;
	reg round_sum_ab_vld;
	always @(posedge clk or negedge rst_n)
		if (~rst_n) begin
			round_sum_ab_vld <= 0;
			round_sum_ab <= 0;
		end
		else if (shift_sum_ab_vld) begin
			round_sum_ab <= shift_sum_ab + round_delay1;
			round_sum_ab_vld <= 1;
		end
		else
			round_sum_ab_vld <= 0;
	reg signed [7:0] nxt_out_data;
	reg nxt_out_data_vld;
	always @(*) begin
		if (_sv2v_0)
			;
		nxt_out_data_vld = 0;
		nxt_out_data = out_data;
		if (round_sum_ab_vld) begin
			nxt_out_data_vld = 1;
			if (round_sum_ab > 127)
				nxt_out_data = 127;
			else if (round_sum_ab < -128)
				nxt_out_data = -128;
			else
				nxt_out_data = round_sum_ab;
		end
	end
	always @(posedge clk or negedge rst_n)
		if (~rst_n) begin
			out_data_vld <= 0;
			out_data <= 0;
		end
		else begin
			out_data_vld <= nxt_out_data_vld;
			out_data <= nxt_out_data;
		end
	initial _sv2v_0 = 0;
endmodule

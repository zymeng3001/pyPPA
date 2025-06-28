module fp_invsqrt_pipe (
	clk,
	rst_n,
	x,
	x_vld,
	y,
	y_vld
);
	input wire clk;
	input wire rst_n;
	input wire [15:0] x;
	input wire x_vld;
	output reg [15:0] y;
	output reg y_vld;
	reg [15:0] startpoint;
	reg [15:0] half_x;
	reg startpoint_half_x_vld;
	wire [15:0] threehalfs;
	assign threehalfs = 16'b0011111111000000;
	always @(posedge clk or negedge rst_n)
		if (~rst_n) begin
			startpoint <= 0;
			startpoint_half_x_vld <= 0;
			half_x <= 0;
		end
		else if (x_vld) begin
			startpoint <= (16'hbe6f - x) >> 1;
			startpoint_half_x_vld <= x_vld;
			half_x <= x - 8'b10000000;
		end
		else
			startpoint_half_x_vld <= 0;
	reg [15:0] startpoint_delay_array [10:0];
	reg [15:0] half_x_delay_array [10:0];
	genvar _gv_i_1;
	generate
		for (_gv_i_1 = 0; _gv_i_1 < 11; _gv_i_1 = _gv_i_1 + 1) begin : genblk1
			localparam i = _gv_i_1;
			if (i == 0) begin : genblk1
				always @(posedge clk or negedge rst_n)
					if (~rst_n) begin
						startpoint_delay_array[i] <= 0;
						half_x_delay_array[i] <= 0;
					end
					else begin
						startpoint_delay_array[i] <= startpoint;
						half_x_delay_array[i] <= half_x;
					end
			end
			else begin : genblk1
				always @(posedge clk or negedge rst_n)
					if (~rst_n) begin
						startpoint_delay_array[i] <= 0;
						half_x_delay_array[i] <= 0;
					end
					else begin
						startpoint_delay_array[i] <= startpoint_delay_array[i - 1];
						half_x_delay_array[i] <= half_x_delay_array[i - 1];
					end
			end
		end
	endgenerate
	reg [15:0] startpoint_square;
	wire [15:0] nxt_startpoint_square;
	reg startpoint_square_vld;
	wire nxt_startpoint_square_vld;
	always @(posedge clk or negedge rst_n)
		if (~rst_n) begin
			startpoint_square <= 0;
			startpoint_square_vld <= 0;
		end
		else begin
			startpoint_square <= nxt_startpoint_square;
			startpoint_square_vld <= nxt_startpoint_square_vld;
		end
	fp_mult_pipe #(
		.sig_width(7),
		.exp_width(8),
		.ieee_compliance(0)
	) startpoint_square_mult_inst(
		.clk(clk),
		.rst_n(rst_n),
		.a(startpoint),
		.b(startpoint),
		.z(nxt_startpoint_square),
		.ab_valid(startpoint_half_x_vld),
		.z_valid(nxt_startpoint_square_vld)
	);
	reg [15:0] half_x_mult_startpoint_square;
	wire [15:0] nxt_half_x_mult_startpoint_square;
	reg half_x_mult_startpoint_square_vld;
	wire nxt_half_x_mult_startpoint_square_vld;
	always @(posedge clk or negedge rst_n)
		if (~rst_n) begin
			half_x_mult_startpoint_square <= 0;
			half_x_mult_startpoint_square_vld <= 0;
		end
		else begin
			half_x_mult_startpoint_square <= nxt_half_x_mult_startpoint_square;
			half_x_mult_startpoint_square_vld <= nxt_half_x_mult_startpoint_square_vld;
		end
	fp_mult_pipe #(
		.sig_width(7),
		.exp_width(8),
		.ieee_compliance(0)
	) half_x_mult_inst(
		.clk(clk),
		.rst_n(rst_n),
		.a(startpoint_square),
		.b(half_x_delay_array[4]),
		.z(nxt_half_x_mult_startpoint_square),
		.ab_valid(startpoint_square_vld),
		.z_valid(nxt_half_x_mult_startpoint_square_vld)
	);
	reg [15:0] threehalfs_sub;
	wire [15:0] nxt_threehlafs_sub;
	reg threehalfs_sub_vld;
	wire nxt_threehalfs_sub_vld;
	assign nxt_threehalfs_sub_vld = half_x_mult_startpoint_square_vld;
	always @(posedge clk or negedge rst_n)
		if (~rst_n) begin
			threehalfs_sub <= 0;
			threehalfs_sub_vld <= 0;
		end
		else begin
			threehalfs_sub <= nxt_threehlafs_sub;
			threehalfs_sub_vld <= nxt_threehalfs_sub_vld;
		end
	DW_fp_sub #(
		.sig_width(7),
		.exp_width(8),
		.ieee_compliance(0)
	) threehalfs_sub_inst(
		.a(threehalfs),
		.b(half_x_mult_startpoint_square),
		.rnd(3'b000),
		.z(nxt_threehlafs_sub),
		.status()
	);
	wire [15:0] nxt_y;
	wire nxt_y_vld;
	always @(posedge clk or negedge rst_n)
		if (~rst_n) begin
			y_vld <= 0;
			y <= 0;
		end
		else begin
			y <= nxt_y;
			y_vld <= nxt_y_vld;
		end
	fp_mult_pipe #(
		.sig_width(7),
		.exp_width(8),
		.ieee_compliance(0)
	) y_mult_inst(
		.clk(clk),
		.rst_n(rst_n),
		.a(threehalfs_sub),
		.b(startpoint_delay_array[10]),
		.z(nxt_y),
		.ab_valid(threehalfs_sub_vld),
		.z_valid(nxt_y_vld)
	);
endmodule

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
	// DW_fp_sub #(
	// 	.sig_width(7),
	// 	.exp_width(8),
	// 	.ieee_compliance(0)
	// ) threehalfs_sub_inst(
	// 	.a(threehalfs),
	// 	.b(half_x_mult_startpoint_square),
	// 	.rnd(3'b000),
	// 	.z(nxt_threehlafs_sub),
	// 	.status()
	// );
	custom_fp_sub #(
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

module custom_fp_sub #(
    parameter integer sig_width = 7,   // number of fraction bits
    parameter integer exp_width = 8,   // number of exponent bits
    parameter integer ieee_compliance = 0  // only basic handling when 0
)(
    input  wire [sig_width+exp_width:0] a,  // input a
    input  wire [sig_width+exp_width:0] b,  // input b
    input  wire [2:0]                   rnd, // rounding mode (ignored)
    output reg  [sig_width+exp_width:0] z,   // result = a - b
    output reg  [7:0]                   status // status flags
);

    // Sign, exponent, mantissa extraction
    wire sign_a = a[sig_width + exp_width];
    wire sign_b = b[sig_width + exp_width];
    wire [exp_width-1:0] exp_a = a[sig_width + exp_width - 1 : sig_width];
    wire [exp_width-1:0] exp_b = b[sig_width + exp_width - 1 : sig_width];
    wire [sig_width-1:0] frac_a = a[sig_width-1:0];
    wire [sig_width-1:0] frac_b = b[sig_width-1:0];

    // 1. Add implicit leading 1 for normalized values
    wire [sig_width:0] mant_a = (|exp_a) ? {1'b1, frac_a} : {1'b0, frac_a};
    wire [sig_width:0] mant_b = (|exp_b) ? {1'b1, frac_b} : {1'b0, frac_b};

    // 2. Align mantissas
    wire [exp_width:0] exp_diff = (exp_a > exp_b) ? (exp_a - exp_b) : (exp_b - exp_a);
    wire [sig_width+2:0] mant_a_align = (exp_a >= exp_b) ? {mant_a, 2'b00} : ({mant_a, 2'b00} >> exp_diff);
    wire [sig_width+2:0] mant_b_align = (exp_b > exp_a) ? {mant_b, 2'b00} : ({mant_b, 2'b00} >> exp_diff);

    // 3. Determine result sign and subtract mantissas
    reg [sig_width+2:0] mant_sub;
    reg [exp_width-1:0] exp_result;
    reg sign_result;

    always @(*) begin
        if ({exp_a, mant_a} >= {exp_b, mant_b}) begin
            mant_sub = mant_a_align - mant_b_align;
            exp_result = (exp_a >= exp_b) ? exp_a : exp_b;
            sign_result = sign_a;
        end else begin
            mant_sub = mant_b_align - mant_a_align;
            exp_result = (exp_b >= exp_a) ? exp_b : exp_a;
            sign_result = ~sign_b;  // a - b = -(b - a)
        end
    end

    // 4. Normalize result
    wire [sig_width-1:0] frac_norm;
    wire [2*sig_width-1:0] frac_norms;
    wire [exp_width-1:0] exp_norm;

    wire [$clog2(sig_width+3)-1:0] shift;
    wire                          valid;
    priority_encoder #(
        .WIDTH(sig_width + 3)
    ) pe_inst (
        .in(mant_sub),
        .shift(shift),
        .valid(valid)
    );

    // then use shift as before
    assign frac_norms = mant_sub << shift;
    assign frac_norm  = frac_norms[sig_width+2:3];
    assign exp_norm   = (exp_result > shift) ? (exp_result - shift) : 0;


    // 5. Final packing
    always @(*) begin
        z = {sign_result, exp_norm, frac_norm};
        status = 8'b0;
        if (mant_sub == 0)
            z = {1'b0, {exp_width{1'b0}}, {sig_width{1'b0}}};  // exact zero
    end

endmodule

module priority_encoder #(
    parameter WIDTH = 16,
    parameter SHIFT_WIDTH = $clog2(WIDTH)
)(
    input  wire [WIDTH-1:0] in,
    output reg  [SHIFT_WIDTH-1:0] shift,
    output wire valid
);

    integer i;
    always @(*) begin
        shift = 0;
        for (i = WIDTH-1; i >= 0; i = i - 1) begin
            if (in[i]) begin
                shift = WIDTH - 1 - i;
            end
        end
    end

    assign valid = |in;
endmodule


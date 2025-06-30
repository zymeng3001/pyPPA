module fp_mult_pipe (
	clk,
	rst_n,
	a,
	b,
	z,
	ab_valid,
	z_valid
);
	reg _sv2v_0;
	parameter sig_width = 23;
	parameter exp_width = 8;
	parameter ieee_compliance = 0;
	parameter stages = 5;
	input wire clk;
	input wire rst_n;
	input wire [sig_width + exp_width:0] a;
	input wire [sig_width + exp_width:0] b;
	output reg [sig_width + exp_width:0] z;
	input wire ab_valid;
	output reg z_valid;
	parameter id_width = 1;
	parameter op_iso_mode = 0;
	parameter no_pm = 1;
	parameter rst_mode = 0;
	parameter en_ubr_flag = 0;
	wire [sig_width + exp_width:0] z_w;
	// DW_fp_mult #(
	// 	sig_width,
	// 	exp_width,
	// 	ieee_compliance,
	// 	en_ubr_flag
	// ) U1(
	// 	.a(a),
	// 	.b(b),
	// 	.rnd(3'b000),
	// 	.z(z_w),
	// 	.status()
	// );

	custom_fp_mult #(
		sig_width,
		exp_width,
		ieee_compliance,
		en_ubr_flag
	) U1(
		.a(a),
		.b(b),
		.rnd(3'b000),
		.z(z_w),
		.status()
	);
	genvar _gv_i_1;
	generate
		for (_gv_i_1 = 0; _gv_i_1 < stages; _gv_i_1 = _gv_i_1 + 1) begin : fp_mult_retiming_stage
			localparam i = _gv_i_1;
			reg [sig_width + exp_width:0] timing_register_float;
			reg timing_register_vld;
			if (i == 0) begin : genblk1
				always @(posedge clk or negedge rst_n)
					if (~rst_n) begin
						timing_register_float <= 0;
						timing_register_vld <= 0;
					end
					else begin
						timing_register_float <= z_w;
						timing_register_vld <= ab_valid;
					end
			end
			else begin : genblk1
				always @(posedge clk or negedge rst_n)
					if (~rst_n) begin
						timing_register_float <= 0;
						timing_register_vld <= 0;
					end
					else begin
						timing_register_float <= fp_mult_retiming_stage[i - 1].timing_register_float;
						timing_register_vld <= fp_mult_retiming_stage[i - 1].timing_register_vld;
					end
			end
		end
	endgenerate
	always @(*) begin
		if (_sv2v_0)
			;
		z = fp_mult_retiming_stage[stages - 1].timing_register_float;
		z_valid = fp_mult_retiming_stage[stages - 1].timing_register_vld;
	end
	initial _sv2v_0 = 0;
endmodule

module custom_fp_mult #(
    parameter integer sig_width       = 23,  // number of significand bits (mantissa)
    parameter integer exp_width       = 8,   // number of exponent bits
    parameter integer ieee_compliance = 0,   // 0 = ignore NaN/Inf
    parameter integer en_ubr_flag     = 0    // unused in this version
)(
    input  wire [sig_width + exp_width:0] a, // input A
    input  wire [sig_width + exp_width:0] b, // input B
    input  wire [2:0] rnd,                   // rounding mode (ignored)
    output reg  [sig_width + exp_width:0] z, // output result
    output reg  [7:0] status                 // status flags
);

    // Unpack inputs
    wire sign_a = a[sig_width + exp_width];
    wire sign_b = b[sig_width + exp_width];
    wire [exp_width-1:0] exp_a = a[sig_width + exp_width - 1 : sig_width];
    wire [exp_width-1:0] exp_b = b[sig_width + exp_width - 1 : sig_width];
    wire [sig_width-1:0] frac_a = a[sig_width-1:0];
    wire [sig_width-1:0] frac_b = b[sig_width-1:0];

    // Compute result sign
    wire sign_z = sign_a ^ sign_b;

    // Add implicit leading 1 for normalized numbers
    wire [sig_width:0] mant_a = (|exp_a) ? {1'b1, frac_a} : {1'b0, frac_a};
    wire [sig_width:0] mant_b = (|exp_b) ? {1'b1, frac_b} : {1'b0, frac_b};

    // Multiply mantissas (full precision)
    wire [2*sig_width+1:0] mant_prod = mant_a * mant_b;

    // Add exponents and subtract bias
    wire signed [exp_width:0] exp_sum = $signed({1'b0, exp_a}) + $signed({1'b0, exp_b}) - ((1 << (exp_width-1)) - 1);

    // Normalize mantissa (check if MSB is at [2*sig_width+1] or [2*sig_width])
    reg [sig_width-1:0] frac_z;
    reg [exp_width-1:0] exp_z;
    always @(*) begin
        if (mant_prod[2*sig_width+1]) begin
            frac_z = mant_prod[2*sig_width:2*sig_width - sig_width + 1];  // drop MSB
            exp_z = exp_sum + 1;
        end else begin
            frac_z = mant_prod[2*sig_width - 1:2*sig_width - sig_width];  // no shift
            exp_z = exp_sum;
        end
    end

    // Compose final result
    always @(*) begin
        // special cases
        if ((a == 0) || (b == 0)) begin
            z = {1'b0, {exp_width{1'b0}}, {sig_width{1'b0}}};  // zero
            status = 8'h01;
        end else begin
            z = {sign_z, exp_z, frac_z};
            status = 8'h00;
        end
    end

endmodule

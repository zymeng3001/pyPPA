module fp_div_pipe (
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
	parameter faithful_round = 0;
	parameter op_iso_mode = 0;
	parameter no_pm = 1;
	parameter rst_mode = 0;
	parameter en_ubr_flag = 0;
	wire [sig_width + exp_width:0] z_w;
	// DW_fp_div #(
	// 	sig_width,
	// 	exp_width,
	// 	ieee_compliance,
	// 	faithful_round,
	// 	en_ubr_flag
	// ) U1(
	// 	.a(a),
	// 	.b(b),
	// 	.rnd(3'b000),
	// 	.z(z_w),
	// 	.status()
	// );
	custom_fp_div #(
		sig_width,
		exp_width,
		ieee_compliance,
		faithful_round,
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
		for (_gv_i_1 = 0; _gv_i_1 < stages; _gv_i_1 = _gv_i_1 + 1) begin : fp_div_retiming_stage
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
						timing_register_float <= fp_div_retiming_stage[i - 1].timing_register_float;
						timing_register_vld <= fp_div_retiming_stage[i - 1].timing_register_vld;
					end
			end
		end
	endgenerate
	always @(*) begin
		if (_sv2v_0)
			;
		z = fp_div_retiming_stage[stages - 1].timing_register_float;
		z_valid = fp_div_retiming_stage[stages - 1].timing_register_vld;
	end
	initial _sv2v_0 = 0;
endmodule

module custom_fp_div #(
    parameter integer sig_width       = 23,  // significand (mantissa) bits (23 for FP32)
    parameter integer exp_width       = 8,   // exponent bits (8 for FP32)
    parameter integer ieee_compliance = 0,   // 1 = handle NaN, Inf, denormals
    parameter integer faithful_round  = 0,   // ignored here (for simplicity)
    parameter integer en_ubr_flag     = 0    // unused: underflow-by-rounding
)(
    input  wire [sig_width+exp_width:0] a,  // floating-point input A
    input  wire [sig_width+exp_width:0] b,  // floating-point input B
    input  wire [2:0]                   rnd, // rounding mode (ignored here)
    output reg  [sig_width+exp_width:0] z,  // result
    output reg  [7:0]                   status // status flags (simplified)
);

    // Decompose inputs
    wire sign_a = a[sig_width+exp_width];
    wire sign_b = b[sig_width+exp_width];
    wire [exp_width-1:0] exp_a = a[sig_width+exp_width-1:sig_width];
    wire [exp_width-1:0] exp_b = b[sig_width+exp_width-1:sig_width];
    wire [sig_width-1:0] frac_a = a[sig_width-1:0];
    wire [sig_width-1:0] frac_b = b[sig_width-1:0];

    // Sign of result
    wire sign_z = sign_a ^ sign_b;

    // Add hidden 1 to normalized mantissa
    wire [sig_width:0] norm_frac_a = (|exp_a) ? {1'b1, frac_a} : {1'b0, frac_a};
    wire [sig_width:0] norm_frac_b = (|exp_b) ? {1'b1, frac_b} : {1'b0, frac_b};

    // Result exponent (biased)
    wire signed [exp_width:0] exp_z = $signed({1'b0, exp_a}) - $signed({1'b0, exp_b}) + (1 << (exp_width-1)) - 1;

    // Mantissa division (simplified version)
    reg [2*sig_width+1:0] frac_div;
    always @(*) begin
        if (norm_frac_b != 0)
            frac_div = (norm_frac_a << sig_width) / norm_frac_b;
        else
            frac_div = 0;
    end

    // Normalize result (assume top sig_width+1 bits are the valid mantissa)
    wire [sig_width-1:0] frac_z = frac_div[sig_width*2: sig_width+1]; // no rounding here

    // Final packing
    always @(*) begin
        if (b == 0) begin
            z = {sign_z, {(exp_width){1'b1}}, {sig_width{1'b0}}}; // Inf
            status = 8'h10; // divide-by-zero flag
        end else if (a == 0) begin
            z = {sign_z, {exp_width{1'b0}}, {sig_width{1'b0}}}; // Zero
            status = 8'h01; // zero result
        end else begin
            z = {sign_z, exp_z[exp_width-1:0], frac_z};
            status = 8'h00;
        end
    end

endmodule


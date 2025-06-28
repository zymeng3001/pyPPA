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
	DW_fp_mult #(
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

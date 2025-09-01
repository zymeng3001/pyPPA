module CLK_GEN (
	asyn_rst,
	ext_clk,
	IIN,
	clk_div_sel,
	clk_gate_en,
	o_chip_clk,
	o_clk_div
);
	input asyn_rst;
	input ext_clk;
	input IIN;
	input [1:0] clk_div_sel;
	input clk_gate_en;
	output wire o_chip_clk;
	output wire o_clk_div;
	wire CLK_CORE_RAW;
	wire CORECLK_div_1;
	wire CORECLK_div_2;
	wire CORECLK_div_4;
	wire core_clk_div;
	ringosc_xcp_top_v3 i_clk(
		.IIN(IIN),
		.CKON(CLK_CORE_RAW),
		.CKOP(),
		.EXTCKIN(1'b0),
		.EXTCKIP(1'b0),
		.CKSEL(1'b1),
		.CKFB_DRV(o_clk_div)
	);
	CLK_DIV u_CLK_DIV(
		.CLK_CORE_RAW(CLK_CORE_RAW),
		.asyn_rst(asyn_rst),
		.CORECLK_div_1(CORECLK_div_1),
		.CORECLK_div_2(CORECLK_div_2),
		.CORECLK_div_4(CORECLK_div_4)
	);
	CLK_DIV_MUX u_CLK_DIV_MUX(
		.clk_gen_div(clk_div_sel),
		.CORECLK_div_1(CORECLK_div_1),
		.CORECLK_div_2(CORECLK_div_2),
		.CORECLK_div_4(CORECLK_div_4),
		.EXT_CLK(ext_clk),
		.core_clk_div(core_clk_div)
	);
	b15cilb01hn1n32x5 u_CLK_GATE(
		.en(clk_gate_en),
		.te(clk_gate_en),
		.clk(core_clk_div),
		.clkout(o_chip_clk)
	);
endmodule
module CLK_DIV (
	CLK_CORE_RAW,
	asyn_rst,
	CORECLK_div_1,
	CORECLK_div_2,
	CORECLK_div_4
);
	input CLK_CORE_RAW;
	input asyn_rst;
	output wire CORECLK_div_1;
	output reg CORECLK_div_2;
	output reg CORECLK_div_4;
	assign CORECLK_div_1 = CLK_CORE_RAW;
	always @(posedge CORECLK_div_1 or posedge asyn_rst)
		if (asyn_rst)
			CORECLK_div_2 <= #(0) 1'b0;
		else
			CORECLK_div_2 <= #(0) ~CORECLK_div_2;
	always @(posedge CORECLK_div_2 or posedge asyn_rst)
		if (asyn_rst)
			CORECLK_div_4 <= #(0) 1'b0;
		else
			CORECLK_div_4 <= #(0) ~CORECLK_div_4;
endmodule
module CLK_DIV_MUX (
	clk_gen_div,
	CORECLK_div_1,
	CORECLK_div_2,
	CORECLK_div_4,
	EXT_CLK,
	core_clk_div
);
	input [1:0] clk_gen_div;
	input CORECLK_div_1;
	input CORECLK_div_2;
	input CORECLK_div_4;
	input EXT_CLK;
	output reg core_clk_div;
	always @(*)
		case (clk_gen_div)
			2'd0: core_clk_div = CORECLK_div_1;
			2'd1: core_clk_div = CORECLK_div_2;
			2'd2: core_clk_div = CORECLK_div_4;
			2'd3: core_clk_div = EXT_CLK;
			default: core_clk_div = CORECLK_div_1;
		endcase
endmodule

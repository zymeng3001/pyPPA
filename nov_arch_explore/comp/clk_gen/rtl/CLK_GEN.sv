module CLK_GEN (
	input           asyn_rst,
    input           ext_clk, //External Clock
	input           IIN, //Ring Oscillator Current Bias
	input	[2-1:0]	clk_div_sel, // Clock GEN Div
    input           clk_gate_en, // Clock Gate
	
	output	logic	o_chip_clk,	//Main clock of chip
    output  logic   o_clk_div //Divided Clock Out
);

	// Signal Declarations
	logic	CLK_CORE_RAW;
	
	logic	CORECLK_div_1;
	logic	CORECLK_div_2;
	logic	CORECLK_div_4;

	logic	core_clk_div;

	// Instantiation of CLK_GEN_CORE
    ringosc_xcp_top_v3 i_clk (.IIN(IIN),.CKON(CLK_CORE_RAW),.CKOP(),.EXTCKIN(1'b0),.EXTCKIP(1'b0),.CKSEL(1'b1),.CKFB_DRV(o_clk_div));

	// Clock Divider Logic
	CLK_DIV u_CLK_DIV (
		.CLK_CORE_RAW(CLK_CORE_RAW),
		.asyn_rst(asyn_rst),
		.CORECLK_div_1(CORECLK_div_1),
		.CORECLK_div_2(CORECLK_div_2),
		.CORECLK_div_4(CORECLK_div_4)
	);

	CLK_DIV_MUX u_CLK_DIV_MUX (
		.clk_gen_div(clk_div_sel),
		.CORECLK_div_1(CORECLK_div_1),
		.CORECLK_div_2(CORECLK_div_2),
		.CORECLK_div_4(CORECLK_div_4),
		.EXT_CLK(ext_clk),
		.core_clk_div(core_clk_div)
	);

    //integrated clock gating cell
    b15cilb01hn1n32x5 u_CLK_GATE (.en(clk_gate_en),.te(clk_gate_en),.clk(core_clk_div),.clkout(o_chip_clk));

endmodule

module CLK_DIV (
    input CLK_CORE_RAW,
    input asyn_rst,
	output logic	CORECLK_div_1,
	output logic	CORECLK_div_2,
	output logic	CORECLK_div_4
);

    // Clock Divider Logic
	assign CORECLK_div_1 = CLK_CORE_RAW;

	always @(posedge CORECLK_div_1 or posedge asyn_rst) begin
    	if(asyn_rst) CORECLK_div_2 <= `SD 1'b0;
    	else CORECLK_div_2 <= `SD ~CORECLK_div_2;
	end
	always @(posedge CORECLK_div_2 or posedge asyn_rst) begin
		if(asyn_rst) CORECLK_div_4 <= `SD 1'b0;
		else CORECLK_div_4 <= `SD ~CORECLK_div_4;
	end
endmodule

module CLK_DIV_MUX (
	input	[2-1:0]	clk_gen_div, // External Clock GEN Div
	input	CORECLK_div_1,
	input	CORECLK_div_2,
	input	CORECLK_div_4,
    input   EXT_CLK,

	output logic	core_clk_div
);

    always@* begin
		case (clk_gen_div)
    		2'd0 : core_clk_div = CORECLK_div_1;
    		2'd1 : core_clk_div = CORECLK_div_2;
    		2'd2 : core_clk_div = CORECLK_div_4;
    		2'd3 : core_clk_div = EXT_CLK;

			default: core_clk_div = CORECLK_div_1;
  		endcase 
	end

endmodule

module ringosc_xcp_top_v3 (
	IIN,
	CKON,
	CKOP,
	EXTCKIN,
	EXTCKIP,
	CKSEL,
	CKFB_DRV
);
	input IIN;
	output wire CKON;
	output wire CKOP;
	input EXTCKIN;
	input EXTCKIP;
	input CKSEL;
	output reg CKFB_DRV;
	reg CLK_RO_RAW;
	initial begin
		CLK_RO_RAW = 1'b0;
		CKFB_DRV = 1'b0;
	end
	always #(1000) CLK_RO_RAW = ~CLK_RO_RAW;
	always #(100000) CKFB_DRV = ~CKFB_DRV;
	assign CKON = CKSEL && CLK_RO_RAW;
endmodule

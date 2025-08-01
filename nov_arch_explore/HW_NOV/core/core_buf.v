module core_buf (
	clk,
	rstn,
	hlink_wdata,
	hlink_wen,
	hlink_rdata,
	hlink_rvalid
);
	parameter CACHE_DATA_WIDTH = 128;
	input clk;
	input rstn;
	input [CACHE_DATA_WIDTH - 1:0] hlink_wdata;
	input hlink_wen;
	output wire [CACHE_DATA_WIDTH - 1:0] hlink_rdata;
	output wire hlink_rvalid;
	reg [CACHE_DATA_WIDTH - 1:0] hlink_reg;
	reg hlink_reg_valid;
	always @(posedge clk or negedge rstn)
		if (!rstn)
			hlink_reg <= 'd0;
		else if (hlink_wen)
			hlink_reg <= hlink_wdata;
	always @(posedge clk or negedge rstn)
		if (!rstn)
			hlink_reg_valid <= 1'b0;
		else
			hlink_reg_valid <= hlink_wen;
	assign hlink_rdata = hlink_reg;
	assign hlink_rvalid = hlink_reg_valid;
endmodule

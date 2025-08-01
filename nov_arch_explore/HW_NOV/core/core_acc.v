module core_acc (
	clk,
	rstn,
	cfg_acc_num,
	idata,
	idata_valid,
	odata,
	odata_valid
);
	parameter IDATA_WIDTH = 25;
	parameter ODATA_BIT = 25;
	parameter CDATA_ACCU_NUM_WIDTH = 10;
	input clk;
	input rstn;
	input [CDATA_ACCU_NUM_WIDTH - 1:0] cfg_acc_num;
	input [IDATA_WIDTH - 1:0] idata;
	input idata_valid;
	output wire signed [ODATA_BIT - 1:0] odata;
	output wire odata_valid;
	wire finish;
	core_acc_ctrl #(.CDATA_ACCU_NUM_WIDTH(CDATA_ACCU_NUM_WIDTH)) acc_counter_inst(
		.clk(clk),
		.rstn(rstn),
		.cfg_acc_num(cfg_acc_num),
		.psum_valid(idata_valid),
		.psum_finish(finish)
	);
	core_acc_mac #(
		.IDATA_WIDTH(IDATA_WIDTH),
		.ODATA_BIT(ODATA_BIT)
	) acc_mac_inst(
		.clk(clk),
		.rstn(rstn),
		.finish(finish),
		.idata(idata),
		.idata_valid(idata_valid),
		.odata(odata),
		.odata_valid(odata_valid)
	);
endmodule
module core_acc_ctrl (
	clk,
	rstn,
	cfg_acc_num,
	psum_valid,
	psum_finish
);
	parameter CDATA_ACCU_NUM_WIDTH = 8;
	input clk;
	input rstn;
	input [CDATA_ACCU_NUM_WIDTH - 1:0] cfg_acc_num;
	input psum_valid;
	output reg psum_finish;
	reg [CDATA_ACCU_NUM_WIDTH - 1:0] psum_cnt;
	always @(posedge clk or negedge rstn)
		if (~rstn)
			psum_cnt <= 0;
		else if ((psum_valid && (psum_cnt == (cfg_acc_num - 1))) && (cfg_acc_num != 0))
			psum_cnt <= 0;
		else if (psum_valid)
			psum_cnt <= psum_cnt + 1;
	always @(posedge clk or negedge rstn)
		if (~rstn)
			psum_finish <= 0;
		else if ((psum_valid && (psum_cnt == (cfg_acc_num - 1))) && (cfg_acc_num != 0))
			psum_finish <= 1;
		else
			psum_finish <= 0;
endmodule
module core_acc_mac (
	clk,
	rstn,
	finish,
	idata,
	idata_valid,
	odata,
	odata_valid
);
	parameter IDATA_WIDTH = 32;
	parameter ODATA_BIT = 32;
	input clk;
	input rstn;
	input finish;
	input [IDATA_WIDTH - 1:0] idata;
	input idata_valid;
	output reg [ODATA_BIT - 1:0] odata;
	output reg odata_valid;
	reg signed [ODATA_BIT - 1:0] acc_reg;
	always @(posedge clk or negedge rstn)
		if (!rstn)
			acc_reg <= 'd0;
		else if (finish) begin
			if (idata_valid)
				acc_reg <= idata;
			else
				acc_reg <= 1'sb0;
		end
		else if (idata_valid)
			acc_reg <= idata + acc_reg;
	always @(posedge clk or negedge rstn)
		if (!rstn)
			odata <= 'd0;
		else if (finish)
			odata <= acc_reg;
	always @(posedge clk or negedge rstn)
		if (!rstn)
			odata_valid <= 1'b0;
		else
			odata_valid <= finish;
endmodule

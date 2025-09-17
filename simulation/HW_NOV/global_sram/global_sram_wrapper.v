module global_sram_wrapper (
	clk,
	rst_n,
	global_sram_waddr,
	global_sram_wen,
	global_sram_wdata,
	global_sram_bwe,
	global_sram_raddr,
	global_sram_ren,
	global_sram_rdata,
	global_sram_rvld,
	global_mem_addr,
	global_mem_wdata,
	global_mem_wen,
	global_mem_rdata,
	global_mem_ren,
	global_mem_rvld
);
	reg _sv2v_0;
	parameter DATA_BIT = 128;
	parameter DEPTH = 32;
	input clk;
	input rst_n;
	input [$clog2(DEPTH) - 1:0] global_sram_waddr;
	input global_sram_wen;
	input [DATA_BIT - 1:0] global_sram_wdata;
	input [DATA_BIT - 1:0] global_sram_bwe;
	input [$clog2(DEPTH) - 1:0] global_sram_raddr;
	input global_sram_ren;
	output wire [DATA_BIT - 1:0] global_sram_rdata;
	output reg global_sram_rvld;
	input wire [7:0] global_mem_addr;
	input wire [15:0] global_mem_wdata;
	input wire global_mem_wen;
	output wire [15:0] global_mem_rdata;
	input wire global_mem_ren;
	output reg global_mem_rvld;
	reg [$clog2(DEPTH) - 1:0] global_sram_waddr_f;
	reg global_sram_wen_f;
	reg [DATA_BIT - 1:0] global_sram_wdata_f;
	reg [DATA_BIT - 1:0] global_sram_bwe_f;
	reg [$clog2(DEPTH) - 1:0] global_sram_raddr_f;
	reg global_sram_ren_f;
	reg [$clog2(DEPTH) - 1:0] interface_waddr;
	reg interface_wen;
	reg [DATA_BIT - 1:0] interface_wdata;
	reg [DATA_BIT - 1:0] interface_bwe;
	reg [$clog2(DEPTH) - 1:0] interface_raddr;
	reg interface_ren;
	reg [7:0] global_mem_addr_delay1;
	reg [7:0] global_mem_addr_delay2;
	always @(posedge clk) begin
		global_mem_addr_delay1 <= global_mem_addr;
		global_mem_addr_delay2 <= global_mem_addr_delay1;
	end
	always @(posedge clk or negedge rst_n)
		if (~rst_n) begin
			interface_wen <= 0;
			interface_waddr <= 0;
			interface_wdata <= 0;
			interface_bwe <= 0;
		end
		else if (global_mem_wen) begin
			interface_wen <= 1;
			interface_waddr <= global_mem_addr[3+:$clog2(DEPTH)];
			interface_wdata[global_mem_addr[2:0] * 16+:16] <= global_mem_wdata;
			interface_bwe[global_mem_addr[2:0] * 16+:16] <= 16'hffff;
		end
		else begin
			interface_bwe <= 0;
			interface_wen <= 0;
		end
	always @(posedge clk or negedge rst_n)
		if (~rst_n) begin
			interface_ren <= 0;
			interface_raddr <= 0;
		end
		else if (global_mem_ren) begin
			interface_ren <= 1;
			interface_raddr <= global_mem_addr[3+:$clog2(DEPTH)];
		end
		else
			interface_ren <= 0;
	always @(posedge clk or negedge rst_n)
		if (~rst_n)
			global_sram_rvld <= 0;
		else if (global_sram_ren)
			global_sram_rvld <= 1;
		else
			global_sram_rvld <= 0;
	always @(posedge clk or negedge rst_n)
		if (~rst_n)
			global_mem_rvld <= 0;
		else if (interface_ren)
			global_mem_rvld <= 1;
		else
			global_mem_rvld <= 0;
	always @(*) begin
		if (_sv2v_0)
			;
		global_sram_waddr_f = global_sram_waddr;
		global_sram_wen_f = global_sram_wen;
		global_sram_wdata_f = global_sram_wdata;
		global_sram_bwe_f = global_sram_bwe;
		global_sram_raddr_f = global_sram_raddr;
		global_sram_ren_f = global_sram_ren;
		if (interface_wen) begin
			global_sram_waddr_f = interface_waddr;
			global_sram_wen_f = interface_wen;
			global_sram_wdata_f = interface_wdata;
			global_sram_bwe_f = interface_bwe;
		end
		else if (interface_ren) begin
			global_sram_raddr_f = interface_raddr;
			global_sram_ren_f = interface_ren;
		end
	end
	assign global_mem_rdata = global_sram_rdata[global_mem_addr_delay2[2:0] * 16+:16];
	mem_dp #(
		.DATA_BIT(DATA_BIT),
		.DEPTH(DEPTH),
		.BWE(1)
	) inst_mem_dp(
		.clk(clk),
		.waddr(global_sram_waddr_f),
		.wen(global_sram_wen_f),
		.bwe(global_sram_bwe_f),
		.wdata(global_sram_wdata_f),
		.raddr(global_sram_raddr_f),
		.ren(global_sram_ren_f),
		.rdata(global_sram_rdata)
	);
	initial _sv2v_0 = 0;
endmodule

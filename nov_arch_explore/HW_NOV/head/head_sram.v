module head_sram (
	clk,
	rstn,
	interface_addr,
	interface_ren,
	interface_rdata,
	interface_rvalid,
	interface_wen,
	interface_wdata,
	raddr,
	ren,
	rdata,
	waddr,
	wen,
	wdata,
	wdata_byte_flag
);
	reg _sv2v_0;
	parameter DATA_WIDTH = 128;
	parameter BANK_NUM = 16;
	parameter BANK_DEPTH = 32;
	parameter ADDR_WIDTH = $clog2(BANK_NUM) + $clog2(BANK_DEPTH);
	parameter SINGLE_SLICE_DATA_BIT = 8;
	input clk;
	input rstn;
	input [$clog2(BANK_DEPTH) + 2:0] interface_addr;
	input interface_ren;
	output wire [15:0] interface_rdata;
	output reg interface_rvalid;
	input interface_wen;
	input [15:0] interface_wdata;
	input [ADDR_WIDTH - 1:0] raddr;
	input ren;
	output wire [DATA_WIDTH - 1:0] rdata;
	input [ADDR_WIDTH - 1:0] waddr;
	input wen;
	input [DATA_WIDTH - 1:0] wdata;
	input [1:0] wdata_byte_flag;
	reg inst_wen;
	reg inst_ren;
	reg [$clog2(BANK_DEPTH) - 1:0] inst_waddr;
	reg [$clog2(BANK_DEPTH) - 1:0] inst_raddr;
	reg [DATA_WIDTH - 1:0] inst_wdata;
	reg [DATA_WIDTH - 1:0] inst_bwe;
	wire [DATA_WIDTH - 1:0] inst_rdata;
	wire [$clog2(BANK_NUM) - 1:0] bank_wsel;
	wire [$clog2(BANK_DEPTH) - 1:0] bank_waddr;
	assign bank_wsel = waddr[ADDR_WIDTH - 1-:$clog2(BANK_NUM)];
	assign bank_waddr = waddr[0+:$clog2(BANK_DEPTH)];
	reg nxt_inst_wen;
	reg [$clog2(BANK_DEPTH) - 1:0] nxt_inst_waddr;
	reg [DATA_WIDTH - 1:0] nxt_inst_wdata;
	reg [DATA_WIDTH - 1:0] nxt_inst_bwe;
	always @(*) begin
		nxt_inst_wen = wen;
		nxt_inst_waddr = bank_waddr;
		nxt_inst_wdata = 0;
		nxt_inst_bwe = 'b0;
		begin : sv2v_autoblock_1
			reg signed [31:0] i;
			for (i = 0; i < BANK_NUM; i = i + 1)
				if (wdata_byte_flag == 0) begin
					if (wen) begin
						nxt_inst_wdata[i * SINGLE_SLICE_DATA_BIT+:SINGLE_SLICE_DATA_BIT] = wdata[i * SINGLE_SLICE_DATA_BIT+:SINGLE_SLICE_DATA_BIT];
						nxt_inst_bwe = {DATA_WIDTH {1'b1}};
					end
				end
				else if (wdata_byte_flag == 1) begin
					if ((i == bank_wsel) && wen) begin
						nxt_inst_wdata[i * 8+:8] = wdata[7:0];
						nxt_inst_bwe[i * 8+:8] = {8 {1'b1}};
					end
				end
				else if (wdata_byte_flag == 2) begin
					if (((i == bank_wsel) && ((((i == 0) || (i == 4)) || (i == 8)) || (i == 12))) && wen && ((i * 8 + 32) <= 128)) begin
						nxt_inst_wdata[i * 8+:32] = wdata[31:0];
						nxt_inst_bwe[i * 8+:32] = {32 {1'b1}};
					end
				end
		end
	end
	always @(posedge clk or negedge rstn)
		if (!rstn) begin
			inst_wen <= 1'sb0;
			inst_waddr <= 'b0;
			inst_wdata <= 1'sb0;
			inst_bwe <= 1'sb0;
		end
		else if (wen) begin
			inst_wen <= nxt_inst_wen;
			inst_waddr <= nxt_inst_waddr;
			inst_wdata <= nxt_inst_wdata;
			inst_bwe <= nxt_inst_bwe;
		end
		else begin
			inst_wen <= 0;
			inst_bwe <= 0;
		end
	always @(*) begin
		inst_ren = 1'b0;
		inst_raddr = 'b0;
		if (ren) begin
			inst_ren = 1'b1;
			inst_raddr = raddr[0+:$clog2(BANK_DEPTH)];
		end
	end
	reg [$clog2(BANK_DEPTH) + 2:0] interface_addr_delay1;
	reg [$clog2(BANK_DEPTH) + 2:0] interface_addr_delay2;
	reg interface_ren_delay1;
	always @(posedge clk or negedge rstn)
		if (~rstn) begin
			interface_ren_delay1 <= 0;
			interface_rvalid <= 0;
		end
		else begin
			interface_ren_delay1 <= interface_ren;
			interface_rvalid <= interface_ren_delay1;
		end
	assign rdata = inst_rdata;
	assign interface_rdata = inst_rdata[interface_addr_delay2[2:0] * 16+:16];
	reg inst_wen_f;
	reg [$clog2(BANK_DEPTH) - 1:0] inst_waddr_f;
	reg [DATA_WIDTH - 1:0] inst_wdata_f;
	reg [DATA_WIDTH - 1:0] inst_bwe_f;
	reg inst_ren_f;
	reg [$clog2(BANK_DEPTH) - 1:0] inst_raddr_f;
	reg interface_inst_ren;
	reg [$clog2(BANK_DEPTH) - 1:0] interface_inst_raddr;
	reg interface_inst_wen;
	reg [$clog2(BANK_DEPTH) - 1:0] interface_inst_waddr;
	reg [DATA_WIDTH - 1:0] interface_inst_wdata;
	reg [DATA_WIDTH - 1:0] interface_inst_bwe;
	always @(posedge clk or negedge rstn)
		if (~rstn) begin
			interface_inst_wen <= 0;
			interface_inst_waddr <= 0;
			interface_inst_wdata <= 0;
			interface_inst_bwe <= 0;
		end
		else if (interface_wen) begin
			interface_inst_wen <= 1;
			interface_inst_waddr <= interface_addr[3+:$clog2(BANK_DEPTH)];
			interface_inst_wdata[interface_addr[2:0] * 16+:16] <= interface_wdata;
			interface_inst_bwe[interface_addr[2:0] * 16+:16] <= 16'hffff;
		end
		else begin
			interface_inst_bwe <= 0;
			interface_inst_wen <= 0;
		end
	always @(posedge clk or negedge rstn)
		if (~rstn) begin
			interface_inst_ren <= 0;
			interface_inst_raddr <= 0;
		end
		else if (interface_ren) begin
			interface_inst_ren <= 1;
			interface_inst_raddr <= interface_addr[3+:$clog2(BANK_DEPTH)];
		end
		else
			interface_inst_ren <= 0;
	always @(*) begin
		if (_sv2v_0)
			;
		if (interface_inst_wen) begin
			inst_wen_f = interface_inst_wen;
			inst_waddr_f = interface_inst_waddr;
			inst_wdata_f = interface_inst_wdata;
			inst_bwe_f = interface_inst_bwe;
		end
		else begin
			inst_wen_f = inst_wen;
			inst_waddr_f = inst_waddr;
			inst_wdata_f = inst_wdata;
			inst_bwe_f = inst_bwe;
		end
	end
	always @(*) begin
		if (_sv2v_0)
			;
		if (interface_inst_ren) begin
			inst_ren_f = interface_inst_ren;
			inst_raddr_f = interface_inst_raddr;
		end
		else begin
			inst_ren_f = inst_ren;
			inst_raddr_f = inst_raddr;
		end
	end
	always @(posedge clk or negedge rstn)
		if (~rstn) begin
			interface_addr_delay1 <= 0;
			interface_addr_delay2 <= 0;
		end
		else begin
			interface_addr_delay1 <= interface_addr;
			interface_addr_delay2 <= interface_addr_delay1;
		end
	mem_dp_sky130_wrapper #(
		.DATA_BIT(SINGLE_SLICE_DATA_BIT * 16),
		.DEPTH(BANK_DEPTH),
		.BWE(1)
	) ram_piece(
		.clk(clk),
		.waddr(inst_waddr_f),
		.wen(inst_wen_f),
		.bwe(inst_bwe_f),
		.wdata(inst_wdata_f),
		.raddr(inst_raddr_f),
		.ren(inst_ren_f),
		.rdata(inst_rdata)
	);
	initial _sv2v_0 = 0;
endmodule


module mem_dp_sky130_wrapper #(
    parameter   DATA_BIT = 32,         
    parameter   DEPTH = 128,          
    parameter   ADDR_BIT = $clog2(DEPTH),  
    parameter   BWE = 0               
)(
    input                       clk,
    input       [ADDR_BIT-1:0]  waddr, 
    input                       wen,    
    input       [DATA_BIT-1:0]  wdata,  
    input       [DATA_BIT-1:0]  bwe,    
    input       [ADDR_BIT-1:0]  raddr,  
    input                       ren,    
    output wire [DATA_BIT-1:0]  rdata   
);

sky130_sram_0kbytes_1r1w_32x128_32 sky130_sram_inst (
`ifdef USE_POWER_PINS
    .vccd1(vccd1),  
    .vssd1(vssd1),  
`endif

    .clk0(clk),            
    .csb0(~wen),           
    .addr0(waddr),         
    .din0(wdata),          
    .clk1(clk),            
    .csb1(~ren),           
    .addr1(raddr),         
    .dout1(rdata)          
);

endmodule
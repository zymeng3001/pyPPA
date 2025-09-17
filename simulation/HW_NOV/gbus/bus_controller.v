module bus_controller (
	clk,
	rst_n,
	bus_packet,
	bus_packet_vld,
	in_bus_packet_array,
	bus_req_array,
	bus_grant_array
);
	reg _sv2v_0;
	input wire clk;
	input wire rst_n;
	localparam integer BUS_CMEM_ADDR_WIDTH = 13;
	localparam integer BUS_CORE_ADDR_WIDTH = 4;
	localparam integer HEAD_SRAM_BIAS_WIDTH = 2;
	localparam integer BUS_DATA_WIDTH = 32;
	output reg [(BUS_DATA_WIDTH + ((HEAD_SRAM_BIAS_WIDTH + BUS_CORE_ADDR_WIDTH) + BUS_CMEM_ADDR_WIDTH)) - 1:0] bus_packet;
	output reg bus_packet_vld;
	input wire [(16 * (BUS_DATA_WIDTH + ((HEAD_SRAM_BIAS_WIDTH + BUS_CORE_ADDR_WIDTH) + BUS_CMEM_ADDR_WIDTH))) - 1:0] in_bus_packet_array;
	input wire [15:0] bus_req_array;
	output wire [15:0] bus_grant_array;
	reg [(BUS_DATA_WIDTH + ((HEAD_SRAM_BIAS_WIDTH + BUS_CORE_ADDR_WIDTH) + BUS_CMEM_ADDR_WIDTH)) - 1:0] nxt_bus_packet;
	reg nxt_bus_packet_vld;
	wire [(16 * (BUS_DATA_WIDTH + ((HEAD_SRAM_BIAS_WIDTH + BUS_CORE_ADDR_WIDTH) + BUS_CMEM_ADDR_WIDTH))) - 1:0] bus_packet_array;
	reg [15:0] bus_grant_array_delay1;
	always @(posedge clk or negedge rst_n)
		if (~rst_n)
			bus_grant_array_delay1 <= 0;
		else
			bus_grant_array_delay1 <= bus_grant_array;
	always @(posedge clk or negedge rst_n)
		if (~rst_n) begin
			bus_packet <= 0;
			bus_packet_vld <= 0;
		end
		else begin
			bus_packet <= nxt_bus_packet;
			bus_packet_vld <= nxt_bus_packet_vld;
		end
	always @(*) begin
		if (_sv2v_0)
			;
		nxt_bus_packet = 0;
		nxt_bus_packet[(((HEAD_SRAM_BIAS_WIDTH + BUS_CORE_ADDR_WIDTH) + BUS_CMEM_ADDR_WIDTH) - 1) - ((((HEAD_SRAM_BIAS_WIDTH + BUS_CORE_ADDR_WIDTH) + BUS_CMEM_ADDR_WIDTH) - 1) - (BUS_CORE_ADDR_WIDTH + (BUS_CMEM_ADDR_WIDTH - 1)))-:((BUS_CORE_ADDR_WIDTH + (BUS_CMEM_ADDR_WIDTH - 1)) >= (BUS_CMEM_ADDR_WIDTH + 0) ? ((BUS_CORE_ADDR_WIDTH + (BUS_CMEM_ADDR_WIDTH - 1)) - (BUS_CMEM_ADDR_WIDTH + 0)) + 1 : ((BUS_CMEM_ADDR_WIDTH + 0) - (BUS_CORE_ADDR_WIDTH + (BUS_CMEM_ADDR_WIDTH - 1))) + 1)] = -1;
		begin : sv2v_autoblock_1
			reg signed [31:0] i;
			for (i = 0; i < 16; i = i + 1)
				if (bus_grant_array_delay1[i] == 1)
					nxt_bus_packet = in_bus_packet_array[i * (BUS_DATA_WIDTH + ((HEAD_SRAM_BIAS_WIDTH + BUS_CORE_ADDR_WIDTH) + BUS_CMEM_ADDR_WIDTH))+:BUS_DATA_WIDTH + ((HEAD_SRAM_BIAS_WIDTH + BUS_CORE_ADDR_WIDTH) + BUS_CMEM_ADDR_WIDTH)];
		end
	end
	always @(posedge clk or negedge rst_n)
		if (~rst_n)
			nxt_bus_packet_vld <= 0;
		else if (|bus_req_array)
			nxt_bus_packet_vld <= 1;
		else
			nxt_bus_packet_vld <= 0;
	arbiter inst_arbiter(
		.clk(clk),
		.rst_n(rst_n),
		.first_priority(16'b0000000000000001),
		.req(bus_req_array),
		.grant(bus_grant_array)
	);
	initial _sv2v_0 = 0;
endmodule

module bus_packet_fifo (
	clk,
	rst_n,
	in_bus_packet,
	wr_en,
	buffer_full,
	out_bus_packet,
	rd_en,
	buffer_empty,
	fifo_full_error
);
	parameter index = 0;
	input wire clk;
	input wire rst_n;
	localparam integer BUS_CMEM_ADDR_WIDTH = 13;
	localparam integer BUS_CORE_ADDR_WIDTH = 4;
	localparam integer HEAD_SRAM_BIAS_WIDTH = 2;
	localparam integer BUS_DATA_WIDTH = 32;
	input wire [(BUS_DATA_WIDTH + ((HEAD_SRAM_BIAS_WIDTH + BUS_CORE_ADDR_WIDTH) + BUS_CMEM_ADDR_WIDTH)) - 1:0] in_bus_packet;
	input wire wr_en;
	output wire buffer_full;
	output reg [(BUS_DATA_WIDTH + ((HEAD_SRAM_BIAS_WIDTH + BUS_CORE_ADDR_WIDTH) + BUS_CMEM_ADDR_WIDTH)) - 1:0] out_bus_packet;
	input wire rd_en;
	output wire buffer_empty;
	output wire fifo_full_error;
	reg [(28 * (BUS_DATA_WIDTH + ((HEAD_SRAM_BIAS_WIDTH + BUS_CORE_ADDR_WIDTH) + BUS_CMEM_ADDR_WIDTH))) - 1:0] bus_packet_buffer;
	reg [4:0] buffer_wr_ptr;
	reg [4:0] buffer_rd_ptr;
	reg [5:0] fifo_cnt;
	assign fifo_full_error = buffer_full;
	assign buffer_full = fifo_cnt == 28;
	assign buffer_empty = fifo_cnt == 0;
	always @(posedge clk or negedge rst_n)
		if (~rst_n)
			fifo_cnt <= 0;
		else if ((wr_en && ~buffer_full) && ~rd_en)
			fifo_cnt <= fifo_cnt + 1;
		else if ((rd_en && ~buffer_empty) && ~wr_en)
			fifo_cnt <= fifo_cnt - 1;
	always @(posedge clk or negedge rst_n)
		if (~rst_n)
			buffer_wr_ptr <= 0;
		else if (wr_en && ~buffer_full) begin
			bus_packet_buffer[buffer_wr_ptr * (BUS_DATA_WIDTH + ((HEAD_SRAM_BIAS_WIDTH + BUS_CORE_ADDR_WIDTH) + BUS_CMEM_ADDR_WIDTH))+:BUS_DATA_WIDTH + ((HEAD_SRAM_BIAS_WIDTH + BUS_CORE_ADDR_WIDTH) + BUS_CMEM_ADDR_WIDTH)] <= in_bus_packet;
			if (buffer_wr_ptr == 27)
				buffer_wr_ptr <= 0;
			else
				buffer_wr_ptr <= buffer_wr_ptr + 1;
		end
	always @(posedge clk or negedge rst_n)
		if (~rst_n) begin
			out_bus_packet <= 1'sb0;
			buffer_rd_ptr <= 1'sb0;
		end
		else if (rd_en && ~buffer_empty) begin
			out_bus_packet <= bus_packet_buffer[buffer_rd_ptr * (BUS_DATA_WIDTH + ((HEAD_SRAM_BIAS_WIDTH + BUS_CORE_ADDR_WIDTH) + BUS_CMEM_ADDR_WIDTH))+:BUS_DATA_WIDTH + ((HEAD_SRAM_BIAS_WIDTH + BUS_CORE_ADDR_WIDTH) + BUS_CMEM_ADDR_WIDTH)];
			if (buffer_rd_ptr == 27)
				buffer_rd_ptr <= 0;
			else
				buffer_rd_ptr <= buffer_rd_ptr + 1;
		end
	always @(negedge clk)
		;
endmodule

module gbus_top (
	clk,
	rst_n,
	hca_gbus_addr_array,
	hca_gbus_wen_array,
	hca_gbus_wdata_array,
	gbus_addr,
	gbus_wen,
	gbus_wdata
);
	reg _sv2v_0;
	parameter HEAD_CORE_NUM = 16;
	parameter GBUS_DATA_WIDTH = 32;
	input wire clk;
	input wire rst_n;
	localparam integer BUS_CMEM_ADDR_WIDTH = 13;
	localparam integer BUS_CORE_ADDR_WIDTH = 4;
	localparam integer HEAD_SRAM_BIAS_WIDTH = 2;
	input wire [(HEAD_CORE_NUM * ((HEAD_SRAM_BIAS_WIDTH + BUS_CORE_ADDR_WIDTH) + BUS_CMEM_ADDR_WIDTH)) - 1:0] hca_gbus_addr_array;
	input wire [HEAD_CORE_NUM - 1:0] hca_gbus_wen_array;
	input wire [(HEAD_CORE_NUM * GBUS_DATA_WIDTH) - 1:0] hca_gbus_wdata_array;
	output wire [((HEAD_SRAM_BIAS_WIDTH + BUS_CORE_ADDR_WIDTH) + BUS_CMEM_ADDR_WIDTH) - 1:0] gbus_addr;
	output wire gbus_wen;
	output wire [GBUS_DATA_WIDTH - 1:0] gbus_wdata;
	localparam integer BUS_DATA_WIDTH = 32;
	reg [(HEAD_CORE_NUM * (BUS_DATA_WIDTH + ((HEAD_SRAM_BIAS_WIDTH + BUS_CORE_ADDR_WIDTH) + BUS_CMEM_ADDR_WIDTH))) - 1:0] fifo_in_bus_packet_array;
	reg [HEAD_CORE_NUM - 1:0] wr_en_array;
	wire [(HEAD_CORE_NUM * (BUS_DATA_WIDTH + ((HEAD_SRAM_BIAS_WIDTH + BUS_CORE_ADDR_WIDTH) + BUS_CMEM_ADDR_WIDTH))) - 1:0] fifo_out_bus_packet_array;
	wire [HEAD_CORE_NUM - 1:0] rd_en_array;
	wire [HEAD_CORE_NUM - 1:0] buffer_empty_array;
	always @(*) begin
		if (_sv2v_0)
			;
		begin : sv2v_autoblock_1
			reg signed [31:0] i;
			for (i = 0; i < HEAD_CORE_NUM; i = i + 1)
				begin
					fifo_in_bus_packet_array[(i * (BUS_DATA_WIDTH + ((HEAD_SRAM_BIAS_WIDTH + BUS_CORE_ADDR_WIDTH) + BUS_CMEM_ADDR_WIDTH))) + (BUS_DATA_WIDTH + (((HEAD_SRAM_BIAS_WIDTH + BUS_CORE_ADDR_WIDTH) + BUS_CMEM_ADDR_WIDTH) - 1))-:((BUS_DATA_WIDTH + (((HEAD_SRAM_BIAS_WIDTH + BUS_CORE_ADDR_WIDTH) + BUS_CMEM_ADDR_WIDTH) - 1)) >= (((HEAD_SRAM_BIAS_WIDTH + BUS_CORE_ADDR_WIDTH) + BUS_CMEM_ADDR_WIDTH) + 0) ? ((BUS_DATA_WIDTH + (((HEAD_SRAM_BIAS_WIDTH + BUS_CORE_ADDR_WIDTH) + BUS_CMEM_ADDR_WIDTH) - 1)) - (((HEAD_SRAM_BIAS_WIDTH + BUS_CORE_ADDR_WIDTH) + BUS_CMEM_ADDR_WIDTH) + 0)) + 1 : ((((HEAD_SRAM_BIAS_WIDTH + BUS_CORE_ADDR_WIDTH) + BUS_CMEM_ADDR_WIDTH) + 0) - (BUS_DATA_WIDTH + (((HEAD_SRAM_BIAS_WIDTH + BUS_CORE_ADDR_WIDTH) + BUS_CMEM_ADDR_WIDTH) - 1))) + 1)] = hca_gbus_wdata_array[i * GBUS_DATA_WIDTH+:GBUS_DATA_WIDTH];
					fifo_in_bus_packet_array[(i * (BUS_DATA_WIDTH + ((HEAD_SRAM_BIAS_WIDTH + BUS_CORE_ADDR_WIDTH) + BUS_CMEM_ADDR_WIDTH))) + (((HEAD_SRAM_BIAS_WIDTH + BUS_CORE_ADDR_WIDTH) + BUS_CMEM_ADDR_WIDTH) - 1)-:(HEAD_SRAM_BIAS_WIDTH + BUS_CORE_ADDR_WIDTH) + BUS_CMEM_ADDR_WIDTH] = hca_gbus_addr_array[i * ((HEAD_SRAM_BIAS_WIDTH + BUS_CORE_ADDR_WIDTH) + BUS_CMEM_ADDR_WIDTH)+:(HEAD_SRAM_BIAS_WIDTH + BUS_CORE_ADDR_WIDTH) + BUS_CMEM_ADDR_WIDTH];
					wr_en_array[i] = hca_gbus_wen_array[i];
				end
		end
	end
	genvar _gv_i_1;
	generate
		for (_gv_i_1 = 0; _gv_i_1 < HEAD_CORE_NUM; _gv_i_1 = _gv_i_1 + 1) begin : bus_packet_fifo_gen_array
			localparam i = _gv_i_1;
			bus_packet_fifo #(.index(i)) inst_bus_packet_fifo(
				.clk(clk),
				.rst_n(rst_n),
				.in_bus_packet(fifo_in_bus_packet_array[i * (BUS_DATA_WIDTH + ((HEAD_SRAM_BIAS_WIDTH + BUS_CORE_ADDR_WIDTH) + BUS_CMEM_ADDR_WIDTH))+:BUS_DATA_WIDTH + ((HEAD_SRAM_BIAS_WIDTH + BUS_CORE_ADDR_WIDTH) + BUS_CMEM_ADDR_WIDTH)]),
				.wr_en(wr_en_array[i]),
				.buffer_full(),
				.out_bus_packet(fifo_out_bus_packet_array[i * (BUS_DATA_WIDTH + ((HEAD_SRAM_BIAS_WIDTH + BUS_CORE_ADDR_WIDTH) + BUS_CMEM_ADDR_WIDTH))+:BUS_DATA_WIDTH + ((HEAD_SRAM_BIAS_WIDTH + BUS_CORE_ADDR_WIDTH) + BUS_CMEM_ADDR_WIDTH)]),
				.rd_en(rd_en_array[i]),
				.buffer_empty(buffer_empty_array[i]),
				.fifo_full_error()
			);
		end
	endgenerate
	wire [(BUS_DATA_WIDTH + ((HEAD_SRAM_BIAS_WIDTH + BUS_CORE_ADDR_WIDTH) + BUS_CMEM_ADDR_WIDTH)) - 1:0] bus_packet;
	wire bus_packet_vld;
	assign gbus_addr = bus_packet[((HEAD_SRAM_BIAS_WIDTH + BUS_CORE_ADDR_WIDTH) + BUS_CMEM_ADDR_WIDTH) - 1-:(HEAD_SRAM_BIAS_WIDTH + BUS_CORE_ADDR_WIDTH) + BUS_CMEM_ADDR_WIDTH];
	assign gbus_wdata = bus_packet[BUS_DATA_WIDTH + (((HEAD_SRAM_BIAS_WIDTH + BUS_CORE_ADDR_WIDTH) + BUS_CMEM_ADDR_WIDTH) - 1)-:((BUS_DATA_WIDTH + (((HEAD_SRAM_BIAS_WIDTH + BUS_CORE_ADDR_WIDTH) + BUS_CMEM_ADDR_WIDTH) - 1)) >= (((HEAD_SRAM_BIAS_WIDTH + BUS_CORE_ADDR_WIDTH) + BUS_CMEM_ADDR_WIDTH) + 0) ? ((BUS_DATA_WIDTH + (((HEAD_SRAM_BIAS_WIDTH + BUS_CORE_ADDR_WIDTH) + BUS_CMEM_ADDR_WIDTH) - 1)) - (((HEAD_SRAM_BIAS_WIDTH + BUS_CORE_ADDR_WIDTH) + BUS_CMEM_ADDR_WIDTH) + 0)) + 1 : ((((HEAD_SRAM_BIAS_WIDTH + BUS_CORE_ADDR_WIDTH) + BUS_CMEM_ADDR_WIDTH) + 0) - (BUS_DATA_WIDTH + (((HEAD_SRAM_BIAS_WIDTH + BUS_CORE_ADDR_WIDTH) + BUS_CMEM_ADDR_WIDTH) - 1))) + 1)];
	assign gbus_wen = bus_packet_vld;
	wire [(HEAD_CORE_NUM * (BUS_DATA_WIDTH + ((HEAD_SRAM_BIAS_WIDTH + BUS_CORE_ADDR_WIDTH) + BUS_CMEM_ADDR_WIDTH))) - 1:0] ctrl_in_bus_packet_array;
	wire [HEAD_CORE_NUM - 1:0] bus_req_array;
	wire [HEAD_CORE_NUM - 1:0] bus_grant_array;
	assign bus_req_array = ~buffer_empty_array;
	assign rd_en_array = bus_grant_array;
	assign ctrl_in_bus_packet_array = fifo_out_bus_packet_array;
	bus_controller inst_bus_ctrl(
		.clk(clk),
		.rst_n(rst_n),
		.bus_packet(bus_packet),
		.bus_packet_vld(bus_packet_vld),
		.in_bus_packet_array(ctrl_in_bus_packet_array),
		.bus_req_array(bus_req_array),
		.bus_grant_array(bus_grant_array)
	);
	initial _sv2v_0 = 0;
endmodule

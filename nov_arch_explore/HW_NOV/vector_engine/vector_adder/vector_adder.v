module vector_adder (
	clk,
	rst_n,
	in_data,
	in_data_vld,
	in_data_addr,
	op_cfg_vld,
	op_cfg,
	out_data,
	out_data_vld,
	out_data_addr,
	in_finish,
	out_finish
);
	input wire clk;
	input wire rst_n;
	input wire [255:0] in_data;
	input wire [7:0] in_data_vld;
	input wire [12:0] in_data_addr;
	input wire op_cfg_vld;
	input wire [40:0] op_cfg;
	output reg [7:0] out_data;
	output reg out_data_vld;
	output wire [12:0] out_data_addr;
	input wire in_finish;
	output wire out_finish;
	localparam SUM_WIDTH = 28;
	reg [40:0] op_cfg_reg;
	always @(posedge clk or negedge rst_n)
		if (~rst_n)
			op_cfg_reg <= 0;
		else if (op_cfg_vld)
			op_cfg_reg <= op_cfg;
	localparam delay_latency = 13;
	genvar _gv_i_1;
	generate
		for (_gv_i_1 = 0; _gv_i_1 < delay_latency; _gv_i_1 = _gv_i_1 + 1) begin : addr_delay_gen_array
			localparam i = _gv_i_1;
			reg [12:0] gen_data_addr;
			reg temp_finish;
			if (i == 0) begin : genblk1
				always @(posedge clk) gen_data_addr <= in_data_addr;
				always @(posedge clk or negedge rst_n)
					if (~rst_n)
						temp_finish <= 0;
					else
						temp_finish <= in_finish;
			end
			else begin : genblk1
				always @(posedge clk) gen_data_addr <= addr_delay_gen_array[i - 1].gen_data_addr;
				always @(posedge clk or negedge rst_n)
					if (~rst_n)
						temp_finish <= 0;
					else
						temp_finish <= addr_delay_gen_array[i - 1].temp_finish;
			end
		end
	endgenerate
	assign out_finish = addr_delay_gen_array[12].temp_finish;
	assign out_data_addr = addr_delay_gen_array[12].gen_data_addr;
	wire [27:0] sum_data;
	wire sum_data_vld;
	reg [199:0] in_data_delay1;
	reg in_data_vld_delay1;
	always @(posedge clk or negedge rst_n)
		if (~rst_n)
			in_data_vld_delay1 <= 0;
		else if (|in_data_vld)
			in_data_vld_delay1 <= 1;
		else
			in_data_vld_delay1 <= 0;
	generate
		for (_gv_i_1 = 0; _gv_i_1 < 8; _gv_i_1 = _gv_i_1 + 1) begin : gen_array_in_data_delay1
			localparam i = _gv_i_1;
			always @(posedge clk or negedge rst_n)
				if (~rst_n)
					in_data_delay1[i * 25+:25] <= 0;
				else if (|in_data_vld)
					in_data_delay1[i * 25+:25] <= in_data[i * 32+:32];
		end
	endgenerate
	wire [7:0] quant_out_data;
	wire quant_out_data_vld;
	reg [7:0] nxt_out_data;
	reg nxt_out_data_vld;
	always @(posedge clk or negedge rst_n)
		if (~rst_n) begin
			nxt_out_data_vld <= 0;
			nxt_out_data <= 0;
		end
		else begin
			nxt_out_data_vld <= quant_out_data_vld;
			nxt_out_data <= quant_out_data;
		end
	always @(posedge clk or negedge rst_n)
		if (~rst_n) begin
			out_data_vld <= 0;
			out_data <= 0;
		end
		else begin
			out_data_vld <= nxt_out_data_vld;
			out_data <= nxt_out_data;
		end
	adder_tree #(
		.MAC_MULT_NUM(8),
		.IDATA_WIDTH(25),
		.ODATA_BIT(SUM_WIDTH)
	) vect_add_tree(
		.clk(clk),
		.rstn(rst_n),
		.idata(in_data_delay1),
		.idata_valid(in_data_vld_delay1),
		.odata(sum_data),
		.odata_valid(sum_data_vld)
	);
	core_quant #(
		.IDATA_WIDTH(SUM_WIDTH),
		.ODATA_BIT(8)
	) vect_core_quant(
		.clk(clk),
		.rstn(rst_n),
		.cfg_quant_scale(op_cfg_reg[30-:10]),
		.cfg_quant_bias(op_cfg_reg[20-:16]),
		.cfg_quant_shift(op_cfg_reg[4-:5]),
		.idata(sum_data),
		.idata_valid(sum_data_vld),
		.odata(quant_out_data),
		.odata_valid(quant_out_data_vld)
	);
endmodule

module core_mac (
	clk,
	rstn,
	idataA,
	idataB,
	idata_valid,
	odata,
	odata_valid
);
	parameter MAC_MULT_NUM = 16;
	parameter IDATA_WIDTH = 8;
	parameter ODATA_BIT = (IDATA_WIDTH * 2) + $clog2(MAC_MULT_NUM);
	input clk;
	input rstn;
	input [(IDATA_WIDTH * MAC_MULT_NUM) - 1:0] idataA;
	input [(IDATA_WIDTH * MAC_MULT_NUM) - 1:0] idataB;
	input idata_valid;
	output wire signed [ODATA_BIT - 1:0] odata;
	output wire odata_valid;
	localparam MAC_ODATA_BIT = (IDATA_WIDTH * 2) + $clog2(MAC_MULT_NUM);
	wire [MAC_ODATA_BIT - 1:0] mac_odata;
	assign odata = $signed(mac_odata);
	wire [((IDATA_WIDTH * 2) * MAC_MULT_NUM) - 1:0] product;
	wire product_valid;
	mul_line #(
		.MAC_MULT_NUM(MAC_MULT_NUM),
		.IDATA_WIDTH(IDATA_WIDTH)
	) mul_inst(
		.clk(clk),
		.rstn(rstn),
		.idataA(idataA),
		.idataB(idataB),
		.idata_valid(idata_valid),
		.odata(product),
		.odata_valid(product_valid)
	);
	adder_tree #(
		.MAC_MULT_NUM(MAC_MULT_NUM),
		.IDATA_WIDTH(IDATA_WIDTH * 2)
	) adt_inst(
		.clk(clk),
		.rstn(rstn),
		.idata(product),
		.idata_valid(product_valid),
		.odata(mac_odata),
		.odata_valid(odata_valid)
	);
endmodule
module mul_line (
	clk,
	rstn,
	idataA,
	idataB,
	idata_valid,
	odata,
	odata_valid
);
	parameter MAC_MULT_NUM = 64;
	parameter IDATA_WIDTH = 8;
	parameter ODATA_BIT = IDATA_WIDTH * 2;
	input clk;
	input rstn;
	input [(IDATA_WIDTH * MAC_MULT_NUM) - 1:0] idataA;
	input [(IDATA_WIDTH * MAC_MULT_NUM) - 1:0] idataB;
	input idata_valid;
	output reg [(ODATA_BIT * MAC_MULT_NUM) - 1:0] odata;
	output reg odata_valid;
	reg [IDATA_WIDTH - 1:0] idataA_reg [0:MAC_MULT_NUM - 1];
	reg [IDATA_WIDTH - 1:0] idataB_reg [0:MAC_MULT_NUM - 1];
	genvar _gv_i_1;
	generate
		for (_gv_i_1 = 0; _gv_i_1 < MAC_MULT_NUM; _gv_i_1 = _gv_i_1 + 1) begin : gen_mul_input
			localparam i = _gv_i_1;
			always @(posedge clk or negedge rstn)
				if (!rstn) begin
					idataA_reg[i] <= 'd0;
					idataB_reg[i] <= 'd0;
				end
				else if (idata_valid) begin
					idataA_reg[i] <= idataA[i * IDATA_WIDTH+:IDATA_WIDTH];
					idataB_reg[i] <= idataB[i * IDATA_WIDTH+:IDATA_WIDTH];
				end
		end
	endgenerate
	wire [ODATA_BIT - 1:0] product [0:MAC_MULT_NUM - 1];
	reg nxt_odata_valid;
	generate
		for (_gv_i_1 = 0; _gv_i_1 < MAC_MULT_NUM; _gv_i_1 = _gv_i_1 + 1) begin : gen_mul
			localparam i = _gv_i_1;
			mul_int #(
				.IDATA_WIDTH(IDATA_WIDTH),
				.ODATA_BIT(ODATA_BIT)
			) mul_inst(
				.idataA(idataA_reg[i]),
				.idataB(idataB_reg[i]),
				.odata(product[i])
			);
		end
		for (_gv_i_1 = 0; _gv_i_1 < MAC_MULT_NUM; _gv_i_1 = _gv_i_1 + 1) begin : gen_mul_output
			localparam i = _gv_i_1;
			always @(posedge clk) odata[i * ODATA_BIT+:ODATA_BIT] <= product[i];
		end
	endgenerate
	always @(posedge clk or negedge rstn)
		if (!rstn) begin
			nxt_odata_valid <= 'd0;
			odata_valid <= 'd0;
		end
		else begin
			nxt_odata_valid <= idata_valid;
			odata_valid <= nxt_odata_valid;
		end
endmodule
module adder_tree (
	clk,
	rstn,
	idata,
	idata_valid,
	odata,
	odata_valid
);
	parameter MAC_MULT_NUM = 64;
	parameter IDATA_WIDTH = 16;
	parameter ODATA_BIT = IDATA_WIDTH + $clog2(MAC_MULT_NUM);
	input clk;
	input rstn;
	input [(IDATA_WIDTH * MAC_MULT_NUM) - 1:0] idata;
	input idata_valid;
	output reg signed [ODATA_BIT - 1:0] odata;
	output reg odata_valid;
	localparam STAGE_NUM = $clog2(MAC_MULT_NUM);
	genvar _gv_i_2;
	genvar _gv_j_1;
	generate
		for (_gv_i_2 = 0; _gv_i_2 < STAGE_NUM; _gv_i_2 = _gv_i_2 + 1) begin : gen_adt_valid
			localparam i = _gv_i_2;
			reg add_valid;
			if (i == 0) begin : genblk1
				always @(posedge clk or negedge rstn)
					if (!rstn)
						add_valid <= 1'b0;
					else
						add_valid <= idata_valid;
			end
			else if ((i % 2) == 1'b0) begin : genblk1
				always @(posedge clk or negedge rstn)
					if (!rstn)
						add_valid <= 1'b0;
					else
						add_valid <= gen_adt_valid[i - 1].add_valid;
			end
			else begin : genblk1
				always @(*) add_valid = gen_adt_valid[i - 1].add_valid;
			end
		end
		for (_gv_i_2 = 0; _gv_i_2 < STAGE_NUM; _gv_i_2 = _gv_i_2 + 1) begin : gen_adt_stage
			localparam i = _gv_i_2;
			localparam OUT_BIT = IDATA_WIDTH + (i + 1'b1);
			localparam OUT_NUM = MAC_MULT_NUM >> (i + 1'b1);
			reg [OUT_BIT - 2:0] add_idata [0:(OUT_NUM * 2) - 1];
			wire [OUT_BIT - 1:0] add_odata [0:OUT_NUM - 1];
			for (_gv_j_1 = 0; _gv_j_1 < OUT_NUM; _gv_j_1 = _gv_j_1 + 1) begin : gen_adt_adder
				localparam j = _gv_j_1;
				if (i == 0) begin : genblk1
					always @(posedge clk or negedge rstn)
						if (!rstn) begin
							add_idata[j * 2] <= 'd0;
							add_idata[(j * 2) + 1] <= 'd0;
						end
						else if (idata_valid) begin
							add_idata[j * 2] <= idata[((j * 2) + 0) * IDATA_WIDTH+:IDATA_WIDTH];
							add_idata[(j * 2) + 1] <= idata[((j * 2) + 1) * IDATA_WIDTH+:IDATA_WIDTH];
						end
				end
				else if ((i % 2) == 0) begin : genblk1
					always @(posedge clk or negedge rstn)
						if (!rstn) begin
							add_idata[j * 2] <= 'd0;
							add_idata[(j * 2) + 1] <= 'd0;
						end
						else if (gen_adt_valid[i - 1].add_valid) begin
							add_idata[j * 2] <= gen_adt_stage[i - 1].add_odata[j * 2];
							add_idata[(j * 2) + 1] <= gen_adt_stage[i - 1].add_odata[(j * 2) + 1];
						end
				end
				else begin : genblk1
					always @(*) begin
						add_idata[j * 2] = gen_adt_stage[i - 1].add_odata[j * 2];
						add_idata[(j * 2) + 1] = gen_adt_stage[i - 1].add_odata[(j * 2) + 1];
					end
				end
				add_int #(
					.IDATA_WIDTH(OUT_BIT - 1),
					.ODATA_BIT(OUT_BIT)
				) adder_inst(
					.idataA(add_idata[j * 2]),
					.idataB(add_idata[(j * 2) + 1]),
					.odata(add_odata[j])
				);
			end
		end
	endgenerate
	reg [ODATA_BIT - 1:0] nxt_odata;
	reg nxt_odata_valid;
	always @(*) begin
		nxt_odata = $signed(gen_adt_stage[STAGE_NUM - 1].add_odata[0]);
		nxt_odata_valid = gen_adt_valid[STAGE_NUM - 1].add_valid;
	end
	always @(posedge clk or negedge rstn)
		if (~rstn) begin
			odata <= 0;
			odata_valid <= 0;
		end
		else begin
			odata <= nxt_odata;
			odata_valid <= nxt_odata_valid;
		end
endmodule

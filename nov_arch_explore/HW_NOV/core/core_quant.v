module core_quant (
	clk,
	rstn,
	cfg_quant_scale,
	cfg_quant_bias,
	cfg_quant_shift,
	idata,
	idata_valid,
	odata,
	odata_valid
);
	parameter IDATA_WIDTH = 25;
	parameter ODATA_BIT = 8;
	parameter CDATA_ACCU_NUM_WIDTH = 10;
	parameter CDATA_SCALE_WIDTH = 10;
	parameter CDATA_BIAS_WIDTH = 16;
	parameter CDATA_SHIFT_WIDTH = 5;
	parameter TEMP_BIT = IDATA_WIDTH + CDATA_SCALE_WIDTH;
	input clk;
	input rstn;
	input [CDATA_SCALE_WIDTH - 1:0] cfg_quant_scale;
	input [CDATA_BIAS_WIDTH - 1:0] cfg_quant_bias;
	input [CDATA_SHIFT_WIDTH - 1:0] cfg_quant_shift;
	input [IDATA_WIDTH - 1:0] idata;
	input idata_valid;
	output reg [ODATA_BIT - 1:0] odata;
	output reg odata_valid;
	reg signed [TEMP_BIT - 1:0] quantized_product;
	reg signed [TEMP_BIT:0] quantized_bias;
	reg signed [TEMP_BIT:0] quantized_bias_reg;
	reg quantized_product_valid;
	reg quantized_bias_valid;
	reg quantized_shift_valid;
	reg quantized_round_valid;
	reg signed [TEMP_BIT:0] quantized_shift;
	reg signed [TEMP_BIT:0] quantized_shift_reg;
	reg signed quantized_round;
	reg signed quantized_round_reg;
	always @(posedge clk or negedge rstn)
		if (!rstn)
			quantized_product <= 'd0;
		else if (idata_valid)
			quantized_product <= $signed(idata) * $signed({1'b0, cfg_quant_scale});
	always @(posedge clk or negedge rstn)
		if (!rstn)
			quantized_product_valid <= 1'b0;
		else
			quantized_product_valid <= idata_valid;
	genvar _gv_i_1;
	generate
		for (_gv_i_1 = 0; _gv_i_1 < 3; _gv_i_1 = _gv_i_1 + 1) begin : quant_scale_retiming_gen_array
			localparam i = _gv_i_1;
			reg quantized_product_valid_delay;
			reg signed [TEMP_BIT - 1:0] quantized_product_delay;
			if (i == 0) begin : genblk1
				always @(posedge clk or negedge rstn)
					if (~rstn) begin
						quantized_product_valid_delay <= 0;
						quantized_product_delay <= 0;
					end
					else begin
						quantized_product_valid_delay <= quantized_product_valid;
						quantized_product_delay <= quantized_product;
					end
			end
			else begin : genblk1
				always @(posedge clk or negedge rstn)
					if (~rstn) begin
						quantized_product_valid_delay <= 0;
						quantized_product_delay <= 0;
					end
					else begin
						quantized_product_valid_delay <= quant_scale_retiming_gen_array[i - 1].quantized_product_valid_delay;
						quantized_product_delay <= quant_scale_retiming_gen_array[i - 1].quantized_product_delay;
					end
			end
		end
	endgenerate
	always @(*) begin
		quantized_bias = $signed(quant_scale_retiming_gen_array[2].quantized_product_delay) + $signed(cfg_quant_bias);
		quantized_round = (cfg_quant_shift > 0 ? quantized_bias[cfg_quant_shift - 1] : 0);
	end
	always @(posedge clk or negedge rstn)
		if (!rstn) begin
			quantized_bias_reg <= 'd0;
			quantized_round_reg <= 1'b0;
		end
		else if (quant_scale_retiming_gen_array[2].quantized_product_valid_delay) begin
			quantized_bias_reg <= quantized_bias;
			quantized_round_reg <= quantized_round;
		end
	always @(posedge clk or negedge rstn)
		if (!rstn)
			quantized_bias_valid <= 1'b0;
		else
			quantized_bias_valid <= quant_scale_retiming_gen_array[2].quantized_product_valid_delay;
	always @(*) begin
		quantized_shift = quantized_bias_reg >>> cfg_quant_shift;
		quantized_shift = (quantized_round_reg ? $signed(quantized_shift) + 1 : $signed(quantized_shift));
	end
	always @(posedge clk or negedge rstn)
		if (!rstn)
			quantized_shift_reg <= 'd0;
		else if (quantized_bias_valid)
			quantized_shift_reg <= quantized_shift;
	always @(posedge clk or negedge rstn)
		if (!rstn)
			quantized_shift_valid <= 1'b0;
		else
			quantized_shift_valid <= quantized_bias_valid;
	reg [ODATA_BIT - 1:0] quantized_overflow;
	always @(*)
		if ((quantized_shift_reg[TEMP_BIT] & ~(&quantized_shift_reg[TEMP_BIT - 1:ODATA_BIT - 1])) || (~quantized_shift_reg[TEMP_BIT] & |quantized_shift_reg[TEMP_BIT - 1:ODATA_BIT - 1]))
			quantized_overflow = {quantized_shift_reg[TEMP_BIT], {ODATA_BIT - 1 {~quantized_shift_reg[TEMP_BIT]}}};
		else
			quantized_overflow = {quantized_shift_reg[TEMP_BIT], quantized_shift_reg[ODATA_BIT - 2:0]};
	always @(posedge clk or negedge rstn)
		if (!rstn)
			odata <= 'd0;
		else if (quantized_shift_valid)
			odata <= quantized_overflow;
	always @(posedge clk or negedge rstn)
		if (!rstn)
			odata_valid <= 'd0;
		else
			odata_valid <= quantized_shift_valid;
endmodule

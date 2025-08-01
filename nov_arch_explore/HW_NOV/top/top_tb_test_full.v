module tb;
	parameter FORCE_TO_BE_RIGHT = 0;
	parameter real CHIP_CLK_FREQ = 500e6;
	parameter real FPGA_CLK_FREQ = 100e6;
	parameter real QSPI_CLK_FREQ = 50e6;
	parameter QKT_MAX_ERR = 6;
	parameter PV_MAX_ERR = 6;
	parameter PROJ_MAX_ERR = 6;
	parameter FFN0_MAX_ERR = 6;
	parameter FFN1_MAX_ERR = 6;
	parameter GQA_EN = 1;
	reg PMU_CFG_EN = 0;
	reg DEEPSLEEP_EN = 0;
	parameter CDATA_ACCU_NUM_WIDTH = 10;
	parameter CDATA_SCALE_WIDTH = 10;
	parameter CDATA_BIAS_WIDTH = 16;
	parameter CDATA_SHIFT_WIDTH = 5;
	localparam K_WEIGHT_ADDR_BASE = 128;
	localparam V_WEIGHT_ADDR_BASE = 256;
	localparam PROJ_WEIGHT_ADDR_BASE = 384;
	localparam FFN0_WEIGHT_ADDR_BASE = 512;
	localparam FFN1_WEIGHT_ADDR_BASE = 1024;
	localparam K_CACHE_ADDR_BASE = 0;
	localparam V_CACHE_ADDR_BASE = 64;
	parameter TB_EMBD_SIZE = 512;
	parameter TB_MAX_CONTEXT_LENGTH = 512;
	parameter TB_TOTAL_CONTEXT_LENGTH = TB_MAX_CONTEXT_LENGTH / 64;
	parameter TB_NUM_USER = 1;
	parameter TB_QKV_WEIGHT_COLS_PER_CORE = (TB_EMBD_SIZE / 8) / 16;
	parameter TB_TOKEN_PER_CORE = TB_MAX_CONTEXT_LENGTH / 16;
	parameter TB_TEST_ITER = TB_TOTAL_CONTEXT_LENGTH * TB_NUM_USER;
	parameter RC_SHIFT = 16;
	parameter RMS_K = 128;
	shortreal attn_rms_dequant_scale_square = 0.0123456;
	shortreal mlp_rms_dequant_scale_square = 0.01404802;
	parameter Q_GEN_SCALE = 120;
	parameter Q_GEN_SHIFT = 15;
	parameter K_GEN_SCALE = 135;
	parameter K_GEN_SHIFT = 15;
	parameter V_GEN_SCALE = 24;
	parameter V_GEN_SHIFT = 13;
	parameter QKT_SCALE = 86;
	parameter QKT_SHIFT = 15;
	shortreal SOFTMAX_DEQUANT_SCALE = 0.06123456;
	shortreal SOFTMAX_QUANT_SCALE = 128;
	parameter PV_SCALE = 227;
	parameter PV_SHIFT = 14;
	parameter PROJ_SCALE = 247;
	parameter PROJ_SHIFT = 19;
	parameter ATTN_RESIDUAL_SCALE_A = 320;
	parameter ATTN_RESIDUAL_SCALE_B = 279;
	parameter ATTN_RESIDUAL_SHIFT = 9;
	parameter FFN0_SCALE = 216;
	parameter FFN0_SHIFT = 16;
	parameter FFN1_SCALE = 201;
	parameter FFN1_SHIFT = 18;
	parameter MLP_RESIDUAL_SCALE_A = 213;
	parameter MLP_RESIDUAL_SCALE_B = 202;
	parameter MLP_RESIDUAL_SHIFT = 8;
	function automatic signed [31:0] sv2v_cast_32_signed;
		input reg signed [31:0] inp;
		sv2v_cast_32_signed = inp;
	endfunction
	function automatic signed [31:0] zero_round;
		input real num;
		zero_round = sv2v_cast_32_signed(num + 0.5);
	endfunction
	function shortreal bitsbfloat16_to_shortreal;
		input [15:0] x;
		reg [31:0] x_float;
		begin
			x_float = {x, 16'b0000000000000000};
			bitsbfloat16_to_shortreal = $bitstoshortreal(x_float);
		end
	endfunction
	function [15:0] shortreal_to_bitsbfloat16;
		input real x;
		reg [31:0] x_float_bits;
		begin
			x_float_bits = $shortrealtobits(x);
			shortreal_to_bitsbfloat16 = x_float_bits[31:16] + x_float_bits[15];
		end
	endfunction
	task QKV_gen_soft;
		input reg [(((TB_NUM_USER * TB_TOTAL_CONTEXT_LENGTH) * TB_EMBD_SIZE) * 8) - 1:0] input_x_array;
		input reg [((TB_EMBD_SIZE * (TB_EMBD_SIZE / 8)) * 8) - 1:0] weights_array;
		input integer scale;
		input integer shift;
		output reg [(((TB_NUM_USER * TB_TOTAL_CONTEXT_LENGTH) * (TB_EMBD_SIZE / 8)) * 8) - 1:0] o_array;
		reg signed [63:0] temp_sum;
		integer round;
		shortreal real_data_array [TB_EMBD_SIZE - 1:0];
		shortreal square_sum;
		shortreal rms;
		shortreal one_over_krms;
		reg signed [31:0] u;
		for (u = 0; u < TB_NUM_USER; u = u + 1)
			begin : sv2v_autoblock_1
				reg signed [31:0] i;
				for (i = 0; i < TB_TOTAL_CONTEXT_LENGTH; i = i + 1)
					begin
						square_sum = 0;
						rms = 0;
						begin : sv2v_autoblock_2
							reg signed [31:0] j;
							for (j = 0; j < TB_EMBD_SIZE; j = j + 1)
								real_data_array[j] = $itor($signed(input_x_array[((((u * TB_TOTAL_CONTEXT_LENGTH) + i) * TB_EMBD_SIZE) + j) * 8+:8]));
						end
						begin : sv2v_autoblock_3
							reg signed [31:0] j;
							for (j = 0; j < RMS_K; j = j + 1)
								square_sum = square_sum + (real_data_array[j] * real_data_array[j]);
						end
						square_sum = square_sum * attn_rms_dequant_scale_square;
						rms = $sqrt((square_sum / RMS_K) * 1.0);
						one_over_krms = 1 / rms;
						begin : sv2v_autoblock_4
							reg signed [31:0] j;
							for (j = 0; j < (TB_EMBD_SIZE / 8); j = j + 1)
								begin
									temp_sum = 0;
									round = 0;
									begin : sv2v_autoblock_5
										reg signed [31:0] k;
										for (k = 0; k < TB_EMBD_SIZE; k = k + 1)
											temp_sum = temp_sum + ($signed(input_x_array[((((u * TB_TOTAL_CONTEXT_LENGTH) + i) * TB_EMBD_SIZE) + k) * 8+:8]) * $signed(weights_array[((k * (TB_EMBD_SIZE / 8)) + j) * 8+:8]));
									end
									temp_sum = $rtoi($itor(temp_sum) * one_over_krms);
									temp_sum = temp_sum * scale;
									round = temp_sum[shift - 1];
									temp_sum = temp_sum >>> shift;
									temp_sum = temp_sum + round;
									if (temp_sum > 127)
										temp_sum = 127;
									if (temp_sum < -128)
										temp_sum = -128;
									o_array[((((u * TB_TOTAL_CONTEXT_LENGTH) + i) * (TB_EMBD_SIZE / 8)) + j) * 8+:8] = temp_sum;
								end
						end
					end
			end
	endtask
	task ATT_QK_soft;
		input reg [(((TB_NUM_USER * TB_TOTAL_CONTEXT_LENGTH) * (TB_EMBD_SIZE / 8)) * 8) - 1:0] input_q_array;
		input reg [(((TB_NUM_USER * TB_TOTAL_CONTEXT_LENGTH) * (TB_EMBD_SIZE / 8)) * 8) - 1:0] input_k_array;
		input integer scale;
		input integer shift;
		output reg [(((TB_NUM_USER * TB_TOTAL_CONTEXT_LENGTH) * TB_TOTAL_CONTEXT_LENGTH) * 8) - 1:0] o_array;
		integer temp_sum;
		integer round;
		reg signed [31:0] u;
		for (u = 0; u < TB_NUM_USER; u = u + 1)
			begin : sv2v_autoblock_6
				reg signed [31:0] i;
				for (i = 0; i < TB_TOTAL_CONTEXT_LENGTH; i = i + 1)
					begin : sv2v_autoblock_7
						reg signed [31:0] j;
						for (j = 0; j < TB_TOTAL_CONTEXT_LENGTH; j = j + 1)
							begin
								temp_sum = 0;
								round = 0;
								begin : sv2v_autoblock_8
									reg signed [31:0] k;
									for (k = 0; k < (TB_EMBD_SIZE / 8); k = k + 1)
										temp_sum = temp_sum + ($signed(input_q_array[((((u * TB_TOTAL_CONTEXT_LENGTH) + i) * (TB_EMBD_SIZE / 8)) + k) * 8+:8]) * $signed(input_k_array[((((u * TB_TOTAL_CONTEXT_LENGTH) + j) * (TB_EMBD_SIZE / 8)) + k) * 8+:8]));
								end
								temp_sum = temp_sum * scale;
								round = temp_sum[shift - 1];
								temp_sum = temp_sum >>> shift;
								temp_sum = temp_sum + round;
								if (temp_sum > 127)
									temp_sum = 127;
								if (temp_sum < -128)
									temp_sum = -128;
								o_array[((((u * TB_TOTAL_CONTEXT_LENGTH) + i) * TB_TOTAL_CONTEXT_LENGTH) + j) * 8+:8] = temp_sum;
							end
					end
			end
	endtask
	task ATT_PV_soft;
		input reg [(((TB_NUM_USER * TB_TOTAL_CONTEXT_LENGTH) * TB_TOTAL_CONTEXT_LENGTH) * 8) - 1:0] input_p_array;
		input reg [(((TB_NUM_USER * TB_TOTAL_CONTEXT_LENGTH) * (TB_EMBD_SIZE / 8)) * 8) - 1:0] input_v_array;
		input integer scale;
		input integer shift;
		output reg [(((TB_NUM_USER * TB_TOTAL_CONTEXT_LENGTH) * (TB_EMBD_SIZE / 8)) * 8) - 1:0] o_array;
		shortreal exp_max_sum;
		reg signed [31:0] qkt_max;
		shortreal softmax_qk;
		integer temp_sum;
		integer round;
		real test_sum;
		reg signed [31:0] u;
		for (u = 0; u < TB_NUM_USER; u = u + 1)
			begin : sv2v_autoblock_9
				reg signed [31:0] i;
				for (i = 0; i < TB_TOTAL_CONTEXT_LENGTH; i = i + 1)
					begin
						qkt_max = -128;
						exp_max_sum = 0;
						begin : sv2v_autoblock_10
							reg signed [31:0] k;
							for (k = 0; k < TB_TOTAL_CONTEXT_LENGTH; k = k + 1)
								if (i < TB_MAX_CONTEXT_LENGTH) begin
									if (k <= i) begin
										if (qkt_max <= $signed(input_p_array[((((u * TB_TOTAL_CONTEXT_LENGTH) + i) * TB_TOTAL_CONTEXT_LENGTH) + k) * 8+:8]))
											qkt_max = $signed(input_p_array[((((u * TB_TOTAL_CONTEXT_LENGTH) + i) * TB_TOTAL_CONTEXT_LENGTH) + k) * 8+:8]);
									end
								end
								else if ((k > (i - TB_MAX_CONTEXT_LENGTH)) && (k <= i)) begin
									if (qkt_max <= $signed(input_p_array[((((u * TB_TOTAL_CONTEXT_LENGTH) + i) * TB_TOTAL_CONTEXT_LENGTH) + k) * 8+:8]))
										qkt_max = $signed(input_p_array[((((u * TB_TOTAL_CONTEXT_LENGTH) + i) * TB_TOTAL_CONTEXT_LENGTH) + k) * 8+:8]);
								end
						end
						begin : sv2v_autoblock_11
							reg signed [31:0] k;
							for (k = 0; k < TB_TOTAL_CONTEXT_LENGTH; k = k + 1)
								if (i < TB_MAX_CONTEXT_LENGTH) begin
									if (k <= i)
										exp_max_sum = exp_max_sum + $exp($itor($signed(input_p_array[((((u * TB_TOTAL_CONTEXT_LENGTH) + i) * TB_TOTAL_CONTEXT_LENGTH) + k) * 8+:8]) - qkt_max) * SOFTMAX_DEQUANT_SCALE);
								end
								else if ((k > (i - TB_MAX_CONTEXT_LENGTH)) && (k <= i))
									exp_max_sum = exp_max_sum + $exp($itor($signed(input_p_array[((((u * TB_TOTAL_CONTEXT_LENGTH) + i) * TB_TOTAL_CONTEXT_LENGTH) + k) * 8+:8]) - qkt_max) * SOFTMAX_DEQUANT_SCALE);
						end
						begin : sv2v_autoblock_12
							reg signed [31:0] j;
							for (j = 0; j < (TB_EMBD_SIZE / 8); j = j + 1)
								begin
									temp_sum = 0;
									round = 0;
									test_sum = 0;
									begin : sv2v_autoblock_13
										reg signed [31:0] k;
										for (k = 0; k < TB_TOTAL_CONTEXT_LENGTH; k = k + 1)
											if (i < TB_MAX_CONTEXT_LENGTH) begin
												if (k <= i) begin
													softmax_qk = $exp($itor($signed(input_p_array[((((u * TB_TOTAL_CONTEXT_LENGTH) + i) * TB_TOTAL_CONTEXT_LENGTH) + k) * 8+:8]) - qkt_max) * SOFTMAX_DEQUANT_SCALE) * SOFTMAX_QUANT_SCALE;
													softmax_qk = zero_round(softmax_qk);
													if (softmax_qk > 127)
														softmax_qk = 127;
													else if (softmax_qk < -128)
														softmax_qk = -128;
													softmax_qk = softmax_qk / exp_max_sum;
													test_sum = test_sum + softmax_qk;
													temp_sum = temp_sum + $rtoi(softmax_qk * $itor($signed(input_v_array[((((u * TB_TOTAL_CONTEXT_LENGTH) + k) * (TB_EMBD_SIZE / 8)) + j) * 8+:8])));
												end
												else
													temp_sum = temp_sum;
											end
											else if ((k > (i - TB_MAX_CONTEXT_LENGTH)) && (k <= i)) begin
												softmax_qk = $exp($itor($signed(input_p_array[((((u * TB_TOTAL_CONTEXT_LENGTH) + i) * TB_TOTAL_CONTEXT_LENGTH) + k) * 8+:8]) - qkt_max) * SOFTMAX_DEQUANT_SCALE) * SOFTMAX_QUANT_SCALE;
												softmax_qk = zero_round(softmax_qk);
												if (softmax_qk > 127)
													softmax_qk = 127;
												else if (softmax_qk < -128)
													softmax_qk = -128;
												softmax_qk = softmax_qk / exp_max_sum;
												temp_sum = temp_sum + $rtoi(softmax_qk * $itor($signed(input_v_array[((((u * TB_TOTAL_CONTEXT_LENGTH) + k) * (TB_EMBD_SIZE / 8)) + j) * 8+:8])));
											end
											else
												temp_sum = temp_sum;
									end
									temp_sum = temp_sum * scale;
									round = temp_sum[shift - 1];
									temp_sum = temp_sum >>> shift;
									temp_sum = temp_sum + round;
									if (temp_sum > 127)
										temp_sum = 127;
									if (temp_sum < -128)
										temp_sum = -128;
									o_array[((((u * TB_TOTAL_CONTEXT_LENGTH) + i) * (TB_EMBD_SIZE / 8)) + j) * 8+:8] = temp_sum;
								end
						end
					end
			end
	endtask
	task PROJ_gen_soft;
		input reg [((((8 * TB_NUM_USER) * TB_TOTAL_CONTEXT_LENGTH) * (TB_EMBD_SIZE / 8)) * 8) - 1:0] ATT_PV_array;
		input reg [(((8 * (TB_EMBD_SIZE / 8)) * TB_EMBD_SIZE) * 8) - 1:0] PROJ_weights_array;
		input integer scale;
		input integer shift;
		output reg [(((TB_NUM_USER * TB_TOTAL_CONTEXT_LENGTH) * TB_EMBD_SIZE) * 8) - 1:0] ATT_PROJ_array;
		reg signed [63:0] temp_sum;
		integer round;
		reg signed [31:0] h;
		reg signed [31:0] l;
		reg signed [31:0] u;
		for (u = 0; u < TB_NUM_USER; u = u + 1)
			begin : sv2v_autoblock_14
				reg signed [31:0] i;
				for (i = 0; i < TB_TOTAL_CONTEXT_LENGTH; i = i + 1)
					begin : sv2v_autoblock_15
						reg signed [31:0] j;
						for (j = 0; j < TB_EMBD_SIZE; j = j + 1)
							begin
								temp_sum = 0;
								round = 0;
								begin : sv2v_autoblock_16
									reg signed [31:0] k;
									for (k = 0; k < TB_EMBD_SIZE; k = k + 1)
										begin
											h = k / (TB_EMBD_SIZE / 8);
											l = k % (TB_EMBD_SIZE / 8);
											temp_sum = temp_sum + ($signed(ATT_PV_array[8 * ((((((h * TB_NUM_USER) + u) * TB_TOTAL_CONTEXT_LENGTH) + i) * (TB_EMBD_SIZE / 8)) + (l * 8))+:64]) * $signed(PROJ_weights_array[((((h * (TB_EMBD_SIZE / 8)) + l) * TB_EMBD_SIZE) + j) * 8+:8]));
										end
								end
								temp_sum = temp_sum * scale;
								round = temp_sum[shift - 1];
								temp_sum = temp_sum >>> shift;
								temp_sum = temp_sum + round;
								if (temp_sum > 127)
									temp_sum = 127;
								if (temp_sum < -128)
									temp_sum = -128;
								ATT_PROJ_array[((((u * TB_TOTAL_CONTEXT_LENGTH) + i) * TB_EMBD_SIZE) + j) * 8+:8] = temp_sum;
							end
					end
			end
	endtask
	task RESIDUAL_gen_soft;
		input reg [(((TB_NUM_USER * TB_TOTAL_CONTEXT_LENGTH) * TB_EMBD_SIZE) * 8) - 1:0] input_x_array;
		input reg [(((TB_NUM_USER * TB_TOTAL_CONTEXT_LENGTH) * TB_EMBD_SIZE) * 8) - 1:0] ATT_PROJ_array;
		input integer scale_A;
		input integer scale_B;
		input integer shift;
		output reg [(((TB_NUM_USER * TB_TOTAL_CONTEXT_LENGTH) * TB_EMBD_SIZE) * 8) - 1:0] ATT_RESIDUAL_array;
		reg signed [63:0] temp;
		integer round;
		reg signed [31:0] h;
		reg signed [31:0] l;
		reg signed [31:0] u;
		for (u = 0; u < TB_NUM_USER; u = u + 1)
			begin : sv2v_autoblock_17
				reg signed [31:0] i;
				for (i = 0; i < TB_TOTAL_CONTEXT_LENGTH; i = i + 1)
					begin : sv2v_autoblock_18
						reg signed [31:0] j;
						for (j = 0; j < TB_EMBD_SIZE; j = j + 1)
							begin
								temp = ($signed(input_x_array[((((u * TB_TOTAL_CONTEXT_LENGTH) + i) * TB_EMBD_SIZE) + j) * 8+:8]) * scale_A) + ($signed(ATT_PROJ_array[((((u * TB_TOTAL_CONTEXT_LENGTH) + i) * TB_EMBD_SIZE) + j) * 8+:8]) * scale_B);
								round = temp[shift - 1];
								temp = temp >>> shift;
								temp = temp + round;
								if (temp > 127)
									temp = 127;
								if (temp < -128)
									temp = -128;
								ATT_RESIDUAL_array[((((u * TB_TOTAL_CONTEXT_LENGTH) + i) * TB_EMBD_SIZE) + j) * 8+:8] = temp;
							end
					end
			end
	endtask
	task FFN0_SOFT;
		input reg [(((TB_NUM_USER * TB_TOTAL_CONTEXT_LENGTH) * TB_EMBD_SIZE) * 8) - 1:0] ATT_RESIDUAL_array;
		input reg [((TB_EMBD_SIZE * ((4 * TB_EMBD_SIZE) / 8)) * 8) - 1:0] FFN0_weights_array;
		input integer scale;
		input integer shift;
		output reg [(((TB_NUM_USER * TB_TOTAL_CONTEXT_LENGTH) * ((4 * TB_EMBD_SIZE) / 8)) * 8) - 1:0] FFN0_array;
		reg signed [63:0] temp_sum;
		integer round;
		shortreal real_data_array [TB_EMBD_SIZE - 1:0];
		shortreal square_sum;
		shortreal rms;
		shortreal one_over_krms;
		reg signed [31:0] u;
		for (u = 0; u < TB_NUM_USER; u = u + 1)
			begin : sv2v_autoblock_19
				reg signed [31:0] i;
				for (i = 0; i < TB_TOTAL_CONTEXT_LENGTH; i = i + 1)
					begin
						square_sum = 0;
						rms = 0;
						begin : sv2v_autoblock_20
							reg signed [31:0] j;
							for (j = 0; j < TB_EMBD_SIZE; j = j + 1)
								real_data_array[j] = $itor($signed(ATT_RESIDUAL_array[((((u * TB_TOTAL_CONTEXT_LENGTH) + i) * TB_EMBD_SIZE) + j) * 8+:8]));
						end
						begin : sv2v_autoblock_21
							reg signed [31:0] j;
							for (j = 0; j < RMS_K; j = j + 1)
								square_sum = square_sum + (real_data_array[j] * real_data_array[j]);
						end
						square_sum = square_sum * mlp_rms_dequant_scale_square;
						rms = $sqrt((square_sum / RMS_K) * 1.0);
						one_over_krms = 1 / rms;
						begin : sv2v_autoblock_22
							reg signed [31:0] j;
							for (j = 0; j < ((4 * TB_EMBD_SIZE) / 8); j = j + 1)
								begin
									temp_sum = 0;
									round = 0;
									begin : sv2v_autoblock_23
										reg signed [31:0] k;
										for (k = 0; k < TB_EMBD_SIZE; k = k + 1)
											temp_sum = temp_sum + ($signed(ATT_RESIDUAL_array[((((u * TB_TOTAL_CONTEXT_LENGTH) + i) * TB_EMBD_SIZE) + k) * 8+:8]) * $signed(FFN0_weights_array[((k * ((4 * TB_EMBD_SIZE) / 8)) + j) * 8+:8]));
									end
									temp_sum = $rtoi($itor(temp_sum) * one_over_krms);
									temp_sum = temp_sum * scale;
									round = temp_sum[shift - 1];
									temp_sum = temp_sum >>> shift;
									temp_sum = temp_sum + round;
									if (temp_sum > 127)
										temp_sum = 127;
									if (temp_sum < -128)
										temp_sum = -128;
									if (temp_sum < 0)
										temp_sum = 0;
									FFN0_array[((((u * TB_TOTAL_CONTEXT_LENGTH) + i) * ((4 * TB_EMBD_SIZE) / 8)) + j) * 8+:8] = temp_sum;
								end
						end
					end
			end
	endtask
	task FFN1_SOFT;
		input reg [((((8 * TB_NUM_USER) * TB_TOTAL_CONTEXT_LENGTH) * ((4 * TB_EMBD_SIZE) / 8)) * 8) - 1:0] FFN0_array;
		input reg [(((8 * ((4 * TB_EMBD_SIZE) / 8)) * TB_EMBD_SIZE) * 8) - 1:0] FFN1_weights_array;
		input integer scale;
		input integer shift;
		output reg [(((TB_NUM_USER * TB_TOTAL_CONTEXT_LENGTH) * TB_EMBD_SIZE) * 8) - 1:0] FFN1_array;
		reg signed [63:0] temp_sum;
		integer round;
		reg signed [31:0] h;
		reg signed [31:0] l;
		reg signed [31:0] u;
		for (u = 0; u < TB_NUM_USER; u = u + 1)
			begin : sv2v_autoblock_24
				reg signed [31:0] i;
				for (i = 0; i < TB_TOTAL_CONTEXT_LENGTH; i = i + 1)
					begin : sv2v_autoblock_25
						reg signed [31:0] j;
						for (j = 0; j < TB_EMBD_SIZE; j = j + 1)
							begin
								temp_sum = 0;
								round = 0;
								begin : sv2v_autoblock_26
									reg signed [31:0] k;
									for (k = 0; k < (4 * TB_EMBD_SIZE); k = k + 1)
										begin
											h = k / ((4 * TB_EMBD_SIZE) / 8);
											l = k % ((4 * TB_EMBD_SIZE) / 8);
											temp_sum = temp_sum + ($signed(FFN0_array[8 * ((((((h * TB_NUM_USER) + u) * TB_TOTAL_CONTEXT_LENGTH) + i) * ((4 * TB_EMBD_SIZE) / 8)) + (l * 8))+:64]) * $signed(FFN1_weights_array[((((h * ((4 * TB_EMBD_SIZE) / 8)) + l) * TB_EMBD_SIZE) + j) * 8+:8]));
										end
								end
								temp_sum = temp_sum * scale;
								round = temp_sum[shift - 1];
								temp_sum = temp_sum >>> shift;
								temp_sum = temp_sum + round;
								if (temp_sum > 127)
									temp_sum = 127;
								if (temp_sum < -128)
									temp_sum = -128;
								FFN1_array[((((u * TB_TOTAL_CONTEXT_LENGTH) + i) * TB_EMBD_SIZE) + j) * 8+:8] = temp_sum;
							end
					end
			end
	endtask
	reg [(((TB_NUM_USER * TB_TOTAL_CONTEXT_LENGTH) * TB_EMBD_SIZE) * 8) - 1:0] input_x_array;
	reg [((TB_EMBD_SIZE * (TB_EMBD_SIZE / 8)) * 8) - 1:0] Q_weights_array [7:0];
	reg [((TB_EMBD_SIZE * (TB_EMBD_SIZE / 8)) * 8) - 1:0] K_weights_array [7:0];
	reg [((TB_EMBD_SIZE * (TB_EMBD_SIZE / 8)) * 8) - 1:0] V_weights_array [7:0];
	reg [(((8 * (TB_EMBD_SIZE / 8)) * TB_EMBD_SIZE) * 8) - 1:0] PROJ_weights_array;
	reg [((TB_EMBD_SIZE * ((4 * TB_EMBD_SIZE) / 8)) * 8) - 1:0] FFN0_weights_array [7:0];
	reg [(((8 * ((4 * TB_EMBD_SIZE) / 8)) * TB_EMBD_SIZE) * 8) - 1:0] FFN1_weights_array;
	wire [(((TB_NUM_USER * TB_TOTAL_CONTEXT_LENGTH) * (TB_EMBD_SIZE / 8)) * 8) - 1:0] K_array [7:0];
	wire [(((TB_NUM_USER * TB_TOTAL_CONTEXT_LENGTH) * (TB_EMBD_SIZE / 8)) * 8) - 1:0] Q_array [7:0];
	wire [(((TB_NUM_USER * TB_TOTAL_CONTEXT_LENGTH) * (TB_EMBD_SIZE / 8)) * 8) - 1:0] V_array [7:0];
	wire [(((TB_NUM_USER * TB_TOTAL_CONTEXT_LENGTH) * TB_TOTAL_CONTEXT_LENGTH) * 8) - 1:0] ATT_QK_array [7:0];
	wire [((((8 * TB_NUM_USER) * TB_TOTAL_CONTEXT_LENGTH) * (TB_EMBD_SIZE / 8)) * 8) - 1:0] ATT_PV_array;
	reg [(((TB_NUM_USER * TB_TOTAL_CONTEXT_LENGTH) * TB_EMBD_SIZE) * 8) - 1:0] ATT_PROJ_array;
	reg [(((TB_NUM_USER * TB_TOTAL_CONTEXT_LENGTH) * TB_EMBD_SIZE) * 8) - 1:0] ATT_RESIDUAL_array;
	wire [((((8 * TB_NUM_USER) * TB_TOTAL_CONTEXT_LENGTH) * ((4 * TB_EMBD_SIZE) / 8)) * 8) - 1:0] FFN0_array;
	reg [(((TB_NUM_USER * TB_TOTAL_CONTEXT_LENGTH) * TB_EMBD_SIZE) * 8) - 1:0] FFN1_array;
	reg [(((TB_NUM_USER * TB_TOTAL_CONTEXT_LENGTH) * TB_EMBD_SIZE) * 8) - 1:0] MLP_RESIDUAL_array;
	reg signed [31:0] error_flag = 0;
	reg signed [7:0] max_value;
	reg signed [7:0] min_value;
	initial begin
		$readmemh("../comp/array/tb/py_head_tb_hex_gen/input_x.hex", input_x_array);
		$readmemh("../comp/array/tb/py_head_tb_hex_gen/Q_weights.hex", Q_weights_array);
		$readmemh("../comp/array/tb/py_head_tb_hex_gen/K_weights.hex", K_weights_array);
		$readmemh("../comp/array/tb/py_head_tb_hex_gen/V_weights.hex", V_weights_array);
		$readmemh("../comp/array/tb/py_head_tb_hex_gen/PROJ_weights.hex", PROJ_weights_array);
		$readmemh("../comp/array/tb/py_head_tb_hex_gen/FFN0_weights.hex", FFN0_weights_array);
		$readmemh("../comp/array/tb/py_head_tb_hex_gen/FFN1_weights.hex", FFN1_weights_array);
	end
	reg [25165823:0] core_wmem_array;
	initial core_wmem_array = 0;
	reg [4095:0] global_mem_array;
	initial global_mem_array = 0;
	reg signed [31:0] WMEM_INIT_FLAG = 0;
	reg [127:0] sram_temp_var;
	integer temp_x;
	integer temp_y;
	genvar _gv_h_1;
	genvar _gv_i_1;
	genvar _gv_k_1;
	generate
		for (_gv_h_1 = 0; _gv_h_1 < 8; _gv_h_1 = _gv_h_1 + 1) begin : genblk1
			localparam h = _gv_h_1;
			for (_gv_i_1 = 0; _gv_i_1 < 16; _gv_i_1 = _gv_i_1 + 1) begin : genblk1
				localparam i = _gv_i_1;
				initial begin
					#(1)
						;
					$display("HEAD[%0d].CORE[%0d] CORE WEIGHT SRAM INIT", h, i);
					begin : sv2v_autoblock_27
						reg signed [31:0] j;
						for (j = 0; j < ((((TB_EMBD_SIZE * TB_EMBD_SIZE) / 8) / 16) / 16); j = j + 1)
							begin
								begin : sv2v_autoblock_28
									reg signed [31:0] k;
									for (k = 0; k < 16; k = k + 1)
										begin
											temp_y = (i * TB_QKV_WEIGHT_COLS_PER_CORE) + (j / (TB_EMBD_SIZE / 16));
											temp_x = ((j % (TB_EMBD_SIZE / 16)) * 16) + k;
											sram_temp_var[k * 8+:8] = Q_weights_array[h][((temp_x * (TB_EMBD_SIZE / 8)) + temp_y) * 8+:8];
										end
								end
								core_wmem_array[((((h * 16) + i) * 1536) + j) * 128+:128] = sram_temp_var;
							end
					end
					begin : sv2v_autoblock_29
						reg signed [31:0] j;
						for (j = 0; j < ((((TB_EMBD_SIZE * TB_EMBD_SIZE) / 8) / 16) / 16); j = j + 1)
							begin
								begin : sv2v_autoblock_30
									reg signed [31:0] k;
									for (k = 0; k < 16; k = k + 1)
										begin
											temp_y = (i * TB_QKV_WEIGHT_COLS_PER_CORE) + (j / (TB_EMBD_SIZE / 16));
											temp_x = ((j % (TB_EMBD_SIZE / 16)) * 16) + k;
											sram_temp_var[k * 8+:8] = K_weights_array[h][((temp_x * (TB_EMBD_SIZE / 8)) + temp_y) * 8+:8];
										end
								end
								core_wmem_array[((((h * 16) + i) * 1536) + (j + K_WEIGHT_ADDR_BASE)) * 128+:128] = sram_temp_var;
							end
					end
					begin : sv2v_autoblock_31
						reg signed [31:0] j;
						for (j = 0; j < ((((TB_EMBD_SIZE * TB_EMBD_SIZE) / 8) / 16) / 16); j = j + 1)
							begin
								begin : sv2v_autoblock_32
									reg signed [31:0] k;
									for (k = 0; k < 16; k = k + 1)
										begin
											temp_y = (i * TB_QKV_WEIGHT_COLS_PER_CORE) + (j / (TB_EMBD_SIZE / 16));
											temp_x = ((j % (TB_EMBD_SIZE / 16)) * 16) + k;
											sram_temp_var[k * 8+:8] = V_weights_array[h][((temp_x * (TB_EMBD_SIZE / 8)) + temp_y) * 8+:8];
										end
								end
								core_wmem_array[((((h * 16) + i) * 1536) + (j + V_WEIGHT_ADDR_BASE)) * 128+:128] = sram_temp_var;
							end
					end
					begin : sv2v_autoblock_33
						reg signed [31:0] j;
						for (j = 0; j < ((((TB_EMBD_SIZE * TB_EMBD_SIZE) / 8) / 16) / 16); j = j + 1)
							begin
								begin : sv2v_autoblock_34
									reg signed [31:0] k;
									for (k = 0; k < 16; k = k + 1)
										begin
											temp_y = (i * (TB_EMBD_SIZE / 16)) + (j / ((TB_EMBD_SIZE / 8) / 16));
											temp_x = ((j % ((TB_EMBD_SIZE / 8) / 16)) * 16) + k;
											sram_temp_var[k * 8+:8] = PROJ_weights_array[((((h * (TB_EMBD_SIZE / 8)) + temp_x) * TB_EMBD_SIZE) + temp_y) * 8+:8];
										end
								end
								core_wmem_array[((((h * 16) + i) * 1536) + (j + PROJ_WEIGHT_ADDR_BASE)) * 128+:128] = sram_temp_var;
							end
					end
					begin : sv2v_autoblock_35
						reg signed [31:0] j;
						for (j = 0; j < (((((4 * TB_EMBD_SIZE) * TB_EMBD_SIZE) / 8) / 16) / 16); j = j + 1)
							begin
								begin : sv2v_autoblock_36
									reg signed [31:0] k;
									for (k = 0; k < 16; k = k + 1)
										begin
											temp_y = (i * (TB_QKV_WEIGHT_COLS_PER_CORE * 4)) + (j / (TB_EMBD_SIZE / 16));
											temp_x = ((j % (TB_EMBD_SIZE / 16)) * 16) + k;
											sram_temp_var[k * 8+:8] = FFN0_weights_array[h][((temp_x * ((4 * TB_EMBD_SIZE) / 8)) + temp_y) * 8+:8];
										end
								end
								core_wmem_array[((((h * 16) + i) * 1536) + (j + FFN0_WEIGHT_ADDR_BASE)) * 128+:128] = sram_temp_var;
							end
					end
					begin : sv2v_autoblock_37
						reg signed [31:0] j;
						for (j = 0; j < (((((4 * TB_EMBD_SIZE) * TB_EMBD_SIZE) / 8) / 16) / 16); j = j + 1)
							begin
								begin : sv2v_autoblock_38
									reg signed [31:0] k;
									for (k = 0; k < 16; k = k + 1)
										begin
											temp_y = (i * (TB_EMBD_SIZE / 16)) + (j / (((4 * TB_EMBD_SIZE) / 8) / 16));
											temp_x = ((j % (((4 * TB_EMBD_SIZE) / 8) / 16)) * 16) + k;
											sram_temp_var[k * 8+:8] = FFN1_weights_array[((((h * ((4 * TB_EMBD_SIZE) / 8)) + temp_x) * TB_EMBD_SIZE) + temp_y) * 8+:8];
										end
								end
								core_wmem_array[((((h * 16) + i) * 1536) + (j + FFN1_WEIGHT_ADDR_BASE)) * 128+:128] = sram_temp_var;
							end
					end
					WMEM_INIT_FLAG = 1;
				end
			end
		end
	endgenerate
	initial begin
		#(10)
			;
		begin : sv2v_autoblock_39
			reg signed [31:0] h;
			for (h = 0; h < 8; h = h + 1)
				begin
					$display("0");
					QKV_gen_soft(input_x_array, Q_weights_array[h], Q_GEN_SCALE, Q_GEN_SHIFT, Q_array[h]);
					$display("1");
					QKV_gen_soft(input_x_array, K_weights_array[h], K_GEN_SCALE, K_GEN_SHIFT, K_array[h]);
					$display("2");
					QKV_gen_soft(input_x_array, V_weights_array[h], V_GEN_SCALE, V_GEN_SHIFT, V_array[h]);
					$display("3");
					ATT_QK_soft(Q_array[h], K_array[h], QKT_SCALE, QKT_SHIFT, ATT_QK_array[h]);
					$display("4");
					ATT_PV_soft(ATT_QK_array[h], V_array[h], PV_SCALE, PV_SHIFT, ATT_PV_array[8 * ((TB_EMBD_SIZE / 8) * (TB_TOTAL_CONTEXT_LENGTH * (h * TB_NUM_USER)))+:8 * ((TB_EMBD_SIZE / 8) * (TB_TOTAL_CONTEXT_LENGTH * TB_NUM_USER))]);
					$display("5");
					max_value = $signed(Q_array[h][0+:8]);
					min_value = max_value;
					begin : sv2v_autoblock_40
						reg signed [31:0] i;
						for (i = 0; i < TB_NUM_USER; i = i + 1)
							begin : sv2v_autoblock_41
								reg signed [31:0] j;
								for (j = 0; j < TB_TOTAL_CONTEXT_LENGTH; j = j + 1)
									begin : sv2v_autoblock_42
										reg signed [31:0] k;
										for (k = 0; k < (TB_EMBD_SIZE / 8); k = k + 1)
											begin
												if ($signed(Q_array[h][((((i * TB_TOTAL_CONTEXT_LENGTH) + j) * (TB_EMBD_SIZE / 8)) + k) * 8+:8]) > $signed(max_value))
													max_value = $signed(Q_array[h][((((i * TB_TOTAL_CONTEXT_LENGTH) + j) * (TB_EMBD_SIZE / 8)) + k) * 8+:8]);
												if ($signed(Q_array[h][((((i * TB_TOTAL_CONTEXT_LENGTH) + j) * (TB_EMBD_SIZE / 8)) + k) * 8+:8]) < $signed(min_value))
													min_value = $signed(Q_array[h][((((i * TB_TOTAL_CONTEXT_LENGTH) + j) * (TB_EMBD_SIZE / 8)) + k) * 8+:8]);
											end
									end
							end
					end
					$display("Max value in Q: %0d", $signed(max_value));
					$display("Min value in Q: %0d", $signed(min_value));
					max_value = $signed(K_array[h][0+:8]);
					min_value = max_value;
					begin : sv2v_autoblock_43
						reg signed [31:0] i;
						for (i = 0; i < TB_NUM_USER; i = i + 1)
							begin : sv2v_autoblock_44
								reg signed [31:0] j;
								for (j = 0; j < TB_TOTAL_CONTEXT_LENGTH; j = j + 1)
									begin : sv2v_autoblock_45
										reg signed [31:0] k;
										for (k = 0; k < (TB_EMBD_SIZE / 8); k = k + 1)
											begin
												if ($signed(K_array[h][((((i * TB_TOTAL_CONTEXT_LENGTH) + j) * (TB_EMBD_SIZE / 8)) + k) * 8+:8]) > $signed(max_value))
													max_value = $signed(K_array[h][((((i * TB_TOTAL_CONTEXT_LENGTH) + j) * (TB_EMBD_SIZE / 8)) + k) * 8+:8]);
												if ($signed(K_array[h][((((i * TB_TOTAL_CONTEXT_LENGTH) + j) * (TB_EMBD_SIZE / 8)) + k) * 8+:8]) < $signed(min_value))
													min_value = $signed(K_array[h][((((i * TB_TOTAL_CONTEXT_LENGTH) + j) * (TB_EMBD_SIZE / 8)) + k) * 8+:8]);
											end
									end
							end
					end
					$display("Max value in K: %0d", $signed(max_value));
					$display("Min value in K: %0d", $signed(min_value));
					max_value = $signed(V_array[h][0+:8]);
					min_value = max_value;
					begin : sv2v_autoblock_46
						reg signed [31:0] i;
						for (i = 0; i < TB_NUM_USER; i = i + 1)
							begin : sv2v_autoblock_47
								reg signed [31:0] j;
								for (j = 0; j < TB_TOTAL_CONTEXT_LENGTH; j = j + 1)
									begin : sv2v_autoblock_48
										reg signed [31:0] k;
										for (k = 0; k < (TB_EMBD_SIZE / 8); k = k + 1)
											begin
												if ($signed(V_array[h][((((i * TB_TOTAL_CONTEXT_LENGTH) + j) * (TB_EMBD_SIZE / 8)) + k) * 8+:8]) > $signed(max_value))
													max_value = $signed(V_array[h][((((i * TB_TOTAL_CONTEXT_LENGTH) + j) * (TB_EMBD_SIZE / 8)) + k) * 8+:8]);
												if ($signed(V_array[h][((((i * TB_TOTAL_CONTEXT_LENGTH) + j) * (TB_EMBD_SIZE / 8)) + k) * 8+:8]) < $signed(min_value))
													min_value = $signed(V_array[h][((((i * TB_TOTAL_CONTEXT_LENGTH) + j) * (TB_EMBD_SIZE / 8)) + k) * 8+:8]);
											end
									end
							end
					end
					$display("Max value in V: %0d", $signed(max_value));
					$display("Min value in V: %0d", $signed(min_value));
					max_value = $signed(ATT_QK_array[h][0+:8]);
					min_value = max_value;
					begin : sv2v_autoblock_49
						reg signed [31:0] i;
						for (i = 0; i < TB_NUM_USER; i = i + 1)
							begin : sv2v_autoblock_50
								reg signed [31:0] j;
								for (j = 0; j < TB_TOTAL_CONTEXT_LENGTH; j = j + 1)
									begin : sv2v_autoblock_51
										reg signed [31:0] k;
										for (k = 0; k < TB_TOTAL_CONTEXT_LENGTH; k = k + 1)
											begin
												if ($signed(ATT_QK_array[h][((((i * TB_TOTAL_CONTEXT_LENGTH) + j) * TB_TOTAL_CONTEXT_LENGTH) + k) * 8+:8]) > $signed(max_value))
													max_value = $signed(ATT_QK_array[h][((((i * TB_TOTAL_CONTEXT_LENGTH) + j) * TB_TOTAL_CONTEXT_LENGTH) + k) * 8+:8]);
												if ($signed(ATT_QK_array[h][((((i * TB_TOTAL_CONTEXT_LENGTH) + j) * TB_TOTAL_CONTEXT_LENGTH) + k) * 8+:8]) < $signed(min_value))
													min_value = $signed(ATT_QK_array[h][((((i * TB_TOTAL_CONTEXT_LENGTH) + j) * TB_TOTAL_CONTEXT_LENGTH) + k) * 8+:8]);
											end
									end
							end
					end
					$display("Max value in QK: %0d", $signed(max_value));
					$display("Min value in QK: %0d", $signed(min_value));
					max_value = $signed(ATT_PV_array[8 * (((h * TB_NUM_USER) * TB_TOTAL_CONTEXT_LENGTH) * (TB_EMBD_SIZE / 8))+:64]);
					min_value = max_value;
					begin : sv2v_autoblock_52
						reg signed [31:0] i;
						for (i = 0; i < TB_NUM_USER; i = i + 1)
							begin : sv2v_autoblock_53
								reg signed [31:0] j;
								for (j = 0; j < TB_TOTAL_CONTEXT_LENGTH; j = j + 1)
									begin : sv2v_autoblock_54
										reg signed [31:0] k;
										for (k = 0; k < (TB_EMBD_SIZE / 8); k = k + 1)
											begin
												if ($signed(ATT_PV_array[8 * ((((((h * TB_NUM_USER) + i) * TB_TOTAL_CONTEXT_LENGTH) + j) * (TB_EMBD_SIZE / 8)) + (k * 8))+:64]) > $signed(max_value))
													max_value = $signed(ATT_PV_array[8 * ((((((h * TB_NUM_USER) + i) * TB_TOTAL_CONTEXT_LENGTH) + j) * (TB_EMBD_SIZE / 8)) + (k * 8))+:64]);
												if ($signed(ATT_PV_array[8 * ((((((h * TB_NUM_USER) + i) * TB_TOTAL_CONTEXT_LENGTH) + j) * (TB_EMBD_SIZE / 8)) + (k * 8))+:64]) < $signed(min_value))
													min_value = $signed(ATT_PV_array[8 * ((((((h * TB_NUM_USER) + i) * TB_TOTAL_CONTEXT_LENGTH) + j) * (TB_EMBD_SIZE / 8)) + (k * 8))+:64]);
											end
									end
							end
					end
					if (h == 0) begin : sv2v_autoblock_55
						reg signed [31:0] kk;
						for (kk = 0; kk < TB_TOTAL_CONTEXT_LENGTH; kk = kk + 1)
							begin
								$display(kk);
								begin : sv2v_autoblock_56
									reg signed [31:0] jj;
									for (jj = 0; jj < (TB_EMBD_SIZE / 8); jj = jj + 1)
										$write("%5d, ", $signed(ATT_PV_array[8 * (((((h * TB_NUM_USER) * TB_TOTAL_CONTEXT_LENGTH) + kk) * (TB_EMBD_SIZE / 8)) + (jj * 8))+:64]));
								end
							end
					end
					$display("Max value in PV: %0d", $signed(max_value));
					$display("Min value in PV: %0d", $signed(min_value));
				end
		end
		PROJ_gen_soft(ATT_PV_array, PROJ_weights_array, PROJ_SCALE, PROJ_SHIFT, ATT_PROJ_array);
		$display("6");
		max_value = $signed(ATT_PROJ_array[0+:8]);
		min_value = max_value;
		begin : sv2v_autoblock_57
			reg signed [31:0] i;
			for (i = 0; i < TB_NUM_USER; i = i + 1)
				begin : sv2v_autoblock_58
					reg signed [31:0] j;
					for (j = 0; j < TB_TOTAL_CONTEXT_LENGTH; j = j + 1)
						begin : sv2v_autoblock_59
							reg signed [31:0] k;
							for (k = 0; k < TB_EMBD_SIZE; k = k + 1)
								begin
									if ($signed(ATT_PROJ_array[((((i * TB_TOTAL_CONTEXT_LENGTH) + j) * TB_EMBD_SIZE) + k) * 8+:8]) > $signed(max_value))
										max_value = $signed(ATT_PROJ_array[((((i * TB_TOTAL_CONTEXT_LENGTH) + j) * TB_EMBD_SIZE) + k) * 8+:8]);
									if ($signed(ATT_PROJ_array[((((i * TB_TOTAL_CONTEXT_LENGTH) + j) * TB_EMBD_SIZE) + k) * 8+:8]) < $signed(min_value))
										min_value = $signed(ATT_PROJ_array[((((i * TB_TOTAL_CONTEXT_LENGTH) + j) * TB_EMBD_SIZE) + k) * 8+:8]);
								end
						end
				end
		end
		$display("Max value in Proj: %0d", $signed(max_value));
		$display("Min value in Proj: %0d", $signed(min_value));
		RESIDUAL_gen_soft(input_x_array, ATT_PROJ_array, ATTN_RESIDUAL_SCALE_A, ATTN_RESIDUAL_SCALE_B, ATTN_RESIDUAL_SHIFT, ATT_RESIDUAL_array);
		$display("7");
		max_value = $signed(ATT_RESIDUAL_array[0+:8]);
		min_value = max_value;
		begin : sv2v_autoblock_60
			reg signed [31:0] i;
			for (i = 0; i < TB_NUM_USER; i = i + 1)
				begin : sv2v_autoblock_61
					reg signed [31:0] j;
					for (j = 0; j < TB_TOTAL_CONTEXT_LENGTH; j = j + 1)
						begin : sv2v_autoblock_62
							reg signed [31:0] k;
							for (k = 0; k < TB_EMBD_SIZE; k = k + 1)
								begin
									if ($signed(ATT_RESIDUAL_array[((((i * TB_TOTAL_CONTEXT_LENGTH) + j) * TB_EMBD_SIZE) + k) * 8+:8]) > $signed(max_value))
										max_value = $signed(ATT_RESIDUAL_array[((((i * TB_TOTAL_CONTEXT_LENGTH) + j) * TB_EMBD_SIZE) + k) * 8+:8]);
									if ($signed(ATT_RESIDUAL_array[((((i * TB_TOTAL_CONTEXT_LENGTH) + j) * TB_EMBD_SIZE) + k) * 8+:8]) < $signed(min_value))
										min_value = $signed(ATT_RESIDUAL_array[((((i * TB_TOTAL_CONTEXT_LENGTH) + j) * TB_EMBD_SIZE) + k) * 8+:8]);
								end
						end
				end
		end
		$display("Max value in Residual: %0d", $signed(max_value));
		$display("Min value in Residual: %0d", $signed(min_value));
		begin : sv2v_autoblock_63
			reg signed [31:0] h;
			for (h = 0; h < 8; h = h + 1)
				begin
					FFN0_SOFT(ATT_RESIDUAL_array, FFN0_weights_array[h], FFN0_SCALE, FFN0_SHIFT, FFN0_array[8 * (((4 * TB_EMBD_SIZE) / 8) * (TB_TOTAL_CONTEXT_LENGTH * (h * TB_NUM_USER)))+:8 * (((4 * TB_EMBD_SIZE) / 8) * (TB_TOTAL_CONTEXT_LENGTH * TB_NUM_USER))]);
					$display("8");
					max_value = $signed(FFN0_array[8 * (((h * TB_NUM_USER) * TB_TOTAL_CONTEXT_LENGTH) * ((4 * TB_EMBD_SIZE) / 8))+:64]);
					min_value = max_value;
					begin : sv2v_autoblock_64
						reg signed [31:0] i;
						for (i = 0; i < TB_NUM_USER; i = i + 1)
							begin : sv2v_autoblock_65
								reg signed [31:0] j;
								for (j = 0; j < TB_TOTAL_CONTEXT_LENGTH; j = j + 1)
									begin : sv2v_autoblock_66
										reg signed [31:0] k;
										for (k = 0; k < ((4 * TB_EMBD_SIZE) / 8); k = k + 1)
											begin
												if ($signed(FFN0_array[8 * ((((((h * TB_NUM_USER) + i) * TB_TOTAL_CONTEXT_LENGTH) + j) * ((4 * TB_EMBD_SIZE) / 8)) + (k * 8))+:64]) > $signed(max_value))
													max_value = $signed(FFN0_array[8 * ((((((h * TB_NUM_USER) + i) * TB_TOTAL_CONTEXT_LENGTH) + j) * ((4 * TB_EMBD_SIZE) / 8)) + (k * 8))+:64]);
												if ($signed(FFN0_array[8 * ((((((h * TB_NUM_USER) + i) * TB_TOTAL_CONTEXT_LENGTH) + j) * ((4 * TB_EMBD_SIZE) / 8)) + (k * 8))+:64]) < $signed(min_value))
													min_value = $signed(FFN0_array[8 * ((((((h * TB_NUM_USER) + i) * TB_TOTAL_CONTEXT_LENGTH) + j) * ((4 * TB_EMBD_SIZE) / 8)) + (k * 8))+:64]);
											end
									end
							end
					end
					$display("Max value in FFN0: %0d", $signed(max_value));
					$display("Min value in FFN0: %0d", $signed(min_value));
				end
		end
		FFN1_SOFT(FFN0_array, FFN1_weights_array, FFN1_SCALE, FFN1_SHIFT, FFN1_array);
		$display("9");
		max_value = $signed(FFN1_array[0+:8]);
		min_value = max_value;
		begin : sv2v_autoblock_67
			reg signed [31:0] i;
			for (i = 0; i < TB_NUM_USER; i = i + 1)
				begin : sv2v_autoblock_68
					reg signed [31:0] j;
					for (j = 0; j < TB_TOTAL_CONTEXT_LENGTH; j = j + 1)
						begin : sv2v_autoblock_69
							reg signed [31:0] k;
							for (k = 0; k < TB_EMBD_SIZE; k = k + 1)
								begin
									if ($signed(FFN1_array[((((i * TB_TOTAL_CONTEXT_LENGTH) + j) * TB_EMBD_SIZE) + k) * 8+:8]) > $signed(max_value))
										max_value = $signed(FFN1_array[((((i * TB_TOTAL_CONTEXT_LENGTH) + j) * TB_EMBD_SIZE) + k) * 8+:8]);
									if ($signed(FFN1_array[((((i * TB_TOTAL_CONTEXT_LENGTH) + j) * TB_EMBD_SIZE) + k) * 8+:8]) < $signed(min_value))
										min_value = $signed(FFN1_array[((((i * TB_TOTAL_CONTEXT_LENGTH) + j) * TB_EMBD_SIZE) + k) * 8+:8]);
								end
						end
				end
		end
		$display("Max value in FFN1: %0d", $signed(max_value));
		$display("Min value in FFN1: %0d", $signed(min_value));
		RESIDUAL_gen_soft(ATT_RESIDUAL_array, FFN1_array, MLP_RESIDUAL_SCALE_A, MLP_RESIDUAL_SCALE_B, MLP_RESIDUAL_SHIFT, MLP_RESIDUAL_array);
		$display("10");
		max_value = $signed(MLP_RESIDUAL_array[0+:8]);
		min_value = max_value;
		begin : sv2v_autoblock_70
			reg signed [31:0] i;
			for (i = 0; i < TB_NUM_USER; i = i + 1)
				begin : sv2v_autoblock_71
					reg signed [31:0] j;
					for (j = 0; j < TB_TOTAL_CONTEXT_LENGTH; j = j + 1)
						begin : sv2v_autoblock_72
							reg signed [31:0] k;
							for (k = 0; k < TB_EMBD_SIZE; k = k + 1)
								begin
									if ($signed(MLP_RESIDUAL_array[((((i * TB_TOTAL_CONTEXT_LENGTH) + j) * TB_EMBD_SIZE) + k) * 8+:8]) > $signed(max_value))
										max_value = $signed(MLP_RESIDUAL_array[((((i * TB_TOTAL_CONTEXT_LENGTH) + j) * TB_EMBD_SIZE) + k) * 8+:8]);
									if ($signed(MLP_RESIDUAL_array[((((i * TB_TOTAL_CONTEXT_LENGTH) + j) * TB_EMBD_SIZE) + k) * 8+:8]) < $signed(min_value))
										min_value = $signed(MLP_RESIDUAL_array[((((i * TB_TOTAL_CONTEXT_LENGTH) + j) * TB_EMBD_SIZE) + k) * 8+:8]);
								end
						end
				end
		end
		$display("Max value in Residual: %0d", $signed(max_value));
		$display("Min value in Residual: %0d", $signed(min_value));
	end
	reg [1:0] user_id;
	reg new_token;
	reg user_first_token;
	reg chip_clk;
	reg fpga_clk;
	reg asyn_rst;
	wire spi_clk;
	wire spi_csn;
	wire spi_mosi;
	wire spi_miso;
	reg qspi_clk;
	reg [15:0] qspi_mosi;
	reg qspi_mosi_valid;
	wire [15:0] qspi_miso;
	wire qspi_miso_valid;
	reg spi_start;
	wire spi_complete;
	reg [40:0] spi_tx_data;
	wire [15:0] spi_rx_data;
	wire spi_rx_valid;
	wire current_token_finish_flag;
	wire current_token_finish_work;
	wire qgen_state_work;
	wire qgen_state_end;
	wire kgen_state_work;
	wire kgen_state_end;
	wire vgen_state_work;
	wire vgen_state_end;
	wire att_qk_state_work;
	wire att_qk_state_end;
	wire att_pv_state_work;
	wire att_pv_state_end;
	wire proj_state_work;
	wire proj_state_end;
	wire ffn0_state_work;
	wire ffn0_state_end;
	wire ffn1_state_work;
	wire ffn1_state_end;
	reg att_qk_done = 0;
	reg att_pv_done = 0;
	reg proj_done = 0;
	reg attn_residual_done = 0;
	reg ffn0_done = 0;
	reg ffn1_done = 0;
	reg mlp_residual_done = 0;
	reg signed [31:0] iter_cnt = 0;
	reg iter_done;
	integer user_total_token_cnt;
	reg [(TB_NUM_USER * $clog2(TB_TOTAL_CONTEXT_LENGTH + 1)) - 1:0] usr_total_token_cnt_array;
	reg [21:0] addr;
	wire [15:0] wdata;
	wire [15:0] rdata;
	reg [4095:0] wdata_array;
	reg [4095:0] rdata_array;
	reg [4095:0] tpu_state_rdata_array;
	reg [7:0] burst_cnt;
	initial begin
		chip_clk = 0;
		#(100) fpga_clk = 0;
		#(450) qspi_clk = 0;
	end
	always begin
		#((1e12 / CHIP_CLK_FREQ) / 2)
			;
		chip_clk = ~chip_clk;
	end
	always begin
		#((1e12 / FPGA_CLK_FREQ) / 2)
			;
		fpga_clk = ~fpga_clk;
	end
	always begin
		#((1e12 / QSPI_CLK_FREQ) / 2)
			;
		qspi_clk = ~qspi_clk;
	end
	host_spi #(
		.DW(41),
		.TX(22),
		.RX(16)
	) host_spi(
		.clk(fpga_clk),
		.rst(asyn_rst),
		.spi_start(spi_start),
		.spi_complete(spi_complete),
		.spi_tx_data(spi_tx_data),
		.spi_rx_data(spi_rx_data),
		.spi_rx_valid(spi_rx_valid),
		.spi_sck(spi_clk),
		.spi_csn(spi_csn),
		.spi_mosi(spi_mosi),
		.spi_miso(spi_miso)
	);
	top top(
		.chip_clk(chip_clk),
		.asyn_rst(asyn_rst),
		.spi_clk(spi_clk),
		.spi_csn(spi_csn),
		.spi_mosi(spi_mosi),
		.spi_miso(spi_miso),
		.qspi_clk(qspi_clk),
		.qspi_mosi(qspi_mosi),
		.qspi_mosi_valid(qspi_mosi_valid),
		.qspi_miso(qspi_miso),
		.qspi_miso_valid(qspi_miso_valid),
		.current_token_finish_flag(current_token_finish_flag),
		.current_token_finish_work(current_token_finish_work),
		.qgen_state_work(qgen_state_work),
		.qgen_state_end(qgen_state_end),
		.kgen_state_work(kgen_state_work),
		.kgen_state_end(kgen_state_end),
		.vgen_state_work(vgen_state_work),
		.vgen_state_end(vgen_state_end),
		.att_qk_state_work(att_qk_state_work),
		.att_qk_state_end(att_qk_state_end),
		.att_pv_state_work(att_pv_state_work),
		.att_pv_state_end(att_pv_state_end),
		.proj_state_work(proj_state_work),
		.proj_state_end(proj_state_end),
		.ffn0_state_work(ffn0_state_work),
		.ffn0_state_end(ffn0_state_end),
		.ffn1_state_work(ffn1_state_work),
		.ffn1_state_end(ffn1_state_end)
	);
	task FPGA_SPI_WR;
		input reg [21:0] addr;
		input reg [15:0] wdata;
		reg [0:1] _sv2v_jump;
		begin
			_sv2v_jump = 2'b00;
			@(negedge fpga_clk)
				;
			spi_start = 1;
			spi_tx_data = {2'b10, addr, 1'b0, wdata};
			@(negedge fpga_clk)
				;
			spi_start = 0;
			while (!(!(_sv2v_jump < 2'b10))) begin
				_sv2v_jump = 2'b00;
				@(negedge fpga_clk)
					;
				if (spi_complete)
					_sv2v_jump = 2'b10;
			end
			if (_sv2v_jump != 2'b11)
				_sv2v_jump = 2'b00;
		end
	endtask
	task FPGA_SPI_RD;
		input reg [21:0] addr;
		output reg [15:0] rdata;
		reg [0:1] _sv2v_jump;
		begin
			_sv2v_jump = 2'b00;
			@(negedge fpga_clk)
				;
			spi_start = 1;
			spi_tx_data = {2'b01, addr, 17'b00000000000000000};
			@(negedge fpga_clk)
				;
			spi_start = 0;
			while (!(!(_sv2v_jump < 2'b10))) begin
				_sv2v_jump = 2'b00;
				@(negedge fpga_clk)
					;
				if (spi_rx_valid) begin
					rdata = spi_rx_data;
					_sv2v_jump = 2'b10;
				end
			end
			if (_sv2v_jump != 2'b11)
				_sv2v_jump = 2'b00;
		end
	endtask
	task FPGA_QSPI_WR;
		input reg [21:0] addr;
		input reg [7:0] burst_cnt;
		input reg [4095:0] wdata_array;
		begin
			@(posedge qspi_clk)
				;
			qspi_mosi_valid = 1;
			qspi_mosi = {addr[5:0], 2'b10, burst_cnt};
			@(posedge qspi_clk)
				;
			qspi_mosi_valid = 1;
			qspi_mosi = addr[21:6];
			begin : sv2v_autoblock_73
				reg signed [31:0] i;
				for (i = 0; i < (burst_cnt + 1); i = i + 1)
					begin
						@(posedge qspi_clk)
							;
						qspi_mosi_valid = 1;
						qspi_mosi = wdata_array[i * 16+:16];
					end
			end
			@(posedge qspi_clk)
				;
			qspi_mosi_valid = 0;
			@(posedge qspi_clk)
				;
		end
	endtask
	task FPGA_QSPI_RD;
		input reg [21:0] addr;
		input reg [7:0] burst_cnt;
		output reg [4095:0] rdata_array;
		reg signed [31:0] temp;
		reg [0:1] _sv2v_jump;
		begin
			temp = 0;
			_sv2v_jump = 2'b00;
			@(posedge qspi_clk)
				;
			qspi_mosi_valid = 1;
			qspi_mosi = {addr[5:0], 2'b01, burst_cnt};
			@(posedge qspi_clk)
				;
			qspi_mosi_valid = 1;
			qspi_mosi = addr[21:6];
			@(posedge qspi_clk)
				;
			qspi_mosi_valid = 0;
			while (!(!(_sv2v_jump < 2'b10))) begin
				_sv2v_jump = 2'b00;
				@(negedge qspi_clk)
					;
				if (qspi_miso_valid) begin
					rdata_array[temp * 16+:16] = qspi_miso;
					temp = temp + 1;
					if (temp == (burst_cnt + 1))
						_sv2v_jump = 2'b10;
				end
			end
			if (_sv2v_jump != 2'b11)
				_sv2v_jump = 2'b00;
			if (_sv2v_jump == 2'b00)
				@(negedge qspi_clk)
					;
		end
	endtask
	initial begin : sv2v_autoblock_74
		reg [0:1] _sv2v_jump;
		_sv2v_jump = 2'b00;
		usr_total_token_cnt_array = 0;
		asyn_rst = 1;
		spi_start = 0;
		spi_tx_data = 0;
		qspi_mosi_valid = 0;
		qspi_mosi = 0;
		repeat (10000) @(negedge chip_clk)
			;
		asyn_rst = 0;
		repeat (100) @(negedge chip_clk)
			;
		$display("Upload wmem");
		begin : sv2v_autoblock_75
			reg signed [31:0] h;
			for (h = 0; h < 8; h = h + 1)
				begin : sv2v_autoblock_76
					reg signed [31:0] c;
					for (c = 0; c < 16; c = c + 1)
						begin
							begin : sv2v_autoblock_77
								reg signed [31:0] i;
								for (i = 0; i < 48; i = i + 1)
									begin
										addr = (((h * 'h40000) + (c * 'h4000)) + (i * 256)) + 4096;
										burst_cnt = 255;
										begin : sv2v_autoblock_78
											reg signed [31:0] j;
											for (j = 0; j < 256; j = j + 1)
												wdata_array[j * 16+:16] = core_wmem_array[(((((h * 16) + c) * 1536) + (((i * 256) / 8) + (j / 8))) * 128) + ((j % 8) * 16)+:16];
										end
										FPGA_QSPI_WR(addr, burst_cnt, wdata_array);
									end
							end
							repeat (30) @(negedge chip_clk)
								;
							$display("HEAD[%0d]CORE[%0d] WMEM finish.", h, c);
						end
				end
		end
		$display("Clean KV CACHE");
		begin : sv2v_autoblock_79
			reg signed [31:0] i;
			for (i = 0; i < TB_NUM_USER; i = i + 1)
				begin
					addr = (22'h200000 + 22'h001000) + 257;
					burst_cnt = 0;
					user_id = i;
					wdata_array[0+:16] = {13'b0000000000000, user_id, 1'b1};
					FPGA_QSPI_WR(addr, burst_cnt, wdata_array);
					repeat (1000) @(negedge chip_clk)
						;
				end
		end
		$display("Upload control register value");
		burst_cnt = 63;
		addr = (22'h200000 + 22'h001000) + 0;
		wdata_array[0+:16] = TB_MAX_CONTEXT_LENGTH;
		wdata_array[16+:16] = TB_QKV_WEIGHT_COLS_PER_CORE;
		wdata_array[32+:16] = TB_MAX_CONTEXT_LENGTH / 16;
		wdata_array[48+:16] = TB_EMBD_SIZE;
		wdata_array[64+:16] = GQA_EN;
		wdata_array[80+:16] = 1;
		wdata_array[96+:16] = 1;
		wdata_array[112+:16] = {14'b00000000000000, DEEPSLEEP_EN, PMU_CFG_EN};
		wdata_array[128+:16] = RC_SHIFT;
		wdata_array[144+:16] = RMS_K;
		wdata_array[160+:16] = shortreal_to_bitsbfloat16(attn_rms_dequant_scale_square);
		wdata_array[176+:16] = shortreal_to_bitsbfloat16(mlp_rms_dequant_scale_square);
		wdata_array[192+:16] = RC_SHIFT;
		wdata_array[208+:16] = shortreal_to_bitsbfloat16(SOFTMAX_DEQUANT_SCALE);
		wdata_array[224+:16] = shortreal_to_bitsbfloat16(SOFTMAX_QUANT_SCALE);
		wdata_array[240+:16] = TB_EMBD_SIZE / 16;
		wdata_array[256+:16] = Q_GEN_SCALE;
		wdata_array[272+:16] = 0;
		wdata_array[288+:16] = Q_GEN_SHIFT;
		wdata_array[304+:16] = TB_EMBD_SIZE / 16;
		wdata_array[320+:16] = K_GEN_SCALE;
		wdata_array[336+:16] = 0;
		wdata_array[352+:16] = K_GEN_SHIFT;
		wdata_array[368+:16] = TB_EMBD_SIZE / 16;
		wdata_array[384+:16] = V_GEN_SCALE;
		wdata_array[400+:16] = 0;
		wdata_array[416+:16] = V_GEN_SHIFT;
		wdata_array[432+:16] = (TB_EMBD_SIZE / 8) / 16;
		wdata_array[448+:16] = QKT_SCALE;
		wdata_array[464+:16] = 0;
		wdata_array[480+:16] = QKT_SHIFT;
		wdata_array[496+:16] = 1;
		wdata_array[512+:16] = PV_SCALE;
		wdata_array[528+:16] = 0;
		wdata_array[544+:16] = PV_SHIFT;
		wdata_array[560+:16] = (TB_EMBD_SIZE / 8) / 16;
		wdata_array[576+:16] = PROJ_SCALE;
		wdata_array[592+:16] = 0;
		wdata_array[608+:16] = PROJ_SHIFT;
		wdata_array[624+:16] = TB_EMBD_SIZE / 16;
		wdata_array[640+:16] = FFN0_SCALE;
		wdata_array[656+:16] = 0;
		wdata_array[672+:16] = FFN0_SHIFT;
		wdata_array[688+:16] = ((4 * TB_EMBD_SIZE) / 8) / 16;
		wdata_array[704+:16] = FFN1_SCALE;
		wdata_array[720+:16] = 0;
		wdata_array[736+:16] = FFN1_SHIFT;
		wdata_array[752+:16] = ATTN_RESIDUAL_SCALE_A;
		wdata_array[768+:16] = ATTN_RESIDUAL_SCALE_B;
		wdata_array[784+:16] = 0;
		wdata_array[800+:16] = ATTN_RESIDUAL_SHIFT;
		wdata_array[816+:16] = MLP_RESIDUAL_SCALE_A;
		wdata_array[832+:16] = MLP_RESIDUAL_SCALE_B;
		wdata_array[848+:16] = 0;
		wdata_array[864+:16] = MLP_RESIDUAL_SHIFT;
		FPGA_QSPI_WR(addr, burst_cnt, wdata_array);
		burst_cnt = 0;
		addr = (22'h200000 + 22'h001000) + 258;
		wdata_array[0+:16] = 1;
		FPGA_QSPI_WR(addr, burst_cnt, wdata_array);
		begin : sv2v_autoblock_80
			reg signed [31:0] i;
			begin : sv2v_autoblock_81
				reg signed [31:0] _sv2v_value_on_break;
				for (i = 0; i < TB_TOTAL_CONTEXT_LENGTH; i = i + 1)
					if (_sv2v_jump < 2'b10) begin
						_sv2v_jump = 2'b00;
						begin : sv2v_autoblock_82
							reg signed [31:0] j;
							begin : sv2v_autoblock_83
								reg signed [31:0] _sv2v_value_on_break;
								reg [0:1] _sv2v_jump_1;
								_sv2v_jump_1 = _sv2v_jump;
								for (j = 0; j < TB_NUM_USER; j = j + 1)
									if (_sv2v_jump < 2'b10) begin
										_sv2v_jump = 2'b00;
										begin : sv2v_autoblock_84
											reg signed [31:0] m;
											for (m = 0; m < (TB_EMBD_SIZE / 16); m = m + 1)
												begin
													begin : sv2v_autoblock_85
														reg signed [31:0] k;
														for (k = 0; k < 16; k = k + 1)
															sram_temp_var[k * 8+:8] = input_x_array[((((j * TB_TOTAL_CONTEXT_LENGTH) + i) * TB_EMBD_SIZE) + ((m * 16) + k)) * 8+:8];
													end
													global_mem_array[m * 128+:128] = sram_temp_var;
												end
										end
										addr = 22'h200000 + 22'h000800;
										burst_cnt = (TB_EMBD_SIZE / 2) - 1;
										begin : sv2v_autoblock_86
											reg signed [31:0] jj;
											for (jj = 0; jj < 256; jj = jj + 1)
												wdata_array[jj * 16+:16] = global_mem_array[((jj / 8) * 128) + ((jj % 8) * 16)+:16];
										end
										FPGA_QSPI_WR(addr, burst_cnt, wdata_array);
										repeat (10) @(negedge chip_clk)
											;
										new_token = 1;
										user_id = j;
										user_first_token = i == 0;
										burst_cnt = 0;
										addr = (22'h200000 + 22'h001000) + 256;
										wdata_array[0+:16] = {12'b000000000000, user_id, user_first_token, new_token};
										FPGA_QSPI_WR(addr, burst_cnt, wdata_array);
										begin : sv2v_autoblock_87
											reg [0:1] _sv2v_jump_2;
											_sv2v_jump_2 = _sv2v_jump;
											while (!(!(_sv2v_jump < 2'b10))) begin
												_sv2v_jump = 2'b00;
												@(negedge chip_clk)
													;
												if (top.array_top_inst.inst_array_ctrl.control_state == 32'd5)
													_sv2v_jump = 2'b10;
											end
											if (_sv2v_jump != 2'b11)
												_sv2v_jump = _sv2v_jump_2;
										end
										if (_sv2v_jump == 2'b00) begin
											$display("USER: %0d DONE Processing, TOKEN_CNT: %0d, TIME: %t", j, usr_total_token_cnt_array[j * $clog2(TB_TOTAL_CONTEXT_LENGTH + 1)+:$clog2(TB_TOTAL_CONTEXT_LENGTH + 1)], $time);
											$display("Iter cnt: %d", iter_cnt);
											$display("QK finish");
											att_qk_done = 1;
											@(negedge chip_clk)
												;
											att_qk_done = 0;
											burst_cnt = 0;
											addr = (22'h200000 + 22'h001000) + 513;
											FPGA_QSPI_RD(addr, burst_cnt, tpu_state_rdata_array);
											$display("State reg[1] = %d", tpu_state_rdata_array[0+:16]);
											begin : sv2v_autoblock_88
												reg [0:1] _sv2v_jump_2;
												_sv2v_jump_2 = _sv2v_jump;
												while (!(!(_sv2v_jump < 2'b10))) begin
													_sv2v_jump = 2'b00;
													@(negedge chip_clk)
														;
													if (top.array_top_inst.inst_array_ctrl.control_state == 32'd6)
														_sv2v_jump = 2'b10;
												end
												if (_sv2v_jump != 2'b11)
													_sv2v_jump = _sv2v_jump_2;
											end
											if (_sv2v_jump == 2'b00) begin
												$display("USER: %0d DONE Processing, TOKEN_CNT: %0d, TIME: %t", j, usr_total_token_cnt_array[j * $clog2(TB_TOTAL_CONTEXT_LENGTH + 1)+:$clog2(TB_TOTAL_CONTEXT_LENGTH + 1)], $time);
												$display("Iter cnt: %d", iter_cnt);
												$display("PV finish");
												att_pv_done = 1;
												@(negedge chip_clk)
													;
												att_pv_done = 0;
												burst_cnt = 0;
												addr = (22'h200000 + 22'h001000) + 513;
												FPGA_QSPI_RD(addr, burst_cnt, tpu_state_rdata_array);
												$display("State reg[1] = %d", tpu_state_rdata_array[0+:16]);
												begin : sv2v_autoblock_89
													reg [0:1] _sv2v_jump_2;
													_sv2v_jump_2 = _sv2v_jump;
													while (!(!(_sv2v_jump < 2'b10))) begin
														_sv2v_jump = 2'b00;
														@(negedge chip_clk)
															;
														if (top.array_top_inst.inst_array_ctrl.control_state == 32'd7)
															_sv2v_jump = 2'b10;
													end
													if (_sv2v_jump != 2'b11)
														_sv2v_jump = _sv2v_jump_2;
												end
												if (_sv2v_jump == 2'b00) begin
													$display("USER: %0d DONE Processing, TOKEN_CNT: %0d, TIME: %t", j, usr_total_token_cnt_array[j * $clog2(TB_TOTAL_CONTEXT_LENGTH + 1)+:$clog2(TB_TOTAL_CONTEXT_LENGTH + 1)], $time);
													$display("Iter cnt: %d", iter_cnt);
													$display("Proj finish");
													proj_done = 1;
													@(negedge chip_clk)
														;
													proj_done = 0;
													burst_cnt = 0;
													addr = (22'h200000 + 22'h001000) + 513;
													FPGA_QSPI_RD(addr, burst_cnt, tpu_state_rdata_array);
													$display("State reg[1] = %d", tpu_state_rdata_array[0+:16]);
													begin : sv2v_autoblock_90
														reg [0:1] _sv2v_jump_2;
														_sv2v_jump_2 = _sv2v_jump;
														while (!(!(_sv2v_jump < 2'b10))) begin
															_sv2v_jump = 2'b00;
															@(negedge chip_clk)
																;
															if (top.array_top_inst.inst_array_ctrl.control_state == 32'd8)
																_sv2v_jump = 2'b10;
														end
														if (_sv2v_jump != 2'b11)
															_sv2v_jump = _sv2v_jump_2;
													end
													if (_sv2v_jump == 2'b00) begin
														$display("USER: %0d DONE Processing, TOKEN_CNT: %0d, TIME: %t", j, usr_total_token_cnt_array[j * $clog2(TB_TOTAL_CONTEXT_LENGTH + 1)+:$clog2(TB_TOTAL_CONTEXT_LENGTH + 1)], $time);
														$display("Iter cnt: %d", iter_cnt);
														$display("FFN0 finish");
														ffn0_done = 1;
														@(negedge chip_clk)
															;
														ffn0_done = 0;
														burst_cnt = 0;
														addr = (22'h200000 + 22'h001000) + 513;
														FPGA_QSPI_RD(addr, burst_cnt, tpu_state_rdata_array);
														$display("State reg[1] = %d", tpu_state_rdata_array[0+:16]);
														begin : sv2v_autoblock_91
															reg [0:1] _sv2v_jump_2;
															_sv2v_jump_2 = _sv2v_jump;
															while (!(!(_sv2v_jump < 2'b10))) begin
																_sv2v_jump = 2'b00;
																burst_cnt = 0;
																addr = (22'h200000 + 22'h001000) + 512;
																FPGA_QSPI_RD(addr, burst_cnt, rdata_array);
																if (rdata_array[0+:16] == 1)
																	_sv2v_jump = 2'b10;
															end
															if (_sv2v_jump != 2'b11)
																_sv2v_jump = _sv2v_jump_2;
														end
														if (_sv2v_jump == 2'b00) begin
															$display("USER: %0d DONE Processing, TOKEN_CNT: %0d, TIME: %t", j, usr_total_token_cnt_array[j * $clog2(TB_TOTAL_CONTEXT_LENGTH + 1)+:$clog2(TB_TOTAL_CONTEXT_LENGTH + 1)], $time);
															$display("Iter cnt: %d", iter_cnt);
															$display("FFN1 finish");
															begin : sv2v_autoblock_92
																reg signed [31:0] ii;
																for (ii = 0; ii < ((TB_EMBD_SIZE / 2) / 64); ii = ii + 1)
																	begin
																		addr = (22'h200000 + 22'h000900) + (64 * ii);
																		burst_cnt = 63;
																		FPGA_QSPI_RD(addr, burst_cnt, rdata_array);
																		begin : sv2v_autoblock_93
																			reg signed [31:0] jj;
																			for (jj = 0; jj < 64; jj = jj + 1)
																				global_mem_array[((((ii * 64) + jj) / 8) * 128) + ((jj % 8) * 16)+:16] = rdata_array[jj * 16+:16];
																		end
																	end
															end
															ffn1_done = 1;
															@(negedge chip_clk)
																;
															ffn1_done = 0;
															burst_cnt = 0;
															addr = (22'h200000 + 22'h001000) + 513;
															FPGA_QSPI_RD(addr, burst_cnt, tpu_state_rdata_array);
															$display("State reg[1] = %d", tpu_state_rdata_array[0+:16]);
															iter_cnt = iter_cnt + 1;
															usr_total_token_cnt_array[j * $clog2(TB_TOTAL_CONTEXT_LENGTH + 1)+:$clog2(TB_TOTAL_CONTEXT_LENGTH + 1)] = usr_total_token_cnt_array[j * $clog2(TB_TOTAL_CONTEXT_LENGTH + 1)+:$clog2(TB_TOTAL_CONTEXT_LENGTH + 1)] + 1;
															if (iter_cnt == 1)
																$display("%32x", top.array_top_inst.inst_residual_sram.ram_piece.mem[0]);
															$display("\n\n\n\n\n\n\n");
															if (iter_cnt == TB_TEST_ITER) begin
																iter_done = 1;
																_sv2v_jump = 2'b10;
															end
														end
													end
												end
											end
										end
										_sv2v_value_on_break = j;
									end
								if (!(_sv2v_jump < 2'b10))
									j = _sv2v_value_on_break;
								if (_sv2v_jump != 2'b11)
									_sv2v_jump = _sv2v_jump_1;
							end
						end
						if (_sv2v_jump == 2'b00) begin
							if (iter_cnt == TB_TEST_ITER) begin
								iter_done = 1;
								_sv2v_jump = 2'b10;
							end
						end
						_sv2v_value_on_break = i;
					end
				if (!(_sv2v_jump < 2'b10))
					i = _sv2v_value_on_break;
				if (_sv2v_jump != 2'b11)
					_sv2v_jump = 2'b00;
			end
		end
		if (_sv2v_jump == 2'b00) begin
			repeat (100) @(negedge chip_clk)
				;
			$finish;
		end
	end
	reg signed [31:0] temp_k;
	genvar _gv_iter_1;
	generate
		for (_gv_h_1 = 0; _gv_h_1 < 8; _gv_h_1 = _gv_h_1 + 1) begin : genblk2
			localparam h = _gv_h_1;
			for (_gv_iter_1 = 0; _gv_iter_1 < 16; _gv_iter_1 = _gv_iter_1 + 1) begin : genblk1
				localparam iter = _gv_iter_1;
				reg [7:0] temp_var;
				initial begin : sv2v_autoblock_94
					reg [0:1] _sv2v_jump;
					_sv2v_jump = 2'b00;
					begin : sv2v_autoblock_95
						reg signed [31:0] p;
						begin : sv2v_autoblock_96
							reg signed [31:0] _sv2v_value_on_break;
							for (p = 0; p < TB_TEST_ITER; p = p + 1)
								if (_sv2v_jump < 2'b10) begin
									_sv2v_jump = 2'b00;
									#(1)
										;
									begin : sv2v_autoblock_97
										reg [0:1] _sv2v_jump_1;
										_sv2v_jump_1 = _sv2v_jump;
										while (!(!(_sv2v_jump < 2'b10))) begin
											_sv2v_jump = 2'b00;
											@(negedge chip_clk)
												;
											if (att_pv_done)
												_sv2v_jump = 2'b10;
										end
										if (_sv2v_jump != 2'b11)
											_sv2v_jump = _sv2v_jump_1;
									end
									if (_sv2v_jump == 2'b00) begin
										user_total_token_cnt = usr_total_token_cnt_array[user_id * $clog2(TB_TOTAL_CONTEXT_LENGTH + 1)+:$clog2(TB_TOTAL_CONTEXT_LENGTH + 1)];
										begin : sv2v_autoblock_98
											reg signed [31:0] j;
											for (j = 0; j < (TB_EMBD_SIZE / 8); j = j + 1)
												begin
													temp_k = j;
													if (iter == (temp_k % 16)) begin
														if ((h % 2) == 0)
															temp_var = top.array_top_inst.inst_head_array.head_gen_array[h / 2].two_heads_inst.inst_head_top_0.inst_head_sram.ram_piece.mem[temp_k / 16][iter * 8+:8];
														else
															temp_var = top.array_top_inst.inst_head_array.head_gen_array[h / 2].two_heads_inst.inst_head_top_1.inst_head_sram.ram_piece.mem[temp_k / 16][iter * 8+:8];
														if ((ATT_PV_array[8 * ((((((h * TB_NUM_USER) + user_id) * TB_TOTAL_CONTEXT_LENGTH) + user_total_token_cnt) * (TB_EMBD_SIZE / 8)) + (j * 8))+:64] !== 'bx) && ($signed(ATT_PV_array[8 * ((((((h * TB_NUM_USER) + user_id) * TB_TOTAL_CONTEXT_LENGTH) + user_total_token_cnt) * (TB_EMBD_SIZE / 8)) + (j * 8))+:64]) !== $signed(temp_var))) begin
															if ($abs(sv2v_cast_32_signed($signed(ATT_PV_array[8 * ((((((h * TB_NUM_USER) + user_id) * TB_TOTAL_CONTEXT_LENGTH) + user_total_token_cnt) * (TB_EMBD_SIZE / 8)) + (j * 8))+:64])) - sv2v_cast_32_signed($signed(temp_var))) > PV_MAX_ERR) begin
																error_flag = 1;
																$display("Wrong!!!, Below Case exceed the error limit");
																$display("ATT_PV_array[%0d][%0d][%0d][%0d]: %0d != HEAD_SRAM[%0d][%0d][%0d][%0d] = %0d", h, user_id, user_total_token_cnt, j, $signed(ATT_PV_array[8 * ((((((h * TB_NUM_USER) + user_id) * TB_TOTAL_CONTEXT_LENGTH) + user_total_token_cnt) * (TB_EMBD_SIZE / 8)) + (j * 8))+:64]), h, user_id, temp_k / 16, iter, $signed(temp_var));
															end
															if (FORCE_TO_BE_RIGHT) begin
																if ((h % 2) == 0)
																	top.array_top_inst.inst_head_array.head_gen_array[h / 2].two_heads_inst.inst_head_top_0.inst_head_sram.ram_piece.mem[temp_k / 16][iter * 8+:8] = $signed(ATT_PV_array[8 * ((((((h * TB_NUM_USER) + user_id) * TB_TOTAL_CONTEXT_LENGTH) + user_total_token_cnt) * (TB_EMBD_SIZE / 8)) + (j * 8))+:64]);
																else
																	top.array_top_inst.inst_head_array.head_gen_array[h / 2].two_heads_inst.inst_head_top_1.inst_head_sram.ram_piece.mem[temp_k / 16][iter * 8+:8] = $signed(ATT_PV_array[8 * ((((((h * TB_NUM_USER) + user_id) * TB_TOTAL_CONTEXT_LENGTH) + user_total_token_cnt) * (TB_EMBD_SIZE / 8)) + (j * 8))+:64]);
															end
														end
													end
												end
										end
									end
									_sv2v_value_on_break = p;
								end
							if (!(_sv2v_jump < 2'b10))
								p = _sv2v_value_on_break;
							if (_sv2v_jump != 2'b11)
								_sv2v_jump = 2'b00;
						end
					end
				end
			end
		end
		for (_gv_iter_1 = 0; _gv_iter_1 < 16; _gv_iter_1 = _gv_iter_1 + 1) begin : genblk3
			localparam iter = _gv_iter_1;
			initial begin : sv2v_autoblock_99
				reg [0:1] _sv2v_jump;
				_sv2v_jump = 2'b00;
				while (!(!(_sv2v_jump < 2'b10))) begin
					_sv2v_jump = 2'b00;
					#(1)
						;
					begin : sv2v_autoblock_100
						reg [0:1] _sv2v_jump_1;
						_sv2v_jump_1 = _sv2v_jump;
						while (!(!(_sv2v_jump < 2'b10))) begin
							_sv2v_jump = 2'b00;
							@(negedge chip_clk)
								;
							if (proj_done)
								_sv2v_jump = 2'b10;
						end
						if (_sv2v_jump != 2'b11)
							_sv2v_jump = _sv2v_jump_1;
					end
					if (_sv2v_jump == 2'b00) begin
						user_total_token_cnt = usr_total_token_cnt_array[user_id * $clog2(TB_TOTAL_CONTEXT_LENGTH + 1)+:$clog2(TB_TOTAL_CONTEXT_LENGTH + 1)];
						begin : sv2v_autoblock_101
							reg signed [31:0] j;
							for (j = 0; j < TB_EMBD_SIZE; j = j + 1)
								begin
									temp_k = j;
									if (iter == (temp_k % 16)) begin
										if ((ATT_PROJ_array[((((user_id * TB_TOTAL_CONTEXT_LENGTH) + user_total_token_cnt) * TB_EMBD_SIZE) + j) * 8+:8] !== 'bx) && ($signed(ATT_PROJ_array[((((user_id * TB_TOTAL_CONTEXT_LENGTH) + user_total_token_cnt) * TB_EMBD_SIZE) + j) * 8+:8]) !== $signed(top.array_top_inst.inst_residual_sram.ram_piece.mem[temp_k / 16][iter * 8+:8]))) begin
											if ($abs(sv2v_cast_32_signed($signed(ATT_PROJ_array[((((user_id * TB_TOTAL_CONTEXT_LENGTH) + user_total_token_cnt) * TB_EMBD_SIZE) + j) * 8+:8])) - sv2v_cast_32_signed($signed(top.array_top_inst.inst_residual_sram.ram_piece.mem[temp_k / 16][iter * 8+:8]))) > PROJ_MAX_ERR) begin
												error_flag = 1;
												$display("Wrong!!!, Below Case exceed the error limit");
												$display("ATT_PROJ_array[%0d][%0d][%0d]: %0d != Residual SRAM[%0d][%0d][%0d] = %0d", user_id, user_total_token_cnt, j, $signed(ATT_PROJ_array[((((user_id * TB_TOTAL_CONTEXT_LENGTH) + user_total_token_cnt) * TB_EMBD_SIZE) + j) * 8+:8]), user_id, temp_k / 16, iter, $signed(top.array_top_inst.inst_residual_sram.ram_piece.mem[temp_k / 16][iter * 8+:8]));
											end
											if (FORCE_TO_BE_RIGHT)
												top.array_top_inst.inst_residual_sram.ram_piece.mem[temp_k / 16][iter * 8+:8] = $signed(ATT_PROJ_array[((((user_id * TB_TOTAL_CONTEXT_LENGTH) + user_total_token_cnt) * TB_EMBD_SIZE) + j) * 8+:8]);
										end
									end
								end
						end
					end
				end
				if (_sv2v_jump != 2'b11)
					_sv2v_jump = 2'b00;
			end
		end
		for (_gv_h_1 = 0; _gv_h_1 < 8; _gv_h_1 = _gv_h_1 + 1) begin : genblk4
			localparam h = _gv_h_1;
			for (_gv_iter_1 = 0; _gv_iter_1 < 16; _gv_iter_1 = _gv_iter_1 + 1) begin : genblk1
				localparam iter = _gv_iter_1;
				reg [7:0] temp_var;
				initial begin : sv2v_autoblock_102
					reg [0:1] _sv2v_jump;
					_sv2v_jump = 2'b00;
					begin : sv2v_autoblock_103
						reg signed [31:0] p;
						begin : sv2v_autoblock_104
							reg signed [31:0] _sv2v_value_on_break;
							for (p = 0; p < TB_TEST_ITER; p = p + 1)
								if (_sv2v_jump < 2'b10) begin
									_sv2v_jump = 2'b00;
									#(1)
										;
									begin : sv2v_autoblock_105
										reg [0:1] _sv2v_jump_1;
										_sv2v_jump_1 = _sv2v_jump;
										while (!(!(_sv2v_jump < 2'b10))) begin
											_sv2v_jump = 2'b00;
											@(negedge chip_clk)
												;
											if (ffn0_done)
												_sv2v_jump = 2'b10;
										end
										if (_sv2v_jump != 2'b11)
											_sv2v_jump = _sv2v_jump_1;
									end
									if (_sv2v_jump == 2'b00) begin
										user_total_token_cnt = usr_total_token_cnt_array[user_id * $clog2(TB_TOTAL_CONTEXT_LENGTH + 1)+:$clog2(TB_TOTAL_CONTEXT_LENGTH + 1)];
										begin : sv2v_autoblock_106
											reg signed [31:0] j;
											for (j = 0; j < ((4 * TB_EMBD_SIZE) / 8); j = j + 1)
												begin
													temp_k = j;
													if (iter == (temp_k % 16)) begin
														if ((h % 2) == 0)
															temp_var = top.array_top_inst.inst_head_array.head_gen_array[h / 2].two_heads_inst.inst_head_top_0.inst_head_sram.ram_piece.mem[temp_k / 16][iter * 8+:8];
														else
															temp_var = top.array_top_inst.inst_head_array.head_gen_array[h / 2].two_heads_inst.inst_head_top_1.inst_head_sram.ram_piece.mem[temp_k / 16][iter * 8+:8];
														if ((FFN0_array[8 * ((((((h * TB_NUM_USER) + user_id) * TB_TOTAL_CONTEXT_LENGTH) + user_total_token_cnt) * ((4 * TB_EMBD_SIZE) / 8)) + (j * 8))+:64] !== 'bx) && ($signed(FFN0_array[8 * ((((((h * TB_NUM_USER) + user_id) * TB_TOTAL_CONTEXT_LENGTH) + user_total_token_cnt) * ((4 * TB_EMBD_SIZE) / 8)) + (j * 8))+:64]) !== $signed(temp_var))) begin
															if ($abs(sv2v_cast_32_signed($signed(FFN0_array[8 * ((((((h * TB_NUM_USER) + user_id) * TB_TOTAL_CONTEXT_LENGTH) + user_total_token_cnt) * ((4 * TB_EMBD_SIZE) / 8)) + (j * 8))+:64])) - sv2v_cast_32_signed($signed(temp_var))) > FFN0_MAX_ERR) begin
																error_flag = 1;
																$display("Wrong!!!, Below Case exceed the error limit");
																$display("FFN0_array[%0d][%0d][%0d][%0d]: %0d != HEAD_SRAM[%0d][%0d][%0d][%0d] = %0d", h, user_id, user_total_token_cnt, j, $signed(FFN0_array[8 * ((((((h * TB_NUM_USER) + user_id) * TB_TOTAL_CONTEXT_LENGTH) + user_total_token_cnt) * ((4 * TB_EMBD_SIZE) / 8)) + (j * 8))+:64]), h, user_id, temp_k / 16, iter, $signed(temp_var));
															end
															if (FORCE_TO_BE_RIGHT) begin
																if ((h % 2) == 0)
																	top.array_top_inst.inst_head_array.head_gen_array[h / 2].two_heads_inst.inst_head_top_0.inst_head_sram.ram_piece.mem[temp_k / 16][iter * 8+:8] = FFN0_array[8 * ((((((h * TB_NUM_USER) + user_id) * TB_TOTAL_CONTEXT_LENGTH) + user_total_token_cnt) * ((4 * TB_EMBD_SIZE) / 8)) + (j * 8))+:64];
																else
																	top.array_top_inst.inst_head_array.head_gen_array[h / 2].two_heads_inst.inst_head_top_1.inst_head_sram.ram_piece.mem[temp_k / 16][iter * 8+:8] = FFN0_array[8 * ((((((h * TB_NUM_USER) + user_id) * TB_TOTAL_CONTEXT_LENGTH) + user_total_token_cnt) * ((4 * TB_EMBD_SIZE) / 8)) + (j * 8))+:64];
															end
														end
													end
												end
										end
									end
									_sv2v_value_on_break = p;
								end
							if (!(_sv2v_jump < 2'b10))
								p = _sv2v_value_on_break;
							if (_sv2v_jump != 2'b11)
								_sv2v_jump = 2'b00;
						end
					end
				end
			end
		end
		for (_gv_iter_1 = 0; _gv_iter_1 < 16; _gv_iter_1 = _gv_iter_1 + 1) begin : genblk5
			localparam iter = _gv_iter_1;
			initial begin : sv2v_autoblock_107
				reg [0:1] _sv2v_jump;
				_sv2v_jump = 2'b00;
				while (!(!(_sv2v_jump < 2'b10))) begin
					_sv2v_jump = 2'b00;
					#(1)
						;
					begin : sv2v_autoblock_108
						reg [0:1] _sv2v_jump_1;
						_sv2v_jump_1 = _sv2v_jump;
						while (!(!(_sv2v_jump < 2'b10))) begin
							_sv2v_jump = 2'b00;
							@(negedge chip_clk)
								;
							if (ffn1_done)
								_sv2v_jump = 2'b10;
						end
						if (_sv2v_jump != 2'b11)
							_sv2v_jump = _sv2v_jump_1;
					end
					if (_sv2v_jump == 2'b00) begin
						user_total_token_cnt = usr_total_token_cnt_array[user_id * $clog2(TB_TOTAL_CONTEXT_LENGTH + 1)+:$clog2(TB_TOTAL_CONTEXT_LENGTH + 1)];
						begin : sv2v_autoblock_109
							reg signed [31:0] j;
							for (j = 0; j < TB_EMBD_SIZE; j = j + 1)
								begin
									temp_k = j;
									if (iter == (temp_k % 16)) begin
										if ((MLP_RESIDUAL_array[((((user_id * TB_TOTAL_CONTEXT_LENGTH) + user_total_token_cnt) * TB_EMBD_SIZE) + j) * 8+:8] !== 'bx) && ($signed(MLP_RESIDUAL_array[((((user_id * TB_TOTAL_CONTEXT_LENGTH) + user_total_token_cnt) * TB_EMBD_SIZE) + j) * 8+:8]) !== $signed(top.array_top_inst.inst_residual_sram.ram_piece.mem[temp_k / 16][iter * 8+:8]))) begin
											if ($abs(sv2v_cast_32_signed($signed(MLP_RESIDUAL_array[((((user_id * TB_TOTAL_CONTEXT_LENGTH) + user_total_token_cnt) * TB_EMBD_SIZE) + j) * 8+:8])) - sv2v_cast_32_signed($signed(top.array_top_inst.inst_residual_sram.ram_piece.mem[temp_k / 16][iter * 8+:8]))) > FFN1_MAX_ERR) begin
												error_flag = 1;
												$display("Wrong!!!, Below Case exceed the error limit");
												$display("MLP_RESIDUAL_array[%0d][%0d][%0d]: %0d != Residual SRAM[%0d][%0d][%0d] = %0d", user_id, user_total_token_cnt, j, $signed(MLP_RESIDUAL_array[((((user_id * TB_TOTAL_CONTEXT_LENGTH) + user_total_token_cnt) * TB_EMBD_SIZE) + j) * 8+:8]), user_id, temp_k / 16, iter, $signed(top.array_top_inst.inst_residual_sram.ram_piece.mem[temp_k / 16][iter * 8+:8]));
											end
										end
									end
								end
						end
					end
				end
				if (_sv2v_jump != 2'b11)
					_sv2v_jump = 2'b00;
			end
		end
		for (_gv_iter_1 = 0; _gv_iter_1 < 16; _gv_iter_1 = _gv_iter_1 + 1) begin : genblk6
			localparam iter = _gv_iter_1;
			initial begin : sv2v_autoblock_110
				reg [0:1] _sv2v_jump;
				_sv2v_jump = 2'b00;
				while (!(!(_sv2v_jump < 2'b10))) begin
					_sv2v_jump = 2'b00;
					#(1)
						;
					begin : sv2v_autoblock_111
						reg [0:1] _sv2v_jump_1;
						_sv2v_jump_1 = _sv2v_jump;
						while (!(!(_sv2v_jump < 2'b10))) begin
							_sv2v_jump = 2'b00;
							@(negedge chip_clk)
								;
							if (ffn1_done)
								_sv2v_jump = 2'b10;
						end
						if (_sv2v_jump != 2'b11)
							_sv2v_jump = _sv2v_jump_1;
					end
					if (_sv2v_jump == 2'b00) begin
						user_total_token_cnt = usr_total_token_cnt_array[user_id * $clog2(TB_TOTAL_CONTEXT_LENGTH + 1)+:$clog2(TB_TOTAL_CONTEXT_LENGTH + 1)];
						begin : sv2v_autoblock_112
							reg signed [31:0] j;
							for (j = 0; j < TB_EMBD_SIZE; j = j + 1)
								begin
									temp_k = j;
									if (iter == (temp_k % 16)) begin
										if ((MLP_RESIDUAL_array[((((user_id * TB_TOTAL_CONTEXT_LENGTH) + user_total_token_cnt) * TB_EMBD_SIZE) + j) * 8+:8] !== 'bx) && ($signed(MLP_RESIDUAL_array[((((user_id * TB_TOTAL_CONTEXT_LENGTH) + user_total_token_cnt) * TB_EMBD_SIZE) + j) * 8+:8]) !== $signed(global_mem_array[((temp_k / 16) * 128) + (iter * 8)+:8]))) begin
											if ($abs(sv2v_cast_32_signed($signed(MLP_RESIDUAL_array[((((user_id * TB_TOTAL_CONTEXT_LENGTH) + user_total_token_cnt) * TB_EMBD_SIZE) + j) * 8+:8])) - sv2v_cast_32_signed($signed(global_mem_array[((temp_k / 16) * 128) + (iter * 8)+:8]))) > FFN1_MAX_ERR) begin
												error_flag = 1;
												$display("Wrong!!!, Below Case exceed the error limit");
												$display("MLP_RESIDUAL_array[%0d][%0d][%0d]: %0d != Read out SRAM[%0d][%0d][%0d] = %0d", user_id, user_total_token_cnt, j, $signed(MLP_RESIDUAL_array[((((user_id * TB_TOTAL_CONTEXT_LENGTH) + user_total_token_cnt) * TB_EMBD_SIZE) + j) * 8+:8]), user_id, temp_k / 16, iter, $signed(global_mem_array[((temp_k / 16) * 128) + (iter * 8)+:8]));
											end
										end
									end
								end
						end
					end
				end
				if (_sv2v_jump != 2'b11)
					_sv2v_jump = 2'b00;
			end
		end
	endgenerate
	always begin
		repeat (10000) @(negedge chip_clk)
			;
		$display("Time: %t", $time());
	end
endmodule

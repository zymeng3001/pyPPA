module array_ctrl (
	clk,
	rst_n,
	control_reg_addr,
	control_reg_wdata,
	control_reg_wen,
	control_reg_rdata,
	control_reg_ren,
	control_reg_rvld,
	control_state,
	control_state_update,
	op_start,
	op_finish,
	op_cfg,
	op_cfg_vld,
	new_token,
	user_first_token,
	user_id,
	current_token_finish,
	usr_cfg,
	usr_cfg_vld,
	cfg_init_success,
	model_cfg,
	model_cfg_vld,
	pmu_cfg,
	pmu_cfg_vld,
	rc_cfg_vld,
	rc_cfg,
	power_mode_en_in,
	debug_mode_en_in,
	debug_mode_bits_in
);
	reg _sv2v_0;
	input wire clk;
	input wire rst_n;
	input wire [5:0] control_reg_addr;
	input wire [15:0] control_reg_wdata;
	input wire control_reg_wen;
	output reg [15:0] control_reg_rdata;
	input wire control_reg_ren;
	output reg control_reg_rvld;
	output reg [31:0] control_state;
	output reg control_state_update;
	output reg op_start;
	input wire op_finish;
	output reg [40:0] op_cfg;
	output reg op_cfg_vld;
	input wire new_token;
	input wire user_first_token;
	input wire [1:0] user_id;
	output reg current_token_finish;
	output reg [11:0] usr_cfg;
	output reg usr_cfg_vld;
	input wire cfg_init_success;
	output reg [29:0] model_cfg;
	output reg model_cfg_vld;
	output reg [3:0] pmu_cfg;
	output reg pmu_cfg_vld;
	output reg rc_cfg_vld;
	output reg [83:0] rc_cfg;
	input wire power_mode_en_in;
	input wire debug_mode_en_in;
	input wire [7:0] debug_mode_bits_in;
	reg [29:0] model_cfg_reg;
	reg [3:0] pmu_cfg_reg;
	reg [83:0] rc_cfg_reg;
	reg [409:0] op_cfg_pkt;
	reg power_mode_en;
	always @(posedge clk or negedge rst_n)
		if (~rst_n)
			power_mode_en <= 0;
		else
			power_mode_en <= power_mode_en_in;
	reg debug_mode_en;
	reg [7:0] debug_mode_bits;
	always @(posedge clk or negedge rst_n)
		if (~rst_n) begin
			debug_mode_en <= 0;
			debug_mode_bits <= 0;
		end
		else begin
			debug_mode_en <= debug_mode_en_in;
			debug_mode_bits <= debug_mode_bits_in;
		end
	reg [5:0] control_reg_addr_delay1;
	reg [15:0] control_reg_wdata_delay1;
	reg control_reg_wen_delay1;
	reg control_reg_ren_delay1;
	always @(posedge clk or negedge rst_n)
		if (~rst_n) begin
			control_reg_addr_delay1 <= 0;
			control_reg_wdata_delay1 <= 0;
			control_reg_wen_delay1 <= 0;
			control_reg_ren_delay1 <= 0;
			control_reg_rvld <= 0;
		end
		else begin
			control_reg_addr_delay1 <= control_reg_addr;
			control_reg_wdata_delay1 <= control_reg_wdata;
			control_reg_wen_delay1 <= control_reg_wen;
			control_reg_ren_delay1 <= control_reg_ren;
			control_reg_rvld <= control_reg_ren_delay1;
		end
	reg [1023:0] control_reg_array;
	always @(posedge clk or negedge rst_n)
		if (~rst_n) begin
			control_reg_array <= 0;
			control_reg_rdata <= 0;
		end
		else begin
			if (control_reg_wen_delay1)
				control_reg_array[control_reg_addr_delay1 * 16+:16] <= control_reg_wdata_delay1;
			if (control_reg_ren_delay1)
				control_reg_rdata <= control_reg_array[control_reg_addr_delay1 * 16+:16];
		end
	always @(*) begin
		if (_sv2v_0)
			;
		model_cfg_reg[29-:10] = control_reg_array[0+:16];
		model_cfg_reg[19-:3] = control_reg_array[16+:16];
		model_cfg_reg[16-:6] = control_reg_array[32+:16];
		model_cfg_reg[10-:10] = control_reg_array[48+:16];
		model_cfg_reg[0] = control_reg_array[64+:16];
		pmu_cfg_reg[3] = control_reg_array[80+:16];
		pmu_cfg_reg[2] = control_reg_array[96+:16];
		pmu_cfg_reg[1] = control_reg_array[112];
		pmu_cfg_reg[0] = control_reg_array[113];
		rc_cfg_reg[83-:5] = control_reg_array[128+:16];
		rc_cfg_reg[78-:10] = control_reg_array[144+:16];
		rc_cfg_reg[68-:16] = control_reg_array[160+:16];
		rc_cfg_reg[52-:16] = control_reg_array[176+:16];
		rc_cfg_reg[36-:5] = control_reg_array[192+:16];
		rc_cfg_reg[31-:16] = control_reg_array[208+:16];
		rc_cfg_reg[15-:16] = control_reg_array[224+:16];
		op_cfg_pkt[409-:10] = control_reg_array[240+:16];
		op_cfg_pkt[399-:10] = control_reg_array[256+:16];
		op_cfg_pkt[389-:16] = control_reg_array[272+:16];
		op_cfg_pkt[373-:5] = control_reg_array[288+:16];
		op_cfg_pkt[368-:10] = control_reg_array[304+:16];
		op_cfg_pkt[358-:10] = control_reg_array[320+:16];
		op_cfg_pkt[348-:16] = control_reg_array[336+:16];
		op_cfg_pkt[332-:5] = control_reg_array[352+:16];
		op_cfg_pkt[327-:10] = control_reg_array[368+:16];
		op_cfg_pkt[317-:10] = control_reg_array[384+:16];
		op_cfg_pkt[307-:16] = control_reg_array[400+:16];
		op_cfg_pkt[291-:5] = control_reg_array[416+:16];
		op_cfg_pkt[286-:10] = control_reg_array[432+:16];
		op_cfg_pkt[276-:10] = control_reg_array[448+:16];
		op_cfg_pkt[266-:16] = control_reg_array[464+:16];
		op_cfg_pkt[250-:5] = control_reg_array[480+:16];
		op_cfg_pkt[245-:10] = control_reg_array[496+:16];
		op_cfg_pkt[235-:10] = control_reg_array[512+:16];
		op_cfg_pkt[225-:16] = control_reg_array[528+:16];
		op_cfg_pkt[209-:5] = control_reg_array[544+:16];
		op_cfg_pkt[204-:10] = control_reg_array[560+:16];
		op_cfg_pkt[194-:10] = control_reg_array[576+:16];
		op_cfg_pkt[184-:16] = control_reg_array[592+:16];
		op_cfg_pkt[168-:5] = control_reg_array[608+:16];
		op_cfg_pkt[163-:10] = control_reg_array[624+:16];
		op_cfg_pkt[153-:10] = control_reg_array[640+:16];
		op_cfg_pkt[143-:16] = control_reg_array[656+:16];
		op_cfg_pkt[127-:5] = control_reg_array[672+:16];
		op_cfg_pkt[122-:10] = control_reg_array[688+:16];
		op_cfg_pkt[112-:10] = control_reg_array[704+:16];
		op_cfg_pkt[102-:16] = control_reg_array[720+:16];
		op_cfg_pkt[86-:5] = control_reg_array[736+:16];
		op_cfg_pkt[81-:10] = control_reg_array[752+:16];
		op_cfg_pkt[71-:10] = control_reg_array[768+:16];
		op_cfg_pkt[61-:16] = control_reg_array[784+:16];
		op_cfg_pkt[45-:5] = control_reg_array[800+:16];
		op_cfg_pkt[40-:10] = control_reg_array[816+:16];
		op_cfg_pkt[30-:10] = control_reg_array[832+:16];
		op_cfg_pkt[20-:16] = control_reg_array[848+:16];
		op_cfg_pkt[4-:5] = control_reg_array[864+:16];
	end
	reg [11:0] usr_cfg_array [3:0];
	localparam op_cfg_vld_delay_num = 5;
	wire op_cfg_vld_delay5;
	assign op_cfg_vld_delay5 = op_cfg_vld_delay_gen_array[4].op_cfg_vld_delay_temp;
	genvar _gv_i_1;
	generate
		for (_gv_i_1 = 0; _gv_i_1 < op_cfg_vld_delay_num; _gv_i_1 = _gv_i_1 + 1) begin : op_cfg_vld_delay_gen_array
			localparam i = _gv_i_1;
			reg op_cfg_vld_delay_temp;
			if (i == 0) begin : genblk1
				always @(posedge clk or negedge rst_n)
					if (~rst_n)
						op_cfg_vld_delay_temp <= 0;
					else
						op_cfg_vld_delay_temp <= op_cfg_vld;
			end
			else begin : genblk1
				always @(posedge clk or negedge rst_n)
					if (~rst_n)
						op_cfg_vld_delay_temp <= 0;
					else
						op_cfg_vld_delay_temp <= op_cfg_vld_delay_gen_array[i - 1].op_cfg_vld_delay_temp;
			end
		end
	endgenerate
	always @(posedge clk or negedge rst_n)
		if (~rst_n) begin
			model_cfg <= 0;
			model_cfg_vld <= 0;
			rc_cfg <= 0;
			rc_cfg_vld <= 0;
			pmu_cfg <= 0;
			pmu_cfg_vld <= 0;
		end
		else if (cfg_init_success) begin
			model_cfg <= model_cfg_reg;
			rc_cfg <= rc_cfg_reg;
			rc_cfg_vld <= 1;
			model_cfg_vld <= 1;
			pmu_cfg <= pmu_cfg_reg;
			pmu_cfg_vld <= 1;
		end
		else begin
			rc_cfg_vld <= 0;
			model_cfg_vld <= 0;
			pmu_cfg_vld <= 0;
		end
	generate
		for (_gv_i_1 = 0; _gv_i_1 < 4; _gv_i_1 = _gv_i_1 + 1) begin : genblk2
			localparam i = _gv_i_1;
			always @(posedge clk or negedge rst_n)
				if (~rst_n) begin
					usr_cfg_array[i][9-:9] <= 0;
					usr_cfg_array[i][11-:2] <= i;
					usr_cfg_array[i][0] <= 1;
				end
				else if (new_token) begin
					if (i == user_id) begin
						if (user_first_token) begin
							usr_cfg_array[i][0] <= 1;
							usr_cfg_array[i][9-:9] <= 0;
							usr_cfg_array[i][11-:2] <= i;
						end
						else begin
							usr_cfg_array[i][11-:2] <= i;
							if (usr_cfg_array[i][9-:9] == (model_cfg_reg[29-:10] - 1)) begin
								usr_cfg_array[i][0] <= 0;
								usr_cfg_array[i][9-:9] <= 0;
							end
							else
								usr_cfg_array[i][9-:9] <= usr_cfg_array[i][9-:9] + 1;
						end
					end
				end
		end
	endgenerate
	reg new_token_delay1;
	reg new_token_delay2;
	reg new_token_delay3;
	reg [1:0] current_user_id;
	always @(posedge clk or negedge rst_n)
		if (~rst_n)
			current_user_id <= 0;
		else if (new_token)
			current_user_id <= user_id;
	always @(posedge clk or negedge rst_n)
		if (~rst_n) begin
			new_token_delay1 <= 0;
			new_token_delay2 <= 0;
			new_token_delay3 <= 0;
		end
		else begin
			new_token_delay1 <= new_token;
			new_token_delay2 <= new_token_delay1;
			new_token_delay3 <= new_token_delay2;
		end
	always @(posedge clk or negedge rst_n)
		if (~rst_n) begin
			usr_cfg <= 0;
			usr_cfg_vld <= 0;
		end
		else if (new_token_delay1) begin
			usr_cfg_vld <= 1;
			usr_cfg <= usr_cfg_array[current_user_id];
		end
		else
			usr_cfg_vld <= 0;
	reg [31:0] nxt_control_state;
	reg nxt_control_state_update;
	reg nxt_op_start;
	reg [40:0] nxt_op_cfg;
	reg nxt_op_cfg_vld;
	reg [31:0] control_state_delay1;
	reg control_state_changed;
	reg nxt_current_token_finish;
	always @(posedge clk or negedge rst_n)
		if (~rst_n)
			control_state_changed <= 0;
		else
			control_state_changed <= control_state_delay1 != control_state;
	always @(posedge clk or negedge rst_n)
		if (~rst_n)
			control_state_delay1 <= 32'd0;
		else
			control_state_delay1 <= control_state;
	always @(*) begin
		if (_sv2v_0)
			;
		nxt_control_state = control_state;
		nxt_control_state_update = 0;
		nxt_op_start = 0;
		nxt_op_cfg = op_cfg;
		nxt_op_cfg_vld = 0;
		nxt_current_token_finish = 0;
		case (control_state)
			32'd0:
				if (new_token_delay3) begin
					nxt_control_state = 32'd1;
					nxt_control_state_update = 1;
				end
			32'd1: begin
				if (control_state_changed) begin
					nxt_op_cfg = op_cfg_pkt[409-:41];
					if (~debug_mode_en)
						nxt_op_cfg_vld = 1;
					else if (debug_mode_bits[0])
						nxt_op_cfg_vld = 1;
					else begin
						nxt_op_cfg_vld = 0;
						nxt_control_state = 32'd2;
						nxt_control_state_update = 1;
					end
				end
				if (op_cfg_vld_delay5)
					nxt_op_start = 1;
				if (op_finish) begin
					nxt_control_state = 32'd2;
					nxt_control_state_update = 1;
				end
			end
			32'd2: begin
				if (control_state_changed) begin
					nxt_op_cfg = op_cfg_pkt[368-:41];
					if (~debug_mode_en)
						nxt_op_cfg_vld = 1;
					else if (debug_mode_bits[1])
						nxt_op_cfg_vld = 1;
					else begin
						nxt_op_cfg_vld = 0;
						nxt_control_state = 32'd3;
						nxt_control_state_update = 1;
					end
				end
				if (op_cfg_vld)
					nxt_op_start = 1;
				if (op_finish) begin
					nxt_control_state = 32'd3;
					nxt_control_state_update = 1;
				end
			end
			32'd3: begin
				if (control_state_changed) begin
					nxt_op_cfg = op_cfg_pkt[327-:41];
					if (~debug_mode_en)
						nxt_op_cfg_vld = 1;
					else if (debug_mode_bits[2])
						nxt_op_cfg_vld = 1;
					else begin
						nxt_op_cfg_vld = 0;
						nxt_control_state = 32'd4;
						nxt_control_state_update = 1;
					end
				end
				if (op_cfg_vld)
					nxt_op_start = 1;
				if (op_finish) begin
					nxt_control_state = 32'd4;
					nxt_control_state_update = 1;
				end
			end
			32'd4: begin
				if (control_state_changed) begin
					nxt_op_cfg = op_cfg_pkt[286-:41];
					if (~debug_mode_en)
						nxt_op_cfg_vld = 1;
					else if (debug_mode_bits[3])
						nxt_op_cfg_vld = 1;
					else begin
						nxt_op_cfg_vld = 0;
						nxt_control_state = 32'd5;
						nxt_control_state_update = 1;
					end
				end
				if (op_cfg_vld)
					nxt_op_start = 1;
				if (op_finish) begin
					nxt_control_state = 32'd5;
					nxt_control_state_update = 1;
				end
			end
			32'd5: begin
				if (control_state_changed) begin
					nxt_op_cfg = op_cfg_pkt[245-:41];
					if (usr_cfg[0])
						nxt_op_cfg[40-:10] = (usr_cfg[9-:9] / 16) + 1;
					else
						nxt_op_cfg[40-:10] = model_cfg_reg[29-:10] / 16;
					if (~debug_mode_en)
						nxt_op_cfg_vld = 1;
					else if (debug_mode_bits[4])
						nxt_op_cfg_vld = 1;
					else begin
						nxt_op_cfg_vld = 0;
						nxt_control_state = 32'd6;
						nxt_control_state_update = 1;
					end
				end
				if (op_cfg_vld)
					nxt_op_start = 1;
				if (op_finish) begin
					nxt_control_state = 32'd6;
					nxt_control_state_update = 1;
				end
			end
			32'd6: begin
				if (control_state_changed) begin
					nxt_op_cfg = op_cfg_pkt[204-:41];
					if (~debug_mode_en)
						nxt_op_cfg_vld = 1;
					else if (debug_mode_bits[5])
						nxt_op_cfg_vld = 1;
					else begin
						nxt_op_cfg_vld = 0;
						nxt_control_state = 32'd7;
						nxt_control_state_update = 1;
					end
				end
				if (op_cfg_vld)
					nxt_op_start = 1;
				if (op_finish) begin
					nxt_control_state = 32'd7;
					nxt_control_state_update = 1;
				end
			end
			32'd7: begin
				if (control_state_changed) begin
					nxt_op_cfg = op_cfg_pkt[163-:41];
					if (~debug_mode_en)
						nxt_op_cfg_vld = 1;
					else if (debug_mode_bits[6])
						nxt_op_cfg_vld = 1;
					else begin
						nxt_op_cfg_vld = 0;
						nxt_control_state = 32'd8;
						nxt_control_state_update = 1;
					end
				end
				if (op_cfg_vld_delay5)
					nxt_op_start = 1;
				if (op_finish) begin
					nxt_control_state = 32'd8;
					nxt_control_state_update = 1;
				end
			end
			32'd8: begin
				if (control_state_changed) begin
					nxt_op_cfg = op_cfg_pkt[122-:41];
					if (~debug_mode_en)
						nxt_op_cfg_vld = 1;
					else if (debug_mode_bits[7])
						nxt_op_cfg_vld = 1;
					else begin
						nxt_op_cfg_vld = 0;
						if (power_mode_en)
							nxt_control_state = 32'd1;
						else
							nxt_control_state = 32'd0;
						nxt_control_state_update = 1;
						nxt_current_token_finish = 1;
					end
				end
				if (op_cfg_vld)
					nxt_op_start = 1;
				if (op_finish) begin
					if (power_mode_en)
						nxt_control_state = 32'd1;
					else
						nxt_control_state = 32'd0;
					nxt_control_state_update = 1;
					nxt_current_token_finish = 1;
				end
			end
		endcase
	end
	always @(posedge clk or negedge rst_n)
		if (~rst_n) begin
			control_state <= 32'd0;
			control_state_update <= 0;
			op_start <= 0;
			op_cfg <= 0;
			op_cfg_vld <= 0;
			current_token_finish <= 0;
		end
		else begin
			control_state <= nxt_control_state;
			control_state_update <= nxt_control_state_update;
			op_start <= nxt_op_start;
			op_cfg <= nxt_op_cfg;
			op_cfg_vld <= nxt_op_cfg_vld;
			current_token_finish <= nxt_current_token_finish;
		end
	initial _sv2v_0 = 0;
endmodule

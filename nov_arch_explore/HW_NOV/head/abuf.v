module abuf (
	clk,
	rst_n,
	start,
	in_data,
	in_data_vld,
	control_state,
	control_state_update,
	model_cfg_vld,
	model_cfg,
	usr_cfg,
	usr_cfg_vld,
	out_data,
	out_data_vld,
	finish_row
);
	reg _sv2v_0;
	input wire clk;
	input wire rst_n;
	input start;
	input wire [127:0] in_data;
	input wire in_data_vld;
	input wire [31:0] control_state;
	input wire control_state_update;
	input wire model_cfg_vld;
	input wire [29:0] model_cfg;
	input wire [11:0] usr_cfg;
	input usr_cfg_vld;
	output reg [127:0] out_data;
	output reg out_data_vld;
	output reg finish_row;
	reg start_reg;
	reg next_finish_row;
	reg reuse_row;
	reg next_reuse_row;
	reg [4095:0] embd_reg;
	reg [4095:0] next_embd_reg;
	reg [127:0] next_out_data;
	reg next_out_data_vld;
	reg [4:0] embd_reg_wr_ptr;
	reg [4:0] next_embd_reg_wr_ptr;
	reg [4:0] embd_reg_rd_ptr;
	reg [4:0] next_embd_reg_rd_ptr;
	reg [9:0] row_iteration_cnt;
	reg [9:0] next_row_iteration_cnt;
	reg [9:0] max_row_iteration_cnt;
	reg [5:0] max_embd_reg_cnt;
	reg [31:0] control_state_reg;
	reg [29:0] model_cfg_reg;
	reg [11:0] usr_cfg_reg;
	always @(posedge clk or negedge rst_n)
		if (~rst_n)
			start_reg <= 0;
		else
			start_reg <= start;
	always @(posedge clk or negedge rst_n)
		if (~rst_n)
			control_state_reg <= 32'd0;
		else if (control_state_update)
			control_state_reg <= control_state;
	always @(posedge clk or negedge rst_n)
		if (~rst_n)
			model_cfg_reg <= 0;
		else if (model_cfg_vld)
			model_cfg_reg <= model_cfg;
	always @(posedge clk or negedge rst_n)
		if (~rst_n)
			usr_cfg_reg <= 0;
		else if (usr_cfg_vld)
			usr_cfg_reg <= usr_cfg;
	always @(posedge clk or negedge rst_n)
		if (~rst_n) begin
			max_row_iteration_cnt <= 0;
			max_embd_reg_cnt <= 0;
		end
		else if (start_reg)
			case (control_state_reg)
				32'd1, 32'd2, 32'd3: begin
					max_row_iteration_cnt <= model_cfg_reg[19-:3];
					max_embd_reg_cnt <= model_cfg_reg[10-:10] / 16;
				end
				32'd4: begin
					max_embd_reg_cnt <= (model_cfg_reg[19-:3] * 16) / 16;
					max_row_iteration_cnt <= model_cfg_reg[16-:6];
				end
				32'd5: begin
					max_row_iteration_cnt <= model_cfg_reg[19-:3];
					if (usr_cfg_reg[0])
						max_embd_reg_cnt <= (usr_cfg_reg[9-:9] / 16) + 1;
					else
						max_embd_reg_cnt <= model_cfg_reg[29-:10] / 16;
				end
				32'd6: begin
					max_row_iteration_cnt <= model_cfg_reg[10-:10] / 16;
					max_embd_reg_cnt <= (model_cfg_reg[19-:3] * 16) / 16;
				end
				32'd7: begin
					max_row_iteration_cnt <= model_cfg_reg[19-:3] * 4;
					max_embd_reg_cnt <= model_cfg_reg[10-:10] / 16;
				end
				32'd8: begin
					max_row_iteration_cnt <= model_cfg_reg[10-:10] / 16;
					max_embd_reg_cnt <= ((model_cfg_reg[19-:3] * 16) * 4) / 16;
				end
			endcase
	always @(*) begin
		if (_sv2v_0)
			;
		next_out_data = 0;
		next_out_data_vld = 0;
		next_row_iteration_cnt = row_iteration_cnt;
		next_embd_reg_wr_ptr = embd_reg_wr_ptr;
		next_embd_reg_rd_ptr = embd_reg_rd_ptr;
		next_finish_row = 0;
		next_reuse_row = reuse_row;
		next_embd_reg = embd_reg;
		case (control_state_reg)
			32'd0: begin
				next_out_data = 0;
				next_out_data_vld = 0;
				next_row_iteration_cnt = 0;
				next_embd_reg_wr_ptr = 0;
				next_embd_reg_rd_ptr = 0;
				next_finish_row = 0;
				next_reuse_row = 0;
				next_embd_reg = 0;
			end
			default:
				if (reuse_row) begin
					next_out_data_vld = 1;
					next_out_data = embd_reg[embd_reg_rd_ptr * 128+:128];
					next_embd_reg_rd_ptr = embd_reg_rd_ptr + 1;
					if ((out_data_vld && (embd_reg_rd_ptr == (max_embd_reg_cnt - 1))) && reuse_row) begin
						if (row_iteration_cnt == (max_row_iteration_cnt - 1)) begin
							next_out_data_vld = 1;
							next_out_data = embd_reg[embd_reg_rd_ptr * 128+:128];
							next_row_iteration_cnt = 0;
							next_embd_reg_rd_ptr = 0;
							next_finish_row = 1;
							next_reuse_row = 0;
							next_embd_reg = 0;
						end
						else begin
							next_out_data_vld = 1;
							next_out_data = embd_reg[embd_reg_rd_ptr * 128+:128];
							next_row_iteration_cnt = row_iteration_cnt + 1;
							next_embd_reg_rd_ptr = 0;
						end
					end
				end
				else if (in_data_vld) begin
					next_out_data_vld = 1;
					next_out_data = in_data;
					next_embd_reg_wr_ptr = embd_reg_wr_ptr + 1;
					if (max_row_iteration_cnt != 1)
						next_embd_reg[embd_reg_wr_ptr * 128+:128] = in_data;
					if (embd_reg_wr_ptr == (max_embd_reg_cnt - 1)) begin
						next_embd_reg_wr_ptr = 0;
						if (max_row_iteration_cnt == 1) begin
							next_row_iteration_cnt = 0;
							next_finish_row = 1;
							next_reuse_row = 0;
						end
						else begin
							next_embd_reg[embd_reg_wr_ptr * 128+:128] = in_data;
							next_reuse_row = 1;
							next_row_iteration_cnt = row_iteration_cnt + 1;
						end
					end
				end
		endcase
	end
	always @(posedge clk or negedge rst_n)
		if (~rst_n) begin
			out_data <= 0;
			out_data_vld <= 0;
			row_iteration_cnt <= 0;
			embd_reg_wr_ptr <= 0;
			embd_reg_rd_ptr <= 0;
			finish_row <= 0;
			reuse_row <= 0;
			embd_reg <= 0;
		end
		else begin
			out_data <= next_out_data;
			out_data_vld <= next_out_data_vld;
			row_iteration_cnt <= next_row_iteration_cnt;
			embd_reg_wr_ptr <= next_embd_reg_wr_ptr;
			embd_reg_rd_ptr <= next_embd_reg_rd_ptr;
			finish_row <= next_finish_row;
			reuse_row <= next_reuse_row;
			embd_reg <= next_embd_reg;
		end
	initial _sv2v_0 = 0;
endmodule

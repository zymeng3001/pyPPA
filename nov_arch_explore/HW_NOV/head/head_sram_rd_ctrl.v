module head_sram_rd_ctrl (
	clk,
	rst_n,
	control_state,
	control_state_update,
	model_cfg_vld,
	model_cfg,
	usr_cfg,
	usr_cfg_vld,
	start,
	finish,
	head_sram_ren,
	head_sram_raddr
);
	reg _sv2v_0;
	input wire clk;
	input wire rst_n;
	input wire [31:0] control_state;
	input wire control_state_update;
	input wire model_cfg_vld;
	input wire [29:0] model_cfg;
	input wire [11:0] usr_cfg;
	input usr_cfg_vld;
	input wire start;
	output reg finish;
	output reg head_sram_ren;
	output reg [8:0] head_sram_raddr;
	reg [31:0] control_state_reg;
	reg [29:0] model_cfg_reg;
	reg [11:0] usr_cfg_reg;
	reg start_reg;
	always @(posedge clk or negedge rst_n)
		if (~rst_n)
			control_state_reg <= 32'd0;
		else if (control_state_update)
			control_state_reg <= control_state;
	always @(posedge clk or negedge rst_n)
		if (~rst_n)
			start_reg <= 0;
		else
			start_reg <= start;
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
	reg nxt_finish;
	reg nxt_head_sram_ren;
	reg [8:0] nxt_head_sram_raddr;
	reg [9:0] max_head_sram_raddr_cnt;
	always @(posedge clk or negedge rst_n)
		if (~rst_n)
			max_head_sram_raddr_cnt <= 0;
		else if (start_reg)
			case (control_state_reg)
				32'd4, 32'd6: max_head_sram_raddr_cnt <= (model_cfg_reg[19-:3] * 16) / 16;
				32'd5:
					if (usr_cfg_reg[0])
						max_head_sram_raddr_cnt <= (usr_cfg_reg[9-:9] / 16) + 1;
					else
						max_head_sram_raddr_cnt <= model_cfg_reg[29-:10] / 16;
				32'd8: max_head_sram_raddr_cnt <= ((model_cfg_reg[19-:3] * 16) * 4) / 16;
			endcase
	always @(*) begin
		if (_sv2v_0)
			;
		nxt_finish = 0;
		nxt_head_sram_ren = 0;
		nxt_head_sram_raddr = head_sram_raddr;
		case (control_state_reg)
			32'd0: nxt_finish = 0;
			32'd1, 32'd2, 32'd3, 32'd7:
				if (start_reg)
					nxt_finish = 1;
				else if (finish)
					nxt_finish = 0;
			32'd4, 32'd5, 32'd6, 32'd8:
				if (start_reg) begin
					nxt_head_sram_ren = 1;
					nxt_head_sram_raddr = 0;
				end
				else if (finish) begin
					nxt_head_sram_ren = 0;
					nxt_head_sram_raddr = 0;
					nxt_finish = 0;
				end
				else if (head_sram_ren & (head_sram_raddr == (max_head_sram_raddr_cnt - 1))) begin
					nxt_head_sram_ren = 0;
					nxt_head_sram_raddr = head_sram_raddr + 1;
					nxt_finish = 1;
				end
				else if (head_sram_ren) begin
					nxt_head_sram_ren = 1;
					nxt_head_sram_raddr = head_sram_raddr + 1;
				end
		endcase
	end
	always @(posedge clk or negedge rst_n)
		if (~rst_n) begin
			finish <= 0;
			head_sram_ren <= 0;
			head_sram_raddr <= 0;
		end
		else begin
			finish <= nxt_finish;
			head_sram_ren <= nxt_head_sram_ren;
			head_sram_raddr <= nxt_head_sram_raddr;
		end
	initial _sv2v_0 = 0;
endmodule

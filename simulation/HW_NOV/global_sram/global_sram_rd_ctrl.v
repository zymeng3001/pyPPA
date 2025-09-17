module global_sram_rd_ctrl (
	clk,
	rst_n,
	control_state,
	control_state_update,
	model_cfg_vld,
	model_cfg,
	start,
	finish,
	vector_out_data_addr,
	vector_out_data_vld,
	global_sram_ren,
	global_sram_raddr
);
	reg _sv2v_0;
	parameter MAX_QKV_WEIGHT_COLS_PER_CORE = 4;
	input wire clk;
	input wire rst_n;
	input wire [31:0] control_state;
	input wire control_state_update;
	input wire model_cfg_vld;
	input wire [29:0] model_cfg;
	input wire start;
	output reg finish;
	input wire [12:0] vector_out_data_addr;
	input wire vector_out_data_vld;
	output reg global_sram_ren;
	output reg [4:0] global_sram_raddr;
	reg [31:0] control_state_reg;
	reg [29:0] model_cfg_reg;
	reg start_reg;
	always @(posedge clk or negedge rst_n)
		if (~rst_n)
			start_reg <= 0;
		else if (start)
			start_reg <= 1;
		else
			start_reg <= 0;
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
	reg nxt_finish;
	reg nxt_global_sram_ren;
	reg [4:0] nxt_global_sram_raddr;
	always @(*) begin
		if (_sv2v_0)
			;
		nxt_global_sram_ren = 0;
		nxt_global_sram_raddr = 0;
		nxt_finish = 0;
		case (control_state_reg)
			32'd0: begin
				nxt_global_sram_ren = 0;
				nxt_global_sram_raddr = 0;
				nxt_finish = 0;
			end
			32'd1, 32'd2, 32'd3, 32'd7:
				if (start_reg) begin
					nxt_global_sram_ren = 1;
					nxt_global_sram_raddr = 0;
				end
				else if (finish) begin
					nxt_global_sram_ren = 0;
					nxt_global_sram_raddr = 0;
					nxt_finish = 0;
				end
				else if (global_sram_raddr == ((model_cfg_reg[10-:10] / 16) - 1)) begin
					nxt_global_sram_ren = 0;
					nxt_global_sram_raddr = global_sram_raddr + 1;
					nxt_finish = 1;
				end
				else if (global_sram_ren) begin
					nxt_global_sram_ren = 1;
					nxt_global_sram_raddr = global_sram_raddr + 1;
				end
			32'd8: begin
				nxt_global_sram_ren = vector_out_data_vld;
				nxt_global_sram_raddr = vector_out_data_addr;
			end
			32'd4, 32'd5, 32'd6:
				if (start_reg)
					nxt_finish = 1;
				else if (finish)
					nxt_finish = 0;
		endcase
	end
	always @(posedge clk or negedge rst_n)
		if (~rst_n) begin
			global_sram_ren <= 0;
			global_sram_raddr <= 0;
			finish <= 0;
		end
		else begin
			global_sram_ren <= nxt_global_sram_ren;
			global_sram_raddr <= nxt_global_sram_raddr;
			finish <= nxt_finish;
		end
	initial _sv2v_0 = 0;
endmodule

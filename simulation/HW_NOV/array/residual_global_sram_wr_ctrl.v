module residual_global_sram_wr_ctrl (
	clk,
	rst_n,
	control_state,
	control_state_update,
	model_cfg_vld,
	model_cfg,
	vector_out_data,
	vector_out_data_vld,
	vector_out_data_addr,
	rs_adder_out_data,
	rs_adder_out_data_vld,
	rs_adder_out_addr,
	residual_sram_wdata_byte_flag,
	residual_sram_wen,
	residual_sram_waddr,
	residual_sram_wdata,
	global_sram_waddr,
	global_sram_wen,
	global_sram_wdata
);
	reg _sv2v_0;
	input wire clk;
	input wire rst_n;
	input wire [31:0] control_state;
	input wire control_state_update;
	input wire model_cfg_vld;
	input wire [29:0] model_cfg;
	input wire [7:0] vector_out_data;
	input wire vector_out_data_vld;
	input wire [12:0] vector_out_data_addr;
	input wire [127:0] rs_adder_out_data;
	input wire rs_adder_out_data_vld;
	input wire [8:0] rs_adder_out_addr;
	output reg residual_sram_wdata_byte_flag;
	output reg residual_sram_wen;
	output reg [8:0] residual_sram_waddr;
	output reg [127:0] residual_sram_wdata;
	output reg [4:0] global_sram_waddr;
	output reg global_sram_wen;
	output reg [127:0] global_sram_wdata;
	reg [31:0] control_state_reg;
	reg [29:0] model_cfg_reg;
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
	reg nxt_residual_sram_wdata_byte_flag;
	reg nxt_residual_sram_wen;
	reg [8:0] nxt_residual_sram_waddr;
	reg [127:0] nxt_residual_sram_wdata;
	reg [4:0] nxt_global_sram_waddr;
	reg nxt_global_sram_wen;
	reg [127:0] nxt_global_sram_wdata;
	always @(*) begin
		if (_sv2v_0)
			;
		nxt_residual_sram_wdata_byte_flag = 0;
		nxt_residual_sram_wen = 0;
		nxt_residual_sram_waddr = 0;
		nxt_residual_sram_wdata = 0;
		nxt_global_sram_waddr = 0;
		nxt_global_sram_wen = 0;
		nxt_global_sram_wdata = 0;
		if (control_state_reg == 32'd6) begin
			nxt_residual_sram_wdata_byte_flag = 1;
			nxt_residual_sram_waddr = vector_out_data_addr;
			nxt_residual_sram_wen = vector_out_data_vld;
			nxt_residual_sram_wdata = vector_out_data;
		end
		else if (control_state_reg == 32'd8) begin
			nxt_residual_sram_wdata_byte_flag = 1;
			nxt_residual_sram_waddr = rs_adder_out_addr;
			nxt_residual_sram_wen = rs_adder_out_data_vld;
			nxt_residual_sram_wdata = rs_adder_out_data[7:0];
		end
		if (control_state_reg == 32'd7) begin
			nxt_global_sram_waddr = rs_adder_out_addr;
			nxt_global_sram_wen = rs_adder_out_data_vld;
			nxt_global_sram_wdata = rs_adder_out_data;
		end
	end
	always @(posedge clk or negedge rst_n)
		if (~rst_n) begin
			residual_sram_wdata_byte_flag <= 0;
			residual_sram_wen <= 0;
			residual_sram_waddr <= 0;
			residual_sram_wdata <= 0;
			global_sram_waddr <= 0;
			global_sram_wen <= 0;
			global_sram_wdata <= 0;
		end
		else begin
			residual_sram_wdata_byte_flag <= nxt_residual_sram_wdata_byte_flag;
			residual_sram_wen <= nxt_residual_sram_wen;
			residual_sram_waddr <= nxt_residual_sram_waddr;
			residual_sram_wdata <= nxt_residual_sram_wdata;
			global_sram_waddr <= nxt_global_sram_waddr;
			global_sram_wen <= nxt_global_sram_wen;
			global_sram_wdata <= nxt_global_sram_wdata;
		end
	initial _sv2v_0 = 0;
endmodule

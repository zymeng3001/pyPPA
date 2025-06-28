module top (
	chip_clk,
	asyn_rst,
	spi_clk,
	spi_csn,
	spi_mosi,
	spi_miso,
	qspi_clk,
	qspi_mosi,
	qspi_mosi_valid,
	qspi_miso,
	qspi_miso_valid,
	current_token_finish_flag,
	current_token_finish_work,
	qgen_state_work,
	qgen_state_end,
	kgen_state_work,
	kgen_state_end,
	vgen_state_work,
	vgen_state_end,
	att_qk_state_work,
	att_qk_state_end,
	att_pv_state_work,
	att_pv_state_end,
	proj_state_work,
	proj_state_end,
	ffn0_state_work,
	ffn0_state_end,
	ffn1_state_work,
	ffn1_state_end
);
	input wire chip_clk;
	input wire asyn_rst;
	input wire spi_clk;
	input wire spi_csn;
	input wire spi_mosi;
	output wire spi_miso;
	input wire qspi_clk;
	input wire [15:0] qspi_mosi;
	input wire qspi_mosi_valid;
	output wire [15:0] qspi_miso;
	output wire qspi_miso_valid;
	output wire current_token_finish_flag;
	output wire current_token_finish_work;
	output wire qgen_state_work;
	output wire qgen_state_end;
	output wire kgen_state_work;
	output wire kgen_state_end;
	output wire vgen_state_work;
	output wire vgen_state_end;
	output wire att_qk_state_work;
	output wire att_qk_state_end;
	output wire att_pv_state_work;
	output wire att_pv_state_end;
	output wire proj_state_work;
	output wire proj_state_end;
	output wire ffn0_state_work;
	output wire ffn0_state_end;
	output wire ffn1_state_work;
	output wire ffn1_state_end;
	wire asyn_rst_n;
	assign asyn_rst_n = ~asyn_rst;
	reg [15:0] spi_rdata;
	reg spi_rvalid;
	wire spi_ren;
	wire [15:0] spi_wdata;
	wire spi_wen;
	wire [21:0] spi_addr;
	spi_slave #(
		.DW(16),
		.AW(24),
		.CNT(6)
	) spi_slave(
		.clk(chip_clk),
		.rst(asyn_rst),
		.rdata(spi_rdata),
		.rvalid(spi_rvalid),
		.ren(spi_ren),
		.wdata(spi_wdata),
		.wen(spi_wen),
		.addr(spi_addr),
		.spi_clk(spi_clk),
		.spi_csn(spi_csn),
		.spi_mosi(spi_mosi),
		.spi_miso(spi_miso)
	);
	reg [15:0] qspi_rdata;
	reg qspi_rvalid;
	wire qspi_ren;
	wire [15:0] qspi_wdata;
	wire qspi_wen;
	wire [21:0] qspi_addr;
	qspi_slave #(
		.DW(16),
		.CTRL_AW(22),
		.MOSI_FIFO_AW(4),
		.MISO_FIFO_AW(6)
	) qspi_slave(
		.clk_slow(qspi_clk),
		.rst_slow(asyn_rst),
		.mosi(qspi_mosi),
		.mosi_valid(qspi_mosi_valid),
		.miso(qspi_miso),
		.miso_valid(qspi_miso_valid),
		.clk_fast(chip_clk),
		.rst_fast(asyn_rst),
		.rdata(qspi_rdata),
		.rvalid(qspi_rvalid),
		.ren(qspi_ren),
		.wdata(qspi_wdata),
		.wen(qspi_wen),
		.addr(qspi_addr),
		.mosi_fifo_wfull(),
		.mosi_fifo_rempty(),
		.miso_fifo_wfull(),
		.miso_fifo_rempty()
	);
	reg [21:0] interface_addr;
	reg interface_wen;
	reg [15:0] interface_wdata;
	reg interface_ren;
	wire [15:0] interface_rdata;
	wire interface_rvalid;
	always @(posedge chip_clk or negedge asyn_rst_n)
		if (~asyn_rst_n)
			interface_addr <= 0;
		else if (spi_wen || spi_ren)
			interface_addr <= spi_addr;
		else if (qspi_ren || qspi_wen)
			interface_addr <= qspi_addr;
	always @(posedge chip_clk or negedge asyn_rst_n)
		if (~asyn_rst_n) begin
			interface_wen <= 0;
			interface_wdata <= 0;
		end
		else if (spi_wen) begin
			interface_wen <= 1;
			interface_wdata <= spi_wdata;
		end
		else if (qspi_wen) begin
			interface_wen <= 1;
			interface_wdata <= qspi_wdata;
		end
		else
			interface_wen <= 0;
	always @(posedge chip_clk or negedge asyn_rst_n)
		if (~asyn_rst_n)
			interface_ren <= 0;
		else if (spi_ren)
			interface_ren <= 1;
		else if (qspi_ren)
			interface_ren <= 1;
		else
			interface_ren <= 0;
	always @(posedge chip_clk or negedge asyn_rst_n)
		if (~asyn_rst_n) begin
			spi_rdata <= 0;
			spi_rvalid <= 0;
			qspi_rvalid <= 0;
			qspi_rdata <= 0;
		end
		else if (interface_rvalid) begin
			spi_rvalid <= 1;
			spi_rdata <= interface_rdata;
			qspi_rvalid <= 1;
			qspi_rdata <= interface_rdata;
		end
		else begin
			spi_rvalid <= 0;
			qspi_rvalid <= 0;
		end
	array_top array_top_inst(
		.clk(chip_clk),
		.rst_n(asyn_rst_n),
		.interface_addr(interface_addr),
		.interface_wen(interface_wen),
		.interface_wdata(interface_wdata),
		.interface_ren(interface_ren),
		.interface_rdata(interface_rdata),
		.interface_rvalid(interface_rvalid),
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
endmodule

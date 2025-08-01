module top(
    input  logic                chip_clk,
    input  logic                asyn_rst, //rst!, not rst_n

    // SPI Domain
    input  logic                spi_clk,        
    input  logic                spi_csn,        // SPI Active Low
    input  logic                spi_mosi,       // Host -> SPI
    output logic                spi_miso,       // Host <- SPI

    // QSPI Domain
    input  logic                qspi_clk,        //qspi clk
    input  logic [15:0]         qspi_mosi,       // Host -> QSPI -> TPU
    input  logic                qspi_mosi_valid,
    output logic [15:0]         qspi_miso,       // Host <- QSPI <- TPU
    output logic                qspi_miso_valid,

    // STATE PIN
    output logic                current_token_finish_flag,
    output logic                current_token_finish_work,

    output logic                qgen_state_work,
    output logic                qgen_state_end,

    output logic                kgen_state_work,
    output logic                kgen_state_end,

    output logic                vgen_state_work,
    output logic                vgen_state_end,

    output logic                att_qk_state_work,
    output logic                att_qk_state_end,

    output logic                att_pv_state_work,
    output logic                att_pv_state_end,

    output logic                proj_state_work,
    output logic                proj_state_end,

    output logic                ffn0_state_work,
    output logic                ffn0_state_end,

    output logic                ffn1_state_work,
    output logic                ffn1_state_end
);
// logic syn_rst_slow;
// logic syn_rst_n_slow;
// logic syn_rst_fast;
// logic syn_rst_n_fast;
// assign syn_rst_n_slow = ~syn_rst_slow;
// assign syn_rst_n_fast = ~syn_rst_fast;

logic asyn_rst_n;
assign asyn_rst_n = ~asyn_rst;
// rst_gen rst_gen(
//     .rst(asyn_rst),
//     .clk_fast(chip_clk),
//     .clk_slow(qspi_clk),
//     .rst_fast(syn_rst_fast),
//     .rst_slow(syn_rst_slow)
// );


logic [`INTERFACE_DATA_WIDTH-1:0]    spi_rdata;
logic                                spi_rvalid;
logic                                spi_ren;
logic [`INTERFACE_DATA_WIDTH-1:0]    spi_wdata;
logic                                spi_wen;
logic [`INTERFACE_ADDR_WIDTH-1:0]    spi_addr;

spi_slave #(
    .DW(`INTERFACE_DATA_WIDTH),   // Data Width
    .AW(`INTERFACE_ADDR_WIDTH+2), // Address and Command Width
    .CNT($clog2(`INTERFACE_DATA_WIDTH + `INTERFACE_ADDR_WIDTH + 2 +1))    // SPI Counter
)spi_slave(
    // Logic Domain
    .clk(chip_clk),
    .rst(asyn_rst),
    .rdata(spi_rdata),          // SPI <- TPU
    .rvalid(spi_rvalid),
    .ren(spi_ren),
    .wdata(spi_wdata),          // SPI -> TPU
    .wen(spi_wen),
    .addr(spi_addr),           // SPI -> TPU
    // output  reg             avalid,

    // SPI Domain
    .spi_clk(spi_clk),
    .spi_csn(spi_csn),        // SPI Active Low
    .spi_mosi(spi_mosi),       // Host -> SPI
    .spi_miso(spi_miso)       // Host <- SPI
);

logic     [`INTERFACE_DATA_WIDTH-1:0]        qspi_rdata;      // Host <- QSPI <- TPU
logic                                        qspi_rvalid;
logic                                        qspi_ren;
logic     [`INTERFACE_DATA_WIDTH-1:0]        qspi_wdata;      // Host -> QSPI -> TPU
logic                                        qspi_wen;
logic     [`INTERFACE_ADDR_WIDTH-1:0]        qspi_addr;       // Host -> QSPI -> TPU



qspi_slave #(
    .DW(`INTERFACE_DATA_WIDTH),
    .CTRL_AW(`INTERFACE_ADDR_WIDTH),
    .MOSI_FIFO_AW(4),
    .MISO_FIFO_AW(6)
)qspi_slave(
    // QSPI Domain
    .clk_slow(qspi_clk),
    .rst_slow(asyn_rst),
    .mosi(qspi_mosi),       // Host -> QSPI -> TPU
    .mosi_valid(qspi_mosi_valid),
    .miso(qspi_miso),       // Host <- QSPI <- TPU
    .miso_valid(qspi_miso_valid),

    // Logic Domain
    .clk_fast(chip_clk),
    .rst_fast(asyn_rst),
    .rdata(qspi_rdata),      // Host <- QSPI <- TPU
    .rvalid(qspi_rvalid),
    .ren(qspi_ren),
    .wdata(qspi_wdata),      // Host -> QSPI -> TPU
    .wen(qspi_wen),
    .addr(qspi_addr),       // Host -> QSPI -> TPU

    // FIFO Check
    .mosi_fifo_wfull(),
    .mosi_fifo_rempty(),
    .miso_fifo_wfull(),
    .miso_fifo_rempty()
);

////////////////////////////////////
//           INTERFACE            //
////////////////////////////////////


logic       [`INTERFACE_ADDR_WIDTH-1:0]              interface_addr;//保证进来的时序是干净的

logic                                                interface_wen;
logic       [`INTERFACE_DATA_WIDTH-1:0]              interface_wdata;

logic                                                interface_ren;
logic       [`INTERFACE_DATA_WIDTH-1:0]              interface_rdata;
logic                                                interface_rvalid;


always_ff @(posedge chip_clk or negedge asyn_rst_n) begin //打拍考虑布线timing
    if(~asyn_rst_n)begin
        interface_addr <= 0;
    end
    else if(spi_wen || spi_ren)begin
        interface_addr <= spi_addr;
    end
    else if(qspi_ren || qspi_wen)begin //qspi has low priority
        interface_addr <= qspi_addr;
    end
end

always_ff @(posedge chip_clk or negedge asyn_rst_n) begin 
    if(~asyn_rst_n)begin
        interface_wen <= 0;
        interface_wdata <= 0;
    end
    else if(spi_wen)begin
        interface_wen <= 1;
        interface_wdata <= spi_wdata;
    end
    else if(qspi_wen)begin //qspi has low priority
        interface_wen <= 1;
        interface_wdata <= qspi_wdata;
    end
    else begin
        interface_wen <= 0;
    end

end

always_ff @(posedge chip_clk or negedge asyn_rst_n) begin 
    if(~asyn_rst_n)begin
        interface_ren <= 0;
    end
    else if(spi_ren)begin
        interface_ren <= 1;
    end
    else if(qspi_ren)begin //qspi has low priority
        interface_ren <= 1;
    end
    else begin
        interface_ren <= 0;
    end
end

always_ff @(posedge chip_clk or negedge asyn_rst_n) begin 
    if(~asyn_rst_n)begin
        spi_rdata <= 0;
        spi_rvalid <= 0;
        qspi_rvalid <= 0;
        qspi_rdata <= 0;
    end
    else if(interface_rvalid)begin
        spi_rvalid <= 1;
        spi_rdata <= interface_rdata;
        qspi_rvalid <= 1;
        qspi_rdata <= interface_rdata;
    end
    else begin
        spi_rvalid <= 0;
        qspi_rvalid <= 0;
    end
end


array_top array_top_inst(
    .clk(chip_clk),
    .rst_n(asyn_rst_n),

    //There should be the ports to initial the cfg, wmem and input data from SPI or someting else
    .interface_addr(interface_addr), //保证进来的时序是干净的

    .interface_wen(interface_wen),
    .interface_wdata(interface_wdata),

    .interface_ren(interface_ren),
    .interface_rdata(interface_rdata),
    .interface_rvalid(interface_rvalid),

    //STATE PIN
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
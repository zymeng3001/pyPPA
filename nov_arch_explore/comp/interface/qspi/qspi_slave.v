// +FHDR========================================================================
//  File Name:      qspi_slave.v
//                  Shiwei Liu (liusw18@gmail.com)
//  Organization:
//  Description:
//     QSPI Top Module with CDC Logic
// -FHDR========================================================================

module qspi_slave #(
    parameter   DW = 16,
    parameter   CTRL_AW = 22,
    parameter   MOSI_FIFO_AW = 8,
    parameter   MISO_FIFO_AW = 8 
)(
    // QSPI Domain
    input                       clk_slow,
    input                       rst_slow,
    input       [DW-1:0]        mosi,       // Host -> QSPI -> TPU
    input                       mosi_valid,
    output      [DW-1:0]        miso,       // Host <- QSPI <- TPU
    output  reg                 miso_valid,

    // Logic Domain
    input                       clk_fast,
    input                       rst_fast,
    input       [DW-1:0]        rdata,      // Host <- QSPI <- TPU
    input                       rvalid,
    output                      ren,
    output      [DW-1:0]        wdata,      // Host -> QSPI -> TPU
    output                      wen,
    output      [CTRL_AW-1:0]   addr,       // Host -> QSPI -> TPU

    // FIFO Check
    output                      mosi_fifo_wfull,
    output                      mosi_fifo_rempty,
    output                      miso_fifo_wfull,
    output                      miso_fifo_rempty
);

// =============================================================================
// Sample and Algin

    reg     [DW-1:0]    mosi_neg, mosi_pos;
    reg                 mosi_valid_neg, mosi_valid_pos;

    always @(negedge clk_slow or posedge rst_slow) begin
        if (rst_slow) begin
            mosi_neg <= 'd0;
            mosi_valid_neg <= 1'b0;
        end
        else begin
            mosi_neg <= mosi;
            mosi_valid_neg <= mosi_valid;
        end 
    end

    always @(posedge clk_slow or posedge rst_slow) begin
        if (rst_slow) begin
            mosi_pos <= 'd0;
            mosi_valid_pos <= 1'b0;
        end
        else begin
            mosi_pos <= mosi_neg;
            mosi_valid_pos <= mosi_valid_neg;
        end
    end

// =============================================================================
// CDC via Async_FIFO

    // QSPI -> TPU
    wire    [DW-1:0]    mosi_fifo;
    reg                 mosi_valid_fifo;

    async_fifo #(.DW(DW), .AW(MOSI_FIFO_AW)) mosi_fifo_inst (
        .wclk                   (clk_slow),
        .wrst                   (rst_slow),
        .wen                    (mosi_valid_pos),
        .wdata                  (mosi_pos),
        .wfull                  (mosi_fifo_wfull),

        .rclk                   (clk_fast),
        .rrst                   (rst_fast),
        .ren                    (~mosi_fifo_rempty),
        .rdata                  (mosi_fifo),
        .rempty                 (mosi_fifo_rempty)
    );

    always @(posedge clk_fast or posedge rst_fast) begin
        if (rst_fast) begin
            mosi_valid_fifo <= 1'b0;
        end
        else begin
            mosi_valid_fifo <= ~mosi_fifo_rempty;
        end
    end

    // QSPI <-TPU
    wire    [DW-1:0]    miso_fifo;
    wire                miso_valid_fifo;

    async_fifo #(.DW(DW), .AW(MISO_FIFO_AW)) miso_fifo_inst (
        .wclk                   (clk_fast),
        .wrst                   (rst_fast),
        .wen                    (miso_valid_fifo),
        .wdata                  (miso_fifo),
        .wfull                  (miso_fifo_wfull),

        .rclk                   (clk_slow),
        .rrst                   (rst_slow),
        .ren                    (~miso_fifo_rempty),
        .rdata                  (miso),
        .rempty                 (miso_fifo_rempty)
    );

    always @(posedge clk_slow or posedge rst_slow) begin
        if (rst_slow) begin
            miso_valid <= 1'b0;
        end
        else begin
            miso_valid <= ~miso_fifo_rempty;
        end
    end

// =============================================================================
// QSPI Controller

    qspi_controller #(.DW(DW), .AW(CTRL_AW)) qspi_inst (
        .clk                    (clk_fast),
        .rst                    (rst_fast),

        .mosi                   (mosi_fifo),        // From FIFO
        .mosi_valid             (mosi_valid_fifo),
        .miso                   (miso_fifo),        // To   FIFO
        .miso_valid             (miso_valid_fifo),

        .rdata                  (rdata),
        .rvalid                 (rvalid),
        .ren                    (ren),
        .wdata                  (wdata),
        .wen                    (wen),
        .addr                   (addr)
    );

endmodule
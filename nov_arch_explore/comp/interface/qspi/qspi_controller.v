// +FHDR========================================================================
//  File Name:      qspi_controller.v
//                  Shiwei Liu (liusw18@gmail.com)
//  Organization:
//  Description:
//      A Modified and Simplied QSPI Slave (16-Bit Data and 1-Bit Valid/Chip Section). 
// -FHDR========================================================================

module qspi_controller #(
    parameter   DW = 16,    // Data Bitwidth
    parameter   AW = 22     // Address Bitwidth
)(
    // Global Signals
    input                   clk,
    input                   rst,

    // Logic Domain
    input       [DW-1:0]    rdata,      // SPI <- TPU
    input                   rvalid,
    output  reg             ren,
    output  reg [DW-1:0]    wdata,      // SPI -> TPU
    output  reg             wen,
    output  reg [AW-1:0]    addr,       // SPI -> TPU

    // QSPI Domain
    input       [DW-1:0]    mosi,       // Host -> QSPI
    input                   mosi_valid,
    output  reg [DW-1:0]    miso,
    output  reg             miso_valid  // Host <- QSPI
);

    parameter       QSPI_IDLE  = 4'b0001,   // Upload Command
                    QSPI_ADDR  = 4'b0010,   // Upload Address
                    QSPI_READ  = 4'b0100,   // Host -> QSPI -> TPU
                    QSPI_WRITE = 4'b1000;   // Host <- QSPI <- TPU
    reg     [3:0]   qspi_state;

    reg     [1:0]   cmd;            // 2'b01: QSPI <- TPU, 2'b10: QSPI -> TPU
    reg     [7:0]   burst_cnt;   // 0-255
    reg     [7:0]   rvalid_cnt;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            qspi_state <= 4'b0;
            cmd        <= 2'b0;
            burst_cnt  <= 8'b0;
            rvalid_cnt <= 8'b0;
            miso       <= 'd0;
            miso_valid <= 1'b0;
            addr       <= 'd0;
            ren        <= 1'b0;
            wdata      <= 'd0;
            wen        <= 1'b0;
        end
        else begin
            case (qspi_state)
                QSPI_IDLE: begin
                    miso_valid <= 1'b0;
                    ren        <= 1'b0;
                    wen        <= 1'b0;
                    if (mosi_valid) begin
                        qspi_state <= QSPI_ADDR;
                        addr[5:0]  <= mosi[15:10];  // Non Conigurable Here
                        cmd        <= mosi[9:8];
                        burst_cnt  <= mosi[7:0];
                    end
                end
                QSPI_ADDR: begin
                    if (mosi_valid) begin
                        addr[21:6] <= mosi[15:0];   // Non Conigurable Here
                        if (cmd == 2'b10) begin     // QSPI -> TPU
                            qspi_state <= QSPI_WRITE;
                        end
                        else begin  // QSPI <- TPU
                            qspi_state <= QSPI_READ;
                            rvalid_cnt <= burst_cnt;
                            ren        <= 1'b1;
                        end
                    end
                end
                QSPI_READ: begin
                    miso_valid <= rvalid;   // RVALID
                    if (rvalid) begin
                        miso <= rdata;
                        if (rvalid_cnt == 8'b0) begin
                            qspi_state <= QSPI_IDLE;
                        end
                        else begin
                            rvalid_cnt <= rvalid_cnt - 1'b1;
                        end
                    end
                    if (burst_cnt == 8'b0) begin    // REN
                        ren <= 1'b0;
                    end
                    else begin
                        burst_cnt <= burst_cnt - 1'b1;
                        addr      <= addr + 1'b1;
                    end
                end
                QSPI_WRITE: begin
                    wen <= mosi_valid;
                    if (mosi_valid) begin
                        wdata <= mosi;
                        if (burst_cnt == 8'b0) begin
                            qspi_state <= QSPI_IDLE;
                        end
                        else begin
                            burst_cnt <= burst_cnt - 1'b1;
                        end
                    end
                    if (wen) begin
                        addr <= addr + 1'b1;
                    end
                end
                default: begin
                    qspi_state <= QSPI_IDLE;
                end
            endcase
        end
    end

endmodule
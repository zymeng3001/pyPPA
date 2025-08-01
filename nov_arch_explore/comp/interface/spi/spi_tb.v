// +FHDR========================================================================
//  File Name:      spi_tb.v
//                  Shiwei Liu (liusw18@gmail.com)
//  Organization:
//  Description:
//      SPI Testbench
// -FHDR========================================================================

`timescale 1ns/10ps

module spi_tb;

// =============================================================================
// 1. Clk and Rst

    parameter   CLK_HOST_HALF_CYCLE = 10; //这里的HOST CLK 是FPGA的逻辑CLK，CHIP CLK 是 CHIP的逻辑CLK，那一定要求是4倍关系吗
    parameter   CLK_CHIP_HALF_CYCLE = 2.5; 

    reg         host_clk;
    reg         host_rst;

    reg         chip_clk;
    reg         chip_rst;

    initial begin
        host_rst = 1;
        chip_rst = 1;
        #(CLK_HOST_HALF_CYCLE * 2);
        host_rst = 0;
        chip_rst = 0;
    end

    initial begin
        host_clk = 1;
        chip_clk = 1;
    end

    always #CLK_HOST_HALF_CYCLE     host_clk = ~host_clk;
    always #CLK_CHIP_HALF_CYCLE     chip_clk = ~chip_clk;

// =============================================================================
// 2. Host and SPI

    parameter   SW = 38;
    parameter   CW = 2;
    parameter   AW = 19;
    parameter   DW = 16;
    parameter   CNT = 6;

    reg     [SW-1:0]    tx_mosi_data;
    reg                 tx_mosi_enable;
    wire                tx_mosi_done;

    wire                spi_clk;
    wire                spi_csn;
    wire                spi_mosi;
    
    spi_master  #(.SW(SW))  master_inst (
        .host_clk           (host_clk),
        .host_rst           (host_rst),

        .tx_mosi_data       (tx_mosi_data),
        .tx_mosi_enable     (tx_mosi_enable),
        .tx_mosi_done       (tx_mosi_done),

        .spi_clk            (spi_clk),
        .spi_csn            (spi_csn),
        .spi_mosi           (spi_mosi)
    );

    reg     [DW-1:0]    miso_data;
    reg                 miso_data_valid;
    wire    [AW-1:0]    mosi_addr;
    wire                mosi_addr_valid;
    wire                mosi_wen;
    wire                mosi_ren;
    wire    [DW-1:0]    mosi_data;
    wire                mosi_data_valid;

    wire                spi_miso;

    spi_slave   #(.CW(CW), .AW(AW), .DW(DW), .CNT(CNT)) slave_inst (
        .clk                (chip_clk),
        .rst                (chip_rst),

        .miso_data          (miso_data),
        .miso_data_valid    (miso_data_valid),
        .mosi_addr          (mosi_addr),
        .mosi_addr_valid    (mosi_addr_valid),
        .mosi_wen           (mosi_wen),
        .mosi_ren           (mosi_ren),
        .mosi_data          (mosi_data),
        .mosi_data_valid    (mosi_data_valid),

        .spi_clk            (spi_clk),
        .spi_csn            (spi_csn),
        .spi_mosi           (spi_mosi),
        .spi_miso           (spi_miso)
    );

    reg     [DW-1:0]    mem [0:3];

    always @(posedge chip_clk) begin
        if (mosi_wen) begin
            mem[mosi_addr[1:0]] <= mosi_data;
        end
    end

    reg     [DW-1:0]    data_d0;
    reg                 data_valid_d0;
    
    always @(posedge chip_clk or posedge chip_rst) begin
        if (chip_rst) begin
            data_d0 <= 'd0;
        end
        else if (mosi_ren) begin
            data_d0 <= mem[mosi_addr[1:0]];
        end
    end

    always @(posedge chip_clk or posedge chip_rst) begin
        if (chip_rst) begin
            miso_data <= 'd0;
        end
        else begin
            miso_data <= data_d0;
        end
    end

    always @(posedge chip_clk or posedge chip_rst) begin
        if (chip_rst) begin
            {miso_data_valid, data_valid_d0} <= 2'd0;
        end
        else begin
            {miso_data_valid, data_valid_d0} <= {data_valid_d0, mosi_ren};
        end
    end

// =============================================================================
// 3. Host and SPI

    initial begin
        tx_mosi_enable = 1'b0;
        #(CLK_HOST_HALF_CYCLE * 2);

        // Write 0
        #(CLK_HOST_HALF_CYCLE * 8);
        tx_mosi_enable = 1'b1;
        tx_mosi_data   = (38'b10 << 36) + (38'd0 << 17) + (16'h1234);
        #(CLK_HOST_HALF_CYCLE * 77 * 2);
        tx_mosi_enable = 1'b0;

        // Write 1
        #(CLK_HOST_HALF_CYCLE * 8);
        tx_mosi_enable = 1'b1;
        tx_mosi_data   = (38'b10 << 36) + (38'd1 << 17) + (16'h5678);
        #(CLK_HOST_HALF_CYCLE * 77 * 2);
        tx_mosi_enable = 1'b0;

        // Write 2
        #(CLK_HOST_HALF_CYCLE * 8);
        tx_mosi_enable = 1'b1;
        tx_mosi_data   = (38'b10 << 36) + (38'd2 << 17) + (16'h4321);
        #(CLK_HOST_HALF_CYCLE * 77 * 2);
        tx_mosi_enable = 1'b0;

        // Write 3
        #(CLK_HOST_HALF_CYCLE * 8);
        tx_mosi_enable = 1'b1;
        tx_mosi_data   = (38'b10 << 36) + (38'd3 << 17) + (16'h8765);
        #(CLK_HOST_HALF_CYCLE * 77 * 2);
        tx_mosi_enable = 1'b0;

        // Read 0
        #(CLK_HOST_HALF_CYCLE * 8);
        tx_mosi_enable = 1'b1;
        tx_mosi_data   = (38'b01 << 36) + (38'd0 << 17) + (16'h1234);
        #(CLK_HOST_HALF_CYCLE * 77 * 2);
        tx_mosi_enable = 1'b0;

        // Read 1
        #(CLK_HOST_HALF_CYCLE * 8);
        tx_mosi_enable = 1'b1;
        tx_mosi_data   = (38'b01 << 36) + (38'd1 << 17) + (16'h5678);
        #(CLK_HOST_HALF_CYCLE * 77 * 2);
        tx_mosi_enable = 1'b0;

        // Read 2
        #(CLK_HOST_HALF_CYCLE * 8);
        tx_mosi_enable = 1'b1;
        tx_mosi_data   = (38'b01 << 36) + (38'd2 << 17) + (16'h4321);
        #(CLK_HOST_HALF_CYCLE * 77 * 2);
        tx_mosi_enable = 1'b0;

        // Read 3
        #(CLK_HOST_HALF_CYCLE * 8);
        tx_mosi_enable = 1'b1;
        tx_mosi_data   = (38'b01 << 36) + (38'd3 << 17) + (16'h8765);
        #(CLK_HOST_HALF_CYCLE * 77 * 2);
        tx_mosi_enable = 1'b0;

        // Finsih
        #(CLK_HOST_HALF_CYCLE * 8);
        $finish;

    end

endmodule
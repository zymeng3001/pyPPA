// +FHDR========================================================================
//  License:
//      Copyright (c) 2022 Authors. All rights reserved.
// =============================================================================
//  File Name:      spi_slave.v
//                  Shiwei Liu (liusw18@fudan.edu.com)
//  Organization:
//  Description:
//      SPI Slave (Only Support Mode 0), CLK >= 4*SCLK
// -FHDR========================================================================
`timescale 1ns/1ns

module spi_slave #(
    parameter   CW = 2,     // Command Width: 10 for Write, 01 for Read
    parameter   AW = 19,    // Address Width
    parameter   DW = 16,    // Data    Width
    parameter   CNT = 6     // Counter Width
)(
    // DLA Domain
    input                   clk,
    input                   rst,
    input       [DW-1:0]    miso_data,
    input                   miso_data_valid,
    output  reg [AW-1:0]    mosi_addr,
    output  reg             mosi_addr_valid,
    output                  mosi_wen,
    output                  mosi_ren,
    output  reg [DW-1:0]    mosi_data,
    output  reg             mosi_data_valid,

    // SPI Domain
    input                   spi_clk,
    input                   spi_csn,    // Activate Low for SPI Enable
    input                   spi_mosi,
    output                  spi_miso
);

    localparam  SW = CW + AW + DW + 1;

// =============================================================================
// MOSI: FPGA -> SPI -> DLA

    // 1. Positive Edge Sampling
    reg     [CNT-1:0]   rx_bit_cnt;

    always @(posedge spi_clk or posedge spi_csn) begin
        if (spi_csn) begin
            rx_bit_cnt <= 'd0;
        end
        else begin
            if (rx_bit_cnt == SW-1) begin
                rx_bit_cnt <= 'd0;
            end
            else begin
                rx_bit_cnt <= rx_bit_cnt + 1'b1;
            end
        end
    end

    // 2. Serial to Parallel
    reg     [SW-1:0]    rx_byte_buf;
    
    always @(posedge spi_clk or posedge spi_csn) begin
        if (spi_csn) begin
            rx_byte_buf <= 'd0;
        end
        else begin
            rx_byte_buf <= {rx_byte_buf[SW-2:0], spi_mosi};
        end
    end

    // 3. Address Valid
    reg                 rx_addr_valid;

    always @(posedge spi_clk or posedge spi_csn) begin
        if (spi_csn) begin
            rx_addr_valid <= 1'b0;
        end
        else begin
            if (rx_bit_cnt == CW+AW-1) begin
                rx_addr_valid <= 1'b1;
            end
            else begin
                rx_addr_valid <= 1'b0;
            end
        end
    end

    // 4. Data Valid
    reg                 rx_data_valid;

    always @(posedge spi_clk or posedge spi_csn) begin
        if (spi_csn) begin
            rx_data_valid <= 1'b0;
        end
        else begin
            if (rx_bit_cnt == SW-1) begin
                rx_data_valid <= 1'b1;
            end
            else begin
                rx_data_valid <= 1'b0;
            end
        end
    end

    // 5. SPI Domain to DLA Domain
    reg     rx_addr_valid_d0, rx_addr_valid_d1;
    reg     rx_data_valid_d0, rx_data_valid_d1;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            {rx_addr_valid_d1, rx_addr_valid_d0} <= 2'b0;
            {rx_data_valid_d1, rx_data_valid_d0} <= 2'b0;
        end
        else begin
            {rx_addr_valid_d1, rx_addr_valid_d0} <= {rx_addr_valid_d0, rx_addr_valid};
            {rx_data_valid_d1, rx_data_valid_d0} <= {rx_data_valid_d0, rx_data_valid};
        end
    end

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            mosi_addr <= 'd0;
            mosi_addr_valid <= 1'b0;
        end
        else if (~rx_addr_valid_d1 && rx_addr_valid_d0) begin
            mosi_addr <= rx_byte_buf[AW-1:0];
            mosi_addr_valid <= 1'b1;
        end
        else begin
            mosi_addr_valid <= 1'b0;
        end
    end

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            mosi_data <= 'd0;
            mosi_data_valid <= 1'b0;
        end
        else if (~rx_data_valid_d1 && rx_data_valid_d0) begin
            mosi_data <= rx_byte_buf[DW-1:0];
            mosi_data_valid <= 1'b1;
        end
        else begin
            mosi_data_valid <= 1'b0;
        end
    end

    // 6. CMD
    reg     [CW-1:0]    mosi_cmd;
    reg                 mosi_cmd_valid;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            mosi_cmd <= 'd0;
            mosi_cmd_valid <= 1'b0;
        end
        else if (~rx_addr_valid_d1 && rx_addr_valid_d0) begin
            mosi_cmd <= rx_byte_buf[AW+CW-1:AW];
            mosi_cmd_valid <= 1'b1;
        end
        else begin
            mosi_cmd_valid <= 1'b0;
        end
    end


// =============================================================================
// MOSI: Address and Command

    assign  mosi_wen = mosi_cmd == 2'b10 && mosi_data_valid;
    assign  mosi_ren = mosi_cmd == 2'b01 && mosi_cmd_valid;

// =============================================================================
// MISO: FPGA <- SPI <- DLA

    // 1. DLA Domain
    reg     [DW-1:0]    tx_byte_buf;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            tx_byte_buf <= 'd0;
        end
        else if (miso_data_valid) begin
            tx_byte_buf <= miso_data;
        end
    end

    // 2. SPI Domain
    reg     [CNT-1:0]    tx_bit_cnt;

    always @(negedge spi_clk or posedge spi_csn) begin
        if (spi_csn) begin
            tx_bit_cnt <= SW - 1;
        end
        else begin
            if (tx_bit_cnt == 0) begin
                tx_bit_cnt <= SW - 1;
            end
            else begin
                tx_bit_cnt <= tx_bit_cnt - 1'b1;
            end
        end
    end

    // 3. SPI MISO Enable 
    reg     spi_miso_enable;

    always @(negedge spi_clk or posedge spi_csn) begin
        if (spi_csn) begin
            spi_miso_enable <= 1'b0;
        end
        else begin
            if (tx_bit_cnt <= DW) begin
                spi_miso_enable <= 1'b1;
            end
            else begin
                spi_miso_enable <= 1'b0;
            end
        end
    end

    wire    [3:0]   tx_addr;
    assign  tx_addr = tx_bit_cnt[3:0];

    assign  spi_miso = spi_miso_enable && mosi_cmd == 2'b01 ? tx_byte_buf[tx_addr] : 1'bz;

endmodule
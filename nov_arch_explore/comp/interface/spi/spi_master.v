// +FHDR========================================================================
//  File Name:      spi_master.v
//                  Shiwei Liu (liusw18@gmail.com)
//  Organization:
//  Description:
//      SPI Master (Only Support Mode 0). Ugly Code, Only for SPI Test.
// -FHDR========================================================================

module spi_master #(
    parameter   SW = 38
)(
    // Host Domain
    input               host_clk,
    input               host_rst,

    // Host Flag
    input   [SW-1:0]    tx_mosi_data,
    input               tx_mosi_enable,
    output  reg         tx_mosi_done,

    //output  reg [15:0]  rx_mosi_data,
    //output  reg         rx_mosi_valid,
    //output  reg         rx_miso_done,

    // SPI Domain
    output  reg         spi_clk,
    output              spi_csn,
    //input               spi_miso, //为什么这个被注释了
    output  reg         spi_mosi
);

    reg     [6:0]   host_spi_state;
    reg             csn;

    assign  spi_csn = csn && tx_mosi_data;

    always @(posedge host_clk or posedge host_rst) begin
        if (host_rst) begin
            host_spi_state <= 7'd0;
            spi_clk        <= 1'b0;
            csn            <= 1'b1;
        end
        else if (tx_mosi_enable) begin

            if (host_spi_state == 76) begin
                host_spi_state <= 0;
            end
            else begin
                host_spi_state <= host_spi_state + 1'b1;
            end

            if (host_spi_state == 76) begin
                csn <= 1'b1;
            end
            else begin
                csn <= 1'b0;
            end

            if (host_spi_state == 75) begin
                tx_mosi_done <= 1'b1;
            end
            else begin
                tx_mosi_done <= 1'b0;
            end

            case (host_spi_state)
                1 ,3 ,5 ,7 ,9 ,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,
                41,43,45,47,49,51,53,55,57,59,61,63,65,67,69,71,73,75: begin
                    spi_clk <= 1'b1;
                    spi_mosi <= spi_mosi;
                end
                76: begin
                    spi_clk  <= 1'b0;
                    spi_mosi <= tx_mosi_data[37];
                end
                0: begin
                    spi_clk  <= 1'b0;
                    spi_mosi <= tx_mosi_data[37];
                end
                2: begin
                    spi_clk  <= 1'b0;
                    spi_mosi <= tx_mosi_data[36];
                end
                4: begin
                    spi_clk  <= 1'b0;
                    spi_mosi <= tx_mosi_data[35];
                end
                6: begin
                    spi_clk  <= 1'b0;
                    spi_mosi <= tx_mosi_data[34];
                end
                8: begin
                    spi_clk  <= 1'b0;
                    spi_mosi <= tx_mosi_data[33];
                end
                10: begin
                    spi_clk  <= 1'b0;
                    spi_mosi <= tx_mosi_data[32];
                end
                12: begin
                    spi_clk  <= 1'b0;
                    spi_mosi <= tx_mosi_data[31];
                end
                14: begin
                    spi_clk  <= 1'b0;
                    spi_mosi <= tx_mosi_data[30];
                end
                16: begin
                    spi_clk  <= 1'b0;
                    spi_mosi <= tx_mosi_data[29];
                end
                18: begin
                    spi_clk  <= 1'b0;
                    spi_mosi <= tx_mosi_data[28];
                end
                20: begin
                    spi_clk  <= 1'b0;
                    spi_mosi <= tx_mosi_data[27];
                end
                22: begin
                    spi_clk  <= 1'b0;
                    spi_mosi <= tx_mosi_data[26];
                end
                24: begin
                    spi_clk  <= 1'b0;
                    spi_mosi <= tx_mosi_data[25];
                end
                26: begin
                    spi_clk  <= 1'b0;
                    spi_mosi <= tx_mosi_data[24];
                end
                28: begin
                    spi_clk  <= 1'b0;
                    spi_mosi <= tx_mosi_data[23];
                end
                30: begin
                    spi_clk  <= 1'b0;
                    spi_mosi <= tx_mosi_data[22];
                end
                32: begin
                    spi_clk  <= 1'b0;
                    spi_mosi <= tx_mosi_data[21];
                end
                34: begin
                    spi_clk  <= 1'b0;
                    spi_mosi <= tx_mosi_data[20];
                end
                36: begin
                    spi_clk  <= 1'b0;
                    spi_mosi <= tx_mosi_data[19];
                end
                38: begin
                    spi_clk  <= 1'b0;
                    spi_mosi <= tx_mosi_data[18];
                end
                40: begin
                    spi_clk  <= 1'b0;
                    spi_mosi <= tx_mosi_data[17];
                end
                42: begin
                    spi_clk  <= 1'b0;
                    spi_mosi <= tx_mosi_data[16];
                end
                44: begin
                    spi_clk  <= 1'b0;
                    spi_mosi <= tx_mosi_data[15];
                end
                46: begin
                    spi_clk  <= 1'b0;
                    spi_mosi <= tx_mosi_data[14];
                end
                48: begin
                    spi_clk  <= 1'b0;
                    spi_mosi <= tx_mosi_data[13];
                end
                50: begin
                    spi_clk  <= 1'b0;
                    spi_mosi <= tx_mosi_data[12];
                end
                52: begin
                    spi_clk  <= 1'b0;
                    spi_mosi <= tx_mosi_data[11];
                end
                54: begin
                    spi_clk  <= 1'b0;
                    spi_mosi <= tx_mosi_data[10];
                end
                56: begin
                    spi_clk  <= 1'b0;
                    spi_mosi <= tx_mosi_data[9];
                end
                58: begin
                    spi_clk  <= 1'b0;
                    spi_mosi <= tx_mosi_data[8];
                end
                60: begin
                    spi_clk  <= 1'b0;
                    spi_mosi <= tx_mosi_data[7];
                end
                62: begin
                    spi_clk  <= 1'b0;
                    spi_mosi <= tx_mosi_data[6];
                end
                64: begin
                    spi_clk  <= 1'b0;
                    spi_mosi <= tx_mosi_data[5];
                end
                66: begin
                    spi_clk  <= 1'b0;
                    spi_mosi <= tx_mosi_data[4];
                end
                68: begin
                    spi_clk  <= 1'b0;
                    spi_mosi <= tx_mosi_data[3];
                end
                70: begin
                    spi_clk  <= 1'b0;
                    spi_mosi <= tx_mosi_data[2];
                end
                72: begin
                    spi_clk  <= 1'b0;
                    spi_mosi <= tx_mosi_data[1];
                end
                74: begin
                    spi_clk  <= 1'b0;
                    spi_mosi <= tx_mosi_data[0];
                end
                default: begin
                    spi_clk  <= 1'b0;
                    spi_mosi <= 1'b0;
                end
            endcase
        end
    end

endmodule
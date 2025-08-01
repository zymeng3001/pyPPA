//chip clk 最好是SPI CLK的10倍以上
//读响应时间要充足的小于1.5 SPI_CLK_CYCLES
//spi指令同步到chip clk需要大概3 clk cycles
//control_reg read chip latency: spi_ren -> spi_rvalid 4
//inst_reg read chip latency: spi_ren -> spi_rvalid 3
//state_reg read chip latency: spi_ren -> spi_rvalid 3
//hear sram read chip latency: spi_ren -> spi_rvalid 7
//wmem and kv cache chip latency: spi_ren -> spi_rvalid 9
//global mem chip latency: spi_ren -> spi_rvalid 5
//residual mem chip latency: spi_ren -> spi_rvalid 5
module tb();

logic                chip_clk;
logic                fpga_clk;
logic                asyn_rst;       //rst!, not rst_n
// SPI Domain
logic                spi_clk;       
logic                spi_csn;        // SPI Active Low
logic                spi_mosi;       // Host -> SPI
logic                spi_miso;       // Host <- SPI

logic                qspi_clk;       //qspi clk
logic [15:0]         qspi_mosi;       // Host -> QSPI -> TPU
logic                qspi_mosi_valid;
logic [15:0]         qspi_miso;       // Host <- QSPI <- TPU
logic                qspi_miso_valid;

//FPGA Domain
logic                                                              spi_start; //looks should be in same phase with spi_tx_data
logic                                                              spi_complete;
logic [`INTERFACE_DATA_WIDTH + `INTERFACE_ADDR_WIDTH+2 + 1-1:0]    spi_tx_data;
logic [`INTERFACE_DATA_WIDTH-1:0]                                  spi_rx_data;
logic                                                              spi_rx_valid;


parameter real  CHIP_CLK_FREQ  = 500e6; //500M
parameter real  FPGA_CLK_FREQ = 100e6;
parameter real  QSPI_CLK_FREQ = 60e6;

//时间刻度和精度都是1ps
initial begin
    chip_clk = 0;
    #100 fpga_clk = 0; //给不同的相位
    #450 qspi_clk = 0;
end
//时间刻度和精度都是1ps
always begin
    #(1e12/CHIP_CLK_FREQ/2);
    chip_clk = ~chip_clk;
end

always begin
    #(1e12/FPGA_CLK_FREQ/2);
    fpga_clk = ~fpga_clk;
end

always begin
    #(1e12/QSPI_CLK_FREQ/2);
    qspi_clk = ~qspi_clk;
end

top top(
    .chip_clk(chip_clk),
    .asyn_rst(asyn_rst), //rst!, not rst_n

    .spi_clk(spi_clk),        
    .spi_csn(spi_csn),        // SPI Active Low
    .spi_mosi(spi_mosi),       // Host -> SPI
    .spi_miso(spi_miso),       // Host <- SPI

    .qspi_clk(qspi_clk),        //qspi clk
    .qspi_mosi(qspi_mosi),       // Host -> QSPI -> TPU
    .qspi_mosi_valid(qspi_mosi_valid),
    .qspi_miso(qspi_miso),       // Host <- QSPI <- TPU
    .qspi_miso_valid(qspi_miso_valid)
);

//{cmd[1:0], addr[21:0], NA, data[15:0]}
//cmd = 2 for write, cmd = 1 for read
host_spi #(
    .DW(`INTERFACE_DATA_WIDTH + `INTERFACE_ADDR_WIDTH+2 + 1),
    .TX(22), //useless
    .RX(`INTERFACE_DATA_WIDTH)
)host_spi (
    // Global Signals
    .clk(fpga_clk),
    .rst(asyn_rst),

    // Host Interface
    .spi_start(spi_start),
    .spi_complete(spi_complete),
    .spi_tx_data(spi_tx_data),
    .spi_rx_data(spi_rx_data),
    .spi_rx_valid(spi_rx_valid),

    // SPI Interface
    .spi_sck(spi_clk),
    .spi_csn(spi_csn),
    .spi_mosi(spi_mosi),
    .spi_miso(spi_miso)
);
task FPGA_SPI_WR(
    input logic [`INTERFACE_ADDR_WIDTH-1:0] addr,
    input logic [`INTERFACE_DATA_WIDTH-1:0] wdata
);
@(negedge fpga_clk);
spi_start = 1;
spi_tx_data = {2'b10, addr, 1'b0, wdata};
@(negedge fpga_clk);
spi_start = 0;
while (1)begin
    @(negedge fpga_clk);
    if(spi_complete)
        break;
end
endtask

task FPGA_SPI_RD(
    input  logic [`INTERFACE_ADDR_WIDTH-1:0] addr,
    output logic [`INTERFACE_DATA_WIDTH-1:0] rdata
);
@(negedge fpga_clk);
spi_start = 1;
spi_tx_data = {2'b01, addr, 1'b0, 16'b0};
@(negedge fpga_clk);
spi_start = 0;
while (1)begin
    @(negedge fpga_clk);
    if(spi_rx_valid)begin
        rdata = spi_rx_data;
        break;
    end
end
endtask

////////////////////////////////////////////////////////////
//                       MAIN                             //
////////////////////////////////////////////////////////////
 logic                                                new_token; //pulse, should be high after global sram accept all the content of the new token
 logic                                                user_first_token; //be in same phase
 logic       [`USER_ID_WIDTH-1:0]                     user_id;// be in same phase

logic                                                 clean_kv_cache; //pulse
 logic       [`USER_ID_WIDTH-1:0]                     clean_kv_cache_user_id; 

logic [`INTERFACE_ADDR_WIDTH-1:0] addr;
logic [`INTERFACE_DATA_WIDTH-1:0] wdata;
logic [`INTERFACE_DATA_WIDTH-1:0] rdata;
int head_sram_wr_done = 0;
int kv_cache_wr_done = 0;
int wmem_wr_done = 0;
int global_sram_wr_done = 0;
int residual_sram_wr_done = 0;
int current_head = 0;
int current_core = 0;

initial begin
    asyn_rst = 1;
    spi_start = 0;
    spi_tx_data = 0;
    qspi_mosi_valid = 0;
    qspi_mosi = 0;

    repeat (1000) @(negedge fpga_clk);
    asyn_rst = 0;
    repeat (100) @(negedge fpga_clk);   
    addr = `STATE_REGISTERS_BASE_ADDR + 0;
    wdata = 16'h1234;
    FPGA_SPI_WR(addr, wdata);
    FPGA_SPI_RD(addr, rdata);
    $display("rdata = %0x",rdata);



    /////////////////////////////////////////////////
    //                NEW TOKEN                    //
    /////////////////////////////////////////////////
    //write
    @(negedge chip_clk);
    new_token = 1;
    user_first_token = 1;
    user_id = 0;
    addr = `INSTRUCTION_REGISTERS_BASE_ADDR;
    wdata = {12'b0 ,user_id ,user_first_token, new_token};
    FPGA_SPI_WR(addr, wdata);

    new_token = 1;
    user_first_token = 1;
    user_id = 1;
    addr = `INSTRUCTION_REGISTERS_BASE_ADDR;
    wdata = {12'b0 ,user_id ,user_first_token, new_token};
    FPGA_SPI_WR(addr, wdata);

    new_token = 0;
    user_first_token = 1;
    user_id = 2;
    addr = `INSTRUCTION_REGISTERS_BASE_ADDR;
    wdata = {12'b0 ,user_id ,user_first_token, new_token};
    FPGA_SPI_WR(addr, wdata);

    new_token = 1;
    user_first_token = 0;
    user_id = 3;
    addr = `INSTRUCTION_REGISTERS_BASE_ADDR;
    wdata = {12'b0 ,user_id ,user_first_token, new_token};
    FPGA_SPI_WR(addr, wdata);



    /////////////////////////////////////////////////
    //                CLEAN KV $                   //
    /////////////////////////////////////////////////
    clean_kv_cache = 1;
    clean_kv_cache_user_id = 0;
    addr = `INSTRUCTION_REGISTERS_BASE_ADDR + 1;
    wdata = {13'b0 ,clean_kv_cache_user_id, clean_kv_cache};
    FPGA_SPI_WR(addr, wdata);

    clean_kv_cache = 1;
    clean_kv_cache_user_id = 1;
    addr = `INSTRUCTION_REGISTERS_BASE_ADDR + 1;
    wdata = {13'b0 ,clean_kv_cache_user_id, clean_kv_cache};
    FPGA_SPI_WR(addr, wdata);

    clean_kv_cache = 0;
    clean_kv_cache_user_id = 2;
    addr = `INSTRUCTION_REGISTERS_BASE_ADDR + 1;
    wdata = {13'b0 ,clean_kv_cache_user_id, clean_kv_cache};
    FPGA_SPI_WR(addr, wdata);

    clean_kv_cache = 1;
    clean_kv_cache_user_id = 3;
    addr = `INSTRUCTION_REGISTERS_BASE_ADDR + 1;
    wdata = {13'b0 ,clean_kv_cache_user_id, clean_kv_cache};
    FPGA_SPI_WR(addr, wdata);

    //////////////////////////////////////////////////
    //                CONFIG INIT                   //
    //////////////////////////////////////////////////

    addr = `INSTRUCTION_REGISTERS_BASE_ADDR + 2;
    wdata = {15'b0, 1'b1};
    FPGA_SPI_WR(addr, wdata);

    repeat(100) @(negedge chip_clk);

    $finish();
end



always begin
    repeat(1000) @(negedge chip_clk);
    $display("Time: %t", $time());
end
endmodule
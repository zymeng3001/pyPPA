//chip clk 最好是SPI CLK的10倍以上
//读响应时间要充足的小于1.5 SPI_CLK_CYCLES
//spi指令同步到chip clk需要大概3 clk cycles
//control_reg read chip latency: spi_ren -> spi_rvalid 4
//inst_reg read chip latency: spi_ren -> spi_rvalid 3
//state_reg read chip latency: spi_ren -> spi_rvalid 3

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
parameter real  QSPI_CLK_FREQ = 50e6;

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
    input  logic [`INTERFACE_ADDR_WIDTH-1:0] addr,
    input  logic [`INTERFACE_DATA_WIDTH-1:0] wdata
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


task FPGA_QSPI_WR(
    input  logic [`INTERFACE_ADDR_WIDTH-1:0]         addr,
    input  logic [7:0]                               burst_cnt,//等于0传递1个数据，等于255传递256个数据
    input  logic [255:0][`INTERFACE_DATA_WIDTH-1:0]  wdata_array
);
    @(posedge qspi_clk);
    qspi_mosi_valid = 1;
    qspi_mosi = {addr[5:0], 2'b10, burst_cnt};
    @(posedge qspi_clk);
    qspi_mosi_valid = 1;
    qspi_mosi = addr[21:6];
    for(int i = 0; i < burst_cnt + 1; i++)begin
        @(posedge qspi_clk);
        qspi_mosi_valid = 1;
        qspi_mosi = wdata_array[i];
    end
    @(posedge qspi_clk);
    qspi_mosi_valid = 0;
    @(posedge qspi_clk);
endtask


task FPGA_QSPI_RD(
    input  logic [`INTERFACE_ADDR_WIDTH-1:0]         addr,
    input  logic [7:0]                               burst_cnt,//等于0传递1个数据，等于63传递64个数据
    output logic [255:0][`INTERFACE_DATA_WIDTH-1:0]  rdata_array
);
    automatic int temp = 0; //不能写成int temp = 0 temp是一个静态变量！！！，所以不会重复定义，导致数据冲突
    @(posedge qspi_clk);
    qspi_mosi_valid = 1;
    qspi_mosi = {addr[5:0], 2'b01, burst_cnt};
    @(posedge qspi_clk);
    qspi_mosi_valid = 1;
    qspi_mosi = addr[21:6];
    @(posedge qspi_clk);
    qspi_mosi_valid = 0;
    while(1)begin
        @(negedge qspi_clk);
        if(qspi_miso_valid)begin
            rdata_array[temp] = qspi_miso;
            temp++;
            if(temp == burst_cnt+1)
                break;
        end
    end
    @(negedge qspi_clk);
endtask
////////////////////////////////////////////////////////////
//                       MAIN                             //
////////////////////////////////////////////////////////////
logic [`INTERFACE_ADDR_WIDTH-1:0] addr;
logic [`INTERFACE_DATA_WIDTH-1:0] wdata;
logic [`INTERFACE_DATA_WIDTH-1:0] rdata;
logic [255:0][`INTERFACE_DATA_WIDTH-1:0] wdata_array;
logic [255:0][`INTERFACE_DATA_WIDTH-1:0] rdata_array;
logic [7:0] burst_cnt;
initial begin
    asyn_rst = 1;
    spi_start = 0;
    spi_tx_data = 0;
    qspi_mosi_valid = 0;
    qspi_mosi = 0;

    repeat (1000) @(negedge fpga_clk);
    asyn_rst = 0;
    repeat (100) @(negedge fpga_clk);   
    addr = `CONTROL_REGISTERS_BASE_ADDR + 0;
    for(int i =0; i < 256; i++)begin
        wdata_array[i] = 'h2024 + i;
    end
    burst_cnt = 63;
    FPGA_QSPI_WR(addr, burst_cnt, wdata_array);
    FPGA_QSPI_RD(addr, burst_cnt, rdata_array);
    for(int i =0; i < 256; i++)begin
        $display("RDATA[%0d] = 0X%0x",i,rdata_array[i]);
    end


    $finish();
end
always begin
    repeat(10000) @(negedge chip_clk);
    $display("Time: %t", $time());
end
endmodule
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
parameter real  FPGA_CLK_FREQ = 80e6;
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
    //                HEAD SRAM                    //
    /////////////////////////////////////////////////
    // head sram
    for(int i = 0; i < 256 * 8; i++)begin  //4KB
        addr = i + (1 << 21);
        wdata = i + 16'h2024;
        FPGA_SPI_WR(addr, wdata);
        // $display("WR[%0d]",i);
    end
    
    repeat(10) @(negedge  chip_clk);
    head_sram_wr_done = 1;
    @(negedge chip_clk)
    head_sram_wr_done = 0;


    for(int i = 0; i < 256 * 8; i++)begin
        addr = i + (1 << 21);
        FPGA_SPI_RD(addr, rdata);
        if(rdata !== i + 16'h2024)
            $display("Head SRAM [%0d] value wrong, Hoped value %4x != %4x.", i, i + 16'h2024, rdata);
    end

    /////////////////////////////////////////////////
    //                REGISTER                     //
    /////////////////////////////////////////////////
    //write
    for(int i = 0; i < 64; i++)begin
        addr = i + `CONTROL_REGISTERS_BASE_ADDR;
        wdata = (i + `CONTROL_REGISTERS_BASE_ADDR)%16'hFFFF;
        FPGA_SPI_WR(addr, wdata);
    end

    for(int i = 0; i < 4; i++)begin
        addr = i + `INSTRUCTION_REGISTERS_BASE_ADDR;
        wdata = (i + `INSTRUCTION_REGISTERS_BASE_ADDR)%16'hFFFF;
        FPGA_SPI_WR(addr, wdata);
    end

    for(int i = 0; i < 4; i++)begin
        addr = i + `STATE_REGISTERS_BASE_ADDR;
        wdata = (i + `STATE_REGISTERS_BASE_ADDR)%16'hFFFF;
        FPGA_SPI_WR(addr, wdata);
    end

    //read
    for(int i = 0; i < 64; i++)begin
        addr = i + `CONTROL_REGISTERS_BASE_ADDR;
        FPGA_SPI_RD(addr, rdata);
        if(rdata !== (i + `CONTROL_REGISTERS_BASE_ADDR)%16'hFFFF)
            $display("Control register [%0d] value wrong, Hoped value %4x != %4x.", i, (i + `CONTROL_REGISTERS_BASE_ADDR)%16'hFFFF, rdata);
    end

    for(int i = 0; i < 4; i++)begin
        addr = i + `INSTRUCTION_REGISTERS_BASE_ADDR;
        FPGA_SPI_RD(addr, rdata);
        if(rdata !== (i + `INSTRUCTION_REGISTERS_BASE_ADDR)%16'hFFFF)
            $display("Instruction register [%0d] value wrong, Hoped value %4x != %4x.", i, (i + `INSTRUCTION_REGISTERS_BASE_ADDR)%16'hFFFF, rdata);
    end

    for(int i = 0; i < 4; i++)begin
        addr = i + `STATE_REGISTERS_BASE_ADDR;
        FPGA_SPI_RD(addr, rdata);
        if(rdata !== (i + `STATE_REGISTERS_BASE_ADDR)%16'hFFFF)
            $display("State register [%0d] value wrong, Hoped value %4x != %4x.", i, (i + `STATE_REGISTERS_BASE_ADDR)%16'hFFFF, rdata);
    end



    /////////////////////////////////////////////////
    //                  KV CACHE                   //
    /////////////////////////////////////////////////

    // 'h40000 = 256K
    // 'h400 = 16K
    for(int h = 0; h < `HEAD_NUM; h++)begin
        for(int c = 0; c < `HEAD_CORE_NUM; c++)begin
            kv_cache_wr_done = 0;
            for(int i = 0; i < 4096; i++)begin  //8KB
                addr = h * 'h40000 + c * 'h4000 + i + 0;
                wdata = (addr + 16'h2024)%16'hFFFF;
                FPGA_SPI_WR(addr, wdata);
                // $display("WR[%0d]",i);
            end
            current_core = c;
            current_head = h;
            repeat(10) @(negedge chip_clk);
            kv_cache_wr_done = 1;
            head_sram_wr_done = 1; //多次检查
            @(negedge chip_clk);
            kv_cache_wr_done = 0;
            head_sram_wr_done = 0;

            for(int i = 0; i < 4096; i++)begin
                addr = h * 'h40000 + c * 'h4000 + i + 0;
                FPGA_SPI_RD(addr, rdata);
                if(rdata !== (addr + 16'h2024)%16'hFFFF)
                    $display("HEAD[%0d]CORE[%0d]KV $[%0d] value wrong, Hoped value %4x != %4x.",h, c, i, (addr + 16'h2024)%16'hFFFF, rdata);
            end

            $display("HEAD[%0d]CORE[%0d] KV $ finish.",h,c);
        end
    end




    /////////////////////////////////////////////////
    //                    WMEM                     //
    /////////////////////////////////////////////////

    // wmem
    // 'h40000 = 256K
    // 'h400 = 16K
    for(int h = 0; h < `HEAD_NUM; h++)begin
        for(int c = 0; c < `HEAD_CORE_NUM; c++)begin
            wmem_wr_done = 0;
            for(int i = 4096; i < 4096 *  4; i++)begin  //8-32KB
                addr = h * 'h40000 + c * 'h4000 + i + 0;
                wdata = (addr + 16'h2024)%16'hFFFF;
                FPGA_SPI_WR(addr, wdata);
                // $display("WR[%0d]",i);
            end
            current_core = c;
            current_head = h;
            repeat(10) @(negedge chip_clk);
            kv_cache_wr_done = 1;
            wmem_wr_done = 1;
            head_sram_wr_done = 1; //多次检查
            @(negedge chip_clk);
            kv_cache_wr_done = 0;
            wmem_wr_done = 0;
            head_sram_wr_done = 0;

            for(int i = 4096; i < 4096 *  4; i++)begin
                addr = h * 'h40000 + c * 'h4000 + i + 0;
                FPGA_SPI_RD(addr, rdata);
                if(rdata !== (addr + 16'h2024)%16'hFFFF)
                    $display("HEAD[%0d]CORE[%0d]WMEM[%0d] value wrong, Hoped value %4x != %4x.",h, c, i, (addr + 16'h2024)%16'hFFFF, rdata);
            end

            $display("HEAD[%0d]CORE[%0d] WMEM finish.",h,c);
        end
    end



    /////////////////////////////////////////////////
    //                 GLOBAL SRAM                 //
    /////////////////////////////////////////////////
    for(int i = 0; i < 256 ; i++)begin  //0.5KB
        addr = i + `GLOBAL_MEM_BASE_ADDR;
        wdata = (addr + 16'h2024)%16'hFFFF;
        FPGA_SPI_WR(addr, wdata);
        // $display("WR[%0d]",i);
    end
    
    repeat(10) @(negedge  chip_clk);
    global_sram_wr_done = 1;
    @(negedge chip_clk)
    global_sram_wr_done = 0;


    for(int i = 0; i < 256; i++)begin
        addr = i + `GLOBAL_MEM_BASE_ADDR;
        FPGA_SPI_RD(addr, rdata);
        if(rdata !== (addr + 16'h2024)%16'hFFFF)
            $display("Global mem [%0d] value wrong, Hoped value %4x != %4x.", i, (addr + 16'h2024)%16'hFFFF, rdata);
    end

    ///////////////////////////////////////////////////
    //                 RESIDUAL SRAM                 //
    ///////////////////////////////////////////////////
    for(int i = 0; i < 256 ; i++)begin  //0.5KB
        addr = i + `RESIDUAL_MEM_BASE_ADDR;
        wdata = (addr + 16'h2024)%16'hFFFF;
        FPGA_SPI_WR(addr, wdata);
        // $display("WR[%0d]",i);
    end
    
    repeat(10) @(negedge  chip_clk);
    residual_sram_wr_done = 1;
    @(negedge chip_clk)
    residual_sram_wr_done = 0;


    for(int i = 0; i < 256; i++)begin
        addr = i + `RESIDUAL_MEM_BASE_ADDR;
        FPGA_SPI_RD(addr, rdata);
        if(rdata !== (addr + 16'h2024)%16'hFFFF)
            $display("Residual mem [%0d] value wrong, Hoped value %4x != %4x.", i, (addr + 16'h2024)%16'hFFFF, rdata);
    end

    $display("Final check");
    kv_cache_wr_done = 1;
    wmem_wr_done = 1;
    head_sram_wr_done = 1; //多次检查
    global_sram_wr_done = 1;
    residual_sram_wr_done = 1;
    @(negedge chip_clk);
    kv_cache_wr_done = 0;
    wmem_wr_done = 0;
    head_sram_wr_done = 0;
    global_sram_wr_done = 0;
    residual_sram_wr_done = 0;

    repeat(100) @(negedge chip_clk);

    $finish();
end
`ifndef TRUE_MEM
/////////////////////////////////////////////////////
//             check head sram                     //
/////////////////////////////////////////////////////
genvar h;
generate
    for(h = 0; h < `HEAD_NUM; h++)begin
        initial begin
            #1;
            while(1)begin
                @(negedge chip_clk);
                if(head_sram_wr_done)
                    break;
            end
            $display("Start check head sram");
            for(int i = 0; i < `HEAD_SRAM_DEPTH; i++)begin
                for(int j =0; j < `MAC_MULT_NUM/2; j++)begin
                    if(h%2==0)begin
                        if (top.array_top_inst.inst_head_array.head_gen_array[h/2].two_heads_inst.inst_head_top_0.inst_head_sram.ram_piece.mem[i][j*16 +: 16] !== (16'h2024 + h * 256 + i * `MAC_MULT_NUM/2 + j))
                            $display("WRONG");
                    end
                    else begin
                        if (top.array_top_inst.inst_head_array.head_gen_array[h/2].two_heads_inst.inst_head_top_1.inst_head_sram.ram_piece.mem[i][j*16 +: 16] !== (16'h2024 + h * 256 + i * `MAC_MULT_NUM/2 + j))
                            $display("WRONG");
                    end
                end
            end
        end
    end
endgenerate

/////////////////////////////////////////////////////
//             check kv cache                      //
/////////////////////////////////////////////////////
genvar c;
generate
    for(h = 0; h < `HEAD_NUM; h++)begin 
        for(c = 0; c < `HEAD_CORE_NUM ; c++)begin
            initial begin
                #1;
                while(1)begin
                    while(1)begin
                        @(negedge chip_clk);
                        if(kv_cache_wr_done)
                            break;
                    end
                    if(h <= current_head && c <= current_core) begin //之前的都检查
                        $display("Start check HEAD[%0D]CORE[%0D] KV CACHE",h,c);
                        for(int i = 0; i < `KV_CACHE_DEPTH_SINGLE_USER*4; i++)begin
                            for(int k = 0; k < `MAC_MULT_NUM/2;k++)begin
                                if(h%2==0)begin
                                    if(top.array_top_inst.inst_head_array.head_gen_array[h/2].two_heads_inst.inst_head_top_0.inst_head_core_array.head_cores_generate_array[c].core_top_inst.mem_inst.kv_cache_inst.kv_cache.mem[i][k*16 +: 16] 
                                        !== (h * 'h40000 + c * 'h4000 + i * `MAC_MULT_NUM/2 + k + 16'h2024)%16'hFFFF)
                                            $display("KV CACHE HEAD[%0D]CORE[%0D] WRONG",h,c);
                                end
                                else begin
                                    if(top.array_top_inst.inst_head_array.head_gen_array[h/2].two_heads_inst.inst_head_top_1.inst_head_core_array.head_cores_generate_array[c].core_top_inst.mem_inst.kv_cache_inst.kv_cache.mem[i][k*16 +: 16] 
                                        !== (h * 'h40000 + c * 'h4000 + i * `MAC_MULT_NUM/2 + k + 16'h2024)%16'hFFFF)
                                            $display("KV CACHE HEAD[%0D]CORE[%0D] WRONG",h,c);
                                end
                            end
                        end
                        $display("Check finish");
                    end
                end
            end
        end
    end
endgenerate

/////////////////////////////////////////////////////
//             check wmem                          //
/////////////////////////////////////////////////////
generate
    for(h = 0; h < `HEAD_NUM; h++)begin 
        for(c = 0; c < `HEAD_CORE_NUM ; c++)begin
            initial begin
                #1;
                while(1)begin
                    while(1)begin
                        @(negedge chip_clk);
                        if(wmem_wr_done)
                            break;
                    end
                    if(h <= current_head && c <= current_core) begin //之前的都检查
                        $display("Start check HEAD[%0D]CORE[%0D] WMEM",h,c);
                        for(int i = 0; i < `WMEM_DEPTH; i++)begin
                            for(int k = 0; k < `MAC_MULT_NUM/2;k++)begin
                                if(h%2==0)begin
                                    if(top.array_top_inst.inst_head_array.head_gen_array[h/2].two_heads_inst.inst_head_top_0.inst_head_core_array.head_cores_generate_array[c].core_top_inst.mem_inst.wmem_inst.inst_mem_sp.mem[i][k*16 +: 16] 
                                        !== (h * 'h40000 + c * 'h4000 + 'h1000 + i * `MAC_MULT_NUM/2 + k + 16'h2024)%16'hFFFF)
                                            $display(" WMEM HEAD[%0D]CORE[%0D] WRONG",h,c);
                                end 
                                else begin
                                    if(top.array_top_inst.inst_head_array.head_gen_array[h/2].two_heads_inst.inst_head_top_1.inst_head_core_array.head_cores_generate_array[c].core_top_inst.mem_inst.wmem_inst.inst_mem_sp.mem[i][k*16 +: 16] 
                                        !== (h * 'h40000 + c * 'h4000 + 'h1000 + i * `MAC_MULT_NUM/2 + k + 16'h2024)%16'hFFFF)
                                            $display(" WMEM HEAD[%0D]CORE[%0D] WRONG",h,c);
                                end
                            end
                        end
                        $display("Check finish");
                    end
                end
            end
        end
    end
endgenerate

/////////////////////////////////////////////////////
//             check global mem                    //
/////////////////////////////////////////////////////
initial begin
    #1;
    while(1)begin
        while(1)begin
            @(negedge chip_clk);
            if(global_sram_wr_done)
                break;
        end
            $display("Start check global mem");
            for(int i = 0; i < `GLOBAL_SRAM_DEPTH; i++)begin
                for(int k = 0; k < `MAC_MULT_NUM/2;k++)begin
                    if(top.array_top_inst.inst_global_sram.inst_mem_dp.mem[i][k*16 +: 16] 
                        !== (`GLOBAL_MEM_BASE_ADDR + i * `MAC_MULT_NUM/2 + k + 16'h2024)%16'hFFFF)
                            $display("Global mem wrong");
                end
            end
            $display("Check finish");

    end
end

/////////////////////////////////////////////////////
//             check residual mem                  //
/////////////////////////////////////////////////////
initial begin
    #1;
    while(1)begin
        while(1)begin
            @(negedge chip_clk);
            if(residual_sram_wr_done)
                break;
        end
            $display("Start check residual mem");
            for(int i = 0; i < `GLOBAL_SRAM_DEPTH; i++)begin
                for(int k = 0; k < `MAC_MULT_NUM/2;k++)begin
                    if(top.array_top_inst.inst_residual_sram.ram_piece.mem[i][k*16 +: 16] 
                        !== (`RESIDUAL_MEM_BASE_ADDR + i * `MAC_MULT_NUM/2 + k + 16'h2024)%16'hFFFF)
                            $display("Residual mem wrong");
                end
            end
            $display("Check finish");

    end
end
`endif

always begin
    repeat(100000) @(negedge chip_clk);
    $display("Time: %t", $time());
end
endmodule
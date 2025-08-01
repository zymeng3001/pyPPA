`timescale 1ns/1ps

module mem_sp_sky130_tb;

    // Parameters
    localparam DATA_BIT = ${sram_width};
    localparam DEPTH    = ${sram_depth};
    localparam ADDR_BIT = $clog2(DEPTH);

    // DUT signals
    reg                    clk;
    reg  [ADDR_BIT-1:0]    addr;
    reg                    wen;
    reg  [DATA_BIT-1:0]    wdata;
    reg  [DATA_BIT-1:0]    bwe;
    reg                    ren;
    wire [DATA_BIT-1:0]    rdata;

    // Instantiate the DUT
    //  mem_sp_sky130 #(
    //      .DATA_BIT(DATA_BIT),
    //      .DEPTH(DEPTH),
    //      .ADDR_BIT(ADDR_BIT),
    //      .BWE(0)
    //  ) dut (
    //      .clk(clk),
    //      .addr(addr),
    //      .wen(wen),
    //      .wdata(wdata),
    //      .bwe(bwe),
    //      .ren(ren),
    //      .rdata(rdata)
    //  );

    // for postsynthesis
   mem_sp_sky130 dut (
       .clk(clk),
       .addr(addr),
       .wen(wen),
       .wdata(wdata),
       .bwe(bwe),
       .ren(ren),
       .rdata(rdata)
   );

    // Clock generation
    initial clk = 0;
    always #2.5 clk = ~clk;  // 100MHz
    
    integer i;

    // Test procedure
    initial begin
        $display("===== Begin SRAM Testbench =====");
        addr = 0;
        wen  = 0;
        ren  = 0;
        wdata = 0;
        bwe = 1;

        repeat (5) @(posedge clk);  // Wait for initialization

        // Write to a few addresses
        for (i = 0; i < 8; i=i+1) begin
            @(posedge clk);
            addr  <= i;
            wdata <= $random;
            wen   <= 1;
            ren   <= 0;
            $display("[WRITE] addr=%0d data=0x%h", i, wdata);
        end
        @(posedge clk);
        wen <= 0;

        // Read back the values
        for (i = 0; i < 8; i=i+1) begin
            @(posedge clk);
            addr <= i;
            ren  <= 1;
            @(posedge clk);
            $display("[READ ] addr=%0d data=0x%h", i, rdata);
            ren <= 0;
        end

        repeat (10) @(posedge clk);
        $display("===== SRAM Test Completed =====");
        $finish;
    end

endmodule

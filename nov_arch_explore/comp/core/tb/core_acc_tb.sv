module tb_core_acc;

    // Testbench parameters
    parameter IDATA_WIDTH = (`IDATA_WIDTH*2+$clog2(`ARR_MAC_NUM));
    parameter ODATA_BIT = `ODATA_WIDTH;
    parameter CDATA_BIT = `ARR_CDATA_BIT;
    parameter CLK_PERIOD = 20; // Clock period in ns

    // Inputs to the DUT
    reg clk;
    reg rstn;
    reg [CDATA_BIT-1:0] cfg_acc_num;
    reg [IDATA_WIDTH-1:0] idata;
    reg idata_valid;

    // Outputs from the DUT
    wire [ODATA_BIT-1:0] odata;
    wire odata_valid;

    // DUT instantiation
    core_acc #(
        .IDATA_WIDTH(IDATA_WIDTH),
        .ODATA_BIT(ODATA_BIT),
        .CDATA_BIT(CDATA_BIT)
    ) dut (
        .clk(clk),
        .rstn(rstn),
        .cfg_acc_num(cfg_acc_num),
        .idata(idata),
        .idata_valid(idata_valid),
        .odata(odata),
        .odata_valid(odata_valid)
    );

    // Clock generation
    always begin
        #(CLK_PERIOD/2) clk=~clk;
    end

    // Testbench stimulus and checks
    initial begin
        // Initialize inputs
        clk = 0;
        rstn = 0;
        cfg_acc_num = 0;
        idata = 0;
        idata_valid = 0;

        // Reset delay
        #(CLK_PERIOD*10) rstn = 1;
        
        // Configure number of accumulations
        cfg_acc_num = 5; // Accumulate 5 values
        @(posedge clk)
        // Send input datasets after reset
        repeat(15) begin
            @(negedge clk) begin
                idata_valid = 1;
                // idata = $random; // Random data for accumulation
                idata = 1;
            end
            @(negedge clk) idata_valid = 0;
            #(CLK_PERIOD*10); // Wait for a few clock cycles
        end

        // Terminate simulation
        #(CLK_PERIOD*100) $finish;
    end

    // Optional: Monitor the output and print
    always @(posedge clk) begin
        if (odata_valid) begin
            $display("Time: %t, Output Data: %0d", $time, odata);
        end
    end

endmodule
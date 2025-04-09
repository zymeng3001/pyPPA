module relu
#(
    parameter integer BUS_NUM = ${head_dim},      
    parameter integer BUS_NUM_WIDTH = $clog2(BUS_NUM), 
    parameter integer SCALA_POS_WIDTH = 5,
    parameter integer FIXED_DATA_WIDTH = 8
)
(
    // Global Signals
    input wire                       clk,
    input wire                       rst_n,

    // Data Signals
    input wire signed [BUS_NUM * FIXED_DATA_WIDTH - 1 : 0] in_fixed_data,
    input wire        [BUS_NUM-1:0]                        in_fixed_data_vld,
    output reg signed [BUS_NUM * FIXED_DATA_WIDTH - 1 : 0] out_fixed_data,
    output reg        [BUS_NUM-1:0]                        out_fixed_data_vld
);    

    integer i;
    // Process input data with ReLU function
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // Reset all outputs
            out_fixed_data_vld <= 0;
            out_fixed_data <= 0;
        end else begin
            for (i = 0; i < BUS_NUM; i = i + 1) begin
                if (in_fixed_data_vld[i]) begin
                    out_fixed_data[i*FIXED_DATA_WIDTH+:FIXED_DATA_WIDTH] <= (in_fixed_data[i*FIXED_DATA_WIDTH + FIXED_DATA_WIDTH - 1] == 0) ? in_fixed_data[i*FIXED_DATA_WIDTH+:FIXED_DATA_WIDTH] : 0;
                    out_fixed_data_vld[i] <= 1;
                end else begin
                    out_fixed_data_vld[i] <= 0;
                end
            end

        end
    end

endmodule


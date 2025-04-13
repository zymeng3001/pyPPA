module silu #(
    parameter integer BUS_NUM = ${head_dim},
    parameter integer SCALA_POS_WIDTH = 5,
    parameter integer FIXED_DATA_WIDTH = 8
)(
    // Global Signals
    input wire clk,
    input wire rst_n,

    // Data Signals
    input wire signed [BUS_NUM * FIXED_DATA_WIDTH - 1 : 0] in_fixed_data,
    input wire        [BUS_NUM-1:0]                        in_fixed_data_vld,
    output reg signed [BUS_NUM * FIXED_DATA_WIDTH - 1 : 0] out_fixed_data,
    output reg        [BUS_NUM-1:0]                        out_fixed_data_vld
);

    integer i;
    reg signed [FIXED_DATA_WIDTH-1:0] x, y;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_fixed_data <= 0;
            out_fixed_data_vld <= 0;
        end else begin
            out_fixed_data_vld <= 0;
            for (i = 0; i < BUS_NUM; i = i + 1) begin
                x = in_fixed_data[i*FIXED_DATA_WIDTH +: FIXED_DATA_WIDTH];

                if (in_fixed_data_vld[i]) begin
                    // Piecewise approximation of SiLU (x * sigmoid(x))
                    if (x < -8'sd96) begin
                        y = 8'sd0;
                    end else if (x < -8'sd32) begin
                        y = (x >>> 2) + 8'sd24; // ~0.25x + 24
                    end else if (x < 0) begin
                        y = (x >>> 1) + 8'sd8;  // ~0.5x + 8
                    end else if (x < 8'sd96) begin
                        y = (x >>> 1) + (x >>> 2) + (x >>> 3); // ~0.875x
                    end else begin
                        y = x;
                    end

                    out_fixed_data[i*FIXED_DATA_WIDTH +: FIXED_DATA_WIDTH] <= y;
                    out_fixed_data_vld[i] <= 1;
                end else begin
                    out_fixed_data[i*FIXED_DATA_WIDTH +: FIXED_DATA_WIDTH] <= 0;
                    out_fixed_data_vld[i] <= 0;
                end
            end
        end
    end

endmodule

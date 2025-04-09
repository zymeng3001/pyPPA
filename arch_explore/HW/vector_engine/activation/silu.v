module silu #(
    parameter BUS_NUM = 8  // Vector size
)(
    input  wire               clk,
    input  wire               rst,
    input  wire               in_valid,
    input  wire signed [7:0]  input_vec [BUS_NUM-1:0],    // fixed-point data，shifted by 5 bits
    output reg  signed [7:0]  output_vec [BUS_NUM-1:0],
    output reg                out_valid
);

    integer i;
    reg signed [7:0] x;
    reg signed [7:0] y;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            for (i = 0; i < BUS_NUM; i = i + 1)
                output_vec[i] <= 0;
            out_valid <= 0;
        end else if (in_valid) begin
            for (i = 0; i < BUS_NUM; i = i + 1) begin
                x = input_vec[i];
                if (x < -96) begin
                    y = 8'sd0;
                end else if (x < -32) begin
                    y = (x >>> 2) + 8'sd24;  // 0.25x + 24
                end else if (x < 0) begin
                    y = (x >>> 1) + 8'sd8;   // 0.5x + 8
                end else if (x < 96) begin
                    y = (x >>> 1) + (x >>> 2) + (x >>> 3);  // 0.875x
                end else begin
                    y = x;  // passthrough for large x
                end
                output_vec[i] <= y;
            end
            out_valid <= 1;
        end else begin
            out_valid <= 0;
        end
    end

endmodule
module gelu #(
    parameter BUS_NUM = ${head_dim}  // Vector size
)(
    input  wire               clk,
    input  wire               rst_n,
    input  wire               in_valid,
    input  wire signed [7:0]  input_vec [BUS_NUM-1:0],    // fixed-point data，shifted by 5 bits
    output reg  signed [7:0]  output_vec [BUS_NUM-1:0],
    output reg                out_valid
);

    integer i;
    reg signed [7:0] x;
    reg signed [7:0] y;

    always @(posedge clk or negedge rst_n) begin
        if (rst) begin
            output_vec[i] <= 0;
            out_valid <= 0;
        end else if (in_valid) begin
            for (i = 0; i < BUS_NUM; i = i + 1) begin
                x = input_vec[i];
                if (x < -96) begin
                    y = 8'sd0;
                end else if (x < -32) begin
                    // y = (3/8)x + 36 ≈ x*0.375 + 36
                    y = (x >>> 2) + (x >>> 3) + 8'sd36;
                end else if (x < 32) begin
                    // y = (3/4)x = x*0.75
                    y = (x >>> 1) + (x >>> 2);
                end else if (x < 96) begin
                    // y = (7/8)x - 8 ≈ x*0.875 - 8
                    y = (x >>> 1) + (x >>> 2) + (x >>> 3) - 8'sd8;
                end else begin
                    y = x;
                end
                output_vec[i] <= y;
            end
            out_valid <= 1;
        end else begin
            out_valid <= 0;
        end
    end

endmodule


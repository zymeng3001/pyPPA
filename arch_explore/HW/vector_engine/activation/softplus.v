module softplus #(
    parameter BUS_NUM = ${head_dim}  // Vector size
)(
    input  wire               clk,
    input  wire               rst_n,
    input  wire               in_valid,
    input  wire signed [7:0]  input_vec [BUS_NUM-1:0],   // fixed-point data，shifted by 5 bits
    output reg  signed [7:0]  output_vec [BUS_NUM-1:0],
    output reg                out_valid
);

    integer i;
    reg signed [7:0] x;
    reg signed [7:0] y;

    always @(posedge clk or negedge rst) begin
        if (rst_n) begin
            output_vec[i] <= 0;
            out_valid <= 0;
        end else if (in_valid) begin
            for (i = 0; i < BUS_NUM; i = i + 1) begin
                x = input_vec[i];
                if (x < -64)
                    y = 8'sd0;
                else if (x >= 64)
                    y = x;
                else
                    y = (x >>> 1) + 8'sd16;  // y ≈ 0.5x + 16
                output_vec[i] <= y;
            end
            out_valid <= 1;
        end else begin
            out_valid <= 0;
        end
    end

endmodule

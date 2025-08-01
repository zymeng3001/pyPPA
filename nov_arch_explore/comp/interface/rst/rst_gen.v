// +FHDR========================================================================
//  File Name:      rst_gen.v
//                  Shiwei Liu (liusw18@gmail.com)
//  Organization:
//  Description:
//     Asynchronous Reset with synchronous Release. Support Multiple Clock Domain.
// -FHDR========================================================================

module rst_gen (
    input               rst,
    input               clk_fast,
    input               clk_slow,
    output  reg         rst_fast,
    output  reg         rst_slow
);

    reg     rst_fast_d0, rst_slow_d0;

    always @(posedge clk_fast or posedge rst) begin
        if (rst) begin
            rst_fast_d0 <= 1'b1;
            rst_fast    <= 1'b1;
        end
        else begin
            rst_fast_d0 <= 1'b0;
            rst_fast    <= rst_fast_d0;
        end
    end

    always @(posedge clk_slow or posedge rst) begin
        if (rst) begin
            rst_slow_d0 <= 1'b1;
            rst_slow    <= 1'b1;
        end
        else begin
            rst_slow_d0 <= 1'b0;
            rst_slow    <= rst_slow_d0;
        end
    end

endmodule
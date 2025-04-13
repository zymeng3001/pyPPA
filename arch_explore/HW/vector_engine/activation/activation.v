module activation
#(
    parameter integer BUS_NUM = ${head_dim},      
    parameter string ACTIVATION = "${activation}",
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


generate;
    if (ACTIVATION == "relu") begin : relu
        relu #(
            .BUS_NUM(BUS_NUM),
            .SCALA_POS_WIDTH(SCALA_POS_WIDTH),
            .FIXED_DATA_WIDTH(FIXED_DATA_WIDTH)
        ) relu_inst (
            .clk(clk),
            .rst_n(rst_n),
            .in_fixed_data(in_fixed_data),
            .in_fixed_data_vld(in_fixed_data_vld),
            .out_fixed_data(out_fixed_data),
            .out_fixed_data_vld(out_fixed_data_vld)
        );
    end else if (ACTIVATION == "gelu") begin : gelu
        gelu #(
            .BUS_NUM(BUS_NUM),
            .SCALA_POS_WIDTH(SCALA_POS_WIDTH),
            .FIXED_DATA_WIDTH(FIXED_DATA_WIDTH)
        ) gelu_inst (
            .clk(clk),
            .rst_n(rst_n),
            .in_fixed_data(in_fixed_data),
            .in_fixed_data_vld(in_fixed_data_vld),
            .out_fixed_data(out_fixed_data),
            .out_fixed_data_vld(out_fixed_data_vld)
        );
    end else if (ACTIVATION == "silu") begin : silu
        silu #(
            .BUS_NUM(BUS_NUM),
            .SCALA_POS_WIDTH(SCALA_POS_WIDTH),
            .FIXED_DATA_WIDTH(FIXED_DATA_WIDTH)
        ) silu_inst (
            .clk(clk),
            .rst_n(rst_n),
            .in_fixed_data(in_fixed_data),
            .in_fixed_data_vld(in_fixed_data_vld),
            .out_fixed_data(out_fixed_data),
            .out_fixed_data_vld(out_fixed_data_vld)
        );
    end else if (ACTIVATION == "softplus") begin : softplus
        softplus #(
            .BUS_NUM(BUS_NUM),
            .SCALA_POS_WIDTH(SCALA_POS_WIDTH),
            .FIXED_DATA_WIDTH(FIXED_DATA_WIDTH)
        ) softplus_inst (
            .clk(clk),
            .rst_n(rst_n),
            .in_fixed_data(in_fixed_data),
            .in_fixed_data_vld(in_fixed_data_vld),
            .out_fixed_data(out_fixed_data),
            .out_fixed_data_vld(out_fixed_data_vld)
        );
    end else begin : default
        initial begin
            $display("Error: Unsupported activation function '%s'", ${activation});
        end
    end

endgenerate



endmodule
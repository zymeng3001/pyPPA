module residual_adder_single_tb();
logic clk;
logic rst_n;

logic                                        scale_vld;
logic        [`CDATA_SCALE_WIDTH-1:0]            scale_a;
logic        [`CDATA_SCALE_WIDTH-1:0]            scale_b;

logic                                        shift_vld;
logic        [`CDATA_SHIFT_WIDTH-1:0]        shift;

logic signed [`IDATA_WIDTH-1:0]              in_data_a;
logic signed [`IDATA_WIDTH-1:0]              in_data_b;
logic                                        in_data_vld;

logic signed [`IDATA_WIDTH-1:0]              out_data;
logic                                        out_data_vld;


always begin
    #(`CLOCK_PERIOD/2.0) clk = ~clk;
end

initial begin
    clk = 0;
    rst_n = 0;
    scale_vld = 0;
    shift_vld = 0;
    in_data_vld = 0;
    repeat(10) @(negedge clk);
    rst_n = 1;
    repeat(10) @(negedge clk);
    scale_a = 8;
    scale_b = 8;
    scale_vld = 1;

    shift = 3;
    shift_vld = 1;
    @(negedge clk);
    scale_vld = 0;
    shift_vld = 0;
    @(negedge clk);
    in_data_vld = 1;
    in_data_a = 1;
    in_data_b = 1;
    @(negedge clk);
    in_data_vld = 1;
    in_data_a = 2;
    in_data_b = 4;
    @(negedge clk);
    in_data_vld = 1;
    in_data_a = 4;
    in_data_b = 25;
    @(negedge clk);
    in_data_vld = 1;
    in_data_a = -4;
    in_data_b = 25;
    @(negedge clk);
    in_data_vld = 1;
    in_data_a = -4;
    in_data_b = -25;
    @(negedge clk);
    in_data_vld = 0;

    repeat(20) @(negedge clk);
    $finish();

    
end

residual_adder_single inst_residual_adder_single(
    .clk(clk),
    .rst_n(rst_n),

    .scale_vld(scale_vld),
    .scale_a(scale_a),
    .scale_b(scale_b),

    .shift_vld(shift_vld),
    .shift(shift),

    .in_data_a(in_data_a),
    .in_data_b(in_data_b),
    .in_data_vld(in_data_vld),

    .out_data(out_data),
    .out_data_vld(out_data_vld)
);

endmodule
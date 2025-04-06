module gelu_int8 #(
    parameter BUS_NUM = 8,  // Vector size
    parameter LUT_ADDR = 8,  // LUT address width
    parameter LUT_DATA = 8  // LUT data width
)(
    input  wire               clk,
    input  wire               rstn,
    input  wire               in_valid,
    input  wire signed [7:0] [BUS_NUM-1:0] input_vec,
    output reg  signed [7:0] [BUS_NUM-1:0] output_vec,
    // LUT Interface
    input       [`LUT_ADDR-1:0] lut_waddr,
    input                       lut_wen,
    input       [`LUT_DATA-1:0]  lut_wdata,
    output reg                out_valid
);

    // LUT-based GELU approximation
    reg signed [7:0] gelu_lut [0:255];  // LUT for -128 to 127

    always @(posedge clk or negedge rstn) begin
        if (rst) begin
            for (i = 0; i < N; i = i + 1)
                output_vec[i] <= 0;
            out_valid <= 0;
        end else if (in_valid) begin
            for (i = 0; i < N; i = i + 1)
                output_vec[i] <= gelu_lut[input_vec[i][7:0]];
            out_valid <= 1;
        end else begin
            out_valid <= 0;
        end
    end

    wire    [`LUT_ADDR-1:0] lut_addr_w;
    wire                    lut_ren;
    wire    [`LUT_DATA-1:0]  lut_rdata;
    reg     exp_valid;

    mem_sp lut_inst (
        .clk                    (clk),
        .addr                   (lut_addr_w),
        .wen                    (lut_wen),
        .wdata                  (lut_wdata),
        .ren                    (lut_ren),
        .rdata                  (lut_rdata)
    ); 
    

endmodule

module mem_sp_gelu 
(
    // Global Signals
    input                       clk,

    // Data Signals
    input       [`LUT_ADDR-1:0]  addr,
    input                       wen,
    input       [`LUT_DATA-1:0]  wdata,
    input                       ren,
    output  reg [`LUT_DATA-1:0]  rdata
);

    // 1. RAM/Memory initialization
    reg signed [`LUT_DATA-1:0]  mem [0:`LUT_DEPTH-1];

    // 2. Write channel
    always @(posedge clk) begin
        if (wen) begin
            mem[addr] <= wdata;
        end
    end

    // 3. Read channel
    always @(posedge clk) begin
        if (ren) begin
            rdata <= mem[addr];
        end
    end

endmodule


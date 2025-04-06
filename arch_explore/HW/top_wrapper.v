module top_wrapper #(
    parameter HNUM = ${n_heads},               // Number of heads
    parameter VNUM = ${n_cols},               // Number of columns
    parameter GBUS_DATA = ${gbus_width},         // Global bus data width
    parameter GBUS_ADDR = 12,
    parameter CDATA_BIT = 8,
    parameter ODATA_BIT = 16,
    parameter IDATA_BIT = 8,
    parameter MAC_NUM = ${gbus_width/8},

    parameter LBUF_DEPTH = 64,
    parameter WMEM_DEPTH = ${wmem_depth},
    parameter CACHE_DEPTH = ${cache_depth},

    parameter softmax_option = ${softmax_option},

    parameter EXP_BIT = 8,
    parameter MAT_BIT = 7,
    parameter LUT_DATA = EXP_BIT + MAT_BIT + 1,
    parameter LUT_ADDR = IDATA_BIT >> 1,
    parameter LUT_DEPTH = 2 ** LUT_ADDR,

    parameter BUS_NUM = ${n_model}
) (
    input clk,
    input rstn
);

    // === Config Signals ===
    wire [CDATA_BIT-1:0] cfg_consmax_shift;
    wire [ODATA_BIT-1:0] cfg_quant_scale, cfg_quant_bias, cfg_quant_shift;

    // === Global Bus and Activation ===
    wire [HNUM-1:0][CDATA_BIT-1:0] cfg_acc_num;
    wire [HNUM-1:0][GBUS_ADDR-1:0] in_GBUS_ADDR;
    wire [HNUM*VNUM-1:0] gbus_wen, gbus_ren;
    wire [HNUM-1:0][GBUS_DATA-1:0] gbus_wdata;
    wire [HNUM-1:0][GBUS_DATA-1:0] gbus_rdata;
    wire [HNUM*VNUM-1:0] gbus_rvalid;

    wire vlink_enable;
    wire [VNUM-1:0][GBUS_DATA-1:0] vlink_wdata;
    wire [VNUM-1:0] vlink_wen;
    wire [VNUM-1:0][GBUS_DATA-1:0] vlink_rdata;
    wire [VNUM-1:0] vlink_rvalid;

    wire [HNUM-1:0][GBUS_DATA-1:0] hlink_wdata;
    wire [HNUM-1:0] hlink_wen;
    wire [HNUM-1:0][GBUS_DATA-1:0] hlink_rdata;
    wire [HNUM-1:0] hlink_rvalid;

    // CMEM interfaces
    wire [HNUM-1:0][VNUM-1:0][GBUS_ADDR-1:0] cmem_waddr, cmem_raddr;
    wire [HNUM-1:0][VNUM-1:0] cmem_wen, cmem_ren;

    // Buffer flags
    wire [HNUM-1:0][VNUM-1:0] lbuf_empty, lbuf_reuse_empty, lbuf_full, lbuf_almost_full;
    wire [HNUM-1:0][VNUM-1:0] lbuf_reuse_ren, lbuf_reuse_rst, lbuf_ren;
    wire [HNUM-1:0][VNUM-1:0] abuf_empty, abuf_reuse_empty, abuf_full, abuf_almost_full;
    wire [HNUM-1:0][VNUM-1:0] abuf_reuse_ren, abuf_reuse_rst, abuf_ren;

   

    // === Instantiate core_array ===
    core_array #(
        .HNUM(HNUM), .VNUM(VNUM), .GBUS_DATA(GBUS_DATA), .GBUS_ADDR(GBUS_ADDR),
        .LBUF_DEPTH(LBUF_DEPTH), .LBUF_DATA(GBUS_DATA), .LBUF_ADDR($clog2(LBUF_DEPTH)),
        .CDATA_BIT(CDATA_BIT), .ODATA_BIT(ODATA_BIT), .IDATA_BIT(IDATA_BIT), .MAC_NUM(MAC_NUM),
        .WMEM_DEPTH(WMEM_DEPTH), .CACHE_DEPTH(CACHE_DEPTH)
    ) u_core_array (
        .clk(clk), .rstn(rstn),
        .cfg_quant_scale(cfg_quant_scale),
        .cfg_quant_bias(cfg_quant_bias),
        .cfg_quant_shift(cfg_quant_shift),
        .cfg_consmax_shift(cfg_consmax_shift),
        .cfg_acc_num(cfg_acc_num),
        .in_GBUS_ADDR(in_GBUS_ADDR),
        .gbus_wen(gbus_wen),
        .gbus_wdata(gbus_wdata),
        .gbus_ren(gbus_ren),
        .gbus_rdata(gbus_rdata),
        .gbus_rvalid(gbus_rvalid),
        .vlink_enable(vlink_enable),
        .vlink_wdata(vlink_wdata),
        .vlink_wen(vlink_wen),
        .vlink_rdata(vlink_rdata),
        .vlink_rvalid(vlink_rvalid),
        .hlink_wdata(hlink_wdata),
        .hlink_wen(hlink_wen),
        .hlink_rdata(hlink_rdata),
        .hlink_rvalid(hlink_rvalid),
        .cmem_waddr(cmem_waddr),
        .cmem_wen(cmem_wen),
        .cmem_raddr(cmem_raddr),
        .cmem_ren(cmem_ren),
        .lbuf_empty(lbuf_empty),
        .lbuf_reuse_empty(lbuf_reuse_empty),
        .lbuf_reuse_ren(lbuf_reuse_ren),
        .lbuf_reuse_rst(lbuf_reuse_rst),
        .lbuf_full(lbuf_full),
        .lbuf_almost_full(lbuf_almost_full),
        .lbuf_ren(lbuf_ren),
        .abuf_empty(abuf_empty),
        .abuf_reuse_empty(abuf_reuse_empty),
        .abuf_reuse_ren(abuf_reuse_ren),
        .abuf_reuse_rst(abuf_reuse_rst),
        .abuf_full(abuf_full),
        .abuf_almost_full(abuf_almost_full),
        .abuf_ren(abuf_ren)
    );

    generate;
        
    endgenerate

    // // === LUT and Softmax I/O ===
    // wire [LUT_ADDR:0] lut_waddr;
    // wire [LUT_DATA-1:0] lut_wdata;
    // wire lut_wen;

    // wire [(GBUS_DATA * HNUM)-1:0] softmax_idata;
    // wire [HNUM-1:0] softmax_idata_valid;
    // wire [(GBUS_DATA * HNUM)-1:0] softmax_odata;
    // wire [(MAC_NUM * HNUM)-1:0] softmax_odata_valid;

    // // === Instantiate consmax_bus (Softmax) ===
    // consmax_bus #(
    //     .IDATA_BIT(IDATA_BIT),
    //     .ODATA_BIT(ODATA_BIT),
    //     .CDATA_BIT(CDATA_BIT),
    //     .EXP_BIT(EXP_BIT),
    //     .MAT_BIT(MAT_BIT),
    //     .LUT_DATA(LUT_DATA),
    //     .LUT_ADDR(LUT_ADDR),
    //     .LUT_DEPTH(LUT_DEPTH),
    //     .GBUS_DATA(GBUS_DATA),
    //     .GBUS_WIDTH(GBUS_DATA/8),
    //     .NUM_HEAD(HNUM),
    //     .NUM_COL(VNUM)
    // ) u_consmax_bus (
    //     .clk(clk),
    //     .rstn(rstn),
    //     .cfg_consmax_shift(cfg_consmax_shift),
    //     .lut_waddr(lut_waddr),
    //     .lut_wen(lut_wen),
    //     .lut_wdata(lut_wdata),
    //     .idata(softmax_idata),
    //     .idata_valid(softmax_idata_valid),
    //     .odata(softmax_odata),
    //     .odata_valid(softmax_odata_valid)
    // );

    //Ports
    reg [CDATA_BIT-1:0] cfg_consmax_shift;
    reg [LUT_ADDR-1:0] lut_waddr;
    reg lut_wen;
    reg [LUT_DATA-1:0] lut_wdata;
    reg [7:0] idata;
    reg idata_valid;
    wire [7:0] odata;
    wire odata_valid;

    softmax  softmax_inst (
        .clk(clk),
        .rst(rst),
        .cfg_consmax_shift(cfg_consmax_shift),
        .lut_waddr(lut_waddr),
        .lut_wen(lut_wen),
        .lut_wdata(lut_wdata),
        .idata(idata),
        .idata_valid(idata_valid),
        .odata(odata),
        .odata_valid(odata_valid)
    );

    
    // Relu Instance
    reg signed [BUS_NUM * 8 - 1 : 0] in_fixed_data;
    reg [BUS_NUM-1:0] in_fixed_data_vld;
    wire [BUS_NUM * 8 - 1 : 0] out_fixed_data;
    wire [BUS_NUM-1:0] out_fixed_data_vld;

    relu relu_inst (
    .clk(clk),
    .rst_n(rst_n),
    .in_fixed_data(in_fixed_data),
    .in_fixed_data_vld(in_fixed_data_vld),
    .out_fixed_data(out_fixed_data),
    .out_fixed_data_vld(out_fixed_data_vld)
    );

endmodule

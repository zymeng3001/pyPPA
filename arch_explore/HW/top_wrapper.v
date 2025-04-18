`define ${activation}
`define ${softmax_choice}

module top_wrapper #(
    parameter HNUM = ${n_heads},               // Number of heads
    parameter VNUM = ${n_cols},               // Number of columns
    parameter GBUS_DATA = ${gbus_width},         // Global bus data width
    parameter GBUS_ADDR = 12,
    parameter CDATA_BIT = 8,
    parameter ODATA_BIT = 16,
    parameter IDATA_BIT = 8,
    parameter MAC_NUM = ${mac_num},

    parameter LBUF_DEPTH = 64,
    parameter WMEM_DEPTH = ${wmem_depth},
    parameter CACHE_DEPTH = ${cache_depth},

    parameter EXP_BIT = 8,
    parameter MAT_BIT = 7,
    parameter LUT_DATA = EXP_BIT + MAT_BIT + 1,
    parameter LUT_ADDR = IDATA_BIT >> 1,
    parameter LUT_DEPTH = 2 ** LUT_ADDR

    parameter integer ACT_BUS_NUM = ${head_dim},      
    parameter integer SCALA_POS_WIDTH = 5,
    parameter integer FIXED_DATA_WIDTH = 8
) (
    input clk,
    input rstn,
    input     [ODATA_BIT-1:0] cfg_quant_scale,
	input     [ODATA_BIT-1:0] cfg_quant_bias,
	input     [ODATA_BIT-1:0] cfg_quant_shift,
	input     [CDATA_BIT-1:0] cfg_consmax_shift, 
	
	input    [HNUM-1:0][CDATA_BIT-1:0]     cfg_acc_num,
	input            [HNUM-1:0][GBUS_ADDR-1:0]     in_GBUS_ADDR,
    input            [HNUM * VNUM - 1:0]                           gbus_wen,
    input            [HNUM-1:0][GBUS_DATA-1:0]     gbus_wdata,   
    input            [HNUM * VNUM - 1:0]                           gbus_ren,
    output   reg   [HNUM-1:0][GBUS_DATA-1:0]     gbus_rdata,     
    output   reg   [HNUM * VNUM - 1:0]        gbus_rvalid,
    // Channel - Core-to-Core Link
    // Vertical for Weight and Key/Value Propagation
    input                                           vlink_enable,
    input            [VNUM-1:0][GBUS_DATA-1:0]     vlink_wdata,
    input            [VNUM-1:0]                    vlink_wen,
    output   reg   [VNUM-1:0][GBUS_DATA-1:0]     vlink_rdata,
    output   reg   [VNUM-1:0]                    vlink_rvalid,
    // Horizontal for Activation Propagation
    input            [HNUM-1:0][GBUS_DATA-1:0]     hlink_wdata,    //hlink_wdata go through reg, to hlink_rdata
    input            [HNUM-1:0]                    hlink_wen,     
    output   reg   [HNUM-1:0][GBUS_DATA-1:0]     hlink_rdata,
    output   reg   [HNUM-1:0]                    hlink_rvalid,
    // Channel - MAC Operation
    // Core Memory Access for Weight and KV Cache
    //input            CMEM_ARR_PACKET                arr_cmem,
	input 	[HNUM-1:0][VNUM-1:0][GBUS_ADDR-1:0]  cmem_waddr,      // Write Value to KV Cache, G_BUS -> KV Cache, debug.
    input 	[HNUM-1:0][VNUM-1:0]    cmem_wen,
    input	[HNUM-1:0][VNUM-1:0][GBUS_ADDR-1:0]  cmem_raddr,
    input	[HNUM-1:0][VNUM-1:0]    cmem_ren,
	
    // Local Buffer Access for Weight and KV Cache
    output           [HNUM-1:0][VNUM-1:0]        lbuf_empty,
    output           [HNUM-1:0][VNUM-1:0]        lbuf_reuse_empty,
    input            [HNUM-1:0][VNUM-1:0]        lbuf_reuse_ren, //reuse pointer logic, when enable
    input            [HNUM-1:0][VNUM-1:0]        lbuf_reuse_rst,  //reuse reset logic, when first round of reset is finished, reset reuse pointer to current normal read pointer value
    output           [HNUM-1:0][VNUM-1:0]        lbuf_full,
    output           [HNUM-1:0][VNUM-1:0]        lbuf_almost_full,
    input            [HNUM-1:0][VNUM-1:0]        lbuf_ren,
    // Local Buffer access for Activation
    output           [HNUM-1:0][VNUM-1:0]        abuf_empty,
    output           [HNUM-1:0][VNUM-1:0]        abuf_reuse_empty,
    input            [HNUM-1:0][VNUM-1:0]        abuf_reuse_ren, //reuse pointer logic, when enable
    input            [HNUM-1:0][VNUM-1:0]        abuf_reuse_rst,  //reuse reset logic, when first round of reset is finished, reset reuse pointer to current normal read pointer value
    output           [HNUM-1:0][VNUM-1:0]        abuf_full,
    output           [HNUM-1:0][VNUM-1:0]        abuf_almost_full,
    input            [HNUM-1:0][VNUM-1:0]        abuf_ren,

    // for activation
    input wire signed [ACT_BUS_NUM * FIXED_DATA_WIDTH - 1 : 0] in_fixed_data,
    input wire        [ACT_BUS_NUM-1:0]                        in_fixed_data_vld,
    output reg signed [ACT_BUS_NUM * FIXED_DATA_WIDTH - 1 : 0] out_fixed_data,
    output reg        [ACT_BUS_NUM-1:0]                        out_fixed_data_vld,

    `ifdef SOFTMAX
    , // Softmax
    input wire [8-1:0] cfg_shift,
    input wire [LUT_ADDR-1:0] lut_waddr,
    input wire lut_wen,
    input wire [LUT_DATA-1:0] lut_wdata,
    input wire [8-1:0] idata,
    input wire idata_valid,
    output wire [8-1:0] odata,
    output wire odata_valid,
    `endif

    `ifdef CONSMAX
    , // Softermax
    input wire [8-1:0] cfg_consmax_shift,
    input wire [LUT_ADDR-1:0] lut_waddr,
    input wire lut_wen,
    input wire [LUT_DATA-1:0] lut_wdata,
    input wire [(GBUS_DATA*NUM_HEAD)-1:0] idata,
    input wire [NUM_HEAD-1:0] idata_valid,
    output wire [(GBUS_DATA*NUM_HEAD)-1:0] odata,
    output wire [(GBUS_WIDTH*NUM_HEAD)-1:0] odata_valid,
    `endif

    `ifdef SOFTERMAX
    , // Softermax
    input wire input_valid,
    input wire signed [7:0] input_vector,
    input wire [$clog2(ROW_WIDTH)-1:0] read_addr,
    output wire norm_valid,
    output wire final_out_valid,
    output wire [15:0] prob_buffer_out,
    `endif
); 

    // === Instantiate core_array ===
    core_array u_core_array (
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


activation activation_inst (
    .clk(clk),
    .rst_n(rstn),
    .in_fixed_data(in_fixed_data),
    .in_fixed_data_vld(in_fixed_data_vld),
    .out_fixed_data(out_fixed_data),
    .out_fixed_data_vld(out_fixed_data_vld)
  );

  softmax_wrapper softmax_wrapper_inst (
    .clk(clk),
    .rst_n(rstn),
    `ifdef SOFTMAX
    .cfg_shift(cfg_shift),
    .lut_waddr(lut_waddr),
    .lut_wen(lut_wen),
    .lut_wdata(lut_wdata),
    .idata(idata),
    .idata_valid(idata_valid),
    .odata(odata),
    .odata_valid(odata_valid),
    `endif
    `ifdef SOFTERMAX
    .input_valid(input_valid),
    .input_vector(input_vector),
    .read_addr(read_addr),
    .norm_valid(norm_valid),
    .final_out_valid(final_out_valid),
    .prob_buffer_out(prob_buffer_out),
    `endif
    `ifdef CONSMAX
    .cfg_consmax_shift(cfg_consmax_shift),
    .lut_waddr(lut_waddr),
    .lut_wen(lut_wen),
    .lut_wdata(lut_wdata),
    .idata(idata),
    .idata_valid(idata_valid),
    .odata(odata),
    .odata_valid(odata_valid),
    `endif
  );


endmodule

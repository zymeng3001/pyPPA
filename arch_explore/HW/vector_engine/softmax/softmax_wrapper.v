`define ${softmax_choice}

module softmax_wrapper
#(
    parameter integer SOFTMAX_NUM = ${max_context_length},
    parameter integer  GBUS_DATA = ${gbus_width}, // Global Bus Data Width
    parameter integer  GBUS_WIDTH = ${gbus_width/8},   // Global Bus Address Width
    parameter integer  NUM_HEAD  = ${n_heads},    // Number of Heads
    parameter integer  NUM_COL  = ${n_cols},     // Number of Columns

    parameter integer  EXP_BIT = 8,    // Exponent
    parameter integer  MAT_BIT = 7,    // Mantissa
    parameter integer  LUT_DATA  = EXP_BIT + MAT_BIT + 1,  // LUT Data Width (in FP)
    parameter integer  LUT_ADDR  = 16,         // LUT Address Width
    parameter integer  LUT_DEPTH = 2 ** LUT_ADDR           // LUT Depth for INT2FP
)(
    // Global Signals
    input wire clk,
    input wire rst_n
);

`ifdef SOFTMAX
    
reg [8-1:0] cfg_consmax_shift;
reg [LUT_ADDR-1:0] lut_waddr;
reg lut_wen;
reg [LUT_DATA-1:0] lut_wdata;
reg [8-1:0] idata;
reg idata_valid;
wire [8-1:0] odata;
wire odata_valid;

softmax # (
  .SOFTMAX_NUM(SOFTMAX_NUM),
)
softmax_inst (
  .clk(clk),
  .rst_n(rst_n),
  .cfg_consmax_shift(cfg_consmax_shift),
  .lut_waddr(lut_waddr),
  .lut_wen(lut_wen),
  .lut_wdata(lut_wdata),
  .idata(idata),
  .idata_valid(idata_valid),
  .odata(odata),
  .odata_valid(odata_valid)
);

`endif

`ifdef SOFTERMAX
    
reg input_valid;
reg signed [7:0] input_vector;
reg [clog2(ROW_WIDTH)-1:0] read_addr;
wire norm_valid;
wire final_out_valid;
wire [15:0] prob_buffer_out;

softermax # (
  .ROW_WIDTH(SOFTMAX_NUM)
)
softermax_inst (
  .rst_n(rst_n),
  .clk(clk),
  .input_valid(input_valid),
  .input_vector(input_vector),
  .read_addr(read_addr),
  .norm_valid(norm_valid),
  .final_out_valid(final_out_valid),
  .prob_buffer_out(prob_buffer_out)
);

`endif 

`ifdef CONSMAX
    

//Ports
reg [8-1:0] cfg_consmax_shift;
reg [LUT_ADDR:0] lut_waddr;
reg lut_wen;
reg [LUT_DATA-1:0] lut_wdata;
reg [(GBUS_DATA*NUM_HEAD)-1:0] idata;
reg [NUM_HEAD-1:0] idata_valid;
wire [(GBUS_DATA*NUM_HEAD)-1:0] odata;
wire [(GBUS_WIDTH*NUM_HEAD)-1:0] odata_valid;

consmax # (
  .GBUS_DATA(GBUS_DATA),
  .GBUS_WIDTH(GBUS_WIDTH),
  .NUM_HEAD(NUM_HEAD),
  .NUM_COL(NUM_COL)
)
consmax_inst (
  .clk(clk),
  .rstn(rst_n),
  .cfg_consmax_shift(cfg_consmax_shift),
  .lut_waddr(lut_waddr),
  .lut_wen(lut_wen),
  .lut_wdata(lut_wdata),
  .idata(idata),
  .idata_valid(idata_valid),
  .odata(odata),
  .odata_valid(odata_valid)
);

    
`endif


endmodule
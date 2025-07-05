// You must supply the following definitions (e.g., in an include file) so that
// the widths of the fixed–point data, floating–point sig/exp fields, etc., match your design:
`define SIG_WIDTH             7
`define EXP_WIDTH             8
`define IDATA_WIDTH           8
`define RECOMPUTE_SCALE_WIDTH 16
`define RMSNORM_RT_NUM        2   // number of retiming registers (example value)

// Also assume that IDLE_STATE and FFN0_STATE are defined (e.g., as parameters or `defines)
`define IDLE_STATE 2'b00
`define FFN0_STATE 2'b01

// For the rc_cfg (RC_CONFIG) packet, we assume it contains the following fields:
//   - rms_K                           : isize bits (here, 32 bits)
//   - mlp_rms_dequant_scale_square    : (sig_width+exp_width+1) bits
//   - attn_rms_dequant_scale_square   : (sig_width+exp_width+1) bits
//   - rms_rc_shift                    : RMS_RC_SHIFT_WIDTH bits (example: 4 bits)
// The overall width is computed below.
module krms #
(
    parameter integer BUS_NUM               = 8,    // Number of input buses
    parameter integer DATA_NUM_WIDTH        = 10,   // Bit–width for vector length/count
    parameter integer FIXED_SQUARE_SUM_WIDTH= 24,
    parameter integer sig_width             = 7,
    parameter integer exp_width             = 8,
    localparam        isize               = 32,
    localparam        isign               = 1,    // 1 means signed
    localparam        ieee_compliance     = 0,
    parameter integer RMS_RC_SHIFT_WIDTH  = 4
)
(
    input                           clk,
    input                           rst_n,
    input                           start,
    input                           rc_cfg_vld,
    // The rc_cfg input is a concatenation of the following fields:
    //   rms_K [isize-1:0]
    //   mlp_rms_dequant_scale_square [(sig_width+exp_width+1)-1:0]
    //   attn_rms_dequant_scale_square [(sig_width+exp_width+1)-1:0]
    //   rms_rc_shift [RMS_RC_SHIFT_WIDTH-1:0]
    // Its overall width is computed below.
    input  [RC_CFG_WIDTH-1:0]       rc_cfg,
    input  [1:0]                  control_state,       // For example, 2–bit state signal
    input                         control_state_update,
    input                         in_fixed_data_vld,
    // in_fixed_data is a BUS_NUM–element vector, each element of width `IDATA_WIDTH.
    // Since plain Verilog does not allow multi–dimensional ports, we flatten it here.
    input  signed [BUS_NUM*`IDATA_WIDTH-1:0] in_fixed_data,
    output reg [`RECOMPUTE_SCALE_WIDTH-1:0]   rc_scale,
    output reg                            rc_scale_vld
);

  //-------------------------------------------------------------------------
  // Local parameters for the RC_CFG field extraction.
  // Assume the following bit–layout for rc_cfg (from LSB to MSB):
  //   [ isize          -1 : 0 ]                             -> rms_K
  //   [ isize+(sig_width+exp_width+1)-1 : isize ]              -> mlp_rms_dequant_scale_square
  //   [ isize+2*(sig_width+exp_width+1)-1 : isize+(sig_width+exp_width+1) ] -> attn_rms_dequant_scale_square
  //   [ isize+2*(sig_width+exp_width+1)+RMS_RC_SHIFT_WIDTH-1 : isize+2*(sig_width+exp_width+1) ] -> rms_rc_shift
  //-------------------------------------------------------------------------
  localparam RC_CFG_RMS_K_MSB           = isize - 1;
  localparam RC_CFG_RMS_K_LSB           = 0;
  localparam RC_CFG_MLP_DEQUANT_MSB     = RC_CFG_RMS_K_MSB + (sig_width+exp_width+1);
  localparam RC_CFG_MLP_DEQUANT_LSB     = RC_CFG_RMS_K_MSB + 1;
  localparam RC_CFG_ATTEN_DEQUANT_MSB    = RC_CFG_MLP_DEQUANT_MSB + (sig_width+exp_width+1);
  localparam RC_CFG_ATTEN_DEQUANT_LSB    = RC_CFG_MLP_DEQUANT_MSB + 1;
  localparam RC_CFG_RMS_RC_SHIFT_MSB    = RC_CFG_ATTEN_DEQUANT_MSB + RMS_RC_SHIFT_WIDTH;
  localparam RC_CFG_RMS_RC_SHIFT_LSB    = RC_CFG_ATTEN_DEQUANT_MSB + 1;
  localparam RC_CFG_WIDTH               = RC_CFG_RMS_RC_SHIFT_MSB + 1;

  //-------------------------------------------------------------------------
  // State and configuration registers
  //-------------------------------------------------------------------------
  // start_flag: remains high once start is asserted, until rc_scale_vld pulses.
  reg start_flag;
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      start_flag <= 1'b0;
    else if (start)
      start_flag <= 1'b1;
    else if (rc_scale_vld)
      start_flag <= 1'b0;
  end

  // rc_cfg_reg: holds the configuration once rc_cfg_vld is high.
  reg [RC_CFG_WIDTH-1:0] rc_cfg_reg;
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      rc_cfg_reg <= {RC_CFG_WIDTH{1'b0}};
    else if (rc_cfg_vld)
      rc_cfg_reg <= rc_cfg;
  end

  // control_state_reg: holds the control state; assume IDLE_STATE is defined externally.
  reg [1:0] control_state_reg;
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      control_state_reg <= `IDLE_STATE;
    else if (control_state_update)
      control_state_reg <= control_state;
  end

  //-------------------------------------------------------------------------
  // Set K from the configuration: extract the field rms_K.
  //-------------------------------------------------------------------------
  wire [isize-1:0] K_ext;
  assign K_ext = rc_cfg_reg[RC_CFG_RMS_K_MSB:RC_CFG_RMS_K_LSB];

  // Conversion of fixed-point K_ext to float.
  reg [sig_width+exp_width:0] float_K;
  wire [sig_width+exp_width:0] nxt_float_K;
  always @(posedge clk) begin
    float_K <= nxt_float_K;
  end

  custom_fp_i2flt #(sig_width, exp_width, isize, isign)
    i2flt_K (
      .a(K_ext),
      .rnd(3'b000),
      .z(nxt_float_K),
      .status()
  );

  //-------------------------------------------------------------------------
  // Select the RMS dequantization scale square
  // Depending on control_state, use either the MLP or Attn value.
  // Both fields are (sig_width+exp_width+1)-bit wide.
  //-------------------------------------------------------------------------
  reg [sig_width+exp_width:0] rms_dequant_scale_square_reg;
  always @(*) begin
    if (control_state_reg == `FFN0_STATE)
      rms_dequant_scale_square_reg = rc_cfg_reg[RC_CFG_MLP_DEQUANT_MSB:RC_CFG_MLP_DEQUANT_LSB];
    else
      rms_dequant_scale_square_reg = rc_cfg_reg[RC_CFG_ATTEN_DEQUANT_MSB:RC_CFG_ATTEN_DEQUANT_LSB];
  end

  //-------------------------------------------------------------------------
  // Calculate the square of each input fixed data value.
  //-------------------------------------------------------------------------
  // Fixed data is provided as a flattened BUS_NUM–element vector (each element is `IDATA_WIDTH bits).
  // Create an internal array (reg array) for the squared values.
  reg fixed_data_square_vld;
  // Use an array of BUS_NUM elements, each 2*DATA_NUM_WIDTH bits wide.
  reg signed [2*DATA_NUM_WIDTH-1:0] fixed_data_square [0:BUS_NUM-1];

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      fixed_data_square_vld <= 1'b0;
    else
      fixed_data_square_vld <= in_fixed_data_vld & start_flag;
  end

  genvar i;
  generate
    for (i = 0; i < BUS_NUM; i = i + 1) begin : fixed_square_sum_generate_array
      // Extract the i-th input from the flattened vector.
      // The slice notation [start +: width] extracts a slice starting at the computed bit.
      always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
          fixed_data_square[i] <= 0;
        else if (in_fixed_data_vld && start_flag)
          fixed_data_square[i] <= $signed(in_fixed_data[(i+1)*`IDATA_WIDTH-1 -: `IDATA_WIDTH]) *
                                    $signed(in_fixed_data[(i+1)*`IDATA_WIDTH-1 -: `IDATA_WIDTH]);
        else
          fixed_data_square[i] <= 0;
      end
    end
  endgenerate

  //-------------------------------------------------------------------------
  // Calculate the square sum (accumulating up to K valid elements)
  //-------------------------------------------------------------------------
  reg [FIXED_SQUARE_SUM_WIDTH-1:0] fixed_square_sum, nxt_fixed_square_sum;
  reg [DATA_NUM_WIDTH-1:0]         square_sum_cnt, nxt_square_sum_cnt;
  reg                              fixed_square_sum_vld;

  integer j;
  always @(*) begin
    nxt_square_sum_cnt   = square_sum_cnt;
    nxt_fixed_square_sum = fixed_square_sum;
    // In this loop, we add the square of each element if there is a valid input
    // and if we have not yet accumulated K values.
    for (j = 0; j < BUS_NUM; j = j + 1) begin
      if (fixed_data_square_vld && (nxt_square_sum_cnt < rc_cfg_reg[RC_CFG_RMS_K_MSB:RC_CFG_RMS_K_LSB])) begin
        nxt_square_sum_cnt   = nxt_square_sum_cnt + 1;
        nxt_fixed_square_sum = nxt_fixed_square_sum + fixed_data_square[j];
      end
    end
  end

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      square_sum_cnt   <= 0;
      fixed_square_sum <= 0;
    end else if (rc_scale_vld) begin  // Reset for the next vector calculation
      square_sum_cnt   <= 0;
      fixed_square_sum <= 0;
    end else if (square_sum_cnt == rc_cfg_reg[RC_CFG_RMS_K_MSB:RC_CFG_RMS_K_LSB])
      square_sum_cnt <= rc_cfg_reg[RC_CFG_RMS_K_MSB:RC_CFG_RMS_K_LSB];
    else if (fixed_data_square_vld) begin
      square_sum_cnt   <= nxt_square_sum_cnt;
      fixed_square_sum <= nxt_fixed_square_sum;
    end
  end

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      fixed_square_sum_vld <= 1'b0;
    else if (fixed_square_sum_vld == 1'b1)
      fixed_square_sum_vld <= 1'b0;
    else if ((nxt_square_sum_cnt == rc_cfg_reg[RC_CFG_RMS_K_MSB:RC_CFG_RMS_K_LSB]) &&
             (square_sum_cnt < rc_cfg_reg[RC_CFG_RMS_K_MSB:RC_CFG_RMS_K_LSB]) &&
             (rc_cfg_reg[RC_CFG_RMS_K_MSB:RC_CFG_RMS_K_LSB] != 0))
      fixed_square_sum_vld <= 1'b1;
  end

  //-------------------------------------------------------------------------
  // Convert the fixed square sum to a floating–point value,
  // then multiply by the dequant scale square.
  //-------------------------------------------------------------------------
  reg [isize-1:0] fixed_square_sum_ext;
  wire [sig_width+exp_width:0] i2flt_square_sum_z;
  reg  [sig_width+exp_width:0] flt_square_sum;
  reg  flt_square_sum_vld;

  always @(*) begin
    fixed_square_sum_ext = fixed_square_sum;
  end

  custom_fp_i2flt #(sig_width, exp_width, isize, isign)
    i2flt_square_sum (
      .a(fixed_square_sum_ext),
      .rnd(3'b000),
      .z(i2flt_square_sum_z),
      .status()
  );

  fp_mult_pipe #(
      .sig_width(sig_width),
      .exp_width(exp_width),
      .ieee_compliance(ieee_compliance),
      .stages(5)
  ) inst_fp_mult_suqare_sum (
      .clk(clk),
      .rst_n(rst_n),
      .a(i2flt_square_sum_z),
      .b(rms_dequant_scale_square_reg),
      .z(flt_square_sum),
      .ab_valid(fixed_square_sum_vld),
      .z_valid(flt_square_sum_vld)
  );

  //-------------------------------------------------------------------------
  // Calculate "Square Sum Over K" via a floating–point division:
  //         square_sum_over_K = flt_square_sum / float_K
  //-------------------------------------------------------------------------
  wire [sig_width+exp_width:0] square_sum_over_K;
  wire                         square_sum_over_K_vld;
  fp_div_pipe #(
      .sig_width(sig_width),
      .exp_width(exp_width),
      .ieee_compliance(ieee_compliance),
      .stages(5)
  ) div_pipe_inst (
      .clk(clk),
      .rst_n(rst_n),
      .a(flt_square_sum),
      .b(float_K),
      .z(square_sum_over_K),
      .ab_valid(flt_square_sum_vld),
      .z_valid(square_sum_over_K_vld)
  );

  //-------------------------------------------------------------------------
  // Retiming registers to delay the division result
  //-------------------------------------------------------------------------
  localparam SLICE_REG_NUM = `RMSNORM_RT_NUM;
  reg [sig_width+exp_width:0] square_sum_over_K_delay;
  reg                         square_sum_over_K_vld_delay;

  // Use a generate loop with an index variable "k" to create SLICE_REG_NUM pipeline stages.
  genvar k;
  generate
    for (k = 0; k < SLICE_REG_NUM; k = k + 1) begin : invsqrt_retiming_gen_array
      reg [sig_width+exp_width:0] timing_register_float;
      reg                         timing_register;
      if (k == 0) begin
        always @(posedge clk or negedge rst_n) begin
          if (!rst_n) begin
            timing_register_float <= 0;
            timing_register       <= 0;
          end else begin
            timing_register_float <= square_sum_over_K;
            timing_register       <= square_sum_over_K_vld;
          end
        end
      end else begin
        always @(posedge clk or negedge rst_n) begin
          if (!rst_n) begin
            timing_register_float <= 0;
            timing_register       <= 0;
          end else begin
            timing_register_float <= invsqrt_retiming_gen_array[k-1].timing_register_float;
            timing_register       <= invsqrt_retiming_gen_array[k-1].timing_register;
          end
        end
      end
    end
  endgenerate

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      square_sum_over_K_delay    <= 0;
      square_sum_over_K_vld_delay<= 0;
    end else begin
      square_sum_over_K_delay    <= invsqrt_retiming_gen_array[SLICE_REG_NUM-1].timing_register_float;
      square_sum_over_K_vld_delay<= invsqrt_retiming_gen_array[SLICE_REG_NUM-1].timing_register;
    end
  end

  //-------------------------------------------------------------------------
  // Calculate One Over RMS via an inverse square root pipeline.
  //-------------------------------------------------------------------------
  wire                         invsqrt_z_vld;
  wire [sig_width+exp_width:0] invsqrt_z;
  reg [sig_width+exp_width:0]  one_over_rms;
  reg                         one_over_rms_vld;

  fp_invsqrt_pipe inst_fp_invsqrt_pipe (
      .clk(clk),
      .rst_n(rst_n),
      .x(square_sum_over_K_delay),
      .x_vld(square_sum_over_K_vld_delay),
      .y(invsqrt_z),
      .y_vld(invsqrt_z_vld)
  );

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      one_over_rms   <= 0;
      one_over_rms_vld <= 0;
    end else begin
      one_over_rms   <= invsqrt_z;
      one_over_rms_vld <= invsqrt_z_vld;
    end
  end

  //-------------------------------------------------------------------------
  // Scale the one_over_rms result to integer (fixed) format.
  // The operation performed is:
  //   one_over_rms_exp_shift = { one_over_rms[MSB], (one_over_rms[LSB:0] + {rc_cfg_reg.rms_rc_shift, {sig_width{1'b0}}} ) }
  //-------------------------------------------------------------------------
  reg [sig_width+exp_width:0] one_over_rms_exp_shift;
  reg                         one_over_rms_exp_shift_vld;

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      one_over_rms_exp_shift     <= 0;
      one_over_rms_exp_shift_vld <= 0;
    end else begin
      one_over_rms_exp_shift_vld <= one_over_rms_vld;
      one_over_rms_exp_shift[sig_width+exp_width] <= one_over_rms[sig_width+exp_width];
      one_over_rms_exp_shift[sig_width+exp_width-1:0] <= one_over_rms[sig_width+exp_width-1:0] +
         { rc_cfg_reg[RC_CFG_RMS_RC_SHIFT_MSB:RC_CFG_RMS_RC_SHIFT_LSB], {sig_width{1'b0}} };
    end
  end

  // Convert the floating–point value to an integer fixed–point value.
  wire [`RECOMPUTE_SCALE_WIDTH-1:0] nxt_rc_scale;
  custom_fp_flt2i #(sig_width, exp_width, `RECOMPUTE_SCALE_WIDTH, isign)
    i2flt_in_data (
      .a(one_over_rms_exp_shift),
      .rnd(3'b000),
      .z(nxt_rc_scale),
      .status()
  );

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      rc_scale     <= 0;
      rc_scale_vld <= 1'b0;
    end else begin
      rc_scale     <= nxt_rc_scale;
      rc_scale_vld <= one_over_rms_exp_shift_vld;
    end
  end

endmodule


module custom_fp_i2flt #(
    parameter integer sig_width = 23,   // Number of significand (mantissa) bits
    parameter integer exp_width = 8,    // Number of exponent bits
    parameter integer isize     = 32,   // Width of integer input
    parameter integer isign     = 1     // 1 = signed, 0 = unsigned
)(
    input  wire [isize-1:0] a,                         // Input integer
    input  wire [2:0]       rnd,                       // Rounding mode (unused)
    output reg  [sig_width + exp_width:0] z,           // Floating-point result
    output reg  [7:0]       status                     // Status flags
);

    // Constants
    localparam integer EXP_BIAS = (1 << (exp_width - 1)) - 1;

    // Internal wires
    reg                  sign;
    reg [isize-1:0]      abs_val;
    integer              msb_index;

    reg [exp_width-1:0]  exponent;
    reg [sig_width-1:0]  mantissa;

    // 1. Get sign and absolute value
    always @(*) begin
        if (isign && a[isize-1]) begin
            sign = 1;
            abs_val = -$signed(a);
        end else begin
            sign = 0;
            abs_val = a;
        end
    end

    // 2. Count leading zeros and calculate exponent
    always @(*) begin
        msb_index = -1;
        for (integer i = isize-1; i >= 0; i = i - 1) begin
            if (abs_val[i] == 1'b1 && msb_index == -1)
                msb_index = i;
        end

        if (msb_index == -1) begin
            exponent = 0;
            mantissa = 0;
        end else begin
            exponent = msb_index + EXP_BIAS;
            if (msb_index >= sig_width)
                mantissa = abs_val[msb_index-1 -: sig_width];
            else
                mantissa = abs_val << (sig_width - msb_index);
        end
    end

    // 3. Pack into FP format
    always @(*) begin
        if (abs_val == 0) begin
            z = 0;
            status = 8'h01;  // zero
        end else begin
            z = {sign, exponent, mantissa};
            status = 8'h00;
        end
    end

endmodule


module custom_fp_flt2i #(
    parameter integer sig_width = 23,                  // Mantissa bits
    parameter integer exp_width = 8,                   // Exponent bits
    parameter integer isize     = 16,                  // Output integer width
    parameter integer isign     = 1                    // 1 = signed output, 0 = unsigned
)(
    input  wire [sig_width + exp_width:0] a,           // Input FP number
    input  wire [2:0] rnd,                             // Rounding mode (ignored)
    output reg  [isize-1:0] z,                         // Integer output
    output reg  [7:0]       status                     // Status flags
);

    localparam integer FP_WIDTH = sig_width + exp_width + 1;
    localparam integer EXP_BIAS = (1 << (exp_width - 1)) - 1;

    // Unpack FP
    wire sign = a[FP_WIDTH-1];
    wire [exp_width-1:0] exp = a[FP_WIDTH-2:sig_width];
    wire [sig_width-1:0] frac = a[sig_width-1:0];

    wire [sig_width:0] mantissa = |exp ? {1'b1, frac} : {1'b0, frac}; // normalized or denorm
    wire signed [exp_width:0] actual_exp = $signed(exp) - EXP_BIAS;

    // Shift mantissa based on exponent
    reg [isize-1:0] shifted;
    reg overflow;

    always @(*) begin
        overflow = 0;

        if (actual_exp < 0) begin
            // Value < 1 → rounds to zero
            shifted = 0;
        end else if (actual_exp > sig_width) begin
            // Too large to represent
            shifted = {isize{1'b1}};
            overflow = 1;
        end else begin
            shifted = mantissa << actual_exp;
            if (shifted >= (1 << isize)) begin
                shifted = {isize{1'b1}};
                overflow = 1;
            end
        end
    end

    always @(*) begin
        if (exp == 0 && frac == 0) begin
            z = 0;
            status = 8'h01;  // zero
        end else if (overflow) begin
            z = isign ? (sign ? {1'b1, {isize-1{1'b0}}} : {1'b0, {isize-1{1'b1}}})
                      : {isize{1'b1}};
            status = 8'h10;  // overflow
        end else begin
            z = isign && sign ? (~shifted + 1'b1) : shifted;
            status = 8'h00;
        end
    end

endmodule



module fp_invsqrt_pipe(
    input               clk,
    input               rst_n,
    input       [15:0]  x,         // input bfloat x, sign: 1, exp: 8, frac: 7
    input               x_vld,
    
    output reg  [15:0]  y,
    output reg          y_vld
);

  //--------------------------------------------------------------------------
  // Latency = 1
  //--------------------------------------------------------------------------
  reg [15:0] startpoint;
  reg [15:0] half_x;
  reg        startpoint_half_x_vld;
  // threehalfs is a constant; note that underscores are removed for Verilog compatibility.
  wire [15:0] threehalfs;
  assign threehalfs = 16'b0011111111000000;  // equivalent to 0_01111111_1000000

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      startpoint              <= 16'd0;
      startpoint_half_x_vld   <= 1'b0;
      half_x                  <= 16'd0;
    end else if (x_vld) begin
      // Ensure proper precedence with parentheses:
      startpoint              <= 16'hBE6F - (x >> 1);
      startpoint_half_x_vld   <= x_vld;
      // Subtract 128 (8'd128) from x; extend constant to 16 bits.
      half_x                  <= x - 16'd128;
    end else begin
      startpoint_half_x_vld   <= 1'b0;
    end
  end

  //--------------------------------------------------------------------------
  // Create delay arrays (latency = 11)
  //--------------------------------------------------------------------------
  reg [15:0] startpoint_delay_array [0:10];
  reg [15:0] half_x_delay_array [0:10];

  genvar i;
  generate
    for (i = 0; i < 11; i = i + 1) begin : delay_array
      if (i == 0) begin : stage0
        always @(posedge clk or negedge rst_n) begin
          if (!rst_n) begin
            startpoint_delay_array[i] <= 16'd0;
            half_x_delay_array[i]     <= 16'd0;
          end else begin
            startpoint_delay_array[i] <= startpoint;
            half_x_delay_array[i]     <= half_x;
          end
        end
      end else begin : stageN
        always @(posedge clk or negedge rst_n) begin
          if (!rst_n) begin
            startpoint_delay_array[i] <= 16'd0;
            half_x_delay_array[i]     <= 16'd0;
          end else begin
            startpoint_delay_array[i] <= startpoint_delay_array[i-1];
            half_x_delay_array[i]     <= half_x_delay_array[i-1];
          end
        end
      end
    end
  endgenerate

  //--------------------------------------------------------------------------
  // Latency = 4+1+1 = 6: Compute startpoint_square with a multiplier pipeline
  //--------------------------------------------------------------------------
  reg [15:0] startpoint_square;
  reg [15:0] nxt_startpoint_square;
  reg        startpoint_square_vld;
  reg        nxt_startpoint_square_vld;

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      startpoint_square     <= 16'd0;
      startpoint_square_vld <= 1'b0;
    end else begin
      startpoint_square     <= nxt_startpoint_square;
      startpoint_square_vld <= nxt_startpoint_square_vld;
    end
  end

  fp_mult_pipe #(
    .sig_width(7),
    .exp_width(8),
    .ieee_compliance(0)
  ) startpoint_square_mult_inst (
    .clk     (clk),
    .rst_n   (rst_n),
    .a       (startpoint),
    .b       (startpoint),
    .z       (nxt_startpoint_square),
    .ab_valid(startpoint_half_x_vld),
    .z_valid (nxt_startpoint_square_vld)
  );

  //--------------------------------------------------------------------------
  // Latency = 1+4+1+4+1 = 11: Multiply half_x and startpoint_square using delay of half_x
  //--------------------------------------------------------------------------
  reg [15:0] half_x_mult_startpoint_square;
  reg [15:0] nxt_half_x_mult_startpoint_square;
  reg        half_x_mult_startpoint_square_vld;
  reg        nxt_half_x_mult_startpoint_square_vld;

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      half_x_mult_startpoint_square     <= 16'd0;
      half_x_mult_startpoint_square_vld <= 1'b0;
    end else begin
      half_x_mult_startpoint_square     <= nxt_half_x_mult_startpoint_square;
      half_x_mult_startpoint_square_vld <= nxt_half_x_mult_startpoint_square_vld;
    end
  end

  fp_mult_pipe #(
    .sig_width(7),
    .exp_width(8),
    .ieee_compliance(0)
  ) half_x_mult_inst (
    .clk     (clk),
    .rst_n   (rst_n),
    .a       (startpoint_square),
    .b       (half_x_delay_array[4]),
    .z       (nxt_half_x_mult_startpoint_square),
    .ab_valid(startpoint_square_vld),
    .z_valid (nxt_half_x_mult_startpoint_square_vld)
  );

  //--------------------------------------------------------------------------
  // Latency = 1+4+1+4+1+1 = 12: Subtract half_x_mult from threehalfs
  //--------------------------------------------------------------------------
  reg [15:0] threehalfs_sub;
  reg [15:0] nxt_threehlafs_sub;
  reg        threehalfs_sub_vld;
  reg        nxt_threehalfs_sub_vld;

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      threehalfs_sub     <= 16'd0;
      threehalfs_sub_vld <= 1'b0;
    end else begin
      threehalfs_sub     <= nxt_threehlafs_sub;
      threehalfs_sub_vld <= nxt_threehalfs_sub_vld;
    end
  end

  custom_fp_sub #(
    .sig_width(7), 
    .exp_width(8), 
    .ieee_compliance(0)
  ) threehalfs_sub_inst (
    .a   (threehalfs),
    .b   (half_x_mult_startpoint_square),
    .rnd (3'b000),
    .z   (nxt_threehlafs_sub),
    .status()
  );

  // Note: The valid signal for the subtraction is simply forwarded from the multiplier.
  always @(*) begin
    nxt_threehalfs_sub_vld = half_x_mult_startpoint_square_vld;
  end

  //--------------------------------------------------------------------------
  // Latency = 1+4+1+4+1+1+4+1 = 17 for y: Multiply threehalfs_sub by a delayed startpoint value
  //--------------------------------------------------------------------------
  reg [15:0] nxt_y;
  reg        nxt_y_vld;

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      y     <= 16'd0;
      y_vld <= 1'b0;
    end else begin
      y     <= nxt_y;
      y_vld <= nxt_y_vld;
    end
  end

  fp_mult_pipe #(
    .sig_width(7),
    .exp_width(8),
    .ieee_compliance(0)
  ) y_mult_inst (
    .clk     (clk),
    .rst_n   (rst_n),
    .a       (threehalfs_sub),
    .b       (startpoint_delay_array[10]),
    .z       (nxt_y),
    .ab_valid(threehalfs_sub_vld),
    .z_valid (nxt_y_vld)
  );

endmodule

module custom_fp_sub #(
    parameter integer sig_width = 7,   // number of fraction bits
    parameter integer exp_width = 8,   // number of exponent bits
    parameter integer ieee_compliance = 0  // only basic handling when 0
)(
    input  wire [sig_width+exp_width:0] a,  // input a
    input  wire [sig_width+exp_width:0] b,  // input b
    input  wire [2:0]                   rnd, // rounding mode (ignored)
    output reg  [sig_width+exp_width:0] z,   // result = a - b
    output reg  [7:0]                   status // status flags
);

    // Sign, exponent, mantissa extraction
    wire sign_a = a[sig_width + exp_width];
    wire sign_b = b[sig_width + exp_width];
    wire [exp_width-1:0] exp_a = a[sig_width + exp_width - 1 : sig_width];
    wire [exp_width-1:0] exp_b = b[sig_width + exp_width - 1 : sig_width];
    wire [sig_width-1:0] frac_a = a[sig_width-1:0];
    wire [sig_width-1:0] frac_b = b[sig_width-1:0];

    // 1. Add implicit leading 1 for normalized values
    wire [sig_width:0] mant_a = (|exp_a) ? {1'b1, frac_a} : {1'b0, frac_a};
    wire [sig_width:0] mant_b = (|exp_b) ? {1'b1, frac_b} : {1'b0, frac_b};

    // 2. Align mantissas
    wire [exp_width:0] exp_diff = (exp_a > exp_b) ? (exp_a - exp_b) : (exp_b - exp_a);
    wire [sig_width+2:0] mant_a_align = (exp_a >= exp_b) ? {mant_a, 2'b00} : ({mant_a, 2'b00} >> exp_diff);
    wire [sig_width+2:0] mant_b_align = (exp_b > exp_a) ? {mant_b, 2'b00} : ({mant_b, 2'b00} >> exp_diff);

    // 3. Determine result sign and subtract mantissas
    reg [sig_width+2:0] mant_sub;
    reg [exp_width-1:0] exp_result;
    reg sign_result;

    always @(*) begin
        if ({exp_a, mant_a} >= {exp_b, mant_b}) begin
            mant_sub = mant_a_align - mant_b_align;
            exp_result = (exp_a >= exp_b) ? exp_a : exp_b;
            sign_result = sign_a;
        end else begin
            mant_sub = mant_b_align - mant_a_align;
            exp_result = (exp_b >= exp_a) ? exp_b : exp_a;
            sign_result = ~sign_b;  // a - b = -(b - a)
        end
    end

    // 4. Normalize result
    integer shift;
    reg [sig_width-1:0] frac_norm;
    reg [2*sig_width-1:0] frac_norms;
    reg [exp_width-1:0] exp_norm;

    always @(*) begin
        shift = 0;
        for (integer i = sig_width+2; i >= 0; i = i - 1) begin
            if (mant_sub[i]) begin
                shift = sig_width + 2 - i;
                break;
            end
        end
        frac_norms = (mant_sub << shift);  
        frac_norm = frac_norms[sig_width+2:3]; // truncate extra bits 
        exp_norm = (exp_result > shift) ? (exp_result - shift) : 0;
    end

    // 5. Final packing
    always @(*) begin
        z = {sign_result, exp_norm, frac_norm};
        status = 8'b0;
        if (mant_sub == 0)
            z = {1'b0, {exp_width{1'b0}}, {sig_width{1'b0}}};  // exact zero
    end

endmodule

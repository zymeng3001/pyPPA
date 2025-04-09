
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

  DW_fp_sub #(
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

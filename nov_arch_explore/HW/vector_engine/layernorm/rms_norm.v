// Copyright (c) 2024, Saligane's Group at University of Michigan and Google Research
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

module rms_norm
#(
    parameter integer BUS_NUM = 8,                     // input bus num
    parameter integer DATA_NUM_WIDTH = 10,             // layernorm vector length reg width
    parameter integer SCALA_POS_WIDTH = 5,
    parameter integer FIXED_SQUARE_SUM_WIDTH = 24,
    // flatten the arrays into a one-dimensional vector
    parameter integer IN_FLOAT_DATA_ARRAY_DEPTH = (${n_embd} / BUS_NUM + 1), 
    parameter integer GAMMA_ARRAY_DEPTH         = IN_FLOAT_DATA_ARRAY_DEPTH,
    parameter integer sig_width = 7,
    parameter integer exp_width = 8,
    parameter integer isize     = 32,
    parameter integer isign     = 1,                   // signed integer number
    parameter integer ieee_compliance = 0              // No support to NaN and denormals
)
(
    input                         clk,
    input                         rst_n,

    // input channel
    input [DATA_NUM_WIDTH-1:0]    in_data_num,
    input                         in_data_num_vld,

    // Flattened bus of gamma values: each element is (sig_width+exp_width+1) bits
    input [BUS_NUM*(sig_width+exp_width+1)-1:0] in_gamma,
    input [BUS_NUM-1:0]           in_gamma_vld,

    // Flattened bus of fixed data (each element is 8-bit signed)
    input signed [BUS_NUM*8-1:0]  in_fixed_data,
    input [BUS_NUM-1:0]           in_fixed_data_vld,

    input signed [SCALA_POS_WIDTH-1:0] in_scale_pos,
    input                         in_scale_pos_vld, 

    input signed [SCALA_POS_WIDTH-1:0] out_scale_pos,  
    input                         out_scale_pos_vld,

    input [DATA_NUM_WIDTH-1:0]    in_K,
    input                         in_K_vld,

    // Flattened bus for output fixed data (each element is 8-bit signed)
    output reg signed [BUS_NUM*8-1:0] out_fixed_data,
    output reg [BUS_NUM-1:0]      out_fixed_data_vld,
    output reg                  out_fixed_data_last  // helps design more efficiently
);

  // --------------------------------------------------------
  // Internal signal declarations (flattened and multi–dim arrays)
  // --------------------------------------------------------
  
  // Registers for scalar values (assigned synchronously)
  reg [DATA_NUM_WIDTH-1:0] data_num;
  reg [DATA_NUM_WIDTH-1:0] K;
  reg [sig_width+exp_width:0] float_K;
  wire [sig_width+exp_width:0] nxt_float_K;
  
  reg signed [SCALA_POS_WIDTH-1:0] in_scale_pos_reg;
  reg signed [SCALA_POS_WIDTH-1:0] out_scale_pos_reg;
  
  // Two-dimensional array signals are declared as arrays of vectors.
  // For example, fixed_data_square[i] is a signed value of width 2*DATA_NUM_WIDTH.
  reg signed [2*DATA_NUM_WIDTH-1:0] fixed_data_square [0:BUS_NUM-1];
  reg [0:BUS_NUM-1]                            fixed_data_square_vld ;
  
  reg [FIXED_SQUARE_SUM_WIDTH-1:0] fixed_square_sum, nxt_fixed_square_sum;
  reg [DATA_NUM_WIDTH-1:0]         square_sum_cnt, nxt_square_sum_cnt;
  reg                              fixed_square_sum_vld;
  
  wire [sig_width+exp_width:0]     i2flt_square_sum_z;
  reg  [sig_width+exp_width:0]     float_square_sum;
  reg                              float_square_sum_vld;
  
  wire [sig_width+exp_width:0]     square_sum_over_K;
  wire                             square_sum_over_K_vld;
  
  reg  [sig_width+exp_width:0]     one_over_rms;
  wire [sig_width+exp_width:0]     invsqrt_z;
  reg                              one_over_rms_vld;
  
  // The following arrays hold BUS_NUM elements; they are implemented as arrays of vectors.
  // (Using indexed array notation available in Verilog–2001; if not supported, further flatten them.)
  wire [sig_width+exp_width:0]     i2flt_in_data_z [0:BUS_NUM-1];
  reg  [sig_width+exp_width:0]     in_float_data [0:BUS_NUM-1];
  reg  [0:BUS_NUM-1]                            in_float_data_vld ;

  // FIFOs for storing float input data and gamma (each slot holds BUS_NUM numbers)
  reg [ (sig_width+exp_width+1)*BUS_NUM -1:0 ] in_float_data_array [0:IN_FLOAT_DATA_ARRAY_DEPTH-1];
  reg [BUS_NUM-1:0]                           in_float_data_vld_array [0:IN_FLOAT_DATA_ARRAY_DEPTH-1];
  reg [ (sig_width+exp_width+1)*BUS_NUM -1:0 ] in_float_data_array_wr_slot;
  reg [ (sig_width+exp_width+1)*BUS_NUM -1:0 ] in_float_data_array_rd_slot;
  reg [BUS_NUM-1:0]                           in_float_data_vld_array_slot;
  reg [$clog2(IN_FLOAT_DATA_ARRAY_DEPTH)-1:0]   in_float_data_array_wr_ptr, in_float_data_array_rd_ptr;

  reg [ (sig_width+exp_width+1)*BUS_NUM -1:0 ] gamma_array [0:GAMMA_ARRAY_DEPTH-1];
  reg [ (sig_width+exp_width+1)*BUS_NUM -1:0 ] gamma_array_wr_slot;
  reg [ (sig_width+exp_width+1)*BUS_NUM -1:0 ] gamma_array_rd_slot;
  reg [BUS_NUM-1:0]                           gamma_vld_array [0:GAMMA_ARRAY_DEPTH-1];
  reg [$clog2(GAMMA_ARRAY_DEPTH)-1:0]           gamma_array_wr_ptr, gamma_array_rd_ptr;
  
  reg                                           internal_array_rd_en;
  reg                                           internal_array_rd_en_delay1;
  reg [$clog2(GAMMA_ARRAY_DEPTH*BUS_NUM)-1:0]     internal_array_rd_cnt;
  
  reg [sig_width+exp_width:0]     x_float [0:BUS_NUM-1];
  reg [sig_width+exp_width:0]     gamma_float [0:BUS_NUM-1];
  reg [BUS_NUM-1:0]               x_and_gamma_vld;
  
  wire [sig_width+exp_width:0]    x_mult_with_gamma [0:BUS_NUM-1];
  wire [BUS_NUM-1:0]              x_mult_with_gamma_vld;
  
  wire [sig_width+exp_width:0]    float_RMSnorm [0:BUS_NUM-1];
  wire [BUS_NUM-1:0]              float_RMSnorm_vld;
  
  reg [sig_width+exp_width:0]     float_RMSnorm_flt2i [0:BUS_NUM-1];
  reg [BUS_NUM-1:0]              float_RMSnorm_flt2i_vld;
  
  wire signed [isize-1:0]         fixed_RMSnorm [0:BUS_NUM-1];
  reg signed [7:0]                nxt_out_fixed_data [0:BUS_NUM-1];
  reg [15:0]                      out_fixed_data_last_cnt;
  
  // Temporary register to hold the extended fixed data.
  // Note: We extract each bus element from the flattened in_fixed_data.
  reg signed [isize-1:0]          in_fixed_data_ext [0:BUS_NUM-1];


  // --------------------------------------------------------
  // Set up the data num (length of RMSnorm vector), scale positions, and gamma array
  // --------------------------------------------------------

  // data_num register
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      data_num <= 0;
    else if (in_data_num_vld)
      data_num <= in_data_num;
  end

  // in_scale_pos_reg
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      in_scale_pos_reg <= 0;
    else if (in_scale_pos_vld)
      in_scale_pos_reg <= in_scale_pos;
  end

  // out_scale_pos_reg
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      out_scale_pos_reg <= 0;
    else if (out_scale_pos_vld)
      out_scale_pos_reg <= out_scale_pos;
  end

  // Gamma array write slot generation. (The flattened vector in_gamma is sliced.)
  genvar i;
  generate
    for (i = 0; i < BUS_NUM; i = i+1) begin : gamma_array_wr_slot_generate_array
      // Using the "-:" slicing operator to extract each BUS_NUM element slice.
      assign gamma_array_wr_slot[(i+1)*(sig_width+exp_width+1)-1 -: (sig_width+exp_width+1)] =
             in_gamma[(i+1)*(sig_width+exp_width+1)-1 -: (sig_width+exp_width+1)];
    end
  endgenerate

  // gamma_array_wr_ptr and storage of the gamma array and its valid bits
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      gamma_array_wr_ptr <= 0;
    else if (out_fixed_data_last)
      gamma_array_wr_ptr <= 0;  // reset for next vector input
    else if (|in_gamma_vld) begin // At least one valid input
      gamma_array[gamma_array_wr_ptr] <= gamma_array_wr_slot;
      gamma_vld_array[gamma_array_wr_ptr] <= in_gamma_vld;
      gamma_array_wr_ptr <= gamma_array_wr_ptr + 1;
    end
  end

  // --------------------------------------------------------
  // Set the K value before each vector computation
  // --------------------------------------------------------
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      K <= 0;
    else if (in_K_vld)
      K <= in_K;
  end

  // Instance: convert fixed K to float (DW_fp_i2flt module)
  always @(posedge clk) begin
    float_K <= nxt_float_K;
  end

  i2flt_rms #(sig_width, exp_width, isize, isign)
    i2flt_K ( 
      .a(K), 
      .rnd(3'b000), 
      .z(nxt_float_K), 
      .status() 
    );

  // --------------------------------------------------------
  // Convert input fixed data to float data and store in an array.
  // --------------------------------------------------------
  generate
    for (i = 0; i < BUS_NUM; i = i+1) begin : i2flt_in_data_generate_array
      // Extract each 8-bit fixed data element from the flattened input.
      always @(*) begin
        in_fixed_data_ext[i] = $signed(in_fixed_data[(i+1)*8-1 -: 8]);
      end

      i2flt_rms #(sig_width, exp_width, isize, isign)
        i2flt_in_data ( 
           .a(in_fixed_data_ext[i]), 
           .rnd(3'b000), 
           .z(i2flt_in_data_z[i]), 
           .status() 
        );
    end
  endgenerate

  // Convert each fixed data value to float (with scaling) and prepare for storage in FIFO.
  generate 
    for (i = 0; i < BUS_NUM; i = i+1) begin : in_fix2float_generate_array
      always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
          in_float_data_vld[i] <= 0;
          in_float_data[i]     <= 0;
        end else begin
          in_float_data_vld[i] <= in_fixed_data_vld[i];
          if (i2flt_in_data_z[i] == 0)
            in_float_data[i] <= 0;
          else begin
            in_float_data[i][sig_width+exp_width-1:0] <= i2flt_in_data_z[i][sig_width+exp_width-1:0] - $signed({in_scale_pos_reg, {sig_width{1'b0}}});
            in_float_data[i][sig_width+exp_width] <= i2flt_in_data_z[i][sig_width+exp_width];
          end
        end
      end
      // Slice each float datum into the write slot vector.
      assign in_float_data_array_wr_slot[(i+1)*(sig_width+exp_width+1)-1 -: (sig_width+exp_width+1)] = in_float_data[i];
      assign in_float_data_vld_array_slot[i] = in_float_data_vld[i];
    end
  endgenerate

  // Write the float data into the FIFO array.
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      in_float_data_array_wr_ptr <= 0;
    end else if (out_fixed_data_last) begin
      in_float_data_array_wr_ptr <= 0;  // reset for next vector input
    end else if (in_float_data_vld > 0) begin  // At least one valid input
      in_float_data_array[in_float_data_array_wr_ptr] <= in_float_data_array_wr_slot;
      in_float_data_vld_array[in_float_data_array_wr_ptr] <= in_float_data_vld_array_slot;
      in_float_data_array_wr_ptr <= in_float_data_array_wr_ptr + 1;
    end
  end

  // --------------------------------------------------------
  // Calculate the square of each fixed input value
  // --------------------------------------------------------
  reg fixed_data_square_input_gating;
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      fixed_data_square_input_gating <= 0;
    else if (out_fixed_data_last)
      fixed_data_square_input_gating <= 0;
    else if (fixed_square_sum_vld)
      fixed_data_square_input_gating <= 1;
  end

  generate 
    for (i = 0; i < BUS_NUM; i = i+1) begin : fixed_square_sum_generate_array
      always @(posedge clk) begin
        if (!fixed_data_square_input_gating) begin
          fixed_data_square[i] <= $signed(in_fixed_data[(i+1)*8-1 -: 8]) * $signed(in_fixed_data[(i+1)*8-1 -: 8]);
          fixed_data_square_vld[i] <= in_fixed_data_vld[i];
        end else begin
          fixed_data_square[i] <= 0;
          fixed_data_square_vld[i] <= 0;
        end
      end
    end
  endgenerate

  // --------------------------------------------------------
  // Calculate the square sum and count up to K inputs.
  // --------------------------------------------------------
  always @(*) begin
    nxt_square_sum_cnt   = square_sum_cnt;
    nxt_fixed_square_sum = fixed_square_sum;
    for (integer i = 0; i < BUS_NUM; i = i + 1) begin
      if (fixed_data_square_vld[i]) begin
        nxt_square_sum_cnt   = nxt_square_sum_cnt + 1;
        nxt_fixed_square_sum = nxt_fixed_square_sum + fixed_data_square[i];
        // if(nxt_square_sum_cnt == K)
        //   disable for;
      end
    end
  end

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      square_sum_cnt   <= 0;
      fixed_square_sum <= 0;
    end else if (out_fixed_data_last) begin  // reset for next vector calculation
      square_sum_cnt   <= 0;
      fixed_square_sum <= 0;
    end else if (square_sum_cnt == K)
      square_sum_cnt <= K;  // update float_square_sum only once
    else if (|fixed_data_square_vld) begin
      square_sum_cnt   <= nxt_square_sum_cnt;
      fixed_square_sum <= nxt_fixed_square_sum;
    end
  end

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      fixed_square_sum_vld <= 0;
    else if (fixed_square_sum_vld == 1)
      fixed_square_sum_vld <= 0;
    else if (nxt_square_sum_cnt == K && square_sum_cnt < K && K != 0)
      fixed_square_sum_vld <= 1;
  end

  // --------------------------------------------------------
  // Convert fixed square sum to a floating–point value.
  // --------------------------------------------------------
  reg [isize-1:0] fixed_square_sum_ext;
  assign fixed_square_sum_ext = fixed_square_sum;
  
  i2flt_rms #(sig_width, exp_width, isize, isign)
    i2flt_square_sum ( 
      .a(fixed_square_sum_ext), 
      .rnd(3'b000), 
      .z(i2flt_square_sum_z), 
      .status() 
    );

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      float_square_sum   <= 0;
      float_square_sum_vld <= 0;
    end else if (out_fixed_data_last) begin  // reset for next vector calculation
      float_square_sum_vld <= 0;
      float_square_sum   <= 0;
    end else if (fixed_square_sum_vld) begin
      float_square_sum_vld <= 1;
      if(i2flt_square_sum_z == 0)
        float_square_sum <= 0;
      else begin
        float_square_sum[sig_width+exp_width-1:0] <= i2flt_square_sum_z[sig_width+exp_width-1:0] - $signed({in_scale_pos_reg, {sig_width{1'b0}}});
        float_square_sum[sig_width+exp_width] <= i2flt_square_sum_z[sig_width+exp_width];
      end
    end else
      float_square_sum_vld <= 0;
  end

  // --------------------------------------------------------
  // Calculate Square Sum Over K using a division pipeline.
  // --------------------------------------------------------
  fp_div_pipe #(
    .sig_width(sig_width),
    .exp_width(exp_width),
    .ieee_compliance(ieee_compliance),
    .stages(4)
  ) div_pipe_inst ( 
    .clk(clk),
    .rst_n(rst_n),
    .a(float_square_sum),
    .b(float_K),
    .z(square_sum_over_K),
    .ab_valid(float_square_sum_vld),
    .z_valid(square_sum_over_K_vld)
  );

  // Delay registers for proper timing
  localparam SLICE_REG_NUM = 1;
  reg [sig_width+exp_width:0] square_sum_over_K_delay;
  reg                         square_sum_over_K_vld_delay;
  genvar j;
  generate
    for (j = 0; j < SLICE_REG_NUM; j = j + 1) begin : invsqrt_retiming_gen_array
      reg [sig_width+exp_width:0] timing_register_float;
      reg                         timing_register;
      if (j == 0) begin
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
            timing_register_float <= invsqrt_retiming_gen_array[j-1].timing_register_float;
            timing_register       <= invsqrt_retiming_gen_array[j-1].timing_register;
          end
        end
      end
    end
  endgenerate

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      square_sum_over_K_delay    <= 0;
      square_sum_over_K_vld_delay <= 0;
    end else begin
      square_sum_over_K_delay    <= invsqrt_retiming_gen_array[SLICE_REG_NUM-1].timing_register_float;
      square_sum_over_K_vld_delay<= invsqrt_retiming_gen_array[SLICE_REG_NUM-1].timing_register;
    end
  end

  // --------------------------------------------------------
  // Calculate One Over RMS using an inverse square root pipeline.
  // --------------------------------------------------------
  wire invsqrt_z_vld;
  fp_invsqrt_pipe inst_fp_invsqrt_pipe(
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

  // --------------------------------------------------------
  // Calculate x multiplied with gamma.
  // --------------------------------------------------------
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      internal_array_rd_en <= 0;
    else if (one_over_rms_vld)
      internal_array_rd_en <= 1;
    else if (internal_array_rd_cnt >= data_num - BUS_NUM && data_num != 0 && internal_array_rd_en)
      internal_array_rd_en <= 0;
  end

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      internal_array_rd_cnt <= 0;
    else if (out_fixed_data_last)
      internal_array_rd_cnt <= 0; // reset for next vector input
    else if (internal_array_rd_en)
      internal_array_rd_cnt <= internal_array_rd_cnt + BUS_NUM;
  end

  // Read pointer update: retrieve data from float data FIFO and gamma array.
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      in_float_data_array_rd_ptr <= 0;
      gamma_array_rd_ptr         <= 0;
      in_float_data_array_rd_slot <= 0;
      gamma_array_rd_slot        <= 0;
    end else if (out_fixed_data_last) begin
      in_float_data_array_rd_ptr <= 0;
      gamma_array_rd_ptr         <= 0;
    end else if (internal_array_rd_en) begin
      in_float_data_array_rd_ptr <= in_float_data_array_rd_ptr + 1;
      gamma_array_rd_ptr         <= gamma_array_rd_ptr + 1;
      in_float_data_array_rd_slot <= in_float_data_array[in_float_data_array_rd_ptr];
      gamma_array_rd_slot         <= gamma_array[gamma_array_rd_ptr];
    end
  end

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      x_and_gamma_vld <= 0;
    else
      x_and_gamma_vld <= in_float_data_vld_array[in_float_data_array_rd_ptr] & {BUS_NUM{internal_array_rd_en}};
  end

  // Extract each bus element from the read slots.
  generate
    for (i = 0; i < BUS_NUM; i = i+1) begin : x_gamma_float_generate_array
      assign x_float[i] = in_float_data_array_rd_slot[(i+1)*(sig_width+exp_width+1)-1 -: (sig_width+exp_width+1)];
      assign gamma_float[i] = gamma_array_rd_slot[(i+1)*(sig_width+exp_width+1)-1 -: (sig_width+exp_width+1)];
    end
  endgenerate

  // Multiply x with gamma.
  generate
    for (i = 0; i < BUS_NUM; i = i+1) begin : x_mult_with_gamma_generate_array
      fp_mult_pipe #(
         .sig_width(sig_width),
         .exp_width(exp_width),
         .ieee_compliance(ieee_compliance),
         .stages(3)
      ) inst_fp_mult_x_gamma( 
         .clk(clk),
         .rst_n(rst_n),
         .a(x_float[i]),
         .b(gamma_float[i]),
         .z(x_mult_with_gamma[i]),
         .ab_valid(x_and_gamma_vld[i]),
         .z_valid(x_mult_with_gamma_vld[i])
      );
    end
  endgenerate

  // --------------------------------------------------------
  // Calculate float RMSnorm by multiplying with one_over_rms.
  // --------------------------------------------------------
  generate
    for (i = 0; i < BUS_NUM; i = i+1) begin : float_RMSnorm_generate_array
      fp_mult_pipe #(
         .sig_width(sig_width),
         .exp_width(exp_width),
         .ieee_compliance(ieee_compliance),
         .stages(3)
      ) inst_fp_mult_float_RMSnorm( 
         .clk(clk),
         .rst_n(rst_n),
         .a(x_mult_with_gamma[i]),
         .b(one_over_rms),
         .z(float_RMSnorm[i]),
         .ab_valid(x_mult_with_gamma_vld[i]),
         .z_valid(float_RMSnorm_vld[i])
      );
    end
  endgenerate

  // --------------------------------------------------------
  // Convert the float RMSnorm to fixed point.
  // --------------------------------------------------------
  generate
    for (i = 0; i < BUS_NUM; i = i+1) begin : fixed_RMSnorm_generate_array
      always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
          float_RMSnorm_flt2i[i]   <= 0;
          float_RMSnorm_flt2i_vld[i] <= 0;
        end else begin
          if (float_RMSnorm[i] == 0)
            float_RMSnorm_flt2i[i] <= 0;
          else begin
            float_RMSnorm_flt2i[i][sig_width+exp_width-1:0] <= float_RMSnorm[i][sig_width+exp_width-1:0] +
                                                                $signed({out_scale_pos_reg, {sig_width{1'b0}}});
            float_RMSnorm_flt2i[i][sig_width+exp_width] <= float_RMSnorm[i][sig_width+exp_width];
          end
          float_RMSnorm_flt2i_vld[i] <= float_RMSnorm_vld[i];
        end
      end

      fp2int_rms #(
          .EXP_BIT(exp_width),
          .MAT_BIT(sig_width),
          .IDATA_BIT(exp_width+sig_width+1),
          .ODATA_BIT(8),
          .CDATA_BIT(8)
        )
        fp2int_in_data_inst ( 
          .idata(float_RMSnorm_flt2i[i]),
          .odata(fixed_RMSnorm[i])
      );

      always @(*) begin
        nxt_out_fixed_data[i] = fixed_RMSnorm[i];
        if ($signed(fixed_RMSnorm[i]) > 127)
          nxt_out_fixed_data[i] = 8'd127;
        else if ($signed(fixed_RMSnorm[i]) < -128)
          nxt_out_fixed_data[i] = -8'd128;
      end

      always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
          out_fixed_data_vld[i] <= 0;
          out_fixed_data[i*8 +: 8] <= 0;
        end else begin
          out_fixed_data_vld[i] <= float_RMSnorm_flt2i_vld[i];
          out_fixed_data[i*8 +: 8] <= nxt_out_fixed_data[i];
        end
      end
    end
  endgenerate

  // --------------------------------------------------------
  // Generate the out_fixed_data_last signal for vector control.
  // --------------------------------------------------------
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      out_fixed_data_last_cnt <= 0;
      out_fixed_data_last     <= 0;
    end else if (out_fixed_data_last_cnt >= data_num - BUS_NUM && data_num != 0 && |float_RMSnorm_flt2i_vld) begin
      out_fixed_data_last     <= 1;
      out_fixed_data_last_cnt <= 0;
    end else if (|float_RMSnorm_flt2i_vld) begin
      out_fixed_data_last_cnt <= out_fixed_data_last_cnt + BUS_NUM;
    end else begin
      out_fixed_data_last <= 0;
    end
  end

endmodule

// Example open–source i2flt module based on concepts from OpenCores FPU projects
module i2flt_rms #(
  parameter SIG_WIDTH = 7,      // Number of fraction bits
  parameter EXP_WIDTH = 8,      // Number of exponent bits
  parameter ISIZE     = 32,     // Integer input width
  parameter ISIGN     = 1       // 1 means signed integer input
)(
  input  [ISIZE-1:0]         a,    // Integer input
  input  [2:0]               rnd,  // Rounding mode (not fully implemented)
  output reg [SIG_WIDTH+EXP_WIDTH:0] z, // Floating–point output {sign, exp, fraction}
  output reg               status  // Status flag (e.g. for exceptions)
);

  // Count Leading Zeros (CLZ)
  // A fully synthesizable unrolled implementation (note: a for–loop here will be unrolled during synthesis)
  function integer clz;
    input [ISIZE-1:0] value;
    integer i;
    integer found;
    begin
      found = 0;
      clz = 0;
      for (i = ISIZE-1; i >= 0; i = i - 1) begin
        if (!found && value[i])
          begin
            clz = ISIZE - i - 1;
            found = 1;
          end
      end
    end
  endfunction

  reg sign;
  reg [ISIZE-1:0] abs_val;
  integer lz;      // count of leading zeros
  integer n;       // number of significant bits
  integer bias;    // exponent bias
  reg [EXP_WIDTH-1:0] exp_field;
  reg [SIG_WIDTH-1:0] frac_field;
  integer shift_amt;

  always @(*) begin
    status = 1'b0;
    // For signed inputs, get sign and absolute value.
    if (ISIGN) begin
      sign = a[ISIZE-1];
      abs_val = (a[ISIZE-1]) ? (~a + 1) : a;
    end else begin
      sign    = 1'b0;
      abs_val = a;
    end

    // Special-case zero.
    if (abs_val == 0) begin
      z = {(SIG_WIDTH+EXP_WIDTH+1){1'b0}};
    end else begin
      lz = clz(abs_val);
      n  = ISIZE - lz;
      bias = (1 << (EXP_WIDTH-1)) - 1;
      exp_field = n - 1 + bias;
      // Normalize the input (truncation only)
      if (n > (SIG_WIDTH + 1)) begin
        shift_amt = n - 1 - SIG_WIDTH;
        frac_field = (abs_val >> shift_amt) & ((1 << SIG_WIDTH) - 1);
      end else begin
        shift_amt = SIG_WIDTH - (n - 1);
        frac_field = (abs_val << shift_amt) & ((1 << SIG_WIDTH) - 1);
      end
      z = {sign, exp_field, frac_field};
    end
  end

endmodule

module fp2int_rms #(
    parameter   EXP_BIT = 8,
    parameter   MAT_BIT = 7,

    parameter   IDATA_BIT = EXP_BIT + MAT_BIT + 1,  // FP-Input
    parameter   ODATA_BIT = 8,  // INT-Output
    parameter   CDATA_BIT = 8   // Config
)(
    // Data Signals
    input   [IDATA_BIT-1:0] idata,
    output  [ODATA_BIT-1:0] odata
);

    localparam  EXP_BASE = 2 ** (EXP_BIT - 1) - 1;

    // Extract Sign, Exponent and Mantissa Field
    reg                     idata_sig;
    reg     [EXP_BIT-1:0]   idata_exp;
    reg     [MAT_BIT:0]     idata_mat;

    always @(*) begin
        idata_sig = idata[IDATA_BIT-1];
        idata_exp = idata[MAT_BIT+:EXP_BIT];
        idata_mat = {1'b1, idata[MAT_BIT-1:0]};
    end

    // Shift and Round Mantissa to Integer
    reg     [MAT_BIT:0]     mat_shift;
    reg     [MAT_BIT:0]     mat_round;

    always @(*) begin
        if (idata_exp >= EXP_BASE) begin    // >= 1.0
            if (MAT_BIT <= ((idata_exp - EXP_BASE))) begin // Overflow
                mat_shift = {(MAT_BIT){1'b1}};
                mat_round = mat_shift;
            end
            else begin
                mat_shift = idata_mat >> (MAT_BIT  - (idata_exp - EXP_BASE));
                mat_round = mat_shift[MAT_BIT:1] + mat_shift[0];
            end
        end
        else begin  // <= 1.0
            if (0 < (EXP_BASE - idata_exp)) begin // Underflow
                mat_shift = {(MAT_BIT){1'b0}};
                mat_round = mat_shift;
            end
            else begin
                mat_shift = idata_mat >> (MAT_BIT + (EXP_BASE - idata_exp));
                mat_round = mat_shift[MAT_BIT:1] + mat_shift[0];
            end
        end
    end

    // Convert to 2's Complementary Integer
    assign  odata = {idata_sig, idata_sig ? (~mat_round[MAT_BIT-:ODATA_BIT] + 1'b1) : 
                                              mat_round[MAT_BIT-:ODATA_BIT]};

endmodule


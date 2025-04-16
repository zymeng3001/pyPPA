module layer_norm #(
    parameter integer BUS_NUM = 8,
    parameter integer DATA_NUM_WIDTH = 10,
    parameter integer SCALA_POS_WIDTH = 5,
    parameter integer FIXED_ACC_WIDTH = 32,
    parameter integer IN_FIXED_DATA_ARRAY_DEPTH = (${n_embd} / BUS_NUM + 1),
    parameter integer DATA_ARRAY_DEPTH_WIDTH = $clog2(IN_FIXED_DATA_ARRAY_DEPTH), 
    parameter integer GAMMA_ARRAY_DEPTH = IN_FIXED_DATA_ARRAY_DEPTH,
    parameter integer BETA_ARRAY_DEPTH = IN_FIXED_DATA_ARRAY_DEPTH,
    parameter integer sig_width = 7,
    parameter integer exp_width = 8,
    parameter integer isize     = 32,
    parameter integer isign     = 1,
    parameter integer ieee_compliance = 0
)(
    input clk,
    input rstn,

    input [DATA_NUM_WIDTH-1:0] in_data_num,
    input                      in_data_num_vld,

    input signed [BUS_NUM*8-1:0] in_fixed_data,
    input [BUS_NUM-1:0] in_fixed_data_vld,

    input [BUS_NUM*(sig_width+exp_width+1)-1:0] in_gamma,
    input [BUS_NUM*(sig_width+exp_width+1)-1:0] in_beta,
    input [BUS_NUM-1:0] in_gamma_vld,
    input [BUS_NUM-1:0] in_beta_vld,

    input signed [SCALA_POS_WIDTH-1:0] in_scale_pos,
    input                              in_scale_pos_vld,

    input signed [SCALA_POS_WIDTH-1:0] out_scale_pos,
    input                              out_scale_pos_vld,

    output reg signed [BUS_NUM*8-1:0] out_fixed_data,
    output reg [BUS_NUM-1:0] out_fixed_data_vld,
    output reg out_fixed_data_last
);

parameter integer COMPUTE_MEAN = 2'b00;
parameter integer COMPUTE_VAR  = 2'b01;
parameter integer COMPUTE_NORM = 2'b10;

integer i;

reg [1:0] state;
reg [1:0] next_state;

reg [DATA_NUM_WIDTH-1:0] data_num;

reg signed [SCALA_POS_WIDTH-1:0] in_scale_pos_reg;
reg signed [SCALA_POS_WIDTH-1:0] out_scale_pos_reg;

// reg for calculating mean value
reg signed [FIXED_ACC_WIDTH-1:0] adder_tree_value;
reg signed adder_tree_value_vld;

reg signed [FIXED_ACC_WIDTH-1:0] adder_tree_value_var;
reg signed adder_tree_value_var_vld;

reg signed [FIXED_ACC_WIDTH-1:0] mean_value;
reg signed mean_value_vld;

reg data_fifo_wren, data_fifo_rden, gamma_fifo_wren, gamma_fifo_rden, beta_fifo_wren, beta_fifo_rden;
reg [BUS_NUM*8-1:0] data_fifo_din;
reg [BUS_NUM*(sig_width+exp_width+1)-1:0] gamma_fifo_din;
reg [BUS_NUM*(sig_width+exp_width+1)-1:0] beta_fifo_din;

wire [BUS_NUM*8-1:0] data_fifo_dout;
wire [BUS_NUM*(sig_width+exp_width+1)-1:0] gamma_fifo_dout;
wire [BUS_NUM*(sig_width+exp_width+1)-1:0] beta_fifo_dout;

reg [BUS_NUM*(sig_width+exp_width+1)-1:0] flt_x_sub_mean;
reg flt_x_sub_mean_vld;
reg [BUS_NUM*(sig_width+exp_width+1)-1:0] x_times_gamma;
reg x_times_gamma_vld;
reg [BUS_NUM*(sig_width+exp_width+1)-1:0] x_times_gamma_over_sqrt_var;
reg x_times_gamma_over_sqrt_var_vld;
reg [BUS_NUM*(sig_width+exp_width+1)-1:0] x_times_gamma_over_sqrt_var_plus_beta;
reg x_times_gamma_over_sqrt_var_plus_beta_vld;
reg [BUS_NUM*(sig_width+exp_width+1)-1:0] float_layernorm_flt2i;
reg [BUS_NUM-1:0] float_layernorm_flt2i_vld;


wire data_fifo_full, data_fifo_empty;
wire gamma_fifo_full, gamma_fifo_empty;
wire beta_fifo_full, beta_fifo_empty;

reg [FIXED_ACC_WIDTH-1: 0] data_sum_acc_reg;
reg                        data_sum_acc_reg_vld;
reg [DATA_ARRAY_DEPTH_WIDTH-1:0] data_sum_acc_reg_cntr;

reg [FIXED_ACC_WIDTH-1: 0] data_var_acc_reg;
reg                        data_var_reg_vld;
reg [DATA_ARRAY_DEPTH_WIDTH-1:0] data_var_reg_cntr;

reg [sig_width+exp_width:0] inv_sqrt_in_reg, inv_sqrt_in_temp;
reg inv_sqrt_in_reg_vld;

reg [sig_width+exp_width:0] inv_sqrt_out_reg;
reg inv_sqrt_out_reg_vld;

// mean value register
reg signed [7:0] mean_value_reg;
reg mean_value_reg_vld;

// x-mean registers
reg signed [BUS_NUM*8-1:0] x_sub_mean;
reg x_sub_mean_vld;

// instantiate adder_tree_layernorm
adder_tree #(
    .ADD_IDATA_BIT(8),
    .ADD_ODATA_BIT(FIXED_ACC_WIDTH),
    .MAC_NUM(BUS_NUM)
) adder_tree_layernorm_inst (
    .clk(clk),
    .rstn(rstn),
    .idata(in_fixed_data),
    .idata_valid((state == COMPUTE_MEAN) ? in_fixed_data_vld : 0),
    .odata(adder_tree_value),
    .odata_valid(adder_tree_value_vld)
);

always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        data_sum_acc_reg <= 0;
        data_sum_acc_reg_vld <= 0;
        data_sum_acc_reg_cntr <= 0;
        state <= 0;
    end else begin
        data_sum_acc_reg <= (data_sum_acc_reg_cntr < IN_FIXED_DATA_ARRAY_DEPTH && adder_tree_value_vld) ? data_sum_acc_reg + adder_tree_value :
                            (data_sum_acc_reg_cntr >= IN_FIXED_DATA_ARRAY_DEPTH) ? 0 : data_sum_acc_reg;
        data_sum_acc_reg_vld <= (data_sum_acc_reg_cntr == IN_FIXED_DATA_ARRAY_DEPTH - 1);
        data_sum_acc_reg_cntr <= (data_sum_acc_reg_cntr < IN_FIXED_DATA_ARRAY_DEPTH && adder_tree_value_vld) ? data_sum_acc_reg_cntr + 1 :
                                 (data_sum_acc_reg_cntr >= IN_FIXED_DATA_ARRAY_DEPTH) ? 0 : data_sum_acc_reg_cntr;
        state <= next_state;
    end
end

// state machine
always @(*) begin
    next_state = state;
    case (state)
        COMPUTE_MEAN: begin
            if (data_sum_acc_reg_vld) begin
                next_state = COMPUTE_VAR;
            end
        end
        COMPUTE_VAR: begin
            if (data_var_reg_vld) begin
                next_state = COMPUTE_NORM;
            end
        end
        COMPUTE_NORM: begin
            if (x_times_gamma_over_sqrt_var_plus_beta_vld) begin
                next_state = COMPUTE_MEAN;
            end
        end
        default: begin
            next_state = COMPUTE_MEAN;
        end
    endcase
end

// mean value register update
// use shift for division
always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        mean_value_reg <= 0;
        mean_value_reg_vld <= 0;
    end else if (data_sum_acc_reg_vld) begin
        mean_value_reg <= data_sum_acc_reg >> (DATA_ARRAY_DEPTH_WIDTH*3);
        mean_value_reg_vld <= 1;
    end else begin
        mean_value_reg_vld <= 0;
    end
end

adder_tree #(
    .ADD_IDATA_BIT(8),
    .ADD_ODATA_BIT(FIXED_ACC_WIDTH),
    .MAC_NUM(BUS_NUM)
) adder_tree_layernorm_var_inst (
    .clk(clk),
    .rstn(rstn),
    .idata(x_sub_mean),
    .idata_valid((state == COMPUTE_VAR) ? in_fixed_data_vld : 0),
    .odata(adder_tree_value_var),
    .odata_valid(adder_tree_value_var_vld)
);
// calculate variance
always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        x_sub_mean <= 0;
        x_sub_mean_vld <= 0;
        data_var_reg_cntr <= 0;
        data_var_reg_vld <= 0;

    end else if (state == COMPUTE_VAR && data_fifo_rden) begin
        for (i = 0; i < BUS_NUM; i = i + 1) begin
            x_sub_mean[i*8+:8] <= (data_fifo_dout[i*8+:8] - mean_value_reg);
        end
        x_sub_mean_vld <= 1;
        data_var_reg_cntr <= data_var_reg_cntr + 1;            
    end else if (state == COMPUTE_VAR && data_fifo_rden && data_var_reg_cntr == IN_FIXED_DATA_ARRAY_DEPTH - 1) begin
        x_sub_mean <= 0;
        x_sub_mean_vld <= 0;
        data_var_reg_cntr <= 0;
        data_var_reg_vld <= 1;
    end
end

always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        data_var_acc_reg <= 0;
    end else if (data_var_reg_cntr < IN_FIXED_DATA_ARRAY_DEPTH && adder_tree_value_var_vld) begin
        data_var_acc_reg <= data_var_acc_reg + adder_tree_value_var;
    end
end

i2flt_rms #(sig_width, exp_width, isize, isign)
i2flt_inv_sqrt (
    .a(data_var_acc_reg + 1),  // set the constant to 1 
    .rnd(3'b00),
    .z(inv_sqrt_in_temp),
    .status()
);

always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        inv_sqrt_in_reg <= 0;
        inv_sqrt_in_reg_vld <= 0;
    end else if (data_var_reg_vld) begin
        inv_sqrt_in_reg <= inv_sqrt_in_temp;
        inv_sqrt_in_reg_vld <= 1;
    end
end

wire [sig_width+exp_width:0] inv_sqrt_out_z;
wire inv_sqrt_out_z_vld;
fp_invsqrt_pipe inst_fp_invsqrt_pipe(
    .clk(clk),
    .rst_n(rstn),
    .x(inv_sqrt_in_reg),
    .x_vld(inv_sqrt_in_reg_vld),
    .y(inv_sqrt_out_z),
    .y_vld(inv_sqrt_out_z_vld)
  );

always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        inv_sqrt_out_reg <= 0;
        inv_sqrt_out_reg_vld <= 0;
    end else begin
        inv_sqrt_out_reg <= inv_sqrt_out_z;
        inv_sqrt_out_reg_vld <= inv_sqrt_out_z_vld;
    end
end

always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        inv_sqrt_out_reg <= 0;
        inv_sqrt_out_reg_vld <= 0;
    end else begin
        inv_sqrt_out_reg <= inv_sqrt_out_z;
        inv_sqrt_out_reg_vld <= inv_sqrt_out_z_vld;
    end
end

wire [BUS_NUM*(sig_width+exp_width)-1:0] flt_x_sub_mean_z;
genvar j;
  generate
    for(j = 0; j < BUS_NUM; j = j + 1) begin : i2flt_array_gen
      i2flt_rms #(sig_width, exp_width, isize, isign)
       i2flt_inv_sqrt_inst (
         .a(data_fifo_dout[j*8+:8]), // set the constant to 1
         .rnd(3'b00),
         .z(flt_x_sub_mean_z[j*(sig_width+exp_width+1)+:sig_width+exp_width+1]),
         .status() // leave unconnected if you don't need to use it
      );
    end
  endgenerate

always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        flt_x_sub_mean <= 0;
        flt_x_sub_mean_vld <= 0;
    end else if (state == COMPUTE_NORM && data_fifo_rden) begin
        flt_x_sub_mean <= flt_x_sub_mean_z;
        flt_x_sub_mean_vld <= 1;
    end else begin
        flt_x_sub_mean_vld <= 0;
    end 
end

// multiply x-mean with gamma
generate 
    for (j = 0; j < BUS_NUM; j = j + 1) begin : x_times_gamma_gen
        fp_mult_pipe #(
            .sig_width(sig_width),
            .exp_width(exp_width),
            .ieee_compliance(ieee_compliance),
            .stages(5)
        ) fp_mult_x_times_gamma (
            .clk(clk),
            .rst_n(rstn),
            .a(flt_x_sub_mean[j*(sig_width+exp_width+1)+:sig_width+exp_width+1]),
            .b(gamma_fifo_dout[j*(sig_width+exp_width+1)+:sig_width+exp_width+1]),
            .ab_valid(state == COMPUTE_NORM && data_fifo_rden),
            .z(x_times_gamma[j*(sig_width+exp_width+1)+:sig_width+exp_width+1]),
            .z_valid(x_times_gamma_vld)
        );
    end
endgenerate

// multiply x-mean with gamma and multiply by inv_sqrt(var)
generate 
    for (j = 0; j < BUS_NUM; j = j + 1) begin : x_times_gamma_over_sqrt_var_gen
        fp_mult_pipe #(
            .sig_width(sig_width),
            .exp_width(exp_width),
            .ieee_compliance(ieee_compliance),
            .stages(5)
        ) fp_mult_x_times_gamma_over_sqrt_var (
            .clk(clk),
            .rst_n(rstn),
            .a(x_times_gamma[j*(sig_width+exp_width+1)+:sig_width+exp_width+1]),
            .b(inv_sqrt_out_reg),
            .ab_valid(inv_sqrt_out_reg_vld),
            .z(x_times_gamma_over_sqrt_var[j*(sig_width+exp_width+1)+:sig_width+exp_width+1]),
            .z_valid(x_times_gamma_over_sqrt_var_vld)
        );
    end
endgenerate

// multiply x-mean with gamma and multiply by inv_sqrt(var) and add beta

wire [BUS_NUM*(sig_width+exp_width+1)-1:0] layer_norm_flt_out_z;
generate 
    for (j = 0; j < BUS_NUM; j = j + 1) begin : x_times_gamma_over_sqrt_var_plus_beta_gen
        custom_fp_sub #(
            .sig_width(sig_width),
            .exp_width(exp_width),
            .ieee_compliance(ieee_compliance)
        ) fp_add_x_times_gamma_over_sqrt_var_plus_beta (
            .a(x_times_gamma_over_sqrt_var[j*(sig_width+exp_width+1)+:sig_width+exp_width+1]),
            .b(beta_fifo_dout[j*(sig_width+exp_width+1)+:sig_width+exp_width+1]),
            .rnd(3'b00),
            .z(layer_norm_flt_out_z[j*(sig_width+exp_width+1)+:sig_width+exp_width+1]),
            .status()
        );
    end
endgenerate

always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        x_times_gamma_over_sqrt_var_plus_beta <= 0;
        x_times_gamma_over_sqrt_var_plus_beta_vld <= 0;
    end else if (state == COMPUTE_NORM && data_fifo_rden) begin
        x_times_gamma_over_sqrt_var_plus_beta <= layer_norm_flt_out_z;
        x_times_gamma_over_sqrt_var_plus_beta_vld <= 1;
    end else begin
        x_times_gamma_over_sqrt_var_plus_beta_vld <= 0;
    end
end

// output the result in fixed data
reg signed [7:0]                nxt_out_fixed_data [0:BUS_NUM-1];
wire signed [8-1:0]         fixed_layernorm [0:BUS_NUM-1];

generate
    for (j = 0; j < BUS_NUM; j = j+1) begin : fixed_layernorm_generate_array
      always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
          float_layernorm_flt2i <= 0;
          float_layernorm_flt2i_vld[j] <= 0;
        end else begin
          float_layernorm_flt2i[j*(sig_width+exp_width+1)+:sig_width+exp_width] <= x_times_gamma_over_sqrt_var_plus_beta[j*(sig_width+exp_width+1)+:sig_width+exp_width] +
                                                                $signed({out_scale_pos_reg, {sig_width{1'b0}}});
          float_layernorm_flt2i[j*(sig_width+exp_width+1)+sig_width+exp_width] <= x_times_gamma_over_sqrt_var_plus_beta[j*(sig_width+exp_width+1)+sig_width+exp_width];
          float_layernorm_flt2i_vld[j] <= x_times_gamma_over_sqrt_var_plus_beta_vld;
        end
      end

      fp2int_rms #(
          .EXP_BIT(exp_width),
          .MAT_BIT(sig_width),
          .IDATA_BIT(exp_width+sig_width+1),
          .ODATA_BIT(8),
          .CDATA_BIT(8)
        )
        fp2int_out_data_inst ( 
          .idata(float_layernorm_flt2i[j*(sig_width+exp_width+1)+:sig_width+exp_width+1]),
          .odata(fixed_layernorm[j])
      );

      always @(*) begin
        nxt_out_fixed_data[j] = fixed_layernorm[j];
        if ($signed(fixed_layernorm[j]) > 127)
          nxt_out_fixed_data[j] = 8'd127;
        else if ($signed(fixed_layernorm[j]) < -128)
          nxt_out_fixed_data[j] = -8'd128;
      end

      always @(posedge clk or negedge rst_n) begin
        if (!rstn) begin
          out_fixed_data_vld[j] <= 0;
          out_fixed_data[j*8 +: 8] <= 0;
        end else begin
          out_fixed_data_vld[j] <= float_layernorm_flt2i_vld[j];
          out_fixed_data[j*8 +: 8] <= nxt_out_fixed_data[j];
        end
      end
    end
  endgenerate



// load the iin_fixed_data_temp fifo from in_fixed_data
fifo #(
    .DEPTH(IN_FIXED_DATA_ARRAY_DEPTH),
    .WIDTH(BUS_NUM*8)
) in_fixed_data_fifo (
    .clk(clk),
    .rstn(rstn),
    .wr_en(data_fifo_wren),
    .rd_en(data_fifo_rden),
    .din((state == COMPUTE_MEAN) ? in_fixed_data : ((state == COMPUTE_VAR) ? x_sub_mean : 0)),
    .dout(data_fifo_dout),
    .full(data_fifo_full),
    .empty(data_fifo_empty)
);

// Gamma array write slot generation. 
fifo #(
    .DEPTH(GAMMA_ARRAY_DEPTH),
    .WIDTH(BUS_NUM*(sig_width+exp_width+1))
) gamma_array_fifo (
    .clk(clk),
    .rstn(rstn),
    .wr_en(gamma_fifo_wren),
    .rd_en(gamma_fifo_rden),
    .din(gamma_fifo_din),
    .dout(gamma_fifo_dout),
    .full(gamma_fifo_full),
    .empty(gamma_fifo_empty)
);

// Beta array write slot generation. 
fifo #(
    .DEPTH(BETA_ARRAY_DEPTH),
    .WIDTH(BUS_NUM*(sig_width+exp_width+1))
) beta_array_fifo (
    .clk(clk),
    .rstn(rstn),
    .wr_en(beta_fifo_wren),
    .rd_en(beta_fifo_rden),
    .din(beta_fifo_din),
    .dout(beta_fifo_dout),
    .full(beta_fifo_full),
    .empty(beta_fifo_empty)
);


// gamma_array_wr_ptr and storage of the gamma array and its valid bits

// data_num register
always @(posedge clk or negedge rstn) begin
if (!rstn)
    data_num <= 0;
else if (in_data_num_vld)
    data_num <= in_data_num;
end

// in_scale_pos_reg
always @(posedge clk or negedge rstn) begin
if (!rstn)
    in_scale_pos_reg <= 0;
else if (in_scale_pos_vld)
    in_scale_pos_reg <= in_scale_pos;
end

// out_scale_pos_reg
always @(posedge clk or negedge rstn) begin
if (!rstn)
    out_scale_pos_reg <= 0;
else if (out_scale_pos_vld)
    out_scale_pos_reg <= out_scale_pos;
end


endmodule









// module adder_tree_layernorm #(
//     parameter ADD_IDATA_BIT = 16,
//     parameter ADD_ODATA_BIT = 16 + $clog2(8),
//     parameter MAC_NUM = 8
// )(
//     // Global Signals
//     input                               clk,
//     input                               rstn,

//     // Data Signals
//     input       [ADD_IDATA_BIT*MAC_NUM-1:0] idata,
//     input                               idata_valid,
//     output  reg [ADD_ODATA_BIT-1:0]         odata,
//     output  reg                         odata_valid
// );

//     localparam  STAGE_NUM = $clog2(MAC_NUM);

//     // Insert a pipeline every two stages
//     // Validation
//     genvar i, j;
//     generate
//         for (i = 0; i < STAGE_NUM; i = i + 1) begin: gen_adt_valid
//             reg             add_valid;

//             if (i == 0) begin   // Input Stage
//                 always @(posedge clk or negedge rstn) begin
//                     if (!rstn) begin
//                         add_valid <= 1'b0;
//                     end
//                     else begin
//                         add_valid <= idata_valid;
//                     end
//                 end
//             end
//             else if (i % 2 == 1'b0) begin   // Even Stage, Insert a pipeline, Start from 0, 2, 4...
//                 always @(posedge clk or negedge rstn) begin
//                     if (!rstn) begin
//                         add_valid <= 1'b0;
//                     end
//                     else begin
//                         add_valid <= gen_adt_valid[i-1].add_valid;
//                     end
//                 end
//             end
//             else begin  // Odd Stage, Combinational, Start from 1, 3, 5...
//                 always @(*) begin
//                     add_valid = gen_adt_valid[i-1].add_valid;
//                 end
//             end
//         end
//     endgenerate

//     // Adder
//     generate
//         for (i = 0; i <STAGE_NUM; i = i + 1) begin: gen_adt_stage
//             localparam  OUT_BIT = ADD_IDATA_BIT + (i + 1'b1);
//             localparam  OUT_NUM = MAC_NUM  >> (i + 1'b1);

//             reg     [OUT_BIT-2:0]   add_idata   [0:OUT_NUM*2-1];
//             wire    [OUT_BIT-1:0]   add_odata   [0:OUT_NUM-1];

//             for (j = 0; j < OUT_NUM; j = j + 1) begin: gen_adt_adder

//                 // Organize adder inputs
//                 if (i == 0) begin   // Input Stage
//                     always @(posedge clk or negedge rstn) begin
//                         if (!rstn) begin
//                             add_idata[j*2]   <= 'd0;
//                             add_idata[j*2+1] <= 'd0;
//                         end
//                         else if (idata_valid) begin
//                             add_idata[j*2]   <= idata[(j*2+0)*ADD_IDATA_BIT+:ADD_IDATA_BIT];
//                             add_idata[j*2+1] <= idata[(j*2+1)*ADD_IDATA_BIT+:ADD_IDATA_BIT];
//                         end
//                     end
//                 end
//                 else if (i % 2 == 0) begin  // Even Stage, Insert a pipeline
//                     always @(posedge clk or negedge rstn) begin
//                         if (!rstn) begin
//                             add_idata[j*2]   <= 'd0;
//                             add_idata[j*2+1] <= 'd0;
//                         end
//                         else if (gen_adt_valid[i-1].add_valid) begin
//                             add_idata[j*2]   <= gen_adt_stage[i-1].add_odata[j*2];
//                             add_idata[j*2+1] <= gen_adt_stage[i-1].add_odata[j*2+1];
//                         end
//                     end
//                 end
//                 else begin  // Odd Stage, Combinational
//                     always @(*) begin
//                         add_idata[j*2]   = gen_adt_stage[i-1].add_odata[j*2];
//                         add_idata[j*2+1] = gen_adt_stage[i-1].add_odata[j*2+1];
//                     end
//                 end

//                 // Adder instantization
//                 add_int #(.ADD_INT_IDATA_BIT(OUT_BIT-1), .ADD_INT_ODATA_BIT(OUT_BIT)) adder_inst (
//                     .idataA                 (add_idata[j*2]),
//                     .idataB                 (add_idata[j*2+1]),
//                     .odata                  (add_odata[j])
//                 );
//             end
//         end
//     endgenerate

//     // Output
//     always @(*) begin
//         odata       = gen_adt_stage[STAGE_NUM-1].add_odata[0];
//         odata_valid = gen_adt_valid[STAGE_NUM-1].add_valid;
//     end
// endmodule 



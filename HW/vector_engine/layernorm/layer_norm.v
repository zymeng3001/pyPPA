module layer_norm #(
    parameter integer BUS_NUM = 8,
    parameter integer DATA_NUM_WIDTH = 10,
    parameter integer SCALA_POS_WIDTH = 5,
    parameter integer FIXED_ACC_WIDTH = 32,
    parameter integer MEAN_ACC_WIDTH = 24,
    parameter integer MEAN_WIDTH = 8,
    parameter integer SQR_WIDTH = 16,
    parameter integer IN_FIXED_DATA_ARRAY_DEPTH = (${n_embd} / BUS_NUM + 1),
    parameter integer DATA_ARRAY_DEPTH_WIDTH = $clog2(IN_FIXED_DATA_ARRAY_DEPTH), 
    parameter integer GAMMA_ARRAY_DEPTH = IN_FIXED_DATA_ARRAY_DEPTH,
    parameter integer BETA_ARRAY_DEPTH = IN_FIXED_DATA_ARRAY_DEPTH,
    parameter integer BF_WIDTH = 16,
    parameter integer sig_width = 7,
    parameter integer exp_width = 8,
    parameter integer isize     = 16,
    parameter integer isign     = 1,
    parameter integer ieee_compliance = 0
)(
    input clk,
    input rstn,

    input [DATA_NUM_WIDTH-1:0] in_data_num,
    input                      in_data_num_vld,

    input signed [BUS_NUM*8-1:0] in_fixed_data,
    input in_fixed_data_vld,

    input [BUS_NUM*(sig_width+exp_width+1)-1:0] in_gamma,
    input [BUS_NUM*(sig_width+exp_width+1)-1:0] in_beta,
    input in_gamma_vld,
    input in_beta_vld,

    input signed [SCALA_POS_WIDTH-1:0] in_scale_pos,
    input                              in_scale_pos_vld,

    input signed [SCALA_POS_WIDTH-1:0] out_scale_pos,
    input                              out_scale_pos_vld,

    output reg signed [BUS_NUM*8-1:0] out_fixed_data,
    output reg out_fixed_data_vld,
    output reg out_fixed_data_last
);
integer i;

parameter integer SETUP = 2'b00;
parameter integer DATA_INPUT = 2'b01;
parameter integer PROCESSING = 2'b10;
parameter integer OUTPUT = 2'b11;

reg [1:0] state, next_state;

// reg for buffering the input data
reg signed [BUS_NUM*8-1:0] in_fixed_data_reg;
reg in_fixed_data_reg_vld;

// reg for holding the input data number
reg [DATA_NUM_WIDTH-1:0] data_bus_num;
reg [DATA_NUM_WIDTH-1:0] data_num_reg;

// reg for holding the input and output scale position
reg signed [SCALA_POS_WIDTH-1:0] in_scale_pos_reg;
reg signed [SCALA_POS_WIDTH-1:0] out_scale_pos_reg;

// signal for controlling the 3 fifos
reg data_fifo_wren, data_fifo_rden, gamma_fifo_wren, gamma_fifo_rden, beta_fifo_wren, beta_fifo_rden;
reg [BUS_NUM*8-1:0] data_fifo_din;
reg [BUS_NUM*(sig_width+exp_width+1)-1:0] gamma_fifo_din;
reg [BUS_NUM*(sig_width+exp_width+1)-1:0] beta_fifo_din;

// register for getting the sum
reg [MEAN_ACC_WIDTH-1: 0] data_sum_acc_reg;
reg                       data_sum_acc_reg_vld;

// register for getting the sum of sqr
reg [FIXED_ACC_WIDTH-1: 0] data_sqr_sum_acc_reg;
reg                        data_sqr_sum_acc_reg_vld;     

// register for holding the mean value
reg [MEAN_WIDTH-1:0] mean_value_reg;
reg mean_value_reg_vld;

// register for holding the mean square value 
reg [SQR_WIDTH-1:0] mean_sqr_reg;
reg mean_sqr_reg_vld;

// register for holding the sqr mean value
reg [SQR_WIDTH-1:0] sqr_mean_reg;
reg sqr_mean_reg_vld;

// reg for hold sqr input data
reg signed [BUS_NUM*SQR_WIDTH-1:0] in_data_sqr_reg;
reg in_data_sqr_reg_vld;

// cnt for data input 
reg [DATA_ARRAY_DEPTH_WIDTH-1:0] data_input_reg_cntr;

// reg for variance
reg [SQR_WIDTH-1:0] var_reg;
reg                 var_reg_vld;

// reg for inv sqrt of variance
wire [SQR_WIDTH-1:0] inv_sqrt_var_reg;
wire                 inv_sqrt_var_reg_vld;

// wire for var i2flt
wire [BF_WIDTH-1:0] var_flt_temp;

// reg for var i2flt
reg [BF_WIDTH-1:0] var_flt_reg;
reg var_flt_reg_vld;

// reg for calculating mean value
wire signed [MEAN_ACC_WIDTH-1:0] adder_tree_value;
wire signed adder_tree_value_vld;

wire signed [FIXED_ACC_WIDTH-1:0] adder_tree_value_sqr;
wire signed adder_tree_value_sqr_vld;

reg adder_tree_sum_input_vld;
reg adder_tree_sqr_sum_input_vld;

// reg for reading out gamma and beta value
wire [BUS_NUM*BF_WIDTH-1:0] gamma_fifo_dout, beta_fifo_dout;
reg gamma_fifo_out_vld, beta_fifo_out_vld;
wire gamma_fifo_empty, beta_fifo_empty;

// reg for reading out data
wire signed [BUS_NUM*8-1:0] data_fifo_dout;
reg signed [BUS_NUM*8-1:0] data_fifo_dout_reg;

reg data_fifo_out_vld, data_fifo_out_pre_vld;
wire data_fifo_empty;

// reg for holding intermidiate mult results
wire [BF_WIDTH-1:0] data_mult_gamma [0:BUS_NUM-1];
wire data_mult_gamma_vld;

// reg for holding data before beta
wire [BF_WIDTH-1:0] data_before_beta [0:BUS_NUM-1];
wire data_before_beta_vld;

// reg [BF_WIDTH-1:0] data_mult_gamma [0:BUS_NUM-1];
// reg data_mult_gamma_vld;

// reg for holding data minus mean value
wire [BF_WIDTH-1:0] data_minus_mean_flt [0:BUS_NUM-1];
reg [BF_WIDTH-1:0] data_minus_mean_flt_reg [0:BUS_NUM-1];
reg data_minus_mean_flt_vld;

// reg for holding output in float
wire [BF_WIDTH-1:0] out_data_flt [0:BUS_NUM-1];
reg  [BF_WIDTH-1:0] out_data_flt_reg [0:BUS_NUM-1];
reg  out_data_flt_reg_vld;

reg beta_rden_pre1, beta_rden_pre2;

// wire for convert output to int8
reg [BF_WIDTH-1:0] layernorm_flt2fp [0:BUS_NUM-1];
wire [BUS_NUM*8-1:0] layernorm_fp;

always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        data_fifo_dout_reg <= 0;
        data_fifo_out_vld <= 0;
        data_fifo_out_pre_vld <= 0;
        // data_mult_gamma_vld <= 0;
        out_data_flt_reg_vld <= 0;
        
        for (i = 0; i < BUS_NUM; i = i + 1) begin
            data_minus_mean_flt_reg[i] <= 0;
            // data_mult_gamma[i] <= 0;
            out_data_flt_reg[i] <= 0;
        end
        data_minus_mean_flt_vld <= 0;

        beta_rden_pre1 <= 0;
        beta_rden_pre2 <= 0;

    end else begin
        for (i = 0; i < BUS_NUM; i = i + 1) begin
            data_fifo_dout_reg[i*8 +: 8] <= data_fifo_dout[i*8 +: 8] - mean_value_reg;
            data_minus_mean_flt_reg[i] <= data_minus_mean_flt[i];
            // data_mult_gamma[i] <= data_mult_gamma_wire[i];
            out_data_flt_reg[i] <= out_data_flt[i];
        end
        data_fifo_out_pre_vld <= !data_fifo_empty && data_fifo_rden;
        data_fifo_out_vld <= data_fifo_out_pre_vld;

        data_minus_mean_flt_vld <= data_fifo_out_vld;
        // data_mult_gamma_vld <= data_mult_gamma_vld_wire;
        out_data_flt_reg_vld <= data_before_beta_vld;

        beta_rden_pre1 <= data_mult_gamma_vld;
        beta_rden_pre2 <= beta_rden_pre1;
    end
end

always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        state <= SETUP;
        in_fixed_data_reg <= 0;
        in_fixed_data_reg_vld <= 0;
    end else begin
        state <= next_state;
        in_fixed_data_reg <= in_fixed_data;
        in_fixed_data_reg_vld <= in_fixed_data_vld;
    end
end

always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        in_data_sqr_reg <= 0;
        in_data_sqr_reg_vld <= 0;
    end else begin
        for (i = 0; i < BUS_NUM; i = i + 1) begin
            if (in_fixed_data_reg_vld) begin
                in_data_sqr_reg[i*SQR_WIDTH +: SQR_WIDTH] <= in_fixed_data_reg[i*8 +: 8] * in_fixed_data_reg[i*8 +: 8];
            end else begin
                in_data_sqr_reg <= in_data_sqr_reg;
            end
        end
        in_data_sqr_reg_vld <= in_fixed_data_reg_vld;
    end
end

always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        data_sum_acc_reg <= 0;
        data_sum_acc_reg_vld <= 0;

        data_sqr_sum_acc_reg <= 0;
        data_sqr_sum_acc_reg_vld <= 0;
    end else begin
        data_sum_acc_reg <= (data_input_reg_cntr < data_bus_num + 3 && adder_tree_value_vld) ? data_sum_acc_reg + adder_tree_value :
                            (data_input_reg_cntr >= data_bus_num + 4) ? 0 : data_sum_acc_reg;
        data_sum_acc_reg_vld <= (data_input_reg_cntr == data_bus_num + 3);

        data_sqr_sum_acc_reg <= (data_input_reg_cntr < data_bus_num + 4 && adder_tree_value_sqr_vld) ? data_sqr_sum_acc_reg + adder_tree_value_sqr :
                            (data_input_reg_cntr >= data_bus_num + 5) ? 0 : data_sqr_sum_acc_reg;
        data_sqr_sum_acc_reg_vld <= (data_input_reg_cntr == data_bus_num + 4);
    end
end

always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        data_input_reg_cntr <= 0;
    end else begin
        data_input_reg_cntr <= (data_input_reg_cntr < data_bus_num && in_fixed_data_reg_vld) ? data_input_reg_cntr + 1 :
                               (data_input_reg_cntr < data_bus_num && !in_fixed_data_reg_vld) ? data_input_reg_cntr :
                               (data_input_reg_cntr >= data_bus_num && data_input_reg_cntr < data_bus_num+5) ? data_input_reg_cntr + 1 : 0;   
    end
end

always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        mean_value_reg <= 0;
        mean_value_reg_vld <= 0;
    end else begin
        mean_value_reg <= (data_sum_acc_reg_vld) ? data_sum_acc_reg / data_num_reg : mean_value_reg;
        mean_value_reg_vld <= (state == SETUP) ? 0 :
                              (data_sum_acc_reg_vld) ? 1 : mean_value_reg_vld;
    end
end

always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        mean_sqr_reg <= 0;
        mean_sqr_reg_vld <= 0;
    end else begin
        mean_sqr_reg <= mean_value_reg_vld ? mean_value_reg * mean_value_reg : mean_sqr_reg;
        mean_sqr_reg_vld <= (state == SETUP) ? 0 :
                            (mean_value_reg_vld) ? 1 : mean_sqr_reg_vld;
    end
end

always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        sqr_mean_reg <= 0;
        sqr_mean_reg_vld <= 0;
    end else begin
        sqr_mean_reg <= (data_sqr_sum_acc_reg_vld) ? data_sqr_sum_acc_reg / data_num_reg : sqr_mean_reg;
        sqr_mean_reg_vld <= (state == SETUP) ? 0 :
                            (data_sqr_sum_acc_reg_vld) ? 1 : sqr_mean_reg_vld;
    end
end

always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        var_reg <= 0;
        var_reg_vld <= 0;
    end else begin
        var_reg <= (sqr_mean_reg_vld && mean_sqr_reg_vld) ? (sqr_mean_reg >> in_scale_pos_reg) - mean_sqr_reg : var_reg;
        var_reg_vld <= (state == SETUP) ? 0 :
                       (sqr_mean_reg_vld && mean_sqr_reg_vld) ? 1 : var_reg_vld;
    end
end

always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        var_flt_reg <= 0;
        var_flt_reg_vld <= 0;
    end else begin
        var_flt_reg <= (var_reg_vld) ? var_flt_temp : var_flt_reg;
        var_flt_reg_vld <= (state == SETUP) ? 0 :
                           (var_reg_vld) ? 1 : var_flt_reg_vld;
    end
end

always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        out_fixed_data <= 0;
        out_fixed_data_vld <= 0;
    end else begin
        out_fixed_data <= layernorm_fp;
        out_fixed_data_vld <= out_data_flt_reg_vld;
    end
end

always @(*) begin
    case (state)
        SETUP: begin
            if (in_data_num_vld) begin
                next_state = DATA_INPUT;
            end else begin
                next_state = SETUP;
            end
        end

        DATA_INPUT: begin
            if (data_input_reg_cntr >= data_bus_num) begin
                next_state = PROCESSING;
            end else begin
                next_state = DATA_INPUT;
            end
        end

        PROCESSING: begin
            if (inv_sqrt_var_reg_vld) begin    // enter parallel output state after inv sqrt variance is ready
                next_state = OUTPUT;
            end else begin
                next_state = PROCESSING;
            end
        end

        OUTPUT: begin
            if (out_scale_pos_vld) begin
                next_state = SETUP;
            end else begin
                next_state = OUTPUT;
            end
        end

        default: next_state = SETUP;
    endcase
end

// data_bus_num register
always @(posedge clk or negedge rstn) begin
if (!rstn) begin
    data_bus_num <= 0;
    data_num_reg <= 0;
end else if (in_data_num_vld) begin
    data_bus_num <= in_data_num / BUS_NUM;
    data_num_reg <= in_data_num;
end
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

always @(*) begin
    adder_tree_sum_input_vld = (state == DATA_INPUT) ? {in_fixed_data_reg_vld} : {1'b0};
end

always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        adder_tree_sqr_sum_input_vld <= 0;
    end else begin
        adder_tree_sqr_sum_input_vld <= adder_tree_sum_input_vld;
    end
end

always @(*) begin
  data_fifo_rden = (state == OUTPUT);
  gamma_fifo_rden = data_fifo_out_vld;
  beta_fifo_rden = beta_rden_pre2;
  data_fifo_wren = (state == DATA_INPUT) && in_fixed_data_reg_vld;
  gamma_fifo_wren = (state == DATA_INPUT) && in_gamma_vld;
  beta_fifo_wren = (state == DATA_INPUT) && in_beta_vld;
  data_fifo_din = in_fixed_data_reg;
  gamma_fifo_din = in_gamma;
  beta_fifo_din = in_beta;

  for (i = 0; i < BUS_NUM; i = i + 1) begin
      layernorm_flt2fp[i][sig_width+:exp_width] = out_data_flt_reg[i][sig_width+:exp_width] + out_scale_pos_reg;
      layernorm_flt2fp[i][sig_width-1:0] = out_data_flt_reg[i][sig_width-1:0];
      layernorm_flt2fp[i][sig_width+exp_width] = out_data_flt_reg[i][sig_width+exp_width];
  end
end

always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        adder_tree_sqr_sum_input_vld <= 0;
    end else begin
        adder_tree_sqr_sum_input_vld <= adder_tree_sum_input_vld;
    end
end

i2flt_rms #(.SIG_WIDTH(sig_width), .EXP_WIDTH(exp_width), .ISIZE(16), .ISIGN(isign))
i2flt_square_sum ( 
    .a(var_reg), 
    .rnd(3'b000), 
    .z(var_flt_temp), 
    .status() 
);

genvar j;
generate
    for (j = 0; j < BUS_NUM; j = j + 1) begin : gen_i2flt_var
        i2flt_rms #(
            .SIG_WIDTH(sig_width),
            .EXP_WIDTH(exp_width),
            .ISIZE(8),
            .ISIGN(isign)
        ) i2flt_var (
            .a(data_fifo_dout_reg[j*8 +: 8]),
            .rnd(3'b000),
            .z(data_minus_mean_flt[j]),
            .status()
        );

        fp2int_rms #(
            .EXP_BIT(exp_width),
            .MAT_BIT(sig_width),
            .IDATA_BIT(exp_width+sig_width+1),
            .ODATA_BIT(8),
            .CDATA_BIT(8)
        )
        fp2int_out_data_inst ( 
            .idata(layernorm_flt2fp[j]),
            .odata(layernorm_fp[j*8 +: 8])
        );

        fp_mult_pipe # (
            .sig_width(sig_width),
            .exp_width(exp_width),
            .ieee_compliance(ieee_compliance),
            .stages(3)  // Number of pipeline stages
        )
        fp_mult_pipe_inst (
            .clk(clk),
            .rst_n(rstn),
            .a(gamma_fifo_dout[j*(sig_width+exp_width+1) +: (sig_width+exp_width+1)]),
            .b(data_minus_mean_flt_reg[j]),
            .ab_valid(data_minus_mean_flt_vld),
            .z(data_mult_gamma[j]),
            .z_valid(data_mult_gamma_vld)
        );

        fp_mult_pipe # (
            .sig_width(sig_width),
            .exp_width(exp_width),
            .ieee_compliance(ieee_compliance),
            .stages(3)  // Number of pipeline stages
        )
        fp_mult_pipe2_inst (
            .clk(clk),
            .rst_n(rstn),
            .a(data_mult_gamma[j]),
            .b(inv_sqrt_var_reg),
            .ab_valid(data_mult_gamma_vld),
            .z(data_before_beta[j]),
            .z_valid(data_before_beta_vld)
        );

        custom_fp_add # (
            .sig_width(sig_width),
            .exp_width(exp_width),
            .ieee_compliance(ieee_compliance)
        )
        custom_fp_add_inst (
            .a(data_before_beta[j]),
            .b(beta_fifo_dout[j*(sig_width+exp_width+1) +: (sig_width+exp_width+1)]),
            .rnd(3'b000),
            .z(out_data_flt[j]),
            .status(_)
        );

    end
endgenerate

fp_invsqrt_pipe fp_invsqrt_pipe_inst (
    .clk(clk),
    .rst_n(rstn),
    .x(var_flt_reg),
    .x_vld(var_flt_reg_vld),
    .y(inv_sqrt_var_reg),
    .y_vld(inv_sqrt_var_reg_vld)
);

adder_tree #(
    .ADD_IDATA_BIT(8),
    .ADD_ODATA_BIT(MEAN_ACC_WIDTH),
    .MAC_NUM(BUS_NUM)
) adder_tree_sum_inst (
    .clk(clk),
    .rstn(rstn),
    .idata(in_fixed_data_reg),
    .idata_valid(adder_tree_sum_input_vld),
    .odata(adder_tree_value),
    .odata_valid(adder_tree_value_vld)
);

adder_tree #(
    .ADD_IDATA_BIT(SQR_WIDTH),
    .ADD_ODATA_BIT(FIXED_ACC_WIDTH),
    .MAC_NUM(BUS_NUM)
) adder_tree_sqr_sum_inst (
    .clk(clk),
    .rstn(rstn),
    .idata(in_data_sqr_reg),
    .idata_valid(adder_tree_sqr_sum_input_vld),
    .odata(adder_tree_value_sqr),
    .odata_valid(adder_tree_value_sqr_vld)
);

// load the iin_fixed_data_temp fifo from in_fixed_data
fifo #(
    .DEPTH(IN_FIXED_DATA_ARRAY_DEPTH),
    .WIDTH(BUS_NUM*8)
) in_fixed_data_fifo (
    .clk(clk),
    .rstn(rstn),
    .wr_en(data_fifo_wren),
    .rd_en(data_fifo_rden),
    // .din((state == COMPUTE_MEAN) ? in_fixed_data : ((state == COMPUTE_VAR) ? x_sub_mean : 0)),
    .din(data_fifo_din),
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

endmodule
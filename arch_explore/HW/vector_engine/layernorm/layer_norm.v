module layer_norm #(
    parameter integer BUS_NUM = 8,
    parameter integer DATA_NUM_WIDTH = 10,
    parameter integer SCALA_POS_WIDTH = 5,
    parameter integer FIXED_ACC_WIDTH = 32,
    parameter integer ADDER_TREE_OUT_WIDTH = 16,
    parameter integer IN_FLOAT_DATA_ARRAY_DEPTH = (${n_embd} / BUS_NUM + 1), 
    parameter integer GAMMA_ARRAY_DEPTH = IN_FLOAT_DATA_ARRAY_DEPTH,
    parameter integer BETA_ARRAY_DEPTH = IN_FLOAT_DATA_ARRAY_DEPTH,
    parameter integer sig_width = 7,
    parameter integer exp_width = 8,
    parameter integer isize     = 32,
    parameter integer isign     = 1,
    parameter integer ieee_compliance = 0
)(
    input clk,
    input rst_n,

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

parameter inetger COMPUTE_MEAN = 2'b00;
parameter integer COMPUTE_VAR  = 2'b01;
parameter integer COMPUTE_norm = 2'b10;

reg [DATA_NUM_WIDTH-1:0] data_num;

reg signed [SCALA_POS_WIDTH-1:0] in_scale_pos_reg;
reg signed [SCALA_POS_WIDTH-1:0] out_scale_pos_reg;

reg [ (sig_width+exp_width+1)*BUS_NUM -1:0 ] gamma_array [0:GAMMA_ARRAY_DEPTH-1];
reg [ (sig_width+exp_width+1)*BUS_NUM -1:0 ] gamma_array_wr_slot;
reg [ (sig_width+exp_width+1)*BUS_NUM -1:0 ] gamma_array_rd_slot;
reg [BUS_NUM-1:0]                           gamma_vld_array [0:GAMMA_ARRAY_DEPTH-1];
reg [$clog2(GAMMA_ARRAY_DEPTH)-1:0]           gamma_array_wr_ptr, gamma_array_rd_ptr;

// reg for beta array
reg [ (sig_width+exp_width+1)*BUS_NUM -1:0 ] beta_array [0:BETA_ARRAY_DEPTH-1];
reg [ (sig_width+exp_width+1)*BUS_NUM -1:0 ] beta_array_wr_slot;
reg [ (sig_width+exp_width+1)*BUS_NUM -1:0 ] beta_array_rd_slot;
reg [BUS_NUM-1:0]                           beta_vld_array [0:BETA_ARRAY_DEPTH-1];
reg [$clog2(BETA_ARRAY_DEPTH)-1:0]           beta_array_wr_ptr, beta_array_rd_ptr;

// reg for calculating mean value
reg signed [ADDER_TREE_OUT_WIDTH-1:0] adder_tree_value;
reg signed adder_tree_value_vld;
reg signed [FIXED_ACC_WIDTH-1:0] mean_value;
reg signed mean_value_vld;

reg signed [8*IN_FLOAT_DATA_ARRAY_DEPTH*BUS_NUM-1:0] in_fixed_data_temp;
reg [IN_FLOAT_DATA_ARRAY_DEPTH-1:0] in_fixed_data_vld_temp;

// load the iin_fixed_data_temp fifo from in_fixed_data
always @(posedge clk or negedge rst_n) begin
if (!rst_n) begin
    in_fixed_data_temp <= 0;
    in_fixed_data_vld_temp <= 0;
end else if (state == COMPUTE_MEAN) begin
    in_fixed_data_temp <= {in_fixed_data_temp[BUS_NUM*8-1:0], in_fixed_data};
    in_fixed_data_vld_temp <= {in_fixed_data_vld_temp[BUS_NUM-1:0], in_fixed_data_vld};
end
end


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

// beta array write slot generation. (The flattened vector in_beta is sliced.)
generate
  for (i = 0; i < BUS_NUM; i = i+1) begin : beta_array_wr_slot_generate_array
    // Using the "-:" slicing operator to extract each BUS_NUM element slice.
    assign beta_array_wr_slot[(i+1)*(sig_width+exp_width+1)-1 -: (sig_width+exp_width+1)] =
           in_beta[(i+1)*(sig_width+exp_width+1)-1 -: (sig_width+exp_width+1)];
  end
endgenerate

// beta_array_wr_ptr and storage of the beta array and its valid bits
always @(posedge clk or negedge rst_n) begin
  if (!rst_n)
    beta_array_wr_ptr <= 0;
  else if (out_fixed_data_last)
    beta_array_wr_ptr <= 0;  // reset for next vector input
  else if (|in_beta_vld) begin // At least one valid input
    beta_array[beta_array_wr_ptr] <= beta_array_wr_slot;
    beta_vld_array[beta_array_wr_ptr] <= in_beta_vld;
    beta_array_wr_ptr <= beta_array_wr_ptr + 1;
  end
end

// instantiate adder_tree_layernorm
adder_tree_layernorm #(
    .ADD_IDATA_BIT(8),
    .ADD_ODATA_BIT(ADDER_TREE_OUT_WIDTH),
    .DATA_NUM(IN_FLOAT_DATA_ARRAY_DEPTH*BUS_NUM)
) adder_tree_layernorm_inst (
    .clk(clk),
    .rstn(rst_n),
    .idata(in_fixed_data_temp),
    .idata_valid(in_fixed_data_vld_temp),
    .odata(adder_tree_value),
    .odata_valid(adder_tree_value_vld)
);




module adder_tree_layernorm #(
    parameter ADD_IDATA_BIT = 16,
    parameter ADD_ODATA_BIT = 16 + $clog2(8),
    parameter DATA_NUM = ${n_embd}
)(
    // Global Signals
    input                               clk,
    input                               rstn,

    // Data Signals
    input       [ADD_IDATA_BIT*MAC_NUM-1:0] idata,
    input                               idata_valid,
    output  reg [ADD_ODATA_BIT-1:0]         odata,
    output  reg                         odata_valid
);

    localparam  STAGE_NUM = $clog2(MAC_NUM);

    // Insert a pipeline every two stages
    // Validation
    genvar i, j;
    generate
        for (i = 0; i < STAGE_NUM; i = i + 1) begin: gen_adt_valid
            reg             add_valid;

            if (i == 0) begin   // Input Stage
                always @(posedge clk or negedge rstn) begin
                    if (!rstn) begin
                        add_valid <= 1'b0;
                    end
                    else begin
                        add_valid <= idata_valid;
                    end
                end
            end
            else if (i % 2 == 1'b0) begin   // Even Stage, Insert a pipeline, Start from 0, 2, 4...
                always @(posedge clk or negedge rstn) begin
                    if (!rstn) begin
                        add_valid <= 1'b0;
                    end
                    else begin
                        add_valid <= gen_adt_valid[i-1].add_valid;
                    end
                end
            end
            else begin  // Odd Stage, Combinational, Start from 1, 3, 5...
                always @(*) begin
                    add_valid = gen_adt_valid[i-1].add_valid;
                end
            end
        end
    endgenerate

    // Adder
    generate
        for (i = 0; i <STAGE_NUM; i = i + 1) begin: gen_adt_stage
            localparam  OUT_BIT = ADD_IDATA_BIT + (i + 1'b1);
            localparam  OUT_NUM = MAC_NUM  >> (i + 1'b1);

            reg     [OUT_BIT-2:0]   add_idata   [0:OUT_NUM*2-1];
            wire    [OUT_BIT-1:0]   add_odata   [0:OUT_NUM-1];

            for (j = 0; j < OUT_NUM; j = j + 1) begin: gen_adt_adder

                // Organize adder inputs
                if (i == 0) begin   // Input Stage
                    always @(posedge clk or negedge rstn) begin
                        if (!rstn) begin
                            add_idata[j*2]   <= 'd0;
                            add_idata[j*2+1] <= 'd0;
                        end
                        else if (idata_valid) begin
                            add_idata[j*2]   <= idata[(j*2+0)*ADD_IDATA_BIT+:ADD_IDATA_BIT];
                            add_idata[j*2+1] <= idata[(j*2+1)*ADD_IDATA_BIT+:ADD_IDATA_BIT];
                        end
                    end
                end
                else if (i % 2 == 0) begin  // Even Stage, Insert a pipeline
                    always @(posedge clk or negedge rstn) begin
                        if (!rstn) begin
                            add_idata[j*2]   <= 'd0;
                            add_idata[j*2+1] <= 'd0;
                        end
                        else if (gen_adt_valid[i-1].add_valid) begin
                            add_idata[j*2]   <= gen_adt_stage[i-1].add_odata[j*2];
                            add_idata[j*2+1] <= gen_adt_stage[i-1].add_odata[j*2+1];
                        end
                    end
                end
                else begin  // Odd Stage, Combinational
                    always @(*) begin
                        add_idata[j*2]   = gen_adt_stage[i-1].add_odata[j*2];
                        add_idata[j*2+1] = gen_adt_stage[i-1].add_odata[j*2+1];
                    end
                end

                // Adder instantization
                add_int #(.ADD_INT_IDATA_BIT(OUT_BIT-1), .ADD_INT_ODATA_BIT(OUT_BIT)) adder_inst (
                    .idataA                 (add_idata[j*2]),
                    .idataB                 (add_idata[j*2+1]),
                    .odata                  (add_odata[j])
                );
            end
        end
    endgenerate

    // Output
    always @(*) begin
        odata       = gen_adt_stage[STAGE_NUM-1].add_odata[0];
        odata_valid = gen_adt_valid[STAGE_NUM-1].add_valid;
    end
endmodule 





endmodule
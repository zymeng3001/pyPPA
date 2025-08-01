module softmax
#(    
    parameter integer BUS_NUM = 8, //input bus num, even number
    parameter integer FIFO_DEPTH = 512, //input fifo depth
    parameter integer VECTOR_LENGTH = 384,
    parameter integer SCALA_POS_WIDTH = 5,
    parameter integer FIXED_DATA_WIDTH = 8,

    parameter integer sig_width = 23,//IEEE 754
    parameter integer exp_width = 8,
    localparam        isize = 32,
    localparam        isign = 1, //signed integer number
    localparam        ieee_compliance = 0 //No support to NaN adn denormals
    localparam        exp_arch = 1
)
(
    // Global Signals
    input                       clk,
    input                       rst_n,

    // Control Signals
    input  logic signed              [SCALA_POS_WIDTH-1:0]           in_scale_pos,
    input  logic                                                     in_scale_pos_vld, 
    input  logic signed              [SCALA_POS_WIDTH-1:0]           out_scale_pos,  
    input  logic                                                     out_scale_pos_vld,

    // Data Signals
    input  logic signed [BUS_NUM-1:0][FIXED_DATA_WIDTH-1:0]          in_fixed_data, //Here signed seems not work!!! Guess: signed now points to BUS_NUM
    input  logic        [BUS_NUM-1:0]                                in_fixed_data_vld,
    output logic signed [BUS_NUM-1:0][FIXED_DATA_WIDTH-1:0]          out_fixed_data,
    output logic        [BUS_NUM-1:0]                                out_fixed_data_vld,
);  
    logic signed              [SCALA_POS_WIDTH-1:0]                      in_scale_pos_reg;
    logic signed              [SCALA_POS_WIDTH-1:0]                      out_scale_pos_reg;

    // Signals for bus_fifo ports
    logic                   wr_en;
    logic                   rd_en;
    logic [IN_DEPTH-1:0][WIDTH-1:0] wr_data;
    logic                   wr_valid;
    logic                   rd_valid;
    logic [IN_DEPTH-1:0][WIDTH-1:0] rd_data;
    logic                   almost_full;
    logic                   full;
    logic                   empty;

/////////////////////////////////////////////////////
//First set the data num (length of RMSnorm vector)//
//and scale position and gamma array               //
/////////////////////////////////////////////////////
    always_ff @(posedge clk or negedge rst_n) begin
        if(~rst_n)
            in_scale_pos_reg <=0;
        else if(in_scale_pos_vld)
            in_scale_pos_reg <= in_scale_pos;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if(~rst_n)
            out_scale_pos_reg <=0;
        else if(out_scale_pos_vld)
            out_scale_pos_reg <= out_scale_pos;
    end

/////////////////////////////////////////////////////
//  FIFO FOR INPUT DATA                            //
//                                                 //
/////////////////////////////////////////////////////
    assign wr_en = |in_fixed_data_vld;
    assign wr_data = in_fixed_data;
    bus_fifo #(
        .SIZE (FIFO_DEPTH),
        .WIDTH (FIXED_DATA_WIDTH),
        .ALERT_DEPTH (3),
        .IN_DEPTH (BUS_NUM)
    ) input_fifo (
        .clk(clk),
        .rstn(rst_n),
        .wr_en(wr_en),
        .rd_en(rd_en),
        .wr_data(wr_data),
        .wr_valid(wr_valid),
        .rd_valid(rd_valid),
        .rd_data(rd_data),
        .almost_full(almost_full),
        .full(full),
        .empty(empty)
    );

/////////////////////////////////////////////////////
//  Get the new max from current input data        //
//                                                 //
/////////////////////////////////////////////////////
    logic signed [FIXED_DATA_WIDTH-1:0]                 max_value_w, new_max_value, old_max_value;
    logic signed [BUS_NUM-1:0][FIXED_DATA_WIDTH-1:0]    in_fixed_data_d;//input data for the next stage
    logic        [BUS_NUM-1:0]                          max_stage_vld;                

    always_comb begin
        max_value_w = {1'b1,{(FIXED_DATA_WIDTH-1){1'b0}}};
        for(int i=0;i<BUS_NUM;i++) begin
            if(in_fixed_data_vld[i])
                max_value_w = (in_fixed_data[i]>max_value_w) ? in_fixed_data[i] : max_value_w;
            else
                break;
        end
    end
    
    always_ff @(posedge clk or negedge rst_n) begin
        if(~rst_n) begin
            new_max_value <=0;
            old_max_value <=0;
            in_fixed_data_d <=0;
        end
        else if(|in_fixed_data_vld) begin
            new_max_value <= max_value_w;
            old_max_value <= new_max_value;
            in_fixed_data_d <= in_fixed_data;
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if(~rst_n) begin
            max_stage_vld<=0;
        end
        else begin
            max_stage_vld<=in_fixed_data_vld;
        end
    end

/////////////////////////////////////////////////////
//  Get xi-max_new, and max_old-max_new            //
//                                                 //
/////////////////////////////////////////////////////
    logic signed [BUS_NUM-1:0][FIXED_DATA_WIDTH-1:0]    xi_max_new; //xi-max_new
    logic signed [FIXED_DATA_WIDTH-1:0]                 max_old_new; //max_old-max_new
    logic        [BUS_NUM-1:0]                          fixed_in_vld;
    logic signed [FIXED_DATA_WIDTH-1:0]                 max_value_d; //max value for the next stage
    
    always_ff @(posedge clk or negedge rst_n) begin
        if(~rst_n) begin
            max_old_new <=0;
            max_value_d <=0;
        end
        else if(|max_stage_vld) begin
            max_old_new <= old_max_value-new_max_value;
            max_value_d <= new_max_value;
        end
    end

    genvar i;
    generate
        for(i=0;i<BUS_NUM;i++) begin
            always_ff @(posedge clk or negedge rst_n) begin
                if(~rst_n) begin
                    xi_max_new[i] <= 0;
                end
                else if(max_stage_vld[i]) begin
                    xi_max_new[i] <= in_fixed_data_d[i]-new_max_value;
                end
            end
        end
    endgenerate

    always_ff @(posedge clk or negedge rst_n) begin
        if(~rst_n) begin
            fixed_in_vld<=0;
        end
        else begin
            fixed_in_vld<=max_stage_vld;
        end
    end

/////////////////////////////////////////////////////
//  Convert xi-max_new, and max_old-max_new to fp  //
//                                                 //
/////////////////////////////////////////////////////
    logic signed [BUS_NUM-1:0][sig_width+exp_width : 0]  i2flt_xi;
    logic signed [sig_width+exp_width : 0]               i2flt_max;
    logic signed [BUS_NUM-1:0][sig_width+exp_width : 0]  in_float_xi;
    logic signed [sig_width+exp_width : 0]               in_float_max;
    logic        [BUS_NUM-1:0]                           in_float_xi_vld;
    logic signed [FIXED_DATA_WIDTH-1:0]                  max_value_dd; //max value for the next stage

    generate
        for(i = 0;i < BUS_NUM;i++)begin : i2flt_xi_stage
            DW_fp_i2flt #(sig_width, exp_width, isize, isign)
            i2flt_xi_inst ( 
                .a($signed(xi_max_new[i])), 
                .rnd(3'b000), 
                .z(i2flt_xi[i]), 
                .status() 
            );
        end
    endgenerate

    generate
        for(i = 0;i < BUS_NUM; i++)begin :  in_fix2float_xi_array
            always_ff @(posedge clk or negedge rst_n)begin
                if(~rst_n)begin
                    in_float_xi_vld[i] <= 0;
                    in_float_xi[i] <= 0;
                end
                else begin
                    in_float_xi_vld[i] <= fixed_in_vld[i];
                    if(fixed_in_vld[i]) begin
                        if(i2flt_xi[i] == 0)begin
                            in_float_xi[i] <= 0;
                        end
                        else begin
                            in_float_xi[i][sig_width+exp_width-1: 0] <= i2flt_xi[i][sig_width+exp_width-1: 0] - $signed({in_scale_pos_reg,{sig_width{1'b0}}}); // i2flt_z_inst * 2^(-in_scale_pos_reg) ;
                            in_float_xi[i][sig_width+exp_width] <= i2flt_xi[i][sig_width+exp_width];
                        end
                    end
                end
            end
        end
    endgenerate

    DW_fp_i2flt #(sig_width, exp_width, isize, isign)
    i2flt_max_inst ( 
        .a($signed(max_old_new)), 
        .rnd(3'b000), 
        .z(i2flt_max), 
        .status() 
    );

    always_ff @(posedge clk or negedge rst_n)begin
        if(~rst_n)begin
            in_float_max <= 0;
        end
        else if(|fixed_in_vld)begin
            max_value_dd <= max_value_d;
            if(i2flt_max == 0)begin
                in_float_max <= 0;
            end
            else begin
                in_float_max[sig_width+exp_width-1: 0] <= i2flt_max[sig_width+exp_width-1: 0] - $signed({in_scale_pos_reg,{sig_width{1'b0}}}); // in_float_max * 2^(-in_scale_pos_reg) ;
                in_float_max[sig_width+exp_width] <= i2flt_max[sig_width+exp_width];
            end
        end
    end

/////////////////////////////////////////////////////
//  Get exponent of xi-max_new and max_old-max_new //
//                                                 //
/////////////////////////////////////////////////////
logic signed    [sig_width+exp_width : 0]  exp_xi_max;
logic signed    [sig_width+exp_width : 0]  exp_max_max;

generate
    for(i = 0;i < BUS_NUM;i++)begin : exp_max_new_old
        DW_fp_exp #(sig_width, exp_width, ieee_compliance, exp_arch) 
        iexp_max_new_old (
        .a(inst_a),
        .z(z_inst),
        .status() );
    end
endgenerate


endmodule


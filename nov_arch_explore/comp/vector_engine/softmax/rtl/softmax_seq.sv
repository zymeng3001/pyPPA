module softmax
#(   
    parameter integer sig_width = 7,//bfloat16
    parameter integer exp_width = 8, 
    parameter integer BUS_NUM = 8, //input bus num, even number
    parameter integer BUS_NUM_WIDTH = 3,
    parameter integer DATA_NUM_WIDTH =  10,
    parameter integer SCALA_POS_WIDTH = 5,
    parameter integer FIXED_DATA_WIDTH = 8,
    parameter integer MEM_WIDTH = (BUS_NUM*FIXED_DATA_WIDTH),
    parameter integer MEM_DEPTH = 512,

    localparam        isize = 32,
    localparam        isign = 1, //signed integer number
    localparam        ieee_compliance = 0, //No support to NaN adn denormals
    localparam        exp_arch = 1, //speed optimization for exp ip
    localparam        ln_arch  = 1, //speed optimization for ln ip
    localparam        ln_extra_prec = 0 //internal extra prec
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
    input  logic                     [DATA_NUM_WIDTH-1:0]            in_data_num,
    input  logic                                                     in_data_num_vld,


    // Data Signals
    input  logic signed [BUS_NUM-1:0][FIXED_DATA_WIDTH-1:0]          in_fixed_data,
    input  logic        [BUS_NUM-1:0]                                in_fixed_data_vld,
    output logic                                                     in_ready,
    output logic signed [BUS_NUM-1:0][FIXED_DATA_WIDTH-1:0]          out_fixed_data,
    output logic        [BUS_NUM-1:0]                                out_fixed_data_vld,
    output logic                                                     out_fixed_last
);  
    logic signed              [SCALA_POS_WIDTH-1:0]                      in_scale_pos_reg;
    logic signed              [SCALA_POS_WIDTH-1:0]                      out_scale_pos_reg;
    logic                     [DATA_NUM_WIDTH-1:0]                       data_num;

    //2 STAGE FSM CONTROL SIGNAL
    logic  first_stage_w_rst,first_stage_r_rst,exp_sum_acc_rst, second_stage_rst;
    assign second_stage_rst = out_fixed_last;
    always_ff @(posedge clk or negedge rst_n) begin
        if(~rst_n) begin
            in_ready<=0;
            exp_sum_acc_rst<=0;
        end
        else begin
            in_ready <= first_stage_w_rst;
            exp_sum_acc_rst<=first_stage_r_rst;
        end
    end

/////////////////////////////////////////////////////
//First set the data num (length of RMSnorm vector)//
//and scale position and gamma array               //
/////////////////////////////////////////////////////
    always_ff @(posedge clk or negedge rst_n) begin
        if(~rst_n)
            data_num <=0;
        else if(in_data_num_vld)
            data_num <= in_data_num;
    end

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
//INPUT FSM Control for input Memory               //
//                                                 //
/////////////////////////////////////////////////////
    logic           [DATA_NUM_WIDTH-1:0]    input_cnt,next_input_cnt;
    logic           [DATA_NUM_WIDTH-1:0]    read_cnt,next_read_cnt;
    logic           [BUS_NUM_WIDTH:0]       last_cnt, last_cnt_w; //record the remain count of last transmission.
    logic           [2:0]                   wstate,next_wstate;
    logic           [2:0]                   rstate,next_rstate;
    logic signed    [FIXED_DATA_WIDTH-1:0]  max_value_w, max_value;
    logic                                   max_vld,next_max_vld;

    always_comb begin //write_count
        next_input_cnt = input_cnt;
        last_cnt_w = last_cnt;
        next_max_vld = max_vld;
        for(int i = 0;i<BUS_NUM;i++)begin
            if(in_fixed_data_vld[i]) begin
                next_input_cnt = next_input_cnt + 1;
                if(next_input_cnt==data_num) begin
                    last_cnt_w = i+1;
                    next_max_vld = 1;
                    break;
                end
            end
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if(~rst_n) begin
            input_cnt <= 0;
        end
        else if(first_stage_w_rst) //TODO: finish processing clean for next round
            input_cnt <= 0;
        else if(input_cnt == data_num) begin
            input_cnt <= input_cnt;
        end
        else if(|in_fixed_data_vld) begin
            input_cnt <= next_input_cnt;
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin //last write value
        if(~rst_n) begin
            last_cnt <= 0;
        end
        else if(first_stage_w_rst) begin//TODO: finish processing clean for next round
            last_cnt <= 0;
        end
        else begin
            last_cnt <= last_cnt_w;
        end
    end

    always_comb begin //read_cnt
        next_read_cnt = read_cnt;
        if((max_vld && rstate == 0)|rstate==1) begin
            next_read_cnt = next_read_cnt+BUS_NUM;
            if(next_read_cnt>=data_num) begin
                next_read_cnt = data_num;
            end
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if(~rst_n) begin
            read_cnt <= 0;
        end
        else if(first_stage_r_rst) //TODO: finish processing, clean for next round
            read_cnt <= 0;
        else if((max_vld && rstate == 0)|rstate==1) begin
            read_cnt <= next_read_cnt;
        end
    end


logic           [BUS_NUM-1:0]           rd_vld,next_rd_vld;
logic                                   last_mem_data,last_mem_data_w; //last rdata from input memory
// Signals for xi_reg ports
logic           [$clog2(MEM_DEPTH)-1:0] xi_waddr,next_xi_waddr;
logic           [$clog2(MEM_DEPTH)-1:0] xi_raddr,next_xi_raddr;
logic                                   xi_wen,next_xi_wen;
logic           [MEM_WIDTH-1:0]         xi_wdata,next_xi_wdata;
logic                                   xi_ren,next_xi_ren;
logic signed    [MEM_WIDTH-1:0]         xi_rdata;
// save the max value for 1st stage's read
logic signed    [FIXED_DATA_WIDTH-1:0]  max_value_d,next_max_value_d;
// save the last cnt value for 1st stage's read
logic           [BUS_NUM_WIDTH:0]       last_cnt_d, next_last_cnt_d;
// 1st stage's write can be perform once 1st stage starts reading
logic                                   next_first_stage_w_rst;

//write state control
always_comb begin
    next_xi_waddr=xi_waddr;
    next_xi_wen=0;
    next_xi_wdata=0;
    next_wstate = wstate;

    if(|in_fixed_data_vld && wstate==0) begin //write first value
        next_xi_wen = 1'b1;
        next_xi_wdata = in_fixed_data;
        next_wstate = 1;
    end
    else if(wstate == 1 && input_cnt == data_num) begin //last write
        next_xi_waddr = 0;
        next_wstate = 2;
    end
    else if(|in_fixed_data_vld && wstate ==1) begin // write
        next_xi_wen = 1'b1;
        next_xi_waddr = xi_waddr + 1'b1;
        next_xi_wdata = in_fixed_data;
    end
    else if(first_stage_w_rst && wstate ==2) begin //TODO: finish round, reset state
        next_wstate = 0;
    end
end

//read state control
always_comb begin
    next_xi_raddr=xi_raddr;
    next_xi_ren = 0;
    next_rstate = rstate;
    last_mem_data_w = 0;
    next_rd_vld = 0;
    next_max_value_d = max_value_d;
    next_last_cnt_d = last_cnt_d;
    next_first_stage_w_rst = 0;

    if(max_vld && rstate == 0) begin //start 1st read
        next_xi_ren = 1'b1;
        next_rstate = 1;
        next_max_value_d = max_value;
        next_last_cnt_d = last_cnt;
        next_first_stage_w_rst = 1;
    end
    else if(rstate == 1) begin //read
        if(read_cnt==data_num) begin //last read
            next_xi_raddr = 0;
            next_rstate = 2;
            last_mem_data_w = 1;
            for(int i=0;i<last_cnt_d;i++) begin
                next_rd_vld[i] = 1'b1;
            end
        end
        else begin
            next_xi_ren = 1'b1;
            next_xi_raddr = xi_raddr+1;
            next_rd_vld = {BUS_NUM{1'b1}};
        end
    end
    else if(first_stage_r_rst && rstate ==2) begin //TODO: finish round, reset state
        next_rstate = 0;
        next_max_value_d = 0;
        next_last_cnt_d = 0;
    end
end

always_ff @(posedge clk or negedge rst_n) begin
    if(~rst_n) begin
        xi_raddr <= 0;
        xi_waddr <= 0;
        xi_wen <= 0;
        xi_wdata <= 0;
        xi_ren <= 0;
        rstate <= 0;
        wstate <= 0;
        rd_vld <= 0;
        last_mem_data <=0;
        max_value_d <=0;
        last_cnt_d <=0;
        first_stage_w_rst <=0;
    end
    else begin
        xi_waddr <= next_xi_waddr;
        xi_raddr <= next_xi_raddr;
        xi_wen <= next_xi_wen;
        xi_wdata <= next_xi_wdata;
        xi_ren <= next_xi_ren;
        rstate <= next_rstate;
        wstate <= next_wstate;
        rd_vld <= next_rd_vld;
        last_mem_data <= last_mem_data_w;
        max_value_d <= next_max_value_d;
        last_cnt_d <= next_last_cnt_d;
        first_stage_w_rst <= next_first_stage_w_rst;
    end
end

mem_dp #(.DATA_BIT(MEM_WIDTH), .DEPTH(MEM_DEPTH))
xi_reg (
    .clk                    (clk),
    .waddr                  (xi_waddr),
    .raddr                  (xi_raddr),
    .wen                    (xi_wen),
    .wdata                  (xi_wdata),
    .ren                    (xi_ren),
    .rdata                  (xi_rdata)
);

/////////////////////////////////////////////////////
//  Get the new max from current input data        //
//                                                 //
/////////////////////////////////////////////////////             
    always_ff @(posedge clk or negedge rst_n) begin
        if(~rst_n) begin
            max_vld <= 0;
        end
        else if(first_stage_w_rst) begin
            max_vld <= 0;
        end
        else begin
            max_vld <= next_max_vld;
        end
    end

    always_comb begin
        max_value_w = max_value;
        for(int i=0;i<BUS_NUM;i++) begin
            if(in_fixed_data_vld[i])
                max_value_w = ($signed(in_fixed_data[i])>$signed(max_value_w)) ? in_fixed_data[i] : max_value_w;
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if(~rst_n) begin
            max_value <={1'b1,{(FIXED_DATA_WIDTH-1){1'b0}}};
        end
        else if (first_stage_w_rst) begin//TODO: finish round, reset state
            max_value <={1'b1,{(FIXED_DATA_WIDTH-1){1'b0}}};
        end
        else if(|in_fixed_data_vld && input_cnt!=data_num) begin
            max_value <= max_value_w;
        end
    end

/////////////////////////////////////////////////////
//  Get the rdata from mem and subtract by max     //
//                                                 //
/////////////////////////////////////////////////////
    logic signed [BUS_NUM-1:0][FIXED_DATA_WIDTH-1:0]    xi_max; //xi-max
    logic        [BUS_NUM-1:0]                          xi_max_vld;
    logic                                               xi_max_last;
    logic signed [BUS_NUM-1:0][FIXED_DATA_WIDTH-1:0]    xi_rdata_split;

    assign xi_rdata_split = xi_rdata;

    genvar i;
    generate
        for(i=0;i<BUS_NUM;i++) begin
            always_ff @(posedge clk or negedge rst_n) begin
                if(~rst_n) begin
                    xi_max[i] <= 0;
                end
                else if(rd_vld[i]) begin
                    xi_max[i] <= xi_rdata_split[i]-max_value_d;
                end
            end
        end
    endgenerate

    always_ff @(posedge clk or negedge rst_n) begin
        if(~rst_n) begin
            xi_max_vld <= 0;
            xi_max_last <= 0;
        end
        else begin
            xi_max_vld<=rd_vld;
            xi_max_last<=last_mem_data;
        end
    end

/////////////////////////////////////////////////////
//  Save xi-max into Memory                        //
//                                                 //
/////////////////////////////////////////////////////
    // Signals for xi_max_mem_reg ports
    logic           [$clog2(MEM_DEPTH)-1:0]     xi_max_mem_raddr,next_xi_max_mem_raddr;
    logic           [$clog2(MEM_DEPTH)-1:0]     xi_max_mem_waddr,next_xi_max_mem_waddr;
    logic                                       xi_max_mem_wen;
    logic           [MEM_WIDTH-1:0]             xi_max_mem_wdata;
    logic                                       xi_max_mem_ren;
    logic signed    [MEM_WIDTH-1:0]             xi_max_mem_rdata;
    logic           [BUS_NUM-1:0]               xi_max_mem_rd_vld;
    logic                                       xi_max_mem_rdata_last;
    // Signals for exp mem controll
    logic           [1:0]                       xi_max_mem_state;
    logic                                       xi_max_mem_readaddr_inc;
    logic           [BUS_NUM_WIDTH:0]           last_cnt_dd; //inherit from previous stage, because it will be updated with new ones.
    logic           [DATA_NUM_WIDTH-1:0]        xi_max_mem_rd_cnt;

    assign xi_max_mem_wen = |xi_max_vld;
    assign xi_max_mem_wdata = xi_max;

    always_comb begin
        next_xi_max_mem_waddr = xi_max_mem_waddr;
        if(first_stage_r_rst) //TODO: clean when finish
            next_xi_max_mem_waddr = 0;
        else if(xi_max_last) //finish last write
            next_xi_max_mem_waddr = 0;
        else if(|xi_max_vld) begin // write addr increment
            next_xi_max_mem_waddr = xi_max_mem_waddr + 1;
        end
    end

    always_comb begin
        next_xi_max_mem_raddr = xi_max_mem_raddr;
        if(second_stage_rst) begin
            next_xi_max_mem_raddr = 0;
        end
        else if(xi_max_mem_readaddr_inc) begin //read addr increment
            next_xi_max_mem_raddr = xi_max_mem_raddr + 1;
        end
    end

    always_ff @(posedge clk or negedge rst_n)begin
        if(~rst_n)begin
            xi_max_mem_waddr <= 0;
            xi_max_mem_raddr <= 0;
        end
        else begin
            xi_max_mem_waddr <= next_xi_max_mem_waddr;
            xi_max_mem_raddr <= next_xi_max_mem_raddr;
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin //read for exp mem is enabled after acc result is valid, this way exp is ready for subtract with ln at ln stage.
        if(~rst_n)begin
            xi_max_mem_state <= 0;
            xi_max_mem_ren <= 0;
            xi_max_mem_readaddr_inc <= 0;
            last_cnt_dd <= 0;
            xi_max_mem_rd_cnt <= 0;
            xi_max_mem_rd_vld <= 0;
            xi_max_mem_rdata_last <= 0;
        end
        else begin
            case(xi_max_mem_state)
                2'b00: begin //IDLE, when adder out last, start first read
                    if(first_stage_r_rst) begin
                        xi_max_mem_ren <= 1'b1;
                        xi_max_mem_state <= 2'b01;
                        xi_max_mem_readaddr_inc <= 1'b1;
                        last_cnt_dd <= last_cnt_d; // last cnt if last input is less than bus_num
                        xi_max_mem_rd_cnt <= xi_max_mem_rd_cnt + BUS_NUM;
                    end
                end
                2'b01: begin //Continue Read, addr increase
                    if(xi_max_mem_rd_cnt == data_num)begin
                        xi_max_mem_ren <= 1'b0;
                        xi_max_mem_state <= 2'b10;
                        for(int i=0;i<BUS_NUM;i++) begin
                            if(i<last_cnt_dd)
                                xi_max_mem_rd_vld[i] <= 1'b1;
                            else
                                xi_max_mem_rd_vld[i] <= 1'b0;
                        end
                        xi_max_mem_rdata_last <= 1;
                    end
                    else if(xi_max_mem_rd_cnt>=data_num-last_cnt_dd) begin
                        xi_max_mem_rd_cnt <= data_num;
                        xi_max_mem_readaddr_inc <= 1'b0;
                        xi_max_mem_rd_vld <= {BUS_NUM{1'b1}};
                    end
                    else begin
                        xi_max_mem_rd_vld <= {BUS_NUM{1'b1}};
                        xi_max_mem_rd_cnt <= xi_max_mem_rd_cnt + BUS_NUM;
                    end
                end
                2'b10: begin //finish read state
                    xi_max_mem_rd_vld <= 0;
                    xi_max_mem_rdata_last <= 0;
                    if(second_stage_rst) begin//TODO: clean when finish
                        last_cnt_dd <= 0;
                        xi_max_mem_rd_cnt <= 0;
                        xi_max_mem_state <= 0;
                    end
                end
            endcase
        end
    end

    mem_dp #(.DATA_BIT(MEM_WIDTH), .DEPTH(MEM_DEPTH))
    xi_max_reg (
        .clk                    (clk),
        .waddr                  (xi_max_mem_waddr),
        .wen                    (xi_max_mem_wen),
        .wdata                  (xi_max_mem_wdata),
        .raddr                  (xi_max_mem_raddr),
        .ren                    (xi_max_mem_ren),
        .rdata                  (xi_max_mem_rdata)
    );

/////////////////////////////////////////////////////
//  Convert xi-max to fp                           //
//                                                 //
/////////////////////////////////////////////////////
    logic signed [BUS_NUM-1:0][sig_width+exp_width : 0]  i2flt_xi_max_w, i2flt_xi_max;
    logic        [BUS_NUM-1:0]                           flt_xi_max_vld;
    logic                                                i2flt_last;

    generate
        for(i = 0;i < BUS_NUM;i++)begin : i2flt_xi_max_stage
            DW_fp_i2flt #(sig_width, exp_width, FIXED_DATA_WIDTH, isign)
            i2flt_xi_max_inst ( 
                .a($signed(xi_max[i])), 
                .rnd(3'b000), 
                .z(i2flt_xi_max_w[i]), 
                .status() 
            );
        end
    endgenerate

    generate
        for(i = 0;i < BUS_NUM; i++)begin
            always_ff @(posedge clk or negedge rst_n)begin
                if(~rst_n)begin
                    flt_xi_max_vld[i] <= 0;
                    i2flt_xi_max[i] <= 0;
                end
                else begin
                    flt_xi_max_vld[i] <= xi_max_vld[i];
                    if(xi_max_vld[i]) begin
                        if(i2flt_xi_max_w[i] == 0)begin
                            i2flt_xi_max[i] <= 0;
                        end
                        else begin
                            i2flt_xi_max[i][sig_width+exp_width-1: 0] <= $signed(i2flt_xi_max_w[i][sig_width+exp_width-1: 0]) - $signed({in_scale_pos_reg,{sig_width{1'b0}}}); // i2flt_z_inst * 2^(-in_scale_pos_reg) ;
                            i2flt_xi_max[i][sig_width+exp_width] <= i2flt_xi_max_w[i][sig_width+exp_width];
                        end
                    end
                    else begin
                        i2flt_xi_max[i] <= 0;
                    end
                end
            end
        end
    endgenerate

    always_ff @(posedge clk or negedge rst_n)begin
        if(~rst_n)begin
            i2flt_last<=0;
        end
        else begin
            i2flt_last<=xi_max_last;
        end
    end

/////////////////////////////////////////////////////
//  Get exponent of xi-max                         //
//                                                 //
/////////////////////////////////////////////////////
    logic signed    [BUS_NUM-1:0][sig_width+exp_width : 0]  exp_xi_max,exp_xi_max_w;
    logic           [BUS_NUM-1:0]                           exp_xi_max_vld;
    logic                                                   exp_last;

    generate
        for(i = 0;i < BUS_NUM;i++)begin : exp_xi_max_stage
            DW_fp_exp #(sig_width, exp_width, ieee_compliance, exp_arch) 
            iexp_xi_max(
            .a(i2flt_xi_max[i]),
            .z(exp_xi_max_w[i]),
            .status() );

            always_ff @(posedge clk or negedge rst_n)begin
                if(~rst_n)begin
                    exp_xi_max[i]<=0;
                    exp_xi_max_vld[i]<=0;
                end
                else begin
                    exp_xi_max_vld[i]<=flt_xi_max_vld[i];
                    if(flt_xi_max_vld[i]) begin
                        exp_xi_max[i]<=exp_xi_max_w[i];
                    end
                    else begin
                        exp_xi_max[i]<=0;
                    end
                end
            end
        end
    endgenerate

    always_ff @(posedge clk or negedge rst_n)begin
        if(~rst_n)begin
            exp_last<=0;
        end
        else begin
            exp_last<=i2flt_last;
        end
    end
    //retiming for exponent unit
    localparam SLICE_REG_NUM = 1;
    logic [BUS_NUM-1:0][sig_width+exp_width : 0] exp_xi_max_delay;
    logic [BUS_NUM-1:0]                          exp_xi_max_vld_delay;
    logic                                        exp_last_delay;

    generate
        for(i = 0;i < SLICE_REG_NUM;i++)begin : exp_xi_max_retiming_stage
            logic [BUS_NUM-1:0][sig_width+exp_width : 0] timing_register_float;
            logic [BUS_NUM-1:0]                          timing_register_vld;
            logic                                        timing_register_last;
            if(i == 0)begin
                always_ff@(posedge clk or negedge rst_n)begin
                    if(~rst_n)begin
                        timing_register_float <= 0;
                        timing_register_vld <= 0;
                        timing_register_last <= 0;
                    end
                    else begin
                        timing_register_float <= exp_xi_max;
                        timing_register_vld <= exp_xi_max_vld;
                        timing_register_last <= exp_last;
                    end
                end
            end
            else begin
                always_ff@(posedge clk or negedge rst_n)begin
                    if(~rst_n) begin
                        timing_register_float <= 0;
                        timing_register_vld <= 0;
                        timing_register_last <= 0;    
                    end
                    else begin
                        timing_register_float <= exp_xi_max_retiming_stage[i-1].timing_register_float;
                        timing_register_vld <= exp_xi_max_retiming_stage[i-1].timing_register_vld;
                        timing_register_last <= exp_xi_max_retiming_stage[i-1].timing_register_last;
                    end
                end
            end
        end
    endgenerate

    always_comb begin
        exp_xi_max_delay = exp_xi_max_retiming_stage[SLICE_REG_NUM-1].timing_register_float;
        exp_xi_max_vld_delay = exp_xi_max_retiming_stage[SLICE_REG_NUM-1].timing_register_vld;
        exp_last_delay = exp_xi_max_retiming_stage[SLICE_REG_NUM-1].timing_register_last;
    end

/////////////////////////////////////////////////////
//  Floating point Adder Tree to add Bus Num of    //
//                exp results                      //
/////////////////////////////////////////////////////
    logic signed    [sig_width+exp_width : 0]               adder_out;
    logic                                                   adder_out_vld;
    logic                                                   adder_out_last;

    assign first_stage_r_rst = adder_out_last;

    fadd_tree #(
        .sig_width(sig_width),
        .exp_width(exp_width),
        .MAC_NUM(BUS_NUM)
    ) fadd_tree_inst (
        .clk(clk),
        .rstn(rst_n),
        .idata(exp_xi_max_delay),
        .idata_valid(exp_xi_max_vld_delay),
        .last_in(exp_last_delay),
        .odata(adder_out),
        .odata_valid(adder_out_vld),
        .last_out(adder_out_last)
    );

/////////////////////////////////////////////////////
//  Accumulate the results from adder tree         //
//                                                 //
/////////////////////////////////////////////////////
    logic signed    [sig_width+exp_width : 0]   acc_out,acc_out_w;
    logic                                       acc_vld;

    DW_fp_add_DG #(sig_width, exp_width, ieee_compliance)
    acc_inst ( .a(acc_out), .b(adder_out), .rnd(3'b000), .DG_ctrl(adder_out_vld), .z(acc_out_w),.status());

    always_ff @(posedge clk or negedge rst_n)begin
        if(~rst_n)begin
            acc_out <= 0;
        end
        else if (exp_sum_acc_rst) begin //TODO: clean when last complete.
            acc_out <= 0;
        end
        else if (adder_out_vld)begin
            acc_out <= acc_out_w;
        end
    end

    always_ff @(posedge clk or negedge rst_n)begin
        if(~rst_n)begin
            acc_vld <= 0;
        end
        else begin
            acc_vld <= adder_out_last;
        end
    end

/////////////////////////////////////////////////////
//  Natural Logarithm                              //
//                                                 //
/////////////////////////////////////////////////////

    logic signed    [sig_width+exp_width : 0]   ln_out,ln_out_w;
    logic                                       ln_vld;

    DW_fp_ln #(sig_width, exp_width, ieee_compliance, ln_extra_prec, ln_arch) 
    inst_ln (
    .a(acc_out),
    .z(ln_out_w),
    .status() );

    always_ff @(posedge clk or negedge rst_n)begin
        if(~rst_n)begin
            ln_out <= 0;
        end
        else if(acc_vld) begin
            ln_out <= ln_out_w;
        end
    end

    always_ff @(posedge clk or negedge rst_n)begin
        if(~rst_n)begin
            ln_vld <= 0;
        end
        else begin
            ln_vld <= acc_vld;
        end
    end

/////////////////////////////////////////////////////
//  Read xi-max (FIXED) and convert                //
//                to Floating point                //
/////////////////////////////////////////////////////
    logic signed    [BUS_NUM-1:0][FIXED_DATA_WIDTH-1:0]     mem_rdata_fixed;
    logic signed    [BUS_NUM-1:0][sig_width+exp_width : 0]  mem_rdata_flt,mem_rdata_flt_w;
    logic           [BUS_NUM-1:0]                           mem_rdata_flt_vld;
    logic                                                   mem_rdata_flt_last;
    
    logic signed                 [sig_width+exp_width : 0]  ln_out_d;
   
   assign mem_rdata_fixed = xi_max_mem_rdata;
    //need to delay ln_out by one cycle to sync with mem_rdata after i2flt conversion
    always_ff @(posedge clk or negedge rst_n)begin
        if(~rst_n)begin
            ln_out_d <= 0;
        end
        else if(ln_vld)begin
            ln_out_d <= ln_out;
        end
    end

    //mem rdata i2flt conversion
     generate
        for(i = 0;i < BUS_NUM;i++)begin : i2flt_mem_rdata_stage
            DW_fp_i2flt #(sig_width, exp_width, FIXED_DATA_WIDTH, isign)
            i2flt_mem_rdata_inst ( 
                .a($signed(mem_rdata_fixed[i])), 
                .rnd(3'b000), 
                .z(mem_rdata_flt_w[i]), 
                .status() 
            );
        end
    endgenerate

    generate
        for(i = 0;i < BUS_NUM; i++)begin
            always_ff @(posedge clk or negedge rst_n)begin
                if(~rst_n)begin
                    mem_rdata_flt_vld[i] <= 0;
                    mem_rdata_flt[i] <= 0;
                end
                else begin
                    mem_rdata_flt_vld[i] <= xi_max_mem_rd_vld[i];
                    if(xi_max_mem_rd_vld[i]) begin
                        if(mem_rdata_flt_w[i] == 0)begin
                            mem_rdata_flt[i] <= 0;
                        end
                        else begin
                            mem_rdata_flt[i][sig_width+exp_width-1: 0] <= $signed(mem_rdata_flt_w[i][sig_width+exp_width-1: 0]) - $signed({in_scale_pos_reg,{sig_width{1'b0}}}); // i2flt_z_inst * 2^(-in_scale_pos_reg) ;
                            mem_rdata_flt[i][sig_width+exp_width] <= mem_rdata_flt_w[i][sig_width+exp_width];
                        end
                    end
                    else begin
                        mem_rdata_flt[i] <= 0;
                    end
                end
            end
        end
    endgenerate

    always_ff @(posedge clk or negedge rst_n)begin
        if(~rst_n)begin
            mem_rdata_flt_last <= 0;
        end
        else begin
            mem_rdata_flt_last <= xi_max_mem_rdata_last;
        end
    end

/////////////////////////////////////////////////////
//  subtract xi-max-mem-rdata with exp sum         //
//                                                 //
/////////////////////////////////////////////////////
    logic signed                 [sig_width+exp_width : 0]  ln_out_d_neg;
    
    logic           [BUS_NUM-1:0]                           subtract_vld;
    logic                                                   subtract_last;
    logic           [BUS_NUM-1:0][sig_width+exp_width : 0]  subtract_result_w, subtract_result;                     
    
    assign ln_out_d_neg = {~ln_out_d[sig_width+exp_width],ln_out_d[sig_width+exp_width-1:0]}; //ln_out_d = -ln_out_d

    generate
        for(i=0;i<BUS_NUM;i++) begin:subtract_stage
            logic subtract_in_vld;
            assign subtract_in_vld = mem_rdata_flt_vld[i];

            DW_fp_add_DG #(sig_width, exp_width, ieee_compliance)
            subtract_inst ( .a(mem_rdata_flt[i]), .b(ln_out_d_neg), .rnd(3'b000), .DG_ctrl(subtract_in_vld), .z(subtract_result_w[i]),.status());

            always_ff @(posedge clk or negedge rst_n)begin
                if(~rst_n)begin
                    subtract_result[i] <= 0;
                end
                else if(subtract_in_vld) begin
                    subtract_result[i] <= subtract_result_w[i];
                end
            end

            always_ff @(posedge clk or negedge rst_n)begin
                if(~rst_n)begin
                    subtract_vld[i]<=0;
                end
                else begin
                    subtract_vld[i]<=subtract_in_vld;
                end
            end
        end
    endgenerate

    always_ff @(posedge clk or negedge rst_n)begin
        if(~rst_n)begin
            subtract_last<=0;
        end
        else begin
            subtract_last<=mem_rdata_flt_last;
        end
    end

/////////////////////////////////////////////////////
//  Final Exponent of the subtracted result        //
//                                                 //
/////////////////////////////////////////////////////
    logic           [BUS_NUM-1:0]                           final_exp_vld;
    logic                                                   final_exp_last;
    logic           [BUS_NUM-1:0][sig_width+exp_width : 0]  final_exp_result_w, final_exp_result;

    generate
        for(i=0;i<BUS_NUM;i++) begin:last_exp_stage

            DW_fp_exp #(sig_width, exp_width, ieee_compliance, exp_arch) 
            final_exp_inst(
            .a(subtract_result[i]),
            .z(final_exp_result_w[i]),
            .status() );

            always_ff @(posedge clk or negedge rst_n)begin
                if(~rst_n)begin
                    final_exp_result[i] <= 0;
                end
                else if(subtract_vld[i]) begin
                    final_exp_result[i] <= final_exp_result_w[i];
                end
            end

            always_ff @(posedge clk or negedge rst_n)begin
                if(~rst_n)begin
                    final_exp_vld[i]<=0;
                end
                else begin
                    final_exp_vld[i]<=subtract_vld[i];
                end
            end
        end
    endgenerate

    always_ff @(posedge clk or negedge rst_n)begin
        if(~rst_n)begin
            final_exp_last<=0;
        end
        else begin
            final_exp_last<=subtract_last;
        end
    end

/////////////////////////////////////////////////////
//  Output float to fixed conversion               //
//                                                 //
/////////////////////////////////////////////////////
    logic signed [BUS_NUM-1:0][sig_width+exp_width : 0]       out_scaled_data;
    logic        [BUS_NUM-1:0]                                out_scaled_data_vld;
    logic                                                     out_scaled_last;
    logic signed [BUS_NUM-1:0][isize-1 : 0]                   out_flt2i_data;
    logic signed [BUS_NUM-1:0][FIXED_DATA_WIDTH-1 : 0]        nxt_out_fixed_data;

    generate
        for(i = 0;i < BUS_NUM; i++)begin : output_flt2fixed_conversion
            always_ff @(posedge clk or negedge rst_n)begin
                if(~rst_n)begin
                    out_scaled_data[i] <= 0;
                    out_scaled_data_vld[i] <= 0;
                end
                else begin
                    if(final_exp_result[i] == 0)begin
                        out_scaled_data[i] <= 0;
                    end
                    else begin
                        out_scaled_data[i][sig_width+exp_width-1: 0] <= $signed(final_exp_result[i][sig_width+exp_width-1: 0]) + $signed({out_scale_pos_reg,{sig_width{1'b0}}});// float_RMSnorm * 2^(out_scale_pos_reg)
                        out_scaled_data[i][sig_width+exp_width] <= final_exp_result[i][sig_width+exp_width];    
                    end
                    out_scaled_data_vld[i] <= final_exp_vld[i];
                end
            end

            DW_fp_flt2i #(sig_width, exp_width, isize, isign)
            output_flt2fixed_inst ( 
                    .a(out_scaled_data[i]), 
                    .rnd(3'b000), 
                    .z(out_flt2i_data[i]), 
                    .status() 
            );

            always_comb begin
                nxt_out_fixed_data[i] = out_flt2i_data[i];
                if($signed(out_flt2i_data[i]) > 127) begin
                    nxt_out_fixed_data[i] = 8'b0111_1111; //127
                end
                else if($signed(out_flt2i_data[i]) < -128) begin
                    nxt_out_fixed_data[i] = 8'b1000_0000; //-128
                end
            end

            always_ff@ (posedge clk or negedge rst_n)begin
                if(~rst_n)begin
                    out_fixed_data_vld[i] <= 0;
                    out_fixed_data[i] <= 0;
                end
                else begin
                    out_fixed_data_vld[i] <= out_scaled_data_vld[i];
                    out_fixed_data[i] <= nxt_out_fixed_data[i];
                end
            end

        end
    endgenerate

    always_ff @(posedge clk or negedge rst_n)begin
        if(~rst_n)begin
            out_scaled_last<=0;
            out_fixed_last<=0;
        end
        else begin
            out_scaled_last<=final_exp_last;
            out_fixed_last<=out_scaled_last;
        end
    end

endmodule
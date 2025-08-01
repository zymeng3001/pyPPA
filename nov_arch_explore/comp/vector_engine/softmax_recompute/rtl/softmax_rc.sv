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

module softmax_rc
#(
    parameter integer GBUS_DATA_WIDTH = `GBUS_DATA_WIDTH,
    parameter integer BUS_NUM = `MAC_MULT_NUM, //input bus num, even number
    parameter integer FIXED_DATA_WIDTH = `IDATA_WIDTH,
    parameter integer EXP_STAGE_DELAY = 1,

    parameter integer sig_width = `SIG_WIDTH,
    parameter integer exp_width = `EXP_WIDTH,
    localparam        isize = 32,
    localparam        isign = 1, //signed integer number
    localparam        ieee_compliance = 0, //No support to NaN adn denormals
    localparam        exp_arch = 1 //speed optimization for exp ip
)(
    input  logic                                                     clk,
    input  logic                                                     rst_n,

    input  CONTROL_STATE                                             control_state,
    input  logic                                                     control_state_update,

    input  logic                                                     rc_cfg_vld,
    input  RC_CONFIG                                                 rc_cfg,

    input  USER_CONFIG                                               usr_cfg,
    input                                                            usr_cfg_vld,

    input  logic                                                     model_cfg_vld,
    input  MODEL_CONFIG                                              model_cfg,
    
    //for PV recompute
    input  logic        [(`MAC_MULT_NUM * FIXED_DATA_WIDTH)-1:0]     in_bus_data, //from head sram during PV
    input  logic                                                     in_bus_data_vld,

    //for getting max
    input  logic                                                     gbus_wen, //from gbus during QK write back
    input  logic        [GBUS_DATA_WIDTH-1:0]                        gbus_wdata,
    input  BUS_ADDR                                                  gbus_addr,
    input  logic                                                     clear, //head finish & control_state_reg == ATT_PV_STATE, clear max and exp_sum

    //to core recompute
    output logic        [`RECOMPUTE_SCALE_WIDTH-1:0]                 rc_scale,
    output logic                                                     rc_scale_vld,

    //to abuf for PV
    output logic        [(`MAC_MULT_NUM * FIXED_DATA_WIDTH)-1:0]     out_bus_data,
    output logic                                                     out_bus_data_vld
);
//input gating of control signal
RC_CONFIG                                               rc_cfg_reg;
CONTROL_STATE                                           control_state_reg;
MODEL_CONFIG                                            model_cfg_reg;

USER_CONFIG                                             usr_cfg_reg;
logic   signed  [GBUS_DATA_WIDTH-1:0]                   gbus_data_reg;
BUS_ADDR                                                gbus_addr_reg;
logic                                                   gbus_data_reg_vld;
logic   signed  [(`MAC_MULT_NUM * FIXED_DATA_WIDTH)-1:0]in_bus_data_reg;
logic                                                   in_bus_data_vld_reg;

//max
logic   signed  [FIXED_DATA_WIDTH-1:0]                  max_reg, max_reg_w;

//xi - max
logic signed [BUS_NUM-1:0][FIXED_DATA_WIDTH:0]          xi_max; //xi-max
logic                                                   xi_max_vld;
logic signed [BUS_NUM-1:0][FIXED_DATA_WIDTH-1:0]        xi_data_split;

//convert xi-max to fp
logic   [BUS_NUM-1:0][sig_width+exp_width : 0]          i2flt_xi_max_dequant;
logic   [BUS_NUM-1:0]                                   i2flt_xi_max_dequant_vld;
logic                                                   i2flt_xi_max_dequant_vld_out;
logic   [sig_width+exp_width : 0]                       softmax_input_dequant_scale_reg;
logic                                                   i2flt_xi_max_vld; 

//get exp of xi-max fp
logic   [BUS_NUM-1:0][sig_width+exp_width : 0]          exp_xi_max,exp_xi_max_w;
logic                                                   exp_xi_max_vld;
logic   [BUS_NUM-1:0][sig_width+exp_width : 0]          exp_xi_max_delay;
logic                                                   exp_xi_max_vld_delay;

//quantize the exp and output to abuf
logic   [sig_width+exp_width : 0]                       softmax_exp_quant_scale_reg;
logic   [BUS_NUM-1:0]                                   exp_xi_max_scaled_vld;
logic                                                   exp_xi_max_scaled_vld_out;

//get exp sum
logic   [$clog2(`MAX_CONTEXT_LENGTH_WITH_GQA):0]        nxt_exp_sum_cnt,exp_sum_cnt;
logic   [BUS_NUM-1:0]                                   nxt_fadd_tree_vld, fadd_tree_vld;
logic   [BUS_NUM-1:0][sig_width+exp_width : 0]          nxt_fadd_tree_in, fadd_tree_in;
logic                                                   nxt_fadd_tree_in_last, fadd_tree_in_last; //indicate last data to acc, pulse.
logic signed    [sig_width+exp_width : 0]               adder_out;
logic                                                   adder_out_vld;
logic signed    [sig_width+exp_width : 0]               acc_out,acc_out_w;
logic                                                   acc_vld;
logic                                                   adder_out_last;

//get 1 over exp sum
logic   [sig_width+exp_width : 0]                       one_over_exp_sum, one_over_exp_sum_w;
logic                                                   one_over_exp_sum_vld;

//get fixed rc scale
logic   [`RECOMPUTE_SCALE_WIDTH-1:0]                    nxt_rc_scale;
logic   [sig_width+exp_width : 0]                       one_over_exp_sum_shift;
logic                                                   one_over_exp_sum_shift_vld;

/////////////////////////////////////////////////////
//      Control Signal Input Gating                //
//                                                 //
/////////////////////////////////////////////////////  
always_ff@(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        rc_cfg_reg <= 0;
    end
    else if(rc_cfg_vld)begin
        rc_cfg_reg <= rc_cfg;
    end
end

always_ff@ (posedge clk or negedge rst_n)begin
    if(~rst_n)
        usr_cfg_reg <= 0;
    else if(usr_cfg_vld)
        usr_cfg_reg <= usr_cfg;
end


always_ff@ (posedge clk or negedge rst_n)begin
    if(~rst_n)
        model_cfg_reg <= 0;
    else if(model_cfg_vld)
        model_cfg_reg <= model_cfg;
end

always_ff@ (posedge clk or negedge rst_n)begin
    if(~rst_n)
        control_state_reg <= IDLE_STATE;
    else if(control_state_update)
        control_state_reg <= control_state;
end

//gate the input, to avoid timing problem caused by long routing from gbus to vector engine
always_ff@ (posedge clk or negedge rst_n)begin
    if(~rst_n) begin
        gbus_data_reg_vld <= 0;
    end
    else if(control_state_reg == ATT_QK_STATE) begin
        gbus_data_reg_vld <= gbus_wen;
    end
end

always_ff@ (posedge clk or negedge rst_n)begin
    if(~rst_n) begin
        gbus_data_reg <= 0;
        gbus_addr_reg <= 0;
    end
    else if(gbus_wen & control_state_reg == ATT_QK_STATE)begin
        gbus_data_reg <= gbus_wdata;
        gbus_addr_reg <= gbus_addr;
    end
end

always_ff@ (posedge clk or negedge rst_n)begin
    if(~rst_n) begin
        in_bus_data_vld_reg <= 0;
    end
    else if(control_state_reg == ATT_PV_STATE) begin
        in_bus_data_vld_reg <= in_bus_data_vld;
    end
end
always_ff@ (posedge clk or negedge rst_n)begin
    if(~rst_n) begin
        in_bus_data_reg <= 0;
    end
    else if(in_bus_data_vld & control_state_reg == ATT_PV_STATE)begin
        in_bus_data_reg <= in_bus_data;
    end
end

/////////////////////////////////////////////////////
//      Get the max from QKt data output from gbus //
//                                                 //
/////////////////////////////////////////////////////

always_comb begin
    max_reg_w = max_reg;
    for(int i = 0; i < 4; i++)begin
        if(gbus_data_reg_vld) begin 
            if( (usr_cfg_reg.user_kv_cache_not_full && (gbus_addr_reg.cmem_addr[0+:$clog2(`HEAD_SRAM_DEPTH)] * `MAC_MULT_NUM + gbus_addr_reg.cmem_addr[$clog2(`HEAD_SRAM_DEPTH)+:$clog2(`MAC_MULT_NUM)] + i <= usr_cfg_reg.user_token_cnt))////kv cache not full
                ||(~usr_cfg_reg.user_kv_cache_not_full) //head sram write addr slice
            )
                max_reg_w = ($signed(gbus_data_reg[FIXED_DATA_WIDTH*i +: FIXED_DATA_WIDTH])>$signed(max_reg_w)) ?  gbus_data_reg[FIXED_DATA_WIDTH*i +: FIXED_DATA_WIDTH]: max_reg_w;
    end
    end
end

always_ff @(posedge clk or negedge rst_n) begin
    if(~rst_n) begin
        max_reg <={1'b1,{(FIXED_DATA_WIDTH-1){1'b0}}};
    end
    else if (clear) begin
        max_reg <={1'b1,{(FIXED_DATA_WIDTH-1){1'b0}}};
    end
    else if(gbus_data_reg_vld) begin
        max_reg <= max_reg_w;
    end
end

/////////////////////////////////////////////////////
//  Get the input from head sram                   //
//  and subtract by max                            //
/////////////////////////////////////////////////////
assign xi_data_split = in_bus_data_reg;

genvar i;
generate
    for(i=0;i<BUS_NUM;i++) begin
        always_ff @(posedge clk or negedge rst_n) begin
            if(~rst_n) begin
                xi_max[i] <= 0;
            end
            else if(in_bus_data_vld_reg) begin
                xi_max[i] <= $signed(xi_data_split[i])-$signed(max_reg);
            end
        end
    end
endgenerate

always_ff @(posedge clk or negedge rst_n) begin
    if(~rst_n) begin
        xi_max_vld <= 0;
    end
    else begin
        xi_max_vld <= in_bus_data_vld_reg;
    end
end

/////////////////////////////////////////////////////
//  Convert xi-max to fp                           //
//                                                 //
/////////////////////////////////////////////////////
assign softmax_input_dequant_scale_reg = rc_cfg_reg.softmax_input_dequant_scale;
assign i2flt_xi_max_dequant_vld_out = &i2flt_xi_max_dequant_vld;

generate
    for(i = 0;i < BUS_NUM;i++)begin : i2flt_xi_max_stage
        logic signed    [isize-1 : 0]               xi_max_ext;
        logic           [sig_width+exp_width : 0]   i2flt_xi_max_w, i2flt_xi_max;  
        assign xi_max_ext = $signed(xi_max[i]);

        DW_fp_i2flt #(sig_width, exp_width, isize, isign)
        i2flt_xi_max_inst ( 
            .a(xi_max_ext), 
            .rnd(3'b000), 
            .z(i2flt_xi_max_w), 
            .status() 
        );

        always_ff @(posedge clk or negedge rst_n) begin
            if(~rst_n) begin
                i2flt_xi_max <= 0;
            end
            else begin
                if(xi_max_vld) begin
                    i2flt_xi_max <= i2flt_xi_max_w;
                end
            end
        end

        fp_mult_pipe#(
            .sig_width(sig_width),
            .exp_width(exp_width),
            .ieee_compliance(ieee_compliance),//NaN and denormals
            .stages(4)
        )inst_fp_dequant( 
            .clk(clk),
            .rst_n(rst_n),
            .a(i2flt_xi_max),
            .b(softmax_input_dequant_scale_reg),
            .z(i2flt_xi_max_dequant[i]),
            .ab_valid(i2flt_xi_max_vld),
            .z_valid(i2flt_xi_max_dequant_vld[i]) //pulse
        );
    end

endgenerate

always_ff @(posedge clk or negedge rst_n) begin
    if(~rst_n) begin
        i2flt_xi_max_vld <= 0;
    end
    else begin
        i2flt_xi_max_vld <= xi_max_vld;
    end
end

/////////////////////////////////////////////////////
//      PV: Get exponent                           //
//                                                 //
/////////////////////////////////////////////////////
generate
    for(i = 0;i < BUS_NUM;i++)begin : exp_xi_max_stage
        DW_fp_exp #(sig_width, exp_width, ieee_compliance, exp_arch) 
        iexp_xi_max(
        .a(i2flt_xi_max_dequant[i]),
        .z(exp_xi_max_w[i]),
        .status() );

        always_ff @(posedge clk or negedge rst_n)begin
            if(~rst_n)begin
                exp_xi_max[i]<=0;
            end
            else if(i2flt_xi_max_dequant_vld_out) begin
                exp_xi_max[i]<=exp_xi_max_w[i];
            end
        end
    end
endgenerate

always_ff @(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        exp_xi_max_vld <= 0;
    end
    else begin
        exp_xi_max_vld<=i2flt_xi_max_dequant_vld_out;
    end
end

//retiming for exponent unit
generate
    for(i = 0;i < EXP_STAGE_DELAY;i++)begin : exp_xi_max_retiming_stage
        logic [BUS_NUM-1:0][sig_width+exp_width : 0] timing_register_float;
        logic [BUS_NUM-1:0]                          timing_register_vld;

        if(i == 0)begin
            always_ff@(posedge clk or negedge rst_n)begin
                if(~rst_n)begin
                    timing_register_float <= 0;
                    timing_register_vld <= 0;
                end
                else begin
                    timing_register_float <= exp_xi_max;
                    timing_register_vld <= exp_xi_max_vld;
                end
            end
        end
        else begin
            always_ff@(posedge clk or negedge rst_n)begin
                if(~rst_n) begin
                    timing_register_float <= 0;
                    timing_register_vld <= 0;  
                end
                else begin
                    timing_register_float <= exp_xi_max_retiming_stage[i-1].timing_register_float;
                    timing_register_vld <= exp_xi_max_retiming_stage[i-1].timing_register_vld;
                end
            end
        end
    end
endgenerate

always_comb begin
    exp_xi_max_delay = exp_xi_max_retiming_stage[EXP_STAGE_DELAY-1].timing_register_float;
    exp_xi_max_vld_delay = exp_xi_max_retiming_stage[EXP_STAGE_DELAY-1].timing_register_vld;
end

/////////////////////////////////////////////////////
//      PV: Quantization the exponent              //
//                                                 //
/////////////////////////////////////////////////////
assign  softmax_exp_quant_scale_reg = rc_cfg_reg.softmax_exp_quant_scale;
assign  exp_xi_max_scaled_vld_out = &exp_xi_max_scaled_vld;
generate
    for(i = 0;i < BUS_NUM;i++)begin : flt2i_quant_stage
        //scale the data
        logic   [sig_width+exp_width : 0]   exp_xi_max_scaled;  

        fp_mult_pipe#(
            .sig_width(sig_width),
            .exp_width(exp_width),
            .ieee_compliance(ieee_compliance),//NaN and denormals
            .stages(5)
        )inst_fp_quant( 
            .clk(clk),
            .rst_n(rst_n),
            .a(exp_xi_max_delay[i]),
            .b(softmax_exp_quant_scale_reg),
            .z(exp_xi_max_scaled),
            .ab_valid(exp_xi_max_vld_delay),
            .z_valid(exp_xi_max_scaled_vld[i]) //pulse
        );
        
        //flt2fixed
        logic   [isize-1:0] exp_xi_max_flt2i;
        DW_fp_flt2i #(sig_width, exp_width, isize, isign)
        exp_flt2i_inst ( 
                .a(exp_xi_max_scaled), 
                .rnd(3'b000), 
                .z(exp_xi_max_flt2i), 
                .status() 
        );

        //clip
        logic   signed   [FIXED_DATA_WIDTH-1:0]  nxt_out_fixed_data;
        always_comb begin
            nxt_out_fixed_data = exp_xi_max_flt2i;
            if($signed(exp_xi_max_flt2i) > 127) begin
                nxt_out_fixed_data = 8'b0111_1111; //127
            end
            else if($signed(exp_xi_max_flt2i) < -128) begin
                nxt_out_fixed_data = 8'b1000_0000; //-128
            end
        end
        
        //output
        always_ff @(posedge clk or negedge rst_n) begin
            if(~rst_n) begin
                out_bus_data[i*FIXED_DATA_WIDTH+:FIXED_DATA_WIDTH] <= 0;
            end
            else if(exp_xi_max_scaled_vld[i]) begin
                out_bus_data[i*FIXED_DATA_WIDTH+:FIXED_DATA_WIDTH] <= nxt_out_fixed_data;
            end
        end
    end
endgenerate

always_ff @(posedge clk or negedge rst_n) begin
    if(~rst_n) begin
        out_bus_data_vld <= 0;
    end
    else begin
        out_bus_data_vld <= exp_xi_max_scaled_vld_out;
    end
end

/////////////////////////////////////////////////////
//   Recompute: Get the exponent sum               //
//                                                 //
/////////////////////////////////////////////////////
always_comb begin
    nxt_exp_sum_cnt = exp_sum_cnt;
    nxt_fadd_tree_in = 0;
    nxt_fadd_tree_vld = 0;
    nxt_fadd_tree_in_last = 0;
    if(clear) begin
        nxt_exp_sum_cnt = 0;
    end
    else if(exp_xi_max_vld_delay) begin
        nxt_fadd_tree_in = exp_xi_max_delay;
        nxt_exp_sum_cnt = exp_sum_cnt;
        for(int m = 0; m<BUS_NUM ; m++) begin
            nxt_fadd_tree_vld[m] = 1'b1;
            //nxt_exp_sum_cnt = nxt_exp_sum_cnt + 1;
            nxt_exp_sum_cnt = exp_sum_cnt + m + 1;
            if(usr_cfg_reg.user_kv_cache_not_full) begin //when kv_cache is not full
                if(nxt_exp_sum_cnt == usr_cfg_reg.user_token_cnt + 1) begin
                    nxt_fadd_tree_in_last = 1'b1;
                    break;
                end
            end
            else begin
                if(nxt_exp_sum_cnt == model_cfg_reg.max_context_length) begin
                    nxt_fadd_tree_in_last = 1'b1;
                    break;
                end
            end
        end
    end
end

always_ff @(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        fadd_tree_in <= 0;
        fadd_tree_vld <= 0;
        exp_sum_cnt <= 0;
        fadd_tree_in_last <= 0;
    end
    else begin
        fadd_tree_in <= nxt_fadd_tree_in;
        fadd_tree_vld <= nxt_fadd_tree_vld;
        exp_sum_cnt <= nxt_exp_sum_cnt;
        fadd_tree_in_last <= nxt_fadd_tree_in_last;
    end
end

fadd_tree #(
    .sig_width(sig_width),
    .exp_width(exp_width),
    .MAC_NUM(BUS_NUM)
) fadd_tree_inst (
    .clk(clk),
    .rstn(rst_n),
    .idata(fadd_tree_in),
    .idata_valid(fadd_tree_vld),
    .last_in(fadd_tree_in_last),
    .odata(adder_out),
    .odata_valid(adder_out_vld),
    .last_out(adder_out_last)
);

DW_fp_add_DG #(sig_width, exp_width, ieee_compliance)
acc_inst ( .a(acc_out), .b(adder_out), .rnd(3'b000), .DG_ctrl(adder_out_vld), .z(acc_out_w),.status());

always_ff @(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        acc_out <= 0;
    end
    else if (clear) begin
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
//   Recompute: Get the 1 over exponent sum        //
//                                                 //
/////////////////////////////////////////////////////
DW_fp_recip #(sig_width, exp_width, ieee_compliance, 0) 
inst_recip (
    .a(acc_out),
    .rnd(3'b000),
    .z(one_over_exp_sum_w),
    .status() 
);

always_ff@(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        one_over_exp_sum_vld <= 0;
        one_over_exp_sum <= 0;
    end
    else begin
        one_over_exp_sum_vld <= acc_vld;
        if(acc_vld)
            one_over_exp_sum <= one_over_exp_sum_w;
    end
end

///////////////////////////////////////////
//Calculate One Over RMS to Integer Scale//
///////////////////////////////////////////
always_ff@(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        one_over_exp_sum_shift_vld <= 0;
        one_over_exp_sum_shift <= 0;
    end
    else begin
        one_over_exp_sum_shift_vld <= one_over_exp_sum_vld;
        if(one_over_exp_sum_vld) begin
            one_over_exp_sum_shift[sig_width+exp_width]      <= one_over_exp_sum[sig_width+exp_width];
            one_over_exp_sum_shift[sig_width+exp_width-1: 0] <= one_over_exp_sum[sig_width+exp_width-1: 0] + $unsigned({rc_cfg_reg.softmax_rc_shift,{sig_width{1'b0}}});
        end 
    end
end

DW_fp_flt2i #(sig_width, exp_width, `RECOMPUTE_SCALE_WIDTH, isign)
    i2flt_in_data ( 
        .a(one_over_exp_sum_shift), 
        .rnd(3'b000), 
        .z(nxt_rc_scale), 
        .status() 
);

always_ff @(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        rc_scale <= 0;
        rc_scale_vld <= 0;
    end
    else begin
        rc_scale_vld <= one_over_exp_sum_shift_vld;
        if(one_over_exp_sum_shift_vld)
            rc_scale <= nxt_rc_scale;
    end
end

endmodule
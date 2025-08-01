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

module head_top #(
    parameter HEAD_INDEX = 0,
    parameter GLOBAL_SRAM_DATA_BIT = (`MAC_MULT_NUM * `IDATA_WIDTH),

    parameter HEAD_CORE_NUM = `HEAD_CORE_NUM,
    parameter GBUS_ADDR_WIDTH = `GBUS_ADDR_WIDTH,
    parameter GBUS_DATA_WIDTH = `GBUS_DATA_WIDTH,

    parameter HLINK_DATA_WIDTH = (`MAC_MULT_NUM * `IDATA_WIDTH),
    parameter VLINK_DATA_WIDTH  = (`MAC_MULT_NUM * `IDATA_WIDTH),
    parameter CDATA_ACCU_NUM_WIDTH = `CDATA_ACCU_NUM_WIDTH,
    parameter CDATA_SCALE_WIDTH = `CDATA_SCALE_WIDTH,
    parameter CDATA_BIAS_WIDTH = `CDATA_BIAS_WIDTH,
    parameter CDATA_SHIFT_WIDTH = `CDATA_SHIFT_WIDTH
)(
    input  logic                                                     clk,
    input  logic                                                     rst_n,

    input  logic                                                     clean_kv_cache, //pulse
    input  logic       [`USER_ID_WIDTH-1:0]                          clean_kv_cache_user_id, //clean the corrsponding user's kv cache to zero

    input  logic           [`HEAD_MEM_ADDR_WIDTH-1:0]                head_mem_addr,
    input  logic           [`INTERFACE_DATA_WIDTH-1:0]               head_mem_wdata,
    input  logic                                                     head_mem_wen,
    output logic           [`INTERFACE_DATA_WIDTH-1:0]               head_mem_rdata,
    input  logic                                                     head_mem_ren,
    output logic                                                     head_mem_rvld,
 
    input  logic         [GLOBAL_SRAM_DATA_BIT-1:0]                  global_sram_rd_data,
    input  logic                                                     global_sram_rd_data_vld,
    
    input  logic         [HEAD_CORE_NUM-1:0][VLINK_DATA_WIDTH-1:0]   vlink_data_in_array,
    input  logic         [HEAD_CORE_NUM-1:0]                         vlink_data_in_vld_array,
    output logic         [HEAD_CORE_NUM-1:0][VLINK_DATA_WIDTH-1:0]   vlink_data_out_array,
    output logic         [HEAD_CORE_NUM-1:0]                         vlink_data_out_vld_array,

    input  CONTROL_STATE                                             control_state,
    input  logic                                                     control_state_update,
                                                                    //start信号具有和control state一样的寄存器delay
    input  logic                                                     start, //start indicate the current opearation starts
    output logic                                                     finish, //finish indicate the current opeatation ends for this ctrl    

    input  logic                                                     op_cfg_vld,      //和control_state的来自core_array的寄存器delay一样
    input  OP_CONFIG                                                 op_cfg,

    input                                                            usr_cfg_vld,
    input  USER_CONFIG                                               usr_cfg,

    input                                                            model_cfg_vld,
    input  MODEL_CONFIG                                              model_cfg,

    input  logic                                                     pmu_cfg_vld,
    input  PMU_CONFIG                                                pmu_cfg, 

    input  logic                                                     rc_cfg_vld,
    input  RC_CONFIG                                                 rc_cfg,//和model config一样在网络初始化的时候配置好就行   

    output BUS_ADDR                                                  gbus_addr_delay1, //latency consideration
    output logic                                                     gbus_wen_delay1,
    output logic                 [GBUS_DATA_WIDTH-1:0]               gbus_wdata_delay1                   
);

BUS_ADDR                                                             gbus_addr;
logic                                                                gbus_wen;
logic                 [GBUS_DATA_WIDTH-1:0]                          gbus_wdata;         

always_ff@(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        gbus_addr_delay1 <= 0;
        gbus_wen_delay1 <= 0;
        gbus_wdata_delay1 <= 0;
    end
    else begin
        gbus_addr_delay1 <= gbus_addr;
        gbus_wen_delay1 <= gbus_wen;
        gbus_wdata_delay1 <= gbus_wdata;
    end
end

logic                                                clean_kv_cache_delay1; //pulse
logic       [`USER_ID_WIDTH-1:0]                     clean_kv_cache_user_id_delay1; //clean the corrsponding user's kv cache to zero

always_ff@(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        clean_kv_cache_delay1 <= 0;
        clean_kv_cache_user_id_delay1 <= 0;
    end
    else begin
        clean_kv_cache_delay1 <= clean_kv_cache;
        clean_kv_cache_user_id_delay1 <= clean_kv_cache_user_id;
    end
end


logic         [(`MAC_MULT_NUM * `IDATA_WIDTH)-1:0]                   abuf_in_data;
logic                                                                abuf_in_data_vld;
logic         [(`MAC_MULT_NUM * `IDATA_WIDTH)-1:0]                   abuf_out_data;
logic                                                                abuf_out_data_vld;

BUS_ADDR      [HEAD_CORE_NUM-1:0]                                    hca_out_gbus_addr_array;  
logic         [HEAD_CORE_NUM-1:0]                                    hca_out_gbus_wen_array;
logic         [HEAD_CORE_NUM-1:0][GBUS_DATA_WIDTH-1:0]               hca_out_gbus_wdata_array;

                                                                    //cfg信号具有和control state一样的寄存器delay
logic                                                                op_cfg_vld_reg;      //和control_state的来自core_array的寄存器delay一样
OP_CONFIG                                                            op_cfg_reg;

logic                                                                usr_cfg_vld_reg;
USER_CONFIG                                                          usr_cfg_reg;

logic                                                                model_cfg_vld_reg;
MODEL_CONFIG                                                         model_cfg_reg;

CONTROL_STATE                                                        control_state_reg;
logic                                                                control_state_update_reg;

logic                                                                rc_cfg_vld_reg;
RC_CONFIG                                                            rc_cfg_reg;

logic                                                                pmu_cfg_vld_reg;
PMU_CONFIG                                                           pmu_cfg_reg;

logic                                                                start_reg;
logic [$clog2(`MAC_MULT_NUM) + $clog2(`HEAD_SRAM_DEPTH)-1:0]         head_sram_waddr;
logic                                                                head_sram_wen;
logic         [(`MAC_MULT_NUM * `IDATA_WIDTH)-1:0]                   head_sram_wdata;

logic [$clog2(`MAC_MULT_NUM) + $clog2(`HEAD_SRAM_DEPTH)-1:0]         head_sram_raddr;           
logic                                                                head_sram_ren;
logic         [(`MAC_MULT_NUM * `IDATA_WIDTH)-1:0]                   head_sram_rdata;
logic                                                                head_sram_rvld;

//finish logic
logic hca_finish;
logic nxt_hca_finish_flag;
logic hca_finish_flag;

logic head_sram_finish;
logic nxt_head_sram_finish_flag;
logic head_sram_finish_flag;

logic nxt_finish;

always_comb begin
    nxt_finish = 0;
    nxt_hca_finish_flag = hca_finish_flag;
    nxt_head_sram_finish_flag = head_sram_finish_flag;

    if(finish)begin
        nxt_finish = 0;
        nxt_hca_finish_flag = 0;
        nxt_head_sram_finish_flag = 0;
    end
    else begin
        if(control_state_reg == ATT_QK_STATE || control_state_reg == ATT_PV_STATE || control_state_reg == FFN0_STATE || control_state_reg == PROJ_STATE || control_state_reg == FFN1_STATE)begin
            if(hca_finish_flag && head_sram_finish_flag && ~gbus_wen_delay1)begin //wait for gbus not busy
                nxt_finish = 1;
            end
        end
        else begin
            if(hca_finish_flag && head_sram_finish_flag)begin
                nxt_finish = 1;
            end
        end

        if(hca_finish)begin
            nxt_hca_finish_flag = 1;
        end

        if(head_sram_finish)begin
            nxt_head_sram_finish_flag = 1;
        end
    end
end

always_ff @(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        finish <= 0;
        head_sram_finish_flag <= 0;
        hca_finish_flag <= 0;
    end
    else begin
        finish <= nxt_finish;
        head_sram_finish_flag <= nxt_head_sram_finish_flag;
        hca_finish_flag <= nxt_hca_finish_flag;
    end
end

logic        [(`MAC_MULT_NUM * `IDATA_WIDTH)-1:0]     softmax_rc_out_bus_data;
logic                                                     softmax_rc_out_bus_data_vld;


////////////////
//INST of ABUF//
////////////////

always_comb begin
    abuf_in_data = 0;
    abuf_in_data_vld = 0;
    if(global_sram_rd_data_vld) begin
        abuf_in_data_vld = 1;
        abuf_in_data = global_sram_rd_data;
    end
    else if(control_state_reg == ATT_PV_STATE && softmax_rc_out_bus_data_vld)begin
        abuf_in_data_vld = 1;
        abuf_in_data = softmax_rc_out_bus_data;
    end
    else if(head_sram_rvld && control_state_reg != ATT_PV_STATE)begin
        abuf_in_data_vld = 1;
        abuf_in_data = head_sram_rdata;
    end
end

abuf inst_abuf(
    .clk(clk),
    .rst_n(rst_n),

    .start(start_reg),

    .in_data(abuf_in_data),
    .in_data_vld(abuf_in_data_vld),

    .control_state(control_state_reg),
    .control_state_update(control_state_update_reg),

    .model_cfg_vld(model_cfg_vld_reg),
    .model_cfg(model_cfg_reg),

    .usr_cfg_vld(usr_cfg_vld_reg),
    .usr_cfg(usr_cfg_reg),

    .out_data(abuf_out_data),
    .out_data_vld(abuf_out_data_vld),

    .finish_row()
);

/////////////////////
//INST of head sram//
/////////////////////
logic [1:0] wdata_byte_flag;



always_comb begin
    wdata_byte_flag = 1; 
    head_sram_waddr = gbus_addr.cmem_addr;
    head_sram_wen = 0;
    head_sram_wdata = gbus_wdata;
    if(control_state_reg == ATT_QK_STATE)begin
        wdata_byte_flag = 2;
    end
    if(gbus_wen && gbus_addr.hs_bias == 1)begin
        head_sram_wen = 1;
    end
    if(control_state_reg == FFN0_STATE)begin
        if(gbus_wdata[`IDATA_WIDTH-1])// RELU (这时只会写一个byte)
            head_sram_wdata = 0;
    end
end
always_ff @(posedge clk or negedge rst_n)begin
    if(~rst_n)
        head_sram_rvld <= 0;
    else if(head_sram_ren)
        head_sram_rvld <= 1;
    else
        head_sram_rvld <= 0;
end
logic  [$clog2(`HEAD_SRAM_DEPTH) + $clog2(`MAC_MULT_NUM/2)-1:0] hs_interface_addr;

logic                                                           hs_interface_ren;
logic                                                           hs_interface_rvalid;
logic  [`INTERFACE_DATA_WIDTH-1:0]                              hs_interface_rdata;
    
logic                                                           hs_interface_wen;
logic  [`INTERFACE_DATA_WIDTH-1:0]                              hs_interface_wdata;

logic           [`INTERFACE_DATA_WIDTH-1:0]                     core_array_mem_rdata;
logic                                                           core_array_mem_rvld;

always_comb begin
    head_mem_rvld = 0;
    head_mem_rdata = 0;
    if(hs_interface_rvalid) begin
        head_mem_rvld = 1;
        head_mem_rdata = hs_interface_rdata;
    end
    else if(core_array_mem_rvld)begin
        head_mem_rvld = 1;
        head_mem_rdata = core_array_mem_rdata;
    end
end

always_comb begin
    hs_interface_addr = head_mem_addr[$clog2(`HEAD_SRAM_DEPTH) + $clog2(`MAC_MULT_NUM/2)-1:0];
    hs_interface_ren = head_mem_ren && head_mem_addr[`HEAD_MEM_ADDR_WIDTH-1];
    hs_interface_wen = head_mem_wen && head_mem_addr[`HEAD_MEM_ADDR_WIDTH-1];
    hs_interface_wdata = head_mem_wdata;
end
head_sram inst_head_sram (
    .clk(clk),
    .rstn(rst_n),
    .interface_addr(hs_interface_addr),

    .interface_ren(hs_interface_ren),
    .interface_rvalid(hs_interface_rvalid),
    .interface_rdata(hs_interface_rdata),
    
    .interface_wen(hs_interface_wen),
    .interface_wdata(hs_interface_wdata),

    .waddr(head_sram_waddr),
    .wen(head_sram_wen),
    .wdata(head_sram_wdata),
    .wdata_byte_flag(wdata_byte_flag),
    .raddr(head_sram_raddr),
    .ren(head_sram_ren),
    .rdata(head_sram_rdata)
);

/////////////////////////////
//INST of head sram rd ctrl//
/////////////////////////////
head_sram_rd_ctrl inst_head_sram_rd_ctrl(
    .clk(clk),
    .rst_n(rst_n),
        
    .control_state(control_state_reg),
    .control_state_update(control_state_update_reg),

    .model_cfg_vld(model_cfg_vld_reg),
    .model_cfg(model_cfg_reg),

    .usr_cfg_vld(usr_cfg_vld_reg),
    .usr_cfg(usr_cfg_reg),

    .start(start_reg),
    .finish(head_sram_finish),

    .head_sram_ren(head_sram_ren),
    .head_sram_raddr(head_sram_raddr)
);




///////////////////////////
//INST of head_core_array//
///////////////////////////
//hca: head_core_array

always_ff @(posedge clk or negedge rst_n)begin //control signals retiming
    if(~rst_n)begin
        op_cfg_vld_reg <= 0;
        op_cfg_reg <= 0;
    end
    else if(op_cfg_vld)begin
        op_cfg_vld_reg <= 1;
        op_cfg_reg <= op_cfg;
    end
    else begin
        op_cfg_vld_reg <= 0;
    end
end

always_ff @(posedge clk or negedge rst_n)begin //control signals retiming
    if(~rst_n)begin
        usr_cfg_vld_reg <= 0;
        usr_cfg_reg <= 0;
    end
    else if(usr_cfg_vld)begin
        usr_cfg_vld_reg <= 1;
        usr_cfg_reg <= usr_cfg;
    end
    else begin
        usr_cfg_vld_reg <= 0;
    end
end

always_ff @(posedge clk or negedge rst_n)begin //control signals retiming
    if(~rst_n)begin
        model_cfg_vld_reg <= 0;
        model_cfg_reg <= 0;
    end
    else if(model_cfg_vld)begin
        model_cfg_vld_reg <= 1;
        model_cfg_reg <= model_cfg;
    end
    else begin
        model_cfg_vld_reg <= 0;
    end
end


always_ff @(posedge clk or negedge rst_n)begin //control signals retiming
    if(~rst_n)begin
        rc_cfg_vld_reg <= 0;
        rc_cfg_reg <= 0;
    end
    else if(rc_cfg_vld)begin
        rc_cfg_vld_reg <= 1;
        rc_cfg_reg <= rc_cfg;
    end
    else begin
        rc_cfg_vld_reg <= 0;
    end
end

always_ff @(posedge clk or negedge rst_n)begin //control signals retiming
    if(~rst_n)begin
        pmu_cfg_vld_reg <= 0;
        pmu_cfg_reg <= 0;
    end
    else if(pmu_cfg_vld)begin
        pmu_cfg_vld_reg <= 1;
        pmu_cfg_reg <= pmu_cfg;
    end
    else begin
        pmu_cfg_vld_reg <= 0;
    end
end

always_ff @(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        control_state_reg <= IDLE_STATE;
        control_state_update_reg <= 0;
    end
    else if(control_state_update) begin
        control_state_update_reg <= 1;
        control_state_reg <=  control_state;
    end
    else begin
        control_state_update_reg <= 0;
    end
end

always_ff @(posedge clk or negedge rst_n)begin
    if(~rst_n)
        start_reg <= 0;
    else if(start)
        start_reg <= 1;
    else
        start_reg <= 0;
end

logic        [`RECOMPUTE_SCALE_WIDTH-1:0]                 rc_scale;
logic                                                     rc_scale_vld;

logic        [`RECOMPUTE_SCALE_WIDTH-1:0]                 krms_rc_scale;
logic                                                     krms_rc_scale_vld;

logic        [`RECOMPUTE_SCALE_WIDTH-1:0]                 softmax_rc_scale;
logic                                                     softmax_rc_scale_vld;

always_comb begin
    rc_scale_vld = 0;
    rc_scale = 0;
    if(krms_rc_scale_vld) begin
        rc_scale_vld = 1;
        rc_scale = krms_rc_scale;
    end
    else if(softmax_rc_scale_vld)begin
        rc_scale = softmax_rc_scale;
        rc_scale_vld = 1;
    end
end


head_core_array#(
    .HEAD_CORE_NUM(HEAD_CORE_NUM),
    .GBUS_ADDR_WIDTH(GBUS_ADDR_WIDTH),
    .GBUS_DATA_WIDTH(GBUS_DATA_WIDTH),
    .HLINK_DATA_WIDTH(HLINK_DATA_WIDTH),
    .HEAD_INDEX(HEAD_INDEX)
) inst_head_core_array
(
    .clk(clk),
    .rst_n(rst_n),

    .clean_kv_cache(clean_kv_cache_delay1),
    .clean_kv_cache_user_id(clean_kv_cache_user_id_delay1),

    .core_array_mem_addr(head_mem_addr),
    .core_array_mem_wdata(head_mem_wdata),
    .core_array_mem_wen(head_mem_wen),
    .core_array_mem_rdata(core_array_mem_rdata),
    .core_array_mem_ren(head_mem_ren),
    .core_array_mem_rvld(core_array_mem_rvld),
 
    //head core array 会有一个gbus write inport，用于更新权重（来自外部mem)，或者kv_cache
    .in_gbus_addr(gbus_addr),  
    .in_gbus_wen(gbus_wen),
    .in_gbus_wdata(gbus_wdata),

    //head core array 会有CORE_NUM个gbus write outport， 用于输出到gbus的arbiter
    .out_gbus_addr_array(hca_out_gbus_addr_array),  
    .out_gbus_wen_array(hca_out_gbus_wen_array),
    .out_gbus_wdata_array(hca_out_gbus_wdata_array),

    //head core array 会有一个 hlink port， 用与接受来自abuf的输入
    .hlink_wdata(abuf_out_data),
    .hlink_wen(abuf_out_data_vld),

    .vlink_data_in_array(vlink_data_in_array),
    .vlink_data_in_vld_array(vlink_data_in_vld_array),
    .vlink_data_out_array(vlink_data_out_array),
    .vlink_data_out_vld_array(vlink_data_out_vld_array),
    
    //recompute logic port
    .rc_scale(rc_scale),
    .rc_scale_vld(rc_scale_vld),
    .rc_scale_clear(finish),

    // operation Config Signals
    .op_cfg_vld(op_cfg_vld_reg),   
    .op_cfg(op_cfg_reg),

    // pmu config signals
    .pmu_cfg_vld(pmu_cfg_vld_reg),   
    .pmu_cfg(pmu_cfg_reg),

    // model Config Signals
    .model_cfg_vld(model_cfg_vld_reg),   
    .model_cfg(model_cfg_reg),

    //recompute config
    .rc_cfg_vld(rc_cfg_vld_reg),
    .rc_cfg(rc_cfg_reg),

    // user Config Signals
    .usr_cfg_vld(usr_cfg_vld_reg),
    .usr_cfg(usr_cfg_reg),

    .control_state(control_state_reg),
    .control_state_update(control_state_update_reg),
    
    .start(start_reg),
    .finish(hca_finish)
); 


////////////////
//INST of GBUS//
////////////////

    
gbus_top #(
    .HEAD_CORE_NUM(HEAD_CORE_NUM)
)inst_gbus_top
(
    .clk(clk),
    .rst_n(rst_n),

    .hca_gbus_addr_array(hca_out_gbus_addr_array),  
    .hca_gbus_wen_array(hca_out_gbus_wen_array),
    .hca_gbus_wdata_array(hca_out_gbus_wdata_array),

    .gbus_addr(gbus_addr),  
    .gbus_wen(gbus_wen),
    .gbus_wdata(gbus_wdata)
);

////////////////
//INST of KRMS//
////////////////

logic        [`MAC_MULT_NUM-1:0][`IDATA_WIDTH-1:0]        nxt_krms_in_fixed_data;
logic                                                     nxt_krms_in_fixed_data_vld;

logic        [`MAC_MULT_NUM-1:0][`IDATA_WIDTH-1:0]        krms_in_fixed_data;
logic                                                     krms_in_fixed_data_vld;
always_comb begin
    nxt_krms_in_fixed_data = 0;
    nxt_krms_in_fixed_data_vld = 0;
    if(control_state_reg == Q_GEN_STATE ||
       control_state_reg == K_GEN_STATE ||
       control_state_reg == V_GEN_STATE ||
       control_state_reg == FFN0_STATE)begin
        nxt_krms_in_fixed_data = global_sram_rd_data;
        nxt_krms_in_fixed_data_vld = global_sram_rd_data_vld;
    end
end

always_ff@(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        krms_in_fixed_data <= 0;
        krms_in_fixed_data_vld <= 0;
    end
    else begin
        krms_in_fixed_data <= nxt_krms_in_fixed_data;
        krms_in_fixed_data_vld <= nxt_krms_in_fixed_data_vld;
    end
end



krms #(
    .BUS_NUM(`MAC_MULT_NUM), //input bus num
    .DATA_NUM_WIDTH($clog2(`MAX_EMBD_SIZE)+1), //layernorm vector length reg width
    .FIXED_SQUARE_SUM_WIDTH(`FIXED_SQUARE_SUM_WIDTH)
)inst_krms(
    .clk(clk),
    .rst_n(rst_n),

    .start(start_reg),
    .rc_cfg_vld(rc_cfg_vld_reg),
    .rc_cfg(rc_cfg_reg),     

    .control_state(control_state_reg),
    .control_state_update(control_state_update_reg),

    .in_fixed_data(krms_in_fixed_data), 
    .in_fixed_data_vld(krms_in_fixed_data_vld),
    
    .rc_scale(krms_rc_scale),
    .rc_scale_vld(krms_rc_scale_vld)
);

//////////////////////
//INST of Softmax rc//
//////////////////////
logic softmax_clear;
assign softmax_clear = finish && control_state_reg == ATT_PV_STATE;




softmax_rc #(
    .GBUS_DATA_WIDTH(`GBUS_DATA_WIDTH),
    .BUS_NUM(`MAC_MULT_NUM),
    .FIXED_DATA_WIDTH(`IDATA_WIDTH),
    .EXP_STAGE_DELAY(`SOFTMAX_RETIMING_NUM)         // Ensure parameters are set correctly
) softmax_rc_inst (
    .clk(clk),
    .rst_n(rst_n),
    
    .control_state(control_state_reg),
    .control_state_update(control_state_update_reg),

    .rc_cfg_vld(rc_cfg_vld_reg),
    .rc_cfg(rc_cfg_reg),

    .model_cfg_vld(model_cfg_vld_reg),
    .model_cfg(model_cfg_reg),

    .usr_cfg(usr_cfg_reg),
    .usr_cfg_vld(usr_cfg_vld_reg),

    .in_bus_data(head_sram_rdata),
    .in_bus_data_vld(head_sram_rvld),

    .gbus_wen(gbus_wen),
    .gbus_wdata(gbus_wdata),
    .gbus_addr(gbus_addr),

    .clear(softmax_clear),

    .rc_scale(softmax_rc_scale),
    .rc_scale_vld(softmax_rc_scale_vld),

    .out_bus_data(softmax_rc_out_bus_data),
    .out_bus_data_vld(softmax_rc_out_bus_data_vld)
);

endmodule
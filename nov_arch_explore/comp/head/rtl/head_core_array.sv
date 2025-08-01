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

module head_core_array#(
    parameter HEAD_CORE_NUM = `HEAD_CORE_NUM,
    parameter GBUS_ADDR_WIDTH = `GBUS_ADDR_WIDTH,
    parameter GBUS_DATA_WIDTH = `GBUS_DATA_WIDTH,
    parameter HLINK_DATA_WIDTH = (`MAC_MULT_NUM * `IDATA_WIDTH),
    parameter VLINK_DATA_WIDTH  = (`MAC_MULT_NUM * `IDATA_WIDTH),
    parameter CDATA_ACCU_NUM_WIDTH = `CDATA_ACCU_NUM_WIDTH,
    parameter CDATA_SCALE_WIDTH = `CDATA_SCALE_WIDTH,
    parameter CDATA_BIAS_WIDTH = `CDATA_BIAS_WIDTH,
    parameter CDATA_SHIFT_WIDTH = `CDATA_SHIFT_WIDTH,
    parameter HEAD_INDEX = 0
)
(
    input  logic                                                     clk,
    input  logic                                                     rst_n,

    input  logic                                                     clean_kv_cache, //pulse
    input  logic       [`USER_ID_WIDTH-1:0]                          clean_kv_cache_user_id, //clean the corrsponding user's kv cache to zero

    input  logic           [`HEAD_MEM_ADDR_WIDTH-1:0]                core_array_mem_addr,
    input  logic           [`INTERFACE_DATA_WIDTH-1:0]               core_array_mem_wdata,
    input  logic                                                     core_array_mem_wen,
    output logic           [`INTERFACE_DATA_WIDTH-1:0]               core_array_mem_rdata,
    input  logic                                                     core_array_mem_ren,
    output logic                                                     core_array_mem_rvld,
 
    //head core array has "gbus write inport"，to update WMEM and KV_CACHE
    input  BUS_ADDR                                                  in_gbus_addr,  
    input  logic                                                     in_gbus_wen,
    input  logic                            [GBUS_DATA_WIDTH-1:0]    in_gbus_wdata,


    //head core array has CORE_NUM number of "gbus write outport"， -> arbiter
    output BUS_ADDR      [HEAD_CORE_NUM-1:0]                         out_gbus_addr_array,  
    output logic         [HEAD_CORE_NUM-1:0]                         out_gbus_wen_array,
    output logic         [HEAD_CORE_NUM-1:0][GBUS_DATA_WIDTH-1:0]    out_gbus_wdata_array,

    //head core array , <- abuf
    input  logic         [HLINK_DATA_WIDTH-1:0]                      hlink_wdata,
    input  logic                                                     hlink_wen,

    input  logic         [HEAD_CORE_NUM-1:0][VLINK_DATA_WIDTH-1:0]   vlink_data_in_array,
    input  logic         [HEAD_CORE_NUM-1:0]                         vlink_data_in_vld_array,
    output logic         [HEAD_CORE_NUM-1:0][VLINK_DATA_WIDTH-1:0]   vlink_data_out_array,
    output logic         [HEAD_CORE_NUM-1:0]                         vlink_data_out_vld_array,

    //recompute logic port
    input  logic         [`RECOMPUTE_SCALE_WIDTH - 1:0]              rc_scale,
    input  logic                                                     rc_scale_vld,
    input  logic                                                     rc_scale_clear,

    // operation Config Signals
    input  logic                                                     op_cfg_vld,      //和control_state的来自core_array的寄存器delay一样
    input  OP_CONFIG                                                 op_cfg,
    input  logic                                                     usr_cfg_vld,
    input  USER_CONFIG                                               usr_cfg,
    input  logic                                                     model_cfg_vld,
    input  MODEL_CONFIG                                              model_cfg,
    input  logic                                                     rc_cfg_vld,
    input  RC_CONFIG                                                 rc_cfg,//和model config一样在网络初始化的时候配置好就行   
    input  logic                                                     pmu_cfg_vld,
    input  PMU_CONFIG                                                pmu_cfg,  

    input  CONTROL_STATE                                             control_state,
    input  logic                                                     control_state_update,

    input  logic                                                     start, //start indicate the current opearation starts
    output logic                                                     finish //finish indicate the current opeatation ends for this ctrl    
); 


genvar i; 




/////////////////
//hlink signals//
/////////////////
logic [HEAD_CORE_NUM-1:0][HLINK_DATA_WIDTH-1:0] hlink_wdata_array;
logic [HEAD_CORE_NUM-1:0]                       hlink_wen_array;
logic [HEAD_CORE_NUM-1:0][HLINK_DATA_WIDTH-1:0] hlink_rdata_array;
logic [HEAD_CORE_NUM-1:0]                       hlink_rvalid_array;

always_comb begin
    for(int i = 0; i < HEAD_CORE_NUM; i++)begin
        if(i == 0)begin
            hlink_wdata_array[i] = hlink_wdata;
            hlink_wen_array[i] = hlink_wen;
        end
        else begin
            hlink_wdata_array[i] = hlink_rdata_array[i-1];
            hlink_wen_array[i] = hlink_rvalid_array[i-1];
        end
    end
end

/////////////////////
//Core finish array//
/////////////////////
logic [HEAD_CORE_NUM-1:0] core_finish_array;
logic [HEAD_CORE_NUM-1:0] core_finish_array_flag;

generate
    for(i = 0; i < HEAD_CORE_NUM ;i++)begin: core_finish_gen_array
        always_ff@ (posedge clk or negedge rst_n)begin
            if(~rst_n)begin
                core_finish_array_flag[i] <= 0;
            end
            else if(finish)begin
                core_finish_array_flag[i] <= 0;
            end
            else if(core_finish_array[i])begin
                core_finish_array_flag[i] <= 1;
            end
        end
    end
endgenerate

always_ff@ (posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        finish <= 0;
    end
    else if(finish)begin
        finish <= 0;
    end
    else if(&core_finish_array_flag)begin
        finish <= 1;
    end
end

///////////////////////////////////////////
//              INTERFACE                //
///////////////////////////////////////////


logic [`HEAD_CORE_NUM-1:0][`CORE_MEM_ADDR_WIDTH-1:0]                nxt_core_mem_addr_array;
logic [`HEAD_CORE_NUM-1:0][`INTERFACE_DATA_WIDTH-1:0]               nxt_core_mem_wdata_array;
logic [`HEAD_CORE_NUM-1:0]                                          nxt_core_mem_wen_array;

logic [`HEAD_CORE_NUM-1:0]                                          nxt_core_mem_ren_array;


logic [`HEAD_CORE_NUM-1:0][`CORE_MEM_ADDR_WIDTH-1:0]                core_mem_addr_array;
logic [`HEAD_CORE_NUM-1:0][`INTERFACE_DATA_WIDTH-1:0]               core_mem_wdata_array;
logic [`HEAD_CORE_NUM-1:0]                                          core_mem_wen_array;
logic [`HEAD_CORE_NUM-1:0][`INTERFACE_DATA_WIDTH-1:0]               core_mem_rdata_array;
logic [`HEAD_CORE_NUM-1:0]                                          core_mem_ren_array;
logic [`HEAD_CORE_NUM-1:0]                                          core_mem_rvld_array;


always_comb begin
    nxt_core_mem_addr_array = core_mem_addr_array;
    nxt_core_mem_wdata_array = core_mem_wdata_array;
    nxt_core_mem_wen_array = 0;
    nxt_core_mem_ren_array = 0;
    for(int i = 0; i < `HEAD_CORE_NUM; i++)begin
        if(~core_array_mem_addr[`HEAD_MEM_ADDR_WIDTH-1])begin //写core mem的
            if(i == core_array_mem_addr[17 : 14])begin
                nxt_core_mem_addr_array[i] = core_array_mem_addr[13:0];
                nxt_core_mem_wdata_array[i] = core_array_mem_wdata;
                nxt_core_mem_wen_array[i] =   core_array_mem_wen;
                nxt_core_mem_ren_array[i] =   core_array_mem_ren;               
            end
        end
    end
end

always_ff @(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
    core_mem_addr_array <= 0;
    core_mem_wdata_array <= 0;
    core_mem_wen_array <= 0;
    core_mem_ren_array <= 0;
    end
    else begin
        core_mem_addr_array <= nxt_core_mem_addr_array;
        core_mem_wdata_array <= nxt_core_mem_wdata_array;
        core_mem_wen_array <= nxt_core_mem_wen_array;
        core_mem_ren_array <= nxt_core_mem_ren_array;
    end
end



logic           [`INTERFACE_DATA_WIDTH-1:0]               nxt_core_array_mem_rdata;
logic                                                     nxt_core_array_mem_rvld;

always_ff @(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        core_array_mem_rvld <= 0;
        core_array_mem_rdata <= 0;
    end
    else begin
        core_array_mem_rvld <= nxt_core_array_mem_rvld;
        core_array_mem_rdata <= nxt_core_array_mem_rdata;
    end
end

always_comb begin
    nxt_core_array_mem_rvld = 0;
    nxt_core_array_mem_rdata = 0;
    for(int i = 0; i < `HEAD_CORE_NUM; i++)begin //TODO,这里有优先级，最后如果有set up问题改为unique case
        if(core_mem_rvld_array[i])begin
            nxt_core_array_mem_rvld = 1;
            nxt_core_array_mem_rdata = core_mem_rdata_array[i];
        end
    end
end


//////////////////////
//Inst of core array//
//////////////////////
generate
    for(i = 0; i < HEAD_CORE_NUM; i++)begin : head_cores_generate_array
        core_top#(.CORE_INDEX(i),
                  .HEAD_INDEX(HEAD_INDEX)) 
        core_top_inst(
            // Global Signals
            .clk(clk),
            .rstn(rst_n),

            .clean_kv_cache(clean_kv_cache),
            .clean_kv_cache_user_id(clean_kv_cache_user_id),

            .core_mem_addr(core_mem_addr_array[i]),
            .core_mem_wdata(core_mem_wdata_array[i]),
            .core_mem_wen(core_mem_wen_array[i]),
            .core_mem_rdata(core_mem_rdata_array[i]),
            .core_mem_ren(core_mem_ren_array[i]),
            .core_mem_rvld(core_mem_rvld_array[i]),

            // Global Config Signals
            .op_cfg_vld(op_cfg_vld),
            .op_cfg(op_cfg),
            .model_cfg_vld(model_cfg_vld),   
            .model_cfg(model_cfg),
            .usr_cfg_vld(usr_cfg_vld),
            .usr_cfg(usr_cfg),
            .rc_cfg_vld(rc_cfg_vld),
            .rc_cfg(rc_cfg),
            .pmu_cfg_vld(pmu_cfg_vld),
            .pmu_cfg(pmu_cfg),

            //GBUS inport
            .in_gbus_addr(in_gbus_addr),     
            .in_gbus_wen(in_gbus_wen),
            .in_gbus_wdata(in_gbus_wdata),

            //GBUS outport
            .out_gbus_addr(out_gbus_addr_array[i]),     
            .out_gbus_wen(out_gbus_wen_array[i]),
            .out_gbus_wdata(out_gbus_wdata_array[i]),

            //recompute logic port
            .rc_scale(rc_scale),
            .rc_scale_vld(rc_scale_vld),
            .rc_scale_clear(rc_scale_clear),

            // Channel - Core-to-Core Link
            .vlink_data_in(vlink_data_in_array[i]),
            .vlink_data_in_vld(vlink_data_in_vld_array[i]),
            .vlink_data_out(vlink_data_out_array[i]),
            .vlink_data_out_vld(vlink_data_out_vld_array[i]),

            .hlink_wdata(hlink_wdata_array[i]),
            .hlink_wen(hlink_wen_array[i]),
            .hlink_rdata(hlink_rdata_array[i]),
            .hlink_rvalid(hlink_rvalid_array[i]),

            .control_state(control_state),
            .control_state_update(control_state_update),

            .start(start),
            .finish(core_finish_array[i])
        );
    end
endgenerate




endmodule
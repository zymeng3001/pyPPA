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

module core_ctrl #(
    parameter HLINK_DATA_WIDTH = (`MAC_MULT_NUM * `IDATA_WIDTH),
    parameter VLINK_DATA_WIDTH  = (`MAC_MULT_NUM * `IDATA_WIDTH),
    parameter GBUS_DATA_WIDTH = `GBUS_DATA_WIDTH,
    parameter IDATA_WIDTH = `IDATA_WIDTH,
    parameter CMEM_ADDR_WIDTH = `CMEM_ADDR_WIDTH,
    parameter CMEM_DATA_WIDTH = (`MAC_MULT_NUM * `IDATA_WIDTH),
    parameter ODATA_BIT = `ODATA_WIDTH,               // Accumulate Sum Bitwidth
    parameter CORE_INDEX = 0,

    parameter   CACHE_DEPTH =             `KV_CACHE_DEPTH_SINGLE_USER_WITH_GQA,                 // Depth
    parameter   CACHE_NUM   =             `MAC_MULT_NUM,
    parameter   CACHE_ADDR_WIDTH  =       ($clog2(CACHE_NUM)+$clog2(CACHE_DEPTH))   // Address Bitwidth
)(
    input  logic                                                    clk,
    input  logic                                                    rst_n,
                    
    input  CONTROL_STATE                                            control_state,
    input  logic                                                    control_state_update,  

    input  USER_CONFIG                                              usr_cfg,

    input  MODEL_CONFIG                                             model_cfg,

                                                                            //start信号具有和control state一样的寄存器delay
    input  logic                                                    start,  //start indicate the current opearation starts
    output logic                                                    finish, //finish indicate the current opeatation ends for this ctrl

    input  logic                                                    hlink_wen, 
    input  logic                                                    hlink_rvalid,
    input  logic          [HLINK_DATA_WIDTH-1:0]                    hlink_rdata,    //注意这里是rdata

    input  logic                                                    cmem_rvalid,
    input  logic          [CMEM_DATA_WIDTH-1:0]                     cmem_rdata,


    input  logic          [ODATA_BIT - 1:0]                         rc_out_data,
    input  logic                                                    rc_out_data_vld,

    input  logic          [IDATA_WIDTH-1:0]                         quant_odata, //for those who write back the byte
    input  logic                                                    quant_odata_valid,     

     
    input  logic          [`GBUS_DATA_WIDTH-1:0]                    parallel_data,
    input  logic                                                    parallel_data_valid,

    output logic                                                    recompute_needed,      

    output logic                                                    self_cmem_ren,
    output logic          [CMEM_ADDR_WIDTH-1:0]                     self_cmem_raddr,

    output logic                                                    mac_opa_vld,
    output logic          [(`MAC_MULT_NUM * `IDATA_WIDTH)-1:0]      mac_opa, //MAC输入A的数据

    output logic                                                    mac_opb_vld,
    output logic          [(`MAC_MULT_NUM * `IDATA_WIDTH)-1:0]      mac_opb, //MAC输入B的数据
    
    output BUS_ADDR                                                 out_gbus_addr,
    output logic                                                    out_gbus_wen,
    output logic          [GBUS_DATA_WIDTH-1:0]                     out_gbus_wdata
);
localparam Q_WEIGHT_ADDR_BASE = 0;
localparam K_WEIGHT_ADDR_BASE = (`MAX_EMBD_SIZE * `MAX_EMBD_SIZE / `HEAD_NUM / `HEAD_CORE_NUM / `MAC_MULT_NUM) * 1;
localparam V_WEIGHT_ADDR_BASE = (`MAX_EMBD_SIZE * `MAX_EMBD_SIZE / `HEAD_NUM / `HEAD_CORE_NUM / `MAC_MULT_NUM) * 2;
localparam PROJ_WEIGHT_ADDR_BASE = (`MAX_EMBD_SIZE * `MAX_EMBD_SIZE / `HEAD_NUM / `HEAD_CORE_NUM / `MAC_MULT_NUM) * 3;
//localparam FFN0_WEIGHT_ADDR_BASE = (`MAX_EMBD_SIZE * `MAX_EMBD_SIZE / `HEAD_NUM / `HEAD_CORE_NUM / `MAC_MULT_NUM) * 4;
//localparam FFN1_WEIGHT_ADDR_BASE = (`MAX_EMBD_SIZE * `MAX_EMBD_SIZE / `HEAD_NUM / `HEAD_CORE_NUM / `MAC_MULT_NUM) * 8;
localparam FFN0_WEIGHT_ADDR_BASE = (`WMEM_DEPTH/`WMEM_NUM_PER_CORE);
localparam FFN1_WEIGHT_ADDR_BASE = (`WMEM_DEPTH/`WMEM_NUM_PER_CORE*2);

localparam K_CACHE_ADDR_BASE = 0;
localparam V_CACHE_ADDR_BASE = (`MAX_CONTEXT_LENGTH * `MAX_EMBD_SIZE / `HEAD_NUM / `HEAD_CORE_NUM / `MAC_MULT_NUM);

CONTROL_STATE                             control_state_reg;
// logic                                     control_state_update_reg;                                                  
logic                                     start_reg;


always_ff@ (posedge clk or negedge rst_n)begin
    if(~rst_n)
        start_reg <= 0;
    else if(start)
        start_reg <= 1;
    else
        start_reg <= 0; //pulse
end

always_ff@ (posedge clk or negedge rst_n)begin
    if(~rst_n)
        control_state_reg <= IDLE_STATE;
    else if(control_state_update)
        control_state_reg <= control_state;
end

always_comb begin
    recompute_needed = 0;
    if(control_state_reg == Q_GEN_STATE ||
       control_state_reg == K_GEN_STATE ||
       control_state_reg == V_GEN_STATE ||
       control_state_reg == ATT_PV_STATE ||
       control_state_reg == FFN0_STATE)begin
        recompute_needed = 1;
    end
end

/////////////////
//control logic//
/////////////////

logic                                                          nxt_finish; //nxt_finish信号应该在write pointer那里控制
    
logic                                                          nxt_self_cmem_ren;
logic          [CMEM_ADDR_WIDTH-1:0]                           nxt_self_cmem_raddr;
    
BUS_ADDR                                                       nxt_out_gbus_addr;
    
logic                                                          nxt_out_gbus_wen;
logic          [GBUS_DATA_WIDTH-1:0]                           nxt_out_gbus_wdata;
    
logic          [`OP_GEN_CNT_WIDTH-1:0]                         op_gen_cnt; //由于我们是权重固定的SA，并且每次只有一行输入向量会参与计算，这个会统计在一次计算中，有多少列权重参与了计算
logic          [`OP_GEN_CNT_WIDTH-1:0]                         nxt_op_gen_cnt;

logic          [`MAX_NUM_USER-1:0][`TOKEN_PER_CORE_WIDTH-1:0]  k_token_per_core_cnt;//count to decide during K gen when to write back to the next core
logic          [`MAX_NUM_USER-1:0][`TOKEN_PER_CORE_WIDTH-1:0]  nxt_k_token_per_core_cnt;

logic          [`MAX_NUM_USER-1:0][$clog2(`HEAD_CORE_NUM)-1:0] k_core_cnt; //count to decide during K gen which core to write back.
logic          [`MAX_NUM_USER-1:0][$clog2(`HEAD_CORE_NUM)-1:0] nxt_k_core_cnt; //count to decide during K gen which core to write back.

logic          [`OP_GEN_CNT_WIDTH-1:0]                         max_op_gen_cnt;

//mac opA data
always_ff @(posedge clk or negedge rst_n) begin
    if(~rst_n)begin
        mac_opa_vld <= 0;
        mac_opa <= 0;
    end
    else if(hlink_rvalid)begin
        mac_opa_vld <= 1; //打一拍
        mac_opa <= hlink_rdata;
    end
    else begin
        mac_opa_vld <= 0;
    end
end

//mac opB data
always_comb begin
    mac_opb_vld = 0;
    mac_opb = 0;
    if(cmem_rvalid)begin
        mac_opb_vld = 1;
        mac_opb = cmem_rdata;
    end
end

//max counter num generation
always_ff @(posedge clk or negedge rst_n) begin
    if(~rst_n)begin
        max_op_gen_cnt <= 0;
    end
    else if(start_reg) begin
        case(control_state_reg)
            Q_GEN_STATE, K_GEN_STATE, V_GEN_STATE, ATT_PV_STATE: begin
                max_op_gen_cnt <= model_cfg.qkv_weight_cols_per_core;
            end
            ATT_QK_STATE: begin
                // if(usr_cfg.user_kv_cache_not_full & (usr_cfg.user_token_cnt<model_cfg.token_per_core))
                //     max_op_gen_cnt <= usr_cfg.user_token_cnt + 1;
                // else
                    max_op_gen_cnt <= model_cfg.token_per_core/4;
            end
            PROJ_STATE, FFN1_STATE: begin
                max_op_gen_cnt <= model_cfg.embd_size / `HEAD_CORE_NUM;
            end
            FFN0_STATE: begin
                max_op_gen_cnt <= model_cfg.qkv_weight_cols_per_core * 4;
            end
        endcase
    end
end

//read pointer
always_comb begin
    nxt_self_cmem_ren = 0;
    nxt_self_cmem_raddr = self_cmem_raddr;
    case (control_state_reg)
        IDLE_STATE: begin
            nxt_self_cmem_ren = 0;
            nxt_self_cmem_raddr = 0;
        end

        Q_GEN_STATE, K_GEN_STATE, V_GEN_STATE, ATT_QK_STATE, ATT_PV_STATE, PROJ_STATE, FFN0_STATE, FFN1_STATE:begin
            if(finish)begin
                nxt_self_cmem_ren = 0;
            end
            else if(hlink_wen)begin 
                nxt_self_cmem_ren = 1;
            end
            else begin
                nxt_self_cmem_ren = 0;
            end

            //nxt_self_cmem_raddr
            if(finish)begin
                nxt_self_cmem_raddr = 0;
            end
            else if(start_reg)begin //这里需要把读地址初始化到对应bias，对于Q gen就是0
                if(control_state_reg == Q_GEN_STATE)
                    nxt_self_cmem_raddr = Q_WEIGHT_ADDR_BASE;
                else if(control_state_reg == V_GEN_STATE)
                    nxt_self_cmem_raddr = V_WEIGHT_ADDR_BASE;
                else if(control_state_reg == K_GEN_STATE)
                    nxt_self_cmem_raddr = K_WEIGHT_ADDR_BASE;
                else if(control_state_reg == PROJ_STATE)
                    nxt_self_cmem_raddr = PROJ_WEIGHT_ADDR_BASE;
                else if(control_state_reg == ATT_QK_STATE)begin
                    nxt_self_cmem_raddr[0 +: $clog2(CACHE_DEPTH)] = K_CACHE_ADDR_BASE;
                    nxt_self_cmem_raddr[CMEM_ADDR_WIDTH-1] = 1; //read the KV CACHE
                end
                else if(control_state_reg == ATT_PV_STATE)begin
                    nxt_self_cmem_raddr[0 +: $clog2(CACHE_DEPTH)] = V_CACHE_ADDR_BASE;
                    nxt_self_cmem_raddr[CMEM_ADDR_WIDTH-1] = 1; //read the KV CACHE
                end
                else if(control_state_reg == FFN0_STATE)
                    nxt_self_cmem_raddr = FFN0_WEIGHT_ADDR_BASE;
                else if(control_state_reg == FFN1_STATE)
                    nxt_self_cmem_raddr = FFN1_WEIGHT_ADDR_BASE;                
            end
            else if(self_cmem_ren)begin
                if(control_state_reg == ATT_PV_STATE)begin
                    nxt_self_cmem_raddr = self_cmem_raddr + model_cfg.qkv_weight_cols_per_core;
                    if(usr_cfg.user_kv_cache_not_full)begin
                        if(self_cmem_raddr[$clog2(`KV_CACHE_DEPTH_SINGLE_USER_WITH_GQA)-1 : 0] >=  (usr_cfg.user_token_cnt / `MAC_MULT_NUM) * model_cfg.qkv_weight_cols_per_core + V_CACHE_ADDR_BASE)//读V矩阵的下一列
                            nxt_self_cmem_raddr = self_cmem_raddr - (usr_cfg.user_token_cnt / `MAC_MULT_NUM) * model_cfg.qkv_weight_cols_per_core + 1;
                    end
                    else begin
                        if(self_cmem_raddr[$clog2(`KV_CACHE_DEPTH_SINGLE_USER_WITH_GQA)-1 : 0] >=  (model_cfg.max_context_length / `MAC_MULT_NUM - 1) * model_cfg.qkv_weight_cols_per_core + V_CACHE_ADDR_BASE)
                            nxt_self_cmem_raddr = self_cmem_raddr - (model_cfg.max_context_length / `MAC_MULT_NUM - 1) * model_cfg.qkv_weight_cols_per_core + 1;
                    end
                end
                else begin
                    nxt_self_cmem_raddr = self_cmem_raddr + 1; 
                end
            end
        end
    endcase
end

//write pointer
always_comb begin
    nxt_out_gbus_addr = out_gbus_addr;
    nxt_out_gbus_wen = 0;
    nxt_out_gbus_wdata = out_gbus_wdata;
    nxt_finish = 0;//nxt_finish信号应该在write pointer那里控制
    nxt_op_gen_cnt = op_gen_cnt;
    nxt_k_token_per_core_cnt = k_token_per_core_cnt;
    nxt_k_core_cnt = k_core_cnt;

    case (control_state_reg)
        IDLE_STATE: begin
            nxt_out_gbus_wen = 0;
            nxt_out_gbus_wdata = 0;
        end

        Q_GEN_STATE, ATT_QK_STATE, ATT_PV_STATE, PROJ_STATE, FFN0_STATE, FFN1_STATE:begin
            if(finish)begin
                nxt_out_gbus_wen = 0;
                nxt_out_gbus_wdata = 0;
            end
            else if(rc_out_data_vld && (control_state_reg == PROJ_STATE || control_state_reg == FFN1_STATE))begin
                nxt_out_gbus_wdata = rc_out_data;
                nxt_out_gbus_wen = 1;
            end
            else if(parallel_data_valid && (control_state_reg == ATT_QK_STATE))begin
                nxt_out_gbus_wdata = parallel_data;
                nxt_out_gbus_wen = 1;
            end
            else if(quant_odata_valid && control_state_reg != PROJ_STATE && control_state_reg != FFN1_STATE && control_state_reg != ATT_QK_STATE)begin //core_odata_valid的pulse数量应该是刚好合适的
                nxt_out_gbus_wdata = quant_odata;
                nxt_out_gbus_wen = 1;
            end

            //gbus writing ptr
            if(finish)begin
                nxt_out_gbus_addr = 0;
                nxt_op_gen_cnt = 0;
            end
            else if(start_reg)begin //这里需要把写地址初始化到对应bias
                // nxt_out_gbus_addr.cmem_addr[$clog2(`HEAD_SRAM_DEPTH) + $clog2(`MAC_MULT_NUM)] = 1; //byte addressable
                nxt_op_gen_cnt = 0;

                if(control_state_reg == PROJ_STATE || control_state_reg == FFN1_STATE)
                    nxt_out_gbus_addr.hs_bias = 2; //写入residual sram
                else
                    nxt_out_gbus_addr.hs_bias = 1; //写入head sram

                if(control_state_reg == Q_GEN_STATE || control_state_reg == ATT_PV_STATE) begin
                    nxt_out_gbus_addr.cmem_addr[$clog2(`HEAD_SRAM_DEPTH)-1:0] = (CORE_INDEX * model_cfg.qkv_weight_cols_per_core) / `MAC_MULT_NUM; //每个core初始的写地址，应该是第CORE_INDEX*`QKV_WEIGHT_COLS_PER_CORE列，由于拼接，就是CORE_INDEX*`QKV_WEIGHT_COLS_PER_CORE/`MAC_MULT_NUM
                    nxt_out_gbus_addr.cmem_addr[$clog2(`HEAD_SRAM_DEPTH) +: $clog2(`MAC_MULT_NUM)] = (CORE_INDEX * model_cfg.qkv_weight_cols_per_core) % `MAC_MULT_NUM; //head sram slice 选择
                end
                else if(control_state_reg == ATT_QK_STATE) begin
                    nxt_out_gbus_addr.cmem_addr[$clog2(`HEAD_SRAM_DEPTH)-1:0] = (CORE_INDEX * model_cfg.token_per_core) / `MAC_MULT_NUM;
                    nxt_out_gbus_addr.cmem_addr[$clog2(`HEAD_SRAM_DEPTH) +: $clog2(`MAC_MULT_NUM)] = (CORE_INDEX * model_cfg.token_per_core) % `MAC_MULT_NUM;
                end
                else if(control_state_reg == PROJ_STATE || control_state_reg == FFN1_STATE) begin
                    nxt_out_gbus_addr.cmem_addr[$clog2(`GLOBAL_SRAM_DEPTH)-1:0] = (CORE_INDEX * (model_cfg.embd_size / `HEAD_CORE_NUM)) / `MAC_MULT_NUM; 
                    nxt_out_gbus_addr.cmem_addr[$clog2(`GLOBAL_SRAM_DEPTH) +: $clog2(`MAC_MULT_NUM)] = (CORE_INDEX * (model_cfg.embd_size / `HEAD_CORE_NUM)) % `MAC_MULT_NUM;
                    // nxt_out_gbus_addr.cmem_addr[$clog2(`GLOBAL_SRAM_DEPTH) + $clog2(`MAC_MULT_NUM)] = 1; //byte addressable
                end
                else if(control_state_reg == FFN0_STATE) begin
                    nxt_out_gbus_addr.cmem_addr[$clog2(`HEAD_SRAM_DEPTH)-1:0] = (CORE_INDEX * model_cfg.qkv_weight_cols_per_core * 4) / `MAC_MULT_NUM; //每个core初始的写地址，应该是第CORE_INDEX*`QKV_WEIGHT_COLS_PER_CORE*4列，由于拼接，就是CORE_INDEX*`QKV_WEIGHT_COLS_PER_CORE*4/`MAC_MULT_NUM
                    nxt_out_gbus_addr.cmem_addr[$clog2(`HEAD_SRAM_DEPTH) +: $clog2(`MAC_MULT_NUM)] = (CORE_INDEX * model_cfg.qkv_weight_cols_per_core * 4) % `MAC_MULT_NUM; //head sram slice 选择
                end

            end
            else if(out_gbus_wen)begin
                if (op_gen_cnt == max_op_gen_cnt-1)begin //写到当前core能写到的最后一列
                    nxt_finish = 1;
                    nxt_op_gen_cnt = 0;
                    nxt_out_gbus_addr = 0;
                end
                else if(op_gen_cnt[$clog2(`MAC_MULT_NUM)-1:0] == `MAC_MULT_NUM-1 || (control_state_reg == ATT_QK_STATE && op_gen_cnt[$clog2(`MAC_MULT_NUM)-1:0] == `MAC_MULT_NUM/4-1))begin //当前地址拼接完成，head/global sram地址+1
                    nxt_out_gbus_addr.cmem_addr[$clog2(`HEAD_SRAM_DEPTH)-1:0] = out_gbus_addr.cmem_addr[$clog2(`HEAD_SRAM_DEPTH)-1:0] + 1;
                    // if(control_state_reg == ATT_QK_STATE)
                    //     nxt_op_gen_cnt = op_gen_cnt + 4;
                    // else
                        nxt_op_gen_cnt = op_gen_cnt + 1;

                    if(control_state_reg == Q_GEN_STATE || control_state_reg == ATT_PV_STATE)
                        nxt_out_gbus_addr.cmem_addr[$clog2(`HEAD_SRAM_DEPTH) +: $clog2(`MAC_MULT_NUM)] = (CORE_INDEX * model_cfg.qkv_weight_cols_per_core) % `MAC_MULT_NUM;
                    else if(control_state_reg == ATT_QK_STATE)
                        nxt_out_gbus_addr.cmem_addr[$clog2(`HEAD_SRAM_DEPTH) +: $clog2(`MAC_MULT_NUM)] = (CORE_INDEX * model_cfg.token_per_core) % `MAC_MULT_NUM;
                    else if(control_state_reg == PROJ_STATE || control_state_reg == FFN1_STATE)begin
                        nxt_out_gbus_addr.cmem_addr[$clog2(`GLOBAL_SRAM_DEPTH) +: $clog2(`MAC_MULT_NUM)] = (CORE_INDEX * (model_cfg.embd_size / `HEAD_CORE_NUM)) % `MAC_MULT_NUM;
                    end
                    else if(control_state_reg == FFN0_STATE)
                        nxt_out_gbus_addr.cmem_addr[$clog2(`HEAD_SRAM_DEPTH) +: $clog2(`MAC_MULT_NUM)] = (CORE_INDEX * model_cfg.qkv_weight_cols_per_core * 4) % `MAC_MULT_NUM;

                end
                else begin
                    if(control_state_reg == PROJ_STATE || control_state_reg == FFN1_STATE)
                        nxt_out_gbus_addr.cmem_addr[$clog2(`GLOBAL_SRAM_DEPTH) +: $clog2(`MAC_MULT_NUM)] = out_gbus_addr.cmem_addr[$clog2(`GLOBAL_SRAM_DEPTH) +: $clog2(`MAC_MULT_NUM)]+1;
                    else if(control_state_reg == ATT_QK_STATE)
                        nxt_out_gbus_addr.cmem_addr[$clog2(`HEAD_SRAM_DEPTH) +: $clog2(`MAC_MULT_NUM)] = out_gbus_addr.cmem_addr[$clog2(`HEAD_SRAM_DEPTH) +: $clog2(`MAC_MULT_NUM)]+4;
                    else
                        nxt_out_gbus_addr.cmem_addr[$clog2(`HEAD_SRAM_DEPTH) +: $clog2(`MAC_MULT_NUM)] = out_gbus_addr.cmem_addr[$clog2(`HEAD_SRAM_DEPTH) +: $clog2(`MAC_MULT_NUM)]+1;
                   
                    // if(control_state_reg == ATT_QK_STATE)
                    //     nxt_op_gen_cnt = op_gen_cnt + 4;
                    // else
                        nxt_op_gen_cnt = op_gen_cnt + 1;
                end
            end
        end
        
        K_GEN_STATE:begin
            if(finish)begin
                nxt_out_gbus_wen = 0;
                nxt_out_gbus_wdata = 0;
            end
            else if(quant_odata_valid)begin //core_odata_valid的pulse数量应该是刚好合适写完K矩阵
                nxt_out_gbus_wdata = quant_odata;
                nxt_out_gbus_wen = 1;
            end
            
            //gbus writing ptr
            if(finish)begin
                nxt_op_gen_cnt = 0;
                nxt_out_gbus_addr = 0;
            end
            else if(start_reg)begin//这里需要把写地址初始化到对应bias
                nxt_out_gbus_addr.hs_bias = 0;
                nxt_out_gbus_addr.cmem_addr[CMEM_ADDR_WIDTH-1] = 1; //write the KV CACHE
                // nxt_out_gbus_addr.cmem_addr[CACHE_ADDR_WIDTH] = 1; //cache address byte addressable
                nxt_out_gbus_addr.cmem_addr[CACHE_ADDR_WIDTH-1:$clog2(CACHE_DEPTH)] = (CORE_INDEX * model_cfg.qkv_weight_cols_per_core) % `MAC_MULT_NUM; //select slice
                if(usr_cfg.user_token_cnt == 0) begin //user的第一个token或者写到最后一行了, 回到初始点
                    nxt_out_gbus_addr.core_addr = 0;
                    nxt_out_gbus_addr.cmem_addr[$clog2(CACHE_DEPTH)-1:0] = K_CACHE_ADDR_BASE + (CORE_INDEX * model_cfg.qkv_weight_cols_per_core)/`MAC_MULT_NUM; //initial bias
                    nxt_k_token_per_core_cnt[usr_cfg.user_id] = 0;
                    nxt_k_core_cnt[usr_cfg.user_id] = 0; //Start from first core
                end
                else if(k_token_per_core_cnt[usr_cfg.user_id] == model_cfg.token_per_core - 1)begin//该换下一个core写了
                    nxt_out_gbus_addr.core_addr = k_core_cnt[usr_cfg.user_id] + 1; //next core
                    nxt_out_gbus_addr.cmem_addr[$clog2(CACHE_DEPTH)-1:0] = K_CACHE_ADDR_BASE + (CORE_INDEX * model_cfg.qkv_weight_cols_per_core)/`MAC_MULT_NUM;
                    nxt_k_token_per_core_cnt[usr_cfg.user_id] = 0;
                    nxt_k_core_cnt[usr_cfg.user_id] = k_core_cnt[usr_cfg.user_id] + 1;
                end
                else begin //在同一个core写下一行
                    nxt_out_gbus_addr.core_addr = k_core_cnt[usr_cfg.user_id]; 
                    nxt_out_gbus_addr.cmem_addr[$clog2(CACHE_DEPTH)-1:0] =  K_CACHE_ADDR_BASE + //k store base addr
                                                                            (CORE_INDEX * model_cfg.qkv_weight_cols_per_core)/`MAC_MULT_NUM + //core index offset
                                                                            (k_token_per_core_cnt[usr_cfg.user_id]+1)*model_cfg.qkv_weight_cols_per_core*`HEAD_CORE_NUM/`MAC_MULT_NUM ; //skip previous tokens
                                                                            // (model_cfg.qkv_weight_cols_per_core/`MAC_MULT_NUM * (`HEAD_CORE_NUM-1)+1);//跳过其他core写到本core的相同token的部分
                    nxt_k_token_per_core_cnt[usr_cfg.user_id] = k_token_per_core_cnt[usr_cfg.user_id] + 1;
                end
            end
            else if(out_gbus_wen)begin
                if(op_gen_cnt == max_op_gen_cnt-1)begin//如果写到当前core能写到列数的最后一列
                    nxt_finish = 1;
                end
                else if(out_gbus_addr.cmem_addr[CACHE_ADDR_WIDTH-1:$clog2(CACHE_DEPTH)]==`MAC_MULT_NUM-1) begin//水平方向Slice都写满了，换下一行KV Cache 写
                    nxt_out_gbus_addr.cmem_addr[$clog2(CACHE_DEPTH)-1:0] = out_gbus_addr.cmem_addr[$clog2(CACHE_DEPTH)-1:0] + 1;
                    nxt_out_gbus_addr.cmem_addr[CACHE_ADDR_WIDTH-1:$clog2(CACHE_DEPTH)] = 0;
                    nxt_op_gen_cnt = op_gen_cnt + 1;
                end
                else begin //先水平方向写kv cache
                    nxt_out_gbus_addr.cmem_addr[CACHE_ADDR_WIDTH-1:$clog2(CACHE_DEPTH)] = out_gbus_addr.cmem_addr[CACHE_ADDR_WIDTH-1:$clog2(CACHE_DEPTH)] + 1;
                    nxt_op_gen_cnt = op_gen_cnt + 1;
                end
            end
        end

        V_GEN_STATE:begin
            if(finish)begin
                nxt_out_gbus_wen = 0;
                nxt_out_gbus_wdata = 0;
            end
            else if(quant_odata_valid)begin //The number of pulses of core_odata_valid should be just enough to write the Q matrix.
                nxt_out_gbus_wdata = quant_odata;
                nxt_out_gbus_wen = 1;
            end

            //gbus writing ptr
            if(finish)begin
                nxt_out_gbus_addr = 0;
                nxt_op_gen_cnt = 0;
            end
            else if(start_reg)begin //initial to bias
                nxt_out_gbus_addr.hs_bias = 0; //write to cmem
                nxt_out_gbus_addr.core_addr = CORE_INDEX;
                nxt_out_gbus_addr.cmem_addr[CMEM_ADDR_WIDTH-1] = 1; //write to the KV CACHE
                // nxt_out_gbus_addr.cmem_addr[CACHE_ADDR_WIDTH] = 1; //cache address byte addressable
                nxt_out_gbus_addr.cmem_addr[CACHE_ADDR_WIDTH-1:$clog2(CACHE_DEPTH)] = usr_cfg.user_token_cnt % `MAC_MULT_NUM; //CACHE SLICE select
                nxt_out_gbus_addr.cmem_addr[$clog2(CACHE_DEPTH)-1:0] = V_CACHE_ADDR_BASE + (usr_cfg.user_token_cnt / `MAC_MULT_NUM) * model_cfg.qkv_weight_cols_per_core; //initial bias
                nxt_op_gen_cnt = 0;
            end
            else if(out_gbus_wen)begin
                if (op_gen_cnt == max_op_gen_cnt-1)begin //写到当前core能写到的最后一列
                    nxt_finish = 1;
                    nxt_op_gen_cnt = 0;
                    nxt_out_gbus_addr = 0;
                end
                else begin
                    nxt_out_gbus_addr.cmem_addr[$clog2(CACHE_DEPTH)-1:0] = out_gbus_addr.cmem_addr[$clog2(CACHE_DEPTH)-1:0] + 1;
                    nxt_op_gen_cnt = op_gen_cnt + 1;
                end
            end
        end

        default: begin

        end
    endcase
end

always_ff @(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        self_cmem_ren <= 0;
        self_cmem_raddr <= 0;
        finish <= 0;
        out_gbus_addr <= 0;
        out_gbus_wen <= 0;
        out_gbus_wdata <= 0;
        op_gen_cnt <= 0;
        k_token_per_core_cnt <= 0;
        k_core_cnt <= 0;
    end
    else begin
        self_cmem_ren <= nxt_self_cmem_ren;
        self_cmem_raddr <= nxt_self_cmem_raddr;
        finish <= nxt_finish;
        out_gbus_addr <= nxt_out_gbus_addr;
        out_gbus_wen <= nxt_out_gbus_wen;
        out_gbus_wdata <= nxt_out_gbus_wdata;
        op_gen_cnt <= nxt_op_gen_cnt;
        k_token_per_core_cnt <= nxt_k_token_per_core_cnt;
        k_core_cnt <= nxt_k_core_cnt;
    end
end



endmodule
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

module head_sram_rd_ctrl(
    input  logic                                                            clk,
    input  logic                                                            rst_n,

    input  CONTROL_STATE                                                    control_state,
    input  logic                                                            control_state_update,

    input  logic                                                            model_cfg_vld,
    input  MODEL_CONFIG                                                     model_cfg,
    input  USER_CONFIG                                                      usr_cfg,
    input                                                                   usr_cfg_vld,

    input  logic                                                            start,

    output logic                                                            finish,

    output logic                                                            head_sram_ren,
    output logic    [$clog2(`MAC_MULT_NUM) + $clog2(`HEAD_SRAM_DEPTH)-1:0]  head_sram_raddr
);


CONTROL_STATE                                control_state_reg;
MODEL_CONFIG                                 model_cfg_reg;
USER_CONFIG                                  usr_cfg_reg;
logic                                        start_reg;

always_ff @(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        control_state_reg <= IDLE_STATE;
    end
    else if(control_state_update)begin
        control_state_reg <= control_state;
    end
end

always_ff @(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        start_reg <= 0;
    end
    else begin
        start_reg <= start;
    end
end

always_ff@ (posedge clk or negedge rst_n)begin
    if(~rst_n)
        model_cfg_reg <= 0;
    else if(model_cfg_vld)
        model_cfg_reg <= model_cfg;
end

always_ff@ (posedge clk or negedge rst_n)begin
    if(~rst_n)
        usr_cfg_reg <= 0;
    else if(usr_cfg_vld)
        usr_cfg_reg <= usr_cfg;
end

logic                                                                nxt_finish;
logic                                                                nxt_head_sram_ren;
logic    [$clog2(`MAC_MULT_NUM) + $clog2(`HEAD_SRAM_DEPTH)-1:0]      nxt_head_sram_raddr;

logic    [$clog2(`MAC_MULT_NUM) + $clog2(`HEAD_SRAM_DEPTH):0]        max_head_sram_raddr_cnt;

// maximum head sram read addr configuration for different state:
always_ff@ (posedge clk or negedge rst_n)begin
    if(~rst_n) begin
        max_head_sram_raddr_cnt <= 0;
    end
    else if(start_reg) begin
        case(control_state_reg)
            ATT_QK_STATE, PROJ_STATE: begin               //head dim
                max_head_sram_raddr_cnt <= model_cfg_reg.qkv_weight_cols_per_core*`HEAD_CORE_NUM/`MAC_MULT_NUM;
            end
            ATT_PV_STATE: begin
                if(usr_cfg_reg.user_kv_cache_not_full) begin //If kv cache not full
                    max_head_sram_raddr_cnt <= usr_cfg_reg.user_token_cnt/`MAC_MULT_NUM + 1; 
                end                                                          
                else begin
                    max_head_sram_raddr_cnt <= model_cfg_reg.max_context_length/`MAC_MULT_NUM;
                end
            end
            FFN1_STATE: begin
                max_head_sram_raddr_cnt <= model_cfg_reg.qkv_weight_cols_per_core*`HEAD_CORE_NUM * 4 /`MAC_MULT_NUM;
            end
        endcase
    end
end

//control logic
always_comb begin
    nxt_finish = 0;
    nxt_head_sram_ren = 0;
    nxt_head_sram_raddr = head_sram_raddr;
    case (control_state_reg)
        IDLE_STATE:begin
            nxt_finish = 0;
        end

        Q_GEN_STATE, K_GEN_STATE, V_GEN_STATE, FFN0_STATE: begin
            if(start_reg)begin
                nxt_finish = 1; //Q_GEN的时候直接finish
            end
            else if(finish)begin
                nxt_finish = 0; 
            end
        end

        ATT_QK_STATE,ATT_PV_STATE,PROJ_STATE,FFN1_STATE: begin
            if(start_reg)begin //Start
                nxt_head_sram_ren = 1;
                nxt_head_sram_raddr = 0;
            end
            else if (finish)begin //rst to zero
                nxt_head_sram_ren = 0;
                nxt_head_sram_raddr = 0;
                nxt_finish = 0; //pulse
            end
            else if(head_sram_ren & (head_sram_raddr == max_head_sram_raddr_cnt - 1))begin //Read to the last part of Q/QKT
                nxt_head_sram_ren = 0;
                nxt_head_sram_raddr = head_sram_raddr + 1;
                nxt_finish = 1;
            end
            else if(head_sram_ren)begin //开始读之后，默认情况下地址+1
                nxt_head_sram_ren = 1;
                nxt_head_sram_raddr = head_sram_raddr + 1;
            end
        end

    endcase
end

always_ff @(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        finish <= 0;
        head_sram_ren <= 0;
        head_sram_raddr <= 0;
    end
    else begin
        finish <= nxt_finish;
        head_sram_ren <= nxt_head_sram_ren;
        head_sram_raddr <= nxt_head_sram_raddr;
    end
end
endmodule
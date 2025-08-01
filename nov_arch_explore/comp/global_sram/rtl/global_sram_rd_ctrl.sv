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

module global_sram_rd_ctrl#( //同时控制GLOBAL SRAM和RESIDUAL SRAM
    parameter MAX_QKV_WEIGHT_COLS_PER_CORE = `MAX_QKV_WEIGHT_COLS_PER_CORE
)(
    input  logic                                        clk,
    input  logic                                        rst_n,
        
    input  CONTROL_STATE                                control_state,
    input  logic                                        control_state_update,
    input  logic                                        model_cfg_vld,
    input  MODEL_CONFIG                                 model_cfg,
                                                        //start信号具有和control state一样的寄存器delay
    input  logic                                        start,
    output logic                                        finish,

    input  logic           [`CMEM_ADDR_WIDTH-1:0]       vector_out_data_addr,
    input  logic                                        vector_out_data_vld,

    output logic                                        global_sram_ren,
    output logic  [$clog2(`GLOBAL_SRAM_DEPTH)-1:0]      global_sram_raddr
);

CONTROL_STATE                                        control_state_reg;
MODEL_CONFIG                                         model_cfg_reg;
// logic                                                control_state_update_reg;
logic                                                start_reg;

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

always_ff@ (posedge clk or negedge rst_n)begin
    if(~rst_n)
        model_cfg_reg <= 0;
    else if(model_cfg_vld)
        model_cfg_reg <= model_cfg;
end

//control logic
logic                                                 nxt_finish;

logic                                                 nxt_global_sram_ren;
logic          [$clog2(`GLOBAL_SRAM_DEPTH)-1:0]       nxt_global_sram_raddr;

always_comb begin
    nxt_global_sram_ren = 0;
    nxt_global_sram_raddr = 0;
    nxt_finish = 0;
    
    case (control_state_reg)
        IDLE_STATE: begin
            nxt_global_sram_ren = 0;
            nxt_global_sram_raddr = 0;
            nxt_finish = 0;
        end

        Q_GEN_STATE, K_GEN_STATE, V_GEN_STATE, FFN0_STATE: begin
            if(start_reg)begin //开始启动
                nxt_global_sram_ren = 1;
                nxt_global_sram_raddr = 0;
            end
            else if (finish)begin //rst to zero
                nxt_global_sram_ren = 0;
                nxt_global_sram_raddr = 0;
                nxt_finish = 0; //pulse
            end
            else if(global_sram_raddr == model_cfg_reg.embd_size/`MAC_MULT_NUM - 1)begin //读到一行的末尾
                nxt_global_sram_ren = 0;
                nxt_global_sram_raddr = global_sram_raddr + 1;
                nxt_finish = 1;
            end
            else if(global_sram_ren)begin //开始读之后，默认情况下地址+1
                nxt_global_sram_ren = 1;
                nxt_global_sram_raddr = global_sram_raddr + 1;
            end
        end

        FFN1_STATE: begin //FFN1 residual
            nxt_global_sram_ren = vector_out_data_vld;
            nxt_global_sram_raddr = vector_out_data_addr; //截取低位
        end

        ATT_QK_STATE, ATT_PV_STATE, PROJ_STATE: begin
            if(start_reg)begin
                nxt_finish = 1;
            end
            else if(finish)begin
                nxt_finish = 0; 
            end
        end


    endcase
end

always_ff @(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        global_sram_ren <= 0;
        global_sram_raddr <= 0;
        finish <= 0;
    end
    else begin
        global_sram_ren <= nxt_global_sram_ren;
        global_sram_raddr <= nxt_global_sram_raddr;
        finish <= nxt_finish;      
    end
end


endmodule
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

module abuf #(
)
(
    input  logic                                        clk,
    input  logic                                        rst_n,

    input                                               start,

    input  logic [(`MAC_MULT_NUM * `IDATA_WIDTH)-1:0]   in_data,
    input  logic                                        in_data_vld,

    input  CONTROL_STATE                                control_state,
    input  logic                                        control_state_update, 

    input  logic                                        model_cfg_vld,
    input  MODEL_CONFIG                                 model_cfg,

    input  USER_CONFIG                                  usr_cfg,
    input                                               usr_cfg_vld,

    output logic [(`MAC_MULT_NUM * `IDATA_WIDTH)-1:0]   out_data,
    output logic                                        out_data_vld,

    output logic                                        finish_row
);

logic                                                                           start_reg;

logic                                                                           next_finish_row;
logic                                                                           reuse_row, next_reuse_row;

logic   [`ABUF_EMBD_REG_DEPTH-1:0][(`MAC_MULT_NUM * `IDATA_WIDTH)-1:0]          embd_reg, next_embd_reg;
logic   [(`MAC_MULT_NUM * `IDATA_WIDTH)-1:0]                                    next_out_data;
logic                                                                           next_out_data_vld;
logic   [$clog2(`ABUF_EMBD_REG_DEPTH)-1:0]                                      embd_reg_wr_ptr, next_embd_reg_wr_ptr;
logic   [$clog2(`ABUF_EMBD_REG_DEPTH)-1:0]                                      embd_reg_rd_ptr, next_embd_reg_rd_ptr;
logic   [`ABUF_MAX_ITER_WIDTH-1:0]                                              row_iteration_cnt, next_row_iteration_cnt;

//max cnt reg for control

logic   [`ABUF_MAX_ITER_WIDTH-1:0]                                              max_row_iteration_cnt; //maximum reuse count
logic   [$clog2(`ABUF_EMBD_REG_DEPTH):0]                                        max_embd_reg_cnt; //maximum number of count embd.

CONTROL_STATE                                                                   control_state_reg;
MODEL_CONFIG                                                                    model_cfg_reg;
USER_CONFIG                                                                     usr_cfg_reg;

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

always_ff@ (posedge clk or negedge rst_n)begin
    if(~rst_n)
        usr_cfg_reg <= 0;
    else if(usr_cfg_vld)
        usr_cfg_reg <= usr_cfg;
end

//Config max counter for Attention Q read, and read of Q will then affect how many K and V we will read.
always_ff@ (posedge clk or negedge rst_n)begin
    if(~rst_n) begin
        max_row_iteration_cnt <= 0;
        max_embd_reg_cnt <= 0;
    end
    else if(start_reg) begin
        case(control_state_reg)
            Q_GEN_STATE, K_GEN_STATE, V_GEN_STATE: begin
                max_row_iteration_cnt <= model_cfg_reg.qkv_weight_cols_per_core;
                max_embd_reg_cnt <= model_cfg_reg.embd_size/`MAC_MULT_NUM;
            end
            ATT_QK_STATE: begin
                max_embd_reg_cnt <= model_cfg_reg.qkv_weight_cols_per_core*`HEAD_CORE_NUM/`MAC_MULT_NUM; //model_cfg_reg.qkv_weight_cols_per_core*`HEAD_CORE_NUM 是 head_dim
                // if(usr_cfg_reg.user_kv_cache_not_full && usr_cfg_reg.user_token_cnt<model_cfg_reg.token_per_core) begin //If kv cache not full and first core's K not full
                //     max_row_iteration_cnt <= usr_cfg_reg.user_token_cnt + 1; //user token cnt counts from 0, if =3, reuse 4 times
                // end
                // else begin
                    max_row_iteration_cnt <= model_cfg_reg.token_per_core;
                // end
            end
            ATT_PV_STATE: begin
                max_row_iteration_cnt <= model_cfg_reg.qkv_weight_cols_per_core;
                if(usr_cfg_reg.user_kv_cache_not_full) begin //If kv cache not full
                    max_embd_reg_cnt <= usr_cfg_reg.user_token_cnt/`MAC_MULT_NUM + 1;
                end                                                                 
                else begin
                    max_embd_reg_cnt <= model_cfg_reg.max_context_length/`MAC_MULT_NUM;
                end
            end
            PROJ_STATE: begin
                max_row_iteration_cnt <= model_cfg_reg.embd_size / `HEAD_CORE_NUM;
                max_embd_reg_cnt <= model_cfg_reg.qkv_weight_cols_per_core*`HEAD_CORE_NUM/`MAC_MULT_NUM;
            end
            FFN0_STATE: begin
                max_row_iteration_cnt <= model_cfg_reg.qkv_weight_cols_per_core * 4;
                max_embd_reg_cnt <= model_cfg_reg.embd_size/`MAC_MULT_NUM;
            end
            FFN1_STATE: begin
                max_row_iteration_cnt <= model_cfg_reg.embd_size / `HEAD_CORE_NUM;
                max_embd_reg_cnt <= model_cfg_reg.qkv_weight_cols_per_core*`HEAD_CORE_NUM*4/`MAC_MULT_NUM;
            end
        endcase
    end
end


always_comb begin
    next_out_data = 0;
    next_out_data_vld = 0;
    next_row_iteration_cnt = row_iteration_cnt;
    next_embd_reg_wr_ptr = embd_reg_wr_ptr;
    next_embd_reg_rd_ptr = embd_reg_rd_ptr;
    next_finish_row = 0;
    next_reuse_row = reuse_row;
    next_embd_reg = embd_reg;
    
    case (control_state_reg)
        IDLE_STATE: begin
            next_out_data = 0;
            next_out_data_vld = 0;
            next_row_iteration_cnt = 0;
            next_embd_reg_wr_ptr = 0;
            next_embd_reg_rd_ptr = 0;
            next_finish_row = 0;
            next_reuse_row = 0;
            next_embd_reg = 0;
        end
        
        // Q_GEN_STATE, K_GEN_STATE, V_GEN_STATE, ATT_QK_STATE, ATT_PV_STATE, PROJ_STATE, FFN0_STATE: begin
        default: begin
            if(reuse_row)begin
                // next_embd_reg_wr_ptr = 0;
                next_out_data_vld = 1;
                next_out_data = embd_reg[embd_reg_rd_ptr];
                next_embd_reg_rd_ptr = embd_reg_rd_ptr + 1;
                if(out_data_vld && (embd_reg_rd_ptr == max_embd_reg_cnt-1) && reuse_row) begin //Last element during reuse
                    
                    if(row_iteration_cnt == max_row_iteration_cnt-1) begin //Last Iteration, proceed to next row
                        next_out_data_vld = 1;
                        next_out_data = embd_reg[embd_reg_rd_ptr];
                        next_row_iteration_cnt = 0;
                        next_embd_reg_rd_ptr = 0;
                        next_finish_row = 1;
                        next_reuse_row = 0;
                        next_embd_reg = 0;
                    end
                    else begin //Same row, next Iteration
                        next_out_data_vld = 1;
                        next_out_data = embd_reg[embd_reg_rd_ptr];
                        next_row_iteration_cnt = row_iteration_cnt + 1;
                        next_embd_reg_rd_ptr = 0;
                    end

                end                
            end
            else if(in_data_vld) begin //New row, 1st Iteration
                next_out_data_vld = 1;
                next_out_data = in_data;
                next_embd_reg_wr_ptr = embd_reg_wr_ptr + 1;

                if(max_row_iteration_cnt != 1) begin //IF there is one iteration, don't need to write embd_reg
                    next_embd_reg[embd_reg_wr_ptr] = in_data;
                end

                if(embd_reg_wr_ptr == max_embd_reg_cnt-1)begin
                    next_embd_reg_wr_ptr = 0;
                    if(max_row_iteration_cnt==1) begin 
                        next_row_iteration_cnt = 0;
                        next_finish_row = 1;
                        next_reuse_row = 0;
                    end
                    else begin
                        next_embd_reg[embd_reg_wr_ptr] = in_data;
                        next_reuse_row = 1;
                        next_row_iteration_cnt = row_iteration_cnt + 1;
                    end 
                end
            end
        end
    endcase
end

always_ff @(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        out_data <= 0;
        out_data_vld <= 0;
        row_iteration_cnt <= 0;
        embd_reg_wr_ptr <= 0;
        embd_reg_rd_ptr <= 0;
        finish_row <= 0;
        reuse_row <= 0;
        embd_reg <= 0;
    end
    else begin
        out_data <= next_out_data;
        out_data_vld <= next_out_data_vld;
        row_iteration_cnt <= next_row_iteration_cnt;
        embd_reg_wr_ptr <= next_embd_reg_wr_ptr;
        embd_reg_rd_ptr <= next_embd_reg_rd_ptr;
        finish_row <= next_finish_row;
        reuse_row <= next_reuse_row;
        embd_reg <= next_embd_reg;
    end
end

endmodule
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

module head_array (
    input  logic                                               clk,
    input  logic                                               rst_n,

    input  logic                                               clean_kv_cache, //pulse
    input  logic       [`USER_ID_WIDTH-1:0]                    clean_kv_cache_user_id, //clean the corrsponding user's kv cache to zero



    //There should be the ports to initial the wmem and global mem from SPI or someting else
    input  logic     [`ARRAY_MEM_ADDR_WIDTH-1:0]               array_mem_addr, //bias 是 0
    input  logic     [`INTERFACE_DATA_WIDTH-1:0]               array_mem_wdata,
    input  logic                                               array_mem_wen,
    output logic     [`INTERFACE_DATA_WIDTH-1:0]               array_mem_rdata,
    input  logic                                               array_mem_ren,
    output logic                                               array_mem_rvld,

    input  logic                                               global_sram_rvld,
    input  logic [(`MAC_MULT_NUM * `IDATA_WIDTH)-1:0]          global_sram_rdata,

    input  CONTROL_STATE                                       control_state,
    input  logic                                               control_state_update,

    input  logic                                               start,
    output logic                                               finish,

    input  OP_CONFIG                                           op_cfg,
    input  logic                                               op_cfg_vld,

    input  USER_CONFIG                                         usr_cfg,
    input  logic                                               usr_cfg_vld,

    input  MODEL_CONFIG                                        model_cfg,
    input  logic                                               model_cfg_vld,
    
    input  logic                                               pmu_cfg_vld,
    input  PMU_CONFIG                                          pmu_cfg, 

    input  logic                                               rc_cfg_vld,
    input  RC_CONFIG                                           rc_cfg,

    output BUS_ADDR   [`HEAD_NUM-1:0]                          gbus_addr_delay1_array,
    output logic      [`HEAD_NUM-1:0]                          gbus_wen_delay1_array,
    output logic      [`HEAD_NUM-1:0][`GBUS_DATA_WIDTH-1:0]    gbus_wdata_delay1_array    
    
);

genvar i;

logic [`HEAD_NUM-1:0] head_finish_array;
logic [`HEAD_NUM-1:0] head_finish_flag_array;

always_ff @(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        finish <= 0;
    end
    else if(finish)begin
        finish <= 0;
    end
    else begin
        finish <= &head_finish_flag_array;
    end
end

generate
    for(i=0; i < `HEAD_NUM; i++)begin
        always_ff@(posedge clk or negedge rst_n)begin
            if(~rst_n)begin
                head_finish_flag_array[i] <= 0;
            end
            else if(finish)begin
                head_finish_flag_array[i] <= 0;                
            end
            else if(head_finish_array[i])begin
                head_finish_flag_array[i] <= 1;
            end
        end
    end
endgenerate


///////////////////////////////////////
//              VLINK                //
///////////////////////////////////////


// logic         [`HEAD_NUM-1:0][`HEAD_CORE_NUM-1:0][(`MAC_MULT_NUM * `IDATA_WIDTH)-1:0]   vlink_data_in_array_array;
// logic         [`HEAD_NUM-1:0][`HEAD_CORE_NUM-1:0]                                       vlink_data_in_vld_array_array;
// logic         [`HEAD_NUM-1:0][`HEAD_CORE_NUM-1:0][(`MAC_MULT_NUM * `IDATA_WIDTH)-1:0]   vlink_data_out_array_array;
// logic         [`HEAD_NUM-1:0][`HEAD_CORE_NUM-1:0]                                       vlink_data_out_vld_array_array;

// generate
//     for(i=0; i < `HEAD_NUM; i++)begin
//         if(i%2==0)begin // i = 0 2 4 6
//             assign vlink_data_in_array_array[i] = vlink_data_out_array_array[i+1];
//             assign vlink_data_in_vld_array_array[i] = vlink_data_out_vld_array_array[i+1];
//         end
//         else begin // i = 1 3 5 7
//             assign vlink_data_in_array_array[i] = vlink_data_out_array_array[i-1];
//             assign vlink_data_in_vld_array_array[i] = vlink_data_out_vld_array_array[i-1];
//         end
//     end
// endgenerate


///////////////////////////////////////////
//              INTERFACE                //
///////////////////////////////////////////


logic [`HEAD_NUM-1:0][`HEAD_MEM_ADDR_WIDTH-1:0]                nxt_head_mem_addr_array;
logic [`HEAD_NUM-1:0][`INTERFACE_DATA_WIDTH-1:0]               nxt_head_mem_wdata_array;
logic [`HEAD_NUM-1:0]                                          nxt_head_mem_wen_array;

logic [`HEAD_NUM-1:0]                                          nxt_head_mem_ren_array;


logic [`HEAD_NUM-1:0][`HEAD_MEM_ADDR_WIDTH-1:0]                head_mem_addr_array;
logic [`HEAD_NUM-1:0][`INTERFACE_DATA_WIDTH-1:0]               head_mem_wdata_array;
logic [`HEAD_NUM-1:0]                                          head_mem_wen_array;
logic [`HEAD_NUM-1:0][`INTERFACE_DATA_WIDTH-1:0]               head_mem_rdata_array;
logic [`HEAD_NUM-1:0]                                          head_mem_ren_array;
logic [`HEAD_NUM-1:0]                                          head_mem_rvld_array;


always_comb begin
    nxt_head_mem_addr_array = head_mem_addr_array;
    nxt_head_mem_wdata_array = head_mem_wdata_array;
    nxt_head_mem_wen_array = 0;
    nxt_head_mem_ren_array = 0;
    for(int i = 0; i < `HEAD_NUM; i++)begin
        if(~array_mem_addr[`ARRAY_MEM_ADDR_WIDTH-1])begin //写core mem的
            if(i == array_mem_addr[`ARRAY_MEM_ADDR_WIDTH-4 +: 3])begin
                nxt_head_mem_addr_array[i] = {1'b0, array_mem_addr[17:0]};
                nxt_head_mem_wdata_array[i] = array_mem_wdata;
                nxt_head_mem_wen_array[i] = array_mem_wen;
                nxt_head_mem_ren_array[i] = array_mem_ren;               
            end
        end
        else begin //可能写head sram的
            if(i == array_mem_addr[10:8])begin
                nxt_head_mem_addr_array[i] = {1'b1, 10'b0, array_mem_addr[7:0]};
                nxt_head_mem_wdata_array[i] = array_mem_wdata;
                nxt_head_mem_wen_array[i] = array_mem_wen;
                nxt_head_mem_ren_array[i] = array_mem_ren;               
            end
        end
    end
end

always_ff @(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        head_mem_addr_array <= 0;
        head_mem_wdata_array <= 0;
        head_mem_wen_array <= 0;
        head_mem_ren_array <= 0;
    end
    else begin
        head_mem_addr_array <= nxt_head_mem_addr_array;
        head_mem_wdata_array <= nxt_head_mem_wdata_array;
        head_mem_wen_array <= nxt_head_mem_wen_array;
        head_mem_ren_array <= nxt_head_mem_ren_array;
    end
end


logic     [`INTERFACE_DATA_WIDTH-1:0]               nxt_array_mem_rdata;
logic                                               nxt_array_mem_rvld;
always_comb begin
    nxt_array_mem_rvld = |head_mem_rvld_array;
    nxt_array_mem_rdata = 0;
    for(int i =0; i < `HEAD_NUM; i++)begin
        if(head_mem_rvld_array[i])begin
            nxt_array_mem_rdata = head_mem_rdata_array[i];
        end
    end
end

always_ff @(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        array_mem_rvld <= 0;
        array_mem_rdata <= 0;
    end
    else begin
        array_mem_rvld <= nxt_array_mem_rvld;
        array_mem_rdata <= nxt_array_mem_rdata;
    end
end

`ifndef APR_TWO_HEADS
generate
    for(i = 0; i < `HEAD_NUM/2; i++)begin: head_gen_array
    two_heads two_heads_inst(
        .clk(clk),
        .rst_n(rst_n),

        .clean_kv_cache(clean_kv_cache), //pulse
        .clean_kv_cache_user_id(clean_kv_cache_user_id), //clean the corrsponding user's kv cache to zero

        .head_mem_addr_0(head_mem_addr_array[2*i]),
        .head_mem_wdata_0(head_mem_wdata_array[2*i]),
        .head_mem_wen_0(head_mem_wen_array[2*i]),
        .head_mem_rdata_0(head_mem_rdata_array[2*i]),
        .head_mem_ren_0(head_mem_ren_array[2*i]),
        .head_mem_rvld_0(head_mem_rvld_array[2*i]),

        .head_mem_addr_1(head_mem_addr_array[2*i+1]),
        .head_mem_wdata_1(head_mem_wdata_array[2*i+1]),
        .head_mem_wen_1(head_mem_wen_array[2*i+1]),
        .head_mem_rdata_1(head_mem_rdata_array[2*i+1]),
        .head_mem_ren_1(head_mem_ren_array[2*i+1]),
        .head_mem_rvld_1(head_mem_rvld_array[2*i+1]),
    
        .global_sram_rd_data(global_sram_rdata),
        .global_sram_rd_data_vld(global_sram_rvld),

        .control_state(control_state),
        .control_state_update(control_state_update),
                                                                        //start信号具有和control state一样的寄存器delay
        .start(start), //start indicate the current opearation starts
        .finish_0(head_finish_array[2*i]), //finish indicate the current opeatation ends for this ctrl    
        .finish_1(head_finish_array[2*i+1]), //finish indicate the current opeatation ends for this ctrl

        .op_cfg_vld(op_cfg_vld),      //和control_state的来自core_array的寄存器delay一样
        .op_cfg(op_cfg),

        .usr_cfg_vld(usr_cfg_vld),
        .usr_cfg(usr_cfg),

        .model_cfg_vld(model_cfg_vld),
        .model_cfg(model_cfg),

        .pmu_cfg_vld(pmu_cfg_vld),
        .pmu_cfg(pmu_cfg), 

        .rc_cfg_vld(rc_cfg_vld),
        .rc_cfg(rc_cfg),//和model config一样在网络初始化的时候配置好就行   

        .gbus_addr_delay1_0(gbus_addr_delay1_array[2*i]), //latency consideration
        .gbus_wen_delay1_0(gbus_wen_delay1_array[2*i]),
        .gbus_wdata_delay1_0(gbus_wdata_delay1_array[2*i]),     

        .gbus_addr_delay1_1(gbus_addr_delay1_array[2*i+1]), //latency consideration
        .gbus_wen_delay1_1(gbus_wen_delay1_array[2*i+1]),
        .gbus_wdata_delay1_1(gbus_wdata_delay1_array[2*i+1])  
    );
    end
endgenerate
`else

generate
    for(i = 0; i < `HEAD_NUM/2; i++)begin: head_gen_array
    two_heads two_heads_inst(
        .clk (clk),
        .rst_n (rst_n),

        .clean_kv_cache (clean_kv_cache),
        .clean_kv_cache_user_id (clean_kv_cache_user_id),

        .head_mem_addr_0 (head_mem_addr_array[2*i]),
        .head_mem_wdata_0 (head_mem_wdata_array[2*i]),
        .head_mem_wen_0 (head_mem_wen_array[2*i]),
        .head_mem_rdata_0 (head_mem_rdata_array[2*i]),
        .head_mem_ren_0 (head_mem_ren_array[2*i]),
        .head_mem_rvld_0 (head_mem_rvld_array[2*i]),

        .head_mem_addr_1 (head_mem_addr_array[2*i+1]),
        .head_mem_wdata_1 (head_mem_wdata_array[2*i+1]),
        .head_mem_wen_1 (head_mem_wen_array[2*i+1]),
        .head_mem_rdata_1 (head_mem_rdata_array[2*i+1]),
        .head_mem_ren_1 (head_mem_ren_array[2*i+1]),
        .head_mem_rvld_1 (head_mem_rvld_array[2*i+1]),

        .global_sram_rd_data (global_sram_rdata),
        .global_sram_rd_data_vld (global_sram_rvld),

        .control_state (control_state),
        .control_state_update (control_state_update),
        
        .start (start),
        .finish_0 (head_finish_array[2*i]),
        .finish_1 (head_finish_array[2*i+1]),

        .op_cfg_vld (op_cfg_vld), 
        .op_cfg_cfg_acc_num_9_(op_cfg.cfg_acc_num[9]),
        .op_cfg_cfg_acc_num_8_(op_cfg.cfg_acc_num[8]),
        .op_cfg_cfg_acc_num_7_(op_cfg.cfg_acc_num[7]),
        .op_cfg_cfg_acc_num_6_(op_cfg.cfg_acc_num[6]),
        .op_cfg_cfg_acc_num_5_(op_cfg.cfg_acc_num[5]),
        .op_cfg_cfg_acc_num_4_(op_cfg.cfg_acc_num[4]),
        .op_cfg_cfg_acc_num_3_(op_cfg.cfg_acc_num[3]),
        .op_cfg_cfg_acc_num_2_(op_cfg.cfg_acc_num[2]),
        .op_cfg_cfg_acc_num_1_(op_cfg.cfg_acc_num[1]),
        .op_cfg_cfg_acc_num_0_(op_cfg.cfg_acc_num[0]),
        .op_cfg_cfg_quant_scale_9_(op_cfg.cfg_quant_scale[9]),
        .op_cfg_cfg_quant_scale_8_(op_cfg.cfg_quant_scale[8]),
        .op_cfg_cfg_quant_scale_7_(op_cfg.cfg_quant_scale[7]),
        .op_cfg_cfg_quant_scale_6_(op_cfg.cfg_quant_scale[6]),
        .op_cfg_cfg_quant_scale_5_(op_cfg.cfg_quant_scale[5]),
        .op_cfg_cfg_quant_scale_4_(op_cfg.cfg_quant_scale[4]),
        .op_cfg_cfg_quant_scale_3_(op_cfg.cfg_quant_scale[3]),
        .op_cfg_cfg_quant_scale_2_(op_cfg.cfg_quant_scale[2]),
        .op_cfg_cfg_quant_scale_1_(op_cfg.cfg_quant_scale[1]),
        .op_cfg_cfg_quant_scale_0_(op_cfg.cfg_quant_scale[0]),
        .op_cfg_cfg_quant_bias_15_(op_cfg.cfg_quant_bias[15]),
        .op_cfg_cfg_quant_bias_14_(op_cfg.cfg_quant_bias[14]),
        .op_cfg_cfg_quant_bias_13_(op_cfg.cfg_quant_bias[13]),
        .op_cfg_cfg_quant_bias_12_(op_cfg.cfg_quant_bias[12]),
        .op_cfg_cfg_quant_bias_11_(op_cfg.cfg_quant_bias[11]),
        .op_cfg_cfg_quant_bias_10_(op_cfg.cfg_quant_bias[10]),
        .op_cfg_cfg_quant_bias_9_(op_cfg.cfg_quant_bias[9]),
        .op_cfg_cfg_quant_bias_8_(op_cfg.cfg_quant_bias[8]),
        .op_cfg_cfg_quant_bias_7_(op_cfg.cfg_quant_bias[7]),
        .op_cfg_cfg_quant_bias_6_(op_cfg.cfg_quant_bias[6]),
        .op_cfg_cfg_quant_bias_5_(op_cfg.cfg_quant_bias[5]),
        .op_cfg_cfg_quant_bias_4_(op_cfg.cfg_quant_bias[4]),
        .op_cfg_cfg_quant_bias_3_(op_cfg.cfg_quant_bias[3]),
        .op_cfg_cfg_quant_bias_2_(op_cfg.cfg_quant_bias[2]),
        .op_cfg_cfg_quant_bias_1_(op_cfg.cfg_quant_bias[1]),
        .op_cfg_cfg_quant_bias_0_(op_cfg.cfg_quant_bias[0]),
        .op_cfg_cfg_quant_shift_4_(op_cfg.cfg_quant_shift[4]),
        .op_cfg_cfg_quant_shift_3_(op_cfg.cfg_quant_shift[3]),
        .op_cfg_cfg_quant_shift_2_(op_cfg.cfg_quant_shift[2]),
        .op_cfg_cfg_quant_shift_1_(op_cfg.cfg_quant_shift[1]),
        .op_cfg_cfg_quant_shift_0_(op_cfg.cfg_quant_shift[0]),

        .usr_cfg_vld(usr_cfg_vld),
        .usr_cfg_user_id_1_(usr_cfg.user_id[1]),
        .usr_cfg_user_id_0_(usr_cfg.user_id[0]),
        .usr_cfg_user_token_cnt_8_(usr_cfg.user_token_cnt[8]),
        .usr_cfg_user_token_cnt_7_(usr_cfg.user_token_cnt[7]),
        .usr_cfg_user_token_cnt_6_(usr_cfg.user_token_cnt[6]),
        .usr_cfg_user_token_cnt_5_(usr_cfg.user_token_cnt[5]),
        .usr_cfg_user_token_cnt_4_(usr_cfg.user_token_cnt[4]),
        .usr_cfg_user_token_cnt_3_(usr_cfg.user_token_cnt[3]),
        .usr_cfg_user_token_cnt_2_(usr_cfg.user_token_cnt[2]),
        .usr_cfg_user_token_cnt_1_(usr_cfg.user_token_cnt[1]),
        .usr_cfg_user_token_cnt_0_(usr_cfg.user_token_cnt[0]),
        .usr_cfg_user_kv_cache_not_full(usr_cfg.user_kv_cache_not_full),

        .model_cfg_vld(model_cfg_vld),   
        .model_cfg_max_context_length_9_(model_cfg.max_context_length[9]),
        .model_cfg_max_context_length_8_(model_cfg.max_context_length[8]),
        .model_cfg_max_context_length_7_(model_cfg.max_context_length[7]),
        .model_cfg_max_context_length_6_(model_cfg.max_context_length[6]),
        .model_cfg_max_context_length_5_(model_cfg.max_context_length[5]),
        .model_cfg_max_context_length_4_(model_cfg.max_context_length[4]),
        .model_cfg_max_context_length_3_(model_cfg.max_context_length[3]),
        .model_cfg_max_context_length_2_(model_cfg.max_context_length[2]),
        .model_cfg_max_context_length_1_(model_cfg.max_context_length[1]),
        .model_cfg_max_context_length_0_(model_cfg.max_context_length[0]),
        .model_cfg_qkv_weight_cols_per_core_2_(model_cfg.qkv_weight_cols_per_core[2]),
        .model_cfg_qkv_weight_cols_per_core_1_(model_cfg.qkv_weight_cols_per_core[1]),
        .model_cfg_qkv_weight_cols_per_core_0_(model_cfg.qkv_weight_cols_per_core[0]),
        .model_cfg_token_per_core_5_(model_cfg.token_per_core[5]),
        .model_cfg_token_per_core_4_(model_cfg.token_per_core[4]),
        .model_cfg_token_per_core_3_(model_cfg.token_per_core[3]),
        .model_cfg_token_per_core_2_(model_cfg.token_per_core[2]),
        .model_cfg_token_per_core_1_(model_cfg.token_per_core[1]),
        .model_cfg_token_per_core_0_(model_cfg.token_per_core[0]),
        .model_cfg_embd_size_9_(model_cfg.embd_size[9]),
        .model_cfg_embd_size_8_(model_cfg.embd_size[8]),
        .model_cfg_embd_size_7_(model_cfg.embd_size[7]),
        .model_cfg_embd_size_6_(model_cfg.embd_size[6]),
        .model_cfg_embd_size_5_(model_cfg.embd_size[5]),
        .model_cfg_embd_size_4_(model_cfg.embd_size[4]),
        .model_cfg_gqa_en(model_cfg.gqa_en),

        .pmu_cfg_vld(pmu_cfg_vld),
        .pmu_cfg_pmu_cfg_bc1(pmu_cfg.pmu_cfg_bc1),
        .pmu_cfg_pmu_cfg_bc2(pmu_cfg.pmu_cfg_bc2),
        .pmu_cfg_pmu_cfg_en(pmu_cfg.pmu_cfg_en),
        .pmu_cfg_deepsleep_en(pmu_cfg.deepsleep_en),
        
        .rc_cfg_vld(rc_cfg_vld),
        .rc_cfg_rms_rc_shift_4_(rc_cfg.rms_rc_shift[4]),
        .rc_cfg_rms_rc_shift_3_(rc_cfg.rms_rc_shift[3]),
        .rc_cfg_rms_rc_shift_2_(rc_cfg.rms_rc_shift[2]),
        .rc_cfg_rms_rc_shift_1_(rc_cfg.rms_rc_shift[1]),
        .rc_cfg_rms_rc_shift_0_(rc_cfg.rms_rc_shift[0]),
        .rc_cfg_rms_K_9_(rc_cfg.rms_K[9]),
        .rc_cfg_rms_K_8_(rc_cfg.rms_K[8]),
        .rc_cfg_rms_K_7_(rc_cfg.rms_K[7]),
        .rc_cfg_rms_K_6_(rc_cfg.rms_K[6]),
        .rc_cfg_rms_K_5_(rc_cfg.rms_K[5]),
        .rc_cfg_rms_K_4_(rc_cfg.rms_K[4]),
        .rc_cfg_rms_K_3_(rc_cfg.rms_K[3]),
        .rc_cfg_rms_K_2_(rc_cfg.rms_K[2]),
        .rc_cfg_rms_K_1_(rc_cfg.rms_K[1]),
        .rc_cfg_rms_K_0_(rc_cfg.rms_K[0]),
        .rc_cfg_attn_rms_dequant_scale_square_15_(rc_cfg.attn_rms_dequant_scale_square[15]),
        .rc_cfg_attn_rms_dequant_scale_square_14_(rc_cfg.attn_rms_dequant_scale_square[14]),
        .rc_cfg_attn_rms_dequant_scale_square_13_(rc_cfg.attn_rms_dequant_scale_square[13]),
        .rc_cfg_attn_rms_dequant_scale_square_12_(rc_cfg.attn_rms_dequant_scale_square[12]),
        .rc_cfg_attn_rms_dequant_scale_square_11_(rc_cfg.attn_rms_dequant_scale_square[11]),
        .rc_cfg_attn_rms_dequant_scale_square_10_(rc_cfg.attn_rms_dequant_scale_square[10]),
        .rc_cfg_attn_rms_dequant_scale_square_9_(rc_cfg.attn_rms_dequant_scale_square[9]),
        .rc_cfg_attn_rms_dequant_scale_square_8_(rc_cfg.attn_rms_dequant_scale_square[8]),
        .rc_cfg_attn_rms_dequant_scale_square_7_(rc_cfg.attn_rms_dequant_scale_square[7]),
        .rc_cfg_attn_rms_dequant_scale_square_6_(rc_cfg.attn_rms_dequant_scale_square[6]),
        .rc_cfg_attn_rms_dequant_scale_square_5_(rc_cfg.attn_rms_dequant_scale_square[5]),
        .rc_cfg_attn_rms_dequant_scale_square_4_(rc_cfg.attn_rms_dequant_scale_square[4]),
        .rc_cfg_attn_rms_dequant_scale_square_3_(rc_cfg.attn_rms_dequant_scale_square[3]),
        .rc_cfg_attn_rms_dequant_scale_square_2_(rc_cfg.attn_rms_dequant_scale_square[2]),
        .rc_cfg_attn_rms_dequant_scale_square_1_(rc_cfg.attn_rms_dequant_scale_square[1]),
        .rc_cfg_attn_rms_dequant_scale_square_0_(rc_cfg.attn_rms_dequant_scale_square[0]),
        .rc_cfg_mlp_rms_dequant_scale_square_15_(rc_cfg.mlp_rms_dequant_scale_square[15]),
        .rc_cfg_mlp_rms_dequant_scale_square_14_(rc_cfg.mlp_rms_dequant_scale_square[14]),
        .rc_cfg_mlp_rms_dequant_scale_square_13_(rc_cfg.mlp_rms_dequant_scale_square[13]),
        .rc_cfg_mlp_rms_dequant_scale_square_12_(rc_cfg.mlp_rms_dequant_scale_square[12]),
        .rc_cfg_mlp_rms_dequant_scale_square_11_(rc_cfg.mlp_rms_dequant_scale_square[11]),
        .rc_cfg_mlp_rms_dequant_scale_square_10_(rc_cfg.mlp_rms_dequant_scale_square[10]),
        .rc_cfg_mlp_rms_dequant_scale_square_9_(rc_cfg.mlp_rms_dequant_scale_square[9]),
        .rc_cfg_mlp_rms_dequant_scale_square_8_(rc_cfg.mlp_rms_dequant_scale_square[8]),
        .rc_cfg_mlp_rms_dequant_scale_square_7_(rc_cfg.mlp_rms_dequant_scale_square[7]),
        .rc_cfg_mlp_rms_dequant_scale_square_6_(rc_cfg.mlp_rms_dequant_scale_square[6]),
        .rc_cfg_mlp_rms_dequant_scale_square_5_(rc_cfg.mlp_rms_dequant_scale_square[5]),
        .rc_cfg_mlp_rms_dequant_scale_square_4_(rc_cfg.mlp_rms_dequant_scale_square[4]),
        .rc_cfg_mlp_rms_dequant_scale_square_3_(rc_cfg.mlp_rms_dequant_scale_square[3]),
        .rc_cfg_mlp_rms_dequant_scale_square_2_(rc_cfg.mlp_rms_dequant_scale_square[2]),
        .rc_cfg_mlp_rms_dequant_scale_square_1_(rc_cfg.mlp_rms_dequant_scale_square[1]),
        .rc_cfg_mlp_rms_dequant_scale_square_0_(rc_cfg.mlp_rms_dequant_scale_square[0]),
        .rc_cfg_softmax_rc_shift_4_(rc_cfg.softmax_rc_shift[4]),
        .rc_cfg_softmax_rc_shift_3_(rc_cfg.softmax_rc_shift[3]),
        .rc_cfg_softmax_rc_shift_2_(rc_cfg.softmax_rc_shift[2]),
        .rc_cfg_softmax_rc_shift_1_(rc_cfg.softmax_rc_shift[1]),
        .rc_cfg_softmax_rc_shift_0_(rc_cfg.softmax_rc_shift[0]),
        .rc_cfg_softmax_input_dequant_scale_15_(rc_cfg.softmax_input_dequant_scale[15]),
        .rc_cfg_softmax_input_dequant_scale_14_(rc_cfg.softmax_input_dequant_scale[14]),
        .rc_cfg_softmax_input_dequant_scale_13_(rc_cfg.softmax_input_dequant_scale[13]),
        .rc_cfg_softmax_input_dequant_scale_12_(rc_cfg.softmax_input_dequant_scale[12]),
        .rc_cfg_softmax_input_dequant_scale_11_(rc_cfg.softmax_input_dequant_scale[11]),
        .rc_cfg_softmax_input_dequant_scale_10_(rc_cfg.softmax_input_dequant_scale[10]),
        .rc_cfg_softmax_input_dequant_scale_9_(rc_cfg.softmax_input_dequant_scale[9]),
        .rc_cfg_softmax_input_dequant_scale_8_(rc_cfg.softmax_input_dequant_scale[8]),
        .rc_cfg_softmax_input_dequant_scale_7_(rc_cfg.softmax_input_dequant_scale[7]),
        .rc_cfg_softmax_input_dequant_scale_6_(rc_cfg.softmax_input_dequant_scale[6]),
        .rc_cfg_softmax_input_dequant_scale_5_(rc_cfg.softmax_input_dequant_scale[5]),
        .rc_cfg_softmax_input_dequant_scale_4_(rc_cfg.softmax_input_dequant_scale[4]),
        .rc_cfg_softmax_input_dequant_scale_3_(rc_cfg.softmax_input_dequant_scale[3]),
        .rc_cfg_softmax_input_dequant_scale_2_(rc_cfg.softmax_input_dequant_scale[2]),
        .rc_cfg_softmax_input_dequant_scale_1_(rc_cfg.softmax_input_dequant_scale[1]),
        .rc_cfg_softmax_input_dequant_scale_0_(rc_cfg.softmax_input_dequant_scale[0]),
        .rc_cfg_softmax_exp_quant_scale_15_(rc_cfg.softmax_exp_quant_scale[15]),
        .rc_cfg_softmax_exp_quant_scale_14_(rc_cfg.softmax_exp_quant_scale[14]),
        .rc_cfg_softmax_exp_quant_scale_13_(rc_cfg.softmax_exp_quant_scale[13]),
        .rc_cfg_softmax_exp_quant_scale_12_(rc_cfg.softmax_exp_quant_scale[12]),
        .rc_cfg_softmax_exp_quant_scale_11_(rc_cfg.softmax_exp_quant_scale[11]),
        .rc_cfg_softmax_exp_quant_scale_10_(rc_cfg.softmax_exp_quant_scale[10]),
        .rc_cfg_softmax_exp_quant_scale_9_(rc_cfg.softmax_exp_quant_scale[9]),
        .rc_cfg_softmax_exp_quant_scale_8_(rc_cfg.softmax_exp_quant_scale[8]),
        .rc_cfg_softmax_exp_quant_scale_7_(rc_cfg.softmax_exp_quant_scale[7]),
        .rc_cfg_softmax_exp_quant_scale_6_(rc_cfg.softmax_exp_quant_scale[6]),
        .rc_cfg_softmax_exp_quant_scale_5_(rc_cfg.softmax_exp_quant_scale[5]),
        .rc_cfg_softmax_exp_quant_scale_4_(rc_cfg.softmax_exp_quant_scale[4]),
        .rc_cfg_softmax_exp_quant_scale_3_(rc_cfg.softmax_exp_quant_scale[3]),
        .rc_cfg_softmax_exp_quant_scale_2_(rc_cfg.softmax_exp_quant_scale[2]),
        .rc_cfg_softmax_exp_quant_scale_1_(rc_cfg.softmax_exp_quant_scale[1]),
        .rc_cfg_softmax_exp_quant_scale_0_(rc_cfg.softmax_exp_quant_scale[0]),

        .gbus_addr_delay1_0_hs_bias_1_          (gbus_addr_delay1_array[2*i].hs_bias[1]),
        .gbus_addr_delay1_0_hs_bias_0_          (gbus_addr_delay1_array[2*i].hs_bias[0]),
        .gbus_addr_delay1_0_core_addr_3_        (gbus_addr_delay1_array[2*i].core_addr[3]),
        .gbus_addr_delay1_0_core_addr_2_        (gbus_addr_delay1_array[2*i].core_addr[2]),
        .gbus_addr_delay1_0_core_addr_1_        (gbus_addr_delay1_array[2*i].core_addr[1]),
        .gbus_addr_delay1_0_core_addr_0_        (gbus_addr_delay1_array[2*i].core_addr[0]),
        .gbus_addr_delay1_0_cmem_addr_12_       (gbus_addr_delay1_array[2*i].cmem_addr[12]),
        .gbus_addr_delay1_0_cmem_addr_11_       (gbus_addr_delay1_array[2*i].cmem_addr[11]),
        .gbus_addr_delay1_0_cmem_addr_10_       (gbus_addr_delay1_array[2*i].cmem_addr[10]),
        .gbus_addr_delay1_0_cmem_addr_9_        (gbus_addr_delay1_array[2*i].cmem_addr[9]),
        .gbus_addr_delay1_0_cmem_addr_8_        (gbus_addr_delay1_array[2*i].cmem_addr[8]),
        .gbus_addr_delay1_0_cmem_addr_7_        (gbus_addr_delay1_array[2*i].cmem_addr[7]),
        .gbus_addr_delay1_0_cmem_addr_6_        (gbus_addr_delay1_array[2*i].cmem_addr[6]),
        .gbus_addr_delay1_0_cmem_addr_5_        (gbus_addr_delay1_array[2*i].cmem_addr[5]),
        .gbus_addr_delay1_0_cmem_addr_4_        (gbus_addr_delay1_array[2*i].cmem_addr[4]),
        .gbus_addr_delay1_0_cmem_addr_3_        (gbus_addr_delay1_array[2*i].cmem_addr[3]),
        .gbus_addr_delay1_0_cmem_addr_2_        (gbus_addr_delay1_array[2*i].cmem_addr[2]),
        .gbus_addr_delay1_0_cmem_addr_1_        (gbus_addr_delay1_array[2*i].cmem_addr[1]),
        .gbus_addr_delay1_0_cmem_addr_0_        (gbus_addr_delay1_array[2*i].cmem_addr[0]),
        .gbus_wen_delay1_0                      (gbus_wen_delay1_array[2*i]),
        .gbus_wdata_delay1_0                    (gbus_wdata_delay1_array[2*i]),

        .gbus_addr_delay1_1_hs_bias_1_          (gbus_addr_delay1_array[2*i+1].hs_bias[1]),                                    
        .gbus_addr_delay1_1_hs_bias_0_          (gbus_addr_delay1_array[2*i+1].hs_bias[0]),                  
        .gbus_addr_delay1_1_core_addr_3_        (gbus_addr_delay1_array[2*i+1].core_addr[3]),               
        .gbus_addr_delay1_1_core_addr_2_        (gbus_addr_delay1_array[2*i+1].core_addr[2]),             
        .gbus_addr_delay1_1_core_addr_1_        (gbus_addr_delay1_array[2*i+1].core_addr[1]),       
        .gbus_addr_delay1_1_core_addr_0_        (gbus_addr_delay1_array[2*i+1].core_addr[0]),         
        .gbus_addr_delay1_1_cmem_addr_12_       (gbus_addr_delay1_array[2*i+1].cmem_addr[12]),    
        .gbus_addr_delay1_1_cmem_addr_11_       (gbus_addr_delay1_array[2*i+1].cmem_addr[11]),      
        .gbus_addr_delay1_1_cmem_addr_10_       (gbus_addr_delay1_array[2*i+1].cmem_addr[10]),          
        .gbus_addr_delay1_1_cmem_addr_9_        (gbus_addr_delay1_array[2*i+1].cmem_addr[9]),        
        .gbus_addr_delay1_1_cmem_addr_8_        (gbus_addr_delay1_array[2*i+1].cmem_addr[8]),      
        .gbus_addr_delay1_1_cmem_addr_7_        (gbus_addr_delay1_array[2*i+1].cmem_addr[7]),     
        .gbus_addr_delay1_1_cmem_addr_6_        (gbus_addr_delay1_array[2*i+1].cmem_addr[6]),        
        .gbus_addr_delay1_1_cmem_addr_5_        (gbus_addr_delay1_array[2*i+1].cmem_addr[5]),    
        .gbus_addr_delay1_1_cmem_addr_4_        (gbus_addr_delay1_array[2*i+1].cmem_addr[4]),       
        .gbus_addr_delay1_1_cmem_addr_3_        (gbus_addr_delay1_array[2*i+1].cmem_addr[3]),       
        .gbus_addr_delay1_1_cmem_addr_2_        (gbus_addr_delay1_array[2*i+1].cmem_addr[2]),    
        .gbus_addr_delay1_1_cmem_addr_1_        (gbus_addr_delay1_array[2*i+1].cmem_addr[1]),              
        .gbus_addr_delay1_1_cmem_addr_0_        (gbus_addr_delay1_array[2*i+1].cmem_addr[0]),   
        .gbus_wen_delay1_1                      (gbus_wen_delay1_array[2*i+1]),            
        .gbus_wdata_delay1_1                    (gbus_wdata_delay1_array[2*i+1])      
    );
    end
endgenerate
`endif

// generate
//     for(i = 0; i < `HEAD_NUM; i++)begin : head_gen_array
//         head_top #(
//             .HEAD_INDEX(i%2)
//         )
//         inst_head_top(
//             .clk(clk),
//             .rst_n(rst_n),

//             .clean_kv_cache(clean_kv_cache),
//             .clean_kv_cache_user_id(clean_kv_cache_user_id),

//             .head_mem_addr(head_mem_addr_array[i]),
//             .head_mem_wdata(head_mem_wdata_array[i]),
//             .head_mem_wen(head_mem_wen_array[i]),
//             .head_mem_rdata(head_mem_rdata_array[i]),
//             .head_mem_ren(head_mem_ren_array[i]),
//             .head_mem_rvld(head_mem_rvld_array[i]),

//             .global_sram_rd_data(global_sram_rdata),
//             .global_sram_rd_data_vld(global_sram_rvld),

//             .vlink_data_in_array(vlink_data_in_array_array[i]),
//             .vlink_data_in_vld_array(vlink_data_in_vld_array_array[i]),
//             .vlink_data_out_array(vlink_data_out_array_array[i]),
//             .vlink_data_out_vld_array(vlink_data_out_vld_array_array[i]),

//             .control_state(control_state),
//             .control_state_update(control_state_update),

//             .start(start),
//             .finish(head_finish_array[i]),

//             // operation Config Signals
//             .op_cfg_vld(op_cfg_vld),   
//             .op_cfg(op_cfg),

//             // model Config Signals
//             .model_cfg_vld(model_cfg_vld),   
//             .model_cfg(model_cfg),

//             // PMU Config Signals
//             .pmu_cfg_vld(pmu_cfg_vld),   
//             .pmu_cfg(pmu_cfg),

//             // user Config Signals
//             .usr_cfg_vld(usr_cfg_vld),
//             .usr_cfg(usr_cfg),

//             .rc_cfg_vld(rc_cfg_vld),
//             .rc_cfg(rc_cfg),

//             .gbus_addr_delay1(gbus_addr_delay1_array[i]),
//             .gbus_wen_delay1(gbus_wen_delay1_array[i]),
//             .gbus_wdata_delay1(gbus_wdata_delay1_array[i])
//         );
//     end
// endgenerate

    
endmodule
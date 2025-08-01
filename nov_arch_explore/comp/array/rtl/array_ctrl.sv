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

module array_ctrl(
    input  logic                                                clk,
    input  logic                                                rst_n,


    //There should be the ports to initial the cfg from SPI or someting else
    

    input  logic [`CONTROL_REGISTERS_ADDR_WIDTH-1:0]            control_reg_addr, 

    input  logic [`CONTROL_REGISTERS_DATA_WIDTH-1:0]            control_reg_wdata,
    input  logic                                                control_reg_wen,

    output logic [`CONTROL_REGISTERS_DATA_WIDTH-1:0]            control_reg_rdata,
    input  logic                                                control_reg_ren,
    output logic                                                control_reg_rvld, //rvld和ren差两个clk cycles，timing consideration (retiming)


    
    //op channel
    output CONTROL_STATE                                        control_state,
    output logic                                                control_state_update,

    output logic                                                op_start, //pulse
    input  logic                                                op_finish,//pulse

    output OP_CONFIG                                            op_cfg,
    output logic                                                op_cfg_vld,

    //user channel
    input  logic                                                new_token, //pulse, should be high after global sram accept all the content of the new token
    input  logic                                                user_first_token, //be in smae phase
    input  logic       [`USER_ID_WIDTH-1:0]                     user_id,// be in same phase

    output logic                                                current_token_finish,
    
    output USER_CONFIG                                          usr_cfg,
    output logic                                                usr_cfg_vld,

    //model channel
    input  logic                                                cfg_init_success, //pulse, when high means that all the config all initial successfully

    output MODEL_CONFIG                                         model_cfg,
    output logic                                                model_cfg_vld,

    output PMU_CONFIG                                           pmu_cfg,
    output logic                                                pmu_cfg_vld,

    output logic                                                rc_cfg_vld,
    output RC_CONFIG                                            rc_cfg,

    //power mode
    input  logic                                                power_mode_en_in,
    input  logic                                                debug_mode_en_in,
    input  logic      [7:0]                                     debug_mode_bits_in //IDLE STATE默认执行
);

MODEL_CONFIG model_cfg_reg;
PMU_CONFIG   pmu_cfg_reg;
RC_CONFIG    rc_cfg_reg;
OP_CONFIG_PKT op_cfg_pkt;

//power mode
logic power_mode_en;
always_ff @(posedge clk or negedge rst_n)begin
    if(~rst_n)
        power_mode_en <= 0;
    else
        power_mode_en <= power_mode_en_in;
end

//debug mode
logic debug_mode_en;
logic [7:0] debug_mode_bits; //IDLE STATE默认执行
always_ff @(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        debug_mode_en <= 0;
        debug_mode_bits <= 0;
    end
    else begin
        debug_mode_en <= debug_mode_en_in;
        debug_mode_bits <= debug_mode_bits_in;
    end
end

`ifdef INTERFACE_ENABLE //disable for functional test for quick simulation
logic [`CONTROL_REGISTERS_ADDR_WIDTH-1:0]            control_reg_addr_delay1; //timing consideration
logic [`CONTROL_REGISTERS_DATA_WIDTH-1:0]            control_reg_wdata_delay1;
logic                                                control_reg_wen_delay1;
logic                                                control_reg_ren_delay1;


always_ff@(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        control_reg_addr_delay1 <= 0; 
        control_reg_wdata_delay1 <= 0;
        control_reg_wen_delay1 <= 0;
        control_reg_ren_delay1 <= 0;
        control_reg_rvld <= 0;
    end
    else begin
        control_reg_addr_delay1 <= control_reg_addr;
        control_reg_wdata_delay1 <=  control_reg_wdata;
        control_reg_wen_delay1 <= control_reg_wen;
        control_reg_ren_delay1 <= control_reg_ren;
        control_reg_rvld <= control_reg_ren_delay1;
    end
end



logic [(2**`CONTROL_REGISTERS_ADDR_WIDTH)-1:0][`CONTROL_REGISTERS_DATA_WIDTH-1:0] control_reg_array;
always_ff @(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        control_reg_array <= 0;
        control_reg_rdata <= 0;
    end
    else begin
        if(control_reg_wen_delay1)begin
            control_reg_array[control_reg_addr_delay1] <= control_reg_wdata_delay1;
        end
        
        if(control_reg_ren_delay1)begin
            control_reg_rdata <= control_reg_array[control_reg_addr_delay1];
        end
    end
end

always_comb begin
    model_cfg_reg.max_context_length = control_reg_array[0];
    model_cfg_reg.qkv_weight_cols_per_core = control_reg_array[1];
    model_cfg_reg.token_per_core = control_reg_array[2];
    model_cfg_reg.embd_size = control_reg_array[3];
    model_cfg_reg.gqa_en = control_reg_array[4];

    pmu_cfg_reg.pmu_cfg_bc1 = control_reg_array[5];
    pmu_cfg_reg.pmu_cfg_bc2 = control_reg_array[6];
    pmu_cfg_reg.pmu_cfg_en = control_reg_array[7][0];
    pmu_cfg_reg.deepsleep_en = control_reg_array[7][1];

    rc_cfg_reg.rms_rc_shift = control_reg_array[8];
    rc_cfg_reg.rms_K = control_reg_array[9];
    rc_cfg_reg.attn_rms_dequant_scale_square = control_reg_array[10];
    rc_cfg_reg.mlp_rms_dequant_scale_square = control_reg_array[11];
    rc_cfg_reg.softmax_rc_shift = control_reg_array[12];
    rc_cfg_reg.softmax_input_dequant_scale = control_reg_array[13];
    rc_cfg_reg.softmax_exp_quant_scale = control_reg_array[14];

    op_cfg_pkt.Q_GEN_CFG.cfg_acc_num = control_reg_array[15];
    op_cfg_pkt.Q_GEN_CFG.cfg_quant_scale = control_reg_array[16];
    op_cfg_pkt.Q_GEN_CFG.cfg_quant_bias = control_reg_array[17];
    op_cfg_pkt.Q_GEN_CFG.cfg_quant_shift = control_reg_array[18];

    op_cfg_pkt.K_GEN_CFG.cfg_acc_num = control_reg_array[19];
    op_cfg_pkt.K_GEN_CFG.cfg_quant_scale = control_reg_array[20];
    op_cfg_pkt.K_GEN_CFG.cfg_quant_bias = control_reg_array[21];
    op_cfg_pkt.K_GEN_CFG.cfg_quant_shift = control_reg_array[22];

    op_cfg_pkt.V_GEN_CFG.cfg_acc_num = control_reg_array[23];
    op_cfg_pkt.V_GEN_CFG.cfg_quant_scale = control_reg_array[24];
    op_cfg_pkt.V_GEN_CFG.cfg_quant_bias = control_reg_array[25];
    op_cfg_pkt.V_GEN_CFG.cfg_quant_shift = control_reg_array[26];

    op_cfg_pkt.ATT_QK_CFG.cfg_acc_num = control_reg_array[27];
    op_cfg_pkt.ATT_QK_CFG.cfg_quant_scale = control_reg_array[28];
    op_cfg_pkt.ATT_QK_CFG.cfg_quant_bias = control_reg_array[29];
    op_cfg_pkt.ATT_QK_CFG.cfg_quant_shift = control_reg_array[30];

    op_cfg_pkt.ATT_PV_CFG.cfg_acc_num = control_reg_array[31];
    op_cfg_pkt.ATT_PV_CFG.cfg_quant_scale = control_reg_array[32];
    op_cfg_pkt.ATT_PV_CFG.cfg_quant_bias = control_reg_array[33];
    op_cfg_pkt.ATT_PV_CFG.cfg_quant_shift = control_reg_array[34];

    op_cfg_pkt.PROJ_CFG.cfg_acc_num = control_reg_array[35];
    op_cfg_pkt.PROJ_CFG.cfg_quant_scale = control_reg_array[36];
    op_cfg_pkt.PROJ_CFG.cfg_quant_bias = control_reg_array[37];
    op_cfg_pkt.PROJ_CFG.cfg_quant_shift = control_reg_array[38];

    op_cfg_pkt.FFN0_CFG.cfg_acc_num = control_reg_array[39];
    op_cfg_pkt.FFN0_CFG.cfg_quant_scale = control_reg_array[40];
    op_cfg_pkt.FFN0_CFG.cfg_quant_bias = control_reg_array[41];
    op_cfg_pkt.FFN0_CFG.cfg_quant_shift = control_reg_array[42];

    op_cfg_pkt.FFN1_CFG.cfg_acc_num = control_reg_array[43];
    op_cfg_pkt.FFN1_CFG.cfg_quant_scale = control_reg_array[44];
    op_cfg_pkt.FFN1_CFG.cfg_quant_bias = control_reg_array[45];
    op_cfg_pkt.FFN1_CFG.cfg_quant_shift = control_reg_array[46];

    op_cfg_pkt.ATTN_RESIDUAL_CFG.cfg_acc_num = control_reg_array[47];
    op_cfg_pkt.ATTN_RESIDUAL_CFG.cfg_quant_scale = control_reg_array[48];
    op_cfg_pkt.ATTN_RESIDUAL_CFG.cfg_quant_bias = control_reg_array[49];
    op_cfg_pkt.ATTN_RESIDUAL_CFG.cfg_quant_shift = control_reg_array[50];

    op_cfg_pkt.MLP_RESIDUAL_CFG.cfg_acc_num = control_reg_array[51];
    op_cfg_pkt.MLP_RESIDUAL_CFG.cfg_quant_scale = control_reg_array[52];
    op_cfg_pkt.MLP_RESIDUAL_CFG.cfg_quant_bias = control_reg_array[53];
    op_cfg_pkt.MLP_RESIDUAL_CFG.cfg_quant_shift = control_reg_array[54];

end

`endif
USER_CONFIG  usr_cfg_array [`MAX_NUM_USER-1:0];

localparam op_cfg_vld_delay_num = 5;
logic                                                op_cfg_vld_delay5; //leave some time for WMEM to wake up
assign op_cfg_vld_delay5 = op_cfg_vld_delay_gen_array[op_cfg_vld_delay_num-1].op_cfg_vld_delay_temp;
genvar i;
generate;
    for(i = 0; i < op_cfg_vld_delay_num; i=i+1)begin : op_cfg_vld_delay_gen_array
        logic op_cfg_vld_delay_temp;
        if(i==0)begin
            always_ff @(posedge clk or negedge rst_n) begin
                if(~rst_n)begin
                    op_cfg_vld_delay_temp <= 0;
                end
                else begin
                    op_cfg_vld_delay_temp <= op_cfg_vld;
                end
            end
        end
        else begin
            always_ff @(posedge clk or negedge rst_n) begin
                if(~rst_n)begin
                    op_cfg_vld_delay_temp <= 0;
                end
                else begin
                    op_cfg_vld_delay_temp <= op_cfg_vld_delay_gen_array[i-1].op_cfg_vld_delay_temp;
                end
            end
        end
    end
endgenerate



//model_cfg and rc_cfg
always_ff @(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        model_cfg <= 0;
        model_cfg_vld <= 0;
        rc_cfg <= 0;
        rc_cfg_vld <=0;
        pmu_cfg <= 0;
        pmu_cfg_vld <= 0;
    end
    else if(cfg_init_success)begin
        model_cfg <= model_cfg_reg;
        rc_cfg <= rc_cfg_reg;
        rc_cfg_vld <= 1;
        model_cfg_vld <= 1;
        pmu_cfg <= pmu_cfg_reg;
        pmu_cfg_vld <= 1;
    end
    else begin
        rc_cfg_vld <= 0;
        model_cfg_vld <= 0;
        pmu_cfg_vld <= 0;
    end
end




//usr_cfg_array
generate;
    for(i = 0; i < `MAX_NUM_USER; i++)begin
        always_ff @(posedge clk or negedge rst_n)begin
            if(~rst_n)begin
                usr_cfg_array[i].user_token_cnt <= 0;
                usr_cfg_array[i].user_id <= i;
                usr_cfg_array[i].user_kv_cache_not_full <= 1;
            end
            else if(new_token)begin
                if(i == user_id)begin
                    if(user_first_token)begin
                        usr_cfg_array[i].user_kv_cache_not_full <= 1;
                        usr_cfg_array[i].user_token_cnt <= 0;
                        usr_cfg_array[i].user_id <= i;
                    end
                    else begin
                        usr_cfg_array[i].user_id <= i;
                        if(usr_cfg_array[i].user_token_cnt == model_cfg_reg.max_context_length - 1)begin
                            usr_cfg_array[i].user_kv_cache_not_full <= 0;
                            usr_cfg_array[i].user_token_cnt <= 0;
                        end
                        else begin
                            usr_cfg_array[i].user_token_cnt <= usr_cfg_array[i].user_token_cnt + 1;
                        end
                    end
                end
            end
        end
    end
endgenerate


//Configure the user config
logic                         new_token_delay1;
logic                         new_token_delay2;
logic                         new_token_delay3;

logic [`USER_ID_WIDTH-1:0]    current_user_id;
always @(posedge clk or negedge rst_n) begin
    if(~rst_n)begin
        current_user_id <= 0;
    end
    else if(new_token)begin
        current_user_id <= user_id;
    end
end


always_ff @(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        new_token_delay1 <= 0;
        new_token_delay2 <= 0;
        new_token_delay3 <= 0;
    end
    else begin
        new_token_delay1 <= new_token;
        new_token_delay2 <= new_token_delay1;
        new_token_delay3 <= new_token_delay2;
    end
end

always_ff @(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        usr_cfg <= 0;
        usr_cfg_vld <= 0;
    end
    else if (new_token_delay1) begin
        usr_cfg_vld <= 1;
        usr_cfg <= usr_cfg_array[current_user_id];
    end
    else begin
        usr_cfg_vld <= 0;
    end
end

//op config
CONTROL_STATE                                        nxt_control_state;
logic                                                nxt_control_state_update;

logic                                                nxt_op_start; //pulse

OP_CONFIG                                            nxt_op_cfg;
logic                                                nxt_op_cfg_vld;

CONTROL_STATE                                        control_state_delay1;
logic                                                control_state_changed;

logic                                                nxt_current_token_finish;

always_ff@(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        control_state_changed <= 0;
    end
    else begin
        control_state_changed <= (control_state_delay1 != control_state);
    end
end

always_ff @(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        control_state_delay1 <= IDLE_STATE;
    end
    else begin
        control_state_delay1 <= control_state;
    end
end

always_comb begin
    nxt_control_state = control_state;
    nxt_control_state_update = 0;
    nxt_op_start = 0;
    nxt_op_cfg = op_cfg;
    nxt_op_cfg_vld = 0;
    nxt_current_token_finish = 0;
    case (control_state)
        IDLE_STATE : begin
            if(new_token_delay3)begin
                nxt_control_state = Q_GEN_STATE;
                nxt_control_state_update = 1;
            end
        end

        Q_GEN_STATE : begin
            if(control_state_changed) begin
                nxt_op_cfg = op_cfg_pkt.Q_GEN_CFG;
                if(~debug_mode_en)
                    nxt_op_cfg_vld = 1;
                else begin
                    if(debug_mode_bits[0])begin
                        nxt_op_cfg_vld = 1;
                    end
                    else begin
                        nxt_op_cfg_vld = 0;
                        nxt_control_state = K_GEN_STATE;
                        nxt_control_state_update = 1;
                    end
                end 
            end
            if(op_cfg_vld_delay5) begin
                nxt_op_start = 1; 
            end
            if(op_finish) begin
                nxt_control_state = K_GEN_STATE;
                nxt_control_state_update = 1;
            end
        end

        K_GEN_STATE : begin
            if(control_state_changed) begin
                nxt_op_cfg = op_cfg_pkt.K_GEN_CFG;
                if(~debug_mode_en)
                    nxt_op_cfg_vld = 1;
                else begin
                    if(debug_mode_bits[1])begin
                        nxt_op_cfg_vld = 1;
                    end
                    else begin
                        nxt_op_cfg_vld = 0;
                        nxt_control_state = V_GEN_STATE;
                        nxt_control_state_update = 1;
                    end
                end 
            end
            if(op_cfg_vld) begin
                nxt_op_start = 1;
            end
            if(op_finish) begin
                nxt_control_state = V_GEN_STATE;
                nxt_control_state_update = 1;
            end
        end

        V_GEN_STATE : begin
            if(control_state_changed) begin
                nxt_op_cfg = op_cfg_pkt.V_GEN_CFG;
                if(~debug_mode_en)
                    nxt_op_cfg_vld = 1;
                else begin
                    if(debug_mode_bits[2])begin
                        nxt_op_cfg_vld = 1;
                    end
                    else begin
                        nxt_op_cfg_vld = 0;
                        nxt_control_state = ATT_QK_STATE;
                        nxt_control_state_update = 1;
                    end
                end 
            end
            if(op_cfg_vld) begin
                nxt_op_start = 1;
            end
            if(op_finish) begin
                nxt_control_state = ATT_QK_STATE;
                nxt_control_state_update = 1;
            end
        end

        ATT_QK_STATE : begin
            if(control_state_changed) begin
                nxt_op_cfg = op_cfg_pkt.ATT_QK_CFG;
                if(~debug_mode_en)
                    nxt_op_cfg_vld = 1;
                else begin
                    if(debug_mode_bits[3])begin
                        nxt_op_cfg_vld = 1;
                    end
                    else begin
                        nxt_op_cfg_vld = 0;
                        nxt_control_state = ATT_PV_STATE;
                        nxt_control_state_update = 1;
                    end
                end 
            end
            if(op_cfg_vld) begin
                nxt_op_start = 1;
            end
            if(op_finish) begin
                nxt_control_state = ATT_PV_STATE;
                nxt_control_state_update = 1;
            end
        end

        ATT_PV_STATE : begin
            if(control_state_changed) begin
                nxt_op_cfg = op_cfg_pkt.ATT_PV_CFG;
                if(usr_cfg.user_kv_cache_not_full)
                    nxt_op_cfg.cfg_acc_num = usr_cfg.user_token_cnt/`MAC_MULT_NUM + 1;
                else
                    nxt_op_cfg.cfg_acc_num = model_cfg_reg.max_context_length/`MAC_MULT_NUM;

                if(~debug_mode_en)
                    nxt_op_cfg_vld = 1;
                else begin
                    if(debug_mode_bits[4])begin
                        nxt_op_cfg_vld = 1;
                    end
                    else begin
                        nxt_op_cfg_vld = 0;
                        nxt_control_state = PROJ_STATE;
                        nxt_control_state_update = 1;
                    end
                end 
            end
            if(op_cfg_vld) begin
                nxt_op_start = 1;
            end
            if(op_finish) begin
                nxt_control_state = PROJ_STATE;
                nxt_control_state_update = 1;
            end
        end

        PROJ_STATE : begin
            if(control_state_changed) begin
                nxt_op_cfg = op_cfg_pkt.PROJ_CFG;
                if(~debug_mode_en)
                    nxt_op_cfg_vld = 1;
                else begin
                    if(debug_mode_bits[5])begin
                        nxt_op_cfg_vld = 1;
                    end
                    else begin
                        nxt_op_cfg_vld = 0;
                        nxt_control_state = FFN0_STATE;
                        nxt_control_state_update = 1;
                    end
                end 
            end
            if(op_cfg_vld) begin
                nxt_op_start = 1;
            end
            if(op_finish) begin
                nxt_control_state = FFN0_STATE;
                nxt_control_state_update = 1;
            end
        end
    
        FFN0_STATE : begin
            if(control_state_changed) begin
                nxt_op_cfg = op_cfg_pkt.FFN0_CFG;
                if(~debug_mode_en)
                    nxt_op_cfg_vld = 1;
                else begin
                    if(debug_mode_bits[6])begin
                        nxt_op_cfg_vld = 1;
                    end
                    else begin
                        nxt_op_cfg_vld = 0;
                        nxt_control_state = FFN1_STATE;
                        nxt_control_state_update = 1;
                    end
                end 
            end
            if(op_cfg_vld_delay5) begin
                nxt_op_start = 1;
            end
            if(op_finish) begin
                nxt_control_state = FFN1_STATE;
                nxt_control_state_update = 1;
            end
        end

        FFN1_STATE : begin
            if(control_state_changed) begin
                nxt_op_cfg = op_cfg_pkt.FFN1_CFG;
                if(~debug_mode_en)
                    nxt_op_cfg_vld = 1;
                else begin
                    if(debug_mode_bits[7])begin
                        nxt_op_cfg_vld = 1;
                    end
                    else begin
                        nxt_op_cfg_vld = 0;
                        if(power_mode_en)
                            nxt_control_state = Q_GEN_STATE;
                        else
                            nxt_control_state = IDLE_STATE;
                        nxt_control_state_update = 1;
                        nxt_current_token_finish = 1;
                    end
                end 
            end
            if(op_cfg_vld) begin
                nxt_op_start = 1;
            end
            if(op_finish) begin
                if(power_mode_en)
                    nxt_control_state = Q_GEN_STATE;
                else
                    nxt_control_state = IDLE_STATE;
                nxt_control_state_update = 1;
                nxt_current_token_finish = 1;
            end
        end

    endcase

end



always_ff @(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        control_state <= IDLE_STATE;
        control_state_update <= 0;
        op_start <= 0;
        op_cfg <= 0;
        op_cfg_vld <= 0;
        current_token_finish <= 0;
    end
    else begin
        control_state <= nxt_control_state;
        control_state_update <= nxt_control_state_update;
        op_start <= nxt_op_start;
        op_cfg <= nxt_op_cfg;
        op_cfg_vld <= nxt_op_cfg_vld;
        current_token_finish <= nxt_current_token_finish;
    end
end

endmodule
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

module array_top (
    input  logic                                                clk,
    input  logic                                                rst_n,

    //There should be the ports to initial the cfg, wmem and input data from SPI or someting else
    input  logic       [`INTERFACE_ADDR_WIDTH-1:0]              interface_addr, //保证进来的时序是干净的

    input  logic                                                interface_wen,
    input  logic       [`INTERFACE_DATA_WIDTH-1:0]              interface_wdata,

    input  logic                                                interface_ren,
    output logic       [`INTERFACE_DATA_WIDTH-1:0]              interface_rdata,
    output logic                                                interface_rvalid,


    //STATE PIN
    output logic                                                current_token_finish_flag,
    output logic                                                current_token_finish_work,

    output logic                                                qgen_state_work,
    output logic                                                qgen_state_end,

    output logic                                                kgen_state_work,
    output logic                                                kgen_state_end,

    output logic                                                vgen_state_work,
    output logic                                                vgen_state_end,

    output logic                                                att_qk_state_work,
    output logic                                                att_qk_state_end,

    output logic                                                att_pv_state_work,
    output logic                                                att_pv_state_end,

    output logic                                                proj_state_work,
    output logic                                                proj_state_end,

    output logic                                                ffn0_state_work,
    output logic                                                ffn0_state_end,

    output logic                                                ffn1_state_work,
    output logic                                                ffn1_state_end
);

  
logic                                                clean_kv_cache; //pulse
logic       [`USER_ID_WIDTH-1:0]                     clean_kv_cache_user_id; //clean the corrsponding user's kv cache to zero


logic                                                new_token; //pulse, should be high after global sram accept all the content of the new token
logic                                                user_first_token; //be in same phase
logic       [`USER_ID_WIDTH-1:0]                     user_id;// be in same phase

logic                                                current_token_finish;

logic                                                cfg_init_success; //pulse, when high means that all the config all initial successfully

logic                                                            control_state_update;
CONTROL_STATE                                                    control_state_delay1;//delay consideration
CONTROL_STATE                                                    control_state;


logic       power_mode_en_in;
logic       debug_mode_en_in;
logic [7:0] debug_mode_bits_in;

//state pin
always_ff@(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        current_token_finish_flag <= 1;
    end
    else if(new_token)begin
        current_token_finish_flag <= 0;
    end
    else if(current_token_finish)begin
        current_token_finish_flag <= 1;
    end
end

always_ff@(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        current_token_finish_work <= 0;
    end
    else begin
        current_token_finish_work <= (control_state_delay1 != IDLE_STATE);
    end
end


logic nxt_qgen_state_work;
logic nxt_qgen_state_end;

logic nxt_kgen_state_work;
logic nxt_kgen_state_end;

logic nxt_vgen_state_work;
logic nxt_vgen_state_end;

logic nxt_att_qk_state_work;
logic nxt_att_qk_state_end;

logic nxt_att_pv_state_work;
logic nxt_att_pv_state_end;

logic nxt_proj_state_work;
logic nxt_proj_state_end;

logic nxt_ffn0_state_work;
logic nxt_ffn0_state_end;

logic nxt_ffn1_state_work;
logic nxt_ffn1_state_end;

always_comb begin
    nxt_qgen_state_work = (control_state_delay1 == Q_GEN_STATE);
    nxt_kgen_state_work = (control_state_delay1 == K_GEN_STATE);
    nxt_vgen_state_work = (control_state_delay1 == V_GEN_STATE);
    nxt_att_qk_state_work = (control_state_delay1 == ATT_QK_STATE);
    nxt_att_pv_state_work = (control_state_delay1 == ATT_PV_STATE);
    nxt_proj_state_work = (control_state_delay1 == PROJ_STATE);
    nxt_ffn0_state_work = (control_state_delay1 == FFN0_STATE);
    nxt_ffn1_state_work = (control_state_delay1 == FFN1_STATE);

    nxt_qgen_state_end = qgen_state_end;
    if(control_state == K_GEN_STATE && control_state_delay1 == Q_GEN_STATE)begin
        nxt_qgen_state_end = 1;
    end
    else if(new_token)begin
        nxt_qgen_state_end = 0;
    end


    nxt_kgen_state_end = kgen_state_end;
    if(control_state == V_GEN_STATE && control_state_delay1 == K_GEN_STATE)begin
        nxt_kgen_state_end = 1;
    end
    else if(new_token)begin
        nxt_kgen_state_end = 0;
    end

    nxt_vgen_state_end = vgen_state_end;
    if(control_state == ATT_QK_STATE && control_state_delay1 == V_GEN_STATE)begin
        nxt_vgen_state_end = 1;
    end
    else if(new_token)begin
        nxt_vgen_state_end = 0;
    end

    nxt_att_qk_state_end = att_qk_state_end;
    if(control_state == ATT_PV_STATE && control_state_delay1 == ATT_QK_STATE)begin
        nxt_att_qk_state_end = 1;
    end
    else if(new_token)begin
        nxt_att_qk_state_end = 0;
    end

    nxt_att_pv_state_end = att_pv_state_end;
    if(control_state == PROJ_STATE && control_state_delay1 == ATT_PV_STATE)begin
        nxt_att_pv_state_end = 1;
    end
    else if(new_token)begin
        nxt_att_pv_state_end = 0;
    end

    nxt_proj_state_end = proj_state_end;
    if(control_state == FFN0_STATE && control_state_delay1 == PROJ_STATE)begin
        nxt_proj_state_end = 1;
    end
    else if(new_token)begin
        nxt_proj_state_end = 0;
    end

    nxt_ffn0_state_end = ffn0_state_end;
    if(control_state == FFN1_STATE && control_state_delay1 == FFN0_STATE)begin
        nxt_ffn0_state_end = 1;
    end
    else if(new_token)begin
        nxt_ffn0_state_end = 0;
    end

    nxt_ffn1_state_end = ffn1_state_end;
    if(control_state == IDLE_STATE && control_state_delay1 == FFN1_STATE)begin
        nxt_ffn1_state_end = 1;
    end
    else if(new_token)begin
        nxt_ffn1_state_end = 0;
    end
end






always_ff @(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        qgen_state_work <= 0;
        qgen_state_end <= 0;

        kgen_state_work <= 0;
        kgen_state_end <= 0;

        vgen_state_work <= 0;
        vgen_state_end <= 0;

        att_qk_state_work <= 0;
        att_qk_state_end <= 0;

        att_pv_state_work <= 0;
        att_pv_state_end <= 0;

        proj_state_work <= 0;
        proj_state_end <= 0;

        ffn0_state_work <= 0;
        ffn0_state_end <= 0;

        ffn1_state_work <= 0;
        ffn1_state_end <= 0;
    end
    else begin
        qgen_state_work <= nxt_qgen_state_work;
        qgen_state_end <= nxt_qgen_state_end;

        kgen_state_work <= nxt_kgen_state_work;
        kgen_state_end <= nxt_kgen_state_end;

        vgen_state_work <= nxt_vgen_state_work;
        vgen_state_end <= nxt_vgen_state_end;

        att_qk_state_work <= nxt_att_qk_state_work;
        att_qk_state_end <= nxt_att_qk_state_end;

        att_pv_state_work <= nxt_att_pv_state_work;
        att_pv_state_end <= nxt_att_pv_state_end;

        proj_state_work <= nxt_proj_state_work;
        proj_state_end <= nxt_proj_state_end;

        ffn0_state_work <= nxt_ffn0_state_work;
        ffn0_state_end <= nxt_ffn0_state_end;

        ffn1_state_work <= nxt_ffn1_state_work;
        ffn1_state_end <= nxt_ffn1_state_end;
    end
end



`ifdef INTERFACE_ENABLE
////////////////////////////////////
//       INSTRUCTION REG          //
////////////////////////////////////

logic      [`INTERFACE_DATA_WIDTH-1:0]               instruction_in_data_0; // {user_id, user_first_token, new_token}, needs to be set to 0 after each new token for the next rising edge detection
logic      [`INTERFACE_DATA_WIDTH-1:0]               instruction_in_data_1; // {user_id, clean_kv_cache}, needs to be set to 0 after each kv cache clean for the next rising edge detection
logic      [`INTERFACE_DATA_WIDTH-1:0]               instruction_in_data_2; // {cfg_init_success}, needs to be set to 0 after each configuration initialization for the next rising edge detection
logic      [`INTERFACE_DATA_WIDTH-1:0]               instruction_in_data_3; // {debug_mode_bits, debug_mode, power_mode}


logic signed [`INTERFACE_ADDR_WIDTH + 2-1:0]             interface_instruction_bias_addr;
logic [`INSTRUCTION_REGISTERS_ADDR_WIDTH-1:0]            instruction_reg_addr;
logic [`INSTRUCTION_REGISTERS_DATA_WIDTH-1:0]            instruction_reg_wdata;
logic                                                    instruction_reg_wen;
logic [`INSTRUCTION_REGISTERS_DATA_WIDTH-1:0]            instruction_reg_rdata;
logic                                                    instruction_reg_ren;
logic                                                    instruction_reg_rvld;

always_comb begin 
    interface_instruction_bias_addr = {1'b0,interface_addr}-{1'b0,`INSTRUCTION_REGISTERS_BASE_ADDR};
    instruction_reg_addr = interface_instruction_bias_addr[`INSTRUCTION_REGISTERS_ADDR_WIDTH-1:0];
    instruction_reg_wdata = interface_wdata;
    instruction_reg_wen = interface_wen && 
                    ($signed(interface_instruction_bias_addr) >= 0 && 
                    $signed(interface_instruction_bias_addr) <= (2 ** `INSTRUCTION_REGISTERS_ADDR_WIDTH)-1);
    instruction_reg_ren = interface_ren && 
                    ($signed(interface_instruction_bias_addr) >= 0 && 
                    $signed(interface_instruction_bias_addr) <= (2 ** `INSTRUCTION_REGISTERS_ADDR_WIDTH)-1);
end

always_ff @(posedge clk or negedge rst_n)begin //write
    if(~rst_n)begin
        instruction_in_data_0 <= 0;
        instruction_in_data_1 <= 0;
        instruction_in_data_2 <= 0;
        instruction_in_data_3 <= 0;
    end
    else if(instruction_reg_wen)begin
        case (instruction_reg_addr)
            2'b00 : instruction_in_data_0 <= instruction_reg_wdata;
            2'b01 : instruction_in_data_1 <= instruction_reg_wdata;
            2'b10 : instruction_in_data_2 <= instruction_reg_wdata;
            2'b11 : instruction_in_data_3 <= instruction_reg_wdata;
        endcase
    end
    else begin //当前token结束，清空instruction

        if(new_token)
            instruction_in_data_0 <= 0;

        if(clean_kv_cache)
            instruction_in_data_1 <= 0;

        if(cfg_init_success)
            instruction_in_data_2 <= 0;

    end
end

always_ff @(posedge clk or negedge rst_n)begin //read
    if(~rst_n)begin
        instruction_reg_rvld <= 0;
        instruction_reg_rdata <= 0;
    end
    else if(instruction_reg_ren)begin
        instruction_reg_rvld <= 1;
        case (instruction_reg_addr)
            2'b00 : instruction_reg_rdata <= instruction_in_data_0;
            2'b01 : instruction_reg_rdata <= instruction_in_data_1;
            2'b10 : instruction_reg_rdata <= instruction_in_data_2;
            2'b11 : instruction_reg_rdata <= instruction_in_data_3;
        endcase
    end
    else begin
        instruction_reg_rvld <= 0;
    end
end

logic      [`INTERFACE_DATA_WIDTH-1:0]               instruction_in_data_0_delay1;
logic      [`INTERFACE_DATA_WIDTH-1:0]               instruction_in_data_1_delay1;
logic      [`INTERFACE_DATA_WIDTH-1:0]               instruction_in_data_2_delay1;

always_ff@(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        instruction_in_data_0_delay1 <= 0;
        instruction_in_data_1_delay1 <= 0;
        instruction_in_data_2_delay1 <= 0;
    end
    else begin
        instruction_in_data_0_delay1 <= instruction_in_data_0;
        instruction_in_data_1_delay1 <= instruction_in_data_1;
        instruction_in_data_2_delay1 <= instruction_in_data_2;
    end
end


///////////////////////////////////////
//             new token             //
///////////////////////////////////////

 logic                                                nxt_new_token; //pulse, should be high after global sram accept all the content of the new token
 logic                                                nxt_user_first_token; //be in same phase
 logic       [`USER_ID_WIDTH-1:0]                     nxt_user_id;// be in same phase

always_comb begin
    nxt_new_token = 0;
    nxt_user_first_token = 0;
    nxt_user_id = instruction_in_data_0[2 + `USER_ID_WIDTH - 1: 2];

    if(new_token)begin
        nxt_new_token = 0; //pulse
        nxt_user_first_token = 0;
    end
    else begin
        nxt_new_token =        instruction_in_data_0[0] & (~instruction_in_data_0_delay1[0]);//catch up the rising edge
        nxt_user_first_token = instruction_in_data_0[1] & (~instruction_in_data_0_delay1[1]);//catch up the rising edge       
    end
end

always_ff @(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        new_token <= 0; 
        user_first_token <= 0; 
        user_id <= 0;
    end
    else begin
        new_token <= nxt_new_token; 
        user_first_token <= nxt_user_first_token; 
        user_id <= nxt_user_id;
    end
end


///////////////////////////////////////
//           clean kv cache          //
///////////////////////////////////////

//instruction register
logic                        nxt_clean_kv_cache;
logic [`USER_ID_WIDTH-1:0]   nxt_clean_kv_cache_user_id;
always_comb begin
    nxt_clean_kv_cache = 0;
    nxt_clean_kv_cache_user_id = instruction_in_data_1[1 + `USER_ID_WIDTH - 1: 1];

    if(clean_kv_cache)begin
        nxt_clean_kv_cache = 0; //pulse
    end
    else begin
        nxt_clean_kv_cache = instruction_in_data_1[0] & (~instruction_in_data_1_delay1[0]);//catch up the rising edge
    end
end


always_ff@(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        clean_kv_cache <= 0;
        clean_kv_cache_user_id <= 0;
    end
    else begin
        clean_kv_cache <= nxt_clean_kv_cache;
        clean_kv_cache_user_id <= nxt_clean_kv_cache_user_id;
    end
end



///////////////////////////////////////
//          cfg_init_success         //
///////////////////////////////////////
logic nxt_cfg_init_success;
always_comb begin
    nxt_cfg_init_success = 0;
    if(cfg_init_success)begin
        nxt_cfg_init_success = 0; //pulse
    end
    else begin
        nxt_cfg_init_success =  instruction_in_data_2[0] & (~instruction_in_data_2_delay1[0]);//catch up the rising edge   
    end
end

always_ff @(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        cfg_init_success <= 0;
    end
    else begin
        cfg_init_success <= nxt_cfg_init_success;
    end
end


///////////////////////////////////////
//             power mode            //
///////////////////////////////////////
always_ff @(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        power_mode_en_in <= 0;
    end
    else begin
        power_mode_en_in <= instruction_in_data_3[0];
    end
end

///////////////////////////////////////
//             debug mode            //
///////////////////////////////////////
always_ff @(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        debug_mode_en_in <= 0;
        debug_mode_bits_in <= 0;
    end
    else begin
        debug_mode_en_in <= instruction_in_data_3[1];
        debug_mode_bits_in <= instruction_in_data_3[9:2];
    end
end




////////////////////////////////////
//           STATE REG            //
////////////////////////////////////
logic      [`INTERFACE_DATA_WIDTH-1:0]               state_out_data_0; //current_token_finish (equal to 1 when curernt token finish, reset to zero when new token)
logic      [`INTERFACE_DATA_WIDTH-1:0]               state_out_data_1; //control state
logic      [`INTERFACE_DATA_WIDTH-1:0]               state_out_data_2; //reserved
logic      [`INTERFACE_DATA_WIDTH-1:0]               state_out_data_3; //reserved

logic signed [`INTERFACE_ADDR_WIDTH + 2-1:0]             interface_state_bias_addr;
logic [`STATE_REGISTERS_ADDR_WIDTH-1:0]                  state_reg_addr;
logic [`STATE_REGISTERS_DATA_WIDTH-1:0]                  state_reg_wdata;
logic                                                    state_reg_wen;
logic [`STATE_REGISTERS_DATA_WIDTH-1:0]                  state_reg_rdata;
logic                                                    state_reg_ren;
logic                                                    state_reg_rvld;

always_comb begin 
    interface_state_bias_addr = {1'b0,interface_addr}-{1'b0,`STATE_REGISTERS_BASE_ADDR};
    state_reg_addr = interface_state_bias_addr[`STATE_REGISTERS_ADDR_WIDTH-1:0];
    state_reg_wdata = interface_wdata;
    state_reg_wen = interface_wen && 
                    ($signed(interface_state_bias_addr) >= 0 && 
                    $signed(interface_state_bias_addr) <= (2 ** `STATE_REGISTERS_ADDR_WIDTH)-1);
    state_reg_ren = interface_ren && 
                    ($signed(interface_state_bias_addr) >= 0 && 
                    $signed(interface_state_bias_addr) <= (2 ** `STATE_REGISTERS_ADDR_WIDTH)-1);
end

always_ff @(posedge clk or negedge rst_n)begin //write
    if(~rst_n)begin
        state_out_data_0 <= 0;
        state_out_data_1 <= 0;
        state_out_data_2 <= 0;
        state_out_data_3 <= 0;
    end
    else if(new_token)begin //当前token开始，清空state
        state_out_data_0 <= 0;
        state_out_data_1 <= 0;
        state_out_data_2 <= 0;
        state_out_data_3 <= 0;
    end
    else if(state_reg_wen)begin
        case (state_reg_addr)
            2'b00 : state_out_data_0 <= state_reg_wdata;
            2'b01 : state_out_data_1 <= state_reg_wdata;
            2'b10 : state_out_data_2 <= state_reg_wdata;
            2'b11 : state_out_data_3 <= state_reg_wdata;
        endcase
    end
    else begin
        if(current_token_finish)
            state_out_data_0 <= 1;
        else if(new_token)
            state_out_data_0 <= 0;

        if(control_state_update)
            state_out_data_1 <= control_state;
    end
end

always_ff @(posedge clk or negedge rst_n)begin //read
    if(~rst_n)begin
        state_reg_rvld <= 0;
        state_reg_rdata <= 0;
    end
    else if(state_reg_ren)begin
        state_reg_rvld <= 1;
        case (state_reg_addr)
            2'b00 : state_reg_rdata <= state_out_data_0;
            2'b01 : state_reg_rdata <= state_out_data_1;
            2'b10 : state_reg_rdata <= state_out_data_2;
            2'b11 : state_reg_rdata <= state_out_data_3;
        endcase
    end
    else begin
        state_reg_rvld <= 0;
    end
end


////////////////////////////////////
//          CONTROL REG           //
////////////////////////////////////

logic [`CONTROL_REGISTERS_ADDR_WIDTH-1:0]    control_reg_addr;
logic signed [`INTERFACE_ADDR_WIDTH + 2-1:0] interface_control_bias_addr;
logic [`CONTROL_REGISTERS_DATA_WIDTH-1:0]    control_reg_wdata;
logic                                        control_reg_wen;
logic [`CONTROL_REGISTERS_DATA_WIDTH-1:0]    control_reg_rdata;
logic                                        control_reg_ren;
logic                                        control_reg_rvld;

always_comb begin 
    interface_control_bias_addr = {1'b0,interface_addr}-{1'b0,`CONTROL_REGISTERS_BASE_ADDR};
    control_reg_addr = interface_control_bias_addr[`CONTROL_REGISTERS_ADDR_WIDTH-1:0];
    control_reg_wdata = interface_wdata;
    control_reg_wen = interface_wen && 
                    ($signed(interface_control_bias_addr) >= 0 && 
                    $signed(interface_control_bias_addr) <= (2 ** `CONTROL_REGISTERS_ADDR_WIDTH)-1);
    control_reg_ren = interface_ren && 
                    ($signed(interface_control_bias_addr) >= 0 && 
                    $signed(interface_control_bias_addr) <= (2 ** `CONTROL_REGISTERS_ADDR_WIDTH)-1);
end

`endif


logic [$clog2(`GLOBAL_SRAM_DEPTH)-1:0]                           global_sram_waddr; 
logic                                                            global_sram_wen;
logic [(`MAC_MULT_NUM * `IDATA_WIDTH)-1:0]                       global_sram_bwe;  
logic [(`MAC_MULT_NUM * `IDATA_WIDTH)-1:0]                       global_sram_wdata;

logic [$clog2(`GLOBAL_SRAM_DEPTH)-1:0]                           global_sram_waddr_delay1; //critical path consideration
logic                                                            global_sram_wen_delay1;
logic [(`MAC_MULT_NUM * `IDATA_WIDTH)-1:0]                       global_sram_bwe_delay1;  
logic [(`MAC_MULT_NUM * `IDATA_WIDTH)-1:0]                       global_sram_wdata_delay1;

always_comb begin
    global_sram_bwe = {(`MAC_MULT_NUM * `IDATA_WIDTH){1'b1}};
end



always_ff @(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        global_sram_waddr_delay1 <= 0;
        global_sram_wen_delay1 <= 0;
        global_sram_wdata_delay1 <= 0;
        global_sram_bwe_delay1 <= 0;
    end
    else begin
        global_sram_waddr_delay1 <= global_sram_waddr;
        global_sram_wen_delay1 <= global_sram_wen;
        global_sram_wdata_delay1 <= global_sram_wdata;
        global_sram_bwe_delay1 <= global_sram_bwe;
    end
end



logic [$clog2(`GLOBAL_SRAM_DEPTH)-1:0]                           global_sram_raddr;
logic                                                            global_sram_ren;
logic                                                            global_sram_rvld;
logic [(`MAC_MULT_NUM * `IDATA_WIDTH)-1:0]                       global_sram_rdata;

logic  [$clog2(`MAC_MULT_NUM) + $clog2(`GLOBAL_SRAM_DEPTH)-1:0]  residual_sram_waddr;
logic                                                            residual_sram_wen;
logic  [(`MAC_MULT_NUM * `IDATA_WIDTH)-1:0]                      residual_sram_wdata;
logic                                                            residual_sram_wdata_byte_flag;

logic   [$clog2(`MAC_MULT_NUM)+$clog2(`GLOBAL_SRAM_DEPTH)-1:0]   residual_sram_raddr;  
logic   [$clog2(`MAC_MULT_NUM)+$clog2(`GLOBAL_SRAM_DEPTH)-1:0]   residual_sram_raddr_delay1;            
logic                                                            residual_sram_ren;
logic         [(`MAC_MULT_NUM * `IDATA_WIDTH)-1:0]               residual_sram_rdata;
logic                                                            residual_sram_rvld;

logic         [(`MAC_MULT_NUM * `IDATA_WIDTH)-1:0]               head_input_sram_rdata;
logic                                                            head_input_sram_rdata_vld;

logic                                                            op_start_delay1;
logic                                                            global_sram_finish;




logic                                                            op_start;
logic                                                            residual_finish;
logic                                                            head_array_finish;
logic                                                            op_finish;

OP_CONFIG                                                        op_cfg;
logic                                                            op_cfg_vld;

USER_CONFIG                                                      usr_cfg;
logic                                                            usr_cfg_vld;

MODEL_CONFIG                                                     model_cfg;
logic                                                            model_cfg_vld;

PMU_CONFIG                                                       pmu_cfg;
logic                                                            pmu_cfg_vld;

logic                                                            rc_cfg_vld;
RC_CONFIG                                                        rc_cfg;

BUS_ADDR   [`HEAD_NUM-1:0]                                       gbus_addr_delay1_array;
logic      [`HEAD_NUM-1:0]                                       gbus_wen_delay1_array;
logic      [`HEAD_NUM-1:0][`GBUS_DATA_WIDTH-1:0]                 gbus_wdata_delay1_array;    

BUS_ADDR   [`HEAD_NUM-1:0]                                       gbus_addr_delay2_array;
logic      [`HEAD_NUM-1:0]                                       gbus_wen_delay2_array;
logic      [`HEAD_NUM-1:0][`GBUS_DATA_WIDTH-1:0]                 gbus_wdata_delay2_array;    

always_ff @(posedge clk or negedge rst_n) begin
    if(~rst_n)begin
        gbus_addr_delay2_array <= 0;
        gbus_wen_delay2_array <= 0;
        gbus_wdata_delay2_array <= 0;    
    end
    else begin
        gbus_addr_delay2_array <= gbus_addr_delay1_array;
        gbus_wen_delay2_array <= gbus_wen_delay1_array;
        gbus_wdata_delay2_array <= gbus_wdata_delay1_array;    
    end
end

always_ff @(posedge clk or negedge rst_n ) begin
    if(~rst_n)begin
        op_start_delay1 <= 0;
    end
    else begin
        op_start_delay1 <= op_start;
    end
end

always_comb begin
    if(control_state_delay1 == FFN0_STATE)begin
        head_input_sram_rdata_vld = global_sram_wen;
        head_input_sram_rdata = global_sram_wdata;
    end
    else if(control_state_delay1 == FFN1_STATE)begin
        head_input_sram_rdata_vld = 0;
        head_input_sram_rdata = 0;
    end
    else begin
        head_input_sram_rdata_vld = global_sram_rvld;
        head_input_sram_rdata = global_sram_rdata;
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

////////////////////////////////////
//           ARRAY MEM            //
////////////////////////////////////

`ifdef INTERFACE_ENABLE
//打一拍
logic [`ARRAY_MEM_ADDR_WIDTH-1:0]               array_mem_addr; //bias 是 0
logic [`INTERFACE_DATA_WIDTH-1:0]               array_mem_wdata;
logic                                           array_mem_wen;
logic [`INTERFACE_DATA_WIDTH-1:0]               array_mem_rdata;
logic                                           array_mem_ren;
logic                                           array_mem_rvld;

logic [`ARRAY_MEM_ADDR_WIDTH-1:0]               nxt_array_mem_addr; //bias 是 0
logic [`INTERFACE_DATA_WIDTH-1:0]               nxt_array_mem_wdata;
logic                                           nxt_array_mem_wen;

logic                                           nxt_array_mem_ren;

always_comb begin 
    nxt_array_mem_addr = interface_addr;
    nxt_array_mem_wdata = interface_wdata;             //head sram       //kv cache and wmem
    nxt_array_mem_wen = interface_wen && (interface_addr <  2048     +       (1<<21));
    nxt_array_mem_ren = interface_ren && (interface_addr <  2048     +       (1<<21));
end


always_ff@(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        array_mem_addr <= 0;
        array_mem_wdata <= 0;
        array_mem_wen <= 0;
        array_mem_ren <= 0;
    end
    else begin
        array_mem_addr <= nxt_array_mem_addr;
        array_mem_wdata <= nxt_array_mem_wdata;
        array_mem_wen <= nxt_array_mem_wen;
        array_mem_ren <= nxt_array_mem_ren;
    end
end



/////////////////////////////////////
//           GLOBAL MEM            //
/////////////////////////////////////
logic [`GLOBAL_MEM_ADDR_WIDTH-1:0]              global_mem_addr;
logic [`GLOBAL_MEM_DATA_WIDTH-1:0]              global_mem_wdata;
logic                                           global_mem_wen;
logic [`GLOBAL_MEM_DATA_WIDTH-1:0]              global_mem_rdata;
logic                                           global_mem_ren;
logic                                           global_mem_rvld;

logic signed [`INTERFACE_ADDR_WIDTH + 2-1:0]    interface_global_bias_addr;

logic [`GLOBAL_MEM_ADDR_WIDTH-1:0]              nxt_global_mem_addr;
logic [`GLOBAL_MEM_DATA_WIDTH-1:0]              nxt_global_mem_wdata;
logic                                           nxt_global_mem_wen;
logic                                           nxt_global_mem_ren;

always_comb begin 
    interface_global_bias_addr = {1'b0,interface_addr}-{1'b0,`GLOBAL_MEM_BASE_ADDR};
    nxt_global_mem_addr = interface_global_bias_addr[`GLOBAL_MEM_ADDR_WIDTH-1:0];
    nxt_global_mem_wdata = interface_wdata;
    nxt_global_mem_wen = interface_wen && 
                    ($signed(interface_global_bias_addr) >= 0 && 
                    $signed(interface_global_bias_addr) <= (2 ** `GLOBAL_MEM_ADDR_WIDTH)-1);
    nxt_global_mem_ren = interface_ren && 
                    ($signed(interface_global_bias_addr) >= 0 && 
                    $signed(interface_global_bias_addr) <= (2 ** `GLOBAL_MEM_ADDR_WIDTH)-1);
end


always_ff @(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        global_mem_addr <= 0;
        global_mem_wdata <= 0;
        global_mem_wen <= 0;
        global_mem_ren <= 0;
    end
    else begin
        global_mem_addr <= nxt_global_mem_addr;
        global_mem_wdata <= nxt_global_mem_wdata;
        global_mem_wen <= nxt_global_mem_wen;
        global_mem_ren <= nxt_global_mem_ren;
    end
end

///////////////////////////////////////
//           RESIDUAL MEM            //
///////////////////////////////////////
logic [`RESIDUAL_MEM_ADDR_WIDTH-1:0]              residual_mem_addr;
logic [`RESIDUAL_MEM_DATA_WIDTH-1:0]              residual_mem_wdata;
logic                                             residual_mem_wen;
logic [`RESIDUAL_MEM_DATA_WIDTH-1:0]              residual_mem_rdata;
logic                                             residual_mem_ren;
logic                                             residual_mem_rvld;

logic signed [`INTERFACE_ADDR_WIDTH + 2-1:0]      interface_residual_bias_addr;

logic [`RESIDUAL_MEM_DATA_WIDTH-1:0]              nxt_residual_mem_addr;
logic [`RESIDUAL_MEM_DATA_WIDTH-1:0]              nxt_residual_mem_wdata;
logic                                             nxt_residual_mem_wen;
logic                                             nxt_residual_mem_ren;

always_comb begin 
    interface_residual_bias_addr = {1'b0,interface_addr}-{1'b0,`RESIDUAL_MEM_BASE_ADDR};
    nxt_residual_mem_addr = interface_residual_bias_addr[`RESIDUAL_MEM_ADDR_WIDTH-1:0];
    nxt_residual_mem_wdata = interface_wdata;
    nxt_residual_mem_wen = interface_wen && 
                    ($signed(interface_residual_bias_addr) >= 0 && 
                    $signed(interface_residual_bias_addr) <= (2 ** `RESIDUAL_MEM_ADDR_WIDTH)-1);
    nxt_residual_mem_ren = interface_ren && 
                    ($signed(interface_residual_bias_addr) >= 0 && 
                    $signed(interface_residual_bias_addr) <= (2 ** `RESIDUAL_MEM_ADDR_WIDTH)-1);
end


always_ff @(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        residual_mem_addr <= 0;
        residual_mem_wdata <= 0;
        residual_mem_wen <= 0;
        residual_mem_ren <= 0;
    end
    else begin
        residual_mem_addr <=  nxt_residual_mem_addr;
        residual_mem_wdata <= nxt_residual_mem_wdata;
        residual_mem_wen <=   nxt_residual_mem_wen;
        residual_mem_ren <=   nxt_residual_mem_ren;
    end
end




always_comb begin
    unique if(control_reg_rvld) begin //unique 互斥
        interface_rvalid = 1;
        interface_rdata = control_reg_rdata;
    end
    else if(state_reg_rvld)begin
        interface_rvalid = 1;
        interface_rdata = state_reg_rdata;
    end
    else if(instruction_reg_rvld)begin
        interface_rvalid = 1;
        interface_rdata = instruction_reg_rdata;
    end
    else if(array_mem_rvld)begin
        interface_rvalid = 1;
        interface_rdata = array_mem_rdata;
    end
    else if(global_mem_rvld)begin
        interface_rvalid = 1;
        interface_rdata = global_mem_rdata;
    end
    else if(residual_mem_rvld)begin
        interface_rvalid = 1;
        interface_rdata = residual_mem_rdata;
    end
    else begin
        interface_rdata = 0;
        interface_rvalid = 0;
    end
end






`endif

head_array inst_head_array(
    .clk(clk),
    .rst_n(rst_n),

    .clean_kv_cache(clean_kv_cache_delay1),
    .clean_kv_cache_user_id(clean_kv_cache_user_id_delay1),

`ifdef INTERFACE_ENABLE
    .array_mem_addr(array_mem_addr), //bias 是 0
    .array_mem_wdata(array_mem_wdata),
    .array_mem_wen(array_mem_wen),
    .array_mem_rdata(array_mem_rdata),
    .array_mem_ren(array_mem_ren),
    .array_mem_rvld(array_mem_rvld),
`else
    .array_mem_addr(22'b0), //bias 是 0
    .array_mem_wdata(16'b0),
    .array_mem_wen(1'b0),
    .array_mem_rdata(),
    .array_mem_ren(1'b0),
    .array_mem_rvld(),
`endif

    .global_sram_rvld(head_input_sram_rdata_vld),
    .global_sram_rdata(head_input_sram_rdata),

    .control_state(control_state),
    .control_state_update(control_state_update),

    .start(op_start),
    .finish(head_array_finish),

    .op_cfg(op_cfg),
    .op_cfg_vld(op_cfg_vld),

    .pmu_cfg_vld(pmu_cfg_vld),   
    .pmu_cfg(pmu_cfg),

    .usr_cfg(usr_cfg),
    .usr_cfg_vld(usr_cfg_vld),

    .model_cfg(model_cfg),
    .model_cfg_vld(model_cfg_vld),

    .rc_cfg_vld(rc_cfg_vld),
    .rc_cfg(rc_cfg),

    .gbus_addr_delay1_array(gbus_addr_delay1_array),
    .gbus_wen_delay1_array(gbus_wen_delay1_array),
    .gbus_wdata_delay1_array(gbus_wdata_delay1_array)
);

logic      [`HEAD_NUM-1:0]                          vector_in_data_vld;
logic      [`HEAD_NUM-1:0]                          vector_in_data_vld_delay1;
logic                 [`IDATA_WIDTH-1:0]            vector_out_data;
logic                                               vector_out_data_vld;
logic           [`CMEM_ADDR_WIDTH-1:0]              vector_out_data_addr;

logic                 [`IDATA_WIDTH-1:0]            vector_out_data_delay1;
logic                                               vector_out_data_vld_delay1;
logic           [`CMEM_ADDR_WIDTH-1:0]              vector_out_data_addr_delay1;

logic                 [`IDATA_WIDTH-1:0]            vector_out_data_delay2;
logic                                               vector_out_data_vld_delay2; //在FFN1做residual时，和global_sram rdata 同相
logic           [`CMEM_ADDR_WIDTH-1:0]              vector_out_data_addr_delay2;

always_ff @(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        vector_in_data_vld_delay1 <= 0;
    end
    else begin
        vector_in_data_vld_delay1 <= vector_in_data_vld;
    end
end

always_ff@(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        control_state_delay1 <= IDLE_STATE;

        vector_out_data_vld_delay1 <= 0;
        vector_out_data_vld_delay2 <= 0;

        vector_out_data_addr_delay1 <= 0;
        vector_out_data_addr_delay2 <= 0;

        vector_out_data_delay1 <= 0;
        vector_out_data_delay2 <= 0;
    end
    else begin
        control_state_delay1 <= control_state;

        vector_out_data_vld_delay1 <= vector_out_data_vld;
        vector_out_data_vld_delay2 <= vector_out_data_vld_delay1;

        vector_out_data_addr_delay1 <= vector_out_data_addr;
        vector_out_data_addr_delay2 <= vector_out_data_addr_delay1;

        vector_out_data_delay1 <= vector_out_data;
        vector_out_data_delay2 <= vector_out_data_delay1;
    end
end


always_comb begin
    if(control_state_delay1 == PROJ_STATE || control_state_delay1 == FFN1_STATE)
        vector_in_data_vld = gbus_wen_delay1_array;
    else
        vector_in_data_vld = 0;
end

logic vec_adder_finish;

always_comb begin
    if(control_state_delay1 == FFN1_STATE)begin
        op_finish = residual_finish;
    end
    else if(control_state_delay1 == PROJ_STATE)begin
        op_finish = vec_adder_finish;
    end
    else begin
        op_finish = head_array_finish;
    end
end




array_ctrl inst_array_ctrl(
    .clk(clk),
    .rst_n(rst_n),
`ifdef INTERFACE_ENABLE
    .control_reg_addr(control_reg_addr), 

    .control_reg_wdata(control_reg_wdata),
    .control_reg_wen(control_reg_wen),

    .control_reg_rdata(control_reg_rdata),
    .control_reg_ren(control_reg_ren),
    .control_reg_rvld(control_reg_rvld),
`endif

    //op channel
    .control_state(control_state),
    .control_state_update(control_state_update),

    .op_start(op_start), //pulse
    .op_finish(op_finish),//pulse

    .op_cfg(op_cfg),
    .op_cfg_vld(op_cfg_vld),

    //user channel
    .new_token(new_token), //pulse, should be high after global sram accept all the content of the new token
    .user_first_token(user_first_token), //be in smae phase
    .user_id(user_id),// be in same phase

    .current_token_finish(current_token_finish),
    
    .usr_cfg(usr_cfg),
    .usr_cfg_vld(usr_cfg_vld),

    .pmu_cfg_vld(pmu_cfg_vld),   
    .pmu_cfg(pmu_cfg),

    //model channel
    .cfg_init_success(cfg_init_success), //pulse, when high means that all the config all initial successfully

    .model_cfg(model_cfg),
    .model_cfg_vld(model_cfg_vld),

    .rc_cfg_vld(rc_cfg_vld),
    .rc_cfg(rc_cfg),

    .power_mode_en_in(power_mode_en_in),
    .debug_mode_en_in(debug_mode_en_in),
    .debug_mode_bits_in(debug_mode_bits_in) //IDLE STATE默认执行   
);

global_sram_rd_ctrl inst_global_sram_rd_ctrl
(
    .clk(clk),
    .rst_n(rst_n),
        
    .control_state(control_state),
    .control_state_update(control_state_update),

    .model_cfg_vld(model_cfg_vld),   
    .model_cfg(model_cfg),
                                                    
    .start(op_start_delay1),
    .finish(global_sram_finish),

    .vector_out_data_addr(vector_out_data_addr),
    .vector_out_data_vld(vector_out_data_vld),

    .global_sram_ren(global_sram_ren),
    .global_sram_raddr(global_sram_raddr)
);

/*
ATTN_RESIDUAL rs_ram + global_sram -> global_sram
FFN_RESIDUAL rs_ram + global_sram -> rs_ram
*/

        

global_sram_wrapper #(
    .DATA_BIT((`MAC_MULT_NUM * `IDATA_WIDTH)),
    .DEPTH(`GLOBAL_SRAM_DEPTH)
)
inst_global_sram (
    .clk(clk),
    .rst_n(rst_n),
    .global_sram_waddr(global_sram_waddr_delay1),
    .global_sram_wen(global_sram_wen_delay1),
    .global_sram_bwe(global_sram_bwe_delay1),
    .global_sram_wdata(global_sram_wdata_delay1),

    .global_sram_raddr(global_sram_raddr),
    .global_sram_ren(global_sram_ren),
    .global_sram_rdata(global_sram_rdata),
    .global_sram_rvld(global_sram_rvld),

`ifdef INTERFACE_ENABLE
    .global_mem_addr(global_mem_addr),
    .global_mem_wdata(global_mem_wdata),
    .global_mem_wen(global_mem_wen),
    .global_mem_rdata(global_mem_rdata),
    .global_mem_ren(global_mem_ren),
    .global_mem_rvld(global_mem_rvld)    
`else
    .global_mem_addr(8'b0),
    .global_mem_wdata(16'b0),
    .global_mem_wen(1'b0),
    .global_mem_rdata(),
    .global_mem_ren(1'b0),
    .global_mem_rvld()    
`endif
);


vector_adder inst_vector_adder(
    .clk(clk),
    .rst_n(rst_n),

    .in_data(gbus_wdata_delay2_array),
    .in_data_vld(vector_in_data_vld_delay1),

    .in_data_addr(gbus_addr_delay2_array[3].cmem_addr), //随便选一个中间的，节省延时

    .op_cfg_vld(op_cfg_vld),
    .op_cfg(op_cfg),

    .out_data(vector_out_data),
    .out_data_vld(vector_out_data_vld),
    .out_data_addr(vector_out_data_addr),

    .in_finish(head_array_finish && (control_state_delay1 == PROJ_STATE ||control_state_delay1==FFN1_STATE )),
    .out_finish(vec_adder_finish)
);



always_comb begin
    residual_sram_ren = 0;
    residual_sram_raddr = 0;
    if(control_state_delay1 == FFN0_STATE)begin
        residual_sram_ren = global_sram_ren;
        residual_sram_raddr = global_sram_raddr; //zero extension
    end
end


always_ff @(posedge clk) begin
    residual_sram_raddr_delay1 <= residual_sram_raddr;
end
always_ff @(posedge clk or negedge rst_n)begin
    if(~rst_n)
        residual_sram_rvld <= 0;
    else if(residual_sram_ren)
        residual_sram_rvld <= 1;
    else
        residual_sram_rvld <= 0;
end

 


logic                                        rs_adder_scale_vld;
logic        [`CDATA_SCALE_WIDTH-1:0]        rs_adder_scale_a;
logic        [`CDATA_SCALE_WIDTH-1:0]        rs_adder_scale_b;

logic                                        rs_adder_shift_vld;
logic        [`CDATA_SHIFT_WIDTH-1:0]        rs_adder_shift;

logic signed [`IDATA_WIDTH*`MAC_MULT_NUM-1:0]                                     rs_adder_in_data_a;
logic signed [`IDATA_WIDTH*`MAC_MULT_NUM-1:0]                                     rs_adder_in_data_b;
logic                                                                             rs_adder_in_data_vld;
logic        [$clog2(`GLOBAL_SRAM_DEPTH)+$clog2(`MAC_MULT_NUM)-1:0]               rs_adder_in_addr;

logic signed [`IDATA_WIDTH*`MAC_MULT_NUM-1:0]                                     rs_adder_out_data;
logic                                                                             rs_adder_out_data_vld;
logic        [$clog2(`GLOBAL_SRAM_DEPTH)+$clog2(`MAC_MULT_NUM)-1:0]               rs_adder_out_addr;

OP_CONFIG                       ATTN_RESIDUAL_CFG_REG;//这个比较特殊，其中的accu num用于scale a, scale 用于 scale b  //buffer (delay consideartion)
OP_CONFIG                       MLP_RESIDUAL_CFG_REG;//这个比较特殊，其中的accu num用于scale a, scale 用于 scale b
always@(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        ATTN_RESIDUAL_CFG_REG <= 0;
        MLP_RESIDUAL_CFG_REG <= 0;
    end
    else begin
        ATTN_RESIDUAL_CFG_REG <= inst_array_ctrl.op_cfg_pkt.ATTN_RESIDUAL_CFG;
        MLP_RESIDUAL_CFG_REG <= inst_array_ctrl.op_cfg_pkt.MLP_RESIDUAL_CFG;
    end
end

always_comb begin
    rs_adder_scale_vld = 0;
    rs_adder_scale_a = 0;
    rs_adder_scale_b = 0;
    rs_adder_shift_vld = 0;
    rs_adder_shift = 0;
    if(op_cfg_vld && (control_state_delay1 == FFN1_STATE))begin
        rs_adder_scale_vld = 1;
        rs_adder_shift_vld = 1;
        rs_adder_scale_a = MLP_RESIDUAL_CFG_REG.cfg_acc_num[`CDATA_SCALE_WIDTH-1:0];
        rs_adder_scale_b = MLP_RESIDUAL_CFG_REG.cfg_quant_scale;
        rs_adder_shift =   MLP_RESIDUAL_CFG_REG.cfg_quant_shift;
    end
    else if(op_cfg_vld && (control_state_delay1 == FFN0_STATE))begin
        rs_adder_scale_vld = 1;
        rs_adder_shift_vld = 1;
        rs_adder_scale_a = ATTN_RESIDUAL_CFG_REG.cfg_acc_num[`CDATA_SCALE_WIDTH-1:0];
        rs_adder_scale_b = ATTN_RESIDUAL_CFG_REG.cfg_quant_scale;
        rs_adder_shift =   ATTN_RESIDUAL_CFG_REG.cfg_quant_shift;
    end
end

always_comb begin
    rs_adder_in_data_a = 0;
    rs_adder_in_data_b = 0;
    rs_adder_in_data_vld = 0;
    rs_adder_in_addr = 0;
    if((control_state_delay1 == FFN0_STATE) && residual_sram_rvld)begin
        rs_adder_in_data_a = global_sram_rdata;
        rs_adder_in_data_b = residual_sram_rdata;
        rs_adder_in_data_vld = residual_sram_rvld;
        rs_adder_in_addr = residual_sram_raddr_delay1;
    end
    else if((control_state_delay1 == FFN1_STATE) && vector_out_data_vld_delay2)begin
        rs_adder_in_data_a = global_sram_rdata[ vector_out_data_addr_delay2[$clog2(`GLOBAL_SRAM_DEPTH) +: $clog2(`MAC_MULT_NUM)]*8 +: 8];
        rs_adder_in_data_b = vector_out_data_delay2;
        rs_adder_in_data_vld = vector_out_data_vld_delay2;
        rs_adder_in_addr = vector_out_data_addr_delay2; //截取低位
    end
end


residual_adder inst_residual_adder(
    .clk(clk),
    .rst_n(rst_n),

    .in_finish(vec_adder_finish && control_state_delay1 == FFN1_STATE),
    .out_finish(residual_finish),

    .scale_vld(rs_adder_scale_vld),
    .scale_a(rs_adder_scale_a),
    .scale_b(rs_adder_scale_b),

    .shift_vld(rs_adder_shift_vld),
    .shift(rs_adder_shift),

    .in_data_a(rs_adder_in_data_a),
    .in_data_b(rs_adder_in_data_b),
    .in_data_vld(rs_adder_in_data_vld),
    .in_addr(rs_adder_in_addr),

    .out_data(rs_adder_out_data),
    .out_data_vld(rs_adder_out_data_vld),
    .out_addr(rs_adder_out_addr)
);


residual_global_sram_wr_ctrl inst_residual_sram_wr_ctrl(
    .clk(clk),
    .rst_n(rst_n),

    .control_state(control_state),
    .control_state_update(control_state_update),

    .model_cfg(model_cfg),
    .model_cfg_vld(model_cfg_vld),

    .vector_out_data(vector_out_data),
    .vector_out_data_vld(vector_out_data_vld),
    .vector_out_data_addr(vector_out_data_addr),

    .rs_adder_out_data(rs_adder_out_data),
    .rs_adder_out_data_vld(rs_adder_out_data_vld),
    .rs_adder_out_addr(rs_adder_out_addr),

    .residual_sram_wdata_byte_flag(residual_sram_wdata_byte_flag),
    .residual_sram_wen(residual_sram_wen),
    .residual_sram_waddr(residual_sram_waddr),
    .residual_sram_wdata(residual_sram_wdata),

    .global_sram_waddr(global_sram_waddr),
    .global_sram_wen(global_sram_wen),
    .global_sram_wdata(global_sram_wdata)
);

residual_sram inst_residual_sram
(
    .clk(clk),
    .rstn(rst_n),

`ifdef INTERFACE_ENABLE
    .interface_addr(residual_mem_addr),
    .interface_ren(residual_mem_ren),
    .interface_rdata(residual_mem_rdata),
    .interface_rvalid(residual_mem_rvld),
    .interface_wen(residual_mem_wen),
    .interface_wdata(residual_mem_wdata),
`else
    .interface_addr(8'b0),
    .interface_ren(1'b0),
    .interface_rdata(),
    .interface_rvalid(),
    .interface_wen(1'b0),
    .interface_wdata(16'b0),
`endif

    .raddr(residual_sram_raddr),
    .ren(residual_sram_ren),
    .rdata(residual_sram_rdata),
    
    .waddr(residual_sram_waddr),
    .wen(residual_sram_wen),
    .wdata(residual_sram_wdata),

    .wdata_byte_flag(residual_sram_wdata_byte_flag) //if wdata_byte_flag == 1, the address is byte based-BUCK
); 


endmodule
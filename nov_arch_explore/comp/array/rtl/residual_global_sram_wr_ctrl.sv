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

module residual_global_sram_wr_ctrl(//同时也会控制global sram的写
    input  logic                                                                    clk,
    input  logic                                                                    rst_n,

    input  CONTROL_STATE                                                            control_state,
    input  logic                                                                    control_state_update,

    input  logic                                                                    model_cfg_vld,
    input  MODEL_CONFIG                                                             model_cfg,

    // output logic                                                                    finish,

    input  logic          [`IDATA_WIDTH-1:0]                                        vector_out_data,
    input  logic                                                                    vector_out_data_vld,
    input  logic          [`CMEM_ADDR_WIDTH-1:0]                                    vector_out_data_addr,

    input  logic        [`IDATA_WIDTH*`MAC_MULT_NUM-1:0]                            rs_adder_out_data,
    input  logic                                                                    rs_adder_out_data_vld,
    input  logic        [$clog2(`GLOBAL_SRAM_DEPTH)+$clog2(`MAC_MULT_NUM)-1:0]      rs_adder_out_addr,

    output logic                                                                    residual_sram_wdata_byte_flag,
    output logic                                                                    residual_sram_wen,
    output logic  [$clog2(`MAC_MULT_NUM)+$clog2(`GLOBAL_SRAM_DEPTH)-1:0]            residual_sram_waddr,
    output logic  [(`MAC_MULT_NUM * `IDATA_WIDTH)-1:0]                              residual_sram_wdata,

    output logic [$clog2(`GLOBAL_SRAM_DEPTH)-1:0]                                   global_sram_waddr,
    output logic                                                                    global_sram_wen,
    output logic [(`MAC_MULT_NUM * `IDATA_WIDTH)-1:0]                               global_sram_wdata            
);

CONTROL_STATE                                        control_state_reg; 
MODEL_CONFIG                                         model_cfg_reg;

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

// logic                                                                  nxt_finish;

logic                                                                  nxt_residual_sram_wdata_byte_flag;
logic                                                                  nxt_residual_sram_wen;
logic     [$clog2(`MAC_MULT_NUM)+$clog2(`GLOBAL_SRAM_DEPTH)-1:0]       nxt_residual_sram_waddr;
logic     [(`MAC_MULT_NUM * `IDATA_WIDTH)-1:0]                         nxt_residual_sram_wdata;


logic [$clog2(`GLOBAL_SRAM_DEPTH)-1:0]                                   nxt_global_sram_waddr;
logic                                                                    nxt_global_sram_wen;
logic [(`MAC_MULT_NUM * `IDATA_WIDTH)-1:0]                               nxt_global_sram_wdata;

// always_ff@(posedge clk or negedge rst_n)begin
//     if(~rst_n)begin
//         finish <= 0;
//     end
//     else begin
//         finish <= nxt_finish;
//     end
// end

// always_comb begin
//     nxt_finish = 0;
//     if(finish)
//         nxt_finish = 0;
//     else if(residual_sram_waddr == model_cfg_reg.embd_size/`MAC_MULT_NUM - 1)begin
//         nxt_finish = 1;
//     end
//     else if(global_sram_waddr == model_cfg_reg.embd_size/`MAC_MULT_NUM - 1)begin
//         nxt_finish = 1;
//     end
// end

always_comb begin
    nxt_residual_sram_wdata_byte_flag = 0;
    nxt_residual_sram_wen = 0;
    nxt_residual_sram_waddr = 0;
    nxt_residual_sram_wdata = 0;

    nxt_global_sram_waddr = 0;
    nxt_global_sram_wen = 0;
    nxt_global_sram_wdata = 0;
    if(control_state_reg == PROJ_STATE)begin
        nxt_residual_sram_wdata_byte_flag = 1;
        nxt_residual_sram_waddr = vector_out_data_addr;
        nxt_residual_sram_wen = vector_out_data_vld;
        nxt_residual_sram_wdata = vector_out_data;
    end
    else if (control_state_reg == FFN1_STATE)begin
        nxt_residual_sram_wdata_byte_flag = 1;
        nxt_residual_sram_waddr = rs_adder_out_addr;
        nxt_residual_sram_wen = rs_adder_out_data_vld;
        nxt_residual_sram_wdata = rs_adder_out_data[`IDATA_WIDTH-1:0];
    end

    if(control_state_reg == FFN0_STATE)begin
        nxt_global_sram_waddr = rs_adder_out_addr; //截取低位
        nxt_global_sram_wen = rs_adder_out_data_vld;
        nxt_global_sram_wdata = rs_adder_out_data;
    end
    
end

always_ff@(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        residual_sram_wdata_byte_flag <= 0;
        residual_sram_wen <= 0;
        residual_sram_waddr <= 0;
        residual_sram_wdata <= 0;
        global_sram_waddr <= 0;
        global_sram_wen <= 0;
        global_sram_wdata <= 0;    
    end
    else begin
        residual_sram_wdata_byte_flag <= nxt_residual_sram_wdata_byte_flag;
        residual_sram_wen <= nxt_residual_sram_wen;
        residual_sram_waddr <= nxt_residual_sram_waddr;
        residual_sram_wdata <= nxt_residual_sram_wdata;
        global_sram_waddr <= nxt_global_sram_waddr;
        global_sram_wen <= nxt_global_sram_wen;
        global_sram_wdata <= nxt_global_sram_wdata;    
    end
end

endmodule
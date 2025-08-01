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
// `include "../includes/CORE_DEF.svh"
module core_top #(
    // 1. Global Bus and Core-to-Core Link
    parameter   GBUS_DATA_WIDTH      = `GBUS_DATA_WIDTH,               // Data Bitwidth
    parameter   GBUS_ADDR_WIDTH      = `GBUS_ADDR_WIDTH,               // Memory Space


    // cmem_addr port
    parameter   CMEM_ADDR_WIDTH      = `CMEM_ADDR_WIDTH,
    parameter   CMEM_DATA_WIDTH      = (`MAC_MULT_NUM * `IDATA_WIDTH),
    
    // vlink data
    parameter   VLINK_DATA_WIDTH     = (`MAC_MULT_NUM * `IDATA_WIDTH),
    // hlinke data
    parameter   HLINK_DATA_WIDTH     = (`MAC_MULT_NUM * `IDATA_WIDTH),

    // 3. Computing Logic
    parameter   MAC_MULT_NUM         = `MAC_MULT_NUM,                 // MAC Line Size
    parameter   IDATA_WIDTH          = `IDATA_WIDTH,               // Input and Output Bitwidth
    parameter   ODATA_BIT            = `ODATA_WIDTH,               // Accumulate Sum Bitwidth

    // 4. Config Signals
    parameter   CDATA_ACCU_NUM_WIDTH = `CDATA_ACCU_NUM_WIDTH,
    parameter   CDATA_SCALE_WIDTH    = `CDATA_SCALE_WIDTH,
    parameter   CDATA_BIAS_WIDTH     = `CDATA_BIAS_WIDTH,
    parameter   CDATA_SHIFT_WIDTH    = `CDATA_SHIFT_WIDTH,

    parameter   HEAD_INDEX           = 0,
    parameter   CORE_INDEX           = 0
)(
    // Global Signals
    input                                       clk,
    input                                       rstn,

    input  logic                                clean_kv_cache, //pulse
    input  logic       [`USER_ID_WIDTH-1:0]     clean_kv_cache_user_id, //clean the corrsponding user's kv cache to zero

    input  logic [`CORE_MEM_ADDR_WIDTH-1:0]     core_mem_addr,
    input  logic [`INTERFACE_DATA_WIDTH-1:0]    core_mem_wdata,
    input  logic                                core_mem_wen,
    output logic [`INTERFACE_DATA_WIDTH-1:0]    core_mem_rdata,
    input  logic                                core_mem_ren,
    output logic                                core_mem_rvld,

    // Global Config Signals
    input                                       op_cfg_vld,
    input       OP_CONFIG                       op_cfg,

    input                                       usr_cfg_vld,
    input       USER_CONFIG                     usr_cfg,

    input                                       model_cfg_vld,
    input       MODEL_CONFIG                    model_cfg,

    input                                       pmu_cfg_vld,
    input       PMU_CONFIG                      pmu_cfg,
     
    input       logic                           rc_cfg_vld,
    input       RC_CONFIG                       rc_cfg,//和model config一样在网络初始化的时候配置好就行        

    input       CONTROL_STATE                   control_state, 
    input       logic                           control_state_update,

    input       logic                           start,  //start信号具有和control state一样的寄存器delay
    output      logic                           finish,

    //GBUS inport
    input         BUS_ADDR                      in_gbus_addr,     
    input                                       in_gbus_wen,
    input         [GBUS_DATA_WIDTH-1:0]         in_gbus_wdata,

    //GBUS outport
    output        BUS_ADDR                      out_gbus_addr,     
    output logic                                out_gbus_wen,
    output logic  [GBUS_DATA_WIDTH-1:0]         out_gbus_wdata,

    //recompute logic
    input  logic [`RECOMPUTE_SCALE_WIDTH - 1:0] rc_scale,
    input  logic                                rc_scale_vld,
    input  logic                                rc_scale_clear,

    // Channel - Core-to-Core Link
    // Vertical for Weight and Key/Value Propagation
    input  logic [VLINK_DATA_WIDTH-1:0]         vlink_data_in,
    input  logic                                vlink_data_in_vld,

    output logic [VLINK_DATA_WIDTH-1:0]         vlink_data_out,        
    output logic                                vlink_data_out_vld,

    input       [HLINK_DATA_WIDTH-1:0]          hlink_wdata,
    input                                       hlink_wen,
    output      [HLINK_DATA_WIDTH-1:0]          hlink_rdata,
    output                                      hlink_rvalid

);

    OP_CONFIG       op_cfg_reg;
    USER_CONFIG     usr_cfg_reg;
    MODEL_CONFIG    model_cfg_reg;
    PMU_CONFIG      pmu_cfg_reg;

    logic       [CDATA_ACCU_NUM_WIDTH-1:0]      cfg_acc_num;
    logic       [CDATA_SCALE_WIDTH-1:0]         cfg_quant_scale;
    logic       [CDATA_BIAS_WIDTH-1:0]          cfg_quant_bias;
    logic       [CDATA_SHIFT_WIDTH-1:0]         cfg_quant_shift;

    logic                                clean_kv_cache_delay1; //pulse
    logic       [`USER_ID_WIDTH-1:0]     clean_kv_cache_user_id_delay1;

    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            op_cfg_reg <= 0;
        end
        else if (op_cfg_vld)begin 
            op_cfg_reg <= op_cfg;
        end
    end
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            pmu_cfg_reg <= 0;
        end
        else if (pmu_cfg_vld)begin 
            pmu_cfg_reg <= pmu_cfg;
        end
    end
    always_comb begin
        cfg_acc_num = op_cfg_reg.cfg_acc_num;
        cfg_quant_scale = op_cfg_reg.cfg_quant_scale;
        cfg_quant_bias = op_cfg_reg.cfg_quant_bias;
        cfg_quant_shift = op_cfg_reg.cfg_quant_shift;
    end
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            model_cfg_reg <= 0;
        end
        else if (model_cfg_vld)begin 
            model_cfg_reg <= model_cfg;
        end
    end
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            usr_cfg_reg <= 0;
        end
        else if (usr_cfg_vld)begin 
            usr_cfg_reg <= usr_cfg;
        end
    end

    always_ff@(posedge clk or negedge rstn)begin
        if(~rstn)begin
            clean_kv_cache_delay1 <= 0;
            clean_kv_cache_user_id_delay1 <= 0;
        end
        else begin
            clean_kv_cache_delay1 <= clean_kv_cache;
            clean_kv_cache_user_id_delay1 <= clean_kv_cache_user_id;
        end
    end

    // =============================================================================
    //core_systolic logic
    // always_ff @( posedge clk or negedge rstn ) begin
    //     core_out_cmem_raddr <= core_in_cmem_raddr;
    //     core_out_cmem_ren <= core_in_cmem_ren;
    // end


    // =============================================================================
    // Core Memory Module for Weight and KV Cache Access
    parameter integer USER_ID_WIDTH = $clog2(`MAX_NUM_USER);

    logic    [CMEM_DATA_WIDTH-1:0]     cmem_wdata;
    logic    [CMEM_ADDR_WIDTH-1:0]     cmem_waddr;
    logic                              cmem_wen;

    logic    [CMEM_DATA_WIDTH-1:0]     cmem_rdata;
    logic    [CMEM_ADDR_WIDTH-1:0]     cmem_raddr; 
    logic                              cmem_ren;
    logic                              cmem_rvalid;

    logic    [CMEM_ADDR_WIDTH-1:0]     self_cmem_raddr; //core self generated cmem_raddr
    logic                              self_cmem_ren;   //core self generated cmem_ren

    always_comb begin //cmem write channel
        cmem_wdata = 0;
        cmem_waddr = 0;
        cmem_wen = 0;        //to core                                        //to this core
        if(in_gbus_wen && (in_gbus_addr.hs_bias==0) && in_gbus_addr.core_addr ==  CORE_INDEX)begin
            cmem_wen = 1;
            cmem_waddr = in_gbus_addr.cmem_addr;
            cmem_wdata = in_gbus_wdata; 
        end
    end

    always_comb begin //cmem read channel
        cmem_raddr = 0;
        cmem_ren = 0;
        // if(core_in_cmem_ren)begin
        //     cmem_ren = 1;
        //     cmem_raddr = core_in_cmem_raddr;
        // end 
        if(self_cmem_ren)begin
            cmem_ren = 1;
            cmem_raddr = self_cmem_raddr;
        end
    end
    
    core_mem   #(
        .CMEM_ADDR_WIDTH(CMEM_ADDR_WIDTH), 
        .CMEM_DATA_WIDTH(CMEM_DATA_WIDTH), 
        .VLINK_DATA_WIDTH(VLINK_DATA_WIDTH),
        .HEAD_INDEX(HEAD_INDEX)
    )
    mem_inst (
        .clk                    (clk),
        .rstn                   (rstn),

        .clean_kv_cache         (clean_kv_cache_delay1),
        .clean_kv_cache_user_id (clean_kv_cache_user_id_delay1),

        .core_mem_addr          (core_mem_addr),
        .core_mem_wdata         (core_mem_wdata),
        .core_mem_wen           (core_mem_wen),
        .core_mem_rdata         (core_mem_rdata),
        .core_mem_ren           (core_mem_ren),
        .core_mem_rvld          (core_mem_rvld),

        .vlink_data_in          (vlink_data_in),
        .vlink_data_in_vld      (vlink_data_in_vld),
        .vlink_data_out         (vlink_data_out),
        .vlink_data_out_vld     (vlink_data_out_vld),

        .usr_cfg                (usr_cfg_reg),
        .model_cfg              (model_cfg_reg),
        .control_state          (control_state),
        .control_state_update   (control_state_update),
        .pmu_cfg                (pmu_cfg_reg), 

        .cmem_waddr             (cmem_waddr),
        .cmem_wen               (cmem_wen),
        .cmem_wdata             (cmem_wdata),

        .cmem_raddr             (cmem_raddr),
        .cmem_ren               (cmem_ren),
        .cmem_rdata             (cmem_rdata),
        .cmem_rvalid            (cmem_rvalid)
    );
    // =============================================================================
    // Core Interface logic



    // =============================================================================
    // Core Buffer Module for Activation Access

    core_buf    #(.CACHE_DATA_WIDTH((`MAC_MULT_NUM * `IDATA_WIDTH))) buf_inst (
        .clk                    (clk),
        .rstn                   (rstn),

        .hlink_wdata            (hlink_wdata),
        .hlink_wen              (hlink_wen),
        .hlink_rdata            (hlink_rdata),
        .hlink_rvalid           (hlink_rvalid)
    );

    // =============================================================================
    // MAC Module
    logic                                                mac_opa_vld;
    logic                         [(`MAC_MULT_NUM * `IDATA_WIDTH)-1:0]  mac_opa;

    logic                                                mac_opb_vld;
    logic                         [(`MAC_MULT_NUM * `IDATA_WIDTH)-1:0]  mac_opb;

    logic     signed              [ODATA_BIT-1:0]        mac_odata;
    logic                                                mac_odata_valid;


    core_mac    #(.MAC_MULT_NUM(MAC_MULT_NUM), .IDATA_WIDTH(IDATA_WIDTH), .ODATA_BIT(ODATA_BIT)) mac_inst (
        .clk                    (clk),
        .rstn                   (rstn),

        .idataA                 (mac_opa),
        .idataB                 (mac_opb),
        .idata_valid            (mac_opa_vld && mac_opb_vld),
        .odata                  (mac_odata),
        .odata_valid            (mac_odata_valid)
    );

    // =============================================================================
    // ACC Module

    logic    signed [ODATA_BIT-1:0]     acc_odata;
    logic                               acc_odata_valid;

    core_acc    #(.IDATA_WIDTH(ODATA_BIT), .ODATA_BIT(ODATA_BIT)) acc_inst (
        .clk                    (clk),
        .rstn                   (rstn),
        .cfg_acc_num            (cfg_acc_num),

        .idata                  ($signed(mac_odata)),
        .idata_valid            (mac_odata_valid),
        .odata                  (acc_odata),
        .odata_valid            (acc_odata_valid)
    );


    // =============================================================================
    // Recompute Module
    logic                                     recompute_needed;
    logic [ODATA_BIT - 1:0]                   rc_out_data;
    logic                                     rc_out_data_vld;
    logic                                     rc_error;
    logic [`RECOMPUTE_SHIFT_WIDTH - 1:0]      rms_rc_shift;
    logic                                                     rc_cfg_vld_reg;
    RC_CONFIG                                                 rc_cfg_reg;

    always_ff @(posedge clk or negedge rstn)begin //control signals retiming
    if(~rstn)begin
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

    assign rms_rc_shift = rc_cfg_reg.rms_rc_shift;

    core_rc #(
        .IN_DATA_WIDTH(ODATA_BIT),
        .OUT_DATA_WIDTH(ODATA_BIT),
        .RECOMPUTE_FIFO_DEPTH(`RECOMPUTE_FIFO_DEPTH),
        .RETIMING_REG_NUM(`RC_RETIMING_REG_NUM)
    )rc_inst(
        .clk(clk),
        .rst_n(rstn),

        .recompute_needed(recompute_needed),
    
        .rc_scale(rc_scale),
        .rc_scale_vld(rc_scale_vld),
        .rc_scale_clear(rc_scale_clear),

        .rms_rc_shift(rms_rc_shift),
    
        .in_data(acc_odata),
        .in_data_vld(acc_odata_valid),

        .out_data(rc_out_data),
        .out_data_vld(rc_out_data_vld),

        .error(rc_error)
    );


    // =============================================================================
    // Quantization Module

    logic signed  [IDATA_WIDTH-1:0]     quant_odata;
    logic                             quant_odata_valid;

    core_quant  #(.IDATA_WIDTH(ODATA_BIT), .ODATA_BIT(IDATA_WIDTH)) quant_inst (
        .clk                    (clk),
        .rstn                   (rstn),

        .cfg_quant_scale        (cfg_quant_scale),
        .cfg_quant_bias         (cfg_quant_bias),
        .cfg_quant_shift        (cfg_quant_shift),

        .idata                  (rc_out_data),
        .idata_valid            (rc_out_data_vld && (inst_core_ctrl.control_state_reg != PROJ_STATE && inst_core_ctrl.control_state_reg != FFN1_STATE)),//gating
        .odata                  (quant_odata),
        .odata_valid            (quant_odata_valid)
    );

    // =============================================================================
    // s2p Module

    logic [`GBUS_DATA_WIDTH-1:0] parallel_data;
    logic                        parallel_data_valid;

    align_s2p #(
        .IDATA_WIDTH(`IDATA_WIDTH),
        .ODATA_BIT(`GBUS_DATA_WIDTH)
    )align_s2p_inst(
        .clk(clk),
        .rstn(rstn),

        .idata(quant_odata),
        .idata_valid(quant_odata_valid && inst_core_ctrl.control_state_reg == ATT_QK_STATE),
        .odata(parallel_data),
        .odata_valid(parallel_data_valid)
    );


    core_ctrl #(
        .CORE_INDEX(CORE_INDEX)
    )inst_core_ctrl(
        .clk                                        (clk),
        .rst_n                                      (rstn),
                        
        .control_state                              (control_state),
        .control_state_update                       (control_state_update), 

        .usr_cfg                                    (usr_cfg_reg),

        .model_cfg                                  (model_cfg_reg),

                                                             //start信号具有和control state一样的寄存器delay
        .start                                      (start), //start indicate the current opearation starts
        .finish                                     (finish), //finish indicate the current opeatation ends for this ctrl
    
        .hlink_wen                                  (hlink_wen),
        .hlink_rvalid                               (hlink_rvalid),
        .hlink_rdata                                (hlink_rdata),    
    
        .cmem_rvalid                                (cmem_rvalid),
        .cmem_rdata                                 (cmem_rdata),
    
        .quant_odata                                (quant_odata), //for those who write back the byte
        .quant_odata_valid                          (quant_odata_valid),

        .parallel_data                              (parallel_data),
        .parallel_data_valid                          (parallel_data_valid),

        .rc_out_data                                (rc_out_data),
        .rc_out_data_vld                            (rc_out_data_vld),

        .recompute_needed                           (recompute_needed),

        .self_cmem_ren                              (self_cmem_ren),
        .self_cmem_raddr                            (self_cmem_raddr),
    
        .mac_opa_vld                                (mac_opa_vld),
        .mac_opa                                    (mac_opa), 
    
        .mac_opb_vld                                (mac_opb_vld),
        .mac_opb                                    (mac_opb), 
        
        .out_gbus_addr                              (out_gbus_addr),
        .out_gbus_wen                               (out_gbus_wen),
        .out_gbus_wdata                             (out_gbus_wdata)
    );
endmodule 
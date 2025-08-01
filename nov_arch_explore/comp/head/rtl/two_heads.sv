module two_heads #(
        parameter VLINK_DATA_WIDTH  = (`MAC_MULT_NUM * `IDATA_WIDTH),
        parameter GLOBAL_SRAM_DATA_BIT = (`MAC_MULT_NUM * `IDATA_WIDTH)
)
(
    input  logic                                                     clk,
    input  logic                                                     rst_n,

    input  logic                                                     clean_kv_cache, //pulse
    input  logic       [`USER_ID_WIDTH-1:0]                          clean_kv_cache_user_id, //clean the corrsponding user's kv cache to zero

    input  logic           [`HEAD_MEM_ADDR_WIDTH-1:0]                head_mem_addr_0,
    input  logic           [`INTERFACE_DATA_WIDTH-1:0]               head_mem_wdata_0,
    input  logic                                                     head_mem_wen_0,
    output logic           [`INTERFACE_DATA_WIDTH-1:0]               head_mem_rdata_0,
    input  logic                                                     head_mem_ren_0,
    output logic                                                     head_mem_rvld_0,

    input  logic           [`HEAD_MEM_ADDR_WIDTH-1:0]                head_mem_addr_1,
    input  logic           [`INTERFACE_DATA_WIDTH-1:0]               head_mem_wdata_1,
    input  logic                                                     head_mem_wen_1,
    output logic           [`INTERFACE_DATA_WIDTH-1:0]               head_mem_rdata_1,
    input  logic                                                     head_mem_ren_1,
    output logic                                                     head_mem_rvld_1,
 
    input  logic         [GLOBAL_SRAM_DATA_BIT-1:0]                  global_sram_rd_data,
    input  logic                                                     global_sram_rd_data_vld,

    input  CONTROL_STATE                                             control_state,
    input  logic                                                     control_state_update,
                                                                    //start信号具有和control state一样的寄存器delay
    input  logic                                                     start, //start indicate the current opearation starts
    output logic                                                     finish_0, //finish indicate the current opeatation ends for this ctrl    
    output logic                                                     finish_1, //finish indicate the current opeatation ends for this ctrl

    input  logic                                                     op_cfg_vld,      //和control_state的来自core_array的寄存器delay一样
    input  OP_CONFIG                                                 op_cfg,

    input                                                            usr_cfg_vld,
    input  USER_CONFIG                                               usr_cfg,

    input                                                            model_cfg_vld,
    input  MODEL_CONFIG                                              model_cfg,

    input  logic                                                     pmu_cfg_vld,
    input  PMU_CONFIG                                                pmu_cfg, 

    input  logic                                                     rc_cfg_vld,
    input  RC_CONFIG                                                 rc_cfg,//和model config一样在网络初始化的时候配置好就行   

    output BUS_ADDR                                                  gbus_addr_delay1_0, //latency consideration
    output logic                                                     gbus_wen_delay1_0,
    output logic                 [`GBUS_DATA_WIDTH-1:0]              gbus_wdata_delay1_0,     

    output BUS_ADDR                                                  gbus_addr_delay1_1, //latency consideration
    output logic                                                     gbus_wen_delay1_1,
    output logic                 [`GBUS_DATA_WIDTH-1:0]              gbus_wdata_delay1_1
);

logic         [`HEAD_CORE_NUM-1:0][VLINK_DATA_WIDTH-1:0]   vlink_data_in_array_0;
logic         [`HEAD_CORE_NUM-1:0]                         vlink_data_in_vld_array_0;
logic         [`HEAD_CORE_NUM-1:0][VLINK_DATA_WIDTH-1:0]   vlink_data_out_array_0;
logic         [`HEAD_CORE_NUM-1:0]                         vlink_data_out_vld_array_0;

logic         [`HEAD_CORE_NUM-1:0][VLINK_DATA_WIDTH-1:0]   vlink_data_in_array_1;
logic         [`HEAD_CORE_NUM-1:0]                         vlink_data_in_vld_array_1;
logic         [`HEAD_CORE_NUM-1:0][VLINK_DATA_WIDTH-1:0]   vlink_data_out_array_1;
logic         [`HEAD_CORE_NUM-1:0]                         vlink_data_out_vld_array_1;

always_comb begin
    vlink_data_in_array_0 = vlink_data_out_array_1;
    vlink_data_in_vld_array_0 = vlink_data_out_vld_array_1;
    vlink_data_in_array_1 = vlink_data_out_array_0;
    vlink_data_in_vld_array_1 = vlink_data_out_vld_array_0;
end



head_top #(
    .HEAD_INDEX(0)
 )
inst_head_top_0(
    .clk(clk),
    .rst_n(rst_n),

    .clean_kv_cache(clean_kv_cache),
    .clean_kv_cache_user_id(clean_kv_cache_user_id),

    .head_mem_addr(head_mem_addr_0),
    .head_mem_wdata(head_mem_wdata_0),
    .head_mem_wen(head_mem_wen_0),
    .head_mem_rdata(head_mem_rdata_0),
    .head_mem_ren(head_mem_ren_0),
    .head_mem_rvld(head_mem_rvld_0),

    .global_sram_rd_data(global_sram_rd_data),
    .global_sram_rd_data_vld(global_sram_rd_data_vld),

    .vlink_data_in_array(vlink_data_in_array_0),
    .vlink_data_in_vld_array(vlink_data_in_vld_array_0),
    .vlink_data_out_array(vlink_data_out_array_0),
    .vlink_data_out_vld_array(vlink_data_out_vld_array_0),

    .control_state(control_state),
    .control_state_update(control_state_update),

    .start(start),
    .finish(finish_0),

    // operation Config Signals
    .op_cfg_vld(op_cfg_vld),   
    .op_cfg(op_cfg),

    // model Config Signals
    .model_cfg_vld(model_cfg_vld),   
    .model_cfg(model_cfg),

    // PMU Config Signals
    .pmu_cfg_vld(pmu_cfg_vld),   
    .pmu_cfg(pmu_cfg),

    // user Config Signals
    .usr_cfg_vld(usr_cfg_vld),
    .usr_cfg(usr_cfg),

    .rc_cfg_vld(rc_cfg_vld),
    .rc_cfg(rc_cfg),

    .gbus_addr_delay1(gbus_addr_delay1_0),
    .gbus_wen_delay1(gbus_wen_delay1_0),
    .gbus_wdata_delay1(gbus_wdata_delay1_0)
);


head_top #(
    .HEAD_INDEX(1)
 )
inst_head_top_1(
    .clk(clk),
    .rst_n(rst_n),

    .clean_kv_cache(clean_kv_cache),
    .clean_kv_cache_user_id(clean_kv_cache_user_id),

    .head_mem_addr(head_mem_addr_1),
    .head_mem_wdata(head_mem_wdata_1),
    .head_mem_wen(head_mem_wen_1),
    .head_mem_rdata(head_mem_rdata_1),
    .head_mem_ren(head_mem_ren_1),
    .head_mem_rvld(head_mem_rvld_1),

    .global_sram_rd_data(global_sram_rd_data),
    .global_sram_rd_data_vld(global_sram_rd_data_vld),

    .vlink_data_in_array(vlink_data_in_array_1),
    .vlink_data_in_vld_array(vlink_data_in_vld_array_1),
    .vlink_data_out_array(vlink_data_out_array_1),
    .vlink_data_out_vld_array(vlink_data_out_vld_array_1),

    .control_state(control_state),
    .control_state_update(control_state_update),

    .start(start),
    .finish(finish_1),

    // operation Config Signals
    .op_cfg_vld(op_cfg_vld),   
    .op_cfg(op_cfg),

    // model Config Signals
    .model_cfg_vld(model_cfg_vld),   
    .model_cfg(model_cfg),

    // PMU Config Signals
    .pmu_cfg_vld(pmu_cfg_vld),   
    .pmu_cfg(pmu_cfg),

    // user Config Signals
    .usr_cfg_vld(usr_cfg_vld),
    .usr_cfg(usr_cfg),

    .rc_cfg_vld(rc_cfg_vld),
    .rc_cfg(rc_cfg),

    .gbus_addr_delay1(gbus_addr_delay1_1),
    .gbus_wen_delay1(gbus_wen_delay1_1),
    .gbus_wdata_delay1(gbus_wdata_delay1_1)
);






endmodule
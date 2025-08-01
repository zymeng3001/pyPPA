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

module core_mem #(
    // cmem_addr port
    parameter   CMEM_ADDR_WIDTH   =       `CMEM_ADDR_WIDTH,
    parameter   CMEM_DATA_WIDTH   =       (`MAC_MULT_NUM * `IDATA_WIDTH),
    parameter   CMEM_USER_WIDTH   =       $clog2(`MAX_NUM_USER),
    
    // vlink data
    parameter   VLINK_DATA_WIDTH  =       (`MAC_MULT_NUM * `IDATA_WIDTH),

    // Single-Port Weight Memory (SRAM). Config the default setting for a real application.
    // The WMEM has the same data bitwidth with GBUS.
    parameter   WMEM_DEPTH =              `WMEM_DEPTH,                  // Depth
    parameter   WMEM_ADDR_WIDTH  =        $clog2(WMEM_DEPTH),    // Address Bitwidth

    // Single-Port KV Cache (SRAM). Config the default setting for a real application.;;llll
    // The KV Cache has the same data bitwith with GBUS.
    parameter   SINGLE_USR_CACHE_DEPTH =  `KV_CACHE_DEPTH_SINGLE_USER,                 // Depth
    parameter   CACHE_NUM   =             `MAC_MULT_NUM,
    parameter   SINGLE_USR_CACHE_ADDR_WIDTH  = ($clog2(CACHE_NUM)+$clog2(SINGLE_USR_CACHE_DEPTH)),   // Address Bitwidth
    parameter   HEAD_INDEX = 0
)(
    //core_mem输出的数据 来自于三个地方，
    //一个是vlink_rdata
    //一个是wmem_rdata，
    //一个是cache_rdata
    //vlink_rdata 来自 vlink_rdata 或者 kv_cache (kv_cache sharing)
    //wmem_rdata 来自 wmem由cmem_raddr控制
    //cache_rdata 来自 kv_cache由cmem_raddr控制 -BUCK

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

    // Channel - Core-to-Core Link (vlink)
    // Global Config Signals
    input       logic [VLINK_DATA_WIDTH-1:0]    vlink_data_in,
    input       logic                           vlink_data_in_vld,

    output      logic [VLINK_DATA_WIDTH-1:0]    vlink_data_out,        
    output      logic                           vlink_data_out_vld,

    input       USER_CONFIG                     usr_cfg,
    input       MODEL_CONFIG                    model_cfg,
    input       CONTROL_STATE                   control_state,
    input       logic                           control_state_update,
    input       PMU_CONFIG                      pmu_cfg,       
    // Channel - Core Memory for MAC Operation
    // Write back results from MAC directly.
    //MSB = 1 is for KV CACHE, MSB = 0 is for WMEM -BUCK
    //不支持同时读和写
    input       [CMEM_ADDR_WIDTH-1:0]           cmem_waddr,         // Assume WMEM_ADDR_WIDTH >= SINGLE_USR_CACHE_ADDR_WIDTH
    input                                       cmem_wen,           // cmem_addr channel 给core自己用的 -BUCK
    input       [CMEM_DATA_WIDTH-1:0]           cmem_wdata,
    // input                               cache_wdata_byte_flag,  //if cache_wdata_byte_flag == 1, the address is byte based-BUCK      
    
    input       [CMEM_ADDR_WIDTH-1:0]           cmem_raddr,
    input                                       cmem_ren,
    output  reg [CMEM_DATA_WIDTH-1:0]           cmem_rdata,
    output  reg                                 cmem_rvalid
);
    localparam K_CACHE_ADDR_BASE = 0;
    localparam V_CACHE_ADDR_BASE = (`MAX_CONTEXT_LENGTH * `MAX_EMBD_SIZE / `HEAD_NUM / `HEAD_CORE_NUM / `MAC_MULT_NUM);

    CONTROL_STATE                             control_state_reg, control_state_reg_d;
    logic                                     state_changed;
    assign state_changed = (control_state_reg != control_state_reg_d);
    always_ff@ (posedge clk or negedge rstn)begin
        if(~rstn)
            control_state_reg <= IDLE_STATE;
        else if(control_state_update)
            control_state_reg <= control_state;
    end
    always_ff@ (posedge clk or negedge rstn)begin
        if(~rstn)
            control_state_reg_d <= IDLE_STATE;
        else
            control_state_reg_d <= control_state_reg;
    end
    // =============================================================================
    // Memory Instantization

    // 1. Single-Port Weight Memory
    reg     [WMEM_ADDR_WIDTH-1:0]     wmem_addr;

    reg                               wmem_wen;
    logic   [CMEM_DATA_WIDTH-1:0]     wmem_bwe;
    reg     [CMEM_DATA_WIDTH-1:0]     wmem_wdata;

    reg                               wmem_ren;
    wire    [CMEM_DATA_WIDTH-1:0]     wmem_rdata;
    reg                               wmem_rvalid; 
    
    //Power Management

    logic                                                wmem_1024_ffn_deepslp;
    logic                                                wmem_1024_ffn_bc1;
    logic                                                wmem_1024_ffn_bc2;

    logic                                                wmem_512_attn_deepslp;
    logic                                                wmem_512_attn_bc1;
    logic                                                wmem_512_attn_bc2;


    always_ff @(posedge clk or negedge rstn) begin
        if(~rstn) begin
            wmem_1024_ffn_deepslp <= 0;
            wmem_1024_ffn_bc1 <= 0;
            wmem_1024_ffn_bc2 <= 0;
            wmem_512_attn_deepslp <= 0;
            wmem_512_attn_bc1 <= 0;
            wmem_512_attn_bc2 <= 0;
        end
        else if(pmu_cfg.pmu_cfg_en) begin
            wmem_1024_ffn_bc1 <= pmu_cfg.pmu_cfg_bc1;
            wmem_512_attn_bc1 <= pmu_cfg.pmu_cfg_bc1;

            wmem_1024_ffn_bc2 <= pmu_cfg.pmu_cfg_bc2;
            wmem_512_attn_bc2 <= pmu_cfg.pmu_cfg_bc2;
            if(pmu_cfg.deepsleep_en)begin
                if(state_changed && control_state_reg == Q_GEN_STATE) begin
                    wmem_1024_ffn_deepslp <= 1;
                    wmem_512_attn_deepslp <= 0;
                end
                else if(control_state_reg==FFN0_STATE && state_changed) begin
                    wmem_1024_ffn_deepslp <= 0;
                    wmem_512_attn_deepslp <= 1;
                end
                else if(control_state_reg == IDLE_STATE) begin
                    wmem_1024_ffn_deepslp <= 1;
                    wmem_512_attn_deepslp <= 1;
                end
            end
        end
    end

    logic [`INTERFACE_DATA_WIDTH-1:0] weight_mem_rdata;
    logic                             weight_mem_rvld;

    wmem wmem_inst
    ( 
        .clk(clk),
        .rstn(rstn),
    
        .wmem_addr(wmem_addr),
    
        .wmem_ren(wmem_ren),
        .wmem_rdata(wmem_rdata),
    
        .wmem_wen(wmem_wen),
        .wmem_wdata(wmem_wdata),
        .wmem_bwe(wmem_bwe),

        .weight_mem_addr(core_mem_addr),
        .weight_mem_wdata(core_mem_wdata),
        .weight_mem_wen(core_mem_wen),//这里不是真正的，需要在内部判断
        .weight_mem_rdata(weight_mem_rdata),
        .weight_mem_ren(core_mem_ren),//这里不是真正的，需要在内部判断
        .weight_mem_rvld(weight_mem_rvld),

        .wmem_1024_ffn_deepslp(wmem_1024_ffn_deepslp),
        .wmem_1024_ffn_bc1(wmem_1024_ffn_bc1),
        .wmem_1024_ffn_bc2(wmem_1024_ffn_bc2),

        .wmem_512_attn_deepslp(wmem_512_attn_deepslp),
        .wmem_512_attn_bc1(wmem_512_attn_bc1),
        .wmem_512_attn_bc2(wmem_512_attn_bc2)
    );


    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            wmem_rvalid <= 1'b0;
        end
        else begin
            wmem_rvalid <= wmem_ren;
        end
    end

    // 2. Single-Port KV Cache
    reg     [SINGLE_USR_CACHE_ADDR_WIDTH-1:0]    cache_waddr;
    reg     [SINGLE_USR_CACHE_ADDR_WIDTH-1:0]    nxt_cache_waddr;
    reg     [SINGLE_USR_CACHE_ADDR_WIDTH-1:0]    cache_raddr;

    reg     [SINGLE_USR_CACHE_ADDR_WIDTH-1:0]   cache_addr;

    reg                                         cache_wen;
    reg                                         nxt_cache_wen;
    reg     [CMEM_DATA_WIDTH-1:0]               cache_wdata;
    reg     [CMEM_DATA_WIDTH-1:0]               nxt_cache_wdata;
    
    reg                                         cache_ren;
    wire    [CMEM_DATA_WIDTH-1:0]               cache_rdata;
    reg                                         cache_rvalid;

    logic                                       cache_wdata_byte_flag;

    ///////////////////////////////////////////////////////////
    //                      INTERFACE                        //
    ///////////////////////////////////////////////////////////
    logic [`INTERFACE_DATA_WIDTH-1:0] kv_mem_rdata;
    logic                             kv_mem_rvld;

    always_comb begin
        core_mem_rvld = 0;
        core_mem_rdata = 0;
        if(kv_mem_rvld)begin
            core_mem_rvld = 1;
            core_mem_rdata = kv_mem_rdata;
        end
        else if(weight_mem_rvld)begin
            core_mem_rvld = 1;
            core_mem_rdata = weight_mem_rdata;
        end
    end 

    

    assign cache_wdata_byte_flag = 1;
//`define CMEM_ADDR_WIDTH (1 + $clog2(`WMEM_DEPTH)) 
                         //1: cache, 0: wmem

//$clog2(ARR_WMEM_DEPTH) > 1 +   $clog2(`MAC_MULT_NUM) + $clog2(ARR_CACHE_DEPTH)
                         //1 is here to indicate whether the KV_CACHE address is byte based


    kv_cache_pkt #(
        .IDATA_WIDTH(CMEM_DATA_WIDTH),
        .ODATA_BIT(CMEM_DATA_WIDTH),     
        .CACHE_NUM(CACHE_NUM),
        .CACHE_DEPTH(SINGLE_USR_CACHE_DEPTH),   
        .CACHE_ADDR_WIDTH(SINGLE_USR_CACHE_ADDR_WIDTH)
    )
    kv_cache_inst(
        .clk(clk),
        .rstn(rstn),
        .clean_kv_cache(clean_kv_cache),
        .clean_kv_cache_user_id(clean_kv_cache_user_id),

        .kv_mem_addr(core_mem_addr),
        .kv_mem_wdata(core_mem_wdata),
        .kv_mem_wen(core_mem_wen), //这里不是真正的，需要在内部判断
        .kv_mem_rdata(kv_mem_rdata),
        .kv_mem_ren(core_mem_ren),//这里不是真正的，需要在内部判断
        .kv_mem_rvld(kv_mem_rvld),
        
        .cache_wen(cache_wen),
        .cache_addr(cache_addr),
        .cache_wdata(cache_wdata),
        .cache_ren(cache_ren),
        .cache_rdata(cache_rdata),
        .cache_wdata_byte_flag(cache_wdata_byte_flag),
        .usr_cfg(usr_cfg)
        // .bc1(pmu_bc1_kvcache),
        // .bc2(pmu_bc2_kvcache),
        // .deepslp(pmu_deepslp_kvcache)
    );

    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            cache_rvalid <= 1'b0;
        end
        else begin
            cache_rvalid <= cache_ren;
        end
    end

    // =============================================================================
    // cmem output Interface

    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            cmem_rvalid <= 1'b0;
        end
        else begin
            cmem_rvalid <= cmem_ren;
        end
    end
    logic       [CMEM_ADDR_WIDTH-1:0]           cmem_raddr_delay1;
    always_ff @ (posedge clk)begin
        cmem_raddr_delay1 <= cmem_raddr;
    end
    //cmem_read
    always @(*) begin
        if (cache_rvalid && cmem_rvalid) begin
            cmem_rdata = cache_rdata;
            if(model_cfg.gqa_en == 1)begin
                if(control_state_reg == ATT_QK_STATE)begin
                    if(HEAD_INDEX % 2 == 0)begin //0 2 4 6
                        if(cmem_raddr_delay1[0 +: $clog2(`KV_CACHE_DEPTH_SINGLE_USER_WITH_GQA)] >= K_CACHE_ADDR_BASE + V_CACHE_ADDR_BASE) begin
                            cmem_rdata = vlink_data_in;
                        end
                    end
                    else begin // 1 3 5 7
                        if(cmem_raddr_delay1[0 +: $clog2(`KV_CACHE_DEPTH_SINGLE_USER_WITH_GQA)] < K_CACHE_ADDR_BASE + V_CACHE_ADDR_BASE) begin
                            cmem_rdata = vlink_data_in;
                        end
                    end
                end
                else if(control_state_reg == ATT_PV_STATE)begin
                    if(HEAD_INDEX % 2 == 0)begin //0 2 4 6
                        if(cmem_raddr_delay1[0 +: $clog2(`KV_CACHE_DEPTH_SINGLE_USER_WITH_GQA)] >= V_CACHE_ADDR_BASE + V_CACHE_ADDR_BASE) begin
                            cmem_rdata = vlink_data_in;
                        end
                    end
                    else begin // 1 3 5 7
                        if(cmem_raddr_delay1[0 +: $clog2(`KV_CACHE_DEPTH_SINGLE_USER_WITH_GQA)] < V_CACHE_ADDR_BASE +  V_CACHE_ADDR_BASE) begin
                            cmem_rdata = vlink_data_in;
                        end
                    end
                end
            end
        end
        else if(wmem_rvalid && cmem_rvalid) begin
            cmem_rdata = wmem_rdata;
        end
        else begin
            cmem_rdata = 0;
        end
    end


    // =============================================================================
    // Weight Memory Interface

    reg     [WMEM_ADDR_WIDTH-1:0]     wmem_waddr;
    reg     [WMEM_ADDR_WIDTH-1:0]     wmem_raddr;

    always @(*) begin
        wmem_waddr = cmem_waddr[WMEM_ADDR_WIDTH-1:0];
        wmem_wen   = cmem_wen && ~cmem_waddr[CMEM_ADDR_WIDTH-1];
        wmem_wdata = cmem_wdata;
        wmem_bwe = {CMEM_DATA_WIDTH{1'b1}};
    end

    always @(*) begin
        wmem_ren = cmem_ren && ~cmem_raddr[CMEM_ADDR_WIDTH-1];
        wmem_raddr = cmem_raddr[WMEM_ADDR_WIDTH-1:0];
        
    end

    //WMEM_ADDR W/R Selection
    always @(*) begin
        if (wmem_wen) begin // Write
            wmem_addr = wmem_waddr;
        end
        else if(wmem_ren)begin // Read
            wmem_addr = wmem_raddr;
        end
        else begin
            wmem_addr = 0;
        end
    end

    // =============================================================================
    // KV Cache Interface
    //V_CACHE_ADDR_BASE刚好是原来KV CACHE的一半的分界线
    always_comb begin
        nxt_cache_wen   = cmem_wen && cmem_waddr[CMEM_ADDR_WIDTH-1];
        nxt_cache_waddr[0 +: $clog2(SINGLE_USR_CACHE_DEPTH)] = cmem_waddr[0 +: $clog2(SINGLE_USR_CACHE_DEPTH)];
        nxt_cache_waddr[(SINGLE_USR_CACHE_ADDR_WIDTH-1) -: $clog2(CACHE_NUM)] = cmem_waddr[$clog2(`KV_CACHE_DEPTH_SINGLE_USER_WITH_GQA) +: $clog2(`MAC_MULT_NUM)];
        nxt_cache_wdata = cmem_wdata;

        if(model_cfg.gqa_en == 1)begin
            if(control_state_reg == K_GEN_STATE)begin
                if(HEAD_INDEX % 2 == 0)begin //0 2 4 6
                    if(cmem_waddr[0 +: $clog2(`KV_CACHE_DEPTH_SINGLE_USER_WITH_GQA)]  >= K_CACHE_ADDR_BASE +  V_CACHE_ADDR_BASE) begin
                        nxt_cache_wen = 0;
                    end
                    else begin
                        nxt_cache_wen   = cmem_wen && cmem_waddr[CMEM_ADDR_WIDTH-1];
                        nxt_cache_waddr[0 +: $clog2(SINGLE_USR_CACHE_DEPTH)] = cmem_waddr[0 +: $clog2(`KV_CACHE_DEPTH_SINGLE_USER_WITH_GQA)];
                        nxt_cache_waddr[(SINGLE_USR_CACHE_ADDR_WIDTH-1) -: $clog2(CACHE_NUM)] = cmem_waddr[$clog2(`KV_CACHE_DEPTH_SINGLE_USER_WITH_GQA) +: $clog2(`MAC_MULT_NUM)];
                        nxt_cache_wdata = cmem_wdata;
                    end
                end
                else begin // 1 3 5 7
                    if(cmem_waddr[0 +: $clog2(`KV_CACHE_DEPTH_SINGLE_USER_WITH_GQA)] < K_CACHE_ADDR_BASE +  V_CACHE_ADDR_BASE) begin
                        nxt_cache_wen = 0;
                    end
                    else begin
                        nxt_cache_wen   = cmem_wen && cmem_waddr[CMEM_ADDR_WIDTH-1];
                        nxt_cache_waddr[0 +: $clog2(SINGLE_USR_CACHE_DEPTH)] = cmem_waddr[0 +: $clog2(`KV_CACHE_DEPTH_SINGLE_USER_WITH_GQA)] - V_CACHE_ADDR_BASE;
                        nxt_cache_waddr[(SINGLE_USR_CACHE_ADDR_WIDTH-1) -: $clog2(CACHE_NUM)] = cmem_waddr[$clog2(`KV_CACHE_DEPTH_SINGLE_USER_WITH_GQA) +: $clog2(`MAC_MULT_NUM)];
                        nxt_cache_wdata = cmem_wdata;
                    end
                end
            end
            else if(control_state_reg == V_GEN_STATE)begin
                if(HEAD_INDEX % 2 == 0)begin //0 2 4 6
                    if(cmem_waddr[0 +: $clog2(`KV_CACHE_DEPTH_SINGLE_USER_WITH_GQA)] >=  V_CACHE_ADDR_BASE + V_CACHE_ADDR_BASE) begin
                        nxt_cache_wen = 0;
                    end
                    else begin
                        nxt_cache_wen   = cmem_wen && cmem_waddr[CMEM_ADDR_WIDTH-1];
                        nxt_cache_waddr[0 +: $clog2(SINGLE_USR_CACHE_DEPTH)] = cmem_waddr[0 +: $clog2(`KV_CACHE_DEPTH_SINGLE_USER_WITH_GQA)];
                        nxt_cache_waddr[(SINGLE_USR_CACHE_ADDR_WIDTH-1) -: $clog2(CACHE_NUM)] = cmem_waddr[$clog2(`KV_CACHE_DEPTH_SINGLE_USER_WITH_GQA) +: $clog2(`MAC_MULT_NUM)];
                        nxt_cache_wdata = cmem_wdata;
                    end
                end
                else begin // 1 3 5 7
                    if(cmem_waddr[0 +: $clog2(`KV_CACHE_DEPTH_SINGLE_USER_WITH_GQA)] < V_CACHE_ADDR_BASE + V_CACHE_ADDR_BASE) begin
                        nxt_cache_wen = 0;
                    end
                    else begin
                        nxt_cache_wen   = cmem_wen && cmem_waddr[CMEM_ADDR_WIDTH-1];
                        nxt_cache_waddr[0 +: $clog2(SINGLE_USR_CACHE_DEPTH)] = cmem_waddr[0 +: $clog2(`KV_CACHE_DEPTH_SINGLE_USER_WITH_GQA)] - V_CACHE_ADDR_BASE;
                        nxt_cache_waddr[(SINGLE_USR_CACHE_ADDR_WIDTH-1) -: $clog2(CACHE_NUM)] = cmem_waddr[$clog2(`KV_CACHE_DEPTH_SINGLE_USER_WITH_GQA) +: $clog2(`MAC_MULT_NUM)];
                        nxt_cache_wdata = cmem_wdata;
                    end
                end
            end
        end
        // cache_wen   = cmem_wen && cmem_waddr[CMEM_ADDR_WIDTH-1];
        // cache_waddr[(SINGLE_USR_CACHE_ADDR_WIDTH-1) -: $clog2(CACHE_NUM)] = cmem_waddr[$clog2(`KV_CACHE_DEPTH_SINGLE_USER_WITH_GQA) +: $clog2(`MAC_MULT_NUM)];
        // cache_waddr[0 +: $clog2(SINGLE_USR_CACHE_DEPTH)] = cmem_waddr[0 +: $clog2(SINGLE_USR_CACHE_DEPTH)];
        // cache_wdata = cmem_wdata;
    end

    always_ff@(posedge clk or negedge rstn)begin
        if(~rstn)begin
            cache_wen <= 0;
            cache_waddr <= 0;
            cache_wdata <= 0;
        end
        else begin
            cache_wen <= nxt_cache_wen;
            cache_waddr <= nxt_cache_waddr;
            cache_wdata <= nxt_cache_wdata;
        end
    end


    always_comb begin
        cache_ren = cmem_ren && cmem_raddr[CMEM_ADDR_WIDTH-1];
        // cache_raddr[(ALL_USR_CACHE_ADDR_WIDTH-1) -: $clog2(CACHE_NUM)] = cmem_raddr[(SINGLE_USR_CACHE_ADDR_WIDTH-1) -: $clog2(CACHE_NUM)];
        // cache_raddr[0 +: $clog2(SINGLE_USR_CACHE_DEPTH)] = cmem_raddr[0 +: $clog2(SINGLE_USR_CACHE_DEPTH)];
        if(model_cfg.gqa_en == 1 && HEAD_INDEX %2 == 1)begin
            cache_raddr[0 +: $clog2(SINGLE_USR_CACHE_DEPTH)] = cmem_raddr[0 +: $clog2(`KV_CACHE_DEPTH_SINGLE_USER_WITH_GQA)] - V_CACHE_ADDR_BASE;
        end
        else begin
            cache_raddr[0 +: $clog2(SINGLE_USR_CACHE_DEPTH)] = cmem_raddr[0 +: $clog2(SINGLE_USR_CACHE_DEPTH)];
        end
        cache_raddr[(SINGLE_USR_CACHE_ADDR_WIDTH-1) -: $clog2(CACHE_NUM)] = 0;
    end

    // CACHE_ADDR W/R Selection
    always @(*) begin
        if (cache_wen) begin // Write
            cache_addr = cache_waddr;
        end
        else if(cache_ren)begin // Read
            cache_addr = cache_raddr;
        end
        else begin
            cache_addr = 0;
        end
    end


    // =============================================================================
    // Core-to-Core Link Channel
    assign vlink_data_out = cache_rdata;
    assign vlink_data_out_vld =  cache_rvalid;




endmodule

module wmem #(
    parameter   DATA_BIT =         (`MAC_MULT_NUM * `IDATA_WIDTH),
    parameter   WMEM_DEPTH =       `WMEM_DEPTH,
    parameter   WMEM_ADDR_WIDTH  =  $clog2(WMEM_DEPTH)
)
( 
    input  logic                                                clk,
    input  logic                                                rstn,

    input  logic                                                wmem_1024_ffn_deepslp,
    input  logic                                                wmem_1024_ffn_bc1,
    input  logic                                                wmem_1024_ffn_bc2,

    input  logic                                                wmem_512_attn_deepslp,
    input  logic                                                wmem_512_attn_bc1,
    input  logic                                                wmem_512_attn_bc2,

    input  logic [`CORE_MEM_ADDR_WIDTH-1:0]                     weight_mem_addr,
    input  logic [`INTERFACE_DATA_WIDTH-1:0]                    weight_mem_wdata,
    input  logic                                                weight_mem_wen,
    output logic [`INTERFACE_DATA_WIDTH-1:0]                    weight_mem_rdata,
    input  logic                                                weight_mem_ren,
    output logic                                                weight_mem_rvld,
  
    input  logic [WMEM_ADDR_WIDTH-1:0]                          wmem_addr,
 
    input  logic                                                wmem_ren,
    output logic [DATA_BIT-1:0]                                 wmem_rdata,
 
    input  logic                                                wmem_wen,
    input  logic [DATA_BIT-1:0]                                 wmem_wdata,
    input  logic [DATA_BIT-1:0]                                 wmem_bwe
);


logic [WMEM_ADDR_WIDTH-1:0]                          wmem_addr_f;
logic                                                wmem_ren_f;
logic                                                wmem_wen_f;
logic [DATA_BIT-1:0]                                 wmem_wdata_f;
logic [DATA_BIT-1:0]                                 wmem_bwe_f;


///////////////////////////////////////////
//              INTERFACE                //
///////////////////////////////////////////
logic                                                   interface_inst_ren;
logic  [$clog2(WMEM_DEPTH)-1:0]                         interface_inst_raddr;

logic                                                   interface_inst_wen;
logic  [$clog2(WMEM_DEPTH)-1:0]                         interface_inst_waddr;
logic  [DATA_BIT-1:0]                                   interface_inst_wdata;
logic  [DATA_BIT-1:0]                                   interface_inst_bwe;

//write
always_ff @(posedge clk or negedge rstn)begin
    if(~rstn)begin
        interface_inst_wen <= 0;
        interface_inst_waddr <= 0;
        interface_inst_wdata <= 0;
        interface_inst_bwe <= 0;
    end
    else begin
        if(weight_mem_wen)begin
            interface_inst_wen <= 1 && (weight_mem_addr[`CORE_MEM_ADDR_WIDTH-1 : `CORE_MEM_ADDR_WIDTH-2] != 0);
            interface_inst_waddr <= {(weight_mem_addr[`CORE_MEM_ADDR_WIDTH-1 -: 2] - 1'b1), weight_mem_addr[`CORE_MEM_ADDR_WIDTH-3 : $clog2(`MAC_MULT_NUM/2)]};
            interface_inst_wdata[weight_mem_addr[$clog2(`MAC_MULT_NUM/2)-1:0] * (2*`IDATA_WIDTH) +: 2*`IDATA_WIDTH] <= weight_mem_wdata;
            interface_inst_bwe  [weight_mem_addr[$clog2(`MAC_MULT_NUM/2)-1:0] * (2*`IDATA_WIDTH) +: 2*`IDATA_WIDTH] <= 16'hFFFF;
        end
        else begin
            interface_inst_bwe <= 0;
            interface_inst_wen <= 0;
        end
    end
end

//read
always_ff @(posedge clk or negedge rstn)begin
    if(~rstn)begin
        interface_inst_ren <= 0;
        interface_inst_raddr <= 0;
    end
    else begin
        if(weight_mem_ren)begin
            interface_inst_ren <= 1 && (weight_mem_addr[`CORE_MEM_ADDR_WIDTH-1 : `CORE_MEM_ADDR_WIDTH-2] != 0);
            interface_inst_raddr <= {(weight_mem_addr[`CORE_MEM_ADDR_WIDTH-1 -: 2] - 1'b1), weight_mem_addr[`CORE_MEM_ADDR_WIDTH-3 : $clog2(`MAC_MULT_NUM/2)]};
        end
        else begin
            interface_inst_ren <= 0;
        end
    end
end
logic [`CORE_MEM_ADDR_WIDTH-1:0] interface_addr_delay1;
logic [`CORE_MEM_ADDR_WIDTH-1:0] interface_addr_delay2;

logic                                                weight_mem_ren_delay1;
always_ff @(posedge clk or negedge rstn)begin
    if(~rstn)begin
        weight_mem_ren_delay1 <= 0;
        weight_mem_rvld <= 0;    
    end
    else begin
        weight_mem_ren_delay1 <= weight_mem_ren;
        weight_mem_rvld <= interface_inst_ren;
    end
end
assign weight_mem_rdata = wmem_rdata[interface_addr_delay2[$clog2(`MAC_MULT_NUM/2)-1:0] * (2*`IDATA_WIDTH) +: 2*`IDATA_WIDTH];

always_ff @(posedge clk or negedge rstn)begin
    if(~rstn)begin
        interface_addr_delay1 <= 0;
        interface_addr_delay2 <= 0;
    end
    else begin
        interface_addr_delay1 <= weight_mem_addr;
        interface_addr_delay2 <= interface_addr_delay1;
    end
end


always_comb begin
    wmem_addr_f = wmem_addr;
    wmem_ren_f = wmem_ren;
    wmem_wen_f = wmem_wen;
    wmem_wdata_f = wmem_wdata;
    wmem_bwe_f = wmem_bwe;
    if(interface_inst_wen)begin
        wmem_wen_f = interface_inst_wen;
        wmem_addr_f = interface_inst_waddr;
        wmem_wdata_f = interface_inst_wdata;
        wmem_bwe_f = interface_inst_bwe;
    end
    else if(interface_inst_ren)begin
        wmem_ren_f = interface_inst_ren;
        wmem_addr_f = interface_inst_raddr;
    end
end

`ifndef TRUE_MEM
 mem_sp #(
    .DATA_BIT(DATA_BIT),
    .DEPTH(WMEM_DEPTH),
    .ADDR_BIT(WMEM_ADDR_WIDTH),
    .BWE(1)
 )inst_mem_sp
(
    // Global Signals
    .clk(clk),
    .addr(wmem_addr_f),

    .wen(wmem_wen_f),
    .bwe(wmem_bwe_f),
    .wdata(wmem_wdata_f),
    .ren(wmem_ren_f),
    .rdata(wmem_rdata)
);

`else
true_mem_wrapper true_mem_wrapper(
    .clk(clk),
    .rst_n(rstn),

    .wmem_addr(wmem_addr_f),
    .wmem_ren(wmem_ren_f),
    .wmem_wen(wmem_wen_f),
    .wmem_wdata(wmem_wdata_f),
    .wmem_bwe(wmem_bwe_f),

    .wmem_rdata(wmem_rdata),

    //power ctrl
    .wmem_1024_ffn_deepslp(wmem_1024_ffn_deepslp),
    .wmem_1024_ffn_bc1(wmem_1024_ffn_bc1),
    .wmem_1024_ffn_bc2(wmem_1024_ffn_bc2),

    .wmem_512_attn_deepslp(wmem_512_attn_deepslp),
    .wmem_512_attn_bc1(wmem_512_attn_bc1),
    .wmem_512_attn_bc2(wmem_512_attn_bc2)
);

`endif

endmodule





/****** KV Cache PKT w/ BIT WRITE ENABLE ********/
module kv_cache_pkt #(
    parameter   IDATA_WIDTH =         (`MAC_MULT_NUM * `IDATA_WIDTH),
    parameter   ODATA_BIT =         (`MAC_MULT_NUM * `IDATA_WIDTH),
    parameter   CACHE_NUM =         `MAC_MULT_NUM,
    parameter   CACHE_DEPTH =       `KV_CACHE_DEPTH_SINGLE_USER,
    parameter   CACHE_ADDR_WIDTH  = ($clog2(CACHE_NUM)+$clog2(CACHE_DEPTH))
)
(
    input                           clk,
    input                           rstn,

    input  logic                                                clean_kv_cache, //pulse
    input  logic       [`USER_ID_WIDTH-1:0]                     clean_kv_cache_user_id, //clean the corrsponding user's kv cache to zero

    input  logic [`CORE_MEM_ADDR_WIDTH-1:0]                     kv_mem_addr,
    input  logic [`INTERFACE_DATA_WIDTH-1:0]                    kv_mem_wdata,
    input  logic                                                kv_mem_wen,
    output logic [`INTERFACE_DATA_WIDTH-1:0]                    kv_mem_rdata,
    input  logic                                                kv_mem_ren,
    output logic                                                kv_mem_rvld,

    input  [CACHE_ADDR_WIDTH-1:0]   cache_addr,

    input                           cache_ren,
    output [ODATA_BIT-1:0]          cache_rdata,
 
    input                           cache_wen,
    input  [IDATA_WIDTH-1:0]        cache_wdata,
    input                           cache_wdata_byte_flag, //if cache_wdata_byte_flag == 1, the address is byte based-BUCK

    input       USER_CONFIG         usr_cfg

);  
    reg                                                       inst_wen;
    reg                                                       inst_ren;
    
    reg  [$clog2(`MAX_NUM_USER*CACHE_DEPTH)-1:0]              inst_waddr;
    reg  [$clog2(`MAX_NUM_USER*CACHE_DEPTH)-1:0]              inst_raddr;
    reg  [$clog2(`MAX_NUM_USER*CACHE_DEPTH)-1:0]              inst_addr;

    reg   [ODATA_BIT-1:0]                                     inst_wdata;
    reg   [ODATA_BIT-1:0]                                     inst_rdata;
    reg   [ODATA_BIT-1:0]                                     inst_bwe;

    wire [$clog2(CACHE_NUM)-1:0]                              bank_sel;       //sel which bank
    wire [$clog2(CACHE_DEPTH)-1:0]                            bank_addr;      //sel the addr in bank
    
    assign bank_sel = cache_addr[(CACHE_ADDR_WIDTH-1) -: $clog2(CACHE_NUM)];
    assign bank_addr = cache_addr[0 +: $clog2(CACHE_DEPTH)];

    //write
    reg                                                       inst_wen_w;
    reg [$clog2(`MAX_NUM_USER*CACHE_DEPTH)-1:0]               inst_waddr_w;
    reg [ODATA_BIT-1:0]                                       inst_wdata_w;
    reg [ODATA_BIT-1:0]                                       inst_bwe_w;


    logic       [`USER_ID_WIDTH-1:0]                     clean_kv_cache_user_id_reg;
    always_ff@(posedge clk or negedge rstn)begin
        if(~rstn)begin
            clean_kv_cache_user_id_reg <= 0;
        end
        else if(clean_kv_cache)begin
            clean_kv_cache_user_id_reg <= clean_kv_cache_user_id;
        end
    end
    logic clean_kv_cache_finish;
    logic nxt_clean_kv_cache_finish;
    logic clean_kv_cache_flag;

    logic                                         clean_wen;
    logic [$clog2(`MAC_MULT_NUM*CACHE_DEPTH)-1:0] clean_addr;
    logic                                         nxt_clean_wen;
    logic [$clog2(`MAC_MULT_NUM*CACHE_DEPTH)-1:0] nxt_clean_addr;
    logic [ODATA_BIT-1:0]                         clean_wdata;

    assign clean_wdata = 0;
    assign clean_finish = clean_kv_cache_finish;

    always_ff@(posedge clk or negedge rstn)begin
        if(~rstn)begin
            clean_kv_cache_flag <= 0;
        end
        else if(clean_kv_cache_finish)begin
            clean_kv_cache_flag <= 0;
        end
        else if (clean_kv_cache)begin
            clean_kv_cache_flag <= 1;
        end
    end

    always_comb begin
        nxt_clean_wen = clean_wen;
        nxt_clean_addr = clean_addr;
        nxt_clean_kv_cache_finish = 0;
        if(clean_kv_cache)begin
            nxt_clean_wen = 0;
            nxt_clean_wen = 1;
            nxt_clean_addr = clean_kv_cache_user_id * `KV_CACHE_DEPTH_SINGLE_USER;
        end
        else if(clean_kv_cache_flag && clean_wen)begin
            if(clean_addr == clean_kv_cache_user_id_reg  * `KV_CACHE_DEPTH_SINGLE_USER + `KV_CACHE_DEPTH_SINGLE_USER -1)begin
                nxt_clean_kv_cache_finish = 1;
                nxt_clean_wen = 0;
                nxt_clean_addr = 0;
            end
            else begin
                nxt_clean_addr = clean_addr + 1;
            end
        end
    end

    always_ff@(posedge clk or negedge rstn)begin
        if(~rstn)begin
            clean_wen <= 0;
            clean_addr <= 0;
            clean_kv_cache_finish <= 0;
        end
        else begin
            clean_wen <= nxt_clean_wen;
            clean_addr <= nxt_clean_addr;
            clean_kv_cache_finish <= nxt_clean_kv_cache_finish;
        end
    end


    always @(*) begin
        inst_wen_w='b0;
        inst_waddr_w= bank_addr + usr_cfg.user_id*CACHE_DEPTH;
        inst_wdata_w='b0;
        inst_bwe_w = {IDATA_WIDTH{1'b0}};
        for(integer i=0;i<CACHE_NUM;i++) begin
            if((i==bank_sel)&&cache_wen) begin
                inst_wen_w = 1'b1;
                inst_wdata_w[i*`IDATA_WIDTH+:`IDATA_WIDTH] = cache_wdata[`IDATA_WIDTH-1:0];
                inst_bwe_w[i*`IDATA_WIDTH+:`IDATA_WIDTH] = {`IDATA_WIDTH{1'b1}};
            end
        end
    end
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            inst_wen<='0;
            inst_waddr<='b0;
            inst_wdata<='0;
            inst_bwe <= '0;
        end
        else if(cache_wen) begin
            inst_wen <= inst_wen_w;
            inst_waddr<=inst_waddr_w;
            inst_wdata<=inst_wdata_w;
            inst_bwe<=inst_bwe_w;
        end
        else begin
            inst_wen <= 0;
            inst_bwe <= 0;
        end
    end

    //read
    reg                             inst_ren_w;
    reg [$clog2(`MAX_NUM_USER*CACHE_DEPTH)-1:0] inst_raddr_w;
    always @(*) begin
        inst_ren_w=1'b0;
        inst_raddr_w='b0;
        if(cache_ren) begin
            inst_ren_w=1'b1;
            inst_raddr_w= bank_addr + usr_cfg.user_id*CACHE_DEPTH;
        end
    end

    always@(*)begin
        inst_ren = inst_ren_w;
        inst_raddr = inst_raddr_w;
    end

    logic [`CORE_MEM_ADDR_WIDTH-1:0] interface_addr_delay1;
    logic [`CORE_MEM_ADDR_WIDTH-1:0] interface_addr_delay2;


    assign cache_rdata = inst_rdata;
    assign kv_mem_rdata = inst_rdata[interface_addr_delay2[$clog2(`MAC_MULT_NUM/2)-1:0] * (2*`IDATA_WIDTH) +: 2*`IDATA_WIDTH];


    //inst addr selection
    assign inst_addr = inst_wen ? inst_waddr : inst_raddr;

    logic  inst_wen_f;
    logic  inst_ren_f;
    logic [$clog2(`MAX_NUM_USER*CACHE_DEPTH)-1:0] inst_addr_f;
    logic [ODATA_BIT-1:0]                         inst_wdata_f;
    logic [ODATA_BIT-1:0]                         inst_bwe_f;


///////////////////////////////////////////
//              INTERFACE                //
///////////////////////////////////////////
logic                                                      interface_inst_ren;
logic  [$clog2(`MAX_NUM_USER*CACHE_DEPTH)-1:0]             interface_inst_raddr;

logic                                                      interface_inst_wen;
logic  [$clog2(`MAX_NUM_USER*CACHE_DEPTH)-1:0]             interface_inst_waddr;
logic  [IDATA_WIDTH-1:0]                                   interface_inst_wdata;
logic  [IDATA_WIDTH-1:0]                                   interface_inst_bwe;

//write
always_ff @(posedge clk or negedge rstn)begin
    if(~rstn)begin
        interface_inst_wen <= 0;
        interface_inst_waddr <= 0;
        interface_inst_wdata <= 0;
        interface_inst_bwe <= 0;
    end
    else begin
        if(kv_mem_wen)begin
            interface_inst_wen <= 1 && (kv_mem_addr[`CORE_MEM_ADDR_WIDTH-1 : `CORE_MEM_ADDR_WIDTH-2] == 0);
            interface_inst_waddr <= kv_mem_addr[$clog2(`MAC_MULT_NUM/2) +: $clog2(`MAX_NUM_USER*CACHE_DEPTH)];
            interface_inst_wdata[kv_mem_addr[$clog2(`MAC_MULT_NUM/2)-1:0] * (2*`IDATA_WIDTH) +: 2*`IDATA_WIDTH] <= kv_mem_wdata;
            interface_inst_bwe  [kv_mem_addr[$clog2(`MAC_MULT_NUM/2)-1:0] * (2*`IDATA_WIDTH) +: 2*`IDATA_WIDTH] <= 16'hFFFF;
        end
        else begin
            interface_inst_bwe <= 0;
            interface_inst_wen <= 0;
        end
    end
end

//read
always_ff @(posedge clk or negedge rstn)begin
    if(~rstn)begin
        interface_inst_ren <= 0;
        interface_inst_raddr <= 0;
    end
    else begin
        if(kv_mem_ren)begin
            interface_inst_ren <= 1 && (kv_mem_addr[`CORE_MEM_ADDR_WIDTH-1 : `CORE_MEM_ADDR_WIDTH-2] == 0);
            interface_inst_raddr <= kv_mem_addr[$clog2(`MAC_MULT_NUM/2) +: $clog2(`MAX_NUM_USER*CACHE_DEPTH)];
        end
        else begin
            interface_inst_ren <= 0;
        end
    end
end

logic                                                kv_mem_ren_delay1;
always_ff @(posedge clk or negedge rstn)begin
    if(~rstn)begin
        kv_mem_ren_delay1 <= 0;
        kv_mem_rvld <= 0;    
    end
    else begin
        kv_mem_ren_delay1 <= kv_mem_ren;
        kv_mem_rvld <= interface_inst_ren;
    end
end




always_ff @(posedge clk or negedge rstn)begin
    if(~rstn)begin
        interface_addr_delay1 <= 0;
        interface_addr_delay2 <= 0;
    end
    else begin
        interface_addr_delay1 <= kv_mem_addr;
        interface_addr_delay2 <= interface_addr_delay1;
    end
end

    always_comb begin
        inst_ren_f = inst_ren;
        inst_wen_f = inst_wen;
        inst_addr_f = inst_addr;
        inst_wdata_f = inst_wdata;
        inst_bwe_f = inst_bwe;

        if(clean_kv_cache_flag)begin
            inst_wen_f = clean_wen;
            inst_addr_f = clean_addr;
            inst_wdata_f = clean_wdata;
            inst_bwe_f = {IDATA_WIDTH{1'b1}};
        end
        else if(interface_inst_wen)begin
            inst_wen_f = interface_inst_wen;
            inst_addr_f = interface_inst_waddr;
            inst_wdata_f = interface_inst_wdata;
            inst_bwe_f = interface_inst_bwe;
        end
        else if(interface_inst_ren)begin
            inst_ren_f = interface_inst_ren;
            inst_addr_f = interface_inst_raddr;
        end
    end


`ifdef TRUE_MEM
   kv_cache_bwe_nopme  kv_cache_bwe_nopme_r(//lintra s-68000
   .clk(clk),             		//Input Clock
   .ren(inst_ren_f),                    	//Read Enable
   .wen(inst_wen_f),                    	//Write Enable
   .adr(inst_addr_f),         //Input Address
   .mc(3'b0),     		//Controls extending write duration
   .mcen(1'b0),     			//Enable read margin control 

   .clkbyp(1'b0),                   	//clock bypass enable  
   .din(inst_wdata_f[63:0]),       //Input Write Data 
   .wbeb(~inst_bwe_f[63:0]),  //Write Bit enable

   .wa(2'b0),
   .wpulse(2'b0),
   .wpulseen(1'b1),
   .fwen(!rstn),

   .q(inst_rdata[63:0])
);

   kv_cache_bwe_nopme  kv_cache_bwe_nopme_l(//lintra s-68000
   .clk(clk),             		//Input Clock
   .ren(inst_ren_f),                    	//Read Enable
   .wen(inst_wen_f),                    	//Write Enable
   .adr(inst_addr_f),         //Input Address
   .mc(3'b0),     		//Controls extending write duration
   .mcen(1'b0),     			//Enable read margin control 

   .clkbyp(1'b0),                   	//clock bypass enable  
   .din(inst_wdata_f[127:64]),       //Input Write Data 
   .wbeb(~inst_bwe_f[127:64]),  //Write Bit enable

   .wa(2'b0),
   .wpulse(2'b0),
   .wpulseen(1'b1),
   .fwen(!rstn),

   .q(inst_rdata[127:64])
);




`else
    mem_sp #(.DATA_BIT(`IDATA_WIDTH*`MAC_MULT_NUM), 
             .DEPTH(CACHE_DEPTH*`MAX_NUM_USER),
             .BWE(1)
            ) 
        kv_cache(
        // Global Signals
        .clk(clk),
        .addr(inst_addr_f),

        .wen(inst_wen_f),
        .wdata(inst_wdata_f),
        .bwe(inst_bwe_f),
        .ren(inst_ren_f),
        .rdata(inst_rdata)
    );
`endif 
endmodule


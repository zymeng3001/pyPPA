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


module core_tb # (
    parameter GBUS_DATA_WIDTH = `GBUS_DATA_WIDTH,
    parameter GBUS_ADDR_WIDTH = `GBUS_ADDR_WIDTH,

    parameter WMEM_DEPTH = `WMEM_DEPTH,
    parameter GSRAM_DEPTH = `GLOBAL_SRAM_DEPTH,

    parameter CACHE_DEPTH = `KV_CACHE_DEPTH_SINGLE_USER,
    parameter CACHE_NUM = `MAC_MULT_NUM,
    parameter CACHE_ADDR_WIDTH  = ($clog2(CACHE_NUM)+$clog2(CACHE_DEPTH)),

    parameter MAC_MULT_NUM = `MAC_MULT_NUM,
    parameter IDATA_WIDTH = `IDATA_WIDTH,
    parameter ODATA_BIT = `ODATA_WIDTH,
    parameter CDATA_ACCU_NUM_WIDTH = `CDATA_ACCU_NUM_WIDTH,
    parameter CDATA_SCALE_WIDTH = `CDATA_SCALE_WIDTH,
    parameter CDATA_BIAS_WIDTH = `CDATA_BIAS_WIDTH,
    parameter CDATA_SHIFT_WIDTH = `CDATA_SHIFT_WIDTH,
                
    parameter   VLINK_DATA_WIDTH  = `GBUS_DATA_WIDTH,
    parameter   HLINK_DATA_WIDTH  = `GBUS_DATA_WIDTH,

    parameter   CMEM_ADDR_WIDTH   = `CMEM_ADDR_WIDTH,
    parameter   CMEM_DATA_WIDTH   = `GBUS_DATA_WIDTH,

    parameter   CORE_INDEX        = 0,

    parameter CFG_ACC_NUM = `MAX_EMBD_SIZE/`MAC_MULT_NUM, 
    parameter CFG_QUANT_SCALE = 1,
    parameter CFG_QUANT_BIAS = 0,
    parameter CFG_QUANT_SHIFT = 9,
    parameter COMPUTE_LENGTH = `MAX_EMBD_SIZE*`QKV_WEIGHT_COLS_PER_CORE / `MAC_MULT_NUM, //这样刚好算Q第一行的前QKV_WEIGHT_COLS_PER_CORE列
    parameter WEIGHT_BASE_ADDR = 0,
    parameter IN_BASE_ADDR = 0
    );


logic clk;
logic rstn;
//Global Config Signals
logic                                   cfg_vld;
logic [CDATA_ACCU_NUM_WIDTH-1:0]        cfg_acc_num;
logic [CDATA_SCALE_WIDTH-1:0]           cfg_quant_scale;
logic [CDATA_BIAS_WIDTH-1:0]            cfg_quant_bias;
logic [CDATA_SHIFT_WIDTH-1:0]           cfg_quant_shift;

CONTROL_STATE                   control_state;
logic                           control_state_update;

logic start;
logic finish;

// Channel - Global Bus to Access Core Memory and MAC Result
logic [GBUS_ADDR_WIDTH-1:0]     in_gbus_addr;
logic                           in_gbus_wen;
logic [GBUS_DATA_WIDTH-1:0]     in_gbus_wdata;

logic  [GBUS_ADDR_WIDTH-1:0]    out_gbus_addr;
logic                           out_gbus_wen;
logic  [GBUS_DATA_WIDTH-1:0]    out_gbus_wdata;

// Vertical for Weight and Key/Value Propagation
logic vlink_enable;
logic [GBUS_DATA_WIDTH-1:0]     vlink_wdata;
logic                           vlink_wen;
logic [GBUS_DATA_WIDTH-1:0]     vlink_rdata; //output
logic                           vlink_rvalid; //output

// Horizontal for Activation Propagation
logic [GBUS_DATA_WIDTH-1:0]     hlink_wdata;
logic                           hlink_wen;
logic [GBUS_DATA_WIDTH-1:0]     hlink_rdata; //output
logic                           hlink_rvalid; //output


// logic [CMEM_ADDR_WIDTH-1:0]     core_in_cmem_raddr;
// logic                           core_in_cmem_ren;
// logic [CMEM_ADDR_WIDTH-1:0]     core_out_cmem_raddr;
// logic                           core_out_cmem_ren;

core_top #(
    .GBUS_DATA_WIDTH(GBUS_DATA_WIDTH)   ,
    .GBUS_ADDR_WIDTH(GBUS_ADDR_WIDTH)   ,
    .WMEM_DEPTH(WMEM_DEPTH)        ,
    .CACHE_DEPTH(CACHE_DEPTH)       ,
    .CACHE_NUM(CACHE_NUM)         ,
    .CMEM_ADDR_WIDTH(CMEM_ADDR_WIDTH)   ,
    .CMEM_DATA_WIDTH(CMEM_DATA_WIDTH)   ,
    .VLINK_DATA_WIDTH(VLINK_DATA_WIDTH)  ,
    .HLINK_DATA_WIDTH(HLINK_DATA_WIDTH)  ,
    .MAC_MULT_NUM(MAC_MULT_NUM)           ,
    .IDATA_WIDTH(IDATA_WIDTH)         ,
    .ODATA_BIT(ODATA_BIT)         ,
    .CORE_INDEX(CORE_INDEX)        

) core_inst
(.*);

always begin
    #50 clk = ~clk;
end

logic [GBUS_DATA_WIDTH-1:0] activation;
logic [GBUS_ADDR_WIDTH-1:0] weight_base_addr;
logic [GBUS_ADDR_WIDTH-1:0] in_base_addr;
logic [GBUS_ADDR_WIDTH-1:0] weight_raddr;
integer compute_length;
logic [GBUS_DATA_WIDTH-1:0] wmem_mem [WMEM_DEPTH-1:0];
logic [GBUS_DATA_WIDTH-1:0] in_mem [GSRAM_DEPTH-1:0];

integer result_file;
integer expected_file;
integer write_back_file;

task single_compute_wmem(input [GBUS_DATA_WIDTH-1:0] activation, input [GBUS_ADDR_WIDTH-1:0] weight_raddr);
    @(negedge clk)
    hlink_wen=1'b1;
    hlink_wdata=activation;
    // force core_inst.cmem_ren=1'b1;
    // // release core_inst.cmem_ren;
    // force core_inst.cmem_raddr=weight_raddr;
    // force core_inst.cmem_raddr[CMEM_ADDR_WIDTH-1]=1'b0; //read wmem

endtask

task burst_compute_wmem(input [GBUS_ADDR_WIDTH-1:0] weight_base_addr,input [GBUS_ADDR_WIDTH-1:0] in_base_addr, input integer compute_length);
    integer w_addr;
    integer in_addr;
    logic [GBUS_DATA_WIDTH-1:0] data;

    w_addr=weight_base_addr;
    in_addr=in_base_addr;

    for(int i=0;i<compute_length;i++) begin
        data=in_mem[in_addr];
        // release core_inst.cmem_raddr;   
        single_compute_wmem(data,w_addr);//w_addr and activation data in same phase for core input？ -BUCK
        w_addr=w_addr+1;
        in_addr=in_addr+1;
    end

    @(negedge clk)
    hlink_wen=1'b0;
    hlink_wdata='b0;
    // force core_inst.cmem_ren=1'b0;
    // force core_inst.cmem_raddr='b0;

endtask

integer expected_result_array [COMPUTE_LENGTH/CFG_ACC_NUM + 1];
integer expected_result_array_wr_ptr = 0;

task expected_compute_wmem (input [GBUS_ADDR_WIDTH-1:0] weight_base_addr,input [GBUS_ADDR_WIDTH-1:0] in_base_addr,input integer compute_length);
    integer offset;
    integer mac_result;
    logic [31:0] quant_result;
    logic [31:0] quant_bias;
    integer rnd_result;
    integer expected_result;
    integer quant_shift;
    logic [GBUS_DATA_WIDTH-1:0] weight;
    logic [MAC_MULT_NUM-1:0][IDATA_WIDTH-1:0] data_split;
    logic [MAC_MULT_NUM-1:0][IDATA_WIDTH-1:0] weight_split;
    logic quant_round;

    offset = 0;
    mac_result = 0;
    for(int i=0;i<compute_length;i++) begin
        data_split = in_mem[in_base_addr+offset];
        weight_split = wmem_mem[weight_base_addr+offset];
        //Multiply Add
        if((i+1)%CFG_ACC_NUM!=0) begin
            for(int k=0;k<MAC_MULT_NUM;k++) begin
                mac_result=$signed(mac_result)+$signed(data_split[k])*$signed(weight_split[k]);
            end
            offset = offset + 1;
            continue;
        end

        for(int k=0;k<MAC_MULT_NUM;k++) begin
            mac_result=$signed(mac_result)+$signed(data_split[k])*$signed(weight_split[k]);
        end
        quant_bias = $signed(mac_result)*$signed(CFG_QUANT_SCALE)+$signed(CFG_QUANT_BIAS) ;
        quant_round = quant_bias[CFG_QUANT_SHIFT-1];
        quant_shift = CFG_QUANT_SHIFT;
        quant_result = $signed(quant_bias)>>>CFG_QUANT_SHIFT;
        rnd_result = quant_result + quant_round;
        if(rnd_result > 127)
            expected_result = 127;
        else if(rnd_result<-128)
            expected_result = -128;
        else
            expected_result = rnd_result;

        $fdisplay(expected_file, "acc_result = %d, quant_bias = %d, quant_result = %d, rnd_result = %d, Result = %d,i=%d",
                                mac_result,   $signed(quant_bias), $signed(quant_result), rnd_result ,expected_result,i);
        expected_result_array[expected_result_array_wr_ptr] = expected_result;
        expected_result_array_wr_ptr = expected_result_array_wr_ptr + 1;
        
        mac_result=0;
        quant_result=0;
        rnd_result=0;
        expected_result=0;

        offset = offset+1;
    end
endtask



initial begin
    @(posedge rstn)
    // core_inst.mem_inst.wmem_inst.INTC_MEM_INIT("../mems/hex/wmem_512.hex");
    $readmemh("../mems/hex/wmem_512.hex",core_inst.mem_inst.wmem_inst.inst_mem_sp.mem);
end


logic [IDATA_WIDTH-1:0] quant_odata_array [COMPUTE_LENGTH/CFG_ACC_NUM + 1];
integer quant_odata_array_wr_ptr = 0;
always begin
    @(negedge clk);
    #0.1;
    if(core_inst.quant_odata_valid) begin
        $fdisplay(result_file, "Result = %d, time: %0t",$signed(core_inst.quant_odata), $time);
        quant_odata_array[quant_odata_array_wr_ptr] = core_inst.quant_odata;
        quant_odata_array_wr_ptr = quant_odata_array_wr_ptr + 1;
    end
        
end
int error_flag = 0;
initial begin
	$fsdbDumpfile("core.fsdb");
	$fsdbDumpvars(0, core_tb);
    result_file = $fopen("../comp/core/tb/results.txt", "w");
    expected_file = $fopen("../comp/core/tb/expected.txt","w");
    write_back_file = $fopen("../comp/core/tb/write_back.txt","w");
    cfg_vld = 0;
    clk = 0;
    rstn = 0;
    control_state = IDLE_STATE;
    control_state_update = 0;
    start = 0;
    hlink_wen = 0;
    //configuration
    cfg_acc_num = CFG_ACC_NUM;
    cfg_quant_scale = CFG_QUANT_SCALE;
    cfg_quant_bias = CFG_QUANT_BIAS;
    cfg_quant_shift = CFG_QUANT_SHIFT;
    @(negedge clk);
    @(negedge clk);
    rstn = 1;
    repeat(10) @(negedge clk);
    control_state =Q_GEN_STATE;
    control_state_update = 1;
    start = 1;
    cfg_vld = 1;
    @(negedge clk);
    control_state_update = 0;
    start = 0;
    cfg_vld = 0;

    $readmemh("../mems/hex/wmem_512.hex",wmem_mem);
    $readmemh("../mems/hex/global_sram.hex",in_mem);



    //default value
    in_gbus_addr = 'b0;
    in_gbus_wen = 1'b0;
    in_gbus_wdata = 'b0;
    vlink_enable = 1'b1;
    vlink_wdata = 'b0;
    vlink_wen = 1'b0;
    hlink_wen = 1'b0;
    hlink_wdata = 'b0;
    // force core_inst.cmem_raddr = 'b0;
    // force core_inst.cmem_ren = 1'b0;

    
    expected_compute_wmem(WEIGHT_BASE_ADDR,IN_BASE_ADDR,COMPUTE_LENGTH);
    burst_compute_wmem(WEIGHT_BASE_ADDR,IN_BASE_ADDR,COMPUTE_LENGTH);
    // release core_inst.cmem_ren;
    // release core_inst.cmem_raddr;
    // repeat(1000) begin
    //     @(negedge clk);
    // end
    

    repeat(5000) begin
        @(negedge clk);
    end
    $fclose(expected_file);
    $fclose(result_file);
    $fclose(write_back_file);
    $display("Start comparing");
    for(int i =0; i < COMPUTE_LENGTH/CFG_ACC_NUM;i++)begin
        if($signed(quant_odata_array[i]) !== expected_result_array[i])begin
            $display("quant_odata_array[%0d]:%0d != expected_result_array[%0d]:%0d",i,$signed(quant_odata_array[i]),i,expected_result_array[i]);
            error_flag = 1;
        end
    end
    quant_odata_array_wr_ptr = 0;
    expected_result_array_wr_ptr = 0;
    if(error_flag == 0)begin
        $display("NO ERRROR");
    end
    else begin
        $display("ERROR!!!");
    end
    $display("Finish successfully");
    $finish;
end

endmodule

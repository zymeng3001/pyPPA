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

module global_sram_wrapper #(
    parameter DATA_BIT = (`MAC_MULT_NUM * `IDATA_WIDTH),
    parameter DEPTH = `GLOBAL_SRAM_DEPTH
) (
    input                                                  clk,
    input                                                  rst_n,
    input  [$clog2(DEPTH)-1:0]                             global_sram_waddr,
    input                                                  global_sram_wen,
    input [DATA_BIT-1:0]                                   global_sram_wdata,
    input [DATA_BIT-1:0]                                   global_sram_bwe,
    input  [$clog2(DEPTH)-1:0]                             global_sram_raddr,
    input                                                  global_sram_ren,
    output logic [DATA_BIT-1:0]                            global_sram_rdata,
    output logic                                           global_sram_rvld,

    //interface
    input  logic [`GLOBAL_MEM_ADDR_WIDTH-1:0]              global_mem_addr,
    input  logic [`GLOBAL_MEM_DATA_WIDTH-1:0]              global_mem_wdata,
    input  logic                                           global_mem_wen,
    output logic [`GLOBAL_MEM_DATA_WIDTH-1:0]              global_mem_rdata,
    input  logic                                           global_mem_ren,
    output logic                                           global_mem_rvld
);


logic  [$clog2(DEPTH)-1:0]                             global_sram_waddr_f;
logic                                                  global_sram_wen_f;
logic [DATA_BIT-1:0]                                   global_sram_wdata_f;
logic [DATA_BIT-1:0]                                   global_sram_bwe_f;
logic  [$clog2(DEPTH)-1:0]                             global_sram_raddr_f;
logic                                                  global_sram_ren_f;

///////////////////////////////////////////
//              INTERFACE                //
///////////////////////////////////////////
logic  [$clog2(DEPTH)-1:0]                             interface_waddr;
logic                                                  interface_wen;
logic [DATA_BIT-1:0]                                   interface_wdata;
logic [DATA_BIT-1:0]                                   interface_bwe;
logic  [$clog2(DEPTH)-1:0]                             interface_raddr;
logic                                                  interface_ren;

logic [`GLOBAL_MEM_ADDR_WIDTH-1:0]              global_mem_addr_delay1;
logic [`GLOBAL_MEM_ADDR_WIDTH-1:0]              global_mem_addr_delay2;
always_ff @(posedge clk) begin
    global_mem_addr_delay1 <= global_mem_addr;
    global_mem_addr_delay2 <= global_mem_addr_delay1;
end


//write
always_ff @(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        interface_wen <= 0;
        interface_waddr <= 0;
        interface_wdata <= 0;
        interface_bwe <= 0;
    end
    else begin
        if(global_mem_wen)begin
            interface_wen <= 1;
            interface_waddr <= global_mem_addr[$clog2(`MAC_MULT_NUM/2) +: $clog2(DEPTH)];
            interface_wdata[global_mem_addr[$clog2(`MAC_MULT_NUM/2)-1:0] * (2*`IDATA_WIDTH) +: 2*`IDATA_WIDTH] <= global_mem_wdata;
            interface_bwe  [global_mem_addr[$clog2(`MAC_MULT_NUM/2)-1:0] * (2*`IDATA_WIDTH) +: 2*`IDATA_WIDTH] <= 16'hFFFF;
        end
        else begin
            interface_bwe <= 0;
            interface_wen <= 0;
        end
    end
end

//read
always_ff @(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        interface_ren <= 0;
        interface_raddr <= 0;
    end
    else begin
        if(global_mem_ren)begin
            interface_ren <= 1;
            interface_raddr <= global_mem_addr[$clog2(`MAC_MULT_NUM/2) +: $clog2(DEPTH)];
        end
        else begin
            interface_ren <= 0;
        end
    end
end


always_ff @(posedge clk or negedge rst_n)begin
    if(~rst_n)
        global_sram_rvld <= 0;
    else if(global_sram_ren)
        global_sram_rvld <= 1;
    else
        global_sram_rvld <= 0;
end

always_ff @(posedge clk or negedge rst_n)begin
    if(~rst_n)
        global_mem_rvld <= 0;
    else if(interface_ren)
        global_mem_rvld <= 1;
    else
        global_mem_rvld <= 0;
end

always_comb begin
    global_sram_waddr_f = global_sram_waddr;
    global_sram_wen_f = global_sram_wen;
    global_sram_wdata_f = global_sram_wdata;
    global_sram_bwe_f = global_sram_bwe;
    global_sram_raddr_f = global_sram_raddr;
    global_sram_ren_f = global_sram_ren;
    if(interface_wen)begin
        global_sram_waddr_f = interface_waddr;
        global_sram_wen_f = interface_wen;
        global_sram_wdata_f = interface_wdata;
        global_sram_bwe_f = interface_bwe;
    end
    else if(interface_ren)begin
        global_sram_raddr_f = interface_raddr;
        global_sram_ren_f = interface_ren;
    end
end

assign global_mem_rdata = global_sram_rdata[global_mem_addr_delay2[$clog2(`MAC_MULT_NUM/2)-1:0] * (2*`IDATA_WIDTH) +: 2*`IDATA_WIDTH];

`ifndef TRUE_MEM
mem_dp  #(.DATA_BIT(DATA_BIT), .DEPTH(DEPTH), .BWE(1)) inst_mem_dp (
    .clk                    (clk),
    .waddr                  (global_sram_waddr_f),
    .wen                    (global_sram_wen_f),
    .bwe                    (global_sram_bwe_f),
    .wdata                  (global_sram_wdata_f),
    .raddr                  (global_sram_raddr_f),
    .ren                    (global_sram_ren_f),
    .rdata                  (global_sram_rdata)
);
`else
    global_sram ram_piece (
    .ickwp0(clk),
    .iwenp0(global_sram_wen_f),
    .iawp0(global_sram_waddr_f),
    .idinp0(global_sram_wdata_f),
    .ibwep0(global_sram_bwe_f),
    .ickrp0(clk),
    .irenp0(global_sram_ren_f),
    .iarp0(global_sram_raddr_f),
    .iclkbyp(1'b1),
    .imce(1'b0),
    .irmce(2'b0),
    .ifuse(1'b1),
    .iwmce(4'b0),
    .odoutp0(global_sram_rdata)
    );
`endif
endmodule

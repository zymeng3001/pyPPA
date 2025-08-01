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
module residual_sram #(
    parameter   DATA_WIDTH =       (`MAC_MULT_NUM * `IDATA_WIDTH),
    parameter   BANK_NUM =         `MAC_MULT_NUM,
    parameter   BANK_DEPTH =       `GLOBAL_SRAM_DEPTH,
    parameter   ADDR_WIDTH  = ($clog2(BANK_NUM)+$clog2(BANK_DEPTH)),
    parameter   SINGLE_SLICE_DATA_BIT = 8
)
(
    input                           clk,
    input                           rstn,

    input  [$clog2(BANK_DEPTH) + $clog2(`MAC_MULT_NUM/2)-1:0] interface_addr,

    input                                                     interface_ren,
    output [`INTERFACE_DATA_WIDTH-1:0]                        interface_rdata,
    output logic                                              interface_rvalid,
    
    input                                                     interface_wen,
    input  [`INTERFACE_DATA_WIDTH-1:0]                        interface_wdata,

    input  [ADDR_WIDTH-1:0]         raddr,
    input                           ren,
    output [DATA_WIDTH-1:0]         rdata,
    
    input  [ADDR_WIDTH-1:0]         waddr,
    input                           wen,
    input  [DATA_WIDTH-1:0]         wdata,

    input                           wdata_byte_flag //if wdata_byte_flag == 1, the address is byte based-BUCK
);  
    reg                                                      inst_wen;
    reg                                                      inst_ren;
    
    reg  [DATA_WIDTH-1:0]                                    inst_bwe;

    reg  [$clog2(BANK_DEPTH)-1:0]                            inst_waddr;
    reg  [$clog2(BANK_DEPTH)-1:0]                            inst_raddr;

    reg  [DATA_WIDTH-1:0]                                    inst_wdata;
    reg  [DATA_WIDTH-1:0]                                    inst_rdata;

    wire [$clog2(BANK_NUM)-1:0]                              bank_wsel;       //sel which bank
    wire [$clog2(BANK_DEPTH)-1:0]                            bank_waddr;      //sel the addr in bank
                                                                              //only write need bank sel

    assign bank_wsel = waddr[(ADDR_WIDTH-1) -: $clog2(BANK_NUM)];
    assign bank_waddr = waddr[0 +: $clog2(BANK_DEPTH)];

    //decoder
    //write
    reg                                                    nxt_inst_wen;
    reg [DATA_WIDTH-1:0]                                   nxt_inst_bwe;
    reg [$clog2(BANK_DEPTH)-1:0]                           nxt_inst_waddr;
    reg [DATA_WIDTH-1:0]                                   nxt_inst_wdata;
    always @(*) begin
        nxt_inst_wen = wen;
        nxt_inst_waddr = bank_waddr;
        nxt_inst_wdata = 'b0;
        nxt_inst_bwe = 'b0;
        for(int i=0; i<BANK_NUM; i++) begin
            if(~wdata_byte_flag)begin
                if(wen)begin
                    nxt_inst_wdata[i*SINGLE_SLICE_DATA_BIT +: SINGLE_SLICE_DATA_BIT] =  wdata[i*SINGLE_SLICE_DATA_BIT +: SINGLE_SLICE_DATA_BIT];
                    nxt_inst_bwe = {DATA_WIDTH{1'b1}};
                end
            end
            else if((i==bank_wsel) && wen) begin
                nxt_inst_bwe[SINGLE_SLICE_DATA_BIT*i+:SINGLE_SLICE_DATA_BIT]= {SINGLE_SLICE_DATA_BIT{1'b1}};
                nxt_inst_wdata[SINGLE_SLICE_DATA_BIT*i+:SINGLE_SLICE_DATA_BIT] = wdata[SINGLE_SLICE_DATA_BIT:0];
            end
        end
    end
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            inst_wen<='0;
            inst_waddr<='b0;
            inst_wdata<='0;
            inst_bwe <='0;
        end
        else if(wen) begin
            inst_wen<=nxt_inst_wen;
            inst_waddr<=nxt_inst_waddr;
            inst_wdata<=nxt_inst_wdata;
            inst_bwe<=nxt_inst_bwe;
        end
        else begin
            inst_wen <= 0;
            inst_bwe <= 0;
        end
    end

    //read
    always @(*) begin
        inst_ren=1'b0;
        inst_raddr='b0;
        if(ren) begin
            inst_ren=1'b1;
            inst_raddr=raddr[0 +: $clog2(BANK_DEPTH)];
        end
    end

    assign rdata = inst_rdata;


    logic [$clog2(BANK_DEPTH) + $clog2(`MAC_MULT_NUM/2)-1:0] interface_addr_delay1;
    logic [$clog2(BANK_DEPTH) + $clog2(`MAC_MULT_NUM/2)-1:0] interface_addr_delay2;
    logic                                                    interface_ren_delay1;

    always_ff @(posedge clk or negedge rstn)begin
        if(~rstn)begin
            interface_ren_delay1 <= 0;
            interface_rvalid <= 0;    
        end
        else begin
            interface_ren_delay1 <= interface_ren;
            interface_rvalid <= interface_ren_delay1;
        end
    end

    assign interface_rdata = inst_rdata[interface_addr_delay2[$clog2(`MAC_MULT_NUM/2)-1:0] * (2*`IDATA_WIDTH) +: 2*`IDATA_WIDTH];


    logic                                                      inst_wen_f;
    logic                                                      inst_ren_f;

    logic  [DATA_WIDTH-1:0]                                    inst_bwe_f;

    logic  [$clog2(BANK_DEPTH)-1:0]                            inst_waddr_f;
    logic  [$clog2(BANK_DEPTH)-1:0]                            inst_raddr_f;

    logic  [DATA_WIDTH-1:0]                                    inst_wdata_f;

///////////////////////////////////////////
//              INTERFACE                //
///////////////////////////////////////////
logic                                                      interface_inst_ren;
logic  [$clog2(BANK_DEPTH)-1:0]                            interface_inst_raddr;

logic                                                      interface_inst_wen;
logic  [$clog2(BANK_DEPTH)-1:0]                            interface_inst_waddr;
logic  [DATA_WIDTH-1:0]                                    interface_inst_wdata;
logic [DATA_WIDTH-1:0]                                     interface_inst_bwe;

//write
always_ff @(posedge clk or negedge rstn)begin
    if(~rstn)begin
        interface_inst_wen <= 0;
        interface_inst_waddr <= 0;
        interface_inst_wdata <= 0;
        interface_inst_bwe <= 0;
    end
    else begin
        if(interface_wen)begin
            interface_inst_wen <= 1;
            interface_inst_waddr <= interface_addr[$clog2(`MAC_MULT_NUM/2) +: $clog2(BANK_DEPTH)];
            interface_inst_wdata[interface_addr[$clog2(`MAC_MULT_NUM/2)-1:0] * (2*`IDATA_WIDTH) +: 2*`IDATA_WIDTH] <= interface_wdata;
            interface_inst_bwe  [interface_addr[$clog2(`MAC_MULT_NUM/2)-1:0] * (2*`IDATA_WIDTH) +: 2*`IDATA_WIDTH] <= 16'hFFFF;
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
        if(interface_ren)begin
            interface_inst_ren <= 1;
            interface_inst_raddr <= interface_addr[$clog2(`MAC_MULT_NUM/2) +: $clog2(BANK_DEPTH)];
        end
        else begin
            interface_inst_ren <= 0;
        end
    end
end


always_comb begin
    if(interface_inst_wen)begin
        inst_wen_f = interface_inst_wen;
        inst_waddr_f = interface_inst_waddr;
        inst_wdata_f = interface_inst_wdata;
        inst_bwe_f = interface_inst_bwe;
    end
    else begin
        inst_wen_f = inst_wen;
        inst_waddr_f = inst_waddr;
        inst_wdata_f = inst_wdata;
        inst_bwe_f = inst_bwe;
    end
end

always_comb begin
    if(interface_inst_ren)begin
        inst_ren_f = interface_inst_ren;
        inst_raddr_f = interface_inst_raddr;
    end
    else begin
        inst_ren_f = inst_ren;
        inst_raddr_f = inst_raddr;
    end
end


always_ff @(posedge clk or negedge rstn)begin
    if(~rstn)begin
        interface_addr_delay1 <= 0;
        interface_addr_delay2 <= 0;
    end
    else begin
        interface_addr_delay1 <= interface_addr;
        interface_addr_delay2 <= interface_addr_delay1;
    end
end

`ifndef TRUE_MEM
    mem_dp  #(.DATA_BIT(SINGLE_SLICE_DATA_BIT*BANK_NUM), .DEPTH(BANK_DEPTH), .BWE(1)) ram_piece(
        .clk                    (clk),
        .waddr                  (inst_waddr_f),
        .wen                    (inst_wen_f),
        .bwe                    (inst_bwe_f),
        .wdata                  (inst_wdata_f),
        .raddr                  (inst_raddr_f),
        .ren                    (inst_ren_f),
        .rdata                  (inst_rdata)
    );
`else
    global_sram ram_piece (
    .ickwp0(clk),
    .iwenp0(inst_wen_f),
    .iawp0(inst_waddr_f),
    .idinp0(inst_wdata_f),
    .ibwep0(inst_bwe_f),
    .ickrp0(clk),
    .irenp0(inst_ren_f),
    .iarp0(inst_raddr_f),
    .iclkbyp(1'b1),
    .imce(1'b0),
    .irmce(2'b0),
    .ifuse(1'b1),
    .iwmce(4'b0),
    .odoutp0(inst_rdata)
    );
`endif
endmodule


module gbus_top #(
    parameter HEAD_CORE_NUM = `HEAD_CORE_NUM,
    parameter GBUS_DATA_WIDTH = `GBUS_DATA_WIDTH
)
(
    input  logic                                                     clk,
    input  logic                                                     rst_n,

    //head core array 会有CORE_NUM个gbus write outport， 用于输出到gbus的arbiter
    input  BUS_ADDR      [HEAD_CORE_NUM-1:0]                         hca_gbus_addr_array,  
    input  logic         [HEAD_CORE_NUM-1:0]                         hca_gbus_wen_array,
    input  logic         [HEAD_CORE_NUM-1:0][GBUS_DATA_WIDTH-1:0]    hca_gbus_wdata_array,

    output BUS_ADDR                                                  gbus_addr,  
    output logic                                                     gbus_wen,
    output logic                            [GBUS_DATA_WIDTH-1:0]    gbus_wdata
);

///////////////////////////
//inst of bus packet fifo//
///////////////////////////

BUS_PACKET   [HEAD_CORE_NUM-1:0]  fifo_in_bus_packet_array;
logic        [HEAD_CORE_NUM-1:0]  wr_en_array;

BUS_PACKET   [HEAD_CORE_NUM-1:0]  fifo_out_bus_packet_array;
logic        [HEAD_CORE_NUM-1:0]  rd_en_array;
logic        [HEAD_CORE_NUM-1:0]  buffer_empty_array;

always_comb begin
    for(int i = 0; i < HEAD_CORE_NUM;i++)begin
        fifo_in_bus_packet_array[i].bus_data =  hca_gbus_wdata_array[i];
        fifo_in_bus_packet_array[i].bus_addr = hca_gbus_addr_array[i];
        wr_en_array[i] = hca_gbus_wen_array[i];
    end
end

genvar i;
generate
    for(i=0; i < HEAD_CORE_NUM;i++)begin : bus_packet_fifo_gen_array
    bus_packet_fifo #(.index(i))inst_bus_packet_fifo(
        .clk(clk),
        .rst_n(rst_n),

        .in_bus_packet(fifo_in_bus_packet_array[i]),
        .wr_en(wr_en_array[i]),
        .buffer_full(), //always be 0(just for format),if 1, wrong!
    
        .out_bus_packet(fifo_out_bus_packet_array[i]),
        .rd_en(rd_en_array[i]),
        .buffer_empty(buffer_empty_array[i]),

        .fifo_full_error() //if fifo is full
    );
    end
endgenerate



//////////////////////////
//inst of bus controller//
//////////////////////////
BUS_PACKET                           bus_packet;
logic                                bus_packet_vld;

assign gbus_addr = bus_packet.bus_addr;
assign gbus_wdata = bus_packet.bus_data;
assign gbus_wen = bus_packet_vld;


BUS_PACKET    [HEAD_CORE_NUM-1:0] ctrl_in_bus_packet_array;
logic         [HEAD_CORE_NUM-1:0] bus_req_array;
logic         [HEAD_CORE_NUM-1:0] bus_grant_array;

assign bus_req_array = ~buffer_empty_array;
assign rd_en_array = bus_grant_array;
assign ctrl_in_bus_packet_array = fifo_out_bus_packet_array;


bus_controller inst_bus_ctrl(
    .clk(clk),
    .rst_n(rst_n),

    //data bus
    .bus_packet(bus_packet),
    .bus_packet_vld(bus_packet_vld),

    //data bus req
    .in_bus_packet_array(ctrl_in_bus_packet_array),
    .bus_req_array(bus_req_array),
    .bus_grant_array(bus_grant_array)
);

endmodule
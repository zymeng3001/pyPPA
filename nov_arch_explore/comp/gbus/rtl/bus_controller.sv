//Notice: Grant is for the next cycle not current cycle
module bus_controller(
    input  logic                               clk,
    input  logic                               rst_n,

    //data bus
    output  BUS_PACKET                         bus_packet,
    output  logic                              bus_packet_vld,

    //data bus req
    input  BUS_PACKET    [`BUS_REQ_MAX_NUM-1:0] in_bus_packet_array,
    input  logic         [`BUS_REQ_MAX_NUM-1:0] bus_req_array,
    output logic         [`BUS_REQ_MAX_NUM-1:0] bus_grant_array
);

BUS_PACKET                                     nxt_bus_packet;
logic                                          nxt_bus_packet_vld;
BUS_PACKET    [`BUS_REQ_MAX_NUM-1:0]            bus_packet_array;
logic         [`BUS_REQ_MAX_NUM-1:0]            bus_grant_array_delay1;

always_ff@(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        bus_grant_array_delay1 <= 0;
    end
    else begin
        bus_grant_array_delay1 <= bus_grant_array;
    end
end

always_ff@(posedge clk or negedge rst_n)begin //pipeline for timing consideration
    if(~rst_n)begin
        bus_packet <= 0;
        bus_packet_vld <= 0;
    end
    else begin
        bus_packet <= nxt_bus_packet;
        bus_packet_vld <= nxt_bus_packet_vld;
    end
end

always_comb begin
    nxt_bus_packet = 0;
    nxt_bus_packet.bus_addr.core_addr = -1;
    for(int i = 0; i < `BUS_REQ_MAX_NUM;i++)begin 
        if(bus_grant_array_delay1[i] == 1) //这样写会有优先级，不过感觉影响不大？如果没有参数化，可以用unique case
            nxt_bus_packet = in_bus_packet_array[i];
    end
    // for(int i = 0; i < `BUS_REQ_MAX_NUM;i++)begin 
    //     if(bus_grant_array[i] == 0)
    //         nxt_bus_packet_array[i] = 0;
    //     else
    //         nxt_bus_packet_array[i] = in_bus_packet_array[i];

    //     nxt_bus_packet = nxt_bus_packet | nxt_bus_packet_array[i];
    // end
end

always_ff@(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        nxt_bus_packet_vld <= 0;
    end
    else if (|bus_req_array)begin
        nxt_bus_packet_vld <= 1;
    end
    else begin
        nxt_bus_packet_vld <= 0;
    end
end


arbiter inst_arbiter
(
.clk                    (clk),
.rst_n                  (rst_n),
.first_priority         (`BUS_REQ_MAX_NUM'b1),//give fixed value
.req                    (bus_req_array),
.grant                  (bus_grant_array)
);

endmodule




module bus_packet_fifo #(
    parameter index = 0
)(
    input   logic           clk,
    input   logic           rst_n,

    input   BUS_PACKET      in_bus_packet,
    input   logic           wr_en,
    output  logic           buffer_full, //always be 0(just for format),if 1, wrong!
    
    output  BUS_PACKET      out_bus_packet,
    input   logic           rd_en,
    output  logic           buffer_empty,

    output  logic           fifo_full_error //if fifo is full
);

BUS_PACKET [`BUS_PACKET_FIFO_DEPTH-1:0]          bus_packet_buffer;                                 
logic      [$clog2(`BUS_PACKET_FIFO_DEPTH)-1:0]  buffer_wr_ptr;
logic      [$clog2(`BUS_PACKET_FIFO_DEPTH)-1:0]  buffer_rd_ptr;
logic      [$clog2(`BUS_PACKET_FIFO_DEPTH):0]    fifo_cnt;

assign fifo_full_error = buffer_full;
assign buffer_full = (fifo_cnt == `BUS_PACKET_FIFO_DEPTH);
assign buffer_empty = (fifo_cnt == 0);


always_ff@(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        fifo_cnt <= 0;
    end
    else begin
        if(wr_en && (~buffer_full) && ~rd_en)begin
            fifo_cnt <= fifo_cnt + 1;
        end
        else if(rd_en && (~buffer_empty) && ~wr_en)begin
            fifo_cnt <= fifo_cnt - 1;
        end
    end
end

//write channel
always_ff @(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        buffer_wr_ptr <= 0;
    end
    else if(wr_en && (~buffer_full))begin
        bus_packet_buffer[buffer_wr_ptr] <= in_bus_packet;
        if(buffer_wr_ptr == `BUS_PACKET_FIFO_DEPTH-1)
            buffer_wr_ptr <= 0;
        else
            buffer_wr_ptr <= buffer_wr_ptr + 1;
    end
end


//read channel
always_ff @(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        out_bus_packet <= '0;
        buffer_rd_ptr  <= '0;
    end
    else if(rd_en && (~buffer_empty))begin
        out_bus_packet <= bus_packet_buffer[buffer_rd_ptr];
        if(buffer_rd_ptr == `BUS_PACKET_FIFO_DEPTH-1)
            buffer_rd_ptr <= 0;
        else
            buffer_rd_ptr <= buffer_rd_ptr + 1;
    end
end

// `ifdef ASSERT_EN 
// always @(negedge clk) begin
//     assert (fifo_full_error != 1 || $time ==0) 
//     else begin
//         $fatal("Error: fifo_full_error is 1 at time %0t", $time);
//     end
// end

// int fifo_cnt_max;
// initial begin
//     fifo_cnt_max = 0;
// end
// always begin
//     @(negedge clk);
//     if(fifo_cnt > fifo_cnt_max)begin
//         fifo_cnt_max = fifo_cnt;
//         $display("Index: %0d, fifo cnt max updates: %0d",index, fifo_cnt_max);
//     end
// end
// `endif 
endmodule


// module bus_packet_fifo (
//     input   logic           clk,
//     input   logic           rst_n,

//     input   BUS_PACKET      in_bus_packet,
//     input   logic           wr_en,
//     output  logic           buffer_full, //always be 0(just for format),if 1, wrong!
    
//     output  BUS_PACKET      out_bus_packet,
//     input   logic           rd_en,
//     output  logic           buffer_empty,

//     output  logic           fifo_full_error //if fifo is full
// );

// BUS_PACKET [`BUS_PACKET_FIFO_DEPTH-1:0]          bus_packet_buffer;
// logic      [$clog2(`BUS_PACKET_FIFO_DEPTH):0]    buffer_wr_ptr; //points to the first empty entry(unless full)
// logic      [$clog2(`BUS_PACKET_FIFO_DEPTH):0]    buffer_rd_ptr; //points to the first valid entry(unless empty)                                         
// logic      [$clog2(`BUS_PACKET_FIFO_DEPTH)-1:0]  buffer_wr_ptr;
// logic      [$clog2(`BUS_PACKET_FIFO_DEPTH)-1:0]  buffer_rd_ptr;
// logic      [$clog2(`BUS_PACKET_FIFO_DEPTH)-1:0]  fifo_cnt;

// assign buffer_wr_ptr = buffer_wr_ptr[$clog2(`BUS_PACKET_FIFO_DEPTH)-1:0];
// assign buffer_rd_ptr = buffer_rd_ptr[$clog2(`BUS_PACKET_FIFO_DEPTH)-1:0];
// assign fifo_full_error = buffer_full;
// assign buffer_full = (buffer_wr_ptr[$clog2(`BUS_PACKET_FIFO_DEPTH)-1:0] == buffer_rd_ptr[$clog2(`BUS_PACKET_FIFO_DEPTH)-1:0]) 
//                     && (buffer_wr_ptr[$clog2(`BUS_PACKET_FIFO_DEPTH)] != buffer_rd_ptr[$clog2(`BUS_PACKET_FIFO_DEPTH)]);
// assign buffer_empty = (buffer_rd_ptr == buffer_wr_ptr);


// always_ff@(posedge clk or negedge rst_n)begin
//     if(~rst_n)begin
//         fifo_cnt <= 0;
//     end
//     else begin
//         if(wr_en && (~buffer_full) && ~rd_en)begin
//             fifo_cnt <= fifo_cnt + 1;
//         end
//         else if(rd_en && (~buffer_empty) && ~wr_en)begin
//             fifo_cnt <= fifo_cnt - 1;
//         end
//     end
// end

// //write channel
// always_ff @(posedge clk or negedge rst_n)begin
//     if(~rst_n)begin
//         buffer_wr_ptr <= 0;
//     end
//     else if(wr_en && (~buffer_full))begin
//         bus_packet_buffer[buffer_wr_ptr] <= in_bus_packet;
//         buffer_wr_ptr <= buffer_wr_ptr + 1;
//     end
// end


// //read channel
// always_ff @(posedge clk or negedge rst_n)begin
//     if(~rst_n)begin
//         out_bus_packet <= '0;
//         buffer_rd_ptr  <= '0;
//     end
//     else if(rd_en && (~buffer_empty))begin
//         out_bus_packet <= bus_packet_buffer[buffer_rd_ptr];
//         buffer_rd_ptr <= buffer_rd_ptr + 1;
//     end
// end

// `ifdef ASSERT_EN 
// always @(posedge clk) begin
//     assert (fifo_full_error != 1) 
//     else begin
//         $fatal("Error: fifo_full_error is 1 at time %0t", $time);
//     end
// end

// int fifo_cnt_max;
// initial begin
//     fifo_cnt_max = 0;
// end
// always begin
//     @(negedge clk);
//     if(fifo_cnt > fifo_cnt_max)begin
//         fifo_cnt_max = fifo_cnt;
//         $display("fifo cnt max updates: %0d",fifo_cnt_max);
//     end
// end
// `endif 
// endmodule

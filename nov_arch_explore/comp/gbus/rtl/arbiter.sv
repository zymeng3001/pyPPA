`ifdef ROUND_ROBIN_ARBITER 
module arbiter
(
input	  logic 	                            clk,
input	  logic 	                            rst_n,
input	  logic [`ARBITER_REQ_WIDTH-1:0]	    first_priority, //give fixed value
input   logic [`ARBITER_REQ_WIDTH-1:0]	    req,
output  logic [`ARBITER_REQ_WIDTH-1:0]	    grant
);
 
logic    [`ARBITER_REQ_WIDTH-1:0]    round_priority;
 
always @(posedge clk or negedge rst_n)
begin
  if(~rst_n)
    round_priority <= first_priority;//round_priority <= (`ARBITER_REQ_WIDTH)'b1;
  else if(|req)
    round_priority <= {grant[`ARBITER_REQ_WIDTH-2:0],grant[`ARBITER_REQ_WIDTH-1]}; // one hot, ring shift left
end
logic	[`ARBITER_REQ_WIDTH*2-1:0]	double_req;
logic	[`ARBITER_REQ_WIDTH*2-1:0]	req_sub_round_priority;
logic	[`ARBITER_REQ_WIDTH*2-1:0]	double_grant;
 
assign double_req = {req,req};
assign req_sub_round_priority = double_req - round_priority;
assign double_grant = double_req & (~req_sub_round_priority);
assign grant = double_grant[`ARBITER_REQ_WIDTH-1:0] | double_grant[`ARBITER_REQ_WIDTH*2-1:`ARBITER_REQ_WIDTH];
 
endmodule
`else
// fix priority arbitor
module arbiter
(
  input		logic 	                            clk,
  input		logic 	                            rst_n,
  input	  logic [`ARBITER_REQ_WIDTH-1:0]	    first_priority, //one hot, who is 1 is the first, who is one the left is the second, so on and cycle back
  input 	logic [`ARBITER_REQ_WIDTH-1:0]	    req,
  output 	logic [`ARBITER_REQ_WIDTH-1:0]	    grant
);
logic	[`ARBITER_REQ_WIDTH*2-1:0]	double_req;
logic	[`ARBITER_REQ_WIDTH*2-1:0]	req_sub_first_priority;
logic	[`ARBITER_REQ_WIDTH*2-1:0]	double_grant;

assign double_req = {req,req};
assign req_sub_first_priority = double_req - first_priority;
assign double_grant = double_req & (~req_sub_first_priority);
assign grant = double_grant[`ARBITER_REQ_WIDTH-1:0] | double_grant[`ARBITER_REQ_WIDTH*2-1:`ARBITER_REQ_WIDTH];
 
endmodule
`endif
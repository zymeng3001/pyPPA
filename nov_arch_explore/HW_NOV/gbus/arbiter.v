module arbiter (
	clk,
	rst_n,
	first_priority,
	req,
	grant
);
	input wire clk;
	input wire rst_n;
	input wire [15:0] first_priority;
	input wire [15:0] req;
	output wire [15:0] grant;
	reg [15:0] round_priority;
	always @(posedge clk or negedge rst_n)
		if (~rst_n)
			round_priority <= first_priority;
		else if (|req)
			round_priority <= {grant[14:0], grant[15]};
	wire [31:0] double_req;
	wire [31:0] req_sub_round_priority;
	wire [31:0] double_grant;
	assign double_req = {req, req};
	assign req_sub_round_priority = double_req - round_priority;
	assign double_grant = double_req & ~req_sub_round_priority;
	assign grant = double_grant[15:0] | double_grant[31:16];
endmodule

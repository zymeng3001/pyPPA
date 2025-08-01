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



//latency = 17

module fp_invsqrt_pipe(
    input  logic                 clk,
    input  logic                 rst_n,
 
    input  logic [15:0]          x, //input bfloat x, sign: 1, exp: 8, frac: 7
    input  logic                 x_vld,

    output logic [15:0]          y,
    output logic                 y_vld
);

//latency = 1
logic [15:0] startpoint; //startpoint for Newton interation
logic [15:0] half_x;
logic        startpoint_half_x_vld;
logic [15:0] threehalfs;

assign threehalfs = 16'b0_01111111_1000000;

always_ff @(posedge clk or negedge rst_n) begin
    if(~rst_n)begin
        startpoint <= 0;
        startpoint_half_x_vld <= 0;
        half_x <= 0;
    end
    else if(x_vld)begin
        startpoint <= 16'hBE6F - x >> 1;
        startpoint_half_x_vld <= x_vld;
        half_x <= x - {1'b1, 7'b0};
    end
    else begin
        startpoint_half_x_vld <= 0;
    end
end

logic [15:0] startpoint_delay_array[10:0];
logic [15:0] half_x_delay_array[10:0];
genvar i;
generate
    for(i = 0;i < 11;i++) begin
        if(i==0) begin
            always_ff@(posedge clk or negedge rst_n)begin
                if(~rst_n)begin
                    startpoint_delay_array[i] <= 0;
                    half_x_delay_array[i] <= 0;
                end
                else begin
                    startpoint_delay_array[i] <= startpoint;
                    half_x_delay_array[i] <= half_x;
                end
            end
        end
        else begin
            always_ff@(posedge clk or negedge rst_n)begin
                if(~rst_n)begin
                    startpoint_delay_array[i] <= 0;
                    half_x_delay_array[i] <= 0;
                end
                else begin
                    startpoint_delay_array[i] <= startpoint_delay_array[i-1];
                    half_x_delay_array[i] <= half_x_delay_array[i-1];
                end
            end
        end
    end
endgenerate

//latency = 4 + 1 + 1= 6
logic [15:0] startpoint_square;
logic [15:0] nxt_startpoint_square;
logic        startpoint_square_vld;
logic        nxt_startpoint_square_vld;



always_ff@(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        startpoint_square <= 0;
        startpoint_square_vld <= 0;
    end
    else begin
        startpoint_square <= nxt_startpoint_square;
        startpoint_square_vld <= nxt_startpoint_square_vld;
    end
end


fp_mult_pipe#(  //latency = 4
    .sig_width(7),
    .exp_width(8),
    .ieee_compliance(0)
)startpoint_square_mult_inst
( 
    .clk(clk),
    .rst_n(rst_n),
    .a(startpoint),
    .b(startpoint),
    .z(nxt_startpoint_square),
    .ab_valid(startpoint_half_x_vld),
    .z_valid(nxt_startpoint_square_vld)
);

//latency = 1 + 4 + 1 + 4 + 1 = 11
logic [15:0] half_x_mult_startpoint_square;
logic [15:0] nxt_half_x_mult_startpoint_square;
logic        half_x_mult_startpoint_square_vld;
logic        nxt_half_x_mult_startpoint_square_vld;



always_ff@(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        half_x_mult_startpoint_square <= 0;
        half_x_mult_startpoint_square_vld <= 0;
    end
    else begin
        half_x_mult_startpoint_square <= nxt_half_x_mult_startpoint_square;
        half_x_mult_startpoint_square_vld <= nxt_half_x_mult_startpoint_square_vld;
    end
end


fp_mult_pipe#(
    .sig_width(7),//latency = 4
    .exp_width(8),
    .ieee_compliance(0)
)half_x_mult_inst
( 
    .clk(clk),
    .rst_n(rst_n),
    .a(startpoint_square),
    .b(half_x_delay_array[4]),
    .z(nxt_half_x_mult_startpoint_square),
    .ab_valid(startpoint_square_vld),
    .z_valid(nxt_half_x_mult_startpoint_square_vld)
);

//latency = 1 + 4 + 1 + 4 + 1 + 1= 12
logic [15:0] threehalfs_sub;
logic [15:0] nxt_threehlafs_sub;
logic        threehalfs_sub_vld;
logic        nxt_threehalfs_sub_vld;

assign nxt_threehalfs_sub_vld = half_x_mult_startpoint_square_vld;
always_ff@(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        threehalfs_sub <= 0;
        threehalfs_sub_vld <= 0;
    end
    else begin
        threehalfs_sub <= nxt_threehlafs_sub;
        threehalfs_sub_vld <= nxt_threehalfs_sub_vld;
    end
end

DW_fp_sub #(.sig_width(7), .exp_width(8), .ieee_compliance(0))
threehalfs_sub_inst 
( 
    .a(threehalfs), 
    .b(half_x_mult_startpoint_square), 
    .rnd(3'b000), 
    .z(nxt_threehlafs_sub), 
    .status() 
);

//latency = 1 + 4 + 1 + 4 + 1 + 1 + 4 + 1= 17 for y
logic [15:0] nxt_y;
logic        nxt_y_vld;

always_ff@(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        y_vld <=0;
        y<=0;
    end
    else begin
        y <= nxt_y;
        y_vld <= nxt_y_vld;
    end
end

fp_mult_pipe#(
    .sig_width(7),
    .exp_width(8),
    .ieee_compliance(0)
)y_mult_inst
( 
    .clk(clk),
    .rst_n(rst_n),
    .a(threehalfs_sub),
    .b(startpoint_delay_array[10]),
    .z(nxt_y),
    .ab_valid(threehalfs_sub_vld),
    .z_valid(nxt_y_vld)
);
endmodule
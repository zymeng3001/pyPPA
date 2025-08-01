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

module fp_invsqrt_pipe_tb ();
function  shortreal get_rand_shortreal();
    shortreal min = 1;
    shortreal max = 400;
    get_rand_shortreal = min + (max-min)*(($urandom)*1.0/32'hffffffff);
endfunction

function shortreal bitsbfloat16_to_shortreal;
    input [15:0] x;
    begin
        logic [31:0] x_float;
        x_float = {x,16'b0};
        bitsbfloat16_to_shortreal = $bitstoshortreal(x_float);
    end
endfunction

function [15:0] shortreal_to_bitsbfloat16;
    input real x;
    begin
        logic [31:0] x_float_bits;
        x_float_bits = $shortrealtobits(x);
        shortreal_to_bitsbfloat16 = x_float_bits[31:16] + x_float_bits[15];
    end
endfunction

function [15:0] bfloat16_invsqrt_soft;
    input [15:0] x;
    shortreal y;
    shortreal x2;
    begin
        logic [15:0] temp;
        temp = 16'hBE6F - x >> 1;
        y = bitsbfloat16_to_shortreal(temp);
        x2 = bitsbfloat16_to_shortreal(x) * 0.5;
        y = y*(1.5-(x2*y*y));
        bfloat16_invsqrt_soft = shortreal_to_bitsbfloat16(y);
    end
endfunction

logic                 clk;
logic                 rst_n;

logic [15:0]          x; //i
logic                 x_vld;
logic [15:0]          y;
logic                 y_vld;

real ratio_error = 1e-2;
fp_invsqrt_pipe inst(
    .clk(clk),
    .rst_n(rst_n),

    .x(x), //input bfloat x, sign: 1, exp: 8, frac: 7
    .x_vld(x_vld),

    .y(y),
    .y_vld(y_vld)
);

shortreal x_float;
shortreal y_float;
shortreal real_y_float;
int ERROR_FLAG;

initial
    clk = 0;
always #50 clk = ~clk;

int cnt;
initial begin
    cnt = 0;
    ERROR_FLAG = 0;
    $display("1.5 bfloat in bits: %16b", shortreal_to_bitsbfloat16(1.5));
    for(int i = 0;i < 100;i++)begin
        x_float = get_rand_shortreal();
        real_y_float = 1/$sqrt(x_float);
        y_float = bitsbfloat16_to_shortreal(bfloat16_invsqrt_soft(shortreal_to_bitsbfloat16(x_float)));
        if($abs(y_float-real_y_float)/$abs(real_y_float) >= ratio_error)begin
            $display("Ratio error too large");
            ERROR_FLAG = 1;
        end
    end

    rst_n = 0;
    x = 0;
    x_vld = 0;
    repeat(10) @(negedge clk);
    rst_n = 1;
    @(negedge clk);
    
    for(int i =0;i < 100;i++)begin
        @(negedge clk);
        x_vld = 1;
        x_float = get_rand_shortreal();
        // x_float = 1.0+i;
        // $display("x_float: %0f",x_float);
        x = shortreal_to_bitsbfloat16(x_float);
        // real_y_float = bitsbfloat16_to_shortreal(bfloat16_invsqrt_soft(x));
        real_y_float = 1/$sqrt(x_float);
    end
    @(negedge clk);
    x_vld = 0;
    repeat(100) @(negedge clk);

    if(ERROR_FLAG == 0)
        $display("PASS");
    else
        $display("Wrong");
    $stop();
    // $display("%f, %f",real_y,y);
end
shortreal real_y_float_delay_array [16:0];
shortreal real_y_float_delay16;
assign real_y_float_delay16 = real_y_float_delay_array[15];
always_ff@(posedge clk)begin
    for(int i = 0;i < 17;i++)begin
        if(i == 0)begin
            real_y_float_delay_array[i] <= real_y_float;
        end
        else begin
            real_y_float_delay_array[i] <= real_y_float_delay_array[i-1];
        end
    end
end

always begin
    @(negedge clk);
    if(y_vld)begin
        // $display("True: %0f, Get: %0f",real_y_float_delay_array[16],bitsbfloat16_to_shortreal(y));
        if($abs(real_y_float_delay_array[16]-bitsbfloat16_to_shortreal(y))/$abs(real_y_float_delay_array[16])>=ratio_error)begin
            $display("Wrong");
            // $finish();
            cnt = cnt  + 1;
            $display("cnt = %0d",cnt);
            ERROR_FLAG = 1;
        end
    end
end
endmodule
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




/**
latency = in_reg + out_reg + stages - 1
**/
module div_pipe_tb ();

parameter sig_width = 23;
parameter exp_width = 8;
parameter ieee_compliance = 0;//NaN and denormals
parameter in_reg = 0;
parameter stages = 6;
parameter out_reg = 1;

logic                                        clk;
logic                                        rst_n;
logic [sig_width+exp_width : 0]              a;
logic [sig_width+exp_width : 0]              b;
logic [sig_width+exp_width : 0]              z;
logic                                        launch;
logic                                        ab_valid;
logic                                        arrive;
logic                                        z_valid;


function  shortreal get_rand_shortreal();
    shortreal min = -10000000;
    shortreal max = 10000000;
    get_rand_shortreal = min + (max-min)*(($urandom)*1.0/32'hffffffff);
endfunction


always
    #50 clk = ~clk;

shortreal real_q = 0;
shortreal real_q_delay1;
shortreal real_q_delay2;
shortreal real_q_delay3;
shortreal real_q_delay4;
shortreal real_q_delay5;
shortreal real_q_delay6;
always @(*)
    real_q = $bitstoshortreal(a)/$bitstoshortreal(b);

always @(posedge clk)begin
    real_q_delay1 <= real_q;
    real_q_delay2 <= real_q_delay1;
    real_q_delay3 <= real_q_delay2;
    real_q_delay4 <= real_q_delay3;
    real_q_delay5 <= real_q_delay4;
    real_q_delay6 <= real_q_delay5;    
end

always begin
    @(negedge clk) 
    if (real_q_delay6 == real_q_delay6) begin //Check NaN
            assert  ($abs(real_q_delay6 - $bitstoshortreal(z)) / $abs(real_q_delay6+1e-10) <= 1e-5)
                else $error("Assertion failed: ratio differ by more than 1e-5");
        end
end

initial begin
    $display("\n\n\n\n\n\div_pipe simulation start.");
    clk = 0;
    rst_n = 0;
    a = 0;
    b = 0;
    launch = 1;
    ab_valid = 0;
    repeat(10) @(negedge clk);
    rst_n = 1;
    repeat(5) @(negedge clk);

    @(negedge clk);
    ab_valid = 1;
    a = $shortrealtobits(get_rand_shortreal());
    b = $shortrealtobits(get_rand_shortreal());
    @(negedge clk);
    ab_valid = 1;
    a = $shortrealtobits(get_rand_shortreal());
    b = $shortrealtobits(get_rand_shortreal());
    @(negedge clk);
    ab_valid = 1;
    a = $shortrealtobits(get_rand_shortreal());
    b = $shortrealtobits(get_rand_shortreal());
    @(negedge clk);
    ab_valid = 1;
    a = $shortrealtobits(get_rand_shortreal());
    b = $shortrealtobits(get_rand_shortreal());
    @(negedge clk);
    ab_valid = 1;
    a = $shortrealtobits(get_rand_shortreal());
    b = $shortrealtobits(get_rand_shortreal());
    @(negedge clk);
    ab_valid = 1;
    a = $shortrealtobits(get_rand_shortreal());
    b = $shortrealtobits(get_rand_shortreal());
    @(negedge clk);
    ab_valid = 1;
    a = $shortrealtobits(get_rand_shortreal());
    b = $shortrealtobits(get_rand_shortreal());
    @(negedge clk);
    ab_valid = 0;
    a = $shortrealtobits(get_rand_shortreal());
    b = $shortrealtobits(get_rand_shortreal());
    @(negedge clk);
    ab_valid = 1;
    a = $shortrealtobits(get_rand_shortreal());
    b = $shortrealtobits(get_rand_shortreal());
    @(negedge clk);
    ab_valid = 1;
    a = $shortrealtobits(get_rand_shortreal());
    b = $shortrealtobits(get_rand_shortreal());
    @(negedge clk);
    ab_valid = 0;
    a = $shortrealtobits(get_rand_shortreal());
    b = $shortrealtobits(get_rand_shortreal());
    repeat(20) @(negedge clk);
    $display("\n\n\n\n\n\div_pipe simulation ends.");
    $stop();
end

div_pipe#(
sig_width,
exp_width,
ieee_compliance,//NaN and denormals
in_reg,
stages,
out_reg
)inst
( 
clk,
rst_n,
a,
b,
z,
ab_valid,
z_valid
);

endmodule
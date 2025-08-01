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

module fp_mult_pipe#(
    parameter sig_width = 23,
    parameter exp_width = 8,
    parameter ieee_compliance = 0,//NaN and denormals
    parameter stages = 5
)( 
input  logic                                        clk,
input  logic                                        rst_n,
input  logic [sig_width+exp_width : 0]              a,
input  logic [sig_width+exp_width : 0]              b,
output logic [sig_width+exp_width : 0]              z,
input  logic                                        ab_valid,
output logic                                        z_valid
);

parameter id_width = 1;
parameter op_iso_mode = 0;
parameter no_pm = 1;
parameter rst_mode = 0;//Asynchronous reset on rst_n
parameter en_ubr_flag = 0;

// DW_lp_piped_fp_mult #(
//     sig_width,
//     exp_width,
//     ieee_compliance,
//     op_iso_mode,
//     id_width,
//     in_reg,
//     stages,
//     out_reg,
//     no_pm,
//     rst_mode)
// U1 (     
//     .clk(clk),
//     .rst_n(rst_n),
//     .a(a),
//     .b(b),
//     .rnd(3'b00),
//     .z(z),
//     .status(),
//     .launch(1'b1),
//     .launch_id(ab_valid),
//     .pipe_full(),
//     .pipe_ovf(),
//     .accept_n(1'b0),
//     .arrive(),
//     .arrive_id(z_valid),
//     .push_out_n(),
//     .pipe_census() 
// );

// Instance of DW_fp_mult
logic [sig_width+exp_width : 0] z_w;

DW_fp_mult 
#(  sig_width, 
    exp_width, 
    ieee_compliance, 
    en_ubr_flag)
U1 ( 
    .a(a),
    .b(b),
    .rnd(3'b00),
    .z(z_w),
    .status() );

genvar i;
generate
    for(i = 0;i < stages;i++)begin : fp_mult_retiming_stage
        logic [sig_width+exp_width : 0] timing_register_float;
        logic                           timing_register_vld;
        if(i == 0)begin
            always_ff@(posedge clk or negedge rst_n)begin
                if(~rst_n)begin
                    timing_register_float <= 0;
                    timing_register_vld <= 0;
                end
                else begin
                    timing_register_float <= z_w;
                    timing_register_vld <= ab_valid;
                end
            end
        end
        else begin
            always_ff@(posedge clk or negedge rst_n)begin
                if(~rst_n) begin
                    timing_register_float <= 0;
                    timing_register_vld <= 0;  
                end
                else begin
                    timing_register_float <= fp_mult_retiming_stage[i-1].timing_register_float;
                    timing_register_vld <= fp_mult_retiming_stage[i-1].timing_register_vld;
                end
            end
        end
    end
endgenerate

always_comb begin
    z = fp_mult_retiming_stage[stages-1].timing_register_float;
    z_valid = fp_mult_retiming_stage[stages-1].timing_register_vld;
end

endmodule
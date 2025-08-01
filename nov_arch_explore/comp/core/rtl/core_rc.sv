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

//core recompute
module core_rc #(
    parameter IN_DATA_WIDTH = 24,
    parameter OUT_DATA_WIDTH = 24,
    parameter RECOMPUTE_FIFO_DEPTH = 4,
    parameter RETIMING_REG_NUM = 4
)(
    input  logic                                     clk,
    input  logic                                     rst_n,

    input  logic                                     recompute_needed,
 
    input  logic [`RECOMPUTE_SCALE_WIDTH - 1:0]      rc_scale,
    input  logic                                     rc_scale_vld,
    input  logic                                     rc_scale_clear,

    input  logic [`RECOMPUTE_SHIFT_WIDTH - 1:0]      rms_rc_shift,
 
    input  logic [IN_DATA_WIDTH - 1:0]               in_data,
    input  logic                                     in_data_vld,

    output logic [OUT_DATA_WIDTH - 1:0]              out_data,
    output logic                                     out_data_vld,

    output logic                                     error
);

logic [`RECOMPUTE_SCALE_WIDTH - 1:0]      rc_scale_reg;
logic                                     rc_scale_reg_available;

always_ff@(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        rc_scale_reg <= 0;
        rc_scale_reg_available <= 0;
    end
    else if(rc_scale_vld)begin
        rc_scale_reg <= rc_scale;
        rc_scale_reg_available <= 1;
    end
    else if(rc_scale_clear)begin
        rc_scale_reg_available <= 0;
    end
end

logic [IN_DATA_WIDTH - 1:0]               fifo_data_in;
logic                                     fifo_wr_en;
logic signed [IN_DATA_WIDTH - 1:0]        fifo_data_out;
logic                                     fifo_rd_en;
logic                                     fifo_empty;
logic                                     fifo_rvld;
//fifo logic
always_comb begin
    fifo_data_in = in_data;
    fifo_wr_en = in_data_vld;
    fifo_rd_en = 1;
    if(recompute_needed)begin
        fifo_rd_en = rc_scale_reg_available;
    end
end
always_ff@(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        fifo_rvld <= 0;
    end
    else begin
        fifo_rvld <= (~fifo_empty) & fifo_rd_en;
    end
end

//mult with rc_scale
//retiming registers
logic signed [`RECOMPUTE_SCALE_WIDTH + IN_DATA_WIDTH - 1:0] rc_data_rt_array [RETIMING_REG_NUM-1:0];//recompute data retiming array
logic  rc_data_vld_rt_array [RETIMING_REG_NUM-1:0];
genvar i;
generate
    for(i = 0; i < RETIMING_REG_NUM; i++)begin
        if(i == 0)begin
            always_ff@(posedge clk or negedge rst_n)begin
                if(~rst_n)begin
                    rc_data_rt_array[i] <= 0;
                    rc_data_vld_rt_array[i] <= 0;
                end
                else begin
                    if(recompute_needed)begin
                        rc_data_rt_array[i] <= $signed(rc_scale_reg) * $signed(fifo_data_out);//只要有一个数是无符号，整个乘法都是无符号乘法
                    end
                    else begin
                        rc_data_rt_array[i] <= $signed(fifo_data_out);
                    end

                    rc_data_vld_rt_array[i] <= fifo_rvld;
                end
            end
        end
        else begin
            always_ff@(posedge clk or negedge rst_n)begin
                if(~rst_n)begin
                    rc_data_rt_array[i] <= 0;
                    rc_data_vld_rt_array[i] <= 0;
                end
                else begin
                    rc_data_rt_array[i] <= rc_data_rt_array[i-1];
                    rc_data_vld_rt_array[i] <= rc_data_vld_rt_array[i-1];
                end
            end
        end
    end
endgenerate

logic signed [`RECOMPUTE_SCALE_WIDTH + IN_DATA_WIDTH - 1:0] rc_data_shift;
logic                                                       rc_data_shift_vld;
logic                                                       round;

//shift
always_ff @(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        rc_data_shift <= 0;
        rc_data_shift_vld <= 0;
        round <= 0;
    end
    else if(rc_data_vld_rt_array[RETIMING_REG_NUM-1]) begin
        if(recompute_needed)begin
            rc_data_shift <= $signed(rc_data_rt_array[RETIMING_REG_NUM-1]) >>> rms_rc_shift;
            round <= rc_data_rt_array[RETIMING_REG_NUM-1][rms_rc_shift-1];
        end
        else begin
            rc_data_shift <= rc_data_rt_array[RETIMING_REG_NUM-1];
        end
        rc_data_shift_vld <= 1;
    end
    else begin
        rc_data_shift_vld <= 0;
    end
end

//round
logic signed [`RECOMPUTE_SCALE_WIDTH + IN_DATA_WIDTH - 1:0]    rc_data_rnd;
logic signed [`RECOMPUTE_SCALE_WIDTH + IN_DATA_WIDTH - 1:0]    nxt_rc_data_rnd;
logic                                                          rc_data_rnd_vld;

always_comb begin
    nxt_rc_data_rnd = 0;
    if(recompute_needed)begin
        nxt_rc_data_rnd = rc_data_shift + round;
        if(nxt_rc_data_rnd > $signed({1'b0, {(OUT_DATA_WIDTH-1){1'b1}}}))begin
            nxt_rc_data_rnd = {1'b0, {(OUT_DATA_WIDTH-1){1'b1}}};
        end
        else if(nxt_rc_data_rnd < $signed({1'b1, {(OUT_DATA_WIDTH-1){1'b0}}}))begin
            nxt_rc_data_rnd = {1'b1, {(OUT_DATA_WIDTH-1){1'b0}}};
        end
    end
    else begin
        nxt_rc_data_rnd = rc_data_shift;
    end
end

always_ff@(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        rc_data_rnd_vld <= 0;
        rc_data_rnd <= 0;
    end
    else begin
        rc_data_rnd_vld <= rc_data_shift_vld;
        rc_data_rnd <= nxt_rc_data_rnd;
    end
end

//output
always_ff @(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        out_data_vld <= 0;
        out_data <= 0;
    end
    else if(recompute_needed)begin
        out_data_vld <= rc_data_rnd_vld;
        out_data <= rc_data_rnd[OUT_DATA_WIDTH-1:0];
    end
    else begin
        out_data_vld <= in_data_vld;
        out_data <= in_data;
    end
end


sync_data_fifo
#(
	.DATA_WIDTH(IN_DATA_WIDTH)  ,								//FIFO width
    .DATA_DEPTH(RECOMPUTE_FIFO_DEPTH)  					        //FIFO depth
)inst_sync_data_fifo
(
	.clk(clk),		
	.rst_n(rst_n),    

	.data_in(fifo_data_in),       
	.wr_en(fifo_wr_en),      

    .rd_en(fifo_rd_en),       
	.data_out(fifo_data_out),	    

	.empty(fifo_empty),	    
	.full(error)	                //when fifo is full, it may generate wrong result
);  

`ifdef  ASSERT_EN
always @(negedge clk) begin
    assert (error != 1  || $time ==0) 
    else begin
        $fatal("Error: FIFO FULL in RMS RC 1 at time %0t", $time);
    end
end
`endif
endmodule

module	sync_data_fifo
#(
	parameter   DATA_WIDTH = 24  ,								//FIFO width
    parameter   DATA_DEPTH = 4  								//FIFO depth
)
(
	input	logic 									clk		,		
	input	logic 									rst_n	,    

	input	logic [DATA_WIDTH-1:0]					data_in	,       
    input	logic 									wr_en	,    
    
    input	logic 									rd_en	,         					                                        
	output	logic [DATA_WIDTH-1:0]				    data_out,	    

	output	logic								    empty	,	    
	output	logic								    full	   //when fifo is full, it may generate wrong result
);                                                              
 
//logic define
logic [DATA_WIDTH - 1 : 0]			fifo_buffer [DATA_DEPTH - 1 : 0];	
logic [$clog2(DATA_DEPTH) : 0]		wr_ptr;							
logic [$clog2(DATA_DEPTH) : 0]		rd_ptr;							
 
//logic define
logic [$clog2(DATA_DEPTH) - 1 : 0]	real_wr_ptr;				
logic [$clog2(DATA_DEPTH) - 1 : 0]	real_rd_ptr;				
logic								wr_ptr_msb;					
logic								rd_ptr_msb;					
 
assign {wr_ptr_msb,real_wr_ptr} = wr_ptr;						
assign {rd_ptr_msb,real_rd_ptr} = rd_ptr;						
 
always_ff @ (posedge clk or negedge rst_n) begin
	if (~rst_n)
		rd_ptr <= 'd0;
	else if (rd_en && !empty)begin								
		data_out <= fifo_buffer[real_rd_ptr];
		rd_ptr <= rd_ptr + 1'd1;
	end
end

always_ff @ (posedge clk or negedge rst_n) begin
	if (~rst_n)
		wr_ptr <= 0;
	else if (!full && wr_en)begin								
		wr_ptr <= wr_ptr + 1'd1;
		fifo_buffer[real_wr_ptr] <= data_in;
	end	
end
 

assign	empty = ( wr_ptr == rd_ptr ) ? 1'b1 : 1'b0;

assign	full  = ( (wr_ptr_msb != rd_ptr_msb ) && ( real_wr_ptr == real_rd_ptr ) )? 1'b1 : 1'b0;
 
endmodule
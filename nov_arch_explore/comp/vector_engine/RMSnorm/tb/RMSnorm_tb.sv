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

module RMSnorm_tb();

parameter integer DATA_NUM = 384;
parameter integer BUS_NUM = 8;
parameter integer DATA_NUM_WIDTH =  10;
parameter integer SCALA_POS_WIDTH = 5;
parameter integer FIXED_SQUARE_SUM_WIDTH = 24;
parameter integer IN_FLOAT_DATA_ARRAY_DEPTH = DATA_NUM / BUS_NUM + 1;
parameter integer K = 123;

parameter integer IN_SCALE_POS = 3;
parameter integer OUT_SCALE_POS = 4;


localparam        sig_width = 7;//bfloat16
localparam        exp_width = 8;
real ratio_error = 1e-2;

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


task RMSnorm_soft(  input  signed [7:0] i_data_array[DATA_NUM-1:0],
                    input  shortreal g_array[DATA_NUM-1:0],
                    input  int in_scale_pos, 
                    input  int out_scale_pos,
                    output signed [7:0] o_data_array[DATA_NUM-1:0],
                    output shortreal print_real_data_array [DATA_NUM-1:0]);
    shortreal real_data_array [DATA_NUM-1:0];
    
    int  int_data_array  [DATA_NUM-1:0];
    shortreal square_sum;
    shortreal rms;
    square_sum = 0;
    rms = 0;

    for(int i =0;i < DATA_NUM; i++)begin
        real_data_array[i] = $itor($signed(i_data_array[i])) * $pow(2, -in_scale_pos);
    end

    for(int i =0;i < K; i++)begin //first K kRMSnorm
        square_sum = square_sum + real_data_array[i] * real_data_array[i];
    end
    rms = $sqrt(square_sum/K);

    for(int i =0;i < DATA_NUM; i++)begin
        real_data_array[i] = real_data_array[i] / rms * g_array[i];
        // $display(real_data_array[i]);
        print_real_data_array[i] = real_data_array[i];
        
        /*
        Is this conversion right? No scale? No bias?
        
        "out_scale_pos" represents the position of the decimal point, which also denotes precision. 
        For example, if it is 1, it indicates that the precision of the output fixed-point number is 0.5. 
        If it is -1, it indicates that the precision of the output fixed-point number is 0.5. 
        Therefore, the floating-point number is first multiplied by 2^out_scale_pos and then converted 
        to an integer to obtain the desired output fixed-point number.

        And this will be eactly same in hardware. (Since DW provides float to integer and integer to float ip)
        (Be careful integer is not fixed-point number).
        */
        real_data_array[i] = real_data_array[i] * $pow(2, out_scale_pos); 
        int_data_array[i] = int'(real_data_array[i]); 
        if(int_data_array[i] > 127)
            o_data_array[i] = 127;
        else if(int_data_array[i]<-128)
            o_data_array[i] = -128;
        else
            o_data_array[i] = int_data_array[i];
    end 
    // $write("float output:");
    // for(int i =0;i <DATA_NUM;i++)
    //     $write("%f ",print_real_data_array[i]);
    // $display();
endtask

function  shortreal get_rand_shortreal();
    shortreal min = -0.2;
    shortreal max = 1;
    get_rand_shortreal = min + (max-min)*(($urandom)*1.0/32'hffffffff);
endfunction

logic signed [7:0] I_DATA_ARRAY                 [DATA_NUM-1:0];
logic signed [7:0] tb_O_DATA_ARRAY              [DATA_NUM-1:0];
shortreal          tb_O_DATA_ARRAY_float        [DATA_NUM-1:0];

logic signed [7:0] RMSnorm_O_DATA_ARRAY         [DATA_NUM-1:0];
shortreal          RMSnorm_O_DATA_ARRAY_float   [DATA_NUM-1:0];

shortreal          G_ARRAY                      [DATA_NUM-1:0];


logic                                                       clk;
logic                                                       rst_n;

logic        [DATA_NUM_WIDTH-1:0]                           in_data_num;
logic                                                       in_data_num_vld;

logic        [BUS_NUM-1:0][sig_width+exp_width : 0]         in_gamma;
logic        [BUS_NUM-1:0]                                  in_gamma_vld;

logic signed [BUS_NUM-1:0][7:0]                             in_fixed_data;
logic        [BUS_NUM-1:0]                                  in_fixed_data_vld;

logic signed              [SCALA_POS_WIDTH-1:0]             in_scale_pos;
logic                                                       in_scale_pos_vld;

logic signed              [SCALA_POS_WIDTH-1:0]             out_scale_pos; 
logic                                                       out_scale_pos_vld;

logic                     [DATA_NUM_WIDTH-1:0]              in_K;
logic                                                       in_K_vld;

logic signed [BUS_NUM-1:0][7:0]                             out_fixed_data;
logic        [BUS_NUM-1:0]                                  out_fixed_data_vld;
logic                                                       out_fixed_data_last; //This signal is to help to design more efficiently

int temp1;
int temp2;

rms_norm
#(
    .BUS_NUM(BUS_NUM),
    .DATA_NUM_WIDTH(DATA_NUM_WIDTH),
    .SCALA_POS_WIDTH(SCALA_POS_WIDTH),
    .FIXED_SQUARE_SUM_WIDTH(FIXED_SQUARE_SUM_WIDTH),
    .IN_FLOAT_DATA_ARRAY_DEPTH(IN_FLOAT_DATA_ARRAY_DEPTH),
    .sig_width(sig_width),
    .exp_width(exp_width)
)inst_RMSnorm
(
    .clk(clk),
    .rst_n(rst_n),

    .in_data_num(in_data_num),
    .in_data_num_vld(in_data_num_vld),

    .in_gamma(in_gamma),
    .in_gamma_vld(in_gamma_vld),
 
    .in_fixed_data(in_fixed_data),
    .in_fixed_data_vld(in_fixed_data_vld),
 
    .in_scale_pos(in_scale_pos),  
    .in_scale_pos_vld(in_scale_pos_vld), 
 
    .out_scale_pos(out_scale_pos),  
    .out_scale_pos_vld(out_scale_pos_vld),

    .in_K(in_K),
    .in_K_vld(in_K_vld),

    .out_fixed_data(out_fixed_data),
    .out_fixed_data_vld(out_fixed_data_vld),
    .out_fixed_data_last(out_fixed_data_last) //This signal is to help to design more efficiently
);


shortreal tb_float_square_sum = 0; 
shortreal tb_one_over_rms = 0;
shortreal temp_var;
int ERROR_FLAG;
int float_square_sum_vld_flag = 0;
int one_over_rms_vld_flag = 0;

shortreal real_gain_vector_array [DATA_NUM-1:0];
shortreal real_input_vector_array [DATA_NUM-1:0];

logic signed [7:0] real_input_vector_array_fixed [DATA_NUM-1:0];


always
    #(`CLOCK_PERIOD/2.0) clk = ~clk;

int file;
shortreal float_value;
int index = 0;

int diff_cnt = 0;


//read file
initial begin
    file = $fopen("../comp/vector_engine/RMSnorm/tb/RMS_gain.txt", "r");
    if (file == 0) begin
      $display("Failed to open file!");
      $finish;
    end
    while (!$feof(file)) begin
      if ($fscanf(file, "%f\n", float_value) == 1) begin
        real_gain_vector_array[index] = float_value;
        index = index + 1;
      end
    end
    $fclose(file);

    index = 0;

    file = $fopen("../comp/vector_engine/RMSnorm/tb/RMS_in.txt", "r");
    if (file == 0) begin
      $display("Failed to open file!");
      $finish;
    end
    while (!$feof(file)) begin
      if ($fscanf(file, "%f\n", float_value) == 1) begin
        real_input_vector_array[index] = float_value;
        index = index + 1;
      end
    end
    $fclose(file);

    index = 0;
    for(int i = 0; i<DATA_NUM;i++)begin
        float_value = real_input_vector_array[i] * $pow(2,IN_SCALE_POS);
        index  = int'(float_value);
        if(index > 127)
            index = 127;
        if(index < -128)
            index = -128;
        real_input_vector_array_fixed[i] = index;
        // $display("%0d",index);
    end



    // $display("Array of float values:");
    // for (int i = 0; i < index; i++) begin
    //   $display("real_input_vector_array[%0d] = %f", i, real_input_vector_array[i]);
    // end
end

initial begin
    //dump waveform
    $fsdbDumpfile("rms_norm.fsdb");
    $fsdbDumpvars(0,"+all",RMSnorm_tb); //"+all" enables all  signal dumping including Mem, packed array, etc.

    `ifdef SYNTH
        $sdf_annotate("../intel_flow/outputs_fc/rms_norm.tttt_0.800v_25c.tttt.sdf", inst_RMSnorm);
    `endif

    $display("\n\n\n\n");
    $display("Start Simulation.");
    $display("\n\n\n\n");
    clk = 0;
    rst_n = 0;
    //INITIAL
    in_data_num = 0;
    in_data_num_vld = 0;
    in_gamma = 0;
    in_gamma_vld = 0;
    in_scale_pos = 0;
    in_scale_pos_vld = 0;
    out_scale_pos = 0;
    out_scale_pos_vld = 0;
    in_fixed_data = 0;
    in_fixed_data_vld = 0;
    in_K = 0;
    in_K_vld = 0;

    repeat(10) @(negedge clk);
    rst_n = 1;

    repeat(10) @(negedge clk);
    in_data_num = DATA_NUM;
    in_data_num_vld = 1;
    in_scale_pos = IN_SCALE_POS;
    in_scale_pos_vld = 1;
    out_scale_pos = OUT_SCALE_POS;
    out_scale_pos_vld = 1;
    @(negedge clk);
    in_data_num_vld = 0;
    in_scale_pos_vld = 0;
    out_scale_pos_vld = 0;  

    #50;
    for(int i =0;i<DATA_NUM;i++)begin
        // I_DATA_ARRAY[i] = $random;
        // I_DATA_ARRAY[i] = 1;
        I_DATA_ARRAY[i] = real_input_vector_array_fixed[i];
        // $display("i=%d,input = %d",i,I_DATA_ARRAY[i]);
        G_ARRAY[i]=real_gain_vector_array[i];
        // G_ARRAY[i] = 1;
        // G_ARRAY[i] = get_rand_shortreal();
        // G_ARRAY[i] = get_rand_shortreal();
    end

    for(int i = 0; i< (DATA_NUM+BUS_NUM-1)/BUS_NUM;i++)begin
        in_gamma = 0;
        in_gamma_vld = 0;
        for(int j =0; j < BUS_NUM; j++)begin
            if(i * BUS_NUM + j < DATA_NUM)begin
                in_gamma_vld[j] = 1;
                in_gamma[j] = shortreal_to_bitsbfloat16(G_ARRAY[i*BUS_NUM+j]);
                // in_gamma[j] = 16'b0_01111111_0000000;
            end
        end
        @(negedge clk);
    end
    in_gamma_vld = 0;

    repeat(10)begin
        diff_cnt = 0;
        ERROR_FLAG = 0;
        for(int i = 0; i <K;i++)begin
            tb_float_square_sum = tb_float_square_sum +  $itor($signed(I_DATA_ARRAY[i])) / $pow(2,IN_SCALE_POS) * $itor($signed(I_DATA_ARRAY[i])) / $pow(2,IN_SCALE_POS);
        end

        RMSnorm_soft(I_DATA_ARRAY, G_ARRAY, IN_SCALE_POS, OUT_SCALE_POS, tb_O_DATA_ARRAY,tb_O_DATA_ARRAY_float);
        
        tb_one_over_rms = 1/$sqrt(tb_float_square_sum/K);


        repeat(2) @(negedge clk);
        in_K = K;
        in_K_vld = 1;
        @(negedge clk);
        in_K_vld = 0;
        

        //VECTOR INPUT
        // repeat(10) @(negedge clk);
        for(int i = 0; i< (DATA_NUM+BUS_NUM-1)/BUS_NUM;i++)begin
            in_fixed_data_vld = 0;
            in_fixed_data = 0;
            for(int j =0; j < BUS_NUM; j++)begin
                if(i * BUS_NUM + j < DATA_NUM)begin
                    in_fixed_data_vld[j] = 1;
                    in_fixed_data[j] = I_DATA_ARRAY[i*BUS_NUM+j];
                end
            end
            @(negedge clk);
        end
        in_fixed_data_vld  = 0;

        // repeat(100000) @(negedge clk);
        // $finish();
        `ifndef SYNTH
            for(int i = 0; i< (DATA_NUM+BUS_NUM-1)/BUS_NUM;i++)begin
                for(int j =0; j < BUS_NUM; j++)begin
                    if(i * BUS_NUM + j < DATA_NUM)begin
                        temp_var = $abs((bitsbfloat16_to_shortreal(inst_RMSnorm.gamma_array[i][j*16+:16])- G_ARRAY[i * BUS_NUM + j])) 
                        / $abs(G_ARRAY[i * BUS_NUM + j]);
                        if(temp_var >= ratio_error ) begin
                            ERROR_FLAG = 1;
                            $display("ERROR-- In gamma Wrong --ERROR");
                        end
                        if(inst_RMSnorm.gamma_vld_array[i][j] != 1)begin
                            ERROR_FLAG = 1;
                            $display("ERROR-- In gamma valid Wrong --ERROR, at %0d",i * BUS_NUM + j);
                        end
                    end
                end
            end

            if(ERROR_FLAG==0)begin
                $display("In gamma Pass!!!");
            end



            while(1)begin
                @(negedge clk);
                if (float_square_sum_vld_flag)
                    break;
            end
            // repeat(100) @(negedge clk);

            $display("fix square sum: %0d, scale pos: %0d",inst_RMSnorm.fixed_square_sum, 2*IN_SCALE_POS);
            $display("float tb square sum: %0f",tb_float_square_sum);
            $display("float square sum: %0f",bitsbfloat16_to_shortreal(inst_RMSnorm.float_square_sum));


            for(int i = 0; i< (DATA_NUM+BUS_NUM-1)/BUS_NUM;i++)begin
                for(int j =0; j < BUS_NUM; j++)begin
                        if(i * BUS_NUM + j < DATA_NUM)begin
                            temp_var = $abs(
                            (bitsbfloat16_to_shortreal(inst_RMSnorm.in_float_data_array[i][j*16+:16])
                            - $itor($signed(I_DATA_ARRAY[i * BUS_NUM + j])) * $pow(2, -IN_SCALE_POS))
                            ) / $itor($signed(I_DATA_ARRAY[i * BUS_NUM + j])) * $pow(2, -IN_SCALE_POS);
                            if(temp_var >= ratio_error ) begin
                                ERROR_FLAG = 1;
                                $display("ERROR-- In fix2float Wrong --ERROR, at %0d, diff: %0f, inst_value: %0f, tb_value: %0f",
                                        i * BUS_NUM + j,    temp_var,   bitsbfloat16_to_shortreal(inst_RMSnorm.in_float_data_array[i][j*32+:32]),    $itor($signed(I_DATA_ARRAY[i * BUS_NUM + j])) * $pow(2, -IN_SCALE_POS));
                            end
                            if(inst_RMSnorm.in_float_data_vld_array[i][j] != 1)begin
                                ERROR_FLAG = 1;
                                $display("ERROR-- In fix2float valid Wrong --ERROR, at %0d",i * BUS_NUM + j);
                            end
                        end
                end
            end
            if(ERROR_FLAG==0)begin
                $display("In fix2float Pass!!!");
            end

            while(1)begin
                @(negedge clk);
                if (one_over_rms_vld_flag)
                    break;
            end
            
            // repeat(100) @(negedge clk);
            if($abs(tb_one_over_rms - bitsbfloat16_to_shortreal(inst_RMSnorm.one_over_rms)) / $abs(tb_one_over_rms) >= ratio_error)begin
                ERROR_FLAG = 1;
                $display("one_over_rms wrong");
            end
            if(ERROR_FLAG==0)begin
                $display("one_over_rms Pass!!!");
            end
        `endif
        while(1)begin
            @(negedge clk);
            if(out_fixed_data_last)
                break;
        end
        repeat(1) @(negedge clk);
        for(int i =0;i <DATA_NUM;i++)begin
            //  $display("float value of RMSnorm: %0f",tb_O_DATA_ARRAY_float[i]);
        end

        `ifndef SYNTH
            for(int i =0;i <DATA_NUM;i++)begin
                if($abs(tb_O_DATA_ARRAY_float[i] - RMSnorm_O_DATA_ARRAY_float[i]) / $abs(tb_O_DATA_ARRAY_float[i]) >= ratio_error)begin
                    ERROR_FLAG = 1;
                    diff_cnt = diff_cnt + 1;
                    $display("float: DUT differ from golden gen i=%d, %0f,   %0f",i,tb_O_DATA_ARRAY_float[i],RMSnorm_O_DATA_ARRAY_float[i]);
                end
            end
            $display("float diff_cnt = %0d when ratio error is: %0f",diff_cnt, ratio_error);
            if(ERROR_FLAG == 0)begin
                $display("Out float PASS!!!!");
            end
        `endif

        diff_cnt = 0;
        for(int i =0;i <DATA_NUM;i++)begin
            temp1 = tb_O_DATA_ARRAY[i];
            temp2 = RMSnorm_O_DATA_ARRAY[i];

            if($abs(temp1 - temp2) == 1)
                diff_cnt = diff_cnt + 1;

            if($abs(temp1 - temp2) > 1)begin
                ERROR_FLAG = 1;
                $display("fixed: DUT differ from golden gen %08b,   %08b",tb_O_DATA_ARRAY[i],RMSnorm_O_DATA_ARRAY[i]);
            end
        end

        $display("fixed diff_cnt = %0d",diff_cnt);

        if(ERROR_FLAG == 0)begin
            $display("Out fix PASS!!!!");
        end

        if(ERROR_FLAG == 0)begin
            $display("ALL PASS!!!!");
        end
        else begin
            $display("SOMETHING WRONG!!!");
        end

        //tb reset for next iteration
        tb_float_square_sum = 0;
        float_square_sum_vld_flag = 0;
        $display("\n\n\n\n\n");

    end

    file = $fopen("../comp/vector_engine/RMSnorm/tb/RMSnorm_fixed.txt", "w");
    for (int i =0;i<DATA_NUM;i++) begin
        $fwrite(file, "%0d\n", RMSnorm_O_DATA_ARRAY[i]);
    end
    $fclose(file);

    `ifndef SYNTH
        file = $fopen("../comp/vector_engine/RMSnorm/tb/RMSnorm_float.txt", "w");
        for (int i =0;i<DATA_NUM;i++) begin
            $fwrite(file, "%.15f\n", RMSnorm_O_DATA_ARRAY_float[i]);
        end
        $fclose(file);
    `endif

    file = $fopen("../comp/vector_engine/RMSnorm/tb/RMS_in_fixed.txt", "w");
    for (int i =0;i<DATA_NUM;i++) begin
        $fwrite(file, "%0d\n", real_input_vector_array_fixed[i]);
    end
    $fclose(file);
    // $stop();
    $finish();
end

int RMSnorm_O_DATA_ARRAY_wr_ptr = 0;
always  begin
    @(negedge clk);
    if(|out_fixed_data_vld)begin
        RMSnorm_O_DATA_ARRAY_wr_ptr = RMSnorm_O_DATA_ARRAY_wr_ptr;
        for(int i =0; i< BUS_NUM;i++)begin
            if(out_fixed_data_vld[i] == 1)begin
                RMSnorm_O_DATA_ARRAY[RMSnorm_O_DATA_ARRAY_wr_ptr] = out_fixed_data[i];
                RMSnorm_O_DATA_ARRAY_wr_ptr = RMSnorm_O_DATA_ARRAY_wr_ptr + 1;
            end
        end
    end
end

`ifndef SYNTH
int RMSnorm_O_DATA_ARRAY_float_wr_ptr = 0;
always  begin
    @(negedge clk);
    if(|inst_RMSnorm.float_RMSnorm_vld)begin
        RMSnorm_O_DATA_ARRAY_float_wr_ptr = RMSnorm_O_DATA_ARRAY_float_wr_ptr;
        for(int i =0; i< BUS_NUM;i++)begin
            if(inst_RMSnorm.float_RMSnorm_vld[i] == 1)begin
                RMSnorm_O_DATA_ARRAY_float[RMSnorm_O_DATA_ARRAY_float_wr_ptr] = bitsbfloat16_to_shortreal(inst_RMSnorm.float_RMSnorm[i]);
                RMSnorm_O_DATA_ARRAY_float_wr_ptr = RMSnorm_O_DATA_ARRAY_float_wr_ptr + 1;
            end
        end
    end
end
`endif

`ifndef SYNTH
always  begin
    @(negedge clk);
    if(inst_RMSnorm.float_square_sum_vld)
        float_square_sum_vld_flag = 1;
end

always  begin
    @(negedge clk);
    if(inst_RMSnorm.one_over_rms_vld)
        one_over_rms_vld_flag = 1;
end
`endif

endmodule
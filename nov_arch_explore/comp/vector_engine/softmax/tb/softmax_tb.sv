`timescale 1ns/1ps

module softmax_tb;

    // Parameters
    parameter integer sig_width = 7;
    parameter integer exp_width = 8;
    parameter integer BUS_NUM = 8;
    parameter integer BUS_NUM_WIDTH = 3;
    parameter integer DATA_NUM = 384;
    parameter integer DATA_NUM_WIDTH = 10;
    parameter integer ITER_NUM = 3;
    parameter integer SCALA_POS_WIDTH = 5;
    parameter integer FIXED_DATA_WIDTH = 8;
    parameter integer MEM_WIDTH = (BUS_NUM*FIXED_DATA_WIDTH);
    parameter integer MEM_DEPTH = 512;

    parameter integer IN_SCALE_POS = -1;
    parameter integer OUT_SCALE_POS = 6;
    
    // Signals
    logic clk;
    logic rst_n;

    logic signed [SCALA_POS_WIDTH-1:0] in_scale_pos;
    logic in_scale_pos_vld; 
    logic signed [SCALA_POS_WIDTH-1:0] out_scale_pos;  
    logic out_scale_pos_vld;
    logic [DATA_NUM_WIDTH-1:0] in_data_num;
    logic in_data_num_vld;

    logic signed [BUS_NUM-1:0][FIXED_DATA_WIDTH-1:0] in_fixed_data;
    logic [BUS_NUM-1:0] in_fixed_data_vld;
    logic               in_ready;
    logic signed [BUS_NUM-1:0][FIXED_DATA_WIDTH-1:0] out_fixed_data;
    logic [BUS_NUM-1:0] out_fixed_data_vld;
    logic out_fixed_last;

    // **important** From Junyi
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

    // Instantiate the Device Under Test (DUT)
    softmax #(
        .sig_width(sig_width),
        .exp_width(exp_width),
        .BUS_NUM(BUS_NUM),
        .BUS_NUM_WIDTH(BUS_NUM_WIDTH),
        .DATA_NUM_WIDTH(DATA_NUM_WIDTH),
        .SCALA_POS_WIDTH(SCALA_POS_WIDTH),
        .FIXED_DATA_WIDTH(FIXED_DATA_WIDTH),
        .MEM_WIDTH(MEM_WIDTH),
        .MEM_DEPTH(MEM_DEPTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .in_scale_pos(in_scale_pos),
        .in_scale_pos_vld(in_scale_pos_vld),
        .out_scale_pos(out_scale_pos),
        .out_scale_pos_vld(out_scale_pos_vld),
        .in_data_num(in_data_num),
        .in_data_num_vld(in_data_num_vld),
        .in_fixed_data(in_fixed_data),
        .in_fixed_data_vld(in_fixed_data_vld),
        .in_ready(in_ready),
        .out_fixed_data(out_fixed_data),
        .out_fixed_data_vld(out_fixed_data_vld),
        .out_fixed_last(out_fixed_last)
    );
    int tb_max=32'h8000_0000;
    logic signed [DATA_NUM-1:0][7:0] tb_xi_max;
    shortreal                        tb_i2flt_xi_max       [DATA_NUM-1:0];
    shortreal                        tb_exp_xi_max         [DATA_NUM-1:0];
    shortreal                        tb_acc_out;
    shortreal                        tb_ln_out;
    shortreal                        tb_subtract_result    [DATA_NUM-1:0];
    shortreal                        tb_final_exp_result   [DATA_NUM-1:0];
    int                              tb_out_fixed_data     [DATA_NUM-1:0];

    shortreal real_input_vector_array [DATA_NUM-1:0];
    logic signed [ITER_NUM-1:0][DATA_NUM-1:0][7:0] real_input_vector_array_fixed;

    int ERROR_FLAG;
    // real ratio_error = 1.5e-2;
    real ratio_error = 1.5e-1;
    real ratio_error_final_exp = 1e-1;

    int stage0_iter = 0;
    int stage1_iter = 0;
    int stage2_iter = 0; 

    //read file
    int file;
    int index = 0;
    shortreal float_value;

    initial begin
        file = $fopen("../comp/vector_engine/softmax/tb/softmax_in.txt", "r");
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
        for(int i = 0; i<ITER_NUM;i++) begin
            for(int j = 0; j<DATA_NUM;j++)begin
                float_value = real_input_vector_array[j] * $pow(2,IN_SCALE_POS)+i;//make sure each iteration has different input
                index  = int'(float_value);
                if(index > 127)
                    index = 127;
                if(index < -128)
                    index = -128;
                real_input_vector_array_fixed[i][j] = index;
                // $display("%0d",index);
            end
        end
    end

    always begin
        #(`CLOCK_PERIOD/2.0) clk = ~clk;
    end

    initial begin
        $fsdbDumpfile("softmax.fsdb");
	    $fsdbDumpvars(0,"+all",softmax_tb); //"+all" enables all  signal dumping including Mem, packed array, etc.
        rst_n = 0;
        clk = 0;

        repeat (5) @(negedge clk);
        
        rst_n = 1;
        $display("***********STARTING TESTBENCH CHECKING***********");
        // Input Data
        in_fixed_data = 0;
        in_fixed_data_vld = 0;

        //config the parameters
        in_scale_pos = IN_SCALE_POS;
        in_scale_pos_vld = 1;
        out_scale_pos = OUT_SCALE_POS;
        out_scale_pos_vld = 1;
        in_data_num = DATA_NUM;
        in_data_num_vld = 1;

        @(negedge clk)
        in_scale_pos_vld = 0;
        out_scale_pos_vld = 0;
        in_data_num_vld = 0;

        //VECTOR INPUT
        for(int i = 0; i< (DATA_NUM+BUS_NUM-1)/BUS_NUM;i++)begin
            in_fixed_data_vld = 0;
            in_fixed_data = 0;
            for(int j =0; j < BUS_NUM; j++)begin
                if(i * BUS_NUM + j < DATA_NUM)begin
                    in_fixed_data_vld[j] = 1;
                    in_fixed_data[j] = real_input_vector_array_fixed[stage0_iter][i*BUS_NUM+j];
                end
            end
            @(negedge clk);
        end
        in_fixed_data_vld  = 0;

        //Print fixed input
        file = $fopen("../comp/vector_engine/softmax/tb/softmax_in_fixed.txt", "w");
        for (int j=0;j<ITER_NUM;j++) begin
            $fwrite(file, "ITER_NUM: %0d\n", j);
            for (int i =0;i<DATA_NUM;i++) begin
                $fwrite(file, "%0d\n", $signed(real_input_vector_array_fixed[j][i]));
            end
        end
        $fclose(file);

        ERROR_FLAG = 0;
        while(1)begin
            @(negedge clk);
            if (out_fixed_last && stage2_iter==ITER_NUM-1)
                break;
        end

        repeat (10) @(posedge clk);

        // $display(bitsbfloat16_to_shortreal(16'hc31e));
        // $display(bitsbfloat16_to_shortreal(16'h3f5d));
        // $display(bitsbfloat16_to_shortreal(16'h3d07));
        $display("***********Simulation Passed***********");
        $finish;
    end

    task softmax_tb_compute(   
        input  signed    [DATA_NUM-1:0][7:0]    i_data_array,
        input  int                              in_scale_pos, 
        input  int                              out_scale_pos,
        output signed    [7:0]                  max,
        output signed    [DATA_NUM-1:0][7:0]    in_max_subtract,
        output shortreal    in_max_subtract_flt [DATA_NUM-1:0],
        output shortreal    exp                 [DATA_NUM-1:0],
        output shortreal    exp_sum,
        output shortreal    ln,
        output shortreal    xi_max_ln           [DATA_NUM-1:0],
        output shortreal    o_data_array_real   [DATA_NUM-1:0],
        output int          o_data_array        [DATA_NUM-1:0]);
        
        shortreal exp_sum_temp;
        integer max_temp;
        integer o_data_array_int   [DATA_NUM-1:0];
        shortreal o_data_array_real_scaled   [DATA_NUM-1:0];

        exp_sum_temp =0;
        //max
        max_temp = 32'h8000_0000;
        for(int i =0;i < DATA_NUM; i++)begin
            max_temp = ($signed(i_data_array[i])>$signed(max_temp)) ? $signed(i_data_array[i]) :$signed(max_temp);
            //$display($signed(max_temp));
        end
        max = max_temp;

        //x-max
        for(int i =0;i < DATA_NUM; i++)begin
            in_max_subtract[i] = $signed(i_data_array[i])-$signed(max);
            //$display("x-max value:%d",in_max_subtract[i]);
        end
        
        //i2flt(x-max)
        for(int i =0;i < DATA_NUM; i++)begin
            in_max_subtract_flt[i] = $itor($signed(in_max_subtract[i])) * $pow(2, -in_scale_pos);
            //$display("flt x-max value:%0f",in_max_subtract_flt[i]);
        end

        //exp(x-max)
        for(int i =0;i < DATA_NUM; i++)begin
            exp[i] = $exp(in_max_subtract_flt[i]);
        end

        //exp sum
        for(int i =0;i < DATA_NUM; i++)begin
            exp_sum_temp = exp_sum_temp+exp[i];
        end
        exp_sum = exp_sum_temp;

        //ln
        ln = $ln(exp_sum);

        //i2flt(xi-max)-ln
        for(int i =0;i < DATA_NUM; i++)begin
            xi_max_ln[i] = in_max_subtract_flt[i]-ln;
        end

        //last exp and get odata
        for(int i=0;i<DATA_NUM;i++) begin
            o_data_array_real[i] = $exp(xi_max_ln[i]);
        end

        //fixed output check
        for(int i =0;i < DATA_NUM; i++)begin
            o_data_array_real_scaled[i] = o_data_array_real[i] * $pow(2, out_scale_pos); 
            o_data_array_int[i] = int'(o_data_array_real_scaled[i]); 
            if(o_data_array_int[i] > 127)
                o_data_array[i] = 127;
            else if(o_data_array_int[i]<-128)
                o_data_array[i] = -128;
            else
                o_data_array[i] = o_data_array_int[i];
        end
    endtask


    // STAGE ITER CNT INC
    always begin
        @(negedge clk);

        `ifndef SYNTH
            if (dut.first_stage_w_rst) begin
                stage0_iter = stage0_iter + 1;
            end
            
            if(dut.first_stage_r_rst) begin
                stage1_iter = stage1_iter + 1;
            end
        `endif

        if(out_fixed_last) begin
            stage2_iter = stage2_iter + 1;
        end
    end

    // VECTOR INPUT CONTINUOUS
    always begin
        @(negedge clk);
        if(in_ready && stage0_iter<ITER_NUM) begin
            for(int i = 0; i< (DATA_NUM+BUS_NUM-1)/BUS_NUM;i++)begin
                in_fixed_data_vld = 0;
                in_fixed_data = 0;
                for(int j =0; j < BUS_NUM; j++)begin
                    if(i * BUS_NUM + j < DATA_NUM)begin
                        in_fixed_data_vld[j] = 1;
                        in_fixed_data[j] = real_input_vector_array_fixed[stage0_iter][i*BUS_NUM+j];
                    end
                end
                @(negedge clk);
            end
            in_fixed_data_vld  = 0;
        end
    end

    int dummy;
    shortreal dummy_real [DATA_NUM-1:0];
    int dummy_int [DATA_NUM-1:0];
    shortreal temp_var;
`ifndef SYNTH
    //checking max value
    //stage 0
    always begin
        @(negedge clk);
        if(dut.max_vld) begin
            // //tb max
            // for(int i=0; i< DATA_NUM;i++) begin
            //     tb_max = ($signed(real_input_vector_array_fixed[stage0_iter][i])>$signed(tb_max)) ? $signed(real_input_vector_array_fixed[stage0_iter][i]) :$signed(tb_max);
            // end
            softmax_tb_compute(real_input_vector_array_fixed[stage0_iter], IN_SCALE_POS, OUT_SCALE_POS, tb_max, dummy, dummy_real, dummy_real, dummy, dummy, dummy_real, dummy_real,dummy_int);
            if($signed(dut.max_value) != tb_max) begin
                ERROR_FLAG = 1;
                $display("Hardware max_value:%d, Testbench max_value:%d",$signed(dut.max_value),$signed(tb_max));
                $display("ERROR-- Max Value Wrong --ERROR");
            end
            else begin
                $display("MAX Passed!!!,ITERATION:%d",stage0_iter);
                //$display("Hardware max_value:%d, Testbench max_value:%d",$signed(dut.max_value),$signed(tb_max));
            end
            while(1)begin
                @(negedge clk);
                if (!dut.max_vld)
                    break;
            end
        end
    end

    //checking subtract value
    //stage 1
    logic signed [DATA_NUM-1:0][7:0] xi_max_hw;
    int xi_max_wr_ptr=0;
    always begin
        @(negedge clk);
        if(dut.xi_max_vld) begin
            //tb xi_max
            softmax_tb_compute(real_input_vector_array_fixed[stage1_iter], IN_SCALE_POS, OUT_SCALE_POS, dummy, tb_xi_max, dummy_real, dummy_real, dummy, dummy, dummy_real, dummy_real,dummy_int);
            for(int i = 0; i< BUS_NUM;i++) begin
                if(dut.xi_max_vld[i]) begin
                    xi_max_hw[xi_max_wr_ptr] = dut.xi_max[i];
                    xi_max_wr_ptr = xi_max_wr_ptr + 1;
                end
            end
            if(xi_max_wr_ptr == DATA_NUM) begin
                xi_max_wr_ptr = 0;
                for(int i=0;i< DATA_NUM;i++) begin
                    if(xi_max_hw[i]!=tb_xi_max[i]) begin
                        ERROR_FLAG = 1;
                        $display("Hardware value:%d, Testbench value:%d",$signed(xi_max_hw[i]),$signed(tb_xi_max[i]));
                        $display("ERROR-- XI-MAX Value Wrong --ERROR,%d",i);
                    end
                end
                if(ERROR_FLAG==0)
                    $display("XI-MAX Passed!!!,ITERATION:%d",stage1_iter);
            end
        end
    end

    //checking fix2fp
    logic signed [DATA_NUM-1:0][exp_width+sig_width:0] i2flt_xi_max_hw;
    int i2flt_ptr=0;
    always begin
        @(negedge clk);
        if(dut.flt_xi_max_vld) begin
            //tb xi_max flt
            softmax_tb_compute(real_input_vector_array_fixed[stage1_iter], IN_SCALE_POS, OUT_SCALE_POS, dummy, dummy, tb_i2flt_xi_max, dummy_real, dummy, dummy, dummy_real, dummy_real,dummy_int);
            for(int i = 0; i< BUS_NUM;i++) begin
                if(dut.flt_xi_max_vld[i]) begin
                    i2flt_xi_max_hw[i2flt_ptr] = dut.i2flt_xi_max[i];
                    i2flt_ptr = i2flt_ptr + 1;
                end
            end
            if(i2flt_ptr == DATA_NUM) begin
                i2flt_ptr = 0;
                for(int i=0;i< DATA_NUM;i++) begin
                    //$display("hardware i2flt:%h,float value:%.5f,expected float value:%.5f",i2flt_xi_max_hw[i],bitsbfloat16_to_shortreal(i2flt_xi_max_hw[i]),tb_i2flt_xi_max[i]);
                    if(tb_i2flt_xi_max[i]>0.00001)
                        temp_var = $abs((bitsbfloat16_to_shortreal(i2flt_xi_max_hw[i])- tb_i2flt_xi_max[i])) 
                        / $abs(tb_i2flt_xi_max[i]);
                    else
                        temp_var = $abs((bitsbfloat16_to_shortreal(i2flt_xi_max_hw[i])- tb_i2flt_xi_max[i]));

                    if(temp_var >= ratio_error) begin
                        ERROR_FLAG = 1;
                        $display("Discrepency: %.15f,position:%d",temp_var,i);
                        $display("ERROR-- XI-MAX fix2fp Wrong --ERROR");
                    end
                end
                if(ERROR_FLAG==0)
                    $display("XI-MAX fix2fp Passed!!!,ITERATION:%d",stage1_iter);
            end
        end
    end

    //checking exp
    logic signed [DATA_NUM-1:0][exp_width+sig_width:0] exp_xi_max_hw;
    int exp_ptr=0;
    always begin
        @(negedge clk);
        if(dut.exp_xi_max_vld) begin
            //tb exp
            softmax_tb_compute(real_input_vector_array_fixed[stage1_iter], IN_SCALE_POS, OUT_SCALE_POS, dummy, dummy, dummy_real, tb_exp_xi_max, dummy, dummy, dummy_real, dummy_real,dummy_int);
            for(int i = 0; i< BUS_NUM;i++) begin
                if(dut.exp_xi_max_vld[i]) begin
                    exp_xi_max_hw[exp_ptr] = dut.exp_xi_max[i];
                    exp_ptr = exp_ptr + 1;
                end
            end
            if(exp_ptr == DATA_NUM) begin
                exp_ptr = 0;
                for(int i=0;i< DATA_NUM;i++) begin
                    //$display("hardware exp:%h,float value:%.5f,expected float value:%.5f",exp_xi_max_hw[i],bitsbfloat16_to_shortreal(exp_xi_max_hw[i]),tb_exp_xi_max[i]);
                    if(tb_exp_xi_max[i]>0.00001)
                        temp_var = $abs((bitsbfloat16_to_shortreal(exp_xi_max_hw[i])- tb_exp_xi_max[i]))
                        / $abs(tb_exp_xi_max[i]);
                    else
                        temp_var = $abs((bitsbfloat16_to_shortreal(exp_xi_max_hw[i])- tb_exp_xi_max[i]));

                    if(temp_var >= ratio_error) begin
                        ERROR_FLAG = 1;
                        $display("hardware exp haha:%h,float value:%.15f,expected float value:%.15f",exp_xi_max_hw[i],bitsbfloat16_to_shortreal(exp_xi_max_hw[i]),tb_exp_xi_max[i]);
                        $display("Discrepency: %.15f,position:%d",temp_var,i);
                        $display("ERROR-- EXP Wrong --ERROR");
                    end
                end
                if(ERROR_FLAG==0)
                    $display("EXP Xi-max Passed!!!,ITERATION:%d",stage1_iter);
            end
        end
    end

    //checking exp_sum
    always begin
        @(negedge clk);
        if(dut.acc_vld) begin
            softmax_tb_compute(real_input_vector_array_fixed[stage1_iter-1], IN_SCALE_POS, OUT_SCALE_POS, dummy, dummy, dummy_real, dummy_real, tb_acc_out, dummy, dummy_real, dummy_real,dummy_int);
            //$display("hardware exp_sum:%h,float value:%.5f,expected float value:%.5f",dut.acc_out,bitsbfloat16_to_shortreal(dut.acc_out),tb_acc_out);
            if(tb_acc_out >0.00001)
                temp_var = $abs((bitsbfloat16_to_shortreal(dut.acc_out)- tb_acc_out))/ $abs(tb_acc_out);
            else
                temp_var = $abs((bitsbfloat16_to_shortreal(dut.acc_out)- tb_acc_out));

            if(temp_var >= ratio_error) begin
                ERROR_FLAG = 1;
                $display("hardware exp_sum:%h,float value:%f,expected float value:%f",dut.acc_out,bitsbfloat16_to_shortreal(dut.acc_out),tb_acc_out);
                $display("Discrepency: %.15f",temp_var);
                $display("ERROR-- exp_sum Value Wrong --ERROR, Iter:%d",stage1_iter-1);
            end
            else begin
                $display("EXP_SUM Passed!!!,ITERATION:%d",stage1_iter-1);
            end
        end
    end

    //checking natural logarithm
    always begin
        @(negedge clk);
        if(dut.ln_vld) begin
            softmax_tb_compute(real_input_vector_array_fixed[stage2_iter], IN_SCALE_POS, OUT_SCALE_POS, dummy, dummy, dummy_real, dummy_real, dummy, tb_ln_out, dummy_real, dummy_real,dummy_int);
            //$display("hardware exp_sum:%h,float value:%.5f,expected float value:%.5f",dut.acc_out,bitsbfloat16_to_shortreal(dut.acc_out),tb_acc_out);
            if(tb_ln_out >0.003)
                temp_var = $abs((bitsbfloat16_to_shortreal(dut.ln_out)- tb_ln_out))/ $abs(tb_ln_out);
            else
                temp_var = $abs((bitsbfloat16_to_shortreal(dut.ln_out)- tb_ln_out));
            
            if(temp_var >= ratio_error) begin
                ERROR_FLAG = 1;
                $display("hardware ln:%h,float value:%.5f,expected float value:%.5f",dut.ln_out,bitsbfloat16_to_shortreal(dut.ln_out),tb_ln_out);
                $display("Discrepency: %.15f",temp_var);
                $display("ERROR-- Ln_out Value Wrong --ERROR");
            end
            else begin
                $display("LN out Passed!!!,ITERATION:%d",stage2_iter);
            end
        end
    end

    //checking subtract result
    logic signed [DATA_NUM-1:0][exp_width+sig_width:0] subtract_result_hw;
    int subtract_ptr=0;
    always begin
        @(negedge clk);
        if(dut.subtract_vld) begin
            //tb exp
            softmax_tb_compute(real_input_vector_array_fixed[stage2_iter], IN_SCALE_POS, OUT_SCALE_POS, dummy, dummy, dummy_real, dummy_real, dummy, dummy, tb_subtract_result, dummy_real,dummy_int);
            for(int i = 0; i< BUS_NUM;i++) begin
                if(dut.subtract_vld[i]) begin
                    subtract_result_hw[subtract_ptr] = dut.subtract_result[i];
                    subtract_ptr = subtract_ptr + 1;
                end
            end
            if(subtract_ptr == DATA_NUM) begin
                subtract_ptr = 0;
                for(int i=0;i< DATA_NUM;i++) begin
                    //$display("hardware exp:%h,float value:%.5f,expected float value:%.5f",exp_xi_max_hw[i],bitsbfloat16_to_shortreal(exp_xi_max_hw[i]),tb_exp_xi_max[i]);
                    if(tb_subtract_result[i] > 0.00001)
                        temp_var = $abs((bitsbfloat16_to_shortreal(subtract_result_hw[i])- tb_subtract_result[i]))
                        / $abs(tb_subtract_result[i]);
                    else
                        temp_var = $abs((bitsbfloat16_to_shortreal(subtract_result_hw[i])- tb_subtract_result[i]));

                    if(temp_var >= ratio_error) begin
                        ERROR_FLAG = 1;
                        $display("Discrepency: %.15f,position:%d",temp_var,i);
                        $display("ERROR-- Subtract Result Wrong --ERROR");
                    end
                end
                if(ERROR_FLAG==0)
                    $display("Subtract Result Passed!!!,ITERATION:%d",stage2_iter);
            end
        end
    end

    //checking final exponential
    logic signed [DATA_NUM-1:0][exp_width+sig_width:0] final_exp_result_hw;
    int final_exp_ptr=0;
    always begin
        @(negedge clk);
        if(dut.final_exp_vld) begin
            //tb final exp
            softmax_tb_compute(real_input_vector_array_fixed[stage2_iter], IN_SCALE_POS, OUT_SCALE_POS, dummy, dummy, dummy_real, dummy_real, dummy, dummy, dummy_real, tb_final_exp_result,dummy_int);
            for(int i = 0; i< BUS_NUM;i++) begin
                if(dut.final_exp_vld[i]) begin
                    final_exp_result_hw[final_exp_ptr] = dut.final_exp_result[i];
                    final_exp_ptr = final_exp_ptr + 1;
                end
            end
            if(final_exp_ptr == DATA_NUM) begin
                final_exp_ptr = 0;
                for(int i=0;i< DATA_NUM;i++) begin
                    //$display("hardware exp:%h,float value:%.5f,expected float value:%.5f",final_exp_result_hw[i],bitsbfloat16_to_shortreal(final_exp_result_hw[i]),tb_final_exp_result[i]);
                    if(tb_final_exp_result[i] >0.00001)
                        temp_var = $abs((bitsbfloat16_to_shortreal(final_exp_result_hw[i])- tb_final_exp_result[i]))
                        / $abs(tb_final_exp_result[i]);
                    else
                        temp_var = $abs((bitsbfloat16_to_shortreal(final_exp_result_hw[i])- tb_final_exp_result[i]));

                    if(temp_var >= ratio_error_final_exp) begin
                        ERROR_FLAG = 1;
                        $display("hardware exp:%h,float value:%.5f,expected float value:%.5f",final_exp_result_hw[i],bitsbfloat16_to_shortreal(final_exp_result_hw[i]),tb_final_exp_result[i]);
                        $display("Discrepency: %.15f,position:%d",temp_var,i);
                        $display("ERROR-- Final Exponential Result Wrong --ERROR");
                    end
                end
                if(ERROR_FLAG==0)
                    $display("Final Exponential Result Passed!!!,ITERATION:%d",stage2_iter);
            end
        end
    end
`endif

    //checking final fixed output
    logic signed [DATA_NUM-1:0][7:0] out_fixed_data_hw;
    int out_ptr=0;
    always begin
        @(negedge clk);
        if(out_fixed_data_vld) begin
            //tb final exp
            for(int i = 0; i< BUS_NUM;i++) begin
                if(out_fixed_data_vld[i]) begin
                    out_fixed_data_hw[out_ptr] = out_fixed_data[i];
                    out_ptr = out_ptr + 1;
                end
            end
            if(out_ptr == DATA_NUM) begin
                softmax_tb_compute(real_input_vector_array_fixed[stage2_iter-1], IN_SCALE_POS, OUT_SCALE_POS, dummy, dummy, dummy_real, dummy_real, dummy, dummy, dummy_real ,dummy_real,tb_out_fixed_data);
                out_ptr = 0;
                for(int i=0;i< DATA_NUM;i++) begin
                    temp_var = $abs($signed(out_fixed_data_hw[i])-tb_out_fixed_data[i]);
                    $display("hardware fixed:%d,expected fixed value:%d,location:%d",out_fixed_data_hw[i],tb_out_fixed_data[i],i);
                    if(temp_var > 1) begin
                        ERROR_FLAG = 1;
                        $display("hardware fixed:%d,expected fixed value:%d,location:%d",out_fixed_data_hw[i],tb_out_fixed_data[i],i);
                        $display("ERROR-- Final fixed Result Wrong --ERROR");
                    end
                end
                if(ERROR_FLAG==0)
                    $display("Final fixed Result Passed!!!,ITERATION:%d",stage2_iter-1);
            end
        end
    end

    always begin
        @(negedge clk);
        if(ERROR_FLAG) begin
            // $display(bitsbfloat16_to_shortreal(16'h3774));
            // $display(bitsbfloat16_to_shortreal(16'h3a9b));

            $display("SOMETHING WRONG!!!, time: %0t",$time);
            $finish;
        end
    end
endmodule
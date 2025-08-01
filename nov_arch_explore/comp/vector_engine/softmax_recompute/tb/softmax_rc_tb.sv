module softmax_rc_tb;

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

parameter integer FIXED_SQUARE_SUM_WIDTH = 24;
parameter BUS_NUM = `MAC_MULT_NUM;
parameter DATA_NUM = 129; //`MAX_CONTEXT_LENGTH
parameter GBUS_DATA_WIDTH = `GBUS_DATA_WIDTH;

parameter RC_SHIFT = 16;
shortreal softmax_dequant_scale = 0.0123456;
shortreal softmax_quant_scale = 123;

logic                                                       clk;
logic                                                       rst_n;

logic                                                       rc_cfg_vld;
RC_CONFIG                                                   rc_cfg;     

logic        [`RECOMPUTE_SCALE_WIDTH-1:0]                   rc_scale;
logic                                                       rc_scale_vld;


CONTROL_STATE control_state;
logic control_state_update;

USER_CONFIG usr_cfg;
logic usr_cfg_vld;

logic [(`MAC_MULT_NUM * `IDATA_WIDTH)-1:0] in_bus_data;
logic in_bus_data_vld;

logic gbus_wen;
logic [GBUS_DATA_WIDTH-1:0] gbus_wdata;
BUS_ADDR gbus_addr;
logic clear;

logic [(`MAC_MULT_NUM * `IDATA_WIDTH)-1:0] out_bus_data;
logic                                      out_bus_data_vld;

shortreal   soft_one_over_exp_sum;
logic   [7:0]   tb_o_data_array [DATA_NUM-1:0];
shortreal   tb_xi_max_flt  [DATA_NUM-1:0];

task softmax_rc_soft(   input   [7:0]       i_data_array    [DATA_NUM-1:0],
                        output  shortreal   one_over_exp_sum,
                        output  [7:0]       o_data_array    [DATA_NUM-1:0],
                        output  shortreal   xi_max_flt      [DATA_NUM-1:0]);
    integer max;
    integer max_temp;
    integer xi_max [DATA_NUM-1:0];
    shortreal xi_max_flt_temp [DATA_NUM-1:0];
    shortreal exp [DATA_NUM-1:0];
    shortreal o_data_array_real_scaled   [DATA_NUM-1:0];
    integer   o_data_array_int   [DATA_NUM-1:0];
    shortreal exp_sum_temp, exp_sum;

    max_temp = 32'h8000_0000;
    for(int i =0;i < DATA_NUM; i++)begin
        max_temp = ($signed(i_data_array[i])>$signed(max_temp)) ? $signed(i_data_array[i]) :$signed(max_temp);
        //$display($signed(max_temp));
    end
    max = max_temp;
    // $display($signed(max));

    for(int i =0;i < DATA_NUM; i++)begin
        xi_max[i] = $signed(i_data_array[i])-$signed(max);
        //$display("x-max value:%d",$signed(xi_max[i]));
    end

    for(int i =0;i < DATA_NUM; i++)begin
        xi_max_flt_temp[i] = $itor($signed(xi_max[i])) * softmax_dequant_scale;
        //$display("flt x-max value:%0f",xi_max_flt[i]);
    end
    xi_max_flt = xi_max_flt_temp;

    for(int i =0;i < DATA_NUM; i++)begin
        exp[i] = $exp(xi_max_flt[i]);
    end

    //FIXED EXP OUTPUT CHECK
    for(int i =0;i < DATA_NUM; i++)begin
        o_data_array_real_scaled[i] = exp[i] * softmax_quant_scale; 
        o_data_array_int[i] = int'(o_data_array_real_scaled[i]); 
        if(o_data_array_int[i] > 127)
            o_data_array[i] = 127;
        else if(o_data_array_int[i]<-128)
            o_data_array[i] = -128;
        else
            o_data_array[i] = o_data_array_int[i];
    end

    exp_sum_temp = 0;
    for(int i =0;i < DATA_NUM; i++)begin
        exp_sum_temp = exp_sum_temp+exp[i];
    end
    exp_sum = exp_sum_temp;

    one_over_exp_sum = 1/exp_sum;
endtask

always begin
    #50 clk = ~clk;
end

logic [7:0] tb_indata_array [DATA_NUM-1:0];

integer iter = 0;
integer iter1 = 0;
initial begin
    for(int i =0; i < DATA_NUM; i++)begin
        tb_indata_array[i] = $random;
    end
    softmax_rc_soft(tb_indata_array, soft_one_over_exp_sum, tb_o_data_array, tb_xi_max_flt);
    $display("Software Value, soft_one_over_exp_sum: %f", soft_one_over_exp_sum);
end

initial begin
    clk = 0;
    rst_n = 0;
    rc_cfg_vld = 0;
    usr_cfg_vld = 0;
    control_state_update = 0;

    rc_cfg.rc_shift_softmax = RC_SHIFT;
    rc_cfg.softmax_input_dequant_scale = shortreal_to_bitsbfloat16(softmax_dequant_scale);
    rc_cfg.softmax_exp_quant_scale =  shortreal_to_bitsbfloat16(softmax_quant_scale);

    in_bus_data = 0;
    in_bus_data_vld = 0;

    gbus_wen = 0;
    gbus_wdata = 0;
    gbus_addr = 0;

    clear = 0;

    usr_cfg.user_token_cnt = DATA_NUM -1;
    usr_cfg.user_kv_cache_not_full = 1;

    repeat(10) @(negedge clk);
    rst_n = 1;
    repeat(10) @(negedge clk);
    rc_cfg_vld = 1;
    usr_cfg_vld = 1;
    
    @(negedge clk);
    rc_cfg_vld = 0;
    usr_cfg_vld = 0;
    control_state_update = 0;

    for(int q=0;q<5;q++) begin
        //QKT
        control_state = ATT_QK_STATE;
        control_state_update = 1;
        @(negedge clk);
        control_state_update = 0;
        @(negedge clk);

        for(int i =0; i< DATA_NUM;i++)begin
            gbus_wdata = tb_indata_array[i];
            gbus_wen = 1;
            @(negedge clk);
            // gbus_addr = gbus_addr + 1;
            //TODO: Add gbus_addr generation logic, but it would be better to test in top testbench
        end
        gbus_wen = 0;

        //PV
        @(negedge clk);
        control_state = ATT_PV_STATE;
        control_state_update = 1;
        @(negedge clk);
        control_state_update = 0;
        @(negedge clk);

        for(int i =0; i< $ceil($itor(DATA_NUM)/`MAC_MULT_NUM);i++)begin
            for(int j = 0; j< `MAC_MULT_NUM;j++)begin
                in_bus_data[j*`IDATA_WIDTH+:`IDATA_WIDTH] = tb_indata_array[i*`MAC_MULT_NUM+j];
            end
            in_bus_data_vld = 1;
            @(negedge clk);
        end
        in_bus_data_vld = 0;

        //Check if we have all the exponent output
        while(1) begin
            @(negedge clk);
            if(iter ==  $ceil($itor(DATA_NUM)/`MAC_MULT_NUM)) begin
                iter = 0;
                iter1 = 0;
                break;
            end
        end
        //wait to get the exponent sum
        repeat(20) @(negedge clk);
        //clear
        clear = 1;
        @(negedge clk);
        clear = 0;
    end
        
    $finish();
end

real ratio_error = 0;
always  begin
    @(negedge clk);
    if(rst_n != 0 && rc_scale_vld)begin
        $display("Hardware Value, one_over_exp_sum: %f",rc_scale * 1.0 /($pow(2, RC_SHIFT)));
        ratio_error = $abs((rc_scale * 1.0 /($pow(2, RC_SHIFT)) - soft_one_over_exp_sum) / soft_one_over_exp_sum);
        $display("*** ratio_error: %f ***", ratio_error);
    end
end

always  begin
    @(negedge clk);
    if(rst_n != 0 && out_bus_data_vld)begin
        for(int j = 0; j< `MAC_MULT_NUM;j++) begin
            if($abs(int'($signed(out_bus_data[j*`IDATA_WIDTH+:`IDATA_WIDTH]))-int'($signed(tb_o_data_array[iter*`MAC_MULT_NUM+j]))) > 1) begin
                $display("Iter: %d, Data position: %d, Hardware output: %d, Software output: %d",iter, j, $signed(out_bus_data[j*`IDATA_WIDTH+:`IDATA_WIDTH]), $signed(tb_o_data_array[iter*`MAC_MULT_NUM+j]));
            end
        end
        iter = iter + 1;
    end
end

// always begin
//     @(negedge clk);
//     if(softmax_rc_inst.i2flt_xi_max_dequant_vld_out) begin
//         for(int j = 0; j< `MAC_MULT_NUM;j++) begin
//             $display("Iter: %d, Data position: %d, Hardware output: %f, Software output: %f",iter1, j, bitsbfloat16_to_shortreal(softmax_rc_inst.i2flt_xi_max_dequant[j]), tb_xi_max_flt[iter1*`MAC_MULT_NUM+j]);
//         end
//         iter1 = iter1 + 1;
//     end
// end

// Instantiate softmax_rc module
softmax_rc #(
    .GBUS_DATA_WIDTH(`GBUS_DATA_WIDTH),
    .BUS_NUM(`MAC_MULT_NUM),
    .FIXED_DATA_WIDTH(`IDATA_WIDTH),
    .EXP_STAGE_DELAY(1)         // Ensure parameters are set correctly
) softmax_rc_inst (
    .clk(clk),
    .rst_n(rst_n),
    .control_state(control_state),
    .control_state_update(control_state_update),
    .rc_cfg_vld(rc_cfg_vld),
    .rc_cfg(rc_cfg),
    .usr_cfg(usr_cfg),
    .usr_cfg_vld(usr_cfg_vld),
    .in_bus_data(in_bus_data),
    .in_bus_data_vld(in_bus_data_vld),
    .gbus_wen(gbus_wen),
    .gbus_wdata(gbus_wdata),
    .gbus_addr(gbus_addr),
    .clear(clear),
    .rc_scale(rc_scale),
    .rc_scale_vld(rc_scale_vld),
    .out_bus_data(out_bus_data),
    .out_bus_data_vld(out_bus_data_vld)
);
    
endmodule
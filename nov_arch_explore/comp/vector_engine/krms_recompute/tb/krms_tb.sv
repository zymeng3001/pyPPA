module krms_tb;

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
parameter DATA_NUM = `MAX_EMBD_SIZE;

parameter RC_SHIFT = 16;
parameter RMS_K = 120;
shortreal rms_dequant_scale_square = 0.0123456;

logic                                                     clk;
logic                                                     rst_n;

logic                                                     start;
logic                                                     rc_cfg_vld;
RC_CONFIG                                                 rc_cfg;     

logic signed [BUS_NUM-1:0][`IDATA_WIDTH-1:0]              in_fixed_data; //vector的长度一定是BUS NUM的倍数
logic                                                     in_fixed_data_vld;

logic        [`RECOMPUTE_SCALE_WIDTH-1:0]                 rc_scale;
logic                                                     rc_scale_vld;

shortreal soft_one_over_krms;

task krms_soft(     input    [7:0] i_data_array[DATA_NUM-1:0],
                    output shortreal one_over_krms);
    shortreal real_data_array [DATA_NUM-1:0];
    shortreal square_sum;
    shortreal rms;
    square_sum = 0;
    rms = 0;

    for(int i =0;i < DATA_NUM; i++)begin
        real_data_array[i] = $itor($signed(i_data_array[i]));
    end

    for(int i =0;i < RMS_K; i++)begin //first K kRMSnorm
        square_sum = square_sum + real_data_array[i] * real_data_array[i];
    end
    square_sum = square_sum * rms_dequant_scale_square;
    rms = $sqrt(square_sum/RMS_K * 1.0);
    one_over_krms = 1/rms;
endtask

always begin
    #50 clk = ~clk;
end

logic [7:0] tb_indata_array [DATA_NUM-1:0];

initial begin
    for(int i =0; i < DATA_NUM; i++)begin
        tb_indata_array[i] = $random;
    end
    krms_soft(tb_indata_array, soft_one_over_krms);
    $display("Software Value, soft_one_over_krms: %f", soft_one_over_krms);
end

initial begin
    clk = 0;
    rst_n = 0;
    rc_cfg_vld = 0;
    in_fixed_data_vld = 0;
    start = 0;

    rc_cfg.rms_rc_shift = RC_SHIFT;
    rc_cfg.rms_K = RMS_K;
    rc_cfg.rms_dequant_scale_square = shortreal_to_bitsbfloat16(rms_dequant_scale_square);

    in_fixed_data = 0;

    repeat(10) @(negedge clk);
    rst_n = 1;
    repeat(10) @(negedge clk);
    rc_cfg_vld = 1;
    start = 1;
    @(negedge clk);
    rc_cfg_vld = 0;
    start = 0;
    @(negedge clk);
    for(int i =0; i< DATA_NUM/`MAC_MULT_NUM;i++)begin
        for(int j = 0; j< `MAC_MULT_NUM;j++)begin
            in_fixed_data[j] = tb_indata_array[i*`MAC_MULT_NUM+j];
        end
        in_fixed_data_vld = 1;
        @(negedge clk);
    end
    in_fixed_data_vld = 0;
    repeat(100) @(negedge clk);
    $finish();
end
real ratio_error = 0;
always  begin
    @(negedge clk);
    if(rst_n != 0 && rc_scale_vld)begin
        $display("Hardware Value, one_over_krms: %f",rc_scale * 1.0 /($pow(2, RC_SHIFT)));
        ratio_error = $abs((rc_scale * 1.0 /($pow(2, RC_SHIFT)) - soft_one_over_krms) / soft_one_over_krms);
        $display("*** ratio_error: %f ***", ratio_error);
    end
end



krms 
#(
    .BUS_NUM(BUS_NUM), //input bus num
    .DATA_NUM_WIDTH($clog2(`MAX_EMBD_SIZE)+1), //layernorm vector length reg width
    .FIXED_SQUARE_SUM_WIDTH(FIXED_SQUARE_SUM_WIDTH)
)inst_krms(
    .clk(clk),
    .rst_n(rst_n),

    .start(start),
    .rc_cfg_vld(rc_cfg_vld),
    .rc_cfg(rc_cfg),     

    .in_fixed_data(in_fixed_data), //vector的长度一定是BUS NUM的倍数
    .in_fixed_data_vld(in_fixed_data_vld),
    
    .rc_scale(rc_scale),
    .rc_scale_vld(rc_scale_vld)
);
    
endmodule
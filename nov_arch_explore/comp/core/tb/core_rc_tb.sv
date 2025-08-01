module core_rc_tb ();
parameter IN_DATA_WIDTH = 24;
parameter OUT_DATA_WIDTH = 24;
parameter RECOMPUTE_FIFO_DEPTH = 4;
parameter RETIMING_REG_NUM = 4;

parameter RECOMPUTE_NEEDED = 1;
parameter RC_SCALE = 'h0abc;
parameter RC_SHIFT = 10;

parameter TEST_ITER = 100;

logic                                     clk;
logic                                     rst_n;
logic                                     recompute_needed;
logic [`RECOMPUTE_SCALE_WIDTH - 1:0]      rc_scale;
logic                                     rc_scale_vld;
logic                                     rc_scale_clear;
logic [`RECOMPUTE_SHIFT_WIDTH - 1:0]      rc_shift;
logic signed [IN_DATA_WIDTH - 1:0]        in_data;
logic                                     in_data_vld;
logic signed [OUT_DATA_WIDTH - 1:0]       out_data;
logic signed[OUT_DATA_WIDTH - 1:0]        out_data_soft;
logic signed[OUT_DATA_WIDTH - 1:0]        out_data_soft_delay;
logic                                     out_data_vld;
logic                                     error;


// int temp;
task rc_soft(input int recompute_needed,
             input int scale,
             input int shift,
             input signed [IN_DATA_WIDTH-1:0] in_data,
             output signed [OUT_DATA_WIDTH-1:0] out_data);
    longint temp;
    int round;
    temp =  in_data * scale;
    // $display(temp);
    round = temp[shift-1];
    temp = temp >>> shift;
    temp = temp + round;
    
    if(temp > $rtoi($pow(2, OUT_DATA_WIDTH-1) - 1)) //
        temp = $rtoi($pow(2, OUT_DATA_WIDTH-1) - 1);
    // $display(temp);
    if(temp < $rtoi(-$pow(2, OUT_DATA_WIDTH-1)))
        temp = $rtoi(-$pow(2, OUT_DATA_WIDTH-1));
    // repeat(5 + RETIMING_REG_NUM) @(negedge clk);
    if(recompute_needed == 0)
        out_data = in_data;
    else 
        out_data = temp[OUT_DATA_WIDTH-1:0];
endtask

logic signed [OUT_DATA_WIDTH - 1:0] out_data_soft_delay_array [5 + RETIMING_REG_NUM-1:0];//recompute data retiming array
assign out_data_soft_delay = out_data_soft_delay_array[5 + RETIMING_REG_NUM-1];
genvar i;
generate
    for(i = 0; i < 5 + RETIMING_REG_NUM; i++)begin
        if(i == 0)begin
            always_ff@(posedge clk or negedge rst_n)begin
                if(~rst_n)begin
                    out_data_soft_delay_array[i] <= 0;
                end
                else begin
                    out_data_soft_delay_array[i] <= out_data_soft;
                end
            end
        end
        else begin
            always_ff@(posedge clk or negedge rst_n)begin
                if(~rst_n)begin
                    out_data_soft_delay_array[i] <= 0;
                end
                else begin
                    out_data_soft_delay_array[i] <= out_data_soft_delay_array[i-1];
                end
            end
        end
    end
endgenerate

always  begin
    #50 clk = ~clk;
end

initial begin
    clk = 0;
    rst_n = 0;
    recompute_needed = RECOMPUTE_NEEDED;
    rc_scale = RC_SCALE;
    rc_shift = RC_SHIFT;

    rc_scale_vld = 0;
    rc_scale_clear = 0;

    in_data_vld = 0;
    in_data = 0;

    repeat(10) @(negedge clk);
    rst_n = 1;
    repeat(10) @(negedge clk);
    rc_scale_vld = 1;
    @(negedge clk);
    rc_scale_vld = 0;
    for(int i = 0; i < TEST_ITER; i++)begin
        in_data = $random;
        in_data_vld = 1;
        rc_soft(recompute_needed, RC_SCALE, RC_SHIFT, in_data, out_data_soft);
        @(negedge clk);
    end
    in_data_vld = 0;
    rc_scale_clear = 1;
    repeat(100) @(negedge clk);
    $finish();
end



core_rc #(
    .IN_DATA_WIDTH(IN_DATA_WIDTH),
    .OUT_DATA_WIDTH(OUT_DATA_WIDTH),
    .RECOMPUTE_FIFO_DEPTH(RECOMPUTE_FIFO_DEPTH),
    .RETIMING_REG_NUM(RETIMING_REG_NUM)
)   inst_core_rc
(
    .clk(clk),
    .rst_n(rst_n),

    .recompute_needed(recompute_needed),
 
    .rc_scale(rc_scale),
    .rc_scale_vld(rc_scale_vld),
    .rc_scale_clear(rc_scale_clear),

    .rc_shift(rc_shift),
 
    .in_data(in_data),
    .in_data_vld(in_data_vld),

    .out_data(out_data),
    .out_data_vld(out_data_vld),

    .error(error)
);
always @(negedge clk) begin
    if(rst_n!=0)
        assert (error != 1'b1) else $error("*****FIFO Full error*****");
    if(rst_n!=0 && out_data_vld)
        assert (out_data_soft_delay == out_data) else $error("*****Outdata error*****");
end


endmodule
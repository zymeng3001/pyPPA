module clk_gen_tb ();

logic           asyn_rst;
logic           ext_clk; //External Clock
logic           clk_gen_iin; //Ring Oscillator Current Bias
logic			clk_gen_en; // Clock GEN En
logic	[2-1:0]	clk_gen_div; // Clock GEN Div
logic           clk_gate_en; // Clock Gate
logic	        o_chip_clk;	//Main clock of chip
logic           o_clk_div; //Divided Clock Out

CLK_GEN DUT (.asyn_rst(asyn_rst),.ext_clk(ext_clk),.clk_gen_iin(clk_gen_iin),.clk_gen_en(clk_gen_en),.clk_gen_div(clk_gen_div),.clk_gate_en(clk_gate_en),.o_chip_clk(o_chip_clk),.o_clk_div(o_clk_div));

initial begin
    ext_clk = 0;
    clk_gen_iin = 0;
    clk_gen_en = 0;
    clk_gen_div = 0;
    clk_gate_en = 0;
    asyn_rst = 0;

    //rst
    #1000000
    asyn_rst = 1;

    //enable clock gen
    #1000000;
    clk_gen_en = 1'b1;

    //enable clock gate
    #1000000;
    clk_gate_en = 1'b1;

    //div2
    #1000000;
    clk_gen_div = 2'b1;

    //div4
    #1000000;
    clk_gen_div = 2'b10;

    //ext
    #1000000;
    clk_gen_div = 2'b11;

    //gate off
    #1000000;
    clk_gate_en = 0;

    #1000000;
    $finish;
    
end

always #3 ext_clk=~ext_clk;

endmodule
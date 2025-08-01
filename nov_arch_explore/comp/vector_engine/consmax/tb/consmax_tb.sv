
module consmax_block_tb;

  // Parameters
  parameter FIXED_BIT = 8;
  parameter EXP_BIT = 8;
  parameter MAT_BIT = 7;
  parameter LUT_DATA = EXP_BIT + MAT_BIT + 1;
  parameter LUT_ADDR = FIXED_BIT >> 1;
  parameter LUT_DEPTH = 2 ** LUT_ADDR;
  parameter integer DATA_NUM_WIDTH = 10;
  parameter integer SCALA_POS_WIDTH = EXP_BIT;

  parameter integer BATCH_SZ = 256;

  // Test cases
  logic [FIXED_BIT-1:0] input_data_pack [BATCH_SZ-1:0];

  // DUT Inputs
  logic clk;
  logic rst_n;
  logic [FIXED_BIT-1:0] in_fixed_data;
  logic in_fixed_data_vld;
  // logic signed [SCALA_POS_WIDTH-1:0] out_scale_pos;
  // logic out_scale_pos_vld;
  logic [LUT_DATA-1:0] out_scale;
  logic out_scale_vld;
  logic [LUT_ADDR:0] lut_waddr;
  logic lut_wen;
  logic [LUT_DATA-1:0] lut_wdata;

  // DUT Outputs
  logic [FIXED_BIT-1:0] out_fixed_data;
  logic out_fixed_data_vld;

  real beta = 300.0;
  real gamma = 1000000.0;

  // Instantiate the DUT
  consmax_block #(
    .FIXED_BIT(FIXED_BIT),
    .EXP_BIT(EXP_BIT),
    .MAT_BIT(MAT_BIT),
    .LUT_DATA(LUT_DATA),
    .LUT_ADDR(LUT_ADDR),
    .LUT_DEPTH(LUT_DEPTH),
    .DATA_NUM_WIDTH(DATA_NUM_WIDTH),
    .SCALA_POS_WIDTH(SCALA_POS_WIDTH)
  ) dut (
    .clk(clk),
    .rst_n(rst_n),
    .in_fixed_data(in_fixed_data),
    .in_fixed_data_vld(in_fixed_data_vld),
    .out_scale(out_scale),
    .out_scale_vld(out_scale_vld),
    .lut_waddr(lut_waddr),
    .lut_wen(lut_wen),
    .lut_wdata(lut_wdata),
    .out_fixed_data(out_fixed_data),
    .out_fixed_data_vld(out_fixed_data_vld)
  );

  // Clock Generation
  always #5 clk = ~clk;

  // Helper function
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

  // Reference Model
  logic [LUT_DATA-1:0] lut_ref [LUT_DEPTH*2-1:0];
  logic [FIXED_BIT-1:0] out_fixed_data_ref;
  logic [LUT_DATA-1:0] exp_rst_ref;
  logic lut_rvalid_ref;
  logic exp_rst_vld_ref;
  logic out_vld_ref;
  
  function [7:0] consmax_rst;
    input [7:0] x;
    int temp_rst;
    begin
        shortreal internal_product;
        internal_product = bitsbfloat16_to_shortreal(lut_ref[x[7:4]+16])*bitsbfloat16_to_shortreal(lut_ref[x[3:0]]);
        // $write("internal_p = %f", internal_product*$pow(2, $itor(out_scale_pos)));
        
        temp_rst = (internal_product*bitsbfloat16_to_shortreal(out_scale) < 2**31 -1) ? int'(internal_product*bitsbfloat16_to_shortreal(out_scale)) : 255;
        consmax_rst = (temp_rst > 255) ? 255 : temp_rst;
    end
   endfunction

   function shortreal internal_product;
    input [7:0] x;
    shortreal rst;
    begin
        shortreal lsb_entry;
        shortreal msb_entry;
        lsb_entry = bitsbfloat16_to_shortreal(lut_ref[x[3:0]]);
        msb_entry = bitsbfloat16_to_shortreal(lut_ref[x[7:4]+16]);
        //$write("\n");
        //$display("lsb: %0.7f, msb: %0.7f", lsb_entry, msb_entry);
        rst = lsb_entry*msb_entry;
	internal_product = rst;
    end
  endfunction

  task init_luts;
    shortreal lut_shortreal;
    for (int i = 0; i < 32; i++) begin
        if (i < 16) begin
            lut_shortreal = $exp(i);
        end else begin
            lut_shortreal = $exp(i*16-beta)/gamma;
        end
        lut_waddr = i;
        lut_wen = 1;
        lut_wdata = shortreal_to_bitsbfloat16(lut_shortreal);
        
        $display("Writing to addr %2d: (hex)%4h, (float)%0.7f", i,lut_wdata, lut_shortreal); 
        
        lut_ref[i] = lut_wdata;
        #10;
    end

    lut_wen = 0;
  endtask

  task print_luts;
    $display("LUT Data:");
    $display("%15s%15s","MSB","LSB");
    for (int i = 0; i < 16; i++) begin
        $write("%0.7f %0.7f  |   ",
            bitsbfloat16_to_shortreal(lut_ref[i+16]),bitsbfloat16_to_shortreal(lut_ref[i]));
//          $display("%h %h",lut_ref[i+16],lut_ref[i]);
//        print_bfloat16(lut_ref[i+16]);
//        print_bfloat16(lut_ref[i]);
        $write("\n");
    end
    $display("");
  endtask //print_luts

  task sequencer;
    for (int i = 0; i < BATCH_SZ; i++) begin
        in_fixed_data = input_data_pack[i];
        in_fixed_data_vld = 1;
        // $display("At time %0d, input data = %0d, ref_output = %d", $time, in_fixed_data, consmax_rst(in_fixed_data));
        #10;
    end
    in_fixed_data_vld = 0;
  endtask

  // randomize the input data
  task gen_rand_input;
    for (int i = 0; i < BATCH_SZ; i++) begin
        input_data_pack[i] = i;
    end
  endtask
  
  // scoreboard

  integer idx = 0;
  always @(negedge clk) begin
    if (out_fixed_data_vld) begin
      //$display("exp_rst_scaled: %4h, final output: %3d", dut.exp_rst_scaled_reg, out_fixed_data);
      //$display("%3d | %4h %4h | %4h | %4h | %3d", dut.in_fixed_data_reg, dut.lut_rdata_reg[0], dut.lut_rdata_reg[1], dut.exp_rst_reg, dut.exp_rst_scaled_reg, dut.out_fixed_data_reg);
      if (out_fixed_data != consmax_rst(input_data_pack[idx])) begin
        $display("!!!Mismatch at index %0d, DUT = %0d, Ref = %0d", idx, out_fixed_data, consmax_rst(input_data_pack[idx]));
      end else begin
        $display("Match at index %0d, DUT = %0d, Ref = %0d", idx, out_fixed_data, consmax_rst(input_data_pack[idx]));
      end
      idx = idx + 1;
    end
  end

  integer idx2 = 0;
  always @(negedge clk) begin
    if (dut.exp_rst_scaled_vld) begin
	
	//$display("exp_rst: %4h, exp_rst_scaled: %4h",dut.exp_rst_reg,dut.exp_rst_scaled_reg);
      if (dut.exp_rst_reg != shortreal_to_bitsbfloat16(internal_product(input_data_pack[idx]))) begin
        //$display("!!!Internal Mismatch at index %0d, DUT = %h, Ref = %h", idx2, dut.exp_rst_reg, shortreal_to_bitsbfloat16(internal_product(input_data_pack[idx2])));
      end else begin
        //$display("Internal Match at index %0d, DUT = %h, Ref = %h", idx2, dut.exp_rst_reg, shortreal_to_bitsbfloat16(internal_product(input_data_pack[idx2])));
      end
      idx2 = idx2 + 1;
    end
  end

  // monitor
  //always @(negedge clk) begin
      //if (in_fixed_data_vld)
      //$display("%3d | %4h %4h | %4h | %4h | %3d (%b %b %b %b %b)", dut.in_fixed_data_reg, dut.lut_rdata_reg[0], dut.lut_rdata_reg[1], dut.exp_rst_reg, dut.exp_rst_scaled_reg, dut.out_fixed_data_reg, dut.lut_ren, dut.lut_rvalid, dut.exp_rst_vld, dut.exp_rst_scaled_vld ,dut.out_fixed_data_vld);
  //end


  // Testbench Routine
  initial begin
    // Initialize Inputs
    clk = 0;
    rst_n = 0;
    in_fixed_data = 0;
    in_fixed_data_vld = 0;
    out_scale = 0;
    out_scale_vld = 0;
    lut_waddr = 0;
    lut_wen = 0;
    lut_wdata = 0;

    // Apply Reset
    #10;
    rst_n = 1;

    // Load LUT with some data
    init_luts();
    print_luts();

    // set the out scale position
    out_scale = 16'h3f00;
    out_scale_vld = 1;
$display("out_scale: %4h %4h", dut.out_scale_reg, out_scale);
    #10;
    $display("out_scale: %h %h", dut.out_scale_reg, out_scale);
    out_scale_vld = 0;
    
    // Generate Random Input Data
    gen_rand_input();
    
    // Run the Sequencer
    sequencer();

    // Finish Simulation
    #100;
    $finish;
  end

endmodule

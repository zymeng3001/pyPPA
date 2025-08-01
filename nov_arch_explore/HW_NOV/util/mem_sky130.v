module mem_sp_sky130 #(
    parameter   DATA_BIT = ${sram_width},
    parameter   DEPTH = ${sram_depth},
    parameter   ADDR_BIT = $clog2(DEPTH),
    parameter   BWE = 0 // bit write enable (currently unused)
)(
    input                       clk,
    input       [ADDR_BIT-1:0]  addr,
    input                       wen,
    input       [DATA_BIT-1:0]  bwe,   // only used if BWE == 1
    input       [DATA_BIT-1:0]  wdata,
    input                       ren,
    output  reg [DATA_BIT-1:0]  rdata
);

    // Constants
    localparam MACRO_WIDTH = 32;
    localparam MACRO_DEPTH = 128;
    localparam NUM_BANKS = DATA_BIT / MACRO_WIDTH;          
    localparam NUM_TILES = DEPTH / MACRO_DEPTH;             
    localparam TILE_ADDR_BITS = $clog2(MACRO_DEPTH);       
    localparam TILE_SEL_BITS = $clog2(NUM_TILES);  

    parameter ADDR_WIDTH = 8;   
    parameter DATA_WIDTH = 33;      

    // Internal signals
    wire [TILE_ADDR_BITS-1:0] local_addr;
    wire [TILE_SEL_BITS-1:0]  tile_sel;

    assign local_addr = addr[TILE_ADDR_BITS-1:0];
    assign tile_sel   = addr[ADDR_BIT-1:TILE_ADDR_BITS];

    genvar bank, tile;
    // wire [MACRO_WIDTH-1:0] rdata_macro [0:NUM_BANKS-1];
    
    generate
    for (bank = 0; bank < NUM_BANKS; bank = bank + 1) begin : bank_gen
        for (tile = 0; tile < NUM_TILES; tile = tile + 1) begin : tile_gen
            wire csb  = ~(tile_sel == tile);  // active low
            wire web  = ~wen;
            wire [ADDR_WIDTH-1:0] addr0 = local_addr;
            wire [DATA_WIDTH-1:0] din0  = wdata[bank*MACRO_WIDTH +: MACRO_WIDTH];
            wire [DATA_WIDTH-1:0] dout0_t;

            sky130_sram_0kbytes_1rw_32x128_32 sram_macro (
                .clk0(clk),
                .csb0(csb),
                .web0(web),
                // .spare_wen0(1'b1),  // enable spare bit write
                .addr0(addr0),
                .din0({1'b0, din0}),  // pad MSB (bit 32) as unused
                .dout0(dout0_t)
            );

            // Read output mux
            always @(posedge clk) begin
                if (ren && (tile_sel == tile))
                    rdata[bank*MACRO_WIDTH +: MACRO_WIDTH] <= dout0_t[MACRO_WIDTH-1:0];
            end
        end
    end
    endgenerate

endmodule


// OpenRAM SRAM model
// Words: 128
// Word size: 32

module sky130_sram_0kbytes_1rw_32x128_32(
// `ifdef USE_POWER_PINS
//     vccd1,
//     vssd1,
// `endif
// Port 0: RW
    // clk0,csb0,web0,spare_wen0,addr0,din0,dout0
    clk0,csb0,web0,addr0,din0,dout0
  );

  parameter DATA_WIDTH = 33 ;
  parameter ADDR_WIDTH = 8 ;
  parameter RAM_DEPTH = 1 << ADDR_WIDTH;
  // FIXME: This delay is arbitrary.
  parameter DELAY = 3 ;
  parameter VERBOSE = 1 ; //Set to 0 to only display warnings
  parameter T_HOLD = 1 ; //Delay to hold dout value after posedge. Value is arbitrary

`ifdef USE_POWER_PINS
    inout vccd1;
    inout vssd1;
`endif
  input  clk0; // clock
  input   csb0; // active low chip select
  input  web0; // active low write control
  input [ADDR_WIDTH-1:0]  addr0;
  // input           spare_wen0; // spare mask
  input [DATA_WIDTH-1:0]  din0;
  output [DATA_WIDTH-1:0] dout0;

  reg [DATA_WIDTH-1:0]    mem [0:RAM_DEPTH-1];

  reg  csb0_reg;
  reg  web0_reg;
  // reg spare_wen0_reg;
  reg [ADDR_WIDTH-1:0]  addr0_reg;
  reg [DATA_WIDTH-1:0]  din0_reg;
  reg [DATA_WIDTH-1:0]  dout0;

  // All inputs are registers
  always @(posedge clk0)
  begin
    csb0_reg = csb0;
    web0_reg = web0;
    // spare_wen0_reg = spare_wen0;
    addr0_reg = addr0;
    din0_reg = din0;
    // #(T_HOLD) dout0 = 32'bx;
    // if ( !csb0_reg && web0_reg && VERBOSE )
    //   $display($time," Reading %m addr0=%b dout0=%b",addr0_reg,mem[addr0_reg]);
    // if ( !csb0_reg && !web0_reg && VERBOSE )
    //   $display($time," Writing %m addr0=%b din0=%b",addr0_reg,din0_reg);
  end


  // Memory Write Block Port 0
  // Write Operation : When web0 = 0, csb0 = 0
  always @ (negedge clk0)
  begin : MEM_WRITE0
    if ( !csb0_reg && !web0_reg ) begin
        mem[addr0_reg][30:0] = din0_reg[30:0];
        // if (spare_wen0_reg)
                // mem[addr0_reg][32] = din0_reg[32];
    end
  end

  // Memory Read Block Port 0
  // Read Operation : When web0 = 1, csb0 = 0
  always @ (negedge clk0)
  begin : MEM_READ0
    if (!csb0_reg && web0_reg)
    //    dout0 <= #(DELAY) mem[addr0_reg];
       dout0 <= mem[addr0_reg];
  end

endmodule

module true_mem_wrapper#(
    parameter   DATA_BIT =         (`MAC_MULT_NUM * `IDATA_WIDTH),
    parameter   WMEM_DEPTH =       `WMEM_DEPTH,
    parameter   WMEM_ADDR_WIDTH  =  $clog2(WMEM_DEPTH)
)(
    input  logic                                                clk,
    input  logic                                                rst_n,

    input  logic [WMEM_ADDR_WIDTH-1:0]                          wmem_addr,
    input  logic                                                wmem_ren,
    input  logic                                                wmem_wen,
    input  logic [DATA_BIT-1:0]                                 wmem_wdata,
    input  logic [DATA_BIT-1:0]                                 wmem_bwe,

    output logic [DATA_BIT-1:0]                                 wmem_rdata,

    //power ctrl
    input  logic wmem_1024_ffn_deepslp,
    input  logic wmem_1024_ffn_bc1,
    input  logic wmem_1024_ffn_bc2,

    input  logic wmem_512_attn_deepslp,
    input  logic wmem_512_attn_bc1,
    input  logic wmem_512_attn_bc2
);
logic wmem_1024_ffn_ren;
logic wmem_1024_ffn_ren_delay1;
logic wmem_1024_ffn_wen;

logic [9:0]   wmem_1024_ffn_adr;
logic [127:0] wmem_1024_ffn_din;
logic [127:0] wmem_1024_ffn_wbeb;
logic [127:0] wmem_1024_ffn_q;

logic wmem_512_attn_ren;
logic wmem_512_attn_ren_delay1;
logic wmem_512_attn_wen;

logic [8:0]   wmem_512_attn_adr;
logic [127:0] wmem_512_attn_din;
logic [127:0] wmem_512_attn_wbeb;
logic [127:0] wmem_512_attn_q;

always_ff @(posedge clk or negedge rst_n)begin
    if(~rst_n) begin
        wmem_1024_ffn_ren_delay1 <= 0;
        wmem_512_attn_ren_delay1 <= 0;
    end
    else begin
        wmem_1024_ffn_ren_delay1 <= wmem_1024_ffn_ren;
        wmem_512_attn_ren_delay1 <= wmem_512_attn_ren;
    end
end

always_comb begin
    if(wmem_1024_ffn_ren_delay1)
        wmem_rdata = wmem_1024_ffn_q;
    else
        wmem_rdata = wmem_512_attn_q;
end


always_comb begin
    wmem_512_attn_ren = wmem_ren && (wmem_addr < 512);
    wmem_512_attn_wen = wmem_wen && (wmem_addr < 512);
    wmem_512_attn_adr = wmem_addr[8:0];
    wmem_512_attn_din = wmem_wdata;
    wmem_512_attn_wbeb = ~wmem_bwe;
end

always_comb begin
    wmem_1024_ffn_ren = wmem_ren && (wmem_addr >= 512);
    wmem_1024_ffn_wen = wmem_wen && (wmem_addr >= 512);
    wmem_1024_ffn_adr = wmem_addr - 512;
    wmem_1024_ffn_din = wmem_wdata;
    wmem_1024_ffn_wbeb = ~wmem_bwe;
end


wmem_512 wmem_512_attn_r(//lintra s-68000
   .clk(clk),             		//Input Clock
   .ren(wmem_512_attn_ren),                    	//Read Enable
   .wen(wmem_512_attn_wen),                    	//Write Enable
   .adr(wmem_512_attn_adr),         //Input Address
   .mc(3'b0),     		//Controls extending write duration
   .mcen(1'b0),     			//Enable read margin control 

   .clkbyp(1'b0),                   	//clock bypass enable  
   .din(wmem_512_attn_din[63:0]),       //Input Write Data 
   .wbeb(wmem_512_attn_wbeb[63:0]),  //Write Bit enable

   .wa(2'b0),
   .wpulse(2'b0),
   .wpulseen(1'b1),
   .fwen(~rst_n),

 
   .deepslp(wmem_512_attn_deepslp),
   .shutoff(1'b0),
   .sleep(1'b0),             		//Light Sleep Enable
   .bc1(wmem_512_attn_bc1),                    	//Voltage control
   .bc2(wmem_512_attn_bc2),    
   .mpr(),


   .q(wmem_512_attn_q[63:0])
);

wmem_512 wmem_512_attn_l(//lintra s-68000
   .clk(clk),             		//Input Clock
   .ren(wmem_512_attn_ren),                    	//Read Enable
   .wen(wmem_512_attn_wen),                    	//Write Enable
   .adr(wmem_512_attn_adr),         //Input Address
   .mc(3'b0),     		//Controls extending write duration
   .mcen(1'b0),     			//Enable read margin control 

   .clkbyp(1'b0),                   	//clock bypass enable  
   .din(wmem_512_attn_din[127:64]),       //Input Write Data 
   .wbeb(wmem_512_attn_wbeb[127:64]),  //Write Bit enable

   .wa(2'b0),
   .wpulse(2'b0),
   .wpulseen(1'b1),
   .fwen(~rst_n),

 
   .deepslp(wmem_512_attn_deepslp),
   .shutoff(1'b0),
   .sleep(1'b0),             		//Light Sleep Enable
   .bc1(wmem_512_attn_bc1),                    	//Voltage control
   .bc2(wmem_512_attn_bc2),    
   .mpr(),


   .q(wmem_512_attn_q[127:64])
);



wmem_1024 wmem_1024_ffn_r(//lintra s-68000
   .clk(clk),             		//Input Clock
   .ren(wmem_1024_ffn_ren),                    	//Read Enable
   .wen(wmem_1024_ffn_wen),                    	//Write Enable
   .adr(wmem_1024_ffn_adr),         //Input Address
   .mc(3'b0),     		//Controls extending write duration
   .mcen(1'b0),     			//Enable read margin control 

   .clkbyp(1'b0),                   	//clock bypass enable  
   .din(wmem_1024_ffn_din[63:0]),       //Input Write Data 
   .wbeb(wmem_1024_ffn_wbeb[63:0]),  //Write Bit enable

   .wa(2'b0),
   .wpulse(2'b0),
   .wpulseen(1'b1),
   .fwen(~rst_n),

 
   .deepslp(wmem_1024_ffn_deepslp),
   .shutoff(1'b0),
   .sleep(1'b0),             		//Light Sleep Enable
   .bc1(wmem_1024_ffn_bc1),                    	//Voltage control
   .bc2(wmem_1024_ffn_bc2),    
   .mpr(),


   .q(wmem_1024_ffn_q[63:0])
);

wmem_1024 wmem_1024_ffn_l(//lintra s-68000
   .clk(clk),             		//Input Clock
   .ren(wmem_1024_ffn_ren),                    	//Read Enable
   .wen(wmem_1024_ffn_wen),                    	//Write Enable
   .adr(wmem_1024_ffn_adr),         //Input Address
   .mc(3'b0),     		//Controls extending write duration
   .mcen(1'b0),     			//Enable read margin control 

   .clkbyp(1'b0),                   	//clock bypass enable  
   .din(wmem_1024_ffn_din[127:64]),       //Input Write Data 
   .wbeb(wmem_1024_ffn_wbeb[127:64]),  //Write Bit enable

   .wa(2'b0),
   .wpulse(2'b0),
   .wpulseen(1'b1),
   .fwen(~rst_n),

 
   .deepslp(wmem_1024_ffn_deepslp),
   .shutoff(1'b0),
   .sleep(1'b0),             		//Light Sleep Enable
   .bc1(wmem_1024_ffn_bc1),                    	//Voltage control
   .bc2(wmem_1024_ffn_bc2),    
   .mpr(),


   .q(wmem_1024_ffn_q[127:64])
);
    
endmodule
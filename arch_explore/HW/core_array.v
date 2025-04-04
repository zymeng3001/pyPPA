module core_array #(
	parameter HNUM = ${n_heads},
    parameter VNUM = ${n_cols},

    parameter GBUS_DATA = ${gbus_width},
    parameter GBUS_ADDR = 12,

    parameter LBUF_DEPTH = 64,
    parameter LBUF_DATA =  64,
    parameter LBUF_ADDR   = $clog2(LBUF_DEPTH),

    parameter CDATA_BIT = 8,

    parameter ODATA_BIT = 16,
    parameter IDATA_BIT = 8,
    parameter MAC_NUM   = ${gbus_width/8},

    parameter   WMEM_DEPTH  = 512,
    parameter   CACHE_DEPTH = ${max_context_length}
)(
    // Global Signals
    input                       clk,
    input                       rstn,
  	input     [ODATA_BIT-1:0] cfg_quant_scale,
	input     [ODATA_BIT-1:0] cfg_quant_bias,
	input     [ODATA_BIT-1:0] cfg_quant_shift,
	input     [CDATA_BIT-1:0] cfg_consmax_shift, 
	
	
	input    [HNUM-1:0][CDATA_BIT-1:0]     cfg_acc_num,
	input            [HNUM-1:0][GBUS_ADDR-1:0]     in_GBUS_ADDR,
    input            [HNUM * VNUM - 1:0]                           gbus_wen,
    input            [HNUM-1:0][GBUS_DATA-1:0]     gbus_wdata,   
    input            [HNUM * VNUM - 1:0]                           gbus_ren,
    output   reg   [HNUM-1:0][GBUS_DATA-1:0]     gbus_rdata,     
    output   reg   [HNUM * VNUM - 1:0]        gbus_rvalid,
    // Channel - Core-to-Core Link
    // Vertical for Weight and Key/Value Propagation
    input                                           vlink_enable,
    input            [VNUM-1:0][GBUS_DATA-1:0]     vlink_wdata,
    input            [VNUM-1:0]                    vlink_wen,
    output   reg   [VNUM-1:0][GBUS_DATA-1:0]     vlink_rdata,
    output   reg   [VNUM-1:0]                    vlink_rvalid,
    // Horizontal for Activation Propagation
    input            [HNUM-1:0][GBUS_DATA-1:0]     hlink_wdata,    //hlink_wdata go through reg, to hlink_rdata
    input            [HNUM-1:0]                    hlink_wen,     
    output   reg   [HNUM-1:0][GBUS_DATA-1:0]     hlink_rdata,
    output   reg   [HNUM-1:0]                    hlink_rvalid,
    // Channel - MAC Operation
    // Core Memory Access for Weight and KV Cache
    //input            CMEM_ARR_PACKET                arr_cmem,
	input 	[HNUM-1:0][VNUM-1:0][GBUS_ADDR-1:0]  cmem_waddr,      // Write Value to KV Cache, G_BUS -> KV Cache, debug.
    input 	[HNUM-1:0][VNUM-1:0]    cmem_wen,
    input	[HNUM-1:0][VNUM-1:0][GBUS_ADDR-1:0]  cmem_raddr,
    input	[HNUM-1:0][VNUM-1:0]    cmem_ren,
	
    // Local Buffer Access for Weight and KV Cache
    output           [HNUM-1:0][VNUM-1:0]        lbuf_empty,
    output           [HNUM-1:0][VNUM-1:0]        lbuf_reuse_empty,
    input            [HNUM-1:0][VNUM-1:0]        lbuf_reuse_ren, //reuse pointer logic, when enable
    input            [HNUM-1:0][VNUM-1:0]        lbuf_reuse_rst,  //reuse reset logic, when first round of reset is finished, reset reuse pointer to current normal read pointer value
    output           [HNUM-1:0][VNUM-1:0]        lbuf_full,
    output           [HNUM-1:0][VNUM-1:0]        lbuf_almost_full,
    input            [HNUM-1:0][VNUM-1:0]        lbuf_ren,
    // Local Buffer access for Activation
    output           [HNUM-1:0][VNUM-1:0]        abuf_empty,
    output           [HNUM-1:0][VNUM-1:0]        abuf_reuse_empty,
    input            [HNUM-1:0][VNUM-1:0]        abuf_reuse_ren, //reuse pointer logic, when enable
    input            [HNUM-1:0][VNUM-1:0]        abuf_reuse_rst,  //reuse reset logic, when first round of reset is finished, reset reuse pointer to current normal read pointer value
    output           [HNUM-1:0][VNUM-1:0]        abuf_full,
    output           [HNUM-1:0][VNUM-1:0]        abuf_almost_full,
    input            [HNUM-1:0][VNUM-1:0]        abuf_ren
);

    //from spi
    //CFG_ARR_PACKET arr_cfg_reg;
    reg       [ODATA_BIT-1:0] reg_cfg_quant_scale;
    reg       [ODATA_BIT-1:0] reg_cfg_quant_bias;
    reg       [ODATA_BIT-1:0] reg_cfg_quant_shift;
    reg       [CDATA_BIT-1:0] reg_cfg_consmax_shift;

    always @(posedge clk or negedge rstn) begin
    if(!rstn) begin
        reg_cfg_quant_scale<={ODATA_BIT{1'b0}};
        reg_cfg_quant_bias<={ODATA_BIT{1'b0}};
        reg_cfg_quant_shift<={ODATA_BIT{1'b0}};
        reg_cfg_consmax_shift<={ODATA_BIT{1'b0}};
    end
    else begin
        reg_cfg_quant_scale<=cfg_quant_scale;
        reg_cfg_quant_bias<=cfg_quant_bias;
        reg_cfg_quant_shift<=cfg_quant_shift;
        reg_cfg_consmax_shift<=cfg_consmax_shift;
    end
    end

    reg [CDATA_BIT-1:0] array_cfg_acc_num_reg_flat [0:(HNUM*VNUM)-1];
    integer k;

    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            for (k = 0; k < HNUM * VNUM; k = k + 1) begin
                array_cfg_acc_num_reg_flat[k] <= 0;
            end
        end else begin
            for (k = 0; k < HNUM * VNUM; k = k + 1) begin
                if (k % VNUM == 0) begin
                    array_cfg_acc_num_reg_flat[k] <= cfg_acc_num[k / VNUM];
                end else begin
                    array_cfg_acc_num_reg_flat[k] <= array_cfg_acc_num_reg_flat[k - 1];
                end
            end
        end
    end

    reg [GBUS_DATA-1:0] vlink_wdata_temp [0:(HNUM * VNUM) - 1];
    reg [GBUS_DATA-1:0] vlink_rdata_temp [0:(HNUM * VNUM) - 1];
    reg vlink_wen_temp [0:(HNUM * VNUM) - 1];
    reg vlink_rvalid_temp [0:(HNUM * VNUM) - 1];

    reg [GBUS_DATA-1:0] hlink_wdata_temp [0:(HNUM * VNUM) - 1];
    reg [GBUS_DATA-1:0] hlink_rdata_temp [0:(HNUM * VNUM) - 1];
    reg hlink_wen_temp [0:(HNUM * VNUM) - 1];
    reg hlink_rvalid_temp [0:(HNUM * VNUM) - 1];

    always @(*) begin
        integer i, j;

        // Compute
        for (i = 0; i < HNUM; i = i + 1) begin
            for (j = 0; j < VNUM; j = j + 1) begin
                // vlink connections
                if (i == 0) begin
                    vlink_wdata_temp[i * VNUM + j] = vlink_wdata[j];
                    vlink_wen_temp[i * VNUM + j] = vlink_wen[j];
                end else begin
                    vlink_wdata_temp[i * VNUM + j] = vlink_rdata_temp[(i-1) * VNUM + j];
                    vlink_wen_temp[i * VNUM + j] = vlink_rvalid_temp[(i-1) * VNUM + j];
                end

                // hlink connections
                if (j == 0) begin
                    hlink_wdata_temp[i * VNUM + j] = hlink_wdata[i];
                    hlink_wen_temp[i * VNUM + j] = hlink_wen[i];
                end else begin
                    hlink_wdata_temp[i * VNUM + j] = hlink_rdata_temp[i * VNUM + (j-1)];
                    hlink_wen_temp[i * VNUM + j] = hlink_rvalid_temp[i * VNUM + (j-1)];
                end

                // Update outputs for boundary conditions
                if (i == HNUM - 1) begin
                    vlink_rdata[j] = vlink_rdata_temp[i * VNUM + j];
                    vlink_rvalid[j] = vlink_rvalid_temp[i * VNUM + j];
                end else begin
                    vlink_rvalid[j] = 1'b0;
                    vlink_rdata[j] = 1'b0;
                end

                if (j == VNUM - 1) begin
                    hlink_rdata[i] = hlink_rdata_temp[i * VNUM + j];
                    hlink_rvalid[i] = hlink_rvalid_temp[i * VNUM + j];
                end else begin
                    hlink_rdata[i] = 1'b0;
                end
            end
        end
    end

    reg [GBUS_ADDR-1:0] GBUS_ADDR_temp_flat [0:HNUM-1][0:VNUM-1];
    reg [GBUS_DATA-1:0] gbus_wdata_temp_flat [0:HNUM-1][0:VNUM-1];
    reg [GBUS_DATA-1:0] gbus_rdata_temp_flat [0:HNUM-1][0:VNUM-1];


    always @(*) begin
        integer i, j;

        // Populate gbus arrays based on conditions
        for (i = 0; i < HNUM; i = i + 1) begin
            for (j = 0; j < VNUM; j = j + 1) begin
                // gbus_wdata_temp: Write data logic
                if (gbus_wen[i * VNUM + j]) begin
                    gbus_wdata_temp_flat[i][j] = gbus_wdata[i];
                end

                // GBUS_ADDR_temp: Address logic
                if (gbus_wen[i * VNUM + j] | gbus_ren[i * VNUM + j]) begin
                    GBUS_ADDR_temp_flat[i][j] = in_GBUS_ADDR[i];
                end
            end
        end

        for (i = 0; i < HNUM; i = i + 1) begin
            gbus_rdata[i] = 0; // Initialize to 0
            for (j = 0; j < VNUM; j = j + 1) begin
                if (gbus_rvalid[i * VNUM + j]) begin
                    gbus_rdata[i] = gbus_rdata_temp_flat[i][j];
                end 
            end
        end
    end

      
    generate
      for (genvar i = 0; i < HNUM; i = i+1) begin : gen_row
        for (genvar j = 0; j < VNUM; j = j+1) begin : gen_col
            	core_top core_top_instance (
                        .clk           (clk),
                        .rstn          (rstn),
                        .cfg_acc_num   (array_cfg_acc_num_reg_flat[i*VNUM+j]),
                        .cfg_quant_scale (cfg_quant_scale),
                        .cfg_quant_bias (cfg_quant_bias),
                        .cfg_quant_shift (cfg_quant_shift),
                        .gbus_addr     (GBUS_ADDR_temp_flat[i][j]), //gbus to weight mem, gbus to kv cache
                        .gbus_wen      (gbus_wen[i * VNUM + j]),
                        .gbus_wdata    (gbus_wdata_temp_flat[i][j]),
                        .gbus_ren      (gbus_ren[i * VNUM + j]),
                        .gbus_rdata    (gbus_rdata_temp_flat[i][j]),
                        .gbus_rvalid   (gbus_rvalid[i * VNUM + j]),
                        .vlink_enable  (vlink_enable),
                        .vlink_wdata   (vlink_wdata_temp[i * VNUM + j]), // access lbuf
                        .vlink_wen     (vlink_wen_temp[i * VNUM + j]),
                        .vlink_rdata   (vlink_rdata_temp[i * VNUM + j]),
                        .vlink_rvalid  (vlink_rvalid_temp[i * VNUM + j]),
                        .hlink_wdata   (hlink_wdata_temp[i * VNUM + j]), // access abuf
                        .hlink_wen     (hlink_wen_temp[i * VNUM + j]),
                        .hlink_rdata   (hlink_rdata_temp[i * VNUM + j]),
                        .hlink_rvalid  (hlink_rvalid_temp[i * VNUM + j]),
                        .cmem_waddr    (cmem_waddr[i][j]),
                        .cmem_wen      (cmem_wen[i][j]), //cmem_wen control, when high, mac output will send to cmem, if gbus_ren is not high previous cycle, mac output will send to gbus too.
                        .cmem_raddr    (cmem_raddr[i][j]),
                        .cmem_ren      (cmem_ren[i][j]),
                        .lbuf_full     (lbuf_full[i][j]),
                        .lbuf_almost_full(lbuf_almost_full[i][j]),
                        .lbuf_empty    (lbuf_empty[i][j]),
                        .lbuf_reuse_empty(lbuf_reuse_empty[i][j]),
                        .lbuf_ren      (lbuf_ren[i][j]),
                        .lbuf_reuse_ren(lbuf_reuse_ren[i][j]),
                        .lbuf_reuse_rst(lbuf_reuse_rst[i][j]),
                        .abuf_full     (abuf_full[i][j]),
                        .abuf_almost_full(abuf_almost_full[i][j]),
                        .abuf_empty    (abuf_empty[i][j]),
                        .abuf_reuse_empty(abuf_reuse_empty[i][j]),
                        .abuf_reuse_ren(abuf_reuse_ren[i][j]),
                        .abuf_reuse_rst(abuf_reuse_rst[i][j]),
                        .abuf_ren      (abuf_ren[i][j])
                    );
        	end
    	end
    endgenerate

endmodule

module ringosc_xcp_top_v3 (
   	input             IIN,
    output logic      CKON,
    output logic      CKOP,
    input             EXTCKIN,
    input             EXTCKIP,
    input             CKSEL,
   	output logic      CKFB_DRV
);

	// Emulation of the configurable Ring Osc
	logic	CLK_RO_RAW;

   	initial begin
    	CLK_RO_RAW = 1'b0;
        CKFB_DRV = 1'b0;
   	end

   	always begin
        #(`CLOCK_PERIOD/2)	CLK_RO_RAW = ~CLK_RO_RAW; // 500MHz
   	end

   	always begin
        #(`CLOCK_PERIOD*50) CKFB_DRV = ~CKFB_DRV; //5MHz
   	end

	// Clock AND gate used to emulate EN
 	assign CKON = CKSEL && CLK_RO_RAW;

endmodule // CLK_GEN_CORE
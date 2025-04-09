module fp_mult_pipe #(
    parameter sig_width       = 23,
    parameter exp_width       = 8,
    parameter ieee_compliance = 0,  // For NaN and denormals handling
    parameter stages          = 5
)(
    input                             clk,
    input                             rst_n,
    input  [sig_width+exp_width:0]    a,
    input  [sig_width+exp_width:0]    b,
    input                             ab_valid,
    output [sig_width+exp_width:0]    z,
    output                            z_valid
);

    // Unused parameters (for compatibility)
    parameter id_width     = 1;
    parameter op_iso_mode  = 0;
    parameter no_pm        = 1;
    parameter rst_mode     = 0; // Asynchronous reset on rst_n
    parameter en_ubr_flag  = 0;

    // Intermediate wire for the multiplication result from DW_fp_mult.
    wire [sig_width+exp_width:0] z_w;
    
    // Instance of the underlying multiplier.
    DW_fp_mult #(
        .sig_width       (sig_width),
        .exp_width       (exp_width),
        .ieee_compliance (ieee_compliance),
        .en_ubr_flag     (en_ubr_flag)
    )
    U1 (
        .a    (a),
        .b    (b),
        .rnd  (3'b00),
        .z    (z_w),
        .status()
    );

    //--------------------------------------------------------------------------
    // Pipeline Registers for Retiming
    //--------------------------------------------------------------------------
    // Declare two arrays for pipelining the result and its valid flag.
    reg [sig_width+exp_width:0] pipe_reg [0:stages-1];
    reg                         pipe_valid [0:stages-1];
    integer i;

    // Stage 0: Latch the output from DW_fp_mult.
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pipe_reg[0]    <= {(sig_width+exp_width+1){1'b0}};
            pipe_valid[0]  <= 1'b0;
        end else begin
            pipe_reg[0]    <= z_w;
            pipe_valid[0]  <= ab_valid;
        end
    end

    // Pipeline the result through the remaining stages.
    genvar j;
    generate
        for (j = 1; j < stages; j = j + 1) begin : pipe_stage
            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    pipe_reg[j]    <= {(sig_width+exp_width+1){1'b0}};
                    pipe_valid[j]  <= 1'b0;
                end else begin
                    pipe_reg[j]    <= pipe_reg[j-1];
                    pipe_valid[j]  <= pipe_valid[j-1];
                end
            end
        end
    endgenerate

    //--------------------------------------------------------------------------
    // Output Assignment
    //--------------------------------------------------------------------------
    assign z      = pipe_reg[stages-1];
    assign z_valid = pipe_valid[stages-1];

endmodule

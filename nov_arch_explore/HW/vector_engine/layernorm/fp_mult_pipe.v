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
    custom_fp_mult #(
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

module custom_fp_mult #(
    parameter integer sig_width       = 23,  // number of significand bits (mantissa)
    parameter integer exp_width       = 8,   // number of exponent bits
    parameter integer ieee_compliance = 0,   // 0 = ignore NaN/Inf
    parameter integer en_ubr_flag     = 0    // unused in this version
)(
    input  wire [sig_width + exp_width:0] a, // input A
    input  wire [sig_width + exp_width:0] b, // input B
    input  wire [2:0] rnd,                   // rounding mode (ignored)
    output reg  [sig_width + exp_width:0] z, // output result
    output reg  [7:0] status                 // status flags
);

    // Unpack inputs
    wire sign_a = a[sig_width + exp_width];
    wire sign_b = b[sig_width + exp_width];
    wire [exp_width-1:0] exp_a = a[sig_width + exp_width - 1 : sig_width];
    wire [exp_width-1:0] exp_b = b[sig_width + exp_width - 1 : sig_width];
    wire [sig_width-1:0] frac_a = a[sig_width-1:0];
    wire [sig_width-1:0] frac_b = b[sig_width-1:0];

    // Compute result sign
    wire sign_z = sign_a ^ sign_b;

    // Add implicit leading 1 for normalized numbers
    wire [sig_width:0] mant_a = (|exp_a) ? {1'b1, frac_a} : {1'b0, frac_a};
    wire [sig_width:0] mant_b = (|exp_b) ? {1'b1, frac_b} : {1'b0, frac_b};

    // Multiply mantissas (full precision)
    wire [2*sig_width+1:0] mant_prod = mant_a * mant_b;

    // Add exponents and subtract bias
    wire signed [exp_width:0] exp_sum = $signed({1'b0, exp_a}) + $signed({1'b0, exp_b}) - ((1 << (exp_width-1)) - 1);

    // Normalize mantissa (check if MSB is at [2*sig_width+1] or [2*sig_width])
    reg [sig_width-1:0] frac_z;
    reg [exp_width-1:0] exp_z;
    always @(*) begin
        if (mant_prod[2*sig_width+1]) begin
            frac_z = mant_prod[2*sig_width:2*sig_width - sig_width + 1];  // drop MSB
            exp_z = exp_sum + 1;
        end else begin
            frac_z = mant_prod[2*sig_width - 1:2*sig_width - sig_width];  // no shift
            exp_z = exp_sum;
        end
    end

    // Compose final result
    always @(*) begin
        // special cases
        if ((a == 0) || (b == 0)) begin
            z = {1'b0, {exp_width{1'b0}}, {sig_width{1'b0}}};  // zero
            status = 8'h01;
        end else begin
            z = {sign_z, exp_z, frac_z};
            status = 8'h00;
        end
    end

endmodule


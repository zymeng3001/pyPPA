
module fp_div_pipe #(
    parameter sig_width       = 23,
    parameter exp_width       = 8,
    parameter ieee_compliance = 0,  // NaN and denormals flag
    parameter stages          = 5
)(
    input                              clk,
    input                              rst_n,
    input  [sig_width+exp_width:0]     a,
    input  [sig_width+exp_width:0]     b,
    input                              ab_valid,
    output [sig_width+exp_width:0]     z,
    output                             z_valid
);

    // Local parameters (unused parameters included for interface compatibility)
    localparam id_width       = 1;
    localparam faithful_round = 0;
    localparam op_iso_mode    = 0;
    localparam no_pm          = 1;
    localparam rst_mode       = 0;  // Asynchronous reset mode
    localparam en_ubr_flag    = 0;

    // Intermediate signal from the floating-point divider
    wire [sig_width+exp_width:0] z_w;
    
    // Instantiate the divider module.
    custom_fp_div #(
        .sig_width       (sig_width),
        .exp_width       (exp_width),
        .ieee_compliance (ieee_compliance),
        .faithful_round  (faithful_round),
        .en_ubr_flag     (en_ubr_flag)
    )
    U1 (
        .a    (a),
        .b    (b),
        .rnd  (3'b00),
        .z    (z_w),
        .status()
    );

    //-------------------------------------------------------------------------
    // Pipeline registers for retiming the divider output.
    // We implement the pipeline using two arrays:
    //   - pipeline[] holds the floating-point data.
    //   - pipeline_vld[] holds the valid flag for each stage.
    //-------------------------------------------------------------------------
    reg [sig_width+exp_width:0] pipeline [0:stages-1];
    reg                         pipeline_vld [0:stages-1];
    
    // Stage 0: latch the output from DW_fp_div.
    integer i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pipeline[0]    <= {(sig_width+exp_width+1){1'b0}};
            pipeline_vld[0] <= 1'b0;
        end else begin
            pipeline[0]    <= z_w;
            pipeline_vld[0] <= ab_valid;
        end
    end

    // For stages 1 to stages-1, pass the data along the pipeline.
    genvar j;
    generate
        for (j = 1; j < stages; j = j + 1) begin : pipe_stage
            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    pipeline[j]    <= {(sig_width+exp_width+1){1'b0}};
                    pipeline_vld[j] <= 1'b0;
                end else begin
                    pipeline[j]    <= pipeline[j-1];
                    pipeline_vld[j] <= pipeline_vld[j-1];
                end
            end
        end
    endgenerate

    //-------------------------------------------------------------------------
    // Output assignment.
    // The final stage’s data and valid signal drive the module outputs.
    //-------------------------------------------------------------------------
    assign z      = pipeline[stages-1];
    assign z_valid = pipeline_vld[stages-1];

endmodule

module custom_fp_div #(
    parameter integer sig_width       = 23,  // significand (mantissa) bits (23 for FP32)
    parameter integer exp_width       = 8,   // exponent bits (8 for FP32)
    parameter integer ieee_compliance = 0,   // 1 = handle NaN, Inf, denormals
    parameter integer faithful_round  = 0,   // ignored here (for simplicity)
    parameter integer en_ubr_flag     = 0    // unused: underflow-by-rounding
)(
    input  wire [sig_width+exp_width:0] a,  // floating-point input A
    input  wire [sig_width+exp_width:0] b,  // floating-point input B
    input  wire [2:0]                   rnd, // rounding mode (ignored here)
    output reg  [sig_width+exp_width:0] z,  // result
    output reg  [7:0]                   status // status flags (simplified)
);

    // Decompose inputs
    wire sign_a = a[sig_width+exp_width];
    wire sign_b = b[sig_width+exp_width];
    wire [exp_width-1:0] exp_a = a[sig_width+exp_width-1:sig_width];
    wire [exp_width-1:0] exp_b = b[sig_width+exp_width-1:sig_width];
    wire [sig_width-1:0] frac_a = a[sig_width-1:0];
    wire [sig_width-1:0] frac_b = b[sig_width-1:0];

    // Sign of result
    wire sign_z = sign_a ^ sign_b;

    // Add hidden 1 to normalized mantissa
    wire [sig_width:0] norm_frac_a = (|exp_a) ? {1'b1, frac_a} : {1'b0, frac_a};
    wire [sig_width:0] norm_frac_b = (|exp_b) ? {1'b1, frac_b} : {1'b0, frac_b};

    // Result exponent (biased)
    wire signed [exp_width:0] exp_z = $signed({1'b0, exp_a}) - $signed({1'b0, exp_b}) + (1 << (exp_width-1)) - 1;

    // Mantissa division (simplified version)
    reg [2*sig_width+1:0] frac_div;
    always @(*) begin
        if (norm_frac_b != 0)
            frac_div = (norm_frac_a << sig_width) / norm_frac_b;
        else
            frac_div = 0;
    end

    // Normalize result (assume top sig_width+1 bits are the valid mantissa)
    wire [sig_width-1:0] frac_z = frac_div[sig_width*2: sig_width+1]; // no rounding here

    // Final packing
    always @(*) begin
        if (b == 0) begin
            z = {sign_z, {(exp_width){1'b1}}, {sig_width{1'b0}}}; // Inf
            status = 8'h10; // divide-by-zero flag
        end else if (a == 0) begin
            z = {sign_z, {exp_width{1'b0}}, {sig_width{1'b0}}}; // Zero
            status = 8'h01; // zero result
        end else begin
            z = {sign_z, exp_z[exp_width-1:0], frac_z};
            status = 8'h00;
        end
    end

endmodule


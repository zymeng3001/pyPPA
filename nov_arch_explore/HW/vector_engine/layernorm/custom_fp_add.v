module custom_fp_add #(
    parameter integer sig_width = 7,   // number of fraction bits
    parameter integer exp_width = 8,   // number of exponent bits
    parameter integer ieee_compliance = 0  // only basic handling when 0
)(
    input  wire [sig_width+exp_width:0] a,  // input a
    input  wire [sig_width+exp_width:0] b,  // input b
    input  wire [2:0]                   rnd, // rounding mode (ignored)
    output reg  [sig_width+exp_width:0] z,   // result = a + b
    output reg  [7:0]                   status // status flags
);

    // Sign, exponent, mantissa extraction
    wire sign_a = a[sig_width + exp_width];
    wire sign_b = b[sig_width + exp_width];
    wire [exp_width-1:0] exp_a = a[sig_width + exp_width - 1 : sig_width];
    wire [exp_width-1:0] exp_b = b[sig_width + exp_width - 1 : sig_width];
    wire [sig_width-1:0] frac_a = a[sig_width-1:0];
    wire [sig_width-1:0] frac_b = b[sig_width-1:0];

    // Add implicit leading 1 for normalized values
    wire [sig_width:0] mant_a = (|exp_a) ? {1'b1, frac_a} : {1'b0, frac_a};
    wire [sig_width:0] mant_b = (|exp_b) ? {1'b1, frac_b} : {1'b0, frac_b};

    // Align mantissas
    wire [exp_width:0] exp_diff = (exp_a > exp_b) ? (exp_a - exp_b) : (exp_b - exp_a);
    wire [sig_width+2:0] mant_a_align = (exp_a >= exp_b) ? {mant_a, 2'b00} : ({mant_a, 2'b00} >> exp_diff);
    wire [sig_width+2:0] mant_b_align = (exp_b > exp_a) ? {mant_b, 2'b00} : ({mant_b, 2'b00} >> exp_diff);

    // Add or subtract based on sign
    reg [sig_width+2:0] mant_add;
    reg [exp_width-1:0] exp_result;
    reg sign_result;

    always @(*) begin
        if (sign_a == sign_b) begin
            mant_add = mant_a_align + mant_b_align;
            sign_result = sign_a;
        end else begin
            if ({exp_a, mant_a} >= {exp_b, mant_b}) begin
                mant_add = mant_a_align - mant_b_align;
                sign_result = sign_a;
            end else begin
                mant_add = mant_b_align - mant_a_align;
                sign_result = sign_b;
            end
        end
        exp_result = (exp_a >= exp_b) ? exp_a : exp_b;
    end

    // Normalize result
    wire [sig_width-1:0] frac_norm;
    wire [2*sig_width-1:0] frac_norms;
    wire [exp_width-1:0] exp_norm;

    wire [$clog2(sig_width+3)-1:0] shift;
    wire                          valid;
    priority_encoder #(
        .WIDTH(sig_width + 3)
    ) pe_inst (
        .in(mant_add),
        .shift(shift),
        .valid(valid)
    );

    assign frac_norms = mant_add << shift;
    assign frac_norm  = frac_norms[sig_width+2:3];
    assign exp_norm   = (exp_result > shift) ? (exp_result - shift) : 0;

    // Final packing
    always @(*) begin
        z = {sign_result, exp_norm, frac_norm};
        status = 8'b0;
        if (mant_add == 0)
            z = {1'b0, {exp_width{1'b0}}, {sig_width{1'b0}}};  // exact zero
    end

endmodule

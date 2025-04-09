
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
    // This instance assumes that a DW_fp_div module is available in your library.
    DW_fp_div #(
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


module softmax_wrapper
#(
    parameter integer sig_width       = 8,
    parameter integer exp_width       = 8,
    parameter integer stages         = 2,
    parameter string  ieee_compliance = "IEEE",
    parameter integer en_ubr_flag     = 0
)(
    // Global Signals
    input wire clk,
    input wire rst_n
);
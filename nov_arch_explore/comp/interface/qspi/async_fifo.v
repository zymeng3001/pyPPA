// +FHDR========================================================================
//  File Name:      async_fifo.v
//                  Shiwei Liu (liusw18@gmail.com)
//  Organization:
//  Description:
//     Asynchronous FIFO
// -FHDR========================================================================

module async_fifo #(
    parameter   DW = 16,
    parameter   AW = 8
)(
    // Write Domain
    input                   wclk,
    input                   wrst,
    input                   wen,
    input       [DW-1:0]    wdata,
    output  reg             wfull,

    // Read Domain
    input                   rclk,
    input                   rrst,
    input                   ren,
    output  reg [DW-1:0]    rdata,
    output  reg             rempty
);

// =============================================================================
// FIFO Memory

    localparam DEPTH = 1 << AW;
    reg     [DW-1:0]    mem [0:DEPTH-1];

    wire    [AW-1:0]    waddr, raddr;

    always @(posedge wclk) begin
        if (wen && ~wfull) begin
            mem[waddr] <= wdata;
        end
    end

    always @(posedge rclk or posedge rrst) begin
        if (rrst) begin
            rdata <= 'd0;
        end
        else if (ren && ~rempty) begin
            rdata <= mem[raddr];
        end
    end

// =============================================================================
// Declaration

    // 1. Read Domain
    reg     [AW:0]  rptr, rbin;             // Gray, Binary Code
    wire    [AW:0]  rgraynext, rbinnext;    // Gray, Binary Code
    reg     [AW:0]  rq1_wptr, rq2_wptr;     // Write (Gray) -> Read (Gray)

    // 2. Write Domain
    reg     [AW:0]  wptr, wbin;             // Gray, Binary Code
    wire    [AW:0]  wgraynext, wbinnext;    // Gray, Binary Code
    reg     [AW:0]  wq1_rptr, wq2_rptr;     // Write (Gray) <- Read (Gray)

// =============================================================================
// Read Domain

    // 1. Read Binary and Gray Code Generation
    always @(posedge rclk or posedge rrst) begin
        if (rrst) begin
            rbin <= 'd0;
            rptr <= 'd0;
        end
        else begin
            rbin <= rbinnext;
            rptr <= rgraynext;
        end
    end

    assign  rbinnext  = rbin + (ren && ~rempty);
    assign  rgraynext = (rbinnext >> 1) ^ rbinnext;
    assign  raddr     = rbin[AW-1:0];

    // 2. CDC: Read Gray Code -> Write Gray Code
    always @(posedge wclk or posedge wrst) begin
        if (wrst) begin
            {wq2_rptr, wq1_rptr} <= 'd0;
        end
        else begin
            {wq2_rptr, wq1_rptr} <= {wq1_rptr, rptr};
        end
    end

// =============================================================================
// Write Domain

    // 1. Write Binary and Gray Code Generation
    always @(posedge wclk or posedge wrst) begin
        if (wrst) begin
            wbin <= 'd0;
            wptr <= 'd0;
        end
        else begin
            wbin <= wbinnext;
            wptr <= wgraynext;
        end
    end

    assign  wbinnext  = wbin + (wen && ~wfull);
    assign  wgraynext = (wbinnext >> 1) ^ wbinnext;
    assign  waddr     = wbin[AW-1:0];

    // 2. CDC: Write Gray Code -> Read Gray Code
    always @(posedge rclk or posedge rrst) begin
        if (rrst) begin
            {rq2_wptr, rq1_wptr} <= 'd0;
        end
        else begin
            {rq2_wptr, rq1_wptr} <= {rq1_wptr, wptr};
        end
    end

// =============================================================================
// Full and Empty Check

    // 1. Read Empty
    wire    rempty_val;
    assign  rempty_val = rgraynext == rq2_wptr; //格雷码判断空满条件

    always @(posedge rclk or posedge rrst) begin
        if (rrst) begin
            rempty <= 1'b1;
        end
        else begin
            rempty <= rempty_val;
        end
    end

    // 2. Write Full
    wire    wfull_val;
    assign  wfull_val = wgraynext == {~wq2_rptr[AW:AW-1], wq2_rptr[AW-2:0]}; //格雷码判断空满条件

    always @(posedge wclk or posedge wrst) begin
        if (wrst) begin
            wfull <= 1'b0;
        end
        else begin
            wfull <= wfull_val;
        end
    end

endmodule
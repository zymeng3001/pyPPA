module fifo #(parameter DEPTH = 16, WIDTH = 8)(
    input wire clk,
    input wire rstn,
    input wire wr_en,           // Write enable
    input wire rd_en,           // Read enable
    input wire [WIDTH-1:0] din, // Data input
    output reg [WIDTH-1:0] dout, // Data output
    output wire full,           // FIFO full flag
    output wire empty           // FIFO empty flag
);

    localparam ADDR_WIDTH = $clog2(DEPTH); // Address width based on depth

    reg [WIDTH-1:0] mem [0:DEPTH-1]; // FIFO memory storage
    reg [ADDR_WIDTH-1:0] wr_ptr, rd_ptr; // Write and read pointers
    reg [ADDR_WIDTH:0] count; // Counter to track FIFO occupancy

    always_ff @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            wr_ptr <= 0;
            rd_ptr <= 0;
            count <= 0;
        end else begin
            // Write operation
            if (wr_en && !full) begin
                mem[wr_ptr] <= din;
                wr_ptr <= wr_ptr + 1;
                count <= count + 1;
            end

            // Read operation
            if (rd_en && !empty) begin
                dout <= mem[rd_ptr];
                rd_ptr <= rd_ptr + 1;
                count <= count - 1;
            end else begin
                dout <= 0;
            end
        end
    end

    // Full and Empty flag wire based on count
    assign full = (count == DEPTH);
    assign empty = (count == 0);

endmodule
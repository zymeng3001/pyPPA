module residual_adder(
    input  logic                                                                                   clk,
    input  logic                                                                                   rst_n,

    input  logic                                                                                   scale_vld,
    input  logic        [`CDATA_SCALE_WIDTH-1:0]                                                       scale_a,
    input  logic        [`CDATA_SCALE_WIDTH-1:0]                                                       scale_b,

    input  logic                                                                                   in_finish,

    input  logic                                                                                   shift_vld,
    input  logic        [`CDATA_SHIFT_WIDTH-1:0]                                                   shift,

    input  logic        [`MAC_MULT_NUM-1:0][`IDATA_WIDTH-1:0]                                      in_data_a,
    input  logic        [`MAC_MULT_NUM-1:0][`IDATA_WIDTH-1:0]                                      in_data_b,
    input  logic                                                                                   in_data_vld,
    input  logic        [$clog2(`GLOBAL_SRAM_DEPTH)+$clog2(`MAC_MULT_NUM)-1:0]                     in_addr,

    output logic        [`MAC_MULT_NUM-1:0][`IDATA_WIDTH-1:0]                                      out_data,
    output logic                                                                                   out_data_vld,
    output logic        [$clog2(`GLOBAL_SRAM_DEPTH)+$clog2(`MAC_MULT_NUM)-1:0]                     out_addr,

    output logic                                                                                   out_finish
);

logic [`MAC_MULT_NUM-1:0] out_data_vld_array;
assign out_data_vld = |out_data_vld_array;

localparam addr_delay = `RESIDUAL_DEQUANT_RETIMING + 6;
genvar i;
generate
    for(i=0; i<addr_delay; i++)begin : adddr_delay_gen_array
        logic  [$clog2(`GLOBAL_SRAM_DEPTH)+$clog2(`MAC_MULT_NUM)-1:0]                   temp_addr;
        logic                                                                           temp_finish;
        if(i==0)begin
            always_ff @(posedge clk or negedge rst_n) begin
                if(~rst_n)begin
                    temp_addr <= 0;
                    temp_finish <= 0;
                end
                else begin
                    temp_addr <= in_addr;
                    temp_finish <= in_finish;
                end
            end
        end
        else begin
            always_ff @(posedge clk or negedge rst_n) begin
                if(~rst_n)begin
                    temp_addr <= 0;
                    temp_finish <= 0;
                end
                else begin
                    temp_addr <= adddr_delay_gen_array[i-1].temp_addr;
                    temp_finish <= adddr_delay_gen_array[i-1].temp_finish;
                end
            end
        end
    end
endgenerate

assign out_addr = adddr_delay_gen_array[addr_delay-1].temp_addr;
assign out_finish = adddr_delay_gen_array[addr_delay-1].temp_finish;

// genvar i;
generate
    for(i = 0; i < `MAC_MULT_NUM; i++)begin : single_residual_adder_gen_array
        residual_adder_single inst_single_adder(
            .clk(clk),
            .rst_n(rst_n),

            .scale_vld(scale_vld),
            .scale_a(scale_a),
            .scale_b(scale_b),

            .shift_vld(shift_vld),
            .shift(shift),

            .in_data_a(in_data_a[i]),
            .in_data_b(in_data_b[i]),
            .in_data_vld(in_data_vld),

            .out_data(out_data[i]),
            .out_data_vld(out_data_vld_array[i])
        );
    end
endgenerate



endmodule



module residual_adder_single(
    input  logic                                        clk,
    input  logic                                        rst_n,

    input  logic                                        scale_vld,
    input  logic        [`CDATA_SCALE_WIDTH-1:0]            scale_a,
    input  logic        [`CDATA_SCALE_WIDTH-1:0]            scale_b,

    input  logic                                        shift_vld,
    input  logic        [`CDATA_SHIFT_WIDTH-1:0]        shift,

    input  logic signed [`IDATA_WIDTH-1:0]              in_data_a,
    input  logic signed [`IDATA_WIDTH-1:0]              in_data_b,
    input  logic                                        in_data_vld,

    output logic signed [`IDATA_WIDTH-1:0]              out_data,
    output logic                                        out_data_vld
);

logic        [`CDATA_SCALE_WIDTH-1:0]            scale_a_reg;
logic        [`CDATA_SCALE_WIDTH-1:0]            scale_b_reg;
logic        [`CDATA_SHIFT_WIDTH-1:0]        shift_reg;

always_ff @(posedge clk or negedge rst_n) begin
    if(~rst_n)begin
        scale_a_reg <= 0;
        scale_b_reg <= 0;
    end
    else if(scale_vld)begin
        scale_a_reg <= scale_a;
        scale_b_reg <= scale_b;
    end
end

always_ff @(posedge clk or negedge rst_n) begin
    if(~rst_n)begin
        shift_reg <= 0;
    end
    else if(shift_vld)begin
        shift_reg <= shift;
    end
end

logic signed [`IDATA_WIDTH-1:0]              in_data_a_delay1; //gating
logic signed [`IDATA_WIDTH-1:0]              in_data_b_delay1;
logic                                        in_data_vld_delay1;

logic signed [`CDATA_SCALE_WIDTH + `IDATA_WIDTH-1:0] dequant_a;
logic signed [`CDATA_SCALE_WIDTH + `IDATA_WIDTH-1:0] dequant_b;
logic                                                dequant_vld;
logic signed [`CDATA_SCALE_WIDTH + `IDATA_WIDTH-1:0] dequant_a_delay;
logic signed [`CDATA_SCALE_WIDTH + `IDATA_WIDTH-1:0] dequant_b_delay;
logic                                                dequant_vld_delay;

always_ff @(posedge clk or negedge rst_n) begin
    if(~rst_n)begin
        in_data_a_delay1 <= 0;
        in_data_b_delay1 <= 0;
        in_data_vld_delay1 <= 0;
    end
    else if(in_data_vld)begin
        in_data_a_delay1 <= in_data_a;
        in_data_b_delay1 <= in_data_b;
        in_data_vld_delay1 <= 1;
    end
    else begin
        in_data_vld_delay1 <= 0;
    end
end

always_ff @(posedge clk or negedge rst_n) begin
    if(~rst_n)begin
        dequant_a <= 0;
        dequant_b <= 0;
        dequant_vld <= 0;
    end
    else if(in_data_vld_delay1)begin
        dequant_a <= $signed(in_data_a_delay1) * $signed({1'b0, scale_a_reg});
        dequant_b <= $signed(in_data_b_delay1) * $signed({1'b0, scale_b_reg});
        dequant_vld <= 1;
    end
    else begin
        dequant_vld <= 0;
    end
end

genvar i;
generate;
    for(i = 0; i < `RESIDUAL_DEQUANT_RETIMING; i++)begin : deqaunt_rt_array
        logic [`CDATA_SCALE_WIDTH + `IDATA_WIDTH-1:0] dequant_a_temp;
        logic [`CDATA_SCALE_WIDTH + `IDATA_WIDTH-1:0] dequant_b_temp;
        logic                                         dequant_vld_temp;
        if(i==0)begin
            always_ff@(posedge clk or negedge rst_n)begin
                if(~rst_n)begin
                    dequant_vld_temp <= 0;
                    dequant_a_temp <= 0;
                    dequant_b_temp <= 0;
                end
                else begin
                    dequant_vld_temp <= dequant_vld;
                    dequant_a_temp <= dequant_a;
                    dequant_b_temp <= dequant_b;
                end
            end
        end
        else begin
            always_ff@(posedge clk or negedge rst_n)begin
                if(~rst_n)begin
                    dequant_vld_temp <= 0;
                    dequant_a_temp <= 0;
                    dequant_b_temp <= 0;
                end
                else begin
                    dequant_vld_temp <= deqaunt_rt_array[i-1].dequant_vld_temp;
                    dequant_a_temp <= deqaunt_rt_array[i-1].dequant_a_temp;
                    dequant_b_temp <= deqaunt_rt_array[i-1].dequant_b_temp;
                end
            end
        end
    end
endgenerate

assign dequant_a_delay = deqaunt_rt_array[`RESIDUAL_DEQUANT_RETIMING-1].dequant_a_temp;
assign dequant_b_delay = deqaunt_rt_array[`RESIDUAL_DEQUANT_RETIMING-1].dequant_b_temp;
assign dequant_vld_delay = deqaunt_rt_array[`RESIDUAL_DEQUANT_RETIMING-1].dequant_vld_temp;

//adder
logic signed [`CDATA_SCALE_WIDTH + `IDATA_WIDTH + 1 -1:0] sum_ab;
logic                                                     sum_ab_vld;
always_ff @(posedge clk or negedge rst_n) begin
    if(~rst_n)begin
        sum_ab <= 0;
        sum_ab_vld <= 0;
    end
    else if (dequant_vld_delay) begin
        sum_ab_vld <= 1;
        sum_ab <= dequant_a_delay + dequant_b_delay;
    end
    else begin
        sum_ab_vld <= 0;
    end
end

//round
logic round;
assign round =  (shift_reg > 0) ? sum_ab[shift_reg-1] : 0;

//shift
logic signed [`CDATA_SCALE_WIDTH + `IDATA_WIDTH + 1 -1:0] shift_sum_ab;
logic                                                     shift_sum_ab_vld;
logic                                                     round_delay1;

always_ff@(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        shift_sum_ab_vld <= 0;
        shift_sum_ab <= 0;
        round_delay1 <= 0;
    end
    else if(sum_ab_vld) begin
        shift_sum_ab <= sum_ab >>> shift_reg;
        shift_sum_ab_vld <= 1;
        round_delay1 <= round;
    end
    else begin
        shift_sum_ab_vld <= 0;
    end
end

//round
logic signed [`CDATA_SCALE_WIDTH + `IDATA_WIDTH + 1 -1:0] round_sum_ab;
logic                                                     round_sum_ab_vld;

always_ff@(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        round_sum_ab_vld <= 0;
        round_sum_ab <= 0;
    end
    else if(shift_sum_ab_vld)begin
        round_sum_ab <= shift_sum_ab + round_delay1;
        round_sum_ab_vld <= 1;
    end
    else begin
        round_sum_ab_vld <= 0;
    end
end

//overflow and underflow;
logic signed [`IDATA_WIDTH-1:0]              nxt_out_data;
logic                                        nxt_out_data_vld;
always_comb begin
    nxt_out_data_vld = 0;
    nxt_out_data = out_data;
    if(round_sum_ab_vld)begin
        nxt_out_data_vld = 1;
        if(round_sum_ab > 127)begin
            nxt_out_data = 127;
        end
        else if(round_sum_ab < -128)begin
            nxt_out_data = -128;
        end
        else begin
            nxt_out_data = round_sum_ab;
        end
    end
end

always_ff@(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        out_data_vld <= 0;
        out_data <= 0;
    end
    else begin
        out_data_vld <= nxt_out_data_vld;
        out_data <= nxt_out_data;
    end
end

endmodule
module vector_adder (
    input  logic                                        clk,
    input  logic                                        rst_n,

    input  logic [`HEAD_NUM-1:0][`GBUS_DATA_WIDTH-1:0]  in_data,
    input  logic [`HEAD_NUM-1:0]                        in_data_vld,

    input  logic [`CMEM_ADDR_WIDTH-1:0]                 in_data_addr,

    input  logic                                        op_cfg_vld,
    input  OP_CONFIG                                    op_cfg,

    output logic                 [`IDATA_WIDTH-1:0]     out_data,
    output logic                                        out_data_vld,
    output logic [`CMEM_ADDR_WIDTH-1:0]                 out_data_addr,

    input  logic                                        in_finish,
    output logic                                        out_finish
);
localparam SUM_WIDTH = `ODATA_WIDTH + $clog2(`HEAD_NUM);
OP_CONFIG                               op_cfg_reg;

always_ff@(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        op_cfg_reg <= 0;
    end
    else if(op_cfg_vld)begin
        op_cfg_reg <= op_cfg;
    end
end

//addr delay for alias
localparam delay_latency = 10 + `QUANT_SCALE_RETIMING;
genvar i;
generate
    for(i = 0; i < delay_latency; i++)begin : addr_delay_gen_array
        logic [`CMEM_ADDR_WIDTH-1:0]                 gen_data_addr;
        logic                                        temp_finish;
        if(i==0)begin
            always_ff@(posedge clk)begin
                gen_data_addr <= in_data_addr;
            end

            always_ff@(posedge clk or negedge rst_n)begin
                if(~rst_n)begin
                    temp_finish <= 0;
                end
                else begin
                    temp_finish <= in_finish;
                end
            end
        end
        else begin
            always_ff@(posedge clk)begin
                gen_data_addr <= addr_delay_gen_array[i-1].gen_data_addr;
            end

            always_ff@(posedge clk or negedge rst_n)begin
                if(~rst_n)begin
                    temp_finish <= 0;
                end
                else begin
                    temp_finish <= addr_delay_gen_array[i-1].temp_finish;
                end
            end
        end
    end
endgenerate
assign out_finish = addr_delay_gen_array[delay_latency-1].temp_finish;
assign out_data_addr = addr_delay_gen_array[delay_latency-1].gen_data_addr;

logic [SUM_WIDTH-1:0] sum_data;
logic                 sum_data_vld;

logic [`HEAD_NUM-1:0][`ODATA_WIDTH-1:0]      in_data_delay1; //截取低位
logic                                        in_data_vld_delay1;

always_ff @(posedge clk or negedge rst_n) begin //in data buffer and gating
    if(~rst_n)begin
        in_data_vld_delay1 <= 0;
    end
    else if(|in_data_vld)begin
        in_data_vld_delay1 <= 1;
    end
    else begin
        in_data_vld_delay1 <= 0;
    end
end

generate
    for(i=0; i < `HEAD_NUM; i++)begin : gen_array_in_data_delay1
        always_ff @(posedge clk or negedge rst_n) begin //in data buffer and gating
            if(~rst_n)begin
                in_data_delay1[i] <= 0;
            end
            else if(|in_data_vld)begin
                in_data_delay1[i] <= in_data[i];//截取低位
            end
        end
    end
endgenerate


logic                 [`IDATA_WIDTH-1:0]     quant_out_data;
logic                                        quant_out_data_vld;

logic                 [`IDATA_WIDTH-1:0]     nxt_out_data;
logic                                        nxt_out_data_vld;

always_ff@(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        nxt_out_data_vld <= 0;
        nxt_out_data <= 0;
    end
    else begin
        nxt_out_data_vld <= quant_out_data_vld;
        nxt_out_data <= quant_out_data;
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

adder_tree #(
    .MAC_MULT_NUM(`HEAD_NUM),
    .IDATA_WIDTH(`ODATA_WIDTH),
    .ODATA_BIT(SUM_WIDTH)
)vect_add_tree(
    // Global Signals
    .clk(clk),
    .rstn(rst_n),

    // Data Signals
    .idata(in_data_delay1),
    .idata_valid(in_data_vld_delay1),
    .odata(sum_data),
    .odata_valid(sum_data_vld)
);


core_quant #(
    .IDATA_WIDTH(SUM_WIDTH),
    .ODATA_BIT(`IDATA_WIDTH)
)vect_core_quant(
    // Global Signals
    .clk(clk),
    .rstn(rst_n),

    // Global Config Signals
    .cfg_quant_scale(op_cfg_reg.cfg_quant_scale),
    .cfg_quant_bias(op_cfg_reg.cfg_quant_bias),
    .cfg_quant_shift(op_cfg_reg.cfg_quant_shift),

    // Data Signals
    .idata(sum_data),
    .idata_valid(sum_data_vld),
    .odata(quant_out_data),
    .odata_valid(quant_out_data_vld)
);
    
endmodule
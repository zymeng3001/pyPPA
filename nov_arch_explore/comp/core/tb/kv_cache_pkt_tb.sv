module tb_kv_cache_pkt;
    // Parameters
    localparam IDATA_WIDTH = (`MAC_MULT_NUM * `IDATA_WIDTH);
    localparam ODATA_BIT = (`MAC_MULT_NUM * `IDATA_WIDTH);
    localparam CACHE_NUM = `MAC_MULT_NUM;
    localparam CACHE_PKT_NUM = `KV_CACHE_PKT_NUM;
    localparam CACHE_DEPTH = 256;
    localparam CACHE_ADDR_WIDTH = ($clog2(CACHE_NUM) + $clog2(CACHE_DEPTH));

    localparam DATA_NUM = 50;

    // Inputs
    reg clk;
    reg rstn;
    reg [CACHE_ADDR_WIDTH-1:0] cache_addr;
    reg cache_ren;
    reg cache_wen;
    reg [IDATA_WIDTH-1:0] cache_wdata;
    reg cache_wdata_byte_flag;
    USER_CONFIG usr_cfg;
    reg [CACHE_PKT_NUM-1:0][CACHE_NUM-1:0]  deepslp;
    reg [CACHE_PKT_NUM-1:0][CACHE_NUM-1:0]  shutoff;
    reg [CACHE_PKT_NUM-1:0][CACHE_NUM-1:0]  sleep;
    reg bc1;
    reg bc2;

    // Outputs
    wire    [ODATA_BIT-1:0] cache_rdata;
    logic    [ODATA_BIT-1:0] cache_rdata_old;
    wire    [CACHE_PKT_NUM-1:0][CACHE_NUM-1:0] mpr;
    logic   [CACHE_PKT_NUM-1:0][CACHE_NUM-1:0] mpr_old;

`ifndef SYNTH 
    kv_cache_pkt #(
        .IDATA_WIDTH(IDATA_WIDTH),
        .ODATA_BIT(ODATA_BIT),
        .CACHE_NUM(CACHE_NUM),
        .CACHE_PKT_NUM(CACHE_PKT_NUM),
        .CACHE_DEPTH(CACHE_DEPTH),
        .CACHE_ADDR_WIDTH(CACHE_ADDR_WIDTH)
    ) dut (
        .clk(clk),
        .rstn(rstn),
        .cache_addr(cache_addr),
        .cache_ren(cache_ren),
        .cache_rdata(cache_rdata),
        .cache_wen(cache_wen),
        .cache_wdata(cache_wdata),
        .cache_wdata_byte_flag(cache_wdata_byte_flag),
        .usr_cfg(usr_cfg),
        .deepslp(deepslp),
        .shutoff(shutoff),
        .sleep(sleep),
        .bc1(bc1),
        .bc2(bc2),
        .mpr(mpr)
    );
`endif

`ifdef SYNTH 
    kv_cache_pkt #(
    ) dut (
        .clk(clk),
        .rstn(rstn),
        .cache_addr(cache_addr),
        .cache_ren(cache_ren),
        .cache_rdata(cache_rdata),
        .cache_wen(cache_wen),
        .cache_wdata(cache_wdata),
        .cache_wdata_byte_flag(cache_wdata_byte_flag),
        .usr_cfg_user_id_1_(usr_cfg.user_id[1]),
        .usr_cfg_user_id_0_(usr_cfg.user_id[0]),
        .usr_cfg_user_token_cnt_7_(0),
        .usr_cfg_user_token_cnt_6_(0),
        .usr_cfg_user_token_cnt_5_(0),
        .usr_cfg_user_token_cnt_4_(0),
        .usr_cfg_user_token_cnt_3_(0),
        .usr_cfg_user_token_cnt_2_(0),
        .usr_cfg_user_token_cnt_1_(0),
        .usr_cfg_user_token_cnt_0_(0),
        .usr_cfg_user_kv_cache_not_full (0),
        .usr_cfg_user_first_token_flag (0),
        .deepslp(deepslp),
        .shutoff(shutoff),
        .sleep(sleep),
        .bc1(bc1),
        .bc2(bc2),
        .mpr(mpr)
    );
`endif

    // Clock generation
    always
        #(`CLOCK_PERIOD/2.0) clk = ~clk;


    reg [IDATA_WIDTH-1:0] input_array [DATA_NUM-1:0];
    integer wake_up_cycles;
    integer sleep_cycles;
    // Test sequence
    initial begin
        $fsdbDumpfile("kv_cache_pkt.fsdb");
        $fsdbDumpvars(0,"+all",tb_kv_cache_pkt); //"+all" enables all  signal dumping including Mem, packed array, etc.

    `ifdef SYNTH
        $sdf_annotate("../intel_flow/outputs_fc/write_data.tttt_0.800v_25c.tttt.sdf", dut);
    `endif
        // Initialize Inputs
        clk = 0;
        rstn = 1;
        cache_addr = 0;
        cache_ren = 0;
        cache_wen = 0;
        cache_wdata = 0;
        cache_wdata_byte_flag = 0; //write the whole wordline
        usr_cfg.user_id = 0;
        deepslp = 0;
        shutoff = 0;
        sleep = 0;
        bc1 = 0;
        bc2 = 0;

        @(negedge clk);
        rstn = 0;
        repeat(5)
        @(negedge clk);
        rstn = 1;

        init_input_array(input_array);

        repeat(5) @(negedge clk);
        // Assert user 3,4's kv cache in sleep mode
        //sleep[1] = {CACHE_NUM{1'b1}};
        deepslp[1] = {CACHE_NUM{1'b1}};

        // Write to cache
        for(int i=0;i<DATA_NUM;i++) begin
            cache_wdata = input_array[i];
            cache_wen = 1;
            @(negedge clk);
            cache_addr = cache_addr + 1;
            cache_wen = 0;
        end
        

        repeat(5) @(negedge clk);
        // Read from cache
        cache_addr = 0;
        for(int i=0;i<DATA_NUM;i++) begin
            cache_ren = 1;
            @(negedge clk);
            if(cache_rdata!=input_array[i]) begin
                $display("Something wrong: wdata:%d, rdata:%d, addr:%d",input_array[i],cache_rdata,i);
            end
            cache_addr = cache_addr + 1;
            cache_ren = 0;
        end

        #100
        //Wake up user 3,4's kv cache, assert user 1,2's kv cache in sleep mode
        //wait for user 1,2's kv cache to sleep
        // sleep[1] = 0;
        // sleep[0] = {CACHE_NUM{1'b1}};
        deepslp[1] = 0;
        deepslp[0] = {CACHE_NUM{1'b1}};
        
        // sleep_cycles = 0;
        // while(1) begin
        //     mpr_old = mpr;
        //     @(negedge clk)
        //     sleep_cycles = sleep_cycles + 1;
        //     if(mpr_old[0]!=mpr[0])
        //         break;
        // end

        cache_addr = 0;
        // Write to user 1,2's kv cache while it is sleep
        for(int i=0;i<DATA_NUM;i++) begin
            cache_wdata = input_array[i];
            cache_wen = 1;
            @(negedge clk);
            cache_addr = cache_addr + 1;
            cache_wen = 0;
        end
        
        cache_rdata_old = cache_rdata;
        cache_addr = 0;
        // Read from cache
        for(int i=0;i<DATA_NUM;i++) begin
            cache_ren = 1;
            @(negedge clk);
            if(cache_rdata != cache_rdata_old)
                $display("Something wrong!");
            cache_addr = cache_addr + 1;
            cache_ren = 0;
        end

        //test wake up cycles
        //Wake up user 1,2's kv cache, assert user 3,4's kv cache in sleep mode
        #100
        // sleep[1] = {CACHE_NUM{1'b1}};
        // sleep[0] = 0;
        deepslp[1] = {CACHE_NUM{1'b1}};
        deepslp[0] = 0;
        // Read from cache
        cache_addr = 0;
        for(int i=0;i<DATA_NUM;i++) begin
            cache_ren = 1;
            @(negedge clk);
            cache_ren = 0;
            if(cache_rdata!=cache_rdata_old) begin
                $display("wake up cycles:%d",i);
                break;
            end
        end


        $finish;
    end

    task init_input_array (output [IDATA_WIDTH-1:0] input_array [DATA_NUM-1:0]);
        for(int i=0;i<DATA_NUM;i++) begin
            input_array[i] = {$random, $random};
        end
    endtask

endmodule
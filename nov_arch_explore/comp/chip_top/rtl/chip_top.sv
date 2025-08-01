module chip_top(
    inout wire [63:0] gpio //gpio[40] is analog
);


////////////////////////////////////////////////////////
//Chip
////////////////////////////////////////////////////////
logic                                               clk ;
logic                                               asyn_rst ; //input gpio[0]

logic                                               IIN; //input gpio[40]
// logic                                               CKON;
logic                                               EXT_CLK_IN; //input gpio[41]
logic                                               CLK_DIV_SEL_0; //input gpio[42]
logic                                               CLK_DIV_SEL_1; //input gpio[43]
logic                                               CLK_GATE_EN; //input gpio[44]
logic                                               CLK_DIV100; //output gpio[45]
// SPI Domain
logic                                               spi_clk;        // gpio[1]      
logic                                               spi_csn;        // SPI Active Low gpio[2]
logic                                               spi_mosi;       // Host -> SPI gpio[3]
logic                                               spi_miso;       // Host <- SPI gpio[4]
// QSPI Domain
logic                                               qspi_clk;        //qspi clk gpio[5]
logic [15:0]                                        qspi_mosi;       // Host -> QSPI -> TPU gpio[6]~gpio[21]
logic                                               qspi_mosi_valid; // gpio[22]
logic [15:0]                                        qspi_miso;       // Host <- QSPI <- TPU gpio[23]~gpio[38]
logic                                               qspi_miso_valid; // gpio[39]
// STATE PIN
logic                                               current_token_finish_flag; //gpio[46]
logic                                               current_token_finish_work; //gpio[47]
logic                                               qgen_state_work; //gpio[48]
logic                                               qgen_state_end; //gpio[49]
logic                                               kgen_state_work; //gpio[50]
logic                                               kgen_state_end; //gpio[51]
logic                                               vgen_state_work; //gpio[52]
logic                                               vgen_state_end; //gpio[53]
logic                                               att_qk_state_work; //gpio[54]
logic                                               att_qk_state_end; //gpio[55]
logic                                               att_pv_state_work; //gpio[56]
logic                                               att_pv_state_end; //gpio[57]
logic                                               proj_state_work; //gpio[58]
logic                                               proj_state_end; //gpio[59]
logic                                               ffn0_state_work; //gpio[60]
logic                                               ffn0_state_end; //gpio[61]
logic                                               ffn1_state_work; //gpio[62]
logic                                               ffn1_state_end; //gpio[63]

CLK_GEN i_clk_gen (.asyn_rst(asyn_rst),.ext_clk(EXT_CLK_IN),.IIN(IIN),.clk_div_sel({CLK_DIV_SEL_1,CLK_DIV_SEL_0}),.clk_gate_en(CLK_GATE_EN),.o_chip_clk(clk),.o_clk_div(CLK_DIV100));


top top(
    .chip_clk(clk),
    .asyn_rst(asyn_rst), //rst!, not rst_n

    // SPI Domain
    .spi_clk(spi_clk),        
    .spi_csn(spi_csn),        // SPI Active Low
    .spi_mosi(spi_mosi),       // Host -> SPI
    .spi_miso(spi_miso),       // Host <- SPI

    // QSPI Domain
    .qspi_clk(qspi_clk),        //qspi clk
    .qspi_mosi(qspi_mosi),       // Host -> QSPI -> TPU
    .qspi_mosi_valid(qspi_mosi_valid),
    .qspi_miso(qspi_miso),       // Host <- QSPI <- TPU
    .qspi_miso_valid(qspi_miso_valid),

    // STATE PIN
    .current_token_finish_flag(current_token_finish_flag),
    .current_token_finish_work(current_token_finish_work),

    .qgen_state_work(qgen_state_work),
    .qgen_state_end(qgen_state_end),

    .kgen_state_work(kgen_state_work),
    .kgen_state_end(kgen_state_end),

    .vgen_state_work(vgen_state_work),
    .vgen_state_end(vgen_state_end),

    .att_qk_state_work(att_qk_state_work),
    .att_qk_state_end(att_qk_state_end),

    .att_pv_state_work(att_pv_state_work),
    .att_pv_state_end(att_pv_state_end),

    .proj_state_work(proj_state_work),
    .proj_state_end(proj_state_end),

    .ffn0_state_work(ffn0_state_work),
    .ffn0_state_end(ffn0_state_end),

    .ffn1_state_work(ffn1_state_work),
    .ffn1_state_end(ffn1_state_end)
);

////////////////////////////////////////////////////////
//gpio input/output
////////////////////////////////////////////////////////

//rstn
sdio_1v8_n1 south_io_0 ( //input
    .outi          (asyn_rst),
    .outi_1v8      (),  
    .pad           (gpio[0]),
    .ana_io_1v8    (),     
    .dq            (1'b0),
    .drv0          (1'b0),
    .drv1          (1'b0),
    .drv2          (1'b0),
    .enabq         (1'b0),
    .enq           (1'b1),
    .pd            (1'b1),
    .ppen          (1'b0),
    .prg_slew      (1'b0),
    .puq           (1'b1),
    .pwrupzhl      (1'b0),
    .pwrup_pull_en (1'b0)
);

sdio_1v8_n1 south_io_1 ( //input
    .outi          (spi_clk),
    .outi_1v8      (),  
    .pad           (gpio[1]),
    .ana_io_1v8    (),     
    .dq            (1'b0),
    .drv0          (1'b0),
    .drv1          (1'b0),
    .drv2          (1'b0),
    .enabq         (1'b0),
    .enq           (1'b1),
    .pd            (1'b1),
    .ppen          (1'b0),
    .prg_slew      (1'b0),
    .puq           (1'b1),
    .pwrupzhl      (1'b0),
    .pwrup_pull_en (1'b0)
);

sdio_1v8_n1 south_io_2 ( //input
    .outi          (spi_csn),
    .outi_1v8      (),  
    .pad           (gpio[2]),
    .ana_io_1v8    (),     
    .dq            (1'b0),
    .drv0          (1'b0),
    .drv1          (1'b0),
    .drv2          (1'b0),
    .enabq         (1'b0),
    .enq           (1'b1),
    .pd            (1'b1),
    .ppen          (1'b0),
    .prg_slew      (1'b0),
    .puq           (1'b1),
    .pwrupzhl      (1'b0),
    .pwrup_pull_en (1'b0)
);

sdio_1v8_n1 south_io_3 ( //input
    .outi          (spi_mosi),
    .outi_1v8      (),  
    .pad           (gpio[3]),
    .ana_io_1v8    (),     
    .dq            (1'b0),
    .drv0          (1'b0),
    .drv1          (1'b0),
    .drv2          (1'b0),
    .enabq         (1'b0),
    .enq           (1'b1),
    .pd            (1'b1),
    .ppen          (1'b0),
    .prg_slew      (1'b0),
    .puq           (1'b1),
    .pwrupzhl      (1'b0),
    .pwrup_pull_en (1'b0)
);

sdio_1v8_n1 south_io_4 ( //output
    .outi          (),
    .outi_1v8      (),     
    .pad           (gpio[4]),
    .ana_io_1v8    (),
    .dq            (~spi_miso),
    .drv0          (1'b0),
    .drv1          (1'b0),
    .drv2          (1'b0),
    .enabq         (1'b0),
    .enq           (1'b0),
    .pd            (1'b1),
    .ppen          (1'b1),
    .prg_slew      (1'b0),
    .puq           (1'b1),
    .pwrupzhl      (1'b0),
    .pwrup_pull_en (1'b0)
);

sdio_1v8_n1 south_io_5 ( //input
    .outi          (qspi_clk),
    .outi_1v8      (),  
    .pad           (gpio[5]),
    .ana_io_1v8    (),     
    .dq            (1'b0),
    .drv0          (1'b0),
    .drv1          (1'b0),
    .drv2          (1'b0),
    .enabq         (1'b0),
    .enq           (1'b1),
    .pd            (1'b1),
    .ppen          (1'b0),
    .prg_slew      (1'b0),
    .puq           (1'b1),
    .pwrupzhl      (1'b0),
    .pwrup_pull_en (1'b0)
);

sdio_1v8_n1 south_io_6 ( //input
    .outi          (qspi_mosi[0]),
    .outi_1v8      (),  
    .pad           (gpio[6]),
    .ana_io_1v8    (),     
    .dq            (1'b0),
    .drv0          (1'b0),
    .drv1          (1'b0),
    .drv2          (1'b0),
    .enabq         (1'b0),
    .enq           (1'b1),
    .pd            (1'b1),
    .ppen          (1'b0),
    .prg_slew      (1'b0),
    .puq           (1'b1),
    .pwrupzhl      (1'b0),
    .pwrup_pull_en (1'b0)
);

sdio_1v8_n1 south_io_7 ( //input
    .outi          (qspi_mosi[1]),
    .outi_1v8      (),  
    .pad           (gpio[7]),
    .ana_io_1v8    (),     
    .dq            (1'b0),
    .drv0          (1'b0),
    .drv1          (1'b0),
    .drv2          (1'b0),
    .enabq         (1'b0),
    .enq           (1'b1),
    .pd            (1'b1),
    .ppen          (1'b0),
    .prg_slew      (1'b0),
    .puq           (1'b1),
    .pwrupzhl      (1'b0),
    .pwrup_pull_en (1'b0)
);

sdio_1v8_n1 south_io_8 ( //input
    .outi          (qspi_mosi[2]),
    .outi_1v8      (),  
    .pad           (gpio[8]),
    .ana_io_1v8    (),     
    .dq            (1'b0),
    .drv0          (1'b0),
    .drv1          (1'b0),
    .drv2          (1'b0),
    .enabq         (1'b0),
    .enq           (1'b1),
    .pd            (1'b1),
    .ppen          (1'b0),
    .prg_slew      (1'b0),
    .puq           (1'b1),
    .pwrupzhl      (1'b0),
    .pwrup_pull_en (1'b0)
);

sdio_1v8_n1 south_io_9 ( //input
    .outi          (qspi_mosi[3]),
    .outi_1v8      (),  
    .pad           (gpio[9]),
    .ana_io_1v8    (),     
    .dq            (1'b0),
    .drv0          (1'b0),
    .drv1          (1'b0),
    .drv2          (1'b0),
    .enabq         (1'b0),
    .enq           (1'b1),
    .pd            (1'b1),
    .ppen          (1'b0),
    .prg_slew      (1'b0),
    .puq           (1'b1),
    .pwrupzhl      (1'b0),
    .pwrup_pull_en (1'b0)
);

sdio_1v8_n1 south_io_10 ( //input
    .outi          (qspi_mosi[4]),
    .outi_1v8      (),  
    .pad           (gpio[10]),
    .ana_io_1v8    (),     
    .dq            (1'b0),
    .drv0          (1'b0),
    .drv1          (1'b0),
    .drv2          (1'b0),
    .enabq         (1'b0),
    .enq           (1'b1),
    .pd            (1'b1),
    .ppen          (1'b0),
    .prg_slew      (1'b0),
    .puq           (1'b1),
    .pwrupzhl      (1'b0),
    .pwrup_pull_en (1'b0)
);

sdio_1v8_n1 south_io_11 ( //input
    .outi          (qspi_mosi[5]),
    .outi_1v8      (),  
    .pad           (gpio[11]),
    .ana_io_1v8    (),     
    .dq            (1'b0),
    .drv0          (1'b0),
    .drv1          (1'b0),
    .drv2          (1'b0),
    .enabq         (1'b0),
    .enq           (1'b1),
    .pd            (1'b1),
    .ppen          (1'b0),
    .prg_slew      (1'b0),
    .puq           (1'b1),
    .pwrupzhl      (1'b0),
    .pwrup_pull_en (1'b0)
);

sdio_1v8_n1 south_io_12 ( //input
    .outi          (qspi_mosi[6]),
    .outi_1v8      (),  
    .pad           (gpio[12]),
    .ana_io_1v8    (),     
    .dq            (1'b0),
    .drv0          (1'b0),
    .drv1          (1'b0),
    .drv2          (1'b0),
    .enabq         (1'b0),
    .enq           (1'b1),
    .pd            (1'b1),
    .ppen          (1'b0),
    .prg_slew      (1'b0),
    .puq           (1'b1),
    .pwrupzhl      (1'b0),
    .pwrup_pull_en (1'b0)
);

sdio_1v8_n1 south_io_13 ( //input
    .outi          (qspi_mosi[7]),
    .outi_1v8      (),  
    .pad           (gpio[13]),
    .ana_io_1v8    (),     
    .dq            (1'b0),
    .drv0          (1'b0),
    .drv1          (1'b0),
    .drv2          (1'b0),
    .enabq         (1'b0),
    .enq           (1'b1),
    .pd            (1'b1),
    .ppen          (1'b0),
    .prg_slew      (1'b0),
    .puq           (1'b1),
    .pwrupzhl      (1'b0),
    .pwrup_pull_en (1'b0)
);

sdio_1v8_n1 south_io_14 ( //input
    .outi          (qspi_mosi[8]),
    .outi_1v8      (),  
    .pad           (gpio[14]),
    .ana_io_1v8    (),     
    .dq            (1'b0),
    .drv0          (1'b0),
    .drv1          (1'b0),
    .drv2          (1'b0),
    .enabq         (1'b0),
    .enq           (1'b1),
    .pd            (1'b1),
    .ppen          (1'b0),
    .prg_slew      (1'b0),
    .puq           (1'b1),
    .pwrupzhl      (1'b0),
    .pwrup_pull_en (1'b0)
);

sdio_1v8_n1 south_io_15 ( //input
    .outi          (qspi_mosi[9]),
    .outi_1v8      (),  
    .pad           (gpio[15]),
    .ana_io_1v8    (),     
    .dq            (1'b0),
    .drv0          (1'b0),
    .drv1          (1'b0),
    .drv2          (1'b0),
    .enabq         (1'b0),
    .enq           (1'b1),
    .pd            (1'b1),
    .ppen          (1'b0),
    .prg_slew      (1'b0),
    .puq           (1'b1),
    .pwrupzhl      (1'b0),
    .pwrup_pull_en (1'b0)
);

sdio_1v8_n1 south_io_16 ( //input
    .outi          (qspi_mosi[10]),
    .outi_1v8      (),  
    .pad           (gpio[16]),
    .ana_io_1v8    (),     
    .dq            (1'b0),
    .drv0          (1'b0),
    .drv1          (1'b0),
    .drv2          (1'b0),
    .enabq         (1'b0),
    .enq           (1'b1),
    .pd            (1'b1),
    .ppen          (1'b0),
    .prg_slew      (1'b0),
    .puq           (1'b1),
    .pwrupzhl      (1'b0),
    .pwrup_pull_en (1'b0)
);

sdio_1v8_n1 south_io_17 ( //input
    .outi          (qspi_mosi[11]),
    .outi_1v8      (),  
    .pad           (gpio[17]),
    .ana_io_1v8    (),     
    .dq            (1'b0),
    .drv0          (1'b0),
    .drv1          (1'b0),
    .drv2          (1'b0),
    .enabq         (1'b0),
    .enq           (1'b1),
    .pd            (1'b1),
    .ppen          (1'b0),
    .prg_slew      (1'b0),
    .puq           (1'b1),
    .pwrupzhl      (1'b0),
    .pwrup_pull_en (1'b0)
);

sdio_1v8_n1 south_io_18 ( //input
    .outi          (qspi_mosi[12]),
    .outi_1v8      (),  
    .pad           (gpio[18]),
    .ana_io_1v8    (),     
    .dq            (1'b0),
    .drv0          (1'b0),
    .drv1          (1'b0),
    .drv2          (1'b0),
    .enabq         (1'b0),
    .enq           (1'b1),
    .pd            (1'b1),
    .ppen          (1'b0),
    .prg_slew      (1'b0),
    .puq           (1'b1),
    .pwrupzhl      (1'b0),
    .pwrup_pull_en (1'b0)
);

sdio_1v8_n1 south_io_19 ( //input
    .outi          (qspi_mosi[13]),
    .outi_1v8      (),  
    .pad           (gpio[19]),
    .ana_io_1v8    (),     
    .dq            (1'b0),
    .drv0          (1'b0),
    .drv1          (1'b0),
    .drv2          (1'b0),
    .enabq         (1'b0),
    .enq           (1'b1),
    .pd            (1'b1),
    .ppen          (1'b0),
    .prg_slew      (1'b0),
    .puq           (1'b1),
    .pwrupzhl      (1'b0),
    .pwrup_pull_en (1'b0)
);

sdio_1v8_n1 south_io_20 ( //input
    .outi          (qspi_mosi[14]),
    .outi_1v8      (),  
    .pad           (gpio[20]),
    .ana_io_1v8    (),     
    .dq            (1'b0),
    .drv0          (1'b0),
    .drv1          (1'b0),
    .drv2          (1'b0),
    .enabq         (1'b0),
    .enq           (1'b1),
    .pd            (1'b1),
    .ppen          (1'b0),
    .prg_slew      (1'b0),
    .puq           (1'b1),
    .pwrupzhl      (1'b0),
    .pwrup_pull_en (1'b0)
);

sdio_1v8_n1 south_io_21 ( //input
    .outi          (qspi_mosi[15]),
    .outi_1v8      (),  
    .pad           (gpio[21]),
    .ana_io_1v8    (),     
    .dq            (1'b0),
    .drv0          (1'b0),
    .drv1          (1'b0),
    .drv2          (1'b0),
    .enabq         (1'b0),
    .enq           (1'b1),
    .pd            (1'b1),
    .ppen          (1'b0),
    .prg_slew      (1'b0),
    .puq           (1'b1),
    .pwrupzhl      (1'b0),
    .pwrup_pull_en (1'b0)
);

sdio_1v8_n1 south_io_22 ( //input
    .outi          (qspi_mosi_valid),
    .outi_1v8      (),  
    .pad           (gpio[22]),
    .ana_io_1v8    (),     
    .dq            (1'b0),
    .drv0          (1'b0),
    .drv1          (1'b0),
    .drv2          (1'b0),
    .enabq         (1'b0),
    .enq           (1'b1),
    .pd            (1'b1),
    .ppen          (1'b0),
    .prg_slew      (1'b0),
    .puq           (1'b1),
    .pwrupzhl      (1'b0),
    .pwrup_pull_en (1'b0)
);

sdio_1v8_n1 south_io_23 ( //output
    .outi          (),
    .outi_1v8      (),     
    .pad           (gpio[23]),
    .ana_io_1v8    (),
    .dq            (~qspi_miso[0]),
    .drv0          (1'b0),
    .drv1          (1'b0),
    .drv2          (1'b0),
    .enabq         (1'b0),
    .enq           (1'b0),
    .pd            (1'b1),
    .ppen          (1'b1),
    .prg_slew      (1'b0),
    .puq           (1'b1),
    .pwrupzhl      (1'b0),
    .pwrup_pull_en (1'b0)
);

sdio_1v8_n1 south_io_24 ( //output
    .outi          (),
    .outi_1v8      (),     
    .pad           (gpio[24]),
    .ana_io_1v8    (),
    .dq            (~qspi_miso[1]),
    .drv0          (1'b0),
    .drv1          (1'b0),
    .drv2          (1'b0),
    .enabq         (1'b0),
    .enq           (1'b0),
    .pd            (1'b1),
    .ppen          (1'b1),
    .prg_slew      (1'b0),
    .puq           (1'b1),
    .pwrupzhl      (1'b0),
    .pwrup_pull_en (1'b0)
);

sdio_1v8_n1 south_io_25 ( //output
    .outi          (),
    .outi_1v8      (),     
    .pad           (gpio[25]),
    .ana_io_1v8    (),
    .dq            (~qspi_miso[2]),
    .drv0          (1'b0),
    .drv1          (1'b0),
    .drv2          (1'b0),
    .enabq         (1'b0),
    .enq           (1'b0),
    .pd            (1'b1),
    .ppen          (1'b1),
    .prg_slew      (1'b0),
    .puq           (1'b1),
    .pwrupzhl      (1'b0),
    .pwrup_pull_en (1'b0)
);

sdio_1v8_n1 south_io_26 ( //output
    .outi          (),
    .outi_1v8      (),     
    .pad           (gpio[26]),
    .ana_io_1v8    (),
    .dq            (~qspi_miso[3]),
    .drv0          (1'b0),
    .drv1          (1'b0),
    .drv2          (1'b0),
    .enabq         (1'b0),
    .enq           (1'b0),
    .pd            (1'b1),
    .ppen          (1'b1),
    .prg_slew      (1'b0),
    .puq           (1'b1),
    .pwrupzhl      (1'b0),
    .pwrup_pull_en (1'b0)
);

sdio_1v8_n1 south_io_27 ( //output
    .outi          (),
    .outi_1v8      (),     
    .pad           (gpio[27]),
    .ana_io_1v8    (),
    .dq            (~qspi_miso[4]),
    .drv0          (1'b0),
    .drv1          (1'b0),
    .drv2          (1'b0),
    .enabq         (1'b0),
    .enq           (1'b0),
    .pd            (1'b1),
    .ppen          (1'b1),
    .prg_slew      (1'b0),
    .puq           (1'b1),
    .pwrupzhl      (1'b0),
    .pwrup_pull_en (1'b0)
);

sdio_1v8_n1 south_io_28 ( //output
    .outi          (),
    .outi_1v8      (),     
    .pad           (gpio[28]),
    .ana_io_1v8    (),
    .dq            (~qspi_miso[5]),
    .drv0          (1'b0),
    .drv1          (1'b0),
    .drv2          (1'b0),
    .enabq         (1'b0),
    .enq           (1'b0),
    .pd            (1'b1),
    .ppen          (1'b1),
    .prg_slew      (1'b0),
    .puq           (1'b1),
    .pwrupzhl      (1'b0),
    .pwrup_pull_en (1'b0)
);

sdio_1v8_n1 south_io_29 ( //output
    .outi          (),
    .outi_1v8      (),     
    .pad           (gpio[29]),
    .ana_io_1v8    (),
    .dq            (~qspi_miso[6]),
    .drv0          (1'b0),
    .drv1          (1'b0),
    .drv2          (1'b0),
    .enabq         (1'b0),
    .enq           (1'b0),
    .pd            (1'b1),
    .ppen          (1'b1),
    .prg_slew      (1'b0),
    .puq           (1'b1),
    .pwrupzhl      (1'b0),
    .pwrup_pull_en (1'b0)
);

sdio_1v8_n1 south_io_30 ( //output
    .outi          (),
    .outi_1v8      (),     
    .pad           (gpio[30]),
    .ana_io_1v8    (),
    .dq            (~qspi_miso[7]),
    .drv0          (1'b0),
    .drv1          (1'b0),
    .drv2          (1'b0),
    .enabq         (1'b0),
    .enq           (1'b0),
    .pd            (1'b1),
    .ppen          (1'b1),
    .prg_slew      (1'b0),
    .puq           (1'b1),
    .pwrupzhl      (1'b0),
    .pwrup_pull_en (1'b0)
);

sdio_1v8_n1 south_io_31 ( //output
    .outi          (),
    .outi_1v8      (),     
    .pad           (gpio[31]),
    .ana_io_1v8    (),
    .dq            (~qspi_miso[8]),
    .drv0          (1'b0),
    .drv1          (1'b0),
    .drv2          (1'b0),
    .enabq         (1'b0),
    .enq           (1'b0),
    .pd            (1'b1),
    .ppen          (1'b1),
    .prg_slew      (1'b0),
    .puq           (1'b1),
    .pwrupzhl      (1'b0),
    .pwrup_pull_en (1'b0)
);

sdio_1v8_n1 south_io_32 ( //output
    .outi          (),
    .outi_1v8      (),     
    .pad           (gpio[32]),
    .ana_io_1v8    (),
    .dq            (~qspi_miso[9]),
    .drv0          (1'b0),
    .drv1          (1'b0),
    .drv2          (1'b0),
    .enabq         (1'b0),
    .enq           (1'b0),
    .pd            (1'b1),
    .ppen          (1'b1),
    .prg_slew      (1'b0),
    .puq           (1'b1),
    .pwrupzhl      (1'b0),
    .pwrup_pull_en (1'b0)
);

sdio_1v8_n1 south_io_33 ( //output
    .outi          (),
    .outi_1v8      (),     
    .pad           (gpio[33]),
    .ana_io_1v8    (),
    .dq            (~qspi_miso[10]),
    .drv0          (1'b0),
    .drv1          (1'b0),
    .drv2          (1'b0),
    .enabq         (1'b0),
    .enq           (1'b0),
    .pd            (1'b1),
    .ppen          (1'b1),
    .prg_slew      (1'b0),
    .puq           (1'b1),
    .pwrupzhl      (1'b0),
    .pwrup_pull_en (1'b0)
);

sdio_1v8_n1 south_io_34 ( //output
    .outi          (),
    .outi_1v8      (),     
    .pad           (gpio[34]),
    .ana_io_1v8    (),
    .dq            (~qspi_miso[11]),
    .drv0          (1'b0),
    .drv1          (1'b0),
    .drv2          (1'b0),
    .enabq         (1'b0),
    .enq           (1'b0),
    .pd            (1'b1),
    .ppen          (1'b1),
    .prg_slew      (1'b0),
    .puq           (1'b1),
    .pwrupzhl      (1'b0),
    .pwrup_pull_en (1'b0)
);

sdio_1v8_n1 south_io_35 ( //output
    .outi          (),
    .outi_1v8      (),     
    .pad           (gpio[35]),
    .ana_io_1v8    (),
    .dq            (~qspi_miso[12]),
    .drv0          (1'b0),
    .drv1          (1'b0),
    .drv2          (1'b0),
    .enabq         (1'b0),
    .enq           (1'b0),
    .pd            (1'b1),
    .ppen          (1'b1),
    .prg_slew      (1'b0),
    .puq           (1'b1),
    .pwrupzhl      (1'b0),
    .pwrup_pull_en (1'b0)
);

sdio_1v8_n1 south_io_36 ( //output
    .outi          (),
    .outi_1v8      (),     
    .pad           (gpio[36]),
    .ana_io_1v8    (),
    .dq            (~qspi_miso[13]),
    .drv0          (1'b0),
    .drv1          (1'b0),
    .drv2          (1'b0),
    .enabq         (1'b0),
    .enq           (1'b0),
    .pd            (1'b1),
    .ppen          (1'b1),
    .prg_slew      (1'b0),
    .puq           (1'b1),
    .pwrupzhl      (1'b0),
    .pwrup_pull_en (1'b0)
);

sdio_1v8_n1 south_io_37 ( //output
    .outi          (),
    .outi_1v8      (),     
    .pad           (gpio[37]),
    .ana_io_1v8    (),
    .dq            (~qspi_miso[14]),
    .drv0          (1'b0),
    .drv1          (1'b0),
    .drv2          (1'b0),
    .enabq         (1'b0),
    .enq           (1'b0),
    .pd            (1'b1),
    .ppen          (1'b1),
    .prg_slew      (1'b0),
    .puq           (1'b1),
    .pwrupzhl      (1'b0),
    .pwrup_pull_en (1'b0)
);

sdio_1v8_n1 south_io_38 ( //output
    .outi          (),
    .outi_1v8      (),     
    .pad           (gpio[38]),
    .ana_io_1v8    (),
    .dq            (~qspi_miso[15]),
    .drv0          (1'b0),
    .drv1          (1'b0),
    .drv2          (1'b0),
    .enabq         (1'b0),
    .enq           (1'b0),
    .pd            (1'b1),
    .ppen          (1'b1),
    .prg_slew      (1'b0),
    .puq           (1'b1),
    .pwrupzhl      (1'b0),
    .pwrup_pull_en (1'b0)
);

sdio_1v8_n1 south_io_39 ( //output
    .outi          (),
    .outi_1v8      (),     
    .pad           (gpio[39]),
    .ana_io_1v8    (),
    .dq            (~qspi_miso_valid),
    .drv0          (1'b0),
    .drv1          (1'b0),
    .drv2          (1'b0),
    .enabq         (1'b0),
    .enq           (1'b0),
    .pd            (1'b1),
    .ppen          (1'b1),
    .prg_slew      (1'b0),
    .puq           (1'b1),
    .pwrupzhl      (1'b0),
    .pwrup_pull_en (1'b0)
);




//clk signals
`ifndef CHIP_TOP_SIMULATION
ana_io_n1 north_io_0 ( //input
    .d             (IIN),
    .pad           (gpio[40])
);
`endif

sdio_1v8_n1 north_io_1 ( //input
    .outi          (EXT_CLK_IN),
    .outi_1v8      (),  
    .pad           (gpio[41]),
    .ana_io_1v8    (),     
    .dq            (1'b0),
    .drv0          (1'b0),
    .drv1          (1'b0),
    .drv2          (1'b0),
    .enabq         (1'b0),
    .enq           (1'b1),
    .pd            (1'b1),
    .ppen          (1'b0),
    .prg_slew      (1'b0),
    .puq           (1'b1),
    .pwrupzhl      (1'b0),
    .pwrup_pull_en (1'b0)
);

sdio_1v8_n1 north_io_2 ( //input
    .outi          (CLK_DIV_SEL_0),
    .outi_1v8      (),  
    .pad           (gpio[42]),
    .ana_io_1v8    (),     
    .dq            (1'b0),
    .drv0          (1'b0),
    .drv1          (1'b0),
    .drv2          (1'b0),
    .enabq         (1'b0),
    .enq           (1'b1),
    .pd            (1'b1),
    .ppen          (1'b0),
    .prg_slew      (1'b0),
    .puq           (1'b1),
    .pwrupzhl      (1'b0),
    .pwrup_pull_en (1'b0)
);

sdio_1v8_n1 north_io_3 ( //input
    .outi          (CLK_DIV_SEL_1),
    .outi_1v8      (),  
    .pad           (gpio[43]),
    .ana_io_1v8    (),     
    .dq            (1'b0),
    .drv0          (1'b0),
    .drv1          (1'b0),
    .drv2          (1'b0),
    .enabq         (1'b0),
    .enq           (1'b1),
    .pd            (1'b1),
    .ppen          (1'b0),
    .prg_slew      (1'b0),
    .puq           (1'b1),
    .pwrupzhl      (1'b0),
    .pwrup_pull_en (1'b0)
);

sdio_1v8_n1 north_io_4 ( //input
    .outi          (CLK_GATE_EN),
    .outi_1v8      (),  
    .pad           (gpio[44]),
    .ana_io_1v8    (),     
    .dq            (1'b0),
    .drv0          (1'b0),
    .drv1          (1'b0),
    .drv2          (1'b0),
    .enabq         (1'b0),
    .enq           (1'b1),
    .pd            (1'b1),
    .ppen          (1'b0),
    .prg_slew      (1'b0),
    .puq           (1'b1),
    .pwrupzhl      (1'b0),
    .pwrup_pull_en (1'b0)
);

sdio_1v8_n1 north_io_5 ( //output
    .outi          (),
    .outi_1v8      (),     
    .pad           (gpio[45]),
    .ana_io_1v8    (),
    .dq            (~CLK_DIV100),
    .drv0          (1'b0),
    .drv1          (1'b0),
    .drv2          (1'b0),
    .enabq         (1'b0),
    .enq           (1'b0),
    .pd            (1'b1),
    .ppen          (1'b1),
    .prg_slew      (1'b0),
    .puq           (1'b1),
    .pwrupzhl      (1'b0),
    .pwrup_pull_en (1'b0)
);

sdio_1v8_n1 north_io_6 ( //output
    .outi          (),
    .outi_1v8      (),     
    .pad           (gpio[46]),
    .ana_io_1v8    (),
    .dq            (~current_token_finish_flag),
    .drv0          (1'b0),
    .drv1          (1'b0),
    .drv2          (1'b0),
    .enabq         (1'b0),
    .enq           (1'b0),
    .pd            (1'b1),
    .ppen          (1'b1),
    .prg_slew      (1'b0),
    .puq           (1'b1),
    .pwrupzhl      (1'b0),
    .pwrup_pull_en (1'b0)
);

sdio_1v8_n1 north_io_7 ( //output
    .outi          (),
    .outi_1v8      (),     
    .pad           (gpio[47]),
    .ana_io_1v8    (),
    .dq            (~current_token_finish_work),
    .drv0          (1'b0),
    .drv1          (1'b0),
    .drv2          (1'b0),
    .enabq         (1'b0),
    .enq           (1'b0),
    .pd            (1'b1),
    .ppen          (1'b1),
    .prg_slew      (1'b0),
    .puq           (1'b1),
    .pwrupzhl      (1'b0),
    .pwrup_pull_en (1'b0)
);

sdio_1v8_n1 north_io_8 ( //output
    .outi          (),
    .outi_1v8      (),     
    .pad           (gpio[48]),
    .ana_io_1v8    (),
    .dq            (~qgen_state_work),
    .drv0          (1'b0),
    .drv1          (1'b0),
    .drv2          (1'b0),
    .enabq         (1'b0),
    .enq           (1'b0),
    .pd            (1'b1),
    .ppen          (1'b1),
    .prg_slew      (1'b0),
    .puq           (1'b1),
    .pwrupzhl      (1'b0),
    .pwrup_pull_en (1'b0)
);

sdio_1v8_n1 north_io_9 ( //output
    .outi          (),
    .outi_1v8      (),     
    .pad           (gpio[49]),
    .ana_io_1v8    (),
    .dq            (~qgen_state_end),
    .drv0          (1'b0),
    .drv1          (1'b0),
    .drv2          (1'b0),
    .enabq         (1'b0),
    .enq           (1'b0),
    .pd            (1'b1),
    .ppen          (1'b1),
    .prg_slew      (1'b0),
    .puq           (1'b1),
    .pwrupzhl      (1'b0),
    .pwrup_pull_en (1'b0)
);

sdio_1v8_n1 north_io_10 ( //output
    .outi          (),
    .outi_1v8      (),     
    .pad           (gpio[50]),
    .ana_io_1v8    (),
    .dq            (~kgen_state_work),
    .drv0          (1'b0),
    .drv1          (1'b0),
    .drv2          (1'b0),
    .enabq         (1'b0),
    .enq           (1'b0),
    .pd            (1'b1),
    .ppen          (1'b1),
    .prg_slew      (1'b0),
    .puq           (1'b1),
    .pwrupzhl      (1'b0),
    .pwrup_pull_en (1'b0)
);

sdio_1v8_n1 north_io_11 ( //output
    .outi          (),
    .outi_1v8      (),     
    .pad           (gpio[51]),
    .ana_io_1v8    (),
    .dq            (~kgen_state_end),
    .drv0          (1'b0),
    .drv1          (1'b0),
    .drv2          (1'b0),
    .enabq         (1'b0),
    .enq           (1'b0),
    .pd            (1'b1),
    .ppen          (1'b1),
    .prg_slew      (1'b0),
    .puq           (1'b1),
    .pwrupzhl      (1'b0),
    .pwrup_pull_en (1'b0)
);

sdio_1v8_n1 north_io_12 ( //output
    .outi          (),
    .outi_1v8      (),     
    .pad           (gpio[52]),
    .ana_io_1v8    (),
    .dq            (~vgen_state_work),
    .drv0          (1'b0),
    .drv1          (1'b0),
    .drv2          (1'b0),
    .enabq         (1'b0),
    .enq           (1'b0),
    .pd            (1'b1),
    .ppen          (1'b1),
    .prg_slew      (1'b0),
    .puq           (1'b1),
    .pwrupzhl      (1'b0),
    .pwrup_pull_en (1'b0)
);

sdio_1v8_n1 north_io_13 ( //output
    .outi          (),
    .outi_1v8      (),     
    .pad           (gpio[53]),
    .ana_io_1v8    (),
    .dq            (~vgen_state_end),
    .drv0          (1'b0),
    .drv1          (1'b0),
    .drv2          (1'b0),
    .enabq         (1'b0),
    .enq           (1'b0),
    .pd            (1'b1),
    .ppen          (1'b1),
    .prg_slew      (1'b0),
    .puq           (1'b1),
    .pwrupzhl      (1'b0),
    .pwrup_pull_en (1'b0)
);

sdio_1v8_n1 north_io_14 ( //output
    .outi          (),
    .outi_1v8      (),     
    .pad           (gpio[54]),
    .ana_io_1v8    (),
    .dq            (~att_qk_state_work),
    .drv0          (1'b0),
    .drv1          (1'b0),
    .drv2          (1'b0),
    .enabq         (1'b0),
    .enq           (1'b0),
    .pd            (1'b1),
    .ppen          (1'b1),
    .prg_slew      (1'b0),
    .puq           (1'b1),
    .pwrupzhl      (1'b0),
    .pwrup_pull_en (1'b0)
);

sdio_1v8_n1 north_io_15 ( //output
    .outi          (),
    .outi_1v8      (),     
    .pad           (gpio[55]),
    .ana_io_1v8    (),
    .dq            (~att_qk_state_end),
    .drv0          (1'b0),
    .drv1          (1'b0),
    .drv2          (1'b0),
    .enabq         (1'b0),
    .enq           (1'b0),
    .pd            (1'b1),
    .ppen          (1'b1),
    .prg_slew      (1'b0),
    .puq           (1'b1),
    .pwrupzhl      (1'b0),
    .pwrup_pull_en (1'b0)
);

sdio_1v8_n1 north_io_16 ( //output
    .outi          (),
    .outi_1v8      (),     
    .pad           (gpio[56]),
    .ana_io_1v8    (),
    .dq            (~att_pv_state_work),
    .drv0          (1'b0),
    .drv1          (1'b0),
    .drv2          (1'b0),
    .enabq         (1'b0),
    .enq           (1'b0),
    .pd            (1'b1),
    .ppen          (1'b1),
    .prg_slew      (1'b0),
    .puq           (1'b1),
    .pwrupzhl      (1'b0),
    .pwrup_pull_en (1'b0)
);

sdio_1v8_n1 north_io_17 ( //output
    .outi          (),
    .outi_1v8      (),     
    .pad           (gpio[57]),
    .ana_io_1v8    (),
    .dq            (~att_pv_state_end),
    .drv0          (1'b0),
    .drv1          (1'b0),
    .drv2          (1'b0),
    .enabq         (1'b0),
    .enq           (1'b0),
    .pd            (1'b1),
    .ppen          (1'b1),
    .prg_slew      (1'b0),
    .puq           (1'b1),
    .pwrupzhl      (1'b0),
    .pwrup_pull_en (1'b0)
);

sdio_1v8_n1 north_io_18 ( //output
    .outi          (),
    .outi_1v8      (),     
    .pad           (gpio[58]),
    .ana_io_1v8    (),
    .dq            (~proj_state_work),
    .drv0          (1'b0),
    .drv1          (1'b0),
    .drv2          (1'b0),
    .enabq         (1'b0),
    .enq           (1'b0),
    .pd            (1'b1),
    .ppen          (1'b1),
    .prg_slew      (1'b0),
    .puq           (1'b1),
    .pwrupzhl      (1'b0),
    .pwrup_pull_en (1'b0)
);

sdio_1v8_n1 north_io_19 ( //output
    .outi          (),
    .outi_1v8      (),     
    .pad           (gpio[59]),
    .ana_io_1v8    (),
    .dq            (~proj_state_end),
    .drv0          (1'b0),
    .drv1          (1'b0),
    .drv2          (1'b0),
    .enabq         (1'b0),
    .enq           (1'b0),
    .pd            (1'b1),
    .ppen          (1'b1),
    .prg_slew      (1'b0),
    .puq           (1'b1),
    .pwrupzhl      (1'b0),
    .pwrup_pull_en (1'b0)
);

sdio_1v8_n1 north_io_20 ( //output
    .outi          (),
    .outi_1v8      (),     
    .pad           (gpio[60]),
    .ana_io_1v8    (),
    .dq            (~ffn0_state_work),
    .drv0          (1'b0),
    .drv1          (1'b0),
    .drv2          (1'b0),
    .enabq         (1'b0),
    .enq           (1'b0),
    .pd            (1'b1),
    .ppen          (1'b1),
    .prg_slew      (1'b0),
    .puq           (1'b1),
    .pwrupzhl      (1'b0),
    .pwrup_pull_en (1'b0)
);

sdio_1v8_n1 north_io_21 ( //output
    .outi          (),
    .outi_1v8      (),     
    .pad           (gpio[61]),
    .ana_io_1v8    (),
    .dq            (~ffn0_state_end),
    .drv0          (1'b0),
    .drv1          (1'b0),
    .drv2          (1'b0),
    .enabq         (1'b0),
    .enq           (1'b0),
    .pd            (1'b1),
    .ppen          (1'b1),
    .prg_slew      (1'b0),
    .puq           (1'b1),
    .pwrupzhl      (1'b0),
    .pwrup_pull_en (1'b0)
);

sdio_1v8_n1 north_io_22 ( //output
    .outi          (),
    .outi_1v8      (),     
    .pad           (gpio[62]),
    .ana_io_1v8    (),
    .dq            (~ffn1_state_work),
    .drv0          (1'b0),
    .drv1          (1'b0),
    .drv2          (1'b0),
    .enabq         (1'b0),
    .enq           (1'b0),
    .pd            (1'b1),
    .ppen          (1'b1),
    .prg_slew      (1'b0),
    .puq           (1'b1),
    .pwrupzhl      (1'b0),
    .pwrup_pull_en (1'b0)
);

sdio_1v8_n1 north_io_23 ( //output
    .outi          (),
    .outi_1v8      (),     
    .pad           (gpio[63]),
    .ana_io_1v8    (),
    .dq            (~ffn1_state_end),
    .drv0          (1'b0),
    .drv1          (1'b0),
    .drv2          (1'b0),
    .enabq         (1'b0),
    .enq           (1'b0),
    .pd            (1'b1),
    .ppen          (1'b1),
    .prg_slew      (1'b0),
    .puq           (1'b1),
    .pwrupzhl      (1'b0),
    .pwrup_pull_en (1'b0)
);


endmodule
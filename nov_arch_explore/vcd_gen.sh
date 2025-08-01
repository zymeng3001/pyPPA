#!/bin/bash

# Exit if any command fails
set -e

# Define paths
TOP_DIR=~/pyPPA/nov_arch_explore
RESULTS_DIR=$TOP_DIR/results/head_top
TB=$TOP_DIR/HW_NOV/head/head_top_tb.v
OUTPUT_EXEC=$TOP_DIR/HW_NOV/head/tb_head_top.vvp
OUTPUT_VCD=tb_head_top.vcd


# Go to working dir
cd $TOP_DIR

echo "Collecting Verilog files from HW_NOV..."
VERILOG_SOURCES="\
HW_NOV/util/pe.v \
HW_NOV/util/mem.v \
HW_NOV/util/align.v \
HW_NOV/util/open_fp_cores.v \
HW_NOV/gbus/arbiter.v \
HW_NOV/gbus/bus_controller.v \
HW_NOV/gbus/bus_packet_fifo.v \
HW_NOV/gbus/gbus_top.v \
HW_NOV/util/fadd_tree.v \
HW_NOV/vector_engine/krms_recompute/rtl/krms.v \
HW_NOV/vector_engine/RMSnorm/rtl/RMSnorm.v \
HW_NOV/vector_engine/RMSnorm/rtl/fp_div_pipe.v \
HW_NOV/vector_engine/RMSnorm/rtl/fp_invsqrt_pipe.v \
HW_NOV/vector_engine/RMSnorm/rtl/fp_mult_pipe.v \
HW_NOV/vector_engine/softmax_recompute/rtl/softmax_rc.v \
HW_NOV/core/core_acc.v \
HW_NOV/core/core_buf.v \
HW_NOV/core/core_ctrl.v \
HW_NOV/core/core_mac.v \
HW_NOV/core/core_mem.v \
HW_NOV/core/core_quant.v \
HW_NOV/core/core_rc.v \
HW_NOV/core/core_top.v \
HW_NOV/head/abuf.v \
HW_NOV/head/head_core_array.v \
HW_NOV/head/head_sram_rd_ctrl.v \
HW_NOV/head/head_sram.v \
HW_NOV/head/head_top.v \
HW_NOV/head/two_heads.v"



echo "Compiling testbench and RTL..."
iverilog -o $OUTPUT_EXEC $TB $VERILOG_SOURCES

echo "Running simulation..."
vvp $OUTPUT_EXEC

echo "Moving VCD to results directory..."
mkdir -p $RESULTS_DIR
mv $OUTPUT_VCD $RESULTS_DIR/head_top.vcd

echo "VCD file available at: $RESULTS_DIR/head_top.vcd"

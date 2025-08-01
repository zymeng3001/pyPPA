export SCRIPTS_DIR=/home/yimengz/pyPPA/scripts/ppa
export RESULTS_DIR=/home/yimengz/pyPPA/nov_arch_explore/results/head_top
export REPORTS_DIR=/home/yimengz/pyPPA/nov_arch_explore/reports/head_top
export PLATFORM_DIR=/home/yimengz/pyPPA/platforms/sky130hd
export TECH_LEF=/home/yimengz/pyPPA/platforms/sky130hd/lef/sky130_fd_sc_hd.tlef
export SC_LEF=/home/yimengz/pyPPA/platforms/sky130hd/lef/sky130_fd_sc_hd_merged.lef
export ADDITIONAL_LEFS="/home/yimengz/pyPPA/platforms/sky130hd/lef/sky130io_fill.lef"
export DESIGN_NAME=head_top
export LIB_FILES="/home/yimengz/pyPPA/platforms/sky130hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib"


export SYNTH_V=$RESULTS_DIR/1_synth_headtop.v

export USE_STA_VCD=0
# export STA_VCD_FILE=$RESULTS_DIR/head_top.vcd
# export STA_TB_DUT_INSTANCE=uut
# export VERILOG_TESTBENCH_MODULE=tb_head_top

# ----------- First TCL script: Power report -----------
echo "[INFO] Running power analysis..."

cat <<EOF > temp_power.tcl
read_lef \$::env(TECH_LEF)
read_lef \$::env(SC_LEF)
read_lef \$::env(ADDITIONAL_LEFS)
read_liberty \$::env(LIB_FILES)
read_liberty /home/yimengz/pyPPA/nov_arch_explore/HW_NOV/util/mem_sp.lib

read_verilog \$::env(SYNTH_V)
read_verilog /home/yimengz/pyPPA/nov_arch_explore/results/head_top/1_synth_coretop.v
link_design \$::env(DESIGN_NAME)

create_clock -name clk -period 100 [get_ports clk]

source \$::env(SCRIPTS_DIR)/write_power_report.tcl
write_power_report "1_synth"
exit
EOF

env SYNTH_V=$SYNTH_V DESIGN_NAME=$DESIGN_NAME SCRIPTS_DIR=$SCRIPTS_DIR openroad temp_power.tcl | tee $REPORTS_DIR/power_report.log

rm -f temp_power.tcl
echo "[INFO] Power analysis finished. Log: $REPORTS_DIR/power_report.log"

# ----------- Second TCL script: Best period report -----------
echo "[INFO] Running best period analysis..."

cat <<EOF > temp_timing.tcl
read_lef \$::env(TECH_LEF)
read_lef \$::env(SC_LEF)
read_lef \$::env(ADDITIONAL_LEFS)
read_liberty \$::env(LIB_FILES)
read_liberty /home/yimengz/pyPPA/nov_arch_explore/HW_NOV/util/mem_sp.lib

read_verilog \$::env(SYNTH_V)
read_verilog /home/yimengz/pyPPA/nov_arch_explore/results/head_top/1_synth_coretop.v
link_design \$::env(DESIGN_NAME)

create_clock -name clk -period 100 [get_ports clk]

source \$::env(SCRIPTS_DIR)/report_best_period.tcl
report_best_period "1_synth"
exit
EOF

env SYNTH_V=$SYNTH_V DESIGN_NAME=$DESIGN_NAME SCRIPTS_DIR=$SCRIPTS_DIR openroad temp_timing.tcl | tee $REPORTS_DIR/best_period.log

rm -f temp_timing.tcl
echo "[INFO] Best period analysis finished. Log: $REPORTS_DIR/best_period.log"

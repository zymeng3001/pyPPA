# ------------------------------
# Icarus Verilog Makefile
# Default TB: pure-Verilog testbench tb_core_top_ppa_v.v
# Usage:
#   make run                     # compile + run
#   make SIM_STD=2012 run        # if RTL uses SV constructs
#   make waves                   # open VCD in GTKWave
#   make clean
# You can override vars: TB, CLKPS, WARM, COOL, TIMEOUT, SEED, DEFS
# ------------------------------

# ---- Paths / files ----
TB        ?= HW_NOV/core/tb/core_top_tbv2.v
SIM       ?= build/sim_coretop_v
VCD       ?= core_top_ppa.vcd

# RTL directories (add/remove as needed)
RTL_DIRS  := HW_NOV/core HW_NOV/array HW_NOV/gbus HW_NOV/clk_gen HW_NOV/chip_top HW_NOV/util

# Include dirs for `sys_defs.svh`, etc. (adjust if yours differ)
INCS      := -I HW_NOV/header_file

# Collect all .v sources (exclude any /tb/ files)
RTL_SRCS  := $(shell find $(RTL_DIRS) -type f -name '*.v' | grep -v '/tb/')
# Ensure TB is not duplicated if it lives under core/tb
RTL_SRCS  := $(filter-out $(TB),$(RTL_SRCS))

# Language standard: 2005 for pure Verilog; switch to 2012 if RTL uses SV features
SIM_STD   ?= 2005
IVER_STD   = -g$(SIM_STD)

# Optional preprocessor defines, e.g. DEFS="-DDEBUG -DSOME_WIDTH=32"
DEFS      ?=

# ---- Plusargs / runtime knobs ----
CLKPS     ?= 3000
WARM      ?= 500
COOL      ?= 200
TIMEOUT   ?= 200000
SEED      ?= 1

# ---- Phonies ----
.PHONY: all run waves clean help sv

all: $(SIM)

build:
	mkdir -p build

$(SIM): build $(TB) $(RTL_SRCS)
	@echo "[Compile] Icarus Verilog $(SIM_STD)"
	iverilog $(IVER_STD) -Wall $(INCS) $(DEFS) -o $@ $(TB) $(RTL_SRCS)

run: $(SIM)
	@echo "[Run] vvp with +CLKPS=$(CLKPS) +WARM=$(WARM) +COOL=$(COOL) +TIMEOUT=$(TIMEOUT) +SEED=$(SEED)"
	vvp $(SIM) +CLKPS=$(CLKPS) +WARM=$(WARM) +COOL=$(COOL) +TIMEOUT=$(TIMEOUT) +SEED=$(SEED)

waves:
	gtkwave $(VCD) || true

clean:
	rm -rf build *.vcd *.lxt *.fst

help:
	@echo "Targets: all, run, waves, clean"
	@echo "Vars: TB, SIM_STD(2005|2012), CLKPS, WARM, COOL, TIMEOUT, SEED, DEFS"
	@echo "Examples:"
	@echo "  make run"
	@echo "  make SIM_STD=2012 run"
	@echo "  make DEFS='-DDEBUG -DSOME_WIDTH=64' run"

# Optional: build & run the SystemVerilog TB (if you also use tb_core_top_ppa.sv)
sv:
	$(MAKE) SIM_STD=2012 TB=HW_NOV/core/tb/tb_core_top_ppa.sv run

proc write_power_report {step} {
	set clks [all_clocks]

	if { [llength $clks] == 0 } {
		utl::warn "FLW" 6 "No clocks found."
	} else {
		set_propagated_clock [all_clocks]

		if {$::env(USE_STA_VCD) && [info exists ::env(STA_VCD_FILE)]} {
			puts "Reading VCD file for setting power activity."
			sta::read_power_activities -scope $::env(VERILOG_TESTBENCH_MODULE)/$::env(STA_TB_DUT_INSTANCE) -vcd $::env(STA_VCD_FILE)
		} else {
			puts "No VCD file found. Using default power activity values."
			sta::set_power_activity -input -activity .1
			sta::set_power_activity -input_port rst -activity 0
		}

		current_design $::env(DESIGN_NAME)

		set power_report_file [file join $::env(REPORTS_DIR) "${step}_power_report.txt"]
		sta::report_power > $power_report_file

		set fp [open $power_report_file w]

        # Initialize total power
        set total_power 0.0

        # Get power for each module
        foreach module [get_designs] {
            current_design $module
            set power [sta::report_power -quiet]  ;# Get power of the module
            puts $fp "$module: $power mW"
            set total_power [expr $total_power + $power]
        }

        # Print total power at the end of the report
        puts $fp "------------------------------------"
        puts $fp "Total Power (Summed from Modules): $total_power mW"
        puts $fp "------------------------------------"

        close $fp

		unset_propagated_clock [all_clocks]
	}
}
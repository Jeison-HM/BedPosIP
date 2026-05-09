# Search Path and Logic Library Setup
set_app_var search_path "$search_path . ./rtl ./libs"
set_app_var target_library "sky130_fd_sc_hd__ff_100C_1v95.db sky130_fd_sc_hd__ss_100C_1v40.db"
# set target_library "reflib.ndm"
set_app_var link_library "* $target_library"

# RTL Reading and Link
analyze -format verilog {top.v mux4_registered.v mux4.v register_bank.v}
elaborate top -parameters "WIDTH=3"
set_top_module top_WIDTH3
# analyze -format verilog {myproject*.v}
# set_top_module myproject

# Constraints Setup
set clk_val 15

create_clock -period $clk_val [get_ports clk]
set_clock_uncertainty -setup [expr $clk_val*0.1] [get_clocks clk]
set_clock_transition -max [expr $clk_val*0.1] [get_clocks clk]
set_clock_latency -source -max [expr $clk_val*0.05] [get_clocks clk]
set_clock_latency -max [expr $clk_val*0.03] [get_clocks clk]

set_input_delay -max [expr $clk_val*0.4] -clock clk [get_ports [remove_from_collection [all_inputs] clk]]
set_output_delay -max [expr $clk_val*0.5] -clock clk [get_ports [all_outputs]]

set_load -max 0.04 [all_outputs]
set_input_transition -min [expr $clk_val*0.01] [remove_from_collection [all_inputs] clk]
set_input_transition -max [expr $clk_val*0.1] [remove_from_collection [all_inputs] clk]

# Pre-compile Reports
report_clock > ./reports/pre_syn_report_clock.rpt
report_clock -skew > ./reports/pre_syn_report_clock_skew.rpt
report_port -verbose > ./reports/pre_syn_report_port_constraints.rpt
check_timing > ./reports/pre_syn_check_timing.rpt
check_design > ./reports/pre_syn_check_design.rpt

write_verilog outputs/unmapped.v

# Compile/Synthesis
compile_fusion -to logic_opto -no_autoungroup

# Post-compile Reports
report_qor > ./reports/post_syn_report_qor.rpt
report_constraints -all_violators > ./reports/post_syn_report_contraints.rpt
report_timing > ./reports/post_syn_report_timing.rpt
report_power > ./reports/post_syn_report_power.rpt

# Save Design
write_verilog outputs/mapped.v
write_sdc outputs/mapped.sdc

# Verify
get_designs
filter_collection [get_cells] "is_mapped != true"

# Exit
return
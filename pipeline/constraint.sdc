# Constraints Setup
set clk_val 10

create_clock -period $clk_val [get_ports ap_clk]
set_clock_uncertainty -setup [expr $clk_val*0.1] [get_clocks ap_clk]
set_clock_transition -max [expr $clk_val*0.1] [get_clocks ap_clk]
set_clock_latency -source -max [expr $clk_val*0.05] [get_clocks ap_clk]
set_clock_latency -max [expr $clk_val*0.03] [get_clocks ap_clk]

set_input_delay -max [expr $clk_val*0.1] -clock ap_clk [get_ports [all_inputs -no_clocks]]
set_output_delay -max [expr $clk_val*0.2] -clock ap_clk [get_ports [all_outputs]]

set_load -max 0.04 [all_outputs]
set_input_transition -min [expr $clk_val*0.01] [all_inputs -no_clocks]
set_input_transition -max [expr $clk_val*0.1] [all_inputs -no_clocks]
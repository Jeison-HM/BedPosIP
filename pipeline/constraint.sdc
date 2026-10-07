# Copyright 2026 Jeison Hernández
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Constraints Setup
set clk_val 20

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
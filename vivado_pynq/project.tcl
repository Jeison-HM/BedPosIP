# Vivado project configuration for PYNQ-Z2 bitstream generation
# Sourced by axi_stream_design.tcl

variable project_name
set project_name "myproject"

# AXI-Stream data widths (must match myproject_axi wrapper)
# Input: 16-bit for ap_ufixed<7,8> pressure_map_t
# Output: 32-bit for ap_fixed<14,9> result_t
variable bit_width_hls_input
set bit_width_hls_input 16

variable bit_width_hls_output
set bit_width_hls_output 32

# Target FPGA part (PYNQ-Z2)
variable part
set part "xc7z020clg400-1"

# Clock period in ns (100 MHz = 10ns)
variable clock_period
set clock_period 10

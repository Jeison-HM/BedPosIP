# Vitis HLS build script for myproject_axi wrapper
# Run: vitis-run --mode hls --tcl run_hls.tcl

set project_dir [file dirname [info script]]
set repo_root [file normalize [file join $project_dir ".." ".."]]
set firmware_dir [file join $repo_root "vitis"]

open_project myproject_axi_prj
set_top myproject_axi

# Add wrapper source (includes myproject.cpp as single translation unit)
add_files [file join $project_dir "myproject_axi.cpp"] \
    -cflags "-I$firmware_dir"

# Add testbench (optional - for C simulation)
# add_files -tb [file join $project_dir "myproject_axi_test.cpp"] \
#     -cflags "-I$firmware_dir"

open_solution "solution1" -flow_target vivado
set_part {xc7z020-clg400-1}

# Clock: 10ns (100 MHz) to match Zynq PS FCLK_CLK0
# The original myproject core was synthesized at 5ns (200 MHz)
# Vivado will handle any clock domain crossing if needed
create_clock -period 10 -name default

# Run C synthesis
csynth_design

# Export packaged IP for Vivado
export_design -format ip_catalog -description "AXI-Stream wrapper for myproject CNN" -vendor "xilinx.com" -library "hls" -display_name "myproject_axi"

exit

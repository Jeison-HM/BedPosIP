# PYNQ-Z2 AXI-Stream Overlay Build Script
# Adapted from hls4ml vivado_accelerator template for manual bitstream generation
# Run in Vivado Tcl Console: source axi_stream_design.tcl

set tcldir [file dirname [info script]]
source [file join $tcldir project.tcl]

# Path to the HLS wrapper IP (must be built first with Vitis HLS)
set wrapper_ip_path [file join $tcldir ".." "vitis_pynq" "myproject_axi_prj" "solution1" "impl" "ip"]

# Create Vivado project inside vivado/ directory
set project_path [file join $tcldir "${project_name}_vivado_accelerator"]
create_project project_1 $project_path -part $part -force

# Try to set PYNQ-Z2 board part (works in GUI mode; xhub boards are not loaded in batch mode)
set board_preset_enabled "0"
if {[catch {
    set_property board_part tul.com.tw:pynq-z2:part0:1.0 [current_project]
    set board_preset_enabled "1"
    puts "INFO: Board part tul.com.tw:pynq-z2:part0:1.0 set successfully."
} err]} {
    puts "WARNING: Could not set board_part in batch mode ($err)."
    puts "WARNING: Proceeding without board preset. Zynq PS will use default settings."
    puts "WARNING: This is normal for batch mode; PYNQ-Z2 boot image handles PS config."
}

# Add IP repository (the AXI-Stream wrapper)
set_property ip_repo_paths $wrapper_ip_path [current_project]
update_ip_catalog

# ------------------------------------------------------------------------------
# Block Design: Zynq PS + AXI DMA + myproject_axi HLS IP
# ------------------------------------------------------------------------------
create_bd_design "design_1"

# Add Zynq7 Processing System
startgroup
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0
endgroup

apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \
    -config [list make_external "FIXED_IO, DDR" apply_board_preset $board_preset_enabled Master "Disable" Slave "Disable"] \
    [get_bd_cells processing_system7_0]

# Enable HP0 port for high-performance DMA access
startgroup
set_property -dict [list CONFIG.PCW_USE_S_AXI_HP0 {1}] [get_bd_cells processing_system7_0]
endgroup

# Add AXI DMA
startgroup
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0
endgroup

# Configure DMA:
# - No scatter-gather (PYNQ requirement)
# - Max buffer length (26 bits = 67 MB)
# - Memory-mapped width: 64-bit (matches PYNQ HP port config)
# - Stream widths: match HLS wrapper (16-bit in, 32-bit out)
set_property -dict [list \
    CONFIG.c_include_sg {0} \
    CONFIG.c_sg_length_width {26} \
    CONFIG.c_sg_include_stscntrl_strm {0} \
    CONFIG.c_m_axi_mm2s_data_width {64} \
    CONFIG.c_m_axis_mm2s_tdata_width $bit_width_hls_input \
    CONFIG.c_mm2s_burst_size {256} \
    CONFIG.c_s_axis_s2mm_tdata_width $bit_width_hls_output \
    CONFIG.c_s_axis_s2mm_data_width {64} \
    CONFIG.c_s2mm_burst_size {256} \
] [get_bd_cells axi_dma_0]

# Connect DMA control port (AXI Lite) to PS GP0
startgroup
apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
    -config { Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} \
    Master {/processing_system7_0/M_AXI_GP0} Slave {/axi_dma_0/S_AXI_LITE} \
    ddr_seg {Auto} intc_ip {New AXI Interconnect} master_apm {0}} \
    [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

# Connect DMA read channel (MM2S) to HP0
apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
    -config { Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} \
    Master {/axi_dma_0/M_AXI_MM2S} Slave {/processing_system7_0/S_AXI_HP0} \
    ddr_seg {Auto} intc_ip {New AXI Interconnect} master_apm {0}} \
    [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
endgroup

# Connect DMA write channel (S2MM) to HP0 (shares interconnect)
apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
    -config { Clk_master {Auto} Clk_slave {/processing_system7_0/FCLK_CLK0 (100 MHz)} \
    Clk_xbar {/processing_system7_0/FCLK_CLK0 (100 MHz)} \
    Master {/axi_dma_0/M_AXI_S2MM} Slave {/processing_system7_0/S_AXI_HP0} \
    ddr_seg {Auto} intc_ip {/axi_mem_intercon} master_apm {0}} \
    [get_bd_intf_pins axi_dma_0/M_AXI_S2MM]

# Add HLS AXI-Stream wrapper IP
startgroup
create_bd_cell -type ip -vlnv xilinx.com:hls:${project_name}_axi:1.0 ${project_name}_axi_0
endgroup

# Connect AXI-Stream: DMA -> HLS IP -> DMA
connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S] [get_bd_intf_pins ${project_name}_axi_0/in_r]
connect_bd_intf_net [get_bd_intf_pins ${project_name}_axi_0/out_r] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]

# Auto-connect clock and reset for HLS IP
apply_bd_automation -rule xilinx.com:bd_rule:clkrst \
    -config { Clk {/processing_system7_0/FCLK_CLK0 (100 MHz)} Freq {100} } \
    [get_bd_pins ${project_name}_axi_0/ap_clk]

# Group DMA + HLS IP into hierarchy (matches PYNQ driver expectation)
group_bd_cells hier_0 [get_bd_cells axi_dma_0] [get_bd_cells ${project_name}_axi_0]

# ------------------------------------------------------------------------------
# Generate bitstream
# ------------------------------------------------------------------------------

# Create HDL wrapper
set bd_file [get_files "${project_path}/project_1.srcs/sources_1/bd/design_1/design_1.bd"]
make_wrapper -files $bd_file -top

set wrapper_file "${project_path}/project_1.srcs/sources_1/bd/design_1/hdl/design_1_wrapper.v"
add_files -norecurse $wrapper_file

# Run implementation
reset_run impl_1
reset_run synth_1
launch_runs impl_1 -to_step write_bitstream -jobs 6
wait_on_run -timeout 360 impl_1

# Open implemented design for inspection
open_run impl_1

# Generate utilization report
report_utilization -file [file join $tcldir "output" "util.rpt"] -hierarchical -hierarchical_percentages

# ------------------------------------------------------------------------------
# Copy deliverables to output directory
# ------------------------------------------------------------------------------
set bitstream_file "${project_path}/project_1.runs/impl_1/design_1_wrapper.bit"
set hwh_file "${project_path}/project_1.gen/sources_1/bd/design_1/hw_handoff/design_1.hwh"

# Fallback for older Vivado versions
if {![file exists $hwh_file]} {
    set hwh_file "${project_path}/project_1.srcs/sources_1/bd/design_1/hw_handoff/design_1.hwh"
}

# Copy bitstream
if {[file exists $bitstream_file]} {
    file copy -force $bitstream_file [file join $tcldir "output" "${project_name}.bit"]
    puts "INFO: Copied bitstream to output/${project_name}.bit"
} else {
    puts "ERROR: Bitstream not found at $bitstream_file"
}

# Copy hardware handoff (.hwh)
if {[file exists $hwh_file]} {
    file copy -force $hwh_file [file join $tcldir "output" "${project_name}.hwh"]
    puts "INFO: Copied hardware handoff to output/${project_name}.hwh"
} else {
    # Explicitly generate hardware handoff if missing
    set bd_file [get_files "${project_path}/project_1.srcs/sources_1/bd/design_1/design_1.bd"]
    open_bd_design $bd_file
    validate_bd_design
    generate_target all [get_files $bd_file]
    # Try again after generation
    if {[file exists $hwh_file]} {
        file copy -force $hwh_file [file join $tcldir "output" "${project_name}.hwh"]
        puts "INFO: Generated and copied hardware handoff to output/${project_name}.hwh"
    } else {
        puts "WARNING: .hwh file could not be found or generated. PYNQ may need manual hardware export."
    }
}

puts "INFO: Bitstream generation complete."
puts "INFO: Project located at: $project_path"
puts "INFO: Output folder: [file join $tcldir "output"]"

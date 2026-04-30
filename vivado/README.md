# PYNQ-Z2 Bitstream Generation - Manual Flow

This directory contains all files needed to manually generate a PYNQ-Z2 bitstream from the hls4ml-generated CNN accelerator.

## Why Manual Flow?

The `backend='vivadoaccelerator'` is **not supported with Keras 3** in current hls4ml. Therefore, we must:
1. Create an AXI-Stream wrapper around the core HLS IP
2. Synthesize the wrapper with Vitis HLS
3. Build the Vivado block design (Zynq PS + DMA + wrapper)
4. Generate the bitstream

## File Structure

```
vivado/
├── project.tcl                    # Project configuration (paths, widths)
├── axi_stream_design.tcl          # Main Vivado build script
├── hls_wrapper/
│   ├── myproject_axi.cpp          # AXI-Stream wrapper source
│   └── run_hls.tcl                # Vitis HLS build script
└── output/                        # Final bitstream + hwh files
```

## Prerequisites

1. **Vitis HLS** installed and sourced (for wrapper synthesis)
2. **Vivado** installed and sourced (for bitstream generation)
3. **PYNQ-Z2 board files** installed in Vivado

## Step-by-Step Build Instructions

### Step 1: Build the AXI-Stream Wrapper (Vitis HLS)

```bash
cd /home/jeison/Nextcloud/PUCMM/ProyectoGrado/vivado/hls_wrapper
vitis-run --mode hls --tcl run_hls.tcl
```

This generates the packaged IP at:
```
hls_wrapper/myproject_axi_prj/solution1/impl/ip/xilinx_com_hls_myproject_axi_1_0.zip
```

**What this does:**
- Takes `myproject_axi.cpp` (the wrapper)
- Includes the core CNN from `vitis/firmware/myproject.cpp`
- Includes headers and weights from `vitis/firmware/`
- Synthesizes to RTL with AXI-Stream interfaces (`in_r`, `out_r`)
- Exports a packaged IP for Vivado

### Step 2: Generate Bitstream (Vivado)

```bash
cd /home/jeison/Nextcloud/PUCMM/ProyectoGrado/vivado
vivado -mode batch -source axi_stream_design.tcl
```

Or open Vivado GUI and run in Tcl Console:
```tcl
cd /home/jeison/Nextcloud/PUCMM/ProyectoGrado/vivado
source axi_stream_design.tcl
```

**What this does:**
- Creates Vivado project in `vivado/myproject_vivado_accelerator/`
- Builds block design: Zynq PS + AXI DMA + `myproject_axi` IP
- Runs synthesis, implementation, and bitstream generation
- Generates utilization report in `vivado/output/util.rpt`

### Step 3: Inspect in Vivado GUI (Optional)

After the script completes:
```bash
vivado myproject_vivado_accelerator/project_1.xpr
```

Then:
- **Open Block Design** → View the visual diagram
- **Open Implemented Design** → Inspect floorplan, routing, timing
- **Reports** → View utilization, timing closure

### Step 4: Package for PYNQ

After bitstream generation:
```bash
cd /home/jeison/Nextcloud/PUCMM/ProyectoGrado/vivado
mkdir -p output

# Copy bitstream
cp myproject_vivado_accelerator/project_1.runs/impl_1/design_1_wrapper.bit \
   output/hls4ml_nn.bit

# Copy hardware handoff (Vivado 2024.x path)
cp myproject_vivado_accelerator/project_1.gen/sources_1/bd/design_1/hw_handoff/design_1.hwh \
   output/hls4ml_nn.hwh

# Copy PYNQ driver
cp /path/to/hls4ml/axi_stream_driver.py output/
```

> **Note:** The `.bit` and `.hwh` files **must have the same basename** for PYNQ to recognize them.

## Architecture

```
+-----------------------------------------------------------+
|                    Zynq Processing System                  |
|  +------------------+    +-----------------------------+  |
|  |  ARM Cortex-A9   |    |  High-Performance AXI (HP0) |  |
|  +------------------+    +-----------------------------+  |
+-----------------------------------------------------------+
          | AXI-Lite                    | AXI (64-bit)
          v                            v
+------------------+           +------------------+
|   AXI DMA Ctrl   |           |  AXI Interconnect |
+------------------+           +------------------+
          |                            |
          | AXI-Stream                 | AXI (64-bit)
          v                            v
+------------------+           +------------------+
|  myproject_axi   |           |  PS DRAM (512MB)  |
|  (HLS CNN IP)    |           +------------------+
+------------------+
```

## Data Flow

1. **Python (PYNQ)** allocates contiguous buffer with input data
2. **DMA** reads data from DRAM via AXI-MM, outputs via AXI-Stream
3. **myproject_axi** receives 64 input beats (16-bit each = 128 bytes/sample)
4. **myproject** core runs CNN inference
5. **myproject_axi** sends 3 output beats (32-bit each = 12 bytes/sample)
6. **DMA** receives AXI-Stream, writes back to DRAM via AXI-MM
7. **Python** reads results from output buffer

## AXI-Stream Data Widths

| Signal | Width | Bytes/Beat | Beats/Sample | Rationale |
|--------|-------|------------|--------------|-----------|
| `in_r` (input) | 16-bit | 2 | 64 | Matches `ap_fixed<14,9>` rounded to byte boundary |
| `out_r` (output) | 32-bit | 4 | 3 | Matches `ap_fixed<19,8>` rounded to byte boundary |

**DMA Configuration:**
- MM2S memory-mapped width: 64-bit (matches PYNQ HP port)
- S2MM memory-mapped width: 64-bit
- Stream widths: 16-bit (in), 32-bit (out)
- Buffer length: 26 bits (max 67 MB transfer)
- Burst size: 256 beats

## Troubleshooting

### "Wrapper IP not found"
Make sure Step 1 completed successfully. Check:
```bash
ls hls_wrapper/myproject_axi_prj/solution1/impl/ip/
```

If the build failed previously, delete the stale project before re-running:
```bash
rm -rf hls_wrapper/myproject_axi_prj
```

### "IP myproject_axi_0 not found in catalog"
The IP name might differ. Check the exact VLNV in:
```bash
grep "spirit:name" hls_wrapper/myproject_axi_prj/solution1/impl/ip/component.xml
```

### DMA transfer hangs
- Ensure `TLAST` is set on the final output beat (handled in wrapper)
- Check DMA buffer length register is set to 26 (not default 14)
- Verify HP port data width is 64-bit in Zynq PS config

## References

- hls4ml VivadoAccelerator backend: https://fastmachinelearning.org/hls4ml/advanced/accelerator.html
- PYNQ DMA Tutorial: https://discuss.pynq.io/t/tutorial-pynq-dma-part-1-hardware-design/3133
- PYNQ Overlay Design: https://discuss.pynq.io/t/tutorial-creating-a-hardware-design-for-pynq/145

// AXI-Stream wrapper for myproject CNN accelerator
// Bridges DMA AXI-Stream interface to myproject() core function
// Target: PYNQ-Z2 with AXI DMA

#include "ap_axi_sdata.h"
#include "hls_stream.h"
#include "ap_fixed.h"

// Include original model firmware (core source compiled as single translation unit)
#include "firmware/myproject.cpp"

// AXI-Stream type definitions
// Input: 16-bit to accommodate ap_fixed<14,9> (byte-aligned for DMA)
// Output: 32-bit to accommodate ap_fixed<19,8> (byte-aligned for DMA)
typedef ap_axiu<16, 0, 0, 0> input_axi_t;
typedef ap_axiu<32, 0, 0, 0> output_axi_t;

// Model dimensions
static const unsigned INPUT_SIZE = 8 * 8 * 1;  // 64 pixels
static const unsigned OUTPUT_SIZE = 3;         // 3 classes

void myproject_axi(
    hls::stream<input_axi_t> &in_r,
    hls::stream<output_axi_t> &out_r
) {
    // Interface pragmas
    #pragma HLS INTERFACE axis port=in_r
    #pragma HLS INTERFACE axis port=out_r
    #pragma HLS INTERFACE ap_ctrl_none port=return

    // Local arrays for model I/O
    pressure_map_t input_data[INPUT_SIZE];
    result_t output_data[OUTPUT_SIZE];

    #pragma HLS ARRAY_PARTITION variable=input_data complete dim=0
    #pragma HLS ARRAY_PARTITION variable=output_data complete dim=0

    // ------------------------------------------------------------------
    // Read input data from AXI-Stream (64 beats, one pixel per beat)
    // ------------------------------------------------------------------
    for (unsigned i = 0; i < INPUT_SIZE; i++) {
        #pragma HLS PIPELINE II=1
        input_axi_t val = in_r.read();
        // Convert 16-bit stream data to pressure_map_t (ap_fixed<14,9>)
        input_data[i] = (pressure_map_t)val.data;
    }

    // ------------------------------------------------------------------
    // Run inference
    // ------------------------------------------------------------------
    myproject(input_data, output_data);

    // ------------------------------------------------------------------
    // Write output data to AXI-Stream (3 beats, one class per beat)
    // ------------------------------------------------------------------
    for (unsigned i = 0; i < OUTPUT_SIZE; i++) {
        #pragma HLS PIPELINE II=1
        output_axi_t val;
        // Convert result_t (ap_fixed<19,8>) to 32-bit stream data
        val.data = (ap_int<32>)output_data[i];
        val.keep = 0xF;   // All 4 bytes valid
        val.strb = 0xF;
        val.last = (i == OUTPUT_SIZE - 1) ? 1 : 0;  // TLAST on final beat
        out_r.write(val);
    }
}

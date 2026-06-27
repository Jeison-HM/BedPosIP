// AXI-Stream wrapper for myproject CNN accelerator
// Bridges DMA AXI-Stream interface to myproject() core function
// Target: PYNQ-Z2 with AXI DMA

#include "ap_axi_sdata.h"
#include "hls_stream.h"
#include "ap_fixed.h"

// Include original model firmware (core source compiled as single translation unit)
#include "firmware/myproject.cpp"

// AXI-Stream type definitions
// Input: 16-bit to accommodate ap_ufixed<7,8> (byte-aligned for DMA)
// Output: 32-bit to accommodate ap_fixed<14,9> (byte-aligned for DMA)
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
    #pragma HLS DATAFLOW

    // Local streams to connect to myproject
    hls::stream<pressure_map_t> input_stream("input_stream");
    hls::stream<result_t> output_stream("output_stream");

    #pragma HLS STREAM variable=input_stream depth=64
    #pragma HLS STREAM variable=output_stream depth=1

    // ------------------------------------------------------------------
    // Read input data from AXI-Stream (64 beats, one pixel per beat)
    // ------------------------------------------------------------------
    for (unsigned i = 0; i < INPUT_SIZE; i++) {
        #pragma HLS PIPELINE II=1
        input_axi_t val = in_r.read();
        pressure_map_t pm;
        pm[0] = val.data;  // ap_uint<16> -> ap_ufixed<7,8>
        input_stream.write(pm);
    }

    // ------------------------------------------------------------------
    // Run inference
    // ------------------------------------------------------------------
    myproject(input_stream, output_stream);

    // ------------------------------------------------------------------
    // Write output data to AXI-Stream (3 beats, one class per beat)
    // Raw bits are preserved (sign-extended 14 -> 32) so the PYNQ driver
    // can decode ap_fixed<14,9> correctly with: result * 2^(9-14)
    // ------------------------------------------------------------------
    result_t res = output_stream.read();
    for (unsigned i = 0; i < OUTPUT_SIZE; i++) {
        #pragma HLS PIPELINE II=1
        output_axi_t val;
        ap_int<14> raw;
        raw.range(13, 0) = res[i].range(13, 0);
        val.data = (ap_int<32>)raw;
        val.keep = 0xF;   // All 4 bytes valid
        val.strb = 0xF;
        val.last = (i == OUTPUT_SIZE - 1) ? 1 : 0;  // TLAST on final beat
        out_r.write(val);
    }
}
<div align="center">
<h1>BedPosIP</h1>
</div>

<!-- GitHub Badges Section -->
<p align="center">
  <a href="https://www.python.org/">
    <img alt="Python" height="24.5px" style="padding-right:5px;" src="https://custom-icon-badges.demolab.com/badge/Python-3.13-2D3436?style=for-the-badge&logo=python&logoColor=white&logoSize=auto&labelColor=3776AB"/>
  </a>
  <a href="https://keras.io/">
    <img alt="Keras" height="24.5px" style="padding-right:5px;" src="https://custom-icon-badges.demolab.com/badge/Keras-3.14.0-2D3436?style=for-the-badge&logo=keras&logoColor=white&logoSize=auto&labelColor=D00000"/>
  </a>
  <a href="https://www.tensorflow.org/">
    <img alt="TensorFlow" height="24.5px" style="padding-right:5px;" src="https://custom-icon-badges.demolab.com/badge/TensorFlow-2.21.0-2D3436?style=for-the-badge&logo=tensorflow&logoColor=white&logoSize=auto&labelColor=FF6F00"/>
  </a>
  <a href="https://fastmachinelearning.org/hls4ml/">
    <img alt="hls4ml" height="24.5px" style="padding-right:5px;" src="https://custom-icon-badges.demolab.com/badge/hls4ml-1.2.0-2D3436?style=for-the-badge&logoColor=white&logoSize=auto&labelColor=1F4E79"/>
  </a>
  <a href="https://www.amd.com/en/products/software/adaptive-socs-and-fpgas/vitis.html">
    <img alt="Vitis" height="24.5px" style="padding-right:5px;" src="https://custom-icon-badges.demolab.com/badge/Vitis-2023.2-2D3436?style=for-the-badge&logo=amd&logoColor=white&logoSize=auto&labelColor=ED1C24"/>
  </a>
  <a href="https://fossi-foundation.org/librelane">
    <img alt="LibreLane" height="24.5px" style="padding-right:5px;" src="https://custom-icon-badges.demolab.com/badge/fossi-LibreLane-622eed?style=for-the-badge&logo=fossi&logoColor=white&logoSize=auto&labelColor=2D3436"/>
  </a>
  <a href="https://github.com/google/skywater-pdk">
    <img alt="SKY130" height="24.5px" style="padding-right:5px;" src="https://custom-icon-badges.demolab.com/badge/SkyWater-130nm-2D3436?style=for-the-badge&logo=skywater&logoColor=white&logoSize=auto&labelColor=3f6b87"/>
  </a>
  <a href="https://www.apache.org/licenses/LICENSE-2.0">
    <img alt="APACHE" height="24.5px" style="padding-right:5px;" src="https://custom-icon-badges.demolab.com/badge/License-APACHE%202.0-D22128?style=for-the-badge&logo=apache&logoColor=white&logoSize=auto&labelColor=2D3436"/>
  </a>
  <br />
</p>

<!-- Description -->
<p align="center">
    <img src="figures/IP_AREA_1.gif" alt="Final GDSII render" width="720" align="center">
    <br>
    <b>Quantized CNN accelerator for patient bed-posture classification, from ML model to ASIC</b><br>
    <i>with <code>HGQ2</code> quantization-aware training, <code>hls4ml</code> code translation, <code>PYNQ-Z2</code> FPGA prototyping<br>
    and synthesized with <code>SkyWater 130nm</code> technology via <code>LibreLane</code>.</i>
</p>

## Methodology

The project follows a four-stage flow [[1](#ref-lane2025)]. Each stage consumes the artifact produced by the previous one, so the design can be stopped, inspected or re-targeted at any boundary.

```mermaid
---
config:
  theme: 'dark'
---
flowchart LR
    %% STEP 1
    subgraph S1 ["<br> STEP 1: TF / Keras / HGQ2"]
        direction TB
        P1["• Data Augmentation<br>• Class Balancing<br>• FP32 Training<br>• QAT with HGQ2"]
        O1("model.keras")
        P1 --- O1
    end

    %% STEP 2
    subgraph S2 ["<br> STEP 2: hls4ml (VitisUnified)"]
        direction TB
        P2["• C++ Firmware Conversion<br>• ap_fixed / ap_ufixed<br>• Reuse Factor<br>• Parallelization Factor"]
        O2("firmware.cpp")
        P2 --- O2
    end

    %% STEP 3
    subgraph S3 ["<br> STEP 3: Vitis / Vivado"]
        direction TB
        P3["• XSA Platform<br>• High-Level Synthesis<br>• Bitstream Generation<br>• AXI-Stream Inference"]
        O3("bitstream.bit")
        P3 --- O3
    end

    %% STEP 4
    subgraph S4 ["<br> STEP 4: LibreLane"]
        direction TB
        P4["• Clean RTL Export<br>• Synthesis Exploration<br>• SKY130 Logic Synthesis<br>• Post-PNR STA"]
        O4("netlist.v")
        P4 --- O4
    end

    %% Process Flow
    S1 --> S2
    S2 --> S3
    S3 --> S4

    %% Colors for Nodes
    classDef step1 fill:#1e3a8a,stroke:#60a5fa,color:#ffffff,stroke-width:2px;
    classDef step2 fill:#064e3b,stroke:#34d399,color:#ffffff,stroke-width:2px;
    classDef step3 fill:#4c1d95,stroke:#a78bfa,color:#ffffff,stroke-width:2px;
    classDef step4 fill:#7c2d12,stroke:#fb923c,color:#ffffff,stroke-width:2px;

    class P1,O1 step1;
    class P2,O2 step2;
    class P3,O3 step3;
    class P4,O4 step4;

    %% Colors for Subgraph Borders
    style S1 stroke:#60a5fa,stroke-width:2px,fill:none,color:#60a5fa
    style S2 stroke:#34d399,stroke-width:2px,fill:none,color:#34d399
    style S3 stroke:#a78bfa,stroke-width:2px,fill:none,color:#a78bfa
    style S4 stroke:#fb923c,stroke-width:2px,fill:none,color:#fb923c

    %% Shapes
    P1@{shape: odd}
    P2@{shape: odd}
    P3@{shape: odd}
    P4@{shape: odd}
```

## Project Structure

```
bedposip/
├── pipeline/                                
│   ├── 01_data_augmentation.ipynb           # Augmentation stages
│   ├── 02_data_preparation.ipynb            # Dataset Preparation
│   ├── 03_train_base_cnn.ipynb              # FP32 baseline + KerasTuner search
│   ├── 04_train_quantized_cnn.ipynb         # HGQ2 QAT
│   ├── 05_gen_bitstream.ipynb               # hls4ml VitisUnified
│   ├── 06_infer_bitstream.ipynb             # PYNQ-Z2 Inference
│   ├── 07_export_ip.ipynb                   # hls4ml Vitis
│   ├── config.json                          # LibreLane design
│   ├── constraint.sdc                       # SDC timing constraints
│   └── input/
│       ├── DatosEntrenamiento.csv           # 192 raw samples
│       └── platform/
│           └── pynq-z2.xsa                  # Board platform for the FPGA flow
│
├── src/bedposip/                            # Installable Python package
│
├── tests/
│   └── test_augmentation.py                 # Unit tests for the augmentation core
│
├── figures/                                 # Assets used by this README
│
├── pyproject.toml                           # Package metadata, deps
├── uv.lock                                  
└── .python-version                          # 3.13
```

### Generated at Runtime

None of the following is tracked by git, each is produced by running the notebooks in order.

```
pipeline/
├── output/
│   ├── posture_classifier_base.keras        # Step 03 — FP32 baseline
│   ├── posture_classifier_tuned.keras       # Step 03 — tuned FP32
│   ├── posture_classifier_qat.keras         # Step 04 — quantized
│   ├── gen_bitstream/                       # Step 05 — VitisUnified build
│   │   └── export/                          
│   │       ├── system.bit                   # Step 06 — ALL 3 FILES BELOW MUST BE COPIED TO THE PYNQ-Z2
│   │       ├── system.hwh                   
│   │       └── axi_stream_driver.py         
│   │
│   └── export_ip/                           # Step 07 — Vitis build
│
├── rtl/                                     # Step 07 — generated .v files
├── runs/                                    # Step 08 — LibreLane runs
└── figures/                                 # Training curves, confusion matrices
```

## Installation

> **Python version:** `>=3.13,<3.14`

```bash
uv sync
source .venv/bin/activate
```

`uv` manages the environment and `pyproject.toml` declares the project dependencies, including the **hls4ml fork** [[2](#ref-jeison-hls4ml)] used for the FPGA flow.

> **Note:** the `bedposip` package must be importable before running any notebook, since every notebook imports from `bedposip`.

### GPU Support Setup: TensorFlow CUDA from pip packages

When installing `tensorflow[and-cuda]`, the CUDA libraries are delivered as separate `nvidia-*` pip packages inside the virtual environment. TensorFlow's dynamic linker does **not** automatically discover these `.so` files because `LD_LIBRARY_PATH` is empty by default, so `tf.config.list_physical_devices('GPU')` returns an empty list even though a GPU is present.

**Solution:** patch `.venv/bin/activate` so that activation automatically appends every `lib/` directory from the `nvidia-*` pip packages to `LD_LIBRARY_PATH`.

First, add the restore block **inside** the `deactivate()` function, right after the `_OLD_VIRTUAL_PYTHONHOME` restoration:

```bash
    if ! [ -z "${_OLD_LD_LIBRARY_PATH+_}" ] ; then
        LD_LIBRARY_PATH="$_OLD_LD_LIBRARY_PATH"
        export LD_LIBRARY_PATH
        unset _OLD_LD_LIBRARY_PATH
    fi
```

Then add the collection block **after** the `VIRTUAL_ENV_PROMPT` export and **before** the `pydoc` alias section:

```bash
# Add NVIDIA CUDA libraries from pip packages to LD_LIBRARY_PATH
_OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH-}"
CUDA_LIBS=""
for libdir in "$VIRTUAL_ENV"/lib/python*/site-packages/nvidia/*/lib; do
    if [ -d "$libdir" ]; then
        if [ -z "$CUDA_LIBS" ]; then
            CUDA_LIBS="$libdir"
        else
            CUDA_LIBS="$CUDA_LIBS:$libdir"
        fi
    fi
done
if [ -n "$CUDA_LIBS" ]; then
    if [ -n "${_OLD_LD_LIBRARY_PATH}" ]; then
        LD_LIBRARY_PATH="$CUDA_LIBS:$_OLD_LD_LIBRARY_PATH"
    else
        LD_LIBRARY_PATH="$CUDA_LIBS"
    fi
    export LD_LIBRARY_PATH
fi
```

Re-activate and verify:

```bash
source .venv/bin/activate
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Training is CPU-capable, but QAT (step 04) is strongly GPU-recommended.

## Toolchain Prerequisites

Steps 05, 07, and 08 require AMD and open-source EDA tooling that is **not** installed by `uv`.

| Tool | Version | Needed for |
| :--- | :--- | :--- |
| Vitis / Vivado | **2023.2** | Steps 05, 07 — HLS synthesis, bitstream, utilization reports ([install guide](https://www.amd.com/en/support/downloads/adaptive-socs-and-fpgas/development-tools/2023-2.html)) |
| PYNQ-Z2 board files | — | Step 05 — supplied via the tracked `pipeline/input/platform/pynq-z2.xsa` ([tutorial](https://github.com/Tanawin1701d/vitis_unified_backend_tutorial/tree/master/platform_setup_tutorial)) |
| LibreLane | recent | Step 08 — ASIC synthesis, place & route, STA, Final GDSII via Nix ([install guide](https://librelane.readthedocs.io/en/latest/installation/nix_installation/index.html)) |

Notebook `05` and `07` load Vitis into `PATH` from inside the notebook, so **edit the `script` variable** in the *Conversion* cell to match your installation:

```python
script = "/home/jeison/AMD/Vitis/2023.2/settings64.sh"
```

Without this, `hls_model.build()` fails because the backend cannot find `vitis_hls`.

### LaTeX for Figures

Notebooks 02–07 render every figure through LaTeX:

```python
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "Computer Modern Serif",
    }
)
```

`text.usetex=True` shells out to a real LaTeX installation, so **TeX must be installed on the machine** or those cells raise. Install a distribution with the Computer Modern fonts.

Debian/Ubuntu

```bash
sudo apt install texlive-latex-recommended texlive-fonts-recommended dvipng
```

Arch

```bash
sudo pacman -Syu texlive-latexrecommended texlive-fontsrecommended dvipng
```

> If you have no TeX and do not need publication-quality figures, set `"text.usetex": False` in the affected notebooks, since nothing else in the pipeline depends on it.

## Dataset

The raw dataset is `pipeline/input/DatosEntrenamiento.csv`, where each sample is an 8x8 pressure map.

<div align="center">

| POSICION  | Samples |
| :-------: | :-----: |
| trunk_L   |      34 |
| trunk_R   |      32 |
| fetus_R   |      32 |
| prone     |      32 |
| fetus_L   |      31 |
| supine    |      31 |
| **Total** | 192     |

</div>

> **Source:** The bed-posture dataset (`DatosEntrenamiento.csv`) was collected in a previous project under Justin Bueno [[3](#ref-bueno2022)] — [LinkedIn](https://www.linkedin.com/in/justin-bueno-d/).

## Pipeline

Run the notebooks **in order**, each with the working directory set to `pipeline/`. Steps 01–04 are CPU/GPU only, step 05 needs Vitis, step 06 runs on the board, step 07 needs Vitis and step 08 needs LibreLane.

### 1.  Data Augmentation — `01_data_augmentation.ipynb`

Three independent stages, spatial translation, per-cell KDE generation, and translation of the generated data, then class balancing to 50 000 samples per class, producing the balanced CSV consumed by the next step.

### 2.  Data Preparation — `02_data_preparation.ipynb`

Stratified 40/20/40 split, the 6 → 3 class merge, and export of the six `.npy` arrays that every later notebook loads through `load_dataset()`.

### 3.  Base CNN Training — `03_train_base_cnn.ipynb`

A Keras 3 FP32 baseline establishing the accuracy ceiling the quantized model is measured against. This architecture was adapted from the _SVHN classifier_ found in [[4](#ref-arrestad2021)].

<p align="center">
    <img src="figures/base_arch.svg" alt="Base FP32 CNN architecture" width="660" align="center">
    <br>
    <i>Base FP32 architecture</i>
</p>

<p align="center">
    <img src="figures/tuned_arch.svg" alt="Tuned FP32 CNN architecture" width="660" align="center">
    <br>
    <i>Tuned FP32 CNN architecture</i>
</p>

### 4.  Quantization-Aware Training — `04_train_quantized_cnn.ipynb`

HGQ2 quantize the Keras layers while keeping the spatial topology identical to the FP32 baseline, so the two stay directly comparable. No softmax is used, the output layer emits raw logits [[5](#ref-hgq2)].

<p align="center">
    <img src="figures/qat_arch.svg" alt="Quantized CNN architecture" width="660" align="center">
    <br>
    <i>Quantized CNN architecture</i>
</p>

### 5.  FPGA Bitstream Generation — `05_gen_bitstream.ipynb`

The `VitisUnified` backend synthesizes, implements and packages the whole design in one call, emitting the bitstream, the utilization reports and the board driver together.

The generated accelerator is placed into a Zynq-7000 platform alongside an AXI DMA, following the tutorial of the `VitisUnified` backend [[6](#ref-tanawin-vitisunified)].

<p align="center">
    <img src="figures/vitis_diagram_empty.svg" alt="Block design baseline" width="640" align="center">
    <br>
    <i>Platform baseline</i>
    <br><br>
    <img src="figures/vitis_diagram_filled.svg" alt="Block design with accelerator" width="640" align="center">
    <br>
    <i>The same design plus <code>myproject_axi_stream_1</code> bound to the DMA streams</i>
</p>

<p align="center">
    <img src="figures/myproject_rtl_standalone.svg" alt="hls4ml generated RTL top level" width="560" align="center">
    <br>
    <i>IP Diagram with axi-stream interface generated by Vitis HLS</i>
</p>

**All three `export/` artifacts (`system.bit`, `system.hwh` and `axi_stream_driver.py`) must be copied to the PYNQ-Z2 board**, and the `.bit`/`.hwh` pair must share the same basename or PYNQ cannot bind the overlay. The driver is generated by the backend and is not part of the repository.

### 6.  On-Board Inference — `06_infer_bitstream.ipynb`

**This notebook runs on the PYNQ-Z2 board, not on the workstation.** The bitstream, the `.hwh` and the generated driver are copied to the board, where the overlay is loaded and the test set is classified in batches. **BATCH** must stay under the **20480-bit** DMA transfer limit. The hardware returns raw logits, so the prediction need a function like argmax.

### 7.  RTL Export for ASIC — `07_export_ip.ipynb`

The same quantized model is re-converted with the plain `Vitis` backend [[7](#ref-fastml-hls4ml)] and built, yielding synthesisable Verilog files in `pipeline/rtl/`. This step produces the clean RTL required for the ASIC flow, no bitstream is generated.

### 8.  ASIC Synthesis & PNR — LibreLane

The ASIC implementation uses LibreLane [[8](#ref-librelane)] with the SkyWater 130nm PDK [[9](#ref-google-skywater)]. This step is **manual** (no notebook) and must be run after step 07 completes. See [Toolchain Prerequisites](#toolchain-prerequisites).

**Run commands:**

```bash
cd pipeline/
nix-shell ~/librelane/shell.nix
librelane config.json
```

The **Classic** flow uses `pipeline/config.json` and `pipeline/constraint.sdc` for timing constraints.

<p align="center">
    <img src="figures/IP_AREA_1.png" alt="Final GDSII render" width="620" align="center">
    <br>
    <i>Final GDSII render of the AREA 1 strategy</i>
</p>

## Reproducibility

Every notebook fixes randomness explicitly:

```python
np.random.seed(42)
keras.utils.set_random_seed(42)
```

## References

<a id="ref-lane2025"></a>
[1] D. M. Lane and A. Sahafi, "ADNA: Automating Application-Specific Integrated Circuit Development of Neural Network Accelerators," *Electronics*, vol. 14, no. 7, art. 1432, 2025, doi: 10.3390/electronics14071432. [https://www.mdpi.com/2079-9292/14/7/1432](https://www.mdpi.com/2079-9292/14/7/1432)

<a id="ref-jeison-hls4ml"></a>
[2] Jeison-HM, "hls4ml — VitisUnifiedClean branch," GitHub. [https://github.com/Jeison-HM/hls4ml/tree/VitisUnifiedClean](https://github.com/Jeison-HM/hls4ml/tree/VitisUnifiedClean)

<a id="ref-bueno2022"></a>
[3] J. E. Bueno Díaz, "Prototipo Funcional de Sistema para Detección y Monitoreo de Áreas Susceptibles a Úlceras por Decúbito mediante Sensores de Presión Textiles," Proyecto de Grado, Pontificia Universidad Católica Madre y Maestra, 2022.

<a id="ref-arrestad2021"></a>
[4] T. Aarrestad et al., "Fast convolutional neural networks on FPGAs with hls4ml," arXiv:2101.05108v2, 2021. [https://arxiv.org/abs/2101.05108v2](https://arxiv.org/abs/2101.05108v2)

<a id="ref-hgq2"></a>
[5] FastML Team, "HGQ2 — hardware-generative quantization," GitHub, 2024. [https://github.com/fastmachinelearning/hgq2](https://github.com/fastmachinelearning/hgq2)

<a id="ref-tanawin-vitisunified"></a>
[6] Tanawin1701d, "hls4ml at VitisUnifiedClean," GitHub. [https://github.com/Tanawin1701d/hls4ml/tree/VitisUnifiedClean](https://github.com/Tanawin1701d/hls4ml/tree/VitisUnifiedClean)

<a id="ref-fastml-hls4ml"></a>
[7] FastML Team, "hls4ml," 2024. [https://github.com/fastmachinelearning/hls4ml](https://github.com/fastmachinelearning/hls4ml)

<a id="ref-librelane"></a>
[8] LibreLane contributors, "LibreLane," FOSSi Foundation. [https://github.com/librelane/librelane](https://github.com/librelane/librelane)

<a id="ref-google-skywater"></a>
[9] Google, "SkyWater130nm open-source process design kit." [https://github.com/google/skywater-pdk](https://github.com/google/skywater-pdk)
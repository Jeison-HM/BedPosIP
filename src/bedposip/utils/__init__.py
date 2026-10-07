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

"""NeuroPosASIC utils submodule.

Exports:
    - :func:`check_layer_trainable_params`: Inspect layer sizes.
    - :func:`plot_training_history`: Plot training curves.
    - :func:`class_report_metric`: Classification report.
    - :func:`save_model`: Model persistence.
    - :func:`inspect_quantization_bits`: Learned bit-width summary.
    - :func:`check_quantizer_homogeneity`: Verify activation homogeneity.
    - :func:`inspect_hls4ml_precision`: hls4ml layer precision inspection.
"""

from bedposip.utils.visualization import (
    check_layer_trainable_params,
    check_quantizer_homogeneity,
    class_report_metric,
    inspect_hls4ml_precision,
    inspect_quantization_bits,
    plot_quantization_bits,
    plot_training_history,
    save_model,
)

__all__ = [
    "check_layer_trainable_params",
    "check_quantizer_homogeneity",
    "class_report_metric",
    "inspect_hls4ml_precision",
    "inspect_quantization_bits",
    "plot_quantization_bits",
    "plot_training_history",
    "save_model",
]

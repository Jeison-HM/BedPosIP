"""NeuroPosASIC utils submodule.

Exports:
    - :func:`check_layer_trainable_params`: Inspect layer sizes.
    - :func:`plot_training_history`: Plot training curves.
    - :func:`class_report_metric`: Classification report.
    - :func:`save_model`: Model persistence.
    - :func:`inspect_quantization_bits`: Learned bit-width summary.
    - :func:`check_quantizer_homogeneity`: Verify activation homogeneity.
"""

from bedposip.utils.visualization import (
    check_layer_trainable_params,
    check_quantizer_homogeneity,
    class_report_metric,
    inspect_quantization_bits,
    plot_quantization_bits,
    plot_training_history,
    save_model,
)

__all__ = [
    "check_layer_trainable_params",
    "check_quantizer_homogeneity",
    "class_report_metric",
    "inspect_quantization_bits",
    "plot_quantization_bits",
    "plot_training_history",
    "save_model",
]

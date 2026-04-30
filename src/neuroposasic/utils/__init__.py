"""NeuroPosASIC utils submodule.

Exports:
    - :func:`check_layer_trainable_params`: Inspect layer sizes.
    - :func:`plot_training_history`: Plot training curves.
    - :func:`class_report_metric`: Classification report.
    - :func:`save_model`: Model persistence.
"""

from neuroposasic.utils.visualization import (
    check_layer_trainable_params,
    class_report_metric,
    plot_training_history,
    save_model,
)

__all__ = [
    "check_layer_trainable_params",
    "plot_training_history",
    "class_report_metric",
    "save_model",
]

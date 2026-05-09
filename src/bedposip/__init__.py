"""NeuroPosASIC package for neural network posture classification accelerator.

This package centralizes shared logic between the training notebooks of
the thesis project, organized into logical submodules:

- :mod:`bedposip.model`: CNN architectures (FP32 and quantized) and training utilities.
- :mod:`bedposip.data`: Data loading, preprocessing, and augmentation.
- :mod:`bedposip.utils`: Visualization, evaluation, and persistence helpers.

Example:
    >>> from bedposip.model import create_model
    >>> from bedposip.data import load_dataset, preprocess_data
    >>> from bedposip.utils import plot_training_history
"""

from bedposip.model import create_model, create_quantized_model, get_callbacks, train_model
from bedposip.data import load_dataset, preprocess_data
from bedposip.utils import (
    check_layer_trainable_params,
    class_report_metric,
    plot_training_history,
    save_model,
)

__all__ = [
    "create_model",
    "create_quantized_model",
    "load_dataset",
    "preprocess_data",
    "get_callbacks",
    "train_model",
    "check_layer_trainable_params",
    "class_report_metric",
    "plot_training_history",
    "save_model",
]

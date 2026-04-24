"""NeuroPosASIC package for neural network posture classification accelerator.

This package centralizes shared logic between the training notebooks of
the thesis project, including:

- Pressure sensor data loading and preprocessing.
- CNN architecture definitions (FP32 and quantized).
- Training and visualization utilities.

Example:
    >>> from neuroposasic import create_model, load_dataset, train_model
    >>> X_train, y_train, *_ = load_dataset()
    >>> model = create_model()
    >>> model.summary()
"""

from neuroposasic.architecture import create_model, create_quantized_model
from neuroposasic.data import load_dataset, preprocess_data
from neuroposasic.training import get_callbacks, train_model
from neuroposasic.utils import (
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

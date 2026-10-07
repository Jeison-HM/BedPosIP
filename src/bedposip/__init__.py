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

"""NeuroPosASIC package for neural network posture classification accelerator.

This package centralizes shared logic between the training notebooks of
the thesis project, organized into logical submodules:

- :mod:`bedposip.model`: CNN architectures (FP32 and quantized) and training utilities.
- :mod:`bedposip.data`: Data loading and augmentation.
- :mod:`bedposip.utils`: Visualization, evaluation, and persistence helpers.

Example:
    >>> from bedposip.model import create_model
    >>> from bedposip.data import load_dataset
    >>> from bedposip.utils import plot_training_history
"""

from bedposip.model import create_model, create_quantized_model, get_callbacks, train_model
from bedposip.data import load_dataset
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
    "get_callbacks",
    "train_model",
    "check_layer_trainable_params",
    "class_report_metric",
    "plot_training_history",
    "save_model",
]

"""NeuroPosASIC model submodule.

Exports:
    - :func:`create_model`: Base FP32 CNN architecture.
    - :func:`create_quantized_model`: Quantized CNN for HGQ2 QAT.
    - :func:`get_callbacks`: Training callback factory.
    - :func:`train_model`: Training orchestration.
    - Pruning is handled via tfmot.sparsity.keras (not this module).
"""

from bedposip.model.architecture import create_model, create_quantized_model
from bedposip.model.training import get_callbacks, train_model

__all__ = [
    "create_model",
    "create_quantized_model",
    "get_callbacks",
    "train_model",
]

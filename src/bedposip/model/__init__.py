"""NeuroPosASIC model submodule.

Exports:
    - :func:`create_model`: Base FP32 CNN architecture.
    - :func:`create_quantized_model`: Quantized CNN for HGQ2 QAT.
    - :func:`get_callbacks`: Training callback factory.
    - :func:`train_model`: Training orchestration.
    - :func:`run_sweep`: QAT bit-width sweep execution.
    - :func:`plot_pareto`: Pareto frontier visualization.
    - :func:`print_pareto_summary`: Pareto summary printer.
"""

from bedposip.model.architecture import create_model, create_quantized_model
from bedposip.model.sweep import plot_pareto, print_pareto_summary, run_sweep
from bedposip.model.training import get_callbacks, train_model

__all__ = [
    "create_model",
    "create_quantized_model",
    "get_callbacks",
    "plot_pareto",
    "print_pareto_summary",
    "run_sweep",
    "train_model",
]

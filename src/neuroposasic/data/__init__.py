"""NeuroPosASIC data submodule.

Exports:
    - :func:`load_dataset`: Load `.npz` datasets.
    - :func:`preprocess_data`: Encode, reshape, and cast data.
    - :func:`row_to_mat` / :func:`mat_to_row`: Row ↔ matrix conversions.
    - :func:`calculate_centroid`: Intensity-weighted centroid.
    - :func:`logic_box`: Bounding-box / displacement logic.
    - :func:`gen_trans`: Spatial translation generator.
    - :func:`spatial_augmentation`: Stage 1 augmentation.
    - :func:`distribution_generation`: Stage 2 KDE-based generation.
    - :func:`dist_spatial_augmentation`: Stage 3 augmentation.
    - :func:`balance_to_target`: Exact-count balancing.
"""

from neuroposasic.data.balancer import balance_to_target
from neuroposasic.data.core import (
    calculate_centroid,
    gen_trans,
    logic_box,
    mat_to_row,
    row_to_mat,
)
from neuroposasic.data.loading import load_dataset, preprocess_data
from neuroposasic.data.stages import (
    dist_spatial_augmentation,
    distribution_generation,
    spatial_augmentation,
)

__all__ = [
    "load_dataset",
    "preprocess_data",
    "row_to_mat",
    "mat_to_row",
    "calculate_centroid",
    "logic_box",
    "gen_trans",
    "spatial_augmentation",
    "distribution_generation",
    "dist_spatial_augmentation",
    "balance_to_target",
]
